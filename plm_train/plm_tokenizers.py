from functools import cached_property

import torch
import tokenizers
from tokenizers.models import BPE
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

from torchtitan.datasets.tokenizer import Tokenizer

CATALYTIC_VOCAB = [
    "<cls>", "<pad>", "<eos>", "<unk>",
    "Z", "X", ".", "-", "|",
    "<mask>",
]
SASA_DISCRETIZATION_BOUNDARIES = [
    0.8,
    4.0,
    9.6,
    16.4,
    24.5,
    32.9,
    42.0,
    51.5,
    61.2,
    70.9,
    81.6,
    93.3,
    107.2,
    125.4,
    151.4,
]


class CatalyticTokenizer(PreTrainedTokenizerFast):

    def __init__(
        self,
        unk_token = "<unk>", 
        cls_token = "<cls>",
        pad_token = "<pad>",
        mask_token = "<mask>",
        eos_token = "<eos>",
        chain_break_token="|",
        **kwargs,
    ):

        self.all_tokens = CATALYTIC_VOCAB
        token_to_id = {token: i for i, token in enumerate(self.all_tokens)}

        # BPE tokenizer is a good character-level tokenizer
        bpe = BPE(token_to_id, merges=[], unk_token=unk_token)
        tokenizer = tokenizers.Tokenizer(bpe)
        special_tokens = [cls_token, pad_token, mask_token, eos_token, chain_break_token]
        tokenizer.add_special_tokens(special_tokens)

        # This is where we configure the automatic addition of special tokens when we call
        # tokenizer(text, add_special_tokens=True). Note that you can also configure how two
        # sequences are merged if you want.
        tokenizer.post_processor = TemplateProcessing(  # type: ignore
            single="<cls> $A <eos>",
            special_tokens=[
                ("<cls>", tokenizer.token_to_id("<cls>")),
                ("<eos>", tokenizer.token_to_id("<eos>")),
            ],
        )

        super().__init__(
            tokenizer_object=tokenizer,
            unk_token=unk_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            eos_token=eos_token,
            **kwargs,
        )


class SasaTokenizer(Tokenizer):
    def __init__(
        self,
        boundaries = SASA_DISCRETIZATION_BOUNDARIES,
    ):
        
        self._boundaries = sorted(boundaries)
        
    # From ESM3 repo
    @cached_property
    def special_tokens(self):
        return ["<pad>", "<motif>", "<unk>"]

    @cached_property
    def vocab(self):
        """Discrete token vocabulary.

        Returns:
            token vocabulary with ranges represented as "<low-high>".
        """
        boundary_strs = ["0"] + [str(b) for b in self._boundaries] + ["inf"]
        range_tokens = [
            f"<{low}-{high}>"
            for low, high in zip(boundary_strs[:-1], boundary_strs[1:])
        ]
        return self.special_tokens + range_tokens

    @cached_property
    def midpoints(self):
        """Midpoints of the SASA token ranges."""
        boundaries = [0] + self._boundaries + [self._boundaries[-1] * 2]
        midpoint_tokens = [
            (float(high) + float(low)) / 2
            for low, high in zip(boundaries[:-1], boundaries[1:])
        ]
        midpoint_tokens = [float("nan"), float("nan"), float("nan")] + midpoint_tokens
        return midpoint_tokens

    @cached_property
    def vocab_to_index(self):
        """Constructs token -> token id mapping."""
        return {word: i for i, word in enumerate(self.vocab)}

    def encode(
        self, values, add_special_tokens = True
    ) -> torch.Tensor:
        """Encodes SASA values as discrete tokens.

        Args:
            values: list of either SASA values or individual tokens. For example
                [1.2, "<pad>", 10.3, <pad>, 0.]
        Returns:
            Token ids as tensor. Adds BOS and EOS special tokens.
        """
        ids = []
        if add_special_tokens:
            ids.append(self.vocab_to_index["<pad>"])  # BOS
        for value in values:
            if isinstance(value, (float, int)):
                bucket = torch.bucketize(value, torch.tensor(self._boundaries))
                token_id = len(self.special_tokens) + bucket
            elif isinstance(value, str):
                token_id = self.vocab_to_index[value]
            else:
                raise TypeError(value)
            ids.append(token_id)
        if add_special_tokens:
            ids.append(self.vocab_to_index["<pad>"])  # EOS

        return torch.tensor(ids, dtype=torch.int64)

    
    def decode(self, encoded: torch.Tensor) -> str:
        """Decodes SASA token ids."""
        return ",".join(self.vocab[i] for i in encoded)