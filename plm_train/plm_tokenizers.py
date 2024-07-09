from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

CATALYTIC_VOCAB = [
    "<cls>", "<pad>", "<eos>", "<unk>",
    "Z", "X", ".", "-", "|",
    "<mask>",
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
        tokenizer = Tokenizer(bpe)
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




