import os
import warnings
import random
import pickle
from typing import Any, Dict, List, Optional
from datasets.distributed import split_dataset_by_node
from functools import partial
from typing import Callable, List, Optional

import torch
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset
from torchdata.stateful_dataloader import StatefulDataLoader
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node

from torchtitan.datasets.tokenizer import Tokenizer
from torchtitan.logging_utils import logger

from fasta_utils import fasta_to_json


# Create json files for training and validation
def split_fasta_to_json(data_dir, input_fasta_files, output_dir, num_training_seq=1000, num_validation_seq=100):
    random_seed = 42
    # Split fasta into json
    output_counts = fasta_to_json(input_fasta_files, output_dir, ntrain_seq=num_training_seq, nval_seq=num_validation_seq, random_seed=random_seed)

    return output_counts


# Modified from torchtitan hf_datasets.py
class HuggingFaceProteinDataset(IterableDataset, Stateful):
    """PyTorch Representation of the HuggingFace Dataset.

    Args:
        dataset_name (str): name of the dataset to load
        dataset_path (Optional[str]):
            Path to the dataset in the file system. If provided, data will be loaded
            from this path instead of downloaded.
        dataset_split (Optional[str]): name of the dataset split to load
        tokenizer (Tokenizer):
            Tokenizer used to encode data. Tokenize must implement an `encode` and `decode` method.
        seq_len (int): max sequence length
        world_size (int): number of data parallel processes participating in training
        rank (int): rank of the current data parallel process
        infinite (bool): whether to loop infinitely over the dataset

    """

    def __init__(
        self,
        dataset_name: str,
        dataset_path: Optional[str],
        dataset_split: Optional[str],
        tokenizer: Tokenizer,
        catalytic_tokenizer: Optional[Tokenizer],
        seq_len: int = 2048,
        world_size: int = 1,
        rank: int = 0,
        infinite: bool = False,
    ) -> None:

        logger.info(f"Preparing {dataset_name} dataset from {dataset_path}")
        ds = load_dataset(dataset_path, split=dataset_split)
        # TODO: support shuffling and checkpointing
        self.dataset_name = dataset_name
        self._data = split_dataset_by_node(ds, rank, world_size)
        self._tokenizer = tokenizer
        self._catalytic_tokenizer = catalytic_tokenizer
        self.seq_len = seq_len
        self.infinite = infinite
        self.num_repeats = 0

        # variables for checkpointing
        self._sample_idx = 0
        self._all_tokens: List[int] = []
        self._all_catalytic_tokens: List[int] = []

    def __iter__(self):
        max_buffer_token_len = 1 + self.seq_len

        while True:
            for sample in self._get_data_iter():
                sample_text = sample["seq"]
                sample_tokens = self._tokenizer.encode(sample_text)
                self._all_tokens.extend(sample_tokens)
                # Catalytic tokens if sample["catayltic_sites"] is available
                if sample["catalytic_sites"]:
                    sample_catalytic_tokens = self._catalytic_tokenizer.encode(sample["catalytic_sites"])
                    self._all_catalytic_tokens.extend(sample_catalytic_tokens)
                self._sample_idx += 1

                while len(self._all_tokens) >= max_buffer_token_len:  # Keep going while the current tokens are larger than the seq_len for the transformer
                    x = torch.LongTensor(self._all_tokens[:max_buffer_token_len])
                    # update tokens to the remaining tokens
                    self._all_tokens = self._all_tokens[max_buffer_token_len:]
                    # if self._all_catalytic_tokens:
                    if 1:
                        y = torch.LongTensor(self._all_catalytic_tokens[:max_buffer_token_len])
                        self._all_catalytic_tokens = self._all_catalytic_tokens[max_buffer_token_len:]
                    # if not self._all_catalytic_tokens:
                    #     print("No catalytic tokens")
                    input = (x[:-1], y[:-1])
                    label = (x[1:], y[1:])
                    # input = (x[:-1], y[:-1]) if self._all_catalytic_tokens else x[:-1]  # Return a tuple if catalytic tokens are available
                    # label = (x[1:], y[1:]) if self._all_catalytic_tokens else x[1:]  # Return a tuple if catalytic tokens are available
                    yield input, label

            if not self.infinite:
                logger.warning(f"Dataset {self.dataset_name} has run out of data.")
                break
            else:
                # Reset offset for the next iteration
                self._sample_idx = 0
                self.num_repeats += 1
                logger.warning(
                    f"Dataset {self.dataset_name} is being re-looped. "
                    "Loss related metrics might be misleading."
                )

    def _get_data_iter(self):
        if self._sample_idx == 0:
            return iter(self._data)

        # Skip samples
        if isinstance(self._data, IterableDataset):
            it = iter(self._data)
            # Naively iterate through the samples as skip may not be supported
            for _ in range(self._sample_idx):
                next(it)
            return it

        # As skipping to the end throws an error in case of map-style dataset, return an empty iterator
        if self._sample_idx == len(self._data):
            return iter([])
        return iter(self._data.skip(self._sample_idx))

    def load_state_dict(self, state_dict):
        self._sample_idx = state_dict["sample_idx"]
        self._all_tokens = state_dict["token_buffer"]
        self._all_catalytic_tokens = state_dict.get("catalytic_token_buffer", [])

    def state_dict(self):
        return {"token_buffer": self._all_tokens, "catalytic_token_buffer": self._all_catalytic_tokens, "sample_idx": self._sample_idx}


class DPAwareDataLoader(StatefulDataLoader, Stateful):
    """
    A wrapper around the StatefulDataLoader that ensures that the state is stored only once per DP rank.
    """

    def __init__(self, dp_rank: int, hf_ds: IterableDataset, batch_size: int):
        super().__init__(hf_ds, batch_size)
        self._dp_rank = dp_rank
        self._rank_id = f"dp_rank_{dp_rank}"

    def state_dict(self) -> Dict[str, Any]:
        # Store state only for dp rank to avoid replicating the same state across other dimensions
        return {self._rank_id: pickle.dumps(super().state_dict())}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        # State being empty is valid, don't log a warning
        if not state_dict:
            return

        if self._rank_id not in state_dict:
            logger.warning(
                f"DataLoader state is empty for dp rank {self._dp_rank}, expected key {self._rank_id}."
            )
            return
        super().load_state_dict(pickle.loads(state_dict[self._rank_id]))


def build_hf_data_loader(
    dataset_name: str,
    dataset_path: Optional[str],
    dataset_split: Optional[str],
    tokenizer: Tokenizer,
    catalytic_tokenizer: Optional[Tokenizer],
    batch_size: int,
    seq_len: int,
    world_size,
    rank,
    infinite: bool = True,
):
    hf_ds = HuggingFaceProteinDataset(
        dataset_name, dataset_path, dataset_split, tokenizer, catalytic_tokenizer, seq_len, world_size, rank, infinite
    )

    return DPAwareDataLoader(rank, hf_ds, batch_size=batch_size)
