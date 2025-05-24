"""
Module to handle preprocessing of text data - tokenizing and loading.
Preprocessing to get the text data ready for training
"""
import os
from typing import List, Dict, Tuple, Optional, Any, cast
import pandas as pd
from datasets import Dataset
import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from transformers.data.data_collator import DataCollatorWithPadding


# Ensure tokenizers parallelism is disabled before any worker fork
TOK_ENV = "TOKENIZERS_PARALLELISM"
if os.getenv(TOK_ENV) is None:
    os.environ[TOK_ENV] = "false"


class TorchReadyDataset(TorchDataset):
    """
    Adapts a Hugging Face Dataset for seamless use with Pytorch DataLoaders.
    """

    def __init__(self, ds: Dataset) -> None:
        self.dataset = ds.with_format("torch")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.dataset[idx]


class TextTokenizer:
    """
    Wraps a multilingual HuggingFace tokenizer for text encoding
    """

    def __init__(
        self,
        model_name: str = "xlm-roberta-base",
        max_length: int = 256
    ):
        self.model_name = model_name
        self.max_length = max_length
        self._tokenizer: Optional[PreTrainedTokenizer] = None

    def _get_tokenizer(self) -> PreTrainedTokenizer:
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        assert self._tokenizer is not None
        return self._tokenizer

    def encode_batch(self, texts: List[str]) -> BatchEncoding:
        """
        Tokenizes a batch of strings into input ids and attention masks
        """
        tokenizer = self._get_tokenizer()
        return tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors=None
        )

    def encode(self, text: str) -> BatchEncoding:
        """
        Tokenize a single text string
        """
        tokenizer = self._get_tokenizer()
        return tokenizer(
            text,
            padding=False,
            truncation=True,
            max_length=self.max_length,
            return_tensors=None
        )

    @property
    def hf_tokenizer(self) -> PreTrainedTokenizer:
        """
        Returns the underlying HuggingFace tokenizer
        """
        return self._get_tokenizer()

    @property
    def vocab_size(self) -> int:
        """
        Returns the size of the tokenizer's vocabulary
        """
        return self._get_tokenizer().vocab_size

    @property
    def pad_token_id(self) -> int:
        """
        Returns the padding token ID of the tokenizer
        """
        tokenizer = self._get_tokenizer()
        if tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer has no pad_token_id.")
        return cast(int, tokenizer.pad_token_id)


class DatasetBuilder:
    """
    Converts a compatible data structure into a HuggingFace dataset
    """

    def __init__(self, tokenizer: TextTokenizer):
        self.tokenizer = tokenizer

    def from_dataframe(
        self,
        df: pd.DataFrame,
        include_labels: bool = True
    ) -> Dataset:
        """
        Wraps a pandas DataFrame in a HF Dataset and tokenizes it
        """
        ds = Dataset.from_pandas(df)
        return self.tokenize_dataset(ds, include_labels)

    def from_dict(
        self,
        data: Dict[str, List[Any]],
        include_labels: bool = False
    ) -> Dataset:
        """
        Wraps a dict of lists in a HF Dataset and tokenizes it
        """
        ds = Dataset.from_dict(data)
        return self.tokenize_dataset(ds, include_labels)

    def tokenize_dataset(
        self,
        ds: Dataset,
        include_labels: bool = True
    ) -> Dataset:
        """
        Applies the tokenizer to a HF Dataset
        """
        def _map(batch):
            enc = self.tokenizer.encode_batch(batch['text'])
            if include_labels and "sentiment" in batch:
                enc["labels"] = [
                    0 if s == 0 else 1 for s in batch["sentiment"]
                ]
            return enc

        remove_cols = []
        if "text" in ds.column_names:
            remove_cols.append("text")
        if not include_labels and "sentiment" in ds.column_names:
            remove_cols.append("sentiment")

        return ds.map(
            _map,
            batched=True,
            remove_columns=remove_cols
        )


class DataLoaderBuilder:
    """
    Builds Pytorch DataLoaders for training, validation, testing and inference
    """

    def __init__(
        self,
        collator: DataCollatorWithPadding,
        batch_size: int = 64,
        num_workers: int = 4
    ):
        self.collator = collator
        self.batch_size = batch_size
        self.num_workers = num_workers

    def get_train_val_loaders(
        self,
        ds: Dataset,
        val_split: float = 0.1,
        seed: int = 42
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Outputs train and validation dataloaders from tokenized Datasets
        """
        splits = ds.train_test_split(test_size=val_split, seed=seed)
        train_ds = TorchReadyDataset(splits["train"])
        val_ds = TorchReadyDataset(splits["test"])

        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collator
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collator
        )
        return train_loader, val_loader

    def get_inference_loader(
        self,
        ds: Dataset,
        batch_size: Optional[int] = None
    ) -> DataLoader:
        """
        Creates a DataLoader for inference or testing
        """
        dataset = TorchReadyDataset(ds)
        bs = batch_size or self.batch_size
        return DataLoader(
            dataset=dataset,
            batch_size=bs,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collator
        )
