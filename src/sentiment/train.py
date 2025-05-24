"""
Train a sentiment classifier end-to-end using an LSTM model
"""
from pathlib import Path
from typing import Tuple, Dict
from transformers.data.data_collator import DataCollatorWithPadding
from transformers.optimization import get_linear_schedule_with_warmup
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
import pandas as pd
from tqdm.auto import tqdm

from src.utils.utils import get_root
from .download_dataset import download_and_extract
from .data_setup import DataLoaderBuilder, TextTokenizer, DatasetBuilder
from .model_builder import SentimentLSTM, SentimentConfig


def prepare_data(
    csv_path: Path,
    batch_size: int,
    val_split: float = 0.1,
    num_workers: int = 4
):
    """
    Downloads the data if unavailable, tokenizes and returns the
    train and validation dataloaders
    """
    download_and_extract()
    # read the csv
    df = pd.read_csv(
        csv_path,
        encoding="latin-1",
        header=None,
        usecols=[0, 5],
        names=["sentiment", "text"]
    )
    # Build the dataset
    tokenizer = TextTokenizer()
    dsbuilder = DatasetBuilder(tokenizer)
    ds = dsbuilder.from_dataframe(df, include_labels=True)
    # collator and loaders
    collator = DataCollatorWithPadding(
        tokenizer.hf_tokenizer,
        padding="longest",
        return_tensors="pt"
    )
    loader = DataLoaderBuilder(
        collator,
        batch_size=batch_size,
        num_workers=num_workers
    )
    train_loader, val_loader = loader.get_train_val_loaders(ds, val_split)
    return train_loader, val_loader, tokenizer


class Trainer:
    """
    Handles the training of the LSTM model
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: LRScheduler,
        criterion: nn.Module
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = next(model.parameters()).device

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[float, int]:
        """
        Runs a single training step on one batch.
        Returns (loss_value, num_correct)
        """
        inputs = batch["input_ids"].to(self.device)
        masks = batch["attention_mask"].to(self.device)
        labels = batch["labels"].long().to(self.device)

        self.optimizer.zero_grad()
        logits = self.model(inputs, masks)
        loss = self.criterion(logits, labels)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        preds = logits.argmax(dim=1)
        correct = (preds == labels).sum().item()
        return loss.item(), correct

    def eval_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[float, int]:
        """
        Runs a single evaluation step on one batch.
        Returns (loss_value, num_correct)
        """
        inputs = batch["input_ids"].to(self.device)
        masks = batch["attention_mask"].to(self.device)
        labels = batch["labels"].long().to(self.device)

        with torch.no_grad():
            logits = self.model(inputs, masks)
            loss = self.criterion(logits, labels)
            preds = logits.argmax(dim=1)
            correct = int((preds == labels).sum().item())
        return loss.item(), correct

    def train_epoch(self, loader: DataLoader) -> Tuple[float, float]:
        """
        Trains all batches for a single epoch
        """
        total_loss, total_correct, total_samples = 0.0, 0, 0
        for batch in tqdm(loader, desc="Training Epoch"):
            loss, correct = self.train_step(batch)
            batch_size = batch['labels'].size(0)
            total_loss += loss * batch_size
            total_correct += correct
            total_samples += batch_size
        return total_loss/total_samples, total_correct/total_samples

    def val_epoch(self, loader: DataLoader) -> Tuple[float, float]:
        """
        Runs all batches in the loader. Returns (avg loss, accuracy) for val
        """
        total_loss, total_correct, total_samples = 0.0, 0, 0
        for batch in tqdm(loader, desc="Validation"):
            loss, correct = self.eval_step(batch)
            batch_size = batch['labels'].size(0)
            total_loss += loss * batch_size
            total_correct += correct
            total_samples += batch_size
        return total_loss/total_samples, total_correct/total_samples

    def run_epochs(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int
    ) -> None:
        """
        Trains the model on the dataset for all the epochs
        """
        best_acc = 0.0
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.val_epoch(val_loader)

            print(
                f"Epoch {epoch}: "
                f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
            )

            if val_acc > best_acc:
                best_acc = val_acc
                root = get_root()
                model_path = root / "models" / "sentiment_model.pth"
                model_path.parent.mkdir(exist_ok=True)
                torch.save(self.model.state_dict(), model_path)
                print(f"→ Saved better model at val_acc={val_acc:.4f}")


def main():
    """
    End-to-end training of the Sentiment140 dataset
    """
    root = get_root()
    filename = "training.1600000.processed.noemoticon.csv"
    csv_path = root.joinpath("data", "SENTIMENT140", filename)
    num_epochs = 3

    train_loader, val_loader, tokenizer = prepare_data(
        csv_path, batch_size=64
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SentimentLSTM(
        tokenizer.hf_tokenizer,
        SentimentConfig(num_classes=2),
        device=str(device)
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=2e-4)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    trainer = Trainer(model, optimizer, scheduler, criterion)
    trainer.run_epochs(train_loader, val_loader, num_epochs)


if __name__ == "__main__":
    main()
