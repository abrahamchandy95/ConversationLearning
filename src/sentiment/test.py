"""
Evaluates a trained sentiment model on the Sentiment140 test dataset.
"""
from pathlib import Path
from typing import Tuple, List
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers.data.data_collator import DataCollatorWithPadding
from tqdm.auto import tqdm

from src.utils.utils import get_root
from .data_setup import TextTokenizer, DatasetBuilder, DataLoaderBuilder
from .model_builder import SentimentConfig, SentimentLSTM


def prepare_test_data(
    test_csv: Path,
    batch_size: int,
    num_workers: int
) -> Tuple[DataLoader, List[int], TextTokenizer]:
    """
    Prepares the test dataset to be passed into the model
    """
    df_test = pd.read_csv(
        test_csv,
        encoding="latin-1",
        header=None,
        usecols=[0, 5],
        names=["sentiment", "text"]
    )
    true_labels = df_test["sentiment"].tolist()

    # build tokenizer without labels
    tokenizer = TextTokenizer()
    dsbuilder = DatasetBuilder(tokenizer)
    ds = dsbuilder.from_dataframe(df_test, include_labels=False)

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
    test_loader = loader.get_inference_loader(ds)
    return test_loader, true_labels, tokenizer


def load_model(tokenizer: TextTokenizer) -> SentimentLSTM:
    """
    Initializes and loads the trained model
    """
    root = get_root()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentimentLSTM(
        tokenizer.hf_tokenizer,
        SentimentConfig(num_classes=2),
        device=str(device)
    ).to(device)
    checkpoint = root / "models" / "sentiment_model.pth"
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    return model


def infer(
        model: SentimentLSTM,
        loader: DataLoader,
        neg_thresh: float = 0.40,
        pos_thresh: float = 0.60
) -> List[int]:
    """
    Runs inference over the test DataLoader
    """
    device = next(model.parameters()).device
    preds: List[int] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Testing"):
            inputs = batch["input_ids"].to(device)
            masks = batch["attention_mask"].to(device)
            logits = model(inputs, masks)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu()
            for p in probs.tolist():
                if p < neg_thresh:
                    preds.append(0)
                elif p > pos_thresh:
                    preds.append(4)
                else:
                    preds.append(2)
    return preds


def compute_accuracy(
        predictions: List[int],
        ground_truths: List[int]
) -> float:
    """
    Computes the accuracy of the trained model
    """
    corrects = sum(p == t for p, t in zip(predictions, ground_truths))
    return (corrects / len(ground_truths))*100


def main():
    """
    Tests the model on the test dataset from Sentiment140
    """
    root = get_root()
    filename = "testdata.manual.2009.06.14.csv"
    test_csv = root.joinpath("data", "SENTIMENT140", filename)
    test_loader, ground_truths, tokenizer = prepare_test_data(
        test_csv, batch_size=64, num_workers=4
    )
    model = load_model(tokenizer)
    preds = infer(model, test_loader)
    acc = compute_accuracy(preds, ground_truths)
    print(f"Test accuracy: {acc:.2f}")


if __name__ == "__main__":
    main()
