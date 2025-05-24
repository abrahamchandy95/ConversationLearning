"""
Run sentiment inference on a single thread_id using a trained SentimentLSTM.
"""
import os
from typing import Tuple, List
import torch
from torch.utils.data import DataLoader
from transformers import logging
from transformers.data.data_collator import DataCollatorWithPadding
import pandas as pd
from supabase import create_client, Client

from src.utils.utils import get_root
from src.utils.db import load_conversation
from .data_setup import TextTokenizer, DatasetBuilder, DataLoaderBuilder
from .model_builder import SentimentLSTM, SentimentConfig


logging.set_verbosity_error()


def prepare_thread_loader(
    client: Client,
    thread_id: str,
    batch_size: int = 32,
    num_workers: int = 2
) -> Tuple[DataLoader, pd.DataFrame, TextTokenizer]:
    """
    Fetches messages for a thread_id from Supabase, aggregates user text,
    tokenizes it, and returns a DataLoader plus metadata.
    """
    df = load_conversation(client, thread_id)

    user_df = df[df['role'] != 'assistant']
    grouped = user_df.groupby('thread_id', as_index=False).agg(
        text=('content', lambda msgs: ' '.join(msgs.tolist()))
    )
    assert isinstance(grouped, pd.DataFrame), "Must be a DataFrame"
    tokenizer = TextTokenizer()
    ds_builder = DatasetBuilder(tokenizer)
    ds = ds_builder.from_dataframe(grouped[['text']], include_labels=False)

    collator = DataCollatorWithPadding(
        tokenizer.hf_tokenizer,
        padding='longest',
        return_tensors='pt'
    )
    loader_builder = DataLoaderBuilder(
        collator,
        batch_size=batch_size,
        num_workers=num_workers
    )
    # no val split, so use inference loader
    thread_loader = loader_builder.get_inference_loader(ds)

    return thread_loader, grouped, tokenizer


def load_model(
        device: torch.device, tokenizer: TextTokenizer
) -> SentimentLSTM:
    """
    Instantiate and load the trained SentimentLSTM model onto device.
    """
    root = get_root()
    model = SentimentLSTM(
        tokenizer.hf_tokenizer,
        SentimentConfig(num_classes=2),
        device=str(device)
    ).to(device)
    checkpoint = root / 'models' / 'sentiment_model.pth'
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    return model


def infer_probabilities(
    model: SentimentLSTM,
    loader: DataLoader
) -> List[float]:
    """
    Runs the model on loader to obtain positive-class probabilities.
    """
    device = next(model.parameters()).device
    probs_list: List[float] = []
    with torch.no_grad():
        for batch in loader:
            inputs = batch['input_ids'].to(device)
            masks = batch['attention_mask'].to(device)
            logits = model(inputs, masks)
            batch_probs = torch.softmax(logits, dim=1)[:, 1].cpu().tolist()
            probs_list.extend(batch_probs)
    return probs_list


def map_probability_to_label(
    prob: float,
    neg_thresh: float,
    pos_thresh: float
) -> int:
    """
    Maps a single probability to a 3-way sentiment label.
    """
    if prob < neg_thresh:
        return 0
    if prob > pos_thresh:
        return 4
    return 2


def run_thread_inference(
    client: Client,
    thread_id: str,
    neg_thresh: float = 0.4,
    pos_thresh: float = 0.6
) -> Tuple[int, float]:
    """
    Returns the sentiment prediction (0,2,4) and P_positive for a given thread.
    """
    loader, _, tokenizer = prepare_thread_loader(client, thread_id)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(device, tokenizer)
    prob = infer_probabilities(model, loader)[0]
    label = map_probability_to_label(prob, neg_thresh, pos_thresh)
    return label, prob


def main():
    """
    Calculates the sentiment of a given thread_id
    """
    client = create_client(
        os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"]
    )
    thread_id = input("Enter thread_id: ").strip()
    label, prob = run_thread_inference(client, thread_id)
    label_map = {0: 'negative', 2: 'neutral', 4: 'positive'}
    print(f"Thread {thread_id} → {label_map[label]} (P_positive={prob:.3f})")


if __name__ == '__main__':
    main()
