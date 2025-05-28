"""
Handles vectorized inference of multiple thread_ids
leveraging faster computation
"""
import os
from typing import List, Tuple, Dict, Set
from pathlib import Path
import sys
from requests.exceptions import HTTPError
from dotenv import load_dotenv
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers.utils import logging
from supabase import create_client, Client
from sqlalchemy.engine.reflection import Inspector
from sqlalchemy import Engine

from src.scores.data_setup import (
    ConversationPreprocessor, ConversationDataset
)
from src.scores.inference import get_target_columns
from src.sentiment.data_setup import TextTokenizer
from src.utils.nlp import translate, extract_topics, count_requests
from src.utils.utils import get_root, load_score_model, load_sentiment_model
from src.utils.device import select_device
from src.utils.db import load_thread_ids, TableFetcher


def aggr_user_messages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a dataframe with only the user messages
    """
    user_df = df[df['role'] != 'assistant']
    return (
        user_df
        .sort_values(['thread_id', 'created_at'])
        .groupby('thread_id', as_index=False)
        .agg(text=('content', lambda msgs: ' '.join(msgs.tolist())))
    )


def compute_interactions(df: pd.DataFrame) -> Dict[str, int]:
    """
    Returns a dict mapping thread_id to count of user messages.
    """
    user_df = df[df['role'] != 'assistant']
    return user_df.groupby('thread_id').size().to_dict()


def prepare_score_loader(
    df: pd.DataFrame,
    batch_size: int = 32
) -> Tuple[DataLoader, pd.DataFrame]:
    """
    Prepares conversation for the scorer model
    """
    conv_df = ConversationPreprocessor(max_length=3000)\
        .preprocess_conversations(df)
    ds = ConversationDataset(conv_df, target_columns=None)
    loader = DataLoader(ds, batch_size=batch_size)
    return loader, conv_df


def infer_scores(
    loader: DataLoader,
    model: torch.nn.Module,
    use_device: torch.device,
    target_dim: int
) -> torch.Tensor:
    """
    Runs batch inference on a DataLoader and returns scores
    """
    scores: List[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for input_ids, attention_mask in loader:
            input_ids = input_ids.to(use_device)
            attention_mask = attention_mask.to(use_device)
            preds = model(input_ids, attention_mask)
            scores.append(preds.cpu())
    return (
        torch.cat(scores, dim=0) if scores else torch.empty((0, target_dim))
    )


def compute_scores(
    df: pd.DataFrame, saved_model_dir: Path
) -> pd.DataFrame:
    """
    Scores a number of conversations extracted from a table in supabase
    """
    device = select_device()
    data_dir = get_root() / 'data'
    target_cols = get_target_columns(data_dir)
    model = load_score_model(saved_model_dir, len(target_cols), device)
    # Preprocess conversations
    loader, conv_df = prepare_score_loader(df)
    scores_tensor = infer_scores(loader, model, device, len(target_cols))
    # Build DataFrame
    scores = scores_tensor.numpy()
    out = pd.DataFrame(scores, columns=target_cols)
    out['thread_id'] = conv_df['thread_id'].tolist()
    cols = ['thread_id'] + list(target_cols)
    return out[cols]


def compute_sentiments(
    df: pd.DataFrame,
    sentiment_model_dir: Path,
    neg_thresh: float = 0.4,
    pos_thresh: float = 0.6
) -> pd.DataFrame:
    """
    Returns sentiments from a number of conversations.
    Three categories of sentiments are positive, negative and neutral
    """

    user_conv = aggr_user_messages(df)
    device = select_device()

    # Build inference loader
    tokenizer = TextTokenizer()
    model = load_sentiment_model(sentiment_model_dir, tokenizer, device)
    model.eval()
    enc = tokenizer.hf_tokenizer(
        user_conv['text'].tolist(),
        padding=True,
        truncation=True,
        max_length=tokenizer.max_length,
        return_tensors='pt'
    ).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(enc['input_ids'], enc['attention_mask'])
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().tolist()

    labels = [
        'negative' if p < neg_thresh else
        'positive' if p > pos_thresh else
        'neutral' for p in probs
    ]

    return pd.DataFrame({
        'thread_id': user_conv['thread_id'],
        'sentiment_label': labels,
        'p_positive': probs
    })


def compute_other_insights(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns other insights from the conversation
    """
    user_conv = aggr_user_messages(df)
    # translate
    user_conv['translation'] = translate(user_conv['text'].tolist())
    # extract topics
    user_conv['central_topics'] = (
        user_conv['translation'].apply(extract_topics)
    )
    requests = user_conv['translation'].apply(count_requests)

    user_conv[['image_requests', 'mindmap_requests']] = pd.DataFrame(
        requests.tolist(), index=user_conv.index
    )
    # map interactions
    interactions = compute_interactions(df)
    user_conv['interactions'] = user_conv['thread_id'].\
        map(interactions).fillna(0).astype(int)

    cols = [
        'thread_id',
        'central_topics',
        'image_requests',
        'mindmap_requests',
        'interactions'
    ]
    return user_conv[cols]


def fetch_new_thread_ids(client: Client) -> Set[str]:
    """Helper function that gets unprocessed conversations"""
    all_ids = load_thread_ids(client, table="chat_messages")
    try:
        completed_ids = load_thread_ids(client, table="chat_inference")
    except HTTPError:
        completed_ids = set()
    return all_ids - completed_ids


def compute_inference(
    df: pd.DataFrame, model_dir: Path
) -> pd.DataFrame:
    scores_df = compute_scores(df, model_dir)
    sentiments_df = compute_sentiments(df, model_dir)
    insights_df = compute_other_insights(df)

    merged = scores_df.merge(sentiments_df, on="thread_id")\
        .merge(insights_df, on="thread_id")
    return merged


def upsert_inference_to_supabase(
    client: Client, inference_df: pd.DataFrame
) -> None:
    """Inserts results to the chat_inference table"""
    records = inference_df.to_dict("records")
    if not records:
        return
    client.table("chat_inference").upsert(records).execute()
    print(f"Upserted {len(records)} rows into chat_inference")


if __name__ == "__main__":
    load_dotenv()
    logging.set_verbosity_error()
    model_dir = get_root() / "models"

    # fetch all distinct thread_ids from 'chat messages' table
    supabase = create_client(
        os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"]
    )
    ids_todo = fetch_new_thread_ids(supabase)
    if not ids_todo:
        print("No new thread ids to process")
        sys.exit(0)

    fetcher = TableFetcher(
        client=supabase,
        table="chat_messages",
    )

    # this returns a DataFrame with all columns from chat_messages for the IDs
    conv_df = fetcher.fetch_all(thread_ids=list(ids_todo))
    inference_df = compute_inference(conv_df, model_dir)
    upsert_inference_to_supabase(supabase, inference_df)
    print(inference_df.head())
