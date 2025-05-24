"""
Compute insights like topics, image counts, interaction counts
for a single thread id
"""
import os
from typing import Dict, Any
import pandas as pd
from transformers.utils import logging
from supabase import Client, create_client

from src.utils.db import load_conversation
from src.utils.nlp import translate, extract_topics, count_requests


def count_interactions(df: pd.DataFrame) -> int:
    """Counts the number of user messages in a thread"""
    return int((df['role'] != 'assistant').sum())


def run_insights(client: Client, thread_id: str) -> Dict[str, Any]:
    """Returns all insights for a single thread id"""
    df = load_conversation(client, thread_id)
    if df.empty:
        raise ValueError(f"No conversation for thread {thread_id}")
    user_df = df[df["role"] != "assistant"]
    full_text = ' '.join(user_df["content"].tolist())

    translated = translate(full_text)
    topics = extract_topics(translated)
    imgs, mms = count_requests(translated)
    interactions = count_interactions(df)

    return {
        "thread_id": thread_id,
        "central_topics": topics,
        "image_requests": imgs,
        "mindmap_requests": mms,
        "interactions": interactions
    }


def main():
    """Connects logic"""
    logging.set_verbosity_error()
    client = create_client(
        os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"]
    )
    # nlp models
    thread_id = input("Enter thread id: ").strip()
    insights = run_insights(client, thread_id)
    for k, v in insights.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
