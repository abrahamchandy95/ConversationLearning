"""
Predicts scores for a single conversation using a pre-trained model.
The script prompts the user for a single thread id, which is the id
for a conversation stored in supabase.
"""

import sys
import os
from typing import List
import torch
import pandas as pd
from pathlib import Path
from supabase import create_client, Client

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import SUPABASE_URL, SUPABASE_SERVICE_KEY
from .data_setup import ConversationPreprocessor
from .model_builder import BuddyRegressionModel

def load_conversation(thread_id: str)-> pd.DataFrame:
    """
    Queries supabase for the precific conversation
    """
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    response = (
        supabase.table("chat_messages")
        .select("*")
        .eq("thread_id", thread_id)
        .execute()
    )
    data = response.data
    df = pd.DataFrame(data)
    if not df.empty:
        df['created_at'] = pd.to_datetime(df['created_at'])
        df = df.sort_values(['thread_id', 'created_at'])
    return df

def get_target_columns(scores_csv: str = "scores.csv")-> List[str]:
    """
    Reads the header of the scores csv file and gets the names of the targets
    """
    root_dir = Path(__file__).resolve().parent.parent
    scores_path = root_dir / "data" / scores_csv
    df = pd.read_csv(scores_path, nrows=0)
    target_columns = df.columns.drop("thread_id").tolist()
    return target_columns

def main():
    thread_id = input("Enter the thread id for the conversation: ").strip()
    if not thread_id:
        print("No Thread id provided")
        sys.exit(1)
    chat = load_conversation(thread_id)
    if chat.empty:
        print(f"No chat found for the thread id: {thread_id}")
        sys.exit(1)
    target_columns = get_target_columns("scores.csv")
    # ensure that the conversation length matches the training max
    preprocessor = ConversationPreprocessor(max_length=3000)
    conv_df = preprocessor.preprocess_conversations(chat)
    tokenized = conv_df['tokenized'].iloc[0]
    input_ids = tokenized['input_ids']
    attention_mask = tokenized['attention_mask']

    # device setup
    device = torch.device(
            "mps" if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    # load trained model
    num_targets = len(target_columns)
    model = BuddyRegressionModel(num_targets=num_targets)

    root_dir = Path(__file__).resolve().parent.parent
    model_path = root_dir / "models" / "conversation_scorer.pth"

    if not model_path.exists():
        print(f"No pre-trained model found at {model_path}.")
        print(
            "Please run 'python -m src.train' (or 'python -m src.train --plot')"
            "to train the model first."
        )
        sys.exit(1)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    # move inputs to device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    with torch.no_grad():
        predictions = model(input_ids, attention_mask)
    predictions_np = predictions.cpu().numpy().squeeze()

    # Print the predicted scores with the target names
    print("\nPredicted Scores:")
    for name, pred in zip(target_columns, predictions_np):
          print(f"{name}: {pred:.3f}")

if __name__ == "__main__":
    main()
