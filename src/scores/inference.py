"""
Predicts scores for a single conversation using a pre-trained model.
The script prompts the user for a single thread id, which is the id
for a conversation stored in supabase.
"""

import sys
from pathlib import Path
from typing import List, Dict
import torch
import pandas as pd
from supabase import create_client, Client
from config import SUPABASE_URL, SUPABASE_SERVICE_KEY
from .model_builder import ConversationScorerModel
from .data_setup import ConversationPreprocessor


def load_conversation(thread_id: str) -> pd.DataFrame:
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


def tokenize_chat(
    chat_df: pd.DataFrame, max_length: int = 3000
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Tokenizes the conversation in a DataFrame
    """
    preproc = ConversationPreprocessor(max_length=max_length)
    conv_df = preproc.preprocess_conversations(chat_df)
    tokens = conv_df.loc[0, 'tokenized']
    return tokens['input_ids'], tokens['attention_mask']


def get_target_columns(
        data_dir: Path, scores_csv: str = "scores.csv"
) -> List[str]:
    """
    Reads the header of the scores csv file and gets the names of the targets
    """
    path = data_dir/scores_csv
    df = pd.read_csv(path, nrows=0)
    cols = [
        c for c in df.columns
        if c != "thread_id" and not c.startswith("Unnamed")
    ]
    return cols


def select_device() -> torch.device:
    """
    Selects the device for Pytorch to use
    """
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def load_model(
    model_dir: Path, num_targets: int, device: torch.device
) -> ConversationScorerModel:
    """
    Loads the pre-trained model
    """
    model = ConversationScorerModel(num_targets=num_targets)
    model_path = model_dir / "conversation_scorer.pth"
    if not model_path.exists():
        print(f"No pre-trained model located at {model_path}")
        sys.exit(1)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    return model.to(device).eval()


def run_inference(thread_id: str) -> Dict[str, float]:
    """
    Converts a target conversation to a dictionary of values to scores
    """
    # resolve paths
    root = Path(__file__).resolve().parents[2]
    data_dir = root / "data"
    model_dir = root / "models"

    # load and prepare chat
    chat_df = load_conversation(thread_id)
    targets = get_target_columns(data_dir)
    input_ids, attention_mask = tokenize_chat(chat_df)

    # load model to device
    device = select_device()
    model = load_model(model_dir, len(targets), device)

    # inference
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    with torch.no_grad():
        preds = model(input_ids, attention_mask)
    scores = preds.cpu().numpy().squeeze()
    return dict(zip(targets, scores.tolist()))


def main():
    thread_id = input("Enter the thread id for the conversation: ").strip()
    if not thread_id:
        print("No Thread id provided")
        sys.exit(1)

    results = run_inference(thread_id)
    print("\nPredicted Scores:")
    for name, val in results.items():
        print(f"{name}: {val:.3f}")


if __name__ == "__main__":
    main()
