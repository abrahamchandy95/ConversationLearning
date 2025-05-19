"""
Utility functions
"""
from pathlib import Path
import pandas as pd
from supabase import create_client, Client
import torch

from src.scores.model_builder import ConversationScorerModel
from src.sentiment.data_setup import TextTokenizer
from src.sentiment.model_builder import SentimentLSTM, SentimentConfig


def get_root() -> Path:
    """
    Returns the root directory of the project
    """
    return Path(__file__).resolve().parent.parent


def load_conversation(
        thread_id: str,
        supabase_url: str,
        supabase_key: str
) -> pd.DataFrame:
    """
    Queries Supabase for chat messages with the given thread id.
    Args:
        thread_id: The thread id for which to load chat messages.
    Returns:
        A pandas DataFrame with chat messages for that thread,
        with 'created_at' converted to datetime and sorted.
    """
    supabase: Client = create_client(supabase_url, supabase_key)
    response = (
        supabase.table("chat_messages")
        .select('*')
        .eq("thread_id", thread_id)
        .execute()
    )
    data = response.data
    df = pd.DataFrame(data)
    if not df.empty:
        df['created_at'] = pd.to_datetime(df['created_at'])
        df = df.sort_values(['thread_id', 'created_at'])
    return df


def load_chats(
    supabase_url: str,
    supabase_service_key: str
) -> pd.DataFrame:
    """
    Loads chat messages from Supabase and returns a DataFrame.

    Args:
        supabase_url: The URL for the Supabase instance.
        supabase_service_key: The Supabase service key for authentication.

    Returns:
        A pandas DataFrame containing chat messages.
    """
    supabase: Client = create_client(supabase_url, supabase_service_key)
    response = supabase.table("chat_messages").\
        select('*', count='exact').execute()
    total_rows = response.count if response.count is not None else 0
    batch_size = 1000
    all_chats = []

    # Fetch data in batches
    for start in range(0, total_rows, batch_size):
        end = min(start + batch_size - 1, total_rows - 1)
        response = supabase.table("chat_messages") \
            .select("*, chat_threads(user_id)") \
            .range(start, end) \
            .execute()

        # Flatten nested chat_threads structure
        for item in response.data:
            if "chat_threads" in item and isinstance(item["chat_threads"], dict):
                item["user_id"] = item["chat_threads"].get("user_id")
            else:
                item["user_id"] = None
            del item["chat_threads"]

        all_chats.extend(response.data)

    # Convert to DataFrame
    df = pd.DataFrame(all_chats)
    df['created_at'] = pd.to_datetime(df['created_at'])
    df.sort_values(['thread_id', 'created_at'], inplace=True)
    return df


def load_scores(
    scores_file: str = "scores.csv"
) -> pd.DataFrame:
    """
    Loads scores data from the data directory in the project's root
    """
    root = get_root()
    path = root / "data" / scores_file
    scores_df: pd.DataFrame = pd.read_csv(path)

    cols = [
        c for c in scores_df.columns
        if not c.startswith("Unnamed")
    ]
    scores = scores_df[cols]
    assert isinstance(scores, pd.DataFrame)
    return scores


def save_model(
    model: torch.nn.Module,
    model_name: str,
    target_dir: str = "models"
) -> None:
    """Saves a PyTorch model to a target directory.

    Args:
        model: A target PyTorch model to save.
        model_name: A filename for the saved model. Should include
            either ".pth" or ".pt" as the file extension.
        target_dir: A directory for saving the model, defaulting to "models"
            (located in the project root).

    Example usage:
        save_model(
            model=model_0,
            model_name="my_regression_model.pth"
        )
    """
    # Assume that this function will be run in src folder
    root = get_root()
    save_dir = root / target_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    assert model_name.endswith(".pth") or model_name.endswith(".pt"), \
        "Model should end with '.pt' or .pth'"
    model_save_path = save_dir / model_name

    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)


def select_device(verbose: bool = False) -> torch.device:
    """
    Returns the available device
    """
    if torch.backends.mps.is_available():
        dev = torch.device("mps")
    elif torch.cuda.is_available():
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")
    if verbose:
        print(f"Using device: {dev}")
    return dev


def load_score_model(
    model_dir: Path,
    num_targets: int,
    device: torch.device
) -> ConversationScorerModel:
    model = ConversationScorerModel(num_targets=num_targets).to(device)
    model.eval()
    ckpt = model_dir / 'conversation_scorer.pth'
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    return model


def load_sentiment_model(
    model_dir: Path,
    tokenizer: TextTokenizer,
    device: torch.device
) -> SentimentLSTM:
    ckpt = Path(model_dir) / 'sentiment_model.pth'
    model = SentimentLSTM(tokenizer.hf_tokenizer, SentimentConfig(
        num_classes=2), device=str(device)).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return model
