from pathlib import Path
import pandas as pd
from supabase import create_client, Client
import torch

def load_conversation(
        thread_id: str, 
        supabase_url: str,
        supabase_key: str        
)-> pd.DataFrame:
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
    response = supabase.table("chat_messages").select("*").execute()
    data = response.data
    df = pd.DataFrame(data)
    df['created_at'] = pd.to_datetime(df['created_at'])
    df = df.sort_values(['thread_id', 'created_at'])

    return df

def load_scores(
    scores_file: str = "scores.csv"
)-> pd.DataFrame:
    """
    Loads scores data from the data directory in the project's root
    """

    current_dir = Path(__file__).resolve().parent  # This is the src directory.
    project_root = current_dir.parent
    scores_path = project_root / "data" / scores_file
    scores_df = pd.read_csv(scores_path)

    return scores_df

def save_model(
    model: torch.nn.Module,
    model_name: str,
    target_dir: str = "models"
):
    """Saves a PyTorch model to a target directory relative to the project root.

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
    current_dir = Path(__file__).resolve().parent  # This is the src directory.
    project_root = current_dir.parent

    # Build the target directory path relative to the project root.
    target_dir_path = project_root / target_dir
    target_dir_path.mkdir(parents=True, exist_ok=True)

    assert model_name.endswith(".pth") or model_name.endswith(".pt"), \
    "Model should end with '.pt' or .pth'"
    model_save_path = target_dir_path / model_name

    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)
