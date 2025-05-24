"""
Utility functions
"""
from pathlib import Path
import pandas as pd
import torch

from src.scores.model_builder import ConversationScorerModel
from src.sentiment.data_setup import TextTokenizer
from src.sentiment.model_builder import SentimentLSTM, SentimentConfig


def get_root() -> Path:
    """
    Returns the root directory of the project
    """
    return Path(__file__).resolve().parent.parent


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
    """Loads the regression model"""
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
    """Loads the model to predict sentiments"""
    ckpt = Path(model_dir) / 'sentiment_model.pth'
    model = SentimentLSTM(tokenizer.hf_tokenizer, SentimentConfig(
        num_classes=2), device=str(device)).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return model
