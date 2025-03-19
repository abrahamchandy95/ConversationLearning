"""
Trains a Pytorch regression model for conversation scoring using data
from Supabase
"""
import argparse
import math
from typing import Tuple, List
import os
import sys

from timeit import default_timer as timer
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from config import SUPABASE_URL, SUPABASE_SERVICE_KEY
from .data_setup import ConversationPreprocessor, ConversationDataset
from .model_builder import BuddyRegressionModel
from .engine import train
from .utils import load_conversation, load_scores, save_model

def load_inputs_and_targets()-> Tuple[pd.DataFrame, List[str]]:
    """
    Loads the inputs from Supabase and our ground-truth values from
    root/data/scores.csv

    Returns:
        merged_df: A pandas DataFrame with merged inputs and targets
        target_names: A list of target column names extracted from
        the scores dataframe
    """
    scores_df = load_scores("scores.csv")
    thread_ids = scores_df['thread_id'].unique().tolist()
    target_names = scores_df.columns.drop("thread_id").tolist()
    # for each thread_id, load and preprocess the conversation, then merge
    preprocessor = ConversationPreprocessor(max_length=3000)
    valid_convs = pd.DataFrame()
    
    for tid in thread_ids:
        chat_df = load_conversation(tid, SUPABASE_URL, SUPABASE_SERVICE_KEY)
        if chat_df.empty:
            continue
        tokenized_chat = preprocessor.preprocess_conversations(chat_df)
        valid_convs = pd.concat([valid_convs, tokenized_chat], ignore_index=True)

    merged_df = valid_convs.merge(scores_df, on="thread_id", how="inner")

    return merged_df, target_names

def plot_predictions_and_targets(
    dataset: torch.utils.data.Dataset,
    model: torch.nn.Module,
    target_names: List[str],
    device: torch.device,
    batch_size: int = 1
) -> None:
    """
    Plots predicted vs. ground truth values for all target columns in a
    grid of subplots.

    Args:
        dataset (torch.utils.data.Dataset): Dataset yielding
        (input_ids, attention_mask, targets).
        model (torch.nn.Module): The trained PyTorch regression model.
        target_columns (List[str]): List of target column names.
        device (torch.device): Device to run inference on.
        batch_size (int): Batch size for the DataLoader.

    Returns:
        None. Displays a grid of scatter plots for all target columns.
    """
    # DataLoader object to iterate over the dataset
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, targets = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            preds = model(input_ids, attention_mask)
            all_preds.append(preds.cpu())
            all_targets.append(targets)

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    num_targets = len(target_names)
    # make a grid for subplots
    cols = math.ceil(math.sqrt(num_targets))
    rows = math.ceil(num_targets/cols)

    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(cols*4, rows*4))

    if num_targets == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx in range(num_targets):
        ax = axes[idx]
        ax.scatter(
            all_targets[:, idx].numpy(), all_preds[:, idx].numpy(), alpha=0.7
        )
        ax.set_xlabel("Ground Truth")
        ax.set_ylabel("Predicted")
        ax.set_title(target_names[idx])

    # delete unused subplots
    for idx in range(num_targets, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train a regression model for conversation scoring, with"
            "an option to see a plot of predicted vs ground truth values"
        )
    )
    parser.add_argument(
            "--plot",
            action="store_true",
            help="Optional: Plot predicted vs ground truth values"
    )
    args = parser.parse_args()

    # Hyperparameters and device setup
    NUM_EPOCHS = 10
    BATCH_SIZE = 2
    LEARNING_RATE = 1e-4
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    # load the scored_conversations and the target names
    conv_df, target_names = load_inputs_and_targets()

    # Create datasets
    dataset = ConversationDataset(conv_df, target_names)

    # create model
    num_targets = len(target_names)
    model = BuddyRegressionModel(num_targets=num_targets)
    model.to(device)

    # Train the model using engine.train
    start_time = timer()
    results = train(
        model=model,
        dataset=dataset,
        epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        split=0.1,
        device=device
    )
    end_time = timer()
    print(f"Total training time: {end_time - start_time:.3f} seconds")
    print("Training Results:")
    print(results)

    # save model
    save_model(
        model, model_name="conversation_scorer.pth", target_dir="models"
    )
    # Optional: Plot predictions vs. ground truth.
    if args.plot:
        plot_predictions_and_targets(
            dataset, model, target_names, device, batch_size=BATCH_SIZE
        )

if __name__ == "__main__":
    main()
