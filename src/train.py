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

def plot_loss_curves(results: dict, save_path: str = "results/loss_curves.png") -> None:
    """Plots training and validation loss curves from training results.
    
    Args:
        results (dict): Dictionary containing training metrics
        save_path (str): Path to save the plot image
    """
    plt.figure(figsize=(10, 6))
    
    # Get loss values
    train_loss = results['train_loss']
    val_loss = results['val_loss']
    epochs = range(1, len(train_loss) + 1)
    
    # Plot losses
    plt.plot(epochs, train_loss, label='Train Loss', marker='o')
    plt.plot(epochs, val_loss, label='Validation Loss', marker='o')
    
    # Add best epoch marker
    best_epoch = val_loss.index(min(val_loss)) + 1
    plt.axvline(x=best_epoch, color='r', linestyle='--', 
                label=f'Best Epoch ({best_epoch})')
    
    # Style plot
    plt.title("Training and Validation Loss Curves")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.grid(True)
    
    # Create results directory if needed
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved loss curves to {save_path}")

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
        plot_loss_curves(results, save_path="results/loss_curves.png")
if __name__ == "__main__":
    main()
