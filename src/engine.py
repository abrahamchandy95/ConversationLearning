from typing import Dict, List, Optional, Tuple
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from .data_setup import ConversationDataset

def train_step(
        model: torch.nn.Module,
        dataloader: DataLoader,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device
)-> float:
    """
        Performs training for one epoch.
    """
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch in dataloader:
        input_ids, attention_mask, targets = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        predictions = model(input_ids, attention_mask)
        loss = loss_fn(predictions, targets)
        loss.backward()
        optimizer.step()

        batch_size = input_ids.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    avg_loss = total_loss / total_samples

    return avg_loss

def val_step(
        model: torch.nn.Module,
        dataloader: DataLoader,
        loss_fn: torch.nn.Module,
        device: torch.device
) -> float:

    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.inference_mode():
        for batch in dataloader:
            input_ids, attention_mask, targets = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            targets = targets.to(device)

            predictions = model(input_ids, attention_mask)
            loss = loss_fn(predictions, targets)
            batch_size = input_ids.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        avg_loss = total_loss / total_samples

    return avg_loss

def compute_metrics(
    dataloader: DataLoader,
    model: torch.nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """
    Computes Mean Asbolute Error (MAE) and R^2 score for a DataLoader
    object.

    Args:
        dataloader: DataLoader yielding (input_ids, attention_mask, targets).
        model: The PyTorch model.
        device: The device to run inference on.

    Returns:
        A tuple (mae, r2) where:
         - mae is the average absolute error.
         - r2 is the coefficient of determination.
    """
    model.eval()
    all_preds = []
    all_targets = []
    with torch.inference_mode():
        for batch in dataloader:
            input_ids, attention_mask, targets = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            targets = targets.to(device)
            preds = model(input_ids, attention_mask)
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # compute MAE
    mae = torch.mean(torch.abs(all_preds - all_targets)).item()

    # compute R^2
    # R^2 = 1 - (Sum of Square of Residuals/Total Sum of Squares)
    ss_res = torch.sum((all_targets - all_preds) ** 2)
    ss_tot = torch.sum((all_targets - torch.mean(all_targets))** 2)
    r_sq = 1 - (ss_res / ss_tot).item() if ss_tot > 0 else float('nan')

    return mae, r_sq


def train(
        model: torch.nn.Module,
        dataset: ConversationDataset,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        split: float = 0.1,
        device: Optional[torch.device] = None
) -> Dict[str, List[float]]:
    """
        Trains the model and returns training and validation losses.

        Args:
            model: The PyTorch model to be trained.
            dataset: A Dataset instance that is Sized and returns
            (input_ids, attention_mask, targets).
            epochs: Number of training epochs.
            batch_size: Batch size.
            learning_rate: Learning rate for the optimizer.
            split: Fraction of the dataset to use for validation.
            device: Will select mps, cuda, or cpu.

        Returns:
            A dictionary of train_loss and val_loss per epoch.
    """
    if device is None:
        device = torch.device(
            "mps" if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
    model.to(device)
    # split the dataset into training and validation sets
    total = len(dataset)
    n_val = int(total * split)
    n_train = total - n_val
    train_dataset, val_dataset = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    results = {
        "train_loss": [],
        "val_loss": [],
        "train_mae": [],
        "val_mae": [],
        "train_rsq": [],
        "val_rsq": []
    }

    for epoch in tqdm(range(epochs), desc="Training epochs"):
        train_loss = train_step(model, train_loader, loss_fn, optimizer, device)
        val_loss = val_step(model, val_loader, loss_fn, device)
        # compute metrics
        train_mae, train_rsq = compute_metrics(train_loader, model, device)
        val_mae, val_rsq = compute_metrics(val_loader, model, device)

        print(
            f"Epoch {epoch+1}/{epochs}: "
            f"Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, "
            f"Train MAE = {train_mae:.4f}, Train R^2 = {train_rsq}, "
            f"Val MAE = {val_mae:.4f}, Val R^2 = {val_rsq}"
        )
        results["train_loss"].append(train_loss)
        results["val_loss"].append(val_loss)
        results["train_mae"].append(train_mae)
        results["train_rsq"].append(train_rsq)
        results["val_mae"].append(val_mae)
        results["val_rsq"].append(val_rsq)

    return results
