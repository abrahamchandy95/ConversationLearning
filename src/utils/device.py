import torch


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
