"""
Builds the neural network model for sentiment analysis
"""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from transformers.tokenization_utils import PreTrainedTokenizer


@dataclass
class SentimentConfig:
    """
    Hyperparameters of the SentimentLSTM
    """
    num_classes: int
    embed_dim: int = 128
    hidden_dim: int = 256
    num_layers = 2
    dropout: float = 0.5


class SentimentLSTM(nn.Module):
    """
    Bidirectional LSTM for sentiment classification
    Args:
        vocab_size: Size of the vocabulary (number of unique tokens).
        embed_dim: Dimensionality of token embeddings.
        hidden_dim: Hidden state size in the LSTM.
        num_layers: Number of LSTM layers.
        num_classes: Number of output classes.
        dropout: Dropout probability between LSTM layers.
        pad_idx: Index for padding tokens (tokenizer.pad_token_id).
        device: Device to run the model on ('cpu' or 'cuda').
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        config: SentimentConfig,
        device: str = "cpu"
    ) -> None:
        super().__init__()
        vocab_size = tokenizer.vocab_size
        pad_idx = tokenizer.pad_token_id

        self.device = torch.device(device)
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=config.embed_dim,
            padding_idx=pad_idx  # type: ignore[arg-type]
        )
        self.lstm = nn.LSTM(
            input_size=config.embed_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.hidden_dim * 2, config.num_classes)
        self.to(self.device)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Implements the forward pass of the model
        """
        # input_ids: (B, T), attention_mask: (B, T)
        lengths = attention_mask.sum(dim=1)  # actual lengths per sample
        embedded = self.embedding(input_ids)  # (B, T, E)
        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        # h_n: (num_layers * 2, B, hidden_dim)
        # take last-layer forward & backward states
        h_fwd = h_n[-2, :, :]    # (B, hidden_dim)
        h_bwd = h_n[-1, :, :]    # (B, hidden_dim)
        h_cat = torch.cat((h_fwd, h_bwd), dim=1)  # (B, hidden_dim*2)
        out = self.dropout(h_cat)
        return self.fc(out)  # (B, num_classes)
