import torch
from torch import nn
import torch.nn.functional as F
from transformers.models.longformer import LongformerModel


class AttentionPooling(nn.Module):
    """
    Learns context-aware importance weights for conversation tokens
    designed for smaller datasets. This uses hierarchial feature learning
    where the features are first projected onto a bottleneck dimension,
    and then to learn relative importance for the final features.
    """

    def __init__(self, hidden_size, bottleneck=64):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_size, bottleneck),
            nn.Tanh(),
            nn.Linear(bottleneck, 1),  # importance scoring
            nn.Dropout(0.1)
        )

    def forward(self, token_embeds, mask):
        """
        Computes weighted average of tokens using learned importance scores
        Args:
            token_embeds: (batch_size, seq_len, hidden_size)
            mask: (batch_size, seq_len) attention mask
        Returns:
            pooled: (batch_size, hidden_size) context-aware representations
        """
        # get attention logits to shape [batch_size, seq_len]
        attn_logits = self.attn(token_embeds).squeeze(-1)
        # apply mask and softmax
        attn_logits = attn_logits.masked_fill(~mask.bool(), -1e9)
        attn_weights = F.softmax(attn_logits, dim=1)

        # compute context-aware pooling
        return torch.sum(token_embeds * attn_weights.unsqueeze(-1), dim=1)


class ConversationScorerModel(nn.Module):
    """
    Module that learns how to score conversations
    """

    def __init__(
        self, num_targets, transformer_name='allenai/longformer-base-4096'
    ):
        super().__init__()
        self.encoder = LongformerModel.from_pretrained(transformer_name)
        for layer in self.encoder.encoder.layer[-2:]:
            for param in layer.parameters():
                param.requires_grad = True
        hidden_size = self.encoder.config.hidden_size

        self.attn_pooler = AttentionPooling(hidden_size)
        self.mean_pool = self._mean_pool

        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.LayerNorm(hidden_size*2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        # multi-head regression: outputs one value per target
        self.reg_head = nn.Linear(hidden_size, num_targets)

    def _mean_pool(self, token_embeds, mask):
        """
        Applies mean pooling on the token embeddings considering
        attention masks
        Parameters:

            token_embeds: The output from the transformer encoder
            that has the shape of (batch_size, seq_length, hidden_dim)
            mask: attention mask with shape (batch_size, seq_length)
            that filters out paddings
        Returns:
            Pooled output, a Tensor of shape (batch_size, hidden_dim)
            containing the pooled representations.
        """
        mask_expanded = mask.unsqueeze(-1).expand(token_embeds.size()).float()
        total_embeds = torch.sum(token_embeds * mask_expanded, dim=1)
        # sum the values along the sequence length
        mask_size = torch.clamp(torch.sum(mask_expanded, dim=1), min=1e-9)
        pooled = total_embeds / mask_size

        return pooled

    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the model
        """
        enc = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = enc.last_hidden_state
        attn_features = self.attn_pooler(last_hidden_state, attention_mask)
        mean_features = self.mean_pool(last_hidden_state, attention_mask)

        pooled = torch.cat([attn_features, mean_features], dim=1)

        # apply fully connected layer with ReLU
        fc_out = self.fc(pooled)
        predictions = self.reg_head(fc_out)

        return predictions
