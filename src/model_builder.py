import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LongformerModel

class BuddyRegressionModel(nn.Module):

    def __init__(
        self, num_targets, transformer_name='allenai/longformer-base-4096'
    ):
        super().__init__()
        self.encoder = LongformerModel.from_pretrained(transformer_name)
        hidden_size = self.encoder.config.hidden_size
        self.fc = nn.Linear(hidden_size, hidden_size)
        # multi-head regression: outputs one value per target
        self.reg_head = nn.Linear(hidden_size, num_targets)

    def mean_pool(self, token_embeds, mask):
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
        enc = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = enc.last_hidden_state
        pooled = self.mean_pool(last_hidden_state, attention_mask)
        # apply fully connected layer with ReLU
        fc_out = F.relu(self.fc(pooled))
        predictions = self.reg_head(fc_out)

        return predictions
