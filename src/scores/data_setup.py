"""
Sets up data to be prepared for the model
"""
from typing import List, Optional, Union, Tuple
import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers.models.longformer import LongformerTokenizer


class ConversationPreprocessor:
    """
    Tokenizes conversations and preprocesses it for machine learning
    """

    def __init__(
            self,
            tokenizer_model: str = 'allenai/longformer-base-4096',
            max_length: int = 3000
    ):
        self.tokenizer = LongformerTokenizer.from_pretrained(tokenizer_model)
        self.max_length = max_length

    def add_role_tokens(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds a role token to each part of a conversation. For example,
        a user would have the <USER> token and the bot will have the
        <BOT> token.
        """
        df = df.copy()
        df['message_modified'] = df.apply(self._add_role, axis=1
                                          )
        return df

    def _add_role(self, row: pd.Series) -> str:
        """
        Helper function to prepend the role to the conversation
        """
        if row['role'] == 'assistant':
            return '<BOT>' + row['content']
        return '<USER>' + row['content']

    def group_conversations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Groups messages by the thread id so that we can get
        get complete conversations between bot and user
        """
        conv_df = df.groupby('thread_id')['message_modified']\
                    .agg("\n".join) \
                    .reset_index() \
                    .rename(columns={'message_modified': 'text'})

        return conv_df

    def tokenize_text(self, text: str) -> dict:
        """
        Tokenizes a conversation text such that it returns a dictionary
        of keys 'input_ids' and values 'attention_masks'. Both are
        Pytorch Tensors
        """
        tokens = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )
        return tokens

    def preprocess_conversations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Completes preprocessing pipeline for the conversations
        1. Adds role tokens
        2. Groups messages by 'thread_id'
        3. Tokenizes each conversation
        Adds the tokens as a column to the input DataFrame
        """
        df_tokens = self.add_role_tokens(df)
        conv_df = self.group_conversations(df_tokens)
        conv_df['tokenized'] = conv_df['text'].apply(self.tokenize_text)

        return conv_df


class ConversationDataset(Dataset):
    """
    Turns the conversation data into datasets
    """

    def __init__(
        self, df_tokens: pd.DataFrame,
        target_columns: Optional[List[str]] = None
    ):
        self.df = df_tokens
        self.target_columns = target_columns

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(
        self,
        idx: Union[int, List[int]]
    ) -> Tuple[
        Union[torch.Tensor, List[torch.Tensor]],
        Union[torch.Tensor, List[torch.Tensor]],
        Optional[torch.Tensor]
    ]:
        if isinstance(idx, int):
            row = self.df.iloc[idx]
            tok = row['tokenized']
            input_ids = tok['input_ids'].squeeze(0)
            attention_mask = tok['attention_mask'].squeeze(0)

            targets: Optional[torch.Tensor] = None
            if self.target_columns:
                vals = row[self.target_columns].to_numpy(dtype='float32')
                targets = torch.tensor(vals, dtype=torch.float)

            return input_ids, attention_mask, targets

        input_ids_list = []
        attention_mask_list = []
        for i in idx:
            row = self.df.iloc[i]
            tok = row['tokenized']
            input_ids_list.append(tok['input_ids'].squeeze(0))
            attention_mask_list.append(tok['attention_mask'].squeeze(0))

        return input_ids_list, attention_mask_list, None
