import os
import time
from flask import Flask, request, jsonify
from src.data_setup import ConversationPreprocessor
from src.model_builder import ConversationScorerModel
from src.utils import load_conversation
import torch
from transformers import logging

logging.set_verbosity_error()
torch.set_num_threads(1)

app = Flask(__name__)

class ModelContainer:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.target_name = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def load_artifacts(self):
            start = time.time()

            scores_df = load_scores("scores.csv")
            self.target_names = scores_df.columns.drop("thread_id").tolist()

            # Model setup with safety
            self.model = BuddyRegressionModel(num_targets=len(self.target_names))
            model_path = os.path.join('models', 'conversation_scorer.pth')
            self.model.load_state_dict(torch.load(model_path, map_local=self.device))
            self.model.eval()
            self.model.to(self.device)

            self.preprocessor = ConversationPreprocessor(max_length=3000)
            print(f"Loaded artifacts in ")
