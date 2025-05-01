"""
Sentiment subpackage: download, preprocessing, training, and inference
for Sentiment140.
"""
import sys
from pathlib import Path

from src.sentiment.download_dataset import download_file, download_and_extract
from src.sentiment.data_setup import (
    TextTokenizer, DatasetBuilder, DataLoaderBuilder
)
from src.sentiment.model_builder import SentimentLSTM, SentimentConfig
# Ensure project root is on sys.path so absolute imports resolve
# __file__ → src/sentiment/__init__.py, so parent.parent.parent → project root
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Expose key modules and classes

__all__ = [
    # download
    "download_file", "download_and_extract",
    # dataset
    "TextTokenizer", "DatasetBuilder", "DataLoaderBuilder",
    # modeling
    "SentimentLSTM", "SentimentConfig",
]

