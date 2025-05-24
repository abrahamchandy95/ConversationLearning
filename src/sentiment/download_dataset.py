"""
Sentiment140 dataset download and extraction
"""
from pathlib import Path
from zipfile import ZipFile, BadZipFile
import requests
from tqdm import tqdm

from src.utils.utils import get_root

__all__ = [
    "_CITATION", "_DESCRIPTION", "_URL",
    "download_file", "download_and_extract"
]

# Metadata
_CITATION = """\
    @article{go2009twitter,
    title={Twitter sentiment classification using distant supervision},
    author={Go, Alec and Bhayani, Richa and Huang, Lei},
    journal={CS224N project report, Stanford},
    volume={1},
    number={12},
    pages={2009},
    year={2009}
    }
"""

_DESCRIPTION = (
    "Sentiment140 consists of Twitter messages with emoticons, "
    "which are used as noisy labels for sentiment classification. "
    "For more details, please refer to the paper."
)

_URL = "http://help.sentiment140.com/home"
_DATA_URL = "https://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"

# Create 'data' directory if it doesn't exist
root = get_root()
DATA_DIR = root / "data"
DATA_DIR.mkdir(exist_ok=True)

# Set up path for the dataset

ZIP_PATH = DATA_DIR / "sentiment_140.zip"
SENTIMENT140_DIR = DATA_DIR / "SENTIMENT140"
TRAIN_CSV = SENTIMENT140_DIR / "training.1600000.processed.noemoticon.csv"
TEST_CSV = SENTIMENT140_DIR / "testdata.manual.2009.06.14.csv"


def download_file(url: str, target_path: Path) -> None:
    """
    Downloads a file given a url to a specific path
    """
    response = requests.get(url, stream=True, timeout=10)
    total_size = int(response.headers.get('content-length', 0))

    with open(target_path, 'wb') as f, tqdm(
        desc=str(target_path),
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            f.write(data)
            pbar.update(len(data))


def download_and_extract():
    """
    Downloads and extracts the training data for sentiment analysis
    """
    if TRAIN_CSV.exists() and TEST_CSV.exists():
        print("SENTIMENT140 data already exist, skipping download")
        return
    SENTIMENT140_DIR.mkdir(exist_ok=True)

    try:
        # Download zip file
        print("Downloading dataset...")
        download_file(_DATA_URL, ZIP_PATH)

        # Extract zip file
        print("Extracting files...")
        with ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(SENTIMENT140_DIR)

        ZIP_PATH.unlink(missing_ok=True)
        print(f"Dataset extracted to {SENTIMENT140_DIR}")
    except (requests.RequestException, OSError, BadZipFile) as e:
        print(f"Error occured: {str(e)}")
        ZIP_PATH.unlink(missing_ok=True)


if __name__ == "__main__":
    download_and_extract()
