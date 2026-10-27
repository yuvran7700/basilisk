
# setup.py
import kagglehub

from config import DATA_DIR, IMAGE_DIR, LABEL_DIR


def download_dataset():
    """Download the Beach Bird Detection Dataset from Kaggle.

    Returns:
        Path: Directory containing the downloaded dataset.
    """
    # Skip download if the dataset already exists
    if IMAGE_DIR.is_dir() and LABEL_DIR.is_dir():
        print(f"Dataset already exists at: {DATA_DIR}")
        return DATA_DIR

    # Create the data directory if it doesn't exist
    DATA_DIR.parent.mkdir(parents=True, exist_ok=True)

    # Download and extract the dataset
    path = kagglehub.dataset_download(
        "liugalhigh/beach-bird",
        output_dir=str(DATA_DIR.parent)
    )

    # Check that the dataset was extracted correctly
    if not IMAGE_DIR.is_dir() or not LABEL_DIR.is_dir():
        raise FileNotFoundError(
            f"Expected dataset folders not found in: {DATA_DIR}"
        )

    print(f"Dataset downloaded to: {path}")
    return DATA_DIR


if __name__ == "__main__":
    download_dataset()
