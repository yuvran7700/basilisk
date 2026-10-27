from pathlib import Path

# Define project and dataset paths
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "Beach Bird Detection Dataset v1.0"

# Define image and annotation directories
IMAGE_DIR = DATA_DIR / "images"
LABEL_DIR = DATA_DIR / "labels"

# Define results directory
RESULTS_DIR = PROJECT_DIR / "results"


# Define image dimensions
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512