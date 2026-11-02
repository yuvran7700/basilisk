#!/bin/bash

set -e

# Navigate to project root
cd "$(dirname "$0")"

echo "Setting up Computer Vision project..."

# 1. Find Python
if command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
elif command -v py &> /dev/null; then
    PYTHON=py
else
    echo "Error: Python is not installed."
    exit 1
fi

# 2. Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    "$PYTHON" -m venv .venv
fi

# 3. Find virtual environment Python
if [ -f ".venv/Scripts/python.exe" ]; then
    VENV_PYTHON=".venv/Scripts/python.exe"
else
    VENV_PYTHON=".venv/bin/python"
fi

# 4. Install dependencies
echo "Installing dependencies..."

"$VENV_PYTHON" -m pip install --upgrade pip
"$VENV_PYTHON" -m pip install -r requirements.txt

# 5. Register Jupyter kernel
echo "Registering Jupyter kernel..."

"$VENV_PYTHON" -m ipykernel install --user \
    --name basilisk-venv \
    --display-name "Python (basilisk .venv)"

# 6. Download dataset
echo "Checking Beach Bird Detection Dataset..."

"$VENV_PYTHON" <<'PYTHON'
import kagglehub
from config import DATA_DIR, IMAGE_DIR, LABEL_DIR

DATA_DIR.parent.mkdir(parents=True, exist_ok=True)

if IMAGE_DIR.is_dir() and LABEL_DIR.is_dir():
    print(f"Dataset already exists: {DATA_DIR}")
    print("Skipping download.")
else:
    path = kagglehub.dataset_download(
        "liugalhigh/beach-bird",
        output_dir=str(DATA_DIR)
    )

    print(f"Dataset downloaded to: {path}")
PYTHON

echo "Setup completed successfully!"
