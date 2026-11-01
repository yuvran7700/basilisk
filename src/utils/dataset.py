import cv2
import numpy as np
import kagglehub

from config import DATA_DIR, IMAGE_DIR, LABEL_DIR, IMAGE_WIDTH, IMAGE_HEIGHT


def download_dataset():
    """Download the Beach Bird Detection Dataset from Kaggle.

    Returns:
        str: Path to the downloaded dataset.
    """
    # Create the data directory if it doesn't exist
    DATA_DIR.parent.mkdir(parents=True, exist_ok=True)

    # Download and extract the dataset
    path = kagglehub.dataset_download(
        "liugalhigh/beach-bird",
        output_dir=str(DATA_DIR.parent)
    )

    print(f"Dataset downloaded to: {path}")
    return path


def load_dataset(split):
    """Load images and their corresponding bounding-box annotations.

    Args:
        split (str): Dataset split ('train', 'val', or 'test').

    Returns:
        list: List of (image, boxes) tuples, where:
            - image: OpenCV BGR image as a NumPy array.
            - boxes: NumPy array of shape (N, 4) in XYXY pixel format.
    """
    # Validate dataset split
    if split not in ["train", "val", "test"]:
        raise ValueError("Split must be 'train', 'val', or 'test'")

    images_path = IMAGE_DIR / split
    labels_path = LABEL_DIR / split

    dataset = []

    # Loop over images in alphabetical order
    for image_file in sorted(images_path.iterdir()):
        if not image_file.is_file():
            continue

        # Load image
        image = cv2.imread(str(image_file))

        if image is None:
            raise ValueError(f"Could not load image: {image_file}")

        # Get corresponding annotation file
        label_path = labels_path / f"{image_file.stem}.txt"

        # Load bounding boxes in XYXY format
        boxes = load_annotations(label_path)

        # Store image, annotations and file name together
        dataset.append((image, boxes, image_file.stem))

    return dataset


def yolo_to_xyxy(x, y, w, h):
    """Convert normalized YOLO coordinates to XYXY pixel coordinates.

    Args:
        x (float): Normalized bounding-box centre x-coordinate.
        y (float): Normalized bounding-box centre y-coordinate.
        w (float): Normalized bounding-box width.
        h (float): Normalized bounding-box height.

    Returns:
        list: Bounding box [x_min, y_min, x_max, y_max] in pixels.
    """
    # Convert normalized coordinates to pixel coordinates
    x_min = (x - w / 2) * IMAGE_WIDTH
    y_min = (y - h / 2) * IMAGE_HEIGHT
    x_max = (x + w / 2) * IMAGE_WIDTH
    y_max = (y + h / 2) * IMAGE_HEIGHT

    return [x_min, y_min, x_max, y_max]


def load_annotations(label_path):
    """Load YOLO annotations and convert them to XYXY pixel coordinates.

    Args:
        label_path (str or Path): Path to the annotation file.

    Returns:
        np.ndarray: Bounding boxes of shape (N, 4) in XYXY pixel format.
            Returns an empty array of shape (0, 4) if no birds are annotated.
    """
    boxes = []

    # Read annotation file
    with open(label_path, "r") as file:
        for line in file:
            class_id, x, y, w, h = map(float, line.split())

            # Convert YOLO bounding box to XYXY format
            box = yolo_to_xyxy(x, y, w, h)

            boxes.append(box)

    return np.array(boxes, dtype=np.float32).reshape(-1, 4)


# Only download when this file is executed directly
if __name__ == "__main__":
    download_dataset()
