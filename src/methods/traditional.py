
# src/methods/traditional.py

from src.methods.base import BaseDetector


class TraditionalDetector(BaseDetector):
    """Bird detection using traditional computer vision techniques."""

    def __init__(self):
        super().__init__(name="traditional")

        # TODO: Initialise method parameters

    def _train(self, dataset):
        """Optional: Implement training or parameter fitting if required."""
        # TODO: Implement if required
        pass

    def predict(self, image):
        """Detect birds in a single image."""
        # TODO: Implement bird detection
        # Return {"boxes": (N, 4), "scores": (N,)}
        raise NotImplementedError

def main():
    # Load training dataset
    train_dataset = load_dataset("train")

    # Initialise and train detector
    detector = TraditionalDetector()
    detector.train(train_dataset)

if __name__ == "__main__":
    main()