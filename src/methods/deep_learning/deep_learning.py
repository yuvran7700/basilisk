
# src/methods/deep_learning.py
from ultralytics import YOLO
from src.methods.base import BaseDetector


class DeepLearningDetector(BaseDetector):
    """Bird detection using deep learning techniques."""

    def __init__(self):
        super().__init__(name="deep_learning")

        self.model = YOLO("yolo11n.pt")


    def _train(self, dataset):
        """Implement deep learning training."""
        print("Using pretrained YOLO model. Training skipped.")

    def predict(self, image):
        """Detect birds in a single image."""
        result = self.model.predict(
            source=image,
            classes=[14],  # COCO bird class
            conf=0.25,
            verbose=False
        )[0]

        return {
            "boxes": result.boxes.xyxy.cpu().numpy(),
            "scores": result.boxes.conf.cpu().numpy()
        }