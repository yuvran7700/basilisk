
import time

from src.methods.base import BaseDetector


class Evaluator:
    """Evaluate bird detection methods using annotated images."""

    def __init__(self, detector: BaseDetector):
        self.detector = detector

    def evaluate(self, dataset):
        """Run the detector on every image in the dataset.

        Compares can be performed using the collected predictions and
        ground-truth annotations. Measures the total inference time
        without including dataset loading.

        Args:
            dataset (list): List of (image, boxes) tuples, where:
                - image: OpenCV BGR image as a NumPy array.
                - boxes: Ground-truth bounding boxes of shape (N, 4)
                  in XYXY pixel coordinates.

        Returns:
            dict: Dictionary containing:
                - predictions: List of prediction dictionaries, each
                  containing "boxes" and "scores".
                - ground_truths: List of ground-truth bounding box arrays.
                - inference_time: Total inference time in seconds.
                - average_inference_time: Average seconds per image.
        """
        predictions = []
        ground_truths = []
        inference_time = 0.0

        # Run inference on each image
        for image, boxes in dataset:

            # Measure prediction time
            start = time.perf_counter()
            prediction = self.detector.predict(image)
            inference_time += time.perf_counter() - start

            # Store predictions and corresponding ground truth
            predictions.append(prediction)
            ground_truths.append(boxes)

        # Calculate average inference time
        average_time = inference_time / len(predictions) if len(predictions) else 0.0

        return {
            "predictions": predictions,
            "ground_truths": ground_truths,
            "inference_time": inference_time,
            "average_inference_time": average_time
        }
