
import time
from src.evaluation.metrics import calculate_map50, compute_metrics

class Evaluator:
    """Evaluate one or more detectors against ground-truth annotations."""

    def __init__(self, detectors):
        self.detectors = detectors

    def evaluate(self, dataset):
        """Evaluate all detectors on the same dataset."""
        results = {}

        for name, detector in self.detectors.items():
            results[name] = self._evaluate_detector(detector, dataset)

        return results

    @staticmethod
    def _evaluate_detector(detector, dataset):
        """Run inference and calculate metrics for a single detector."""
        predictions = []
        ground_truths = []
        inference_time = 0.0

        # Run inference on each image
        for image, boxes in dataset:

            # Measure prediction time
            start = time.perf_counter()
            prediction = detector.predict(image)
            inference_time += time.perf_counter() - start

            # Store predictions and corresponding ground truth
            predictions.append(prediction)
            ground_truths.append(boxes)

        # Calculate average inference time
        average_time = (
            inference_time / len(predictions)
            if predictions else 0.0
        )

        # Calculate evaluation metrics
        metrics = compute_metrics(predictions, ground_truths)
        map50 = calculate_map50(predictions, ground_truths)

        return {
            "predictions": predictions,
            "ground_truths": ground_truths,
            "inference_time": inference_time,
            "average_inference_time": average_time,
            "map50": map50,
            **metrics
        }
