
import time
import matplotlib.pyplot as plt
from src.methods.base import BaseDetector
from src.evaluation.metrics import calculate_map50, compute_metrics

class Evaluator:
    """Evaluate bird detection methods using annotated images."""

    def __init__(self, detector: BaseDetector):
        self.detector = detector

    def evaluate(self, dataset, title):
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
        
        # Calculate evalulation metrics
        metrics = compute_metrics(predictions, ground_truths)
        map50 = calculate_map50(predictions, ground_truths)

        results = {
            "predictions": predictions,
            "ground_truths": ground_truths,
            "inference_time": inference_time,
            "average_inference_time": average_time,
            "map50": map50,
            **metrics
        }

        self.print_metrics(results, title)

        return results

    
    def print_metrics(self, results, title):
        """Display evaluation metrics in a report-ready table."""

        data = [
            ["True Positives", f"{results['tp']:,}"],
            ["False Positives", f"{results['fp']:,}"],
            ["False Negatives", f"{results['fn']:,}"],
            ["Precision", f"{results['precision']:.4f}"],
            ["Recall", f"{results['recall']:.4f}"],
            ["F1-score", f"{results['f1']:.4f}"],
            ["mAP@0.5", f"{results['map50']:.4f}"],
            ["Total inference time (s)", f"{results['inference_time']:.4f}"],
            ["Average inference time (s)", f"{results['average_inference_time']:.4f}"]
        ]

        fig, ax = plt.subplots(figsize=(6.5, 3.5))
        ax.axis("off")

        table = ax.table(
            cellText=data,
            colLabels=["Metric", "Value"],
            cellLoc="left",
            colLoc="left",
            colWidths=[0.7, 0.3],
            loc="center"
        )

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.4)

        # Remove all borders and use a white background
        for (row, col), cell in table.get_celld().items():
            cell.set_facecolor("white")
            cell.set_edgecolor("#333333")
            cell.set_linewidth(0)
            cell.PAD = 0.05

            # Right-align numerical values
            if col == 1:
                cell.get_text().set_ha("right")

            # Bold column headers
            if row == 0:
                cell.get_text().set_weight("bold")
                cell.visible_edges = "TB"
                cell.set_linewidth(0.8)

            # Bottom border
            if row == len(data):
                cell.visible_edges = "B"
                cell.set_linewidth(0.8)

        ax.set_title(title, fontsize=12, fontweight="bold", pad=15)

        plt.tight_layout()
        plt.show()
