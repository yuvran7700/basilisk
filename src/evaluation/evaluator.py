
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

    
    def print_metrics(self, results, title):
        """Display detection metrics in a formatted Matplotlib table.

        Presents detection counts, precision, recall, F1-score, mAP@0.5,
        and inference times in a report-ready table.

        Args:
            results (dict): Dictionary of evaluation metrics returned
                by the evaluate() method.
            title (str): Title displayed above the metrics table.

        Returns:
            None: Displays the table without returning a value.
        """

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
