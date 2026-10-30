import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision


def calculate_precision_recall_f1(tp, fp, fn):
    """Calculate precision, recall and F1 from detection counts."""
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def calculate_map50(predictions, ground_truths):
    """Calculate mAP@0.5 for bird detection using TorchMetrics."""
    metric = MeanAveragePrecision(
        box_format="xyxy",
        iou_thresholds=[0.5],
        max_detection_thresholds=[1, 10, 300]
    )

    preds = []
    targets = []

    for prediction, boxes in zip(predictions, ground_truths):
        pred_boxes = torch.as_tensor(
            prediction["boxes"], dtype=torch.float32
        ).reshape(-1, 4)

        pred_scores = torch.as_tensor(
            prediction["scores"], dtype=torch.float32
        ).reshape(-1)

        gt_boxes = torch.as_tensor(
            boxes, dtype=torch.float32
        ).reshape(-1, 4)

        # All objects belong to class 0 (bird)
        preds.append({
            "boxes": pred_boxes,
            "scores": pred_scores,
            "labels": torch.zeros(len(pred_boxes), dtype=torch.int64)
        })

        targets.append({
            "boxes": gt_boxes,
            "labels": torch.zeros(len(gt_boxes), dtype=torch.int64)
        })

    # Calculate mAP@0.5
    metric.update(preds, targets)

    return metric.compute()["map_50"].item()