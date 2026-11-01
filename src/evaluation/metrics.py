import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import numpy as np

# Bounding box evaluation metrics (IoU, NMS, mAP) adapted from 
# ExplainingAI's video tutorial: "Mean Average Precision (mAP) | Explanation and Implementation for Object Detection"
# Code source: https://github.com/explainingai-code/DetectionUtils/blob/main/compute_ap.py
def get_iou(det, gt):
    """method that calculcates the IoU between the detected box and the ground truth boxes"""
    dt_x1, dt_y1, dt_x2, dt_y2 = det
    gt_x1, gt_y1, gt_x2, gt_y2 = gt

    x_left = max(dt_x1, gt_x1)
    y_top = max(dt_y1, gt_y1)
    x_right = min(dt_x2, gt_x2)
    y_bottom = min(dt_y2, gt_y2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    area_intersection = (x_right - x_left) * (y_bottom - y_top)
    det_area = (dt_x2 - dt_x1) * (dt_y2 - dt_y1)
    gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
    area_union = float(det_area + gt_area - area_intersection + 1E-6) # to avoid zero division
    iou = area_intersection / area_union
    return iou


# This implementation is based on the evaluation metrics and toolkit from:
# Padilla, R., Passos, W.L., Dias, T.L.B., Netto, S.L., & da Silva, E.A.B. (2021).
# "A Comparative Analysis of Object Detection Metrics with a Companion Open-Source Toolkit."
# Electronics, 10(3), 279. https://doi.org/10.3390/electronics10030279
# Repository: https://github.com/rafaelpadilla/review_object_detection_metrics
def compute_metrics(detections, ground_truths, iou_threshold=0.5):
    """Calculate precision, recall and F1-score for object detection."""

    if len(detections) != len(ground_truths):
        raise ValueError(
            "Number of predictions and ground truths must match"
        )

    total_tp = 0
    total_fp = 0
    total_fn = 0

    # Compare predictions and ground_truths for each image
    for prediction, ground_truths in zip(detections, ground_truths):
        pred_boxes = prediction["boxes"]
        scores = prediction["scores"]

        sorted_indices = np.argsort(scores)[::-1]
        pred_boxes = pred_boxes[sorted_indices]
        matched = set()

        # Check each predicted bounding box
        for pred_box in pred_boxes:
            best_iou = 0
            best_idx = -1

            # Find the ground-truth box with highest IoU
            for i, gt_box in enumerate(ground_truths):
                if i in matched:
                    continue

                iou = get_iou(pred_box, gt_box)

                if iou > best_iou:
                    best_iou = iou
                    best_idx = i

            # Assign detections as TP or FP
            if best_iou >= iou_threshold: #gt was not matched to any detection
                total_tp += 1 # detection is set as true positive
                matched.add(best_idx)
            else:
                total_fp += 1 # detection is set as false positive

        # Count ground-truth objects that were missed
        total_fn += len(ground_truths) - len(matched)

    # Calculate metrics
    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

    print(total_tp, total_fp, total_fn)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# Rework this code, this is just GPT garbage rn, implement by yourself. Figure out if COCO vs Pascal VOC. AuC or interporalted percision metrics for the dataset. 
#Read through the dataset
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