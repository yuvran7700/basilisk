import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

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
    area_union = float(det_area + gt_area - area_intersection + 1E-6)
    iou = area_intersection / area_union
    return iou


# def compute_metrics(detections, ground_truths, iou_threshold=0.5):
#     """
#     Calculate precision, recall and F1-score for object detection.

#     Args:
#         detections: List of prediction dictionaries containing boxes and scores.
#         ground_truths: List of ground-truth bounding boxes for each image.
#         iou_threshold: Minimum IoU required for a correct detection.

#     Returns:
#         Dictionary containing precision, recall and F1-score.
#     """

#     # Initialise total counts across all images
#     total_tp = 0
#     total_fp = 0
#     total_fn = 0

#     # Evaluate predictions for each image
#     for prediction, annotations in zip(detections, ground_truths):

#         # Extract predicted bounding boxes and confidence scores
#         pred_boxes = prediction["boxes"]
#         pred_scores = prediction["scores"]

#         # Sort predictions by confidence (highest first)
#         sorted_indices = np.argsort(pred_scores)[::-1]
#         pred_boxes = pred_boxes[sorted_indices]

#         # Track ground-truth boxes that have already been matched
#         matched_gt = set()
#         true_positive = 0

#         # Check each predicted bounding box
#         for pred_box in pred_boxes:

#             best_iou = 0
#             best_idx = -1

#             # Find the ground-truth box with the highest IoU
#             for i, gt_box in enumerate(annotations):

#                 iou = get_iou(pred_box[:4], gt_box)

#                 if iou > best_iou:
#                     best_iou = iou
#                     best_idx = i

#             # A correct detection must meet the IoU threshold
#             # and match a ground-truth box only once
#             if best_iou >= iou_threshold and best_idx not in matched_gt:
#                 true_positive += 1
#                 matched_gt.add(best_idx)

#         # Calculate false positives and false negatives
#         false_positive = len(pred_boxes) - true_positive
#         false_negative = len(annotations) - true_positive

#         # Accumulate results across all images
#         total_tp += true_positive
#         total_fp += false_positive
#         total_fn += false_negative

#     # Precision: proportion of predictions that were correct
#     precision = (
#         total_tp / (total_tp + total_fp)
#         if total_tp + total_fp > 0 else 0
#     )

#     # Recall: proportion of actual objects successfully detected
#     recall = (
#         total_tp / (total_tp + total_fn)
#         if total_tp + total_fn > 0 else 0
#     )

#     # F1-score: balance between precision and recall
#     f1_score = (
#         2 * precision * recall / (precision + recall)
#         if precision + recall > 0 else 0
#     )

#     return {
#         "precision": precision,
#         "recall": recall,
#         "f1": f1_score
#     }

def compute_metrics(detections, ground_truths, iou_threshold=0.5):
    """Calculate precision, recall and F1-score for object detection."""

    total_tp = 0
    total_fp = 0
    total_fn = 0

    # Compare predictions and annotations for each image
    for prediction, annotations in zip(detections, ground_truths):
        pred_boxes = prediction["boxes"]
        matched = set()

        # Check each predicted bounding box
        for pred_box in pred_boxes:
            best_iou = 0
            best_idx = -1

            # Find the ground-truth box with highest IoU
            for i, gt_box in enumerate(annotations):
                iou = get_iou(pred_box, gt_box)

                if iou > best_iou:
                    best_iou = iou
                    best_idx = i

            # Count correct detections
            if best_iou >= iou_threshold and best_idx not in matched:
                total_tp += 1
                matched.add(best_idx)
            else:
                total_fp += 1

        # Count ground-truth objects that were missed
        total_fn += len(annotations) - len(matched)

    # Calculate metrics
    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

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