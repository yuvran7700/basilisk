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



def compute_metrics(detections, ground_truths, iou_threshold=0.5):
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for det, ann in zip(detections, ground_truths):
        matched_detections = set()
        true_positive = 0

        for a in ann:
            best_iou = 0
            best_idx = -1

            for i, d in enumerate(det):
                if i in matched_detections:
                    continue

                iou = get_iou(d[:4], a)

                if iou > best_iou:
                    best_iou = iou
                    best_idx = i

            if best_iou >= iou_threshold:
                true_positive += 1
                matched_detections.add(best_idx)

        false_positive = len(det) - true_positive
        false_negative = len(ann) - true_positive

        total_tp += true_positive
        total_fp += false_positive
        total_fn += false_negative

    precision = (
        total_tp / (total_tp + total_fp)
        if total_tp + total_fp > 0 else 0
    )

    recall = (
        total_tp / (total_tp + total_fn)
        if total_tp + total_fn > 0 else 0
    )

    f1_score = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0 else 0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1_score
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
        max_detection_thresholds=[1, 10, 300],
        extended_summary=True 
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
    results = metric.compute()

    map_50 = results["map_50"].item()

    precisions_at_iou50 = results["precision"][0, :, 0, 0, 2]
    recalls_at_iou50 = results["recall"][0, 0, 0, 2]

    valid_precisions = precisions_at_iou50[precisions_at_iou50 >= 0]
    
    precision = valid_precisions.mean().item() if len(valid_precisions) > 0 else 0.0
    recall = recalls_at_iou50.max().item() # Max recall achieved under top limit
    
    # Calculate F1 from the aggregated values
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "map_50": map_50,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }