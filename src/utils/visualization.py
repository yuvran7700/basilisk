# Common bounding box visualisation
import cv2
import matplotlib.pyplot as plt

def visualise_predictions(image, prediction, ground_truth):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    pred_img = image.copy()
    gt_img = image.copy()

    # Draw predicted bounding boxes and confidence scores
    for box, score in zip(prediction["boxes"], prediction["scores"]):
        x1, y1, x2, y2 = map(int, box)

        cv2.rectangle(pred_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            pred_img, f"{score:.2f}", (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1
        )

    # Draw ground-truth bounding boxes
    for box in ground_truth:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(gt_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Display predictions
    axes[0].imshow(cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"Predictions ({len(prediction['boxes'])} birds)")

    # Display ground truth
    axes[1].imshow(cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"Ground Truth ({len(ground_truth)} birds)")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()