import matplotlib.pyplot as plt


def print_metrics(results, title):
    """Display a comparison table of evaluation metrics.

    Args:
        results (dict): Maps column names to evaluation results.
        title (str): Title displayed above the table.
    """

    metrics = [
        ("tp", "True Positives"),
        ("fp", "False Positives"),
        ("fn", "False Negatives"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1-score"),
        ("map50", "mAP@0.5"),
        ("inference_time", "Total inference time (s)"),
        ("average_inference_time", "Average inference time (s)")
    ]

    # Column names (models or severity levels)
    columns = list(results.keys())

    # Build table rows
    data = []

    for metric, label in metrics:
        row = [label]

        for column in columns:
            value = results[column][metric]

            if metric in ("tp", "fp", "fn"):
                row.append(f"{value:,}")
            else:
                row.append(f"{value:.4f}")

        data.append(row)

    # Create table
    fig, ax = plt.subplots(
        figsize=(max(6.5, 2 * (len(columns) + 1)), 3.5)
    )
    ax.axis("off")

    table = ax.table(
        cellText=data,
        colLabels=["Metric", *columns],
        cellLoc="left",
        colLoc="left",
        loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)

    # Format table
    for (row, col), cell in table.get_celld().items():
        cell.set_facecolor("white")
        cell.set_edgecolor("#333333")
        cell.set_linewidth(0)
        cell.PAD = 0.05

        if col > 0:
            cell.get_text().set_ha("right")

        if row == 0:
            cell.get_text().set_weight("bold")
            cell.visible_edges = "TB"
            cell.set_linewidth(0.8)

        if row == len(data):
            cell.visible_edges = "B"
            cell.set_linewidth(0.8)

    ax.set_title(title, fontsize=12, fontweight="bold", pad=15)

    plt.tight_layout()
    plt.show()


def display_results(results, seed=None):
    for name, evaluations in results.items():
        for severity, detector_results in enumerate(evaluations, start=1):

            if name == "CleanExperiment":
                title = "Clean Evaluation"
            elif len(evaluations) > 1:
                title = f"{name} - Severity {severity}"
            else:
                title = f"{name} - Random Severity"

            if seed is not None:
                title += f" (Seed: {seed})"

            print_metrics(detector_results, title)