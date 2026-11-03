# src/evaluation/visualization.py
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

METRICS = [
    ("tp", "TP"),
    ("fp", "FP"),
    ("fn", "FN"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1", "F1"),
    ("map50", "mAP@0.5"),
    ("inference_time", "Total (s)"),
    ("average_inference_time", "Avg (s)")
]

BEST_METRICS = {"precision", "recall", "f1", "map50"}
OUTPUT_DIR = Path("results/tables")

def format_metric(value, metric):
    """Format metric values consistently."""
    if metric in ("tp", "fp", "fn"):
        return f"{value:,}"
    return f"{value:.3f}"


def render_table(data, headers, title, best_rows=None, seed=None):
    """Render and export an IEEE-style table."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    with plt.rc_context({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "pdf.fonttype": 42
    }):
        fig, ax = plt.subplots(
            figsize=(7.1, max(1.4, 0.6 + 0.35 * len(data)))
        )
        ax.axis("off")

        table = ax.table(
            cellText=data,
            colLabels=headers,
            colWidths=[0.16] + [0.093] * (len(headers) - 1),
            cellLoc="center",
            colLoc="center",
            loc="center"
        )

        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)

        # IEEE-style horizontal rules
        for (row, col), cell in table.get_celld().items():
            cell.set_facecolor("white")
            cell.set_edgecolor("black")
            cell.set_linewidth(0)
            cell.visible_edges = ""
            cell.PAD = 0.04

            if row == 0:
                cell.get_text().set_weight("bold")
                cell.visible_edges = "TB"
                cell.set_linewidth(0.8)

            elif row == len(data):
                cell.visible_edges = "B"
                cell.set_linewidth(0.8)

            if col == 0:
                cell.get_text().set_ha("left")

        # Highlight best-performing values
        for row, col in best_rows or []:
            cell = table[(row, col)]
            cell.get_text().set_weight("bold")
            cell.set_facecolor("#ECF4C9")

        ax.set_title(title, fontsize=9, fontweight="bold", pad=10)
        
        if seed is not None:
            fig.text(
                0.95, 0.02,
                f"Seed: {seed}",
                ha="right",
                fontsize=7
            )

        plt.tight_layout(rect=[0, 0.08, 1, 1])

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        filename = "".join(
            c if c.isalnum() or c in "-_" else "_"
            for c in title
        )

        if seed is not None:
            filename += f"_seed{seed}"

        filename += f"_{timestamp}"

        fig.savefig(
            OUTPUT_DIR / f"{filename}.pdf",
            bbox_inches="tight",
            pad_inches=0.05
        )

        plt.show()
        plt.close(fig)


def print_metrics(results, title, seed=None):
    """Compare models in a single IEEE-style table."""

    models = list(results)

    data = [
        [model] + [
            format_metric(results[model][metric], metric)
            for metric, _ in METRICS
        ]
        for model in models
    ]

    best_rows = []

    for col, (metric, _) in enumerate(METRICS, start=1):
        if metric in BEST_METRICS:
            best = max(results[model][metric] for model in models)

            for row, model in enumerate(models, start=1):
                if results[model][metric] == best:
                    best_rows.append((row, col))

    headers = ["Model"] + [label for _, label in METRICS]

    render_table(data, headers, title, best_rows, seed)


def print_sweep_metrics(evaluations, title, seed=None):
    """Compare severity levels in a separate table for each model."""

    models = list(evaluations[0])

    for model in models:
        data = [
            [severity] + [
                format_metric(results[model][metric], metric)
                for metric, _ in METRICS
            ]
            for severity, results in enumerate(evaluations, start=1)
        ]

        best_rows = []

        for col, (metric, _) in enumerate(METRICS, start=1):
            if metric in BEST_METRICS:
                best = max(r[model][metric] for r in evaluations)

                for row, results in enumerate(evaluations, start=1):
                    if results[model][metric] == best:
                        best_rows.append((row, col))

        headers = ["Severity"] + [label for _, label in METRICS]

        render_table(
            data,
            headers,
            f"{title} - {model}",
            best_rows,
            seed
        )


def display_results(results, seed=None):
    """Display normal evaluations and severity sweeps."""

    for name, evaluations in results.items():

        if name == "CleanExperiment":
            print_metrics(evaluations[0], "Clean Evaluation", seed)

        elif len(evaluations) > 1:
            print_sweep_metrics(evaluations, name, seed)

        else:
            print_metrics(
                evaluations[0],
                f"{name} - Random Severity",
                seed
            )
