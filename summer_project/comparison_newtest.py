import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

RESULT_DIRS = {
    "Base FP16": Path("baseline_results_newtest/metrics.json"),
    "Base NF4": Path("baseline_nf4_results_newtest/metrics.json"),
    "Fine-tuned NF4 + LoRA": Path("finetuned_results_newtest/metrics.json"),
}

OUTPUT_DIR = Path("comparison_charts_newtest")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# METRICS TO COMPARE
# ============================================================

METRICS = {
    "mae": {
        "label": "MAE",
        "filename": "mae_comparison.png",
        "lower_is_better": True,
    },

    "rmse": {
        "label": "RMSE",
        "filename": "rmse_comparison.png",
        "lower_is_better": True,
    },

    "exact_total_score_accuracy": {
        "label": "Exact Total-Score Accuracy",
        "filename": "exact_accuracy_comparison.png",
        "lower_is_better": False,
    },

    "rubric_level_accuracy": {
        "label": "Rubric-Level Accuracy",
        "filename": "rubric_accuracy_comparison.png",
        "lower_is_better": False,
    },

    "pearson_r": {
        "label": "Pearson Correlation (r)",
        "filename": "pearson_comparison.png",
        "lower_is_better": False,
    },

    "spearman_rho": {
        "label": "Spearman Correlation (ρ)",
        "filename": "spearman_comparison.png",
        "lower_is_better": False,
    },
}


# ============================================================
# LOAD METRICS
# ============================================================

all_metrics = {}

print("=" * 70)
print("LOADING EVALUATION METRICS")
print("=" * 70)

for model_name, metrics_file in RESULT_DIRS.items():

    if not metrics_file.exists():

        raise FileNotFoundError(
            f"Metrics file not found:\n{metrics_file}"
        )

    with open(
        metrics_file,
        "r",
        encoding="utf-8"
    ) as f:

        all_metrics[model_name] = json.load(f)

    print(f"Loaded: {model_name}")
    print(f"       {metrics_file}")


# ============================================================
# PRINT RAW COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("METRIC COMPARISON")
print("=" * 70)

for metric_key, config in METRICS.items():

    print(f"\n{config['label']}:")

    for model_name in RESULT_DIRS:

        value = all_metrics[model_name].get(
            metric_key
        )

        if value is None:
            print(f"  {model_name}: N/A")
        else:
            print(f"  {model_name}: {value:.4f}")


# ============================================================
# GENERATE INDIVIDUAL CHARTS
# ============================================================

for metric_key, config in METRICS.items():

    model_names = []
    values = []

    for model_name in RESULT_DIRS:

        value = all_metrics[model_name].get(
            metric_key
        )

        if value is not None:

            model_names.append(model_name)
            values.append(float(value))


    if not values:
        continue


    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    plt.figure(figsize=(10, 6))

    bars = plt.bar(
        model_names,
        values
    )


    # --------------------------------------------------------
    # Add value labels
    # --------------------------------------------------------

    for bar, value in zip(bars, values):

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=11
        )


    # --------------------------------------------------------
    # Formatting
    # --------------------------------------------------------

    plt.title(
        f"{config['label']} Comparison",
        fontsize=16,
        fontweight="bold"
    )

    plt.ylabel(
        config["label"],
        fontsize=12
    )

    plt.xlabel(
        "Model",
        fontsize=12
    )

    plt.xticks(
        rotation=15,
        ha="right"
    )

    plt.grid(
        axis="y",
        linestyle="--",
        alpha=0.3
    )

    plt.tight_layout()


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_file = (
        OUTPUT_DIR
        / config["filename"]
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"\nSaved: {output_file}"
    )


# ============================================================
# COMBINED NORMALIZED COMPARISON
# ============================================================

# This chart is useful because MAE/RMSE and correlation/
# accuracy metrics have different scales.

normalization = {}

for metric_key in METRICS:

    values = []

    for model_name in RESULT_DIRS:

        value = all_metrics[model_name].get(
            metric_key
        )

        if value is not None:
            values.append(float(value))


    if not values:
        continue


    min_value = min(values)
    max_value = max(values)

    if max_value == min_value:

        normalized = {
            model_name: 1.0
            for model_name in RESULT_DIRS
        }

    else:

        normalized = {}

        for model_name in RESULT_DIRS:

            value = all_metrics[model_name].get(
                metric_key
            )

            if value is None:
                continue

            value = float(value)

            # For error metrics, lower is better.
            if METRICS[metric_key]["lower_is_better"]:

                score = (
                    max_value - value
                ) / (
                    max_value - min_value
                )

            else:

                score = (
                    value - min_value
                ) / (
                    max_value - min_value
                )

            normalized[model_name] = score


    normalization[metric_key] = normalized


# ============================================================
# PLOT NORMALIZED METRICS
# ============================================================

metric_labels = [
    METRICS[key]["label"]
    for key in normalization
]

model_names = list(RESULT_DIRS.keys())

x = np.arange(
    len(metric_labels)
)

width = 0.25


plt.figure(
    figsize=(14, 7)
)


for i, model_name in enumerate(model_names):

    values = []

    for metric_key in normalization:

        values.append(
            normalization[
                metric_key
            ].get(
                model_name,
                np.nan
            )
        )


    plt.bar(
        x + (i - 1) * width,
        values,
        width,
        label=model_name
    )


plt.title(
    "Normalized Performance Comparison",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel(
    "Evaluation Metric",
    fontsize=12
)

plt.ylabel(
    "Normalized Performance (Higher is Better)",
    fontsize=12
)

plt.xticks(
    x,
    metric_labels,
    rotation=25,
    ha="right"
)

plt.ylim(
    0,
    1.1
)

plt.legend()

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.3
)

plt.tight_layout()


combined_file = (
    OUTPUT_DIR
    / "normalized_metrics_comparison.png"
)

plt.savefig(
    combined_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    f"\nSaved: {combined_file}"
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("COMPARISON COMPLETE")
print("=" * 70)

print(f"\nAll charts are saved in:")
print(f"  {OUTPUT_DIR}/")

print("\nGenerated charts:")

for file in sorted(
    OUTPUT_DIR.glob("*.png")
):

    print(f"  - {file.name}")