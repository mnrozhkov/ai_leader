"""Table and plot helpers for notebook output."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol, cast

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .config import MODEL_PALETTE


class _HasQualityMetrics(Protocol):
    """Notebook `ModelRun` and similar objects expose quality_metrics."""

    quality_metrics: dict[str, Any]


_EVALUATION_METRIC_KEYS: tuple[str, ...] = (
    "category_accuracy",
    "category_f1_macro",
    "department_accuracy",
    "department_f1_macro",
    "exact_route_match_rate",
    "misroute_rate",
    "row_count",
)


def normalize_confusion_rows(matrix: list[list[int]]) -> list[list[float]]:
    """Return row fractions for a count matrix (each row sums to 1.0; empty rows → zeros)."""
    out: list[list[float]] = []
    for row in matrix:
        total = sum(row)
        if total == 0:
            out.append([0.0] * len(row))
        else:
            out.append([c / total for c in row])
    return out


def plot_confusion_matrix_relative_and_counts(
    matrix: list[list[int]],
    labels: Iterable[str],
    *,
    title: str,
    row_count: int,
) -> None:
    """Two panels: row fractions [0, 1] (left) and integer counts (right)."""
    label_list = list(labels)
    norm = normalize_confusion_rows(matrix)
    fig_title = f"{title} (N rows: {row_count})"
    fig, (ax_rel, ax_cnt) = plt.subplots(
        1,
        2,
        figsize=(13.0, 5.5),
        sharey=True,
    )
    with plt.rc_context({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 11}):
        sns.heatmap(
            norm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            linewidths=0.6,
            linecolor="white",
            xticklabels=label_list,
            yticklabels=label_list,
            ax=ax_rel,
            vmin=0.0,
            vmax=1.0,
            annot_kws={"size": 9, "weight": "medium"},
            cbar_kws={"label": "Row fraction"},
        )
        ax_rel.set_xlabel("Predicted")
        ax_rel.set_ylabel("Actual")
        ax_rel.set_title("Relative (per actual row)", fontweight="semibold", pad=8)

        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            linewidths=0.6,
            linecolor="white",
            xticklabels=label_list,
            yticklabels=label_list,
            ax=ax_cnt,
            annot_kws={"size": 9, "weight": "medium"},
            cbar_kws={"label": "Count"},
        )
        ax_cnt.set_xlabel("Predicted")
        ax_cnt.set_ylabel("")
        ax_cnt.set_title("Counts", fontweight="semibold", pad=8)

    fig.suptitle(fig_title, fontweight="semibold", y=1.02)
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def display_evaluation_results(*, eval_results: _HasQualityMetrics) -> None:
    """Print core routing metrics and show confusion matrices (relative + counts)."""
    qm = eval_results.quality_metrics
    row = {k: qm.get(k) for k in _EVALUATION_METRIC_KEYS}
    metrics_series = pd.Series({k: row[k] for k in _EVALUATION_METRIC_KEYS})
    print(metrics_series.to_string())
    n = int(row["row_count"] or 0)

    cat = qm.get("confusion_category")
    if isinstance(cat, dict) and "matrix" in cat and "labels" in cat:
        plot_confusion_matrix_relative_and_counts(
            cast(list[list[int]], cat["matrix"]),
            cast(list[str], cat["labels"]),
            title="Category confusion",
            row_count=n,
        )

    dept = qm.get("confusion_department")
    if isinstance(dept, dict) and "matrix" in dept and "labels" in dept:
        plot_confusion_matrix_relative_and_counts(
            cast(list[list[int]], dept["matrix"]),
            cast(list[str], dept["labels"]),
            title="Department confusion",
            row_count=n,
        )


_CONFIDENCE_COLORS = {
    "High": "#55A868",
    "Medium": "#DD8452",
    "Low": "#C44E52",
    "Unknown": "#8C8C8C",
}


def _short_name(model: str) -> str:
    return model.rsplit("/", 1)[-1]


def plot_confusion_matrix(
    matrix: list[list[int]],
    labels: Iterable[str],
    *,
    title: str,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    with plt.rc_context({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 11}):
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            linewidths=0.6,
            linecolor="white",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            annot_kws={"size": 10, "weight": "medium"},
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(title, fontweight="semibold", pad=10)
    fig.tight_layout()
    return fig


def plot_quality_vs_cost(results: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    df = results.sort_values("cost_per_message_usd").copy()

    with plt.rc_context(
        {"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 11, "legend.fontsize": 9}
    ):
        for _, row in df.iterrows():
            m = str(row["model"])
            cost = float(row["cost_per_message_usd"])
            em = float(row["exact_match_rate"])
            color = MODEL_PALETTE.get(m, "#4C72B0")
            ax.scatter(
                cost,
                em,
                s=200,
                color=color,
                edgecolors="white",
                linewidths=1.2,
                zorder=3,
            )
            ax.annotate(
                _short_name(m),
                (cost, em),
                textcoords="offset points",
                xytext=(12, -6),
                fontsize=9,
                fontweight="bold",
                color=color,
            )

        ax.set_xlabel("Cost per message (USD)")
        ax.set_ylabel("Exact route match rate")
        ax.set_title("Quality vs Cost", fontweight="semibold", pad=10)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:.4f}"))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{round(x * 100)}%"))
        ax.grid(True, alpha=0.25, linestyle=":", zorder=0)
        ax.set_axisbelow(True)

    fig.tight_layout()
    return fig


def plot_confidence_buckets(bucket_df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    df = bucket_df.copy()
    df["confidence"] = df["confidence"].astype(str)
    colors = [_CONFIDENCE_COLORS.get(str(c), "#8C8C8C") for c in df["confidence"]]
    xs = range(len(df))
    counts = [int(round(float(v))) for v in df["count"]]

    with plt.rc_context({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 11}):
        bars = ax.bar(
            list(xs),
            counts,
            color=colors,
            edgecolor="white",
            linewidth=1.0,
        )
        ax.bar_label(
            bars,
            labels=[str(n) for n in counts],
            padding=3,
            fontsize=9,
            fontweight="medium",
        )
        ax.set_xticks(list(xs))
        ax.set_xticklabels(df["confidence"].tolist())
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Count")
        ax.set_title("Confidence buckets", fontweight="semibold", pad=10)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(round(x))}"))
        ax.grid(True, axis="y", alpha=0.25, linestyle=":", zorder=0)
        ax.set_axisbelow(True)

    fig.tight_layout()
    return fig
