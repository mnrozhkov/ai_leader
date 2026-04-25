"""Table and plot helpers for notebook output."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol, cast

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .config import MODEL_PALETTE, MODEL_REGISTRY
from .decision import DecisionSummary

_DISPLAY_DECIMALS = 3


def _display_round(value: Any) -> Any:
    """Round numerics for tables and plot labels (see ``_DISPLAY_DECIMALS``)."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return round(float(value), _DISPLAY_DECIMALS)
    return value


def round_numeric_frame(df: pd.DataFrame, *, decimals: int = _DISPLAY_DECIMALS) -> pd.DataFrame:
    """Return a copy with numeric columns rounded (string columns unchanged)."""
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col].dtype):
            out[col] = pd.to_numeric(out[col], errors="coerce").round(decimals)
    return out


def save_figure(fig: plt.Figure, path: str | Path, *, dpi: int = 150) -> Path:
    """Save a matplotlib figure to disk and close it. Returns the resolved path."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return dest


class _HasQualityMetrics(Protocol):
    """Notebook `ModelRun` and similar objects expose quality_metrics."""

    quality_metrics: dict[str, Any]


class _HasPredictions(Protocol):
    """Notebook `ModelRun` and similar objects expose predictions."""

    predictions: pd.DataFrame


class _HasQualityAndPredictions(Protocol):
    """Notebook `ModelRun` shape used by diagnostics helper."""

    quality_metrics: dict[str, Any]
    predictions: pd.DataFrame


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
            fmt=".3f",
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
    n = int(qm.get("row_count") or 0)
    row = {k: _display_round(qm.get(k)) for k in _EVALUATION_METRIC_KEYS}
    metrics_series = pd.Series({k: row[k] for k in _EVALUATION_METRIC_KEYS})
    print(metrics_series.to_string())

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


def _first_existing_column(columns: list[str], candidates: list[str]) -> str | None:
    for col in candidates:
        if col in columns:
            return col
    return None


def department_mistakes_table(
    *,
    eval_df: pd.DataFrame,
    predictions: pd.DataFrame,
    max_rows: int = 10,
) -> pd.DataFrame:
    """Return top department routing mistakes for diagnostics."""
    target_department_col = "Routing to Department"
    mistakes = eval_df[["Request Text", target_department_col]].copy()
    pred_department_col = _first_existing_column(
        predictions.columns.tolist(),
        [
            "[Agent] Routing to Department",
            "Routed to Department",
            "Routing to Department",
            "routing_to_department",
        ],
    )
    if pred_department_col is None:
        print("No department prediction column found. Available columns:")
        print(sorted(predictions.columns.tolist()))
        out = mistakes.head(0).copy()
        out["predicted_department"] = pd.Series(dtype="string")
        return out

    mistakes["predicted_department"] = (
        predictions[pred_department_col].reindex(mistakes.index).astype(str)
    )
    out = mistakes[
        mistakes[target_department_col].astype(str) != mistakes["predicted_department"].astype(str)
    ].head(max_rows)
    return out.rename(
        columns={
            target_department_col: "Department",
            "predicted_department": "Predicted Department",
        }
    )


def display_evaluation_with_department_mistakes(
    *,
    eval_results: _HasQualityAndPredictions,
    eval_df: pd.DataFrame,
    max_rows: int = 10,
) -> pd.DataFrame:
    """Display evaluation charts and return a compact department mistake table."""
    display_evaluation_results(eval_results=eval_results)
    return department_mistakes_table(
        eval_df=eval_df,
        predictions=eval_results.predictions.copy(),
        max_rows=max_rows,
    )


def get_high_confident_errors(
    *,
    eval_df: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Return high-confidence department routing errors.

    Filters predictions where ``Confidence == "High"`` and keeps rows where
    predicted department differs from reference department.
    """
    joined = eval_df.merge(predictions, on="row_id", how="inner")

    if "Confidence" not in joined.columns:
        return pd.DataFrame(
            columns=[
                "row_id",
                "Request Text",
                "Category",
                "Department",
                "Predicted Category",
                "Predicted Department",
                "Confidence",
            ]
        )

    high_conf_mask = joined["Confidence"] == "High"
    errors = joined.loc[
        high_conf_mask
        & (joined["Routing to Department"] != joined["[Agent] Routing to Department"])
    ].copy()

    display_df = errors.rename(
        columns={
            "Routing to Department": "Department",
            "category": "Predicted Category",
            "[Agent] Routing to Department": "Predicted Department",
        }
    )
    cols = [
        "row_id",
        "Request Text",
        "Category",
        "Department",
        "Predicted Category",
        "Predicted Department",
        "Confidence",
    ]
    present_cols = [c for c in cols if c in display_df.columns]
    return display_df[present_cols].copy()


def display_high_confident_errors(
    *,
    eval_df: pd.DataFrame,
    predictions: pd.DataFrame,
    max_rows: int = 10,
) -> pd.DataFrame:
    """Display high-confidence department routing errors and return shown rows."""
    from IPython.display import display

    joined = eval_df.merge(predictions, on="row_id", how="inner")
    errors = get_high_confident_errors(eval_df=eval_df, predictions=predictions)
    high_conf_total = int((joined["Confidence"] == "High").sum()) if "Confidence" in joined else 0
    shown = errors.head(max_rows).copy()

    error_rate = len(errors) / max(1, high_conf_total)
    print(f"High-confidence department errors: {len(errors)}")
    print(f"Error rate within High-confidence predictions: {error_rate:.1%}")
    print("Reminder: some rows may indicate labeling ambiguity/data issues, not only model issues.")
    display(shown)
    return shown


def display_high_confidence_department_errors(
    *,
    eval_df: pd.DataFrame,
    predictions: pd.DataFrame,
    max_rows: int = 10,
) -> pd.DataFrame:
    """Backward-compatible alias for previous helper name."""
    return display_high_confident_errors(
        eval_df=eval_df,
        predictions=predictions,
        max_rows=max_rows,
    )


def display_run_comparison_table(
    *,
    baseline: _HasQualityMetrics,
    new: _HasQualityMetrics,
    new_label: str = "improved prompt",
    eval_df: pd.DataFrame | None = None,
    show_all: bool = False,
) -> pd.DataFrame:
    """Display baseline vs new run metrics table and return it.

    Rows include quality, safety (when ``eval_df`` is provided), cost, and latency
    metrics commonly used across Notebooks A and B.
    """
    from IPython.display import display

    from .evaluation import compute_safety_metrics

    baseline_q = baseline.quality_metrics
    new_q = new.quality_metrics

    baseline_cost = getattr(baseline, "cost_metrics", {}) or {}
    new_cost = getattr(new, "cost_metrics", {}) or {}
    baseline_latency = getattr(baseline, "latency_metrics", {}) or {}
    new_latency = getattr(new, "latency_metrics", {}) or {}

    baseline_safety: dict[str, Any] = {}
    new_safety: dict[str, Any] = {}
    if eval_df is not None and hasattr(baseline, "predictions") and hasattr(new, "predictions"):
        baseline_safety = compute_safety_metrics(eval_df, baseline.predictions)
        new_safety = compute_safety_metrics(eval_df, new.predictions)

    all_metric_rows = [
        "misroute_rate",
        "department_accuracy",
        "category_accuracy",
        "exact_route_match_rate",
        "unsafe_auto_route_rate",
        "auto_route_coverage",
        "auto_route_precision",
        "manual_review_rate",
        "cost_per_message_usd",
        "monthly_cost_usd",
        "median_latency_ms",
        "p95_latency_ms",
    ]
    default_metric_rows = [
        "department_accuracy",
        "category_accuracy",
        "unsafe_auto_route_rate",
        "monthly_cost_usd",
        "p95_latency_ms",
    ]
    metric_rows = all_metric_rows if show_all else default_metric_rows

    baseline_values = {
        "misroute_rate": baseline_q.get("misroute_rate"),
        "department_accuracy": baseline_q.get("department_accuracy"),
        "category_accuracy": baseline_q.get("category_accuracy"),
        "exact_route_match_rate": baseline_q.get("exact_route_match_rate"),
        "unsafe_auto_route_rate": baseline_safety.get("unsafe_auto_route_rate"),
        "auto_route_coverage": baseline_safety.get("auto_route_coverage"),
        "auto_route_precision": baseline_safety.get("auto_route_precision"),
        "manual_review_rate": baseline_safety.get("manual_review_rate"),
        "cost_per_message_usd": baseline_cost.get("cost_per_message_usd"),
        "monthly_cost_usd": baseline_cost.get("monthly_cost_usd"),
        "median_latency_ms": baseline_latency.get("median_latency_ms"),
        "p95_latency_ms": baseline_latency.get("p95_latency_ms"),
    }
    new_values = {
        "misroute_rate": new_q.get("misroute_rate"),
        "department_accuracy": new_q.get("department_accuracy"),
        "category_accuracy": new_q.get("category_accuracy"),
        "exact_route_match_rate": new_q.get("exact_route_match_rate"),
        "unsafe_auto_route_rate": new_safety.get("unsafe_auto_route_rate"),
        "auto_route_coverage": new_safety.get("auto_route_coverage"),
        "auto_route_precision": new_safety.get("auto_route_precision"),
        "manual_review_rate": new_safety.get("manual_review_rate"),
        "cost_per_message_usd": new_cost.get("cost_per_message_usd"),
        "monthly_cost_usd": new_cost.get("monthly_cost_usd"),
        "median_latency_ms": new_latency.get("median_latency_ms"),
        "p95_latency_ms": new_latency.get("p95_latency_ms"),
    }

    baseline_series = pd.Series(
        [_display_round(baseline_values[m]) for m in metric_rows],
        index=metric_rows,
        dtype="object",
    )
    new_series = pd.Series(
        [_display_round(new_values[m]) for m in metric_rows],
        index=metric_rows,
        dtype="object",
    )
    delta_series = pd.to_numeric(new_series, errors="coerce") - pd.to_numeric(
        baseline_series,
        errors="coerce",
    )

    comparison_table = pd.DataFrame(
        {
            "baseline": baseline_series,
            new_label: new_series,
            "delta_vs_baseline": delta_series.map(_display_round),
        },
        index=metric_rows,
    )
    display(comparison_table)
    return comparison_table


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
    """Scatter: department accuracy vs cost.

    Uses ``monthly_cost_usd`` when available, otherwise falls back to
    ``cost_per_message_usd``.
    """
    use_monthly = "monthly_cost_usd" in results.columns and (results["monthly_cost_usd"].sum() > 0)
    cost_col = "monthly_cost_usd" if use_monthly else "cost_per_message_usd"
    x_label = (
        "Estimated monthly cost (USD, 20 000 msgs)" if use_monthly else "Cost per message (USD)"
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    df = results.sort_values(cost_col).copy()

    with plt.rc_context(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
        }
    ):
        for _, row in df.iterrows():
            m = str(row["model"])
            cost = float(row[cost_col])
            dept_acc = float(row["department_accuracy"])
            color = MODEL_PALETTE.get(m, "#4C72B0")
            ax.scatter(
                cost,
                dept_acc,
                s=220,
                color=color,
                edgecolors="white",
                linewidths=1.2,
                zorder=3,
            )
            ax.annotate(
                _short_name(m),
                (cost, dept_acc),
                textcoords="offset points",
                xytext=(12, -6),
                fontsize=9,
                fontweight="bold",
                color=color,
            )

        ax.set_xlabel(x_label)
        ax.set_ylabel("Department routing accuracy")
        ax.set_title("Quality vs Cost", fontweight="semibold", pad=10)
        dec = _DISPLAY_DECIMALS
        if use_monthly:
            ax.xaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, _: f"${x:.{dec}f}"),
            )
        else:
            ax.xaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, _: f"${x:.{dec}f}"),
            )
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x * 100:.{dec}f}%"),
        )
        ax.grid(True, alpha=0.25, linestyle=":", zorder=0)
        ax.set_axisbelow(True)

    fig.tight_layout()
    return fig


def plot_metrics_for_n_shot(
    df: pd.DataFrame,
    *,
    best_model: str,
    n_shot_col: str = "n_shot",
    department_accuracy_col: str = "department_accuracy",
    department_f1_col: str = "department_f1_macro",
    category_accuracy_col: str = "category_accuracy",
    category_f1_col: str = "category_f1_macro",
    cost_total_col: str = "cost_total_usd",
    figsize: tuple[float, float] = (7, 4),
    save_dir: str | Path | None = None,
) -> list[plt.Figure]:
    """Plot department/category accuracy + F1 vs `n_shot`, with total cost bars.

    Returns a list of figures (department, category). When *save_dir* is given,
    figures are saved there and closed automatically.
    """
    required = {
        n_shot_col,
        department_accuracy_col,
        department_f1_col,
        category_accuracy_col,
        category_f1_col,
        cost_total_col,
    }
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    plot_df = df.sort_values(n_shot_col).copy()
    color = MODEL_PALETTE.get(best_model, "#4C72B0")
    short = _short_name(best_model)
    nshots = plot_df[n_shot_col].tolist()

    figs: list[plt.Figure] = []
    for target, acc_col, f1_col, fname in (
        (
            "Department routing",
            department_accuracy_col,
            department_f1_col,
            "few_shot_department.png",
        ),
        ("Category", category_accuracy_col, category_f1_col, "few_shot_category.png"),
    ):
        fig, ax1 = plt.subplots(figsize=figsize)
        ax1.plot(
            plot_df[n_shot_col],
            plot_df[acc_col],
            marker="o",
            lw=2,
            label="Accuracy",
            color=color,
        )
        ax1.plot(
            plot_df[n_shot_col],
            plot_df[f1_col],
            marker="s",
            lw=2,
            label="F1 macro",
            color=color,
            alpha=0.6,
            linestyle="--",
        )
        ax1.set_xlabel("Few-shot examples")
        ax1.set_ylabel("Score")
        ax1.set_ylim(0, 1.05)
        ax1.set_xticks(nshots)
        ax1.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x:.{_DISPLAY_DECIMALS}f}"),
        )
        ax1.legend(loc="lower right", fontsize=9)

        ax2 = ax1.twinx()
        ax2.bar(
            plot_df[n_shot_col],
            plot_df[cost_total_col],
            width=0.35,
            alpha=0.15,
            color=color,
            label="Cost (USD)",
        )
        ax2.set_ylabel("Total cost (USD) on eval slice")
        ax2.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x:.{_DISPLAY_DECIMALS}f}"),
        )
        ax2.legend(loc="upper left", fontsize=8)

        ax1.set_title(
            f"{short} — {target} (metrics & cost vs n-shot)",
            fontweight="bold",
        )
        fig.tight_layout()

        if save_dir is not None:
            save_figure(fig, Path(save_dir) / fname)
        else:
            plt.show()
            plt.close(fig)
        figs.append(fig)
    return figs


def plot_latency_for_n_shots(
    df: pd.DataFrame,
    *,
    best_model: str,
    n_shot_col: str = "n_shot",
    median_latency_col: str = "median_latency_ms",
    mean_latency_col: str = "mean_latency_ms",
    p95_latency_col: str = "p95_latency_ms",
    figsize: tuple[float, float] = (7, 4),
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot median/mean/p95 latency vs `n_shot`.

    Returns the figure. When *save_path* is given, saves and closes automatically.
    """
    required = {
        n_shot_col,
        median_latency_col,
        mean_latency_col,
        p95_latency_col,
    }
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    plot_df = df.sort_values(n_shot_col).copy()
    color = MODEL_PALETTE.get(best_model, "#4C72B0")
    short = _short_name(best_model)
    nshots = plot_df[n_shot_col].tolist()

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(
        plot_df[n_shot_col],
        plot_df[median_latency_col],
        marker="o",
        lw=2,
        label="Median latency (ms)",
        color=color,
    )
    ax.plot(
        plot_df[n_shot_col],
        plot_df[mean_latency_col],
        marker="^",
        lw=2,
        label="Mean latency (ms)",
        color=color,
        linestyle="--",
        alpha=0.75,
    )
    ax.plot(
        plot_df[n_shot_col],
        plot_df[p95_latency_col],
        marker="s",
        lw=2,
        label="p95 latency (ms)",
        color=color,
        linestyle=":",
        alpha=0.75,
    )
    ax.set_xlabel("Few-shot examples")
    ax.set_ylabel("Latency (ms)")
    ax.set_xticks(nshots)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:.{_DISPLAY_DECIMALS}f}"),
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title(f"{short} — Latency vs n-shot", fontweight="bold")
    fig.tight_layout()

    if save_path is not None:
        save_figure(fig, save_path)
    else:
        plt.show()
        plt.close(fig)
    return fig


def plot_safety_comparison(results: pd.DataFrame) -> plt.Figure:
    """Grouped bar chart comparing safety metrics across models.

    Expects columns: model, auto_route_coverage, auto_route_precision,
    unsafe_auto_route_rate, manual_review_rate.
    """
    metrics = [
        ("auto_route_coverage", "Auto-route Coverage"),
        ("auto_route_precision", "Auto-route Precision"),
        ("unsafe_auto_route_rate", "Unsafe Auto-route Rate"),
        ("manual_review_rate", "Manual Review Rate"),
    ]
    present = [(col, label) for col, label in metrics if col in results.columns]
    if not present:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, "No safety data", ha="center", va="center")
        return fig

    df = results.copy()
    df["short_model"] = df["model"].apply(_short_name)
    n_models = len(df)
    n_metrics = len(present)
    x = range(n_models)
    width = 0.8 / n_metrics

    fig, ax = plt.subplots(figsize=(max(7, n_models * 1.8), 5))
    with plt.rc_context(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
        }
    ):
        for i, (col, label) in enumerate(present):
            offset = (i - n_metrics / 2 + 0.5) * width
            bars = ax.bar(
                [xi + offset for xi in x],
                df[col].tolist(),
                width=width,
                label=label,
                edgecolor="white",
                linewidth=0.8,
            )
            dec = _DISPLAY_DECIMALS
            for bar in bars:
                h = bar.get_height()
                if h > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        h + 0.01,
                        f"{h:.{dec}f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

        ax.set_xticks(list(x))
        ax.set_xticklabels(df["short_model"].tolist(), rotation=30, ha="right")
        ax.set_ylabel("Rate")
        ax.set_ylim(0, 1.15)
        dec = _DISPLAY_DECIMALS
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{v * 100:.{dec}f}%"),
        )
        ax.set_title(
            "Safety Metrics — Model Comparison",
            fontweight="semibold",
            pad=10,
        )
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, axis="y", alpha=0.25, linestyle=":", zorder=0)
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
        dec = _DISPLAY_DECIMALS
        ax.bar_label(
            bars,
            labels=[f"{float(n):.{dec}f}" for n in counts],
            padding=3,
            fontsize=9,
            fontweight="medium",
        )
        ax.set_xticks(list(xs))
        ax.set_xticklabels(df["confidence"].tolist())
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Count")
        ax.set_title("Confidence buckets", fontweight="semibold", pad=10)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x:.{dec}f}"),
        )
        ax.grid(True, axis="y", alpha=0.25, linestyle=":", zorder=0)
        ax.set_axisbelow(True)

    fig.tight_layout()
    return fig


# --- Notebook A: compact tables and single-shot figure display ----------------


def show_figure(fig: plt.Figure) -> None:
    """Show *fig* once in Jupyter ``inline`` backends, then close it.

    Using ``display(fig)`` alone often produces **two** identical figures because
    the inline backend also flushes open figures at the end of the cell.
    """
    from IPython.display import display

    display(fig)
    plt.close(fig)


def quality_metrics_summary_table(
    quality_metrics: dict[str, Any],
    *,
    show_all: bool = False,
) -> pd.DataFrame:
    row: dict[str, Any] = {
        "Department accuracy": _display_round(quality_metrics["department_accuracy"]),
        "Category accuracy": _display_round(quality_metrics["category_accuracy"]),
        "Rows evaluated": _display_round(quality_metrics.get("row_count", 0)),
    }
    if show_all:
        row = {
            "Misroute rate": _display_round(quality_metrics["misroute_rate"]),
            **row,
        }
    return pd.DataFrame([row])


def display_quality_metrics(*, quality_metrics: dict[str, Any], show_all: bool = False) -> None:
    from IPython.display import display

    display(quality_metrics_summary_table(quality_metrics, show_all=show_all))


def safety_metrics_summary_table(safety_metrics: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Auto-route coverage": _display_round(safety_metrics["auto_route_coverage"]),
                "Auto-route precision": _display_round(safety_metrics["auto_route_precision"]),
                "Unsafe auto-route rate": _display_round(safety_metrics["unsafe_auto_route_rate"]),
                "Manual review rate": _display_round(safety_metrics["manual_review_rate"]),
            },
        ],
    )


def display_safety_metrics(*, safety_metrics: dict[str, float]) -> None:
    from IPython.display import display

    display(safety_metrics_summary_table(safety_metrics))


def cost_metrics_summary_table(cost_metrics: dict[str, Any]) -> pd.DataFrame:
    cost_per_message = cost_metrics.get("cost_per_message_usd")
    cost_per_1k = None if cost_per_message is None else float(cost_per_message) * 1000.0
    return pd.DataFrame(
        [
            {
                "Cost per 1,000 messages (USD)": _display_round(cost_per_1k),
                "Monthly cost (USD)": _display_round(cost_metrics.get("monthly_cost_usd")),
                "Annual cost (USD)": _display_round(cost_metrics.get("annual_cost_usd")),
            },
        ],
    )


def display_cost_metrics(*, cost_metrics: dict[str, Any]) -> None:
    from IPython.display import display

    display(cost_metrics_summary_table(cost_metrics))


def cost_assumptions_table(
    *,
    model: str,
    cost_metrics: dict[str, Any],
    monthly_messages: int,
) -> pd.DataFrame:
    pricing = MODEL_REGISTRY.get(model, {})
    return pd.DataFrame(
        [
            {"Assumption": "Model", "Value": model},
            {"Assumption": "Messages per month", "Value": monthly_messages},
            {
                "Assumption": "Avg input tokens per message",
                "Value": float(cost_metrics.get("avg_prompt_tokens") or 0.0),
            },
            {
                "Assumption": "Avg output tokens per message",
                "Value": float(cost_metrics.get("avg_completion_tokens") or 0.0),
            },
            {
                "Assumption": "Input price (USD / 1M tokens)",
                "Value": float(pricing.get("input") or 0.0),
            },
            {
                "Assumption": "Output price (USD / 1M tokens)",
                "Value": float(pricing.get("output") or 0.0),
            },
            {"Assumption": "Cost source", "Value": str(cost_metrics.get("cost_source", "unknown"))},
        ],
    )


def token_economics_table(*, model: str, cost_metrics: dict[str, Any]) -> pd.DataFrame:
    pricing = MODEL_REGISTRY.get(model, {})
    avg_prompt_tokens = float(cost_metrics.get("avg_prompt_tokens") or 0.0)
    avg_completion_tokens = float(cost_metrics.get("avg_completion_tokens") or 0.0)
    input_price_per_1m = float(pricing.get("input") or 0.0)
    output_price_per_1m = float(pricing.get("output") or 0.0)

    input_cost_per_message = avg_prompt_tokens * input_price_per_1m / 1_000_000
    output_cost_per_message = avg_completion_tokens * output_price_per_1m / 1_000_000

    return pd.DataFrame(
        [
            {
                "Token type": "Input",
                "Avg tokens / message": avg_prompt_tokens,
                "Price (USD / 1M tokens)": input_price_per_1m,
                "Cost / message (USD)": input_cost_per_message,
            },
            {
                "Token type": "Output",
                "Avg tokens / message": avg_completion_tokens,
                "Price (USD / 1M tokens)": output_price_per_1m,
                "Cost / message (USD)": output_cost_per_message,
            },
            {
                "Token type": "Total",
                "Avg tokens / message": avg_prompt_tokens + avg_completion_tokens,
                "Price (USD / 1M tokens)": float("nan"),
                "Cost / message (USD)": float(cost_metrics.get("cost_per_message_usd") or 0.0),
            },
        ],
    )


def cost_projection_table(*, cost_metrics: dict[str, Any], monthly_messages: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Metric": "Messages per month", "Value": monthly_messages},
            {
                "Metric": "Cost per message (USD)",
                "Value": float(cost_metrics.get("cost_per_message_usd") or 0.0),
            },
            {
                "Metric": "Monthly cost (USD)",
                "Value": float(cost_metrics.get("monthly_cost_usd") or 0.0),
            },
            {
                "Metric": "Annual cost (USD)",
                "Value": float(cost_metrics.get("annual_cost_usd") or 0.0),
            },
        ],
    )


def display_cost_breakdown(
    *,
    model: str,
    cost_metrics: dict[str, Any],
    monthly_messages: int,
) -> None:
    from IPython.display import display

    display(
        cost_assumptions_table(
            model=model, cost_metrics=cost_metrics, monthly_messages=monthly_messages
        )
    )
    display(token_economics_table(model=model, cost_metrics=cost_metrics))


def display_cost_projection(
    *,
    cost_metrics: dict[str, Any],
    monthly_messages: int,
) -> None:
    from IPython.display import display

    display(cost_projection_table(cost_metrics=cost_metrics, monthly_messages=monthly_messages))
    monthly_cost = float(cost_metrics.get("monthly_cost_usd") or 0.0)
    annual_cost = float(cost_metrics.get("annual_cost_usd") or 0.0)
    print(
        f"At {monthly_messages:,} messages/month, projected cost is "
        f"${monthly_cost:,.2f}/month (~${annual_cost:,.2f}/year)."
    )


def latency_metrics_summary_table(latency_metrics: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Median latency (ms)": _display_round(latency_metrics.get("median_latency_ms")),
                "p95 latency (ms)": _display_round(latency_metrics.get("p95_latency_ms")),
                "Latency source": latency_metrics.get("latency_source"),
            },
        ],
    )


def display_latency_metrics(*, latency_metrics: dict[str, Any]) -> None:
    from IPython.display import display

    display(latency_metrics_summary_table(latency_metrics))


def display_prediction_confidence_chart(*, predictions: pd.DataFrame) -> None:
    from IPython.display import Markdown, display

    if "Confidence" not in predictions.columns:
        display(
            Markdown("*No `Confidence` column in model output — skipped confidence chart.*"),
        )
        return
    bucket_df = predictions["Confidence"].fillna("Unknown").value_counts().reset_index()
    bucket_df.columns = pd.Index(["confidence", "count"])
    show_figure(plot_confidence_buckets(bucket_df))


def mvp_decision_summary_table(decision: DecisionSummary) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for d in decision.dimensions:
        rows.append(
            {
                "Dimension": d.name.replace("_", " ").title(),
                "Status": d.status,
                "Value": _display_round(d.value),
                "Pass threshold (≤)": _display_round(d.threshold_pass),
            },
        )
    return pd.DataFrame(rows)


def display_mvp_decision(*, decision: DecisionSummary) -> None:
    from IPython.display import display

    display(mvp_decision_summary_table(decision))
