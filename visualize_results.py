#!/usr/bin/env python3
"""
Make plots from aggregated simplification metrics.

Expected CSV columns (your current format):
dataset,model,variant,n,error_rate,words_mean,fk_grade_mean,flesch_mean,fog_mean,smog_mean,
rougeL_mean,trigram_copy_mean,compression_mean

Usage:
  python make_plots.py --csv new_data_metrics.csv --outdir figs
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ---------------------------
# Data loading / preparation
# ---------------------------

METRIC_COLS = {
    "Words": "words_mean",
    "FKGL": "fk_grade_mean",
    "FRE": "flesch_mean",
    "Fog": "fog_mean",
    "SMOG": "smog_mean",
    "ROUGE_L": "rougeL_mean",
    "TriCopy": "trigram_copy_mean",
    "Comp": "compression_mean",
}

VARIANT_ORDER = ["easy_general", "adult_non_technical", "adult_technical"]
DATASET_TITLE = {"arxiv": "arXiv", "wiki": "Wikipedia"}

MARKER_BY_VARIANT = {
    "easy_general": "o",
    "adult_non_technical": "s",
    "adult_technical": "^",
}


def load_and_prepare(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Basic sanity checks
    required = ["dataset", "model", "variant"] + list(METRIC_COLS.values())
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    # Rename to nicer internal names (keep original columns too if you want)
    df = df.copy()
    for nice, raw in METRIC_COLS.items():
        df[nice] = df[raw]

    # Treat reuse/comp as undefined for original row (helps plotting)
    is_orig = df["variant"].eq("original")
    df.loc[is_orig, ["ROUGE_L", "TriCopy", "Comp"]] = np.nan

    # Compute dataset baselines from the "original" rows
    base = (
        df[df["variant"].eq("original")]
        .groupby("dataset")[["FKGL", "FRE", "Fog", "SMOG", "Words"]]
        .mean()
        .rename(
            columns={
                "FKGL": "FKGL_base",
                "FRE": "FRE_base",
                "Fog": "Fog_base",
                "SMOG": "SMOG_base",
                "Words": "Words_base",
            }
        )
        .reset_index()
    )

    df = df.merge(base, on="dataset", how="left")

    # Delta metrics (improvement = positive)
    df["dFKGL"] = df["FKGL_base"] - df["FKGL"]
    df["dFRE"] = df["FRE"] - df["FRE_base"]
    df["dFog"] = df["Fog_base"] - df["Fog"]
    df["dSMOG"] = df["SMOG_base"] - df["SMOG"]

    # Novelty is often more intuitive (higher is better)
    df["TriNovelty"] = 1.0 - df["TriCopy"]

    return df


def add_simplification_avg(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a derived "simplification_avg" row per (dataset, model) as the mean of:
      easy_general + adult_non_technical
    """
    simp = df[df["variant"].isin(["easy_general", "adult_non_technical"])].copy()
    agg_cols = [
        "Words", "FKGL", "FRE", "Fog", "SMOG", "ROUGE_L", "TriCopy", "TriNovelty", "Comp",
        "dFKGL", "dFRE", "dFog", "dSMOG",
        "Words_base", "FKGL_base", "FRE_base", "Fog_base", "SMOG_base",
    ]

    simp_avg = (
        simp.groupby(["dataset", "model"], as_index=False)[agg_cols]
        .mean(numeric_only=True)
    )
    simp_avg["variant"] = "simplification_avg"

    # Keep n/error_rate if present by averaging (safe for your constant values)
    for extra in ["n", "error_rate"]:
        if extra in df.columns:
            simp_avg[extra] = (
                simp.groupby(["dataset", "model"], as_index=False)[extra].mean(numeric_only=True)[extra]
            )

    out = pd.concat([df, simp_avg], ignore_index=True)
    return out


# ---------------------------
# Plot helpers
# ---------------------------

def ensure_outdir(outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)


def savefig_all_formats(fig: plt.Figure, outdir: str, stem: str) -> None:
    png_path = os.path.join(outdir, f"{stem}.png")
    pdf_path = os.path.join(outdir, f"{stem}.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def get_model_colors(models: List[str]) -> Dict[str, Tuple[float, float, float, float]]:
    """
    Stable mapping model -> color using matplotlib default cycle.
    """
    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not prop_cycle:
        # fallback
        prop_cycle = ["C0", "C1", "C2", "C3", "C4", "C5"]
    colors = {}
    for i, m in enumerate(models):
        colors[m] = prop_cycle[i % len(prop_cycle)]
    return colors


# ---------------------------
# Plots
# ---------------------------

def plot_tradeoff_scatter(
    df: pd.DataFrame,
    dataset: str,
    outdir: str,
    y_col: str,
    y_label: str,
    x_col: str = "TriNovelty",
    x_label: str = "Trigram novelty (1 - TriCopy)",
    size_col: str = "Comp",
    title_suffix: str = "",
) -> None:
    """
    Scatter: x vs y, color=model, marker=variant, point size ~ Comp.
    """
    d = df[(df["dataset"] == dataset) & (df["variant"].isin(VARIANT_ORDER))].copy()
    d = d.dropna(subset=[x_col, y_col])

    models = sorted(d["model"].unique().tolist())
    colors = get_model_colors(models)

    # Scale point sizes for visual balance
    sizes = 220.0 * (d[size_col].fillna(1.0).to_numpy())
    sizes = np.clip(sizes, 60.0, 600.0)

    fig = plt.figure()
    ax = plt.gca()

    # Plot each point (model x variant)
    for i, row in enumerate(d.itertuples(index=False)):
        ax.scatter(
            getattr(row, x_col),
            getattr(row, y_col),
            s=float(sizes[i]),
            marker=MARKER_BY_VARIANT.get(row.variant, "o"),
            color=colors.get(row.model, "C0"),
            alpha=0.9,
            edgecolors="none",
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    title = f"{DATASET_TITLE.get(dataset, dataset)} trade-off{title_suffix}"
    ax.set_title(title)

    # Legends: models (colors) + variants (markers)
    model_handles = [
        Line2D([0], [0], marker="o", linestyle="", color=colors[m], label=m, markersize=7)
        for m in models
    ]
    variant_handles = [
        Line2D([0], [0], marker=MARKER_BY_VARIANT[v], linestyle="", color="black", label=v, markersize=7)
        for v in VARIANT_ORDER
    ]

    leg1 = ax.legend(handles=model_handles, title="Model", loc="best", frameon=True)
    ax.add_artist(leg1)
    ax.legend(handles=variant_handles, title="Prompt", loc="lower right", frameon=True)

    # Light grid helps readability
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.7)

    stem = f"scatter_{dataset}_{y_col}_vs_{x_col}"
    savefig_all_formats(fig, outdir, stem)


def plot_grouped_bars(
    df: pd.DataFrame,
    dataset: str,
    outdir: str,
    metric: str,
    ylabel: str,
) -> None:
    """
    Grouped bars by model, bars=variants.
    """
    d = df[(df["dataset"] == dataset) & (df["variant"].isin(VARIANT_ORDER))].copy()

    models = sorted(d["model"].unique().tolist())
    x = np.arange(len(models))
    width = 0.22

    fig = plt.figure()
    ax = plt.gca()

    # Plot each variant as a bar group
    for j, v in enumerate(VARIANT_ORDER):
        dv = d[d["variant"] == v].set_index("model").reindex(models)
        ax.bar(x + (j - 1) * width, dv[metric].to_numpy(), width=width, label=v)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=0)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{DATASET_TITLE.get(dataset, dataset)}: {ylabel} by model and prompt")
    ax.legend(loc="best", frameon=True)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.8, alpha=0.7)

    stem = f"bars_{dataset}_{metric}"
    savefig_all_formats(fig, outdir, stem)


def plot_slope_simplavg_to_technical(
    df_with_avg: pd.DataFrame,
    dataset: str,
    outdir: str,
    metric: str,
    ylabel: str,
) -> None:
    """
    Slope plot connecting simplification_avg -> adult_technical per model.
    Requires df with variant == simplification_avg already added.
    """
    d = df_with_avg[df_with_avg["dataset"] == dataset].copy()

    simp = d[d["variant"] == "simplification_avg"].set_index("model")
    tech = d[d["variant"] == "adult_technical"].set_index("model")

    models = sorted(list(set(simp.index).intersection(set(tech.index))))
    colors = get_model_colors(models)

    fig = plt.figure()
    ax = plt.gca()

    x0, x1 = 0, 1
    for m in models:
        y0 = float(simp.loc[m, metric])
        y1 = float(tech.loc[m, metric])
        ax.plot([x0, x1], [y0, y1], marker="o", color=colors[m], label=m)

    ax.set_xticks([x0, x1])
    ax.set_xticklabels(["simplification_avg", "adult_technical"])
    ax.set_ylabel(ylabel)
    ax.set_title(f"{DATASET_TITLE.get(dataset, dataset)}: shift from simplification → technical")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.8, alpha=0.7)

    # Keep one legend entry per model
    ax.legend(loc="best", frameon=True)

    stem = f"slope_{dataset}_simplavg_to_technical_{metric}"
    savefig_all_formats(fig, outdir, stem)


def plot_reuse_vs_length(
    df: pd.DataFrame,
    dataset: str,
    outdir: str,
    x_col: str = "Comp",
    y_col: str = "TriCopy",
) -> None:
    """
    Scatter Comp vs TriCopy, marker=variant, color=model.
    """
    d = df[(df["dataset"] == dataset) & (df["variant"].isin(VARIANT_ORDER))].copy()
    d = d.dropna(subset=[x_col, y_col])

    models = sorted(d["model"].unique().tolist())
    colors = get_model_colors(models)

    fig = plt.figure()
    ax = plt.gca()

    for row in d.itertuples(index=False):
        ax.scatter(
            getattr(row, x_col),
            getattr(row, y_col),
            s=220,
            marker=MARKER_BY_VARIANT.get(row.variant, "o"),
            color=colors.get(row.model, "C0"),
            alpha=0.9,
            edgecolors="none",
        )

    ax.set_xlabel("Compression ratio (Comp = W_gen / W_orig)")
    ax.set_ylabel("TriCopy (lower = less surface reuse)")
    ax.set_title(f"{DATASET_TITLE.get(dataset, dataset)}: reuse vs verbosity")
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.7)

    # Legends
    model_handles = [
        Line2D([0], [0], marker="o", linestyle="", color=colors[m], label=m, markersize=7)
        for m in models
    ]
    variant_handles = [
        Line2D([0], [0], marker=MARKER_BY_VARIANT[v], linestyle="", color="black", label=v, markersize=7)
        for v in VARIANT_ORDER
    ]
    leg1 = ax.legend(handles=model_handles, title="Model", loc="best", frameon=True)
    ax.add_artist(leg1)
    ax.legend(handles=variant_handles, title="Prompt", loc="lower right", frameon=True)

    stem = f"scatter_{dataset}_reuse_vs_length"
    savefig_all_formats(fig, outdir, stem)


# ---------------------------
# Main
# ---------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to aggregated metrics CSV")
    ap.add_argument("--outdir", default="figs", help="Output directory for figures")
    args = ap.parse_args()

    ensure_outdir(args.outdir)

    df = load_and_prepare(args.csv)
    df_avg = add_simplification_avg(df)

    for dataset in sorted(df["dataset"].unique()):
        # Trade-off: (1 - TriCopy) vs dFKGL
        plot_tradeoff_scatter(
            df=df,
            dataset=dataset,
            outdir=args.outdir,
            y_col="dFKGL",
            y_label="ΔFKGL (FKGL_orig − FKGL_gen)",
            x_col="TriNovelty",
            x_label="Trigram novelty (1 − TriCopy)",
            size_col="Comp",
        )

        # Trade-off: (1 - TriCopy) vs dFRE
        plot_tradeoff_scatter(
            df=df,
            dataset=dataset,
            outdir=args.outdir,
            y_col="dFRE",
            y_label="ΔFRE (FRE_gen − FRE_orig)",
            x_col="TriNovelty",
            x_label="Trigram novelty (1 − TriCopy)",
            size_col="Comp",
        )

        # Grouped bars: improvement in FKGL and FRE
        plot_grouped_bars(
            df=df,
            dataset=dataset,
            outdir=args.outdir,
            metric="dFKGL",
            ylabel="ΔFKGL (orig − gen)",
        )
        plot_grouped_bars(
            df=df,
            dataset=dataset,
            outdir=args.outdir,
            metric="dFRE",
            ylabel="ΔFRE (gen − orig)",
        )

        # Slope: simplification_avg -> adult_technical
        plot_slope_simplavg_to_technical(
            df_with_avg=df_avg,
            dataset=dataset,
            outdir=args.outdir,
            metric="dFKGL",
            ylabel="ΔFKGL (orig − gen)",
        )
        plot_slope_simplavg_to_technical(
            df_with_avg=df_avg,
            dataset=dataset,
            outdir=args.outdir,
            metric="Comp",
            ylabel="Compression ratio (Comp)",
        )

        # Reuse vs length
        plot_reuse_vs_length(
            df=df,
            dataset=dataset,
            outdir=args.outdir,
            x_col="Comp",
            y_col="TriCopy",
        )

    print(f"Done. Figures written to: {args.outdir}")


if __name__ == "__main__":
    main()
