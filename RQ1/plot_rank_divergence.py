#!/usr/bin/env python3
"""
RQ1 Figure: Bump charts showing construct rank divergence
between established (PVQ-40, PVQ-21) and ecological methods.

Usage:
    python RQ1/plot_rank_divergence.py
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ANALYSIS = ROOT / "results" / "RQ1" / "analysis.json"
DEFAULT_OUT_DIR = ROOT / "results" / "RQ1" / "figures"

VALUE_COLORS = {
    "Self-Direction": "#E63946",
    "Self_Direction": "#E63946",
    "Stimulation":    "#F4A261",
    "Hedonism":       "#E9C46A",
    "Achievement":    "#264653",
    "Power":          "#2A9D8F",
    "Security":       "#457B9D",
    "Conformity":     "#1D3557",
    "Tradition":      "#6D6875",
    "Universalism":   "#06D6A0",
    "Benevolence":    "#118AB2",
}

DISPLAY_NAMES = {
    "Self_Direction": "Self-Direction",
    "Self-Direction": "Self-Direction",
    "Stimulation": "Stimulation",
    "Hedonism": "Hedonism",
    "Achievement": "Achievement",
    "Power": "Power",
    "Security": "Security",
    "Conformity": "Conformity",
    "Tradition": "Tradition",
    "Universalism": "Universalism",
    "Benevolence": "Benevolence",
}

MODEL_SHORT = {
    "Qwen2.5-72B-Instruct": "Qwen2.5-72B",
    "Qwen2.5-7B-Instruct": "Qwen2.5-7B",
    "Qwen3-235B-A22B-Instruct-2507-FP8": "Qwen3-235B",
    "Qwen3-30B-A3B-Instruct-2507": "Qwen3-30B",
    "gemma-3-27b-it": "Gemma-3-27B",
    "gemma-3-4b-it": "Gemma-3-4B",
    "gpt-oss-120b": "GPT-oss-120B",
    "gpt-oss-20b": "GPT-oss-20B",
}


def _canon(name):
    return name.replace("-", "_").replace(" ", "_")


def load_data(analysis_path: Path):
    data = json.loads(analysis_path.read_text())
    ranks = data["analysis_2_rank_comparison"]
    rho_data = data["analysis_1_spearman_rho"]
    return ranks, rho_data


def get_rho(rho_data, model, key):
    entry = rho_data.get(model, {}).get(key, {})
    return entry.get("observed_rho")


def style_panel(ax, method_labels, n_ranks, show_ylabel=False):
    """Apply a paper-friendly shared style to each bump chart panel."""
    x_pos = np.arange(len(method_labels))

    for xpos in x_pos:
        ax.axvline(x=xpos, color="#E6E6E6", linewidth=0.8, zorder=0)

    ax.set_xlim(-0.45, len(method_labels) - 0.55)
    ax.set_ylim(n_ranks + 0.5, 0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(method_labels, fontsize=8, fontweight="bold")
    ax.yaxis.set_major_locator(mticker.MultipleLocator(1))
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", labelsize=7, length=0)
    ax.grid(axis="y", color="#D9D9D9", alpha=0.7, linewidth=0.6)
    ax.set_axisbelow(True)

    if show_ylabel:
        ax.set_ylabel("Rank", fontsize=8, fontweight="bold")

    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#BDBDBD")
    ax.spines["bottom"].set_linewidth(0.8)


def add_panel_header(ax, title):
    ax.set_title(title, fontsize=9.5, fontweight="bold", pad=10)


def draw_bump_panel(ax, model, rank_data, rho_pvq, rho_pvq21):
    """Draw one bump chart panel for PVQ values."""
    pvq_data = rank_data.get("pvq_values", {})

    methods = ["PVQ", "PVQ21", "ecological"]
    method_labels = ["PVQ-40", "PVQ-21", "VP"]
    x_pos = [0, 1, 2]

    construct_ranks = {}
    for method in methods:
        items = pvq_data.get(method, [])
        for item in items:
            canon = _canon(item["name"])
            if canon not in construct_ranks:
                construct_ranks[canon] = {}
            construct_ranks[canon][method] = item["rank"]

    n = max(
        max(r.values()) for r in construct_ranks.values() if r
    )

    for canon, method_ranks in construct_ranks.items():
        if len(method_ranks) < 2:
            continue

        ys = [method_ranks.get(m) for m in methods]
        xs = [x_pos[i] for i, y in enumerate(ys) if y is not None]
        ys_clean = [y for y in ys if y is not None]

        color = VALUE_COLORS.get(canon, VALUE_COLORS.get(canon.replace("_", "-"), "#999"))
        display = DISPLAY_NAMES.get(canon, canon)

        ax.plot(
            xs,
            ys_clean,
            color=color,
            linewidth=2.0,
            alpha=0.9,
            marker="o",
            markersize=5.2,
            zorder=3,
        )

        if ys[0] is not None:
            ax.text(
                -0.08,
                ys[0],
                display,
                ha="right",
                va="center",
                fontsize=6.7,
                color=color,
                fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", pad=0.3, alpha=0.9),
            )
        if ys[-1] is not None:
            ax.text(
                2.08,
                ys[-1],
                display,
                ha="left",
                va="center",
                fontsize=6.7,
                color=color,
                fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", pad=0.3, alpha=0.9),
            )

    style_panel(ax, method_labels, n)

    short = MODEL_SHORT.get(model, model[:20])
    add_panel_header(ax, short)


def draw_bfi_panel(ax, model, rank_data, rho_bfi44, rho_bfi10):
    """Draw one bump chart panel for BFI traits."""
    bfi_data = rank_data.get("bfi_traits", {})

    methods = ["BFI44", "BFI10", "ecological"]
    method_labels = ["BFI-44", "BFI-10", "VP"]
    x_pos = [0, 1, 2]

    BFI_COLORS = {
        "Openness": "#E63946",
        "Conscientiousness": "#457B9D",
        "Extraversion": "#F4A261",
        "Agreeableness": "#06D6A0",
        "Neuroticism": "#6D6875",
    }

    construct_ranks = {}
    for method in methods:
        items = bfi_data.get(method, [])
        for item in items:
            name = item["name"]
            if name not in construct_ranks:
                construct_ranks[name] = {}
            construct_ranks[name][method] = item["rank"]

    n = max(max(r.values()) for r in construct_ranks.values() if r)

    for name, method_ranks in construct_ranks.items():
        ys = [method_ranks.get(m) for m in methods]
        xs = [x_pos[i] for i, y in enumerate(ys) if y is not None]
        ys_clean = [y for y in ys if y is not None]

        color = BFI_COLORS.get(name, "#999")

        ax.plot(
            xs,
            ys_clean,
            color=color,
            linewidth=2.3,
            alpha=0.9,
            marker="o",
            markersize=6,
            zorder=3,
        )

        if ys[0] is not None:
            ax.text(
                -0.08,
                ys[0],
                name,
                ha="right",
                va="center",
                fontsize=7.3,
                color=color,
                fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", pad=0.3, alpha=0.9),
            )
        if ys[-1] is not None:
            ax.text(
                2.08,
                ys[-1],
                name,
                ha="left",
                va="center",
                fontsize=7.3,
                color=color,
                fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", pad=0.3, alpha=0.9),
            )

    style_panel(ax, method_labels, n)

    short = MODEL_SHORT.get(model, model[:20])
    add_panel_header(ax, short)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analysis-path",
        type=Path,
        default=DEFAULT_ANALYSIS,
        help="Path to analysis JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for output figures. Defaults to same parent as analysis file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    analysis_path = args.analysis_path
    if not analysis_path.is_absolute():
        analysis_path = ROOT / analysis_path
    out_dir = args.output_dir
    if out_dir is None:
        out_dir = analysis_path.parent / "figures"
    elif not out_dir.is_absolute():
        out_dir = ROOT / out_dir

    ranks, rho_data = load_data(analysis_path)
    models = sorted(ranks.keys())
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── PVQ bump chart (4×2) ──
    fig, axes = plt.subplots(2, 4, figsize=(22, 10), sharey=True)

    for idx, model in enumerate(models):
        row, col = divmod(idx, 4)
        ax = axes[row][col]
        rho_pvq = get_rho(rho_data, model, "PVQ_pvq_values")
        rho_pvq21 = get_rho(rho_data, model, "PVQ21_pvq_values")
        draw_bump_panel(ax, model, ranks[model], rho_pvq, rho_pvq21)
        if col == 0:
            ax.set_ylabel("Rank", fontsize=8, fontweight="bold")

    for idx in range(len(models), axes.size):
        axes.flat[idx].set_visible(False)

    fig.tight_layout(pad=2.0, w_pad=2.4, h_pad=2.8)
    path = out_dir / "rank_divergence_pvq.pdf"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"Saved: {path.relative_to(ROOT)}")
    plt.close(fig)

    # ── BFI bump chart (4×2) ──
    fig2, axes2 = plt.subplots(2, 4, figsize=(22, 9), sharey=True)

    for idx, model in enumerate(models):
        row, col = divmod(idx, 4)
        ax = axes2[row][col]
        rho_bfi44 = get_rho(rho_data, model, "BFI44_bfi_traits")
        rho_bfi10 = get_rho(rho_data, model, "BFI10_bfi_traits")
        draw_bfi_panel(ax, model, ranks[model], rho_bfi44, rho_bfi10)
        if col == 0:
            ax.set_ylabel("Rank", fontsize=8, fontweight="bold")

    for idx in range(len(models), axes2.size):
        axes2.flat[idx].set_visible(False)

    fig2.tight_layout(pad=2.0, w_pad=2.4, h_pad=2.8)
    path2 = out_dir / "rank_divergence_bfi.pdf"
    fig2.savefig(path2, dpi=200, bbox_inches="tight")
    fig2.savefig(path2.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"Saved: {path2.relative_to(ROOT)}")
    plt.close(fig2)


if __name__ == "__main__":
    main()
