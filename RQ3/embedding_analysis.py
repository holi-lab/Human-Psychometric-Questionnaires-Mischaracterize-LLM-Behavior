#!/usr/bin/env python3
"""
RQ3: Embedding-based analysis — Construct Recognition & Lexical-Semantic Clustering

Investigates why construct structure appears in established questionnaires
but not in VP (ecological) scenarios through two complementary lenses:

  A. Item-Construct Semantic Similarity (Construct Recognition Cue):
     Cosine similarity between item text and construct names/definitions.
     High similarity → model can infer what an item measures from text alone.
     Computed with both construct names and full definitions.

  B. Lexical-Semantic Clustering (LSC):
     Intra-construct vs inter-construct pairwise item similarity.
     High gap → items within a construct share surface wording,
     potentially generating construct structure without true understanding.

Uses general-purpose sentence-transformers to measure intrinsic textual
properties (model-agnostic: any LLM would pick up the same cues).

Usage:
    python RQ3/embedding_analysis.py

Outputs:
    results/RQ3/embedding_analysis/
"""

import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

ROOT = Path(__file__).resolve().parent.parent
SURVEYS_DIR = ROOT / "surveys"
OUT_DIR = ROOT / "results" / "RQ3" / "embedding_analysis"
FIG_DIR = OUT_DIR / "figures"

VP_POSITIVE_THRESHOLD = 0.3
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"

CONSTRUCT_DEFS = {
    "pvq_values": {
        "Power": "Social status and prestige, control or dominance over people and resources",
        "Achievement": "Personal success through demonstrating competence according to social standards",
        "Hedonism": "Pleasure and sensuous gratification for oneself",
        "Stimulation": "Excitement, novelty, and challenge in life",
        "Self-Direction": "Independent thought and action - choosing, creating, exploring",
        "Universalism": "Understanding, appreciation, tolerance, and protection for the welfare of all people and for nature",
        "Benevolence": "Preserving and enhancing the welfare of people with whom one is in frequent personal contact",
        "Tradition": "Respect, commitment, and acceptance of the customs and ideas that one's culture or religion provides",
        "Conformity": "Restraint of actions, inclinations, and impulses likely to upset or harm others and violate social expectations or norms",
        "Security": "Safety, harmony, and stability of society, of relationships, and of self",
    },
    "bfi_traits": {
        "Openness": "The tendency to be imaginative, curious, and receptive to new ideas, experiences, and perspectives",
        "Conscientiousness": "The tendency to be organized, responsible, hardworking, and goal-directed",
        "Extraversion": "The tendency to be sociable, assertive, active, and to experience positive emotions",
        "Agreeableness": "The tendency to be compassionate, cooperative, trusting, and helpful toward others",
        "Neuroticism": "The tendency to experience negative emotions such as anxiety, depression, and emotional instability",
    },
}

DISPLAY_NAME_MAP = {
    "Self_Direction": "Self-Direction",
    "Self_Enhancement": "Self-Enhancement",
    "Self_Transcendence": "Self-Transcendence",
    "Openness_to_Change": "Openness to Change",
}


def load_json(path: Path):
    return json.loads(path.read_text())


def display_name(name: str) -> str:
    return DISPLAY_NAME_MAP.get(name, name.replace("_", " "))


# ── Data Loading ──────────────────────────────────────────────────────


def load_established_items(survey_name: str) -> list[dict]:
    survey = load_json(SURVEYS_DIR / f"{survey_name}.json")
    construct_key = "value" if survey_name.startswith("PVQ") else "trait"
    items = []
    for item_id in sorted(survey.keys(), key=int):
        item_data = survey[item_id]
        text = item_data.get("en_neutral", "").strip()
        construct = item_data["meta_data"][construct_key]
        items.append({"id": f"{survey_name}_{item_id}", "text": text, "construct": construct})
    return items


def load_vp_items() -> list[dict]:
    scenarios = load_json(SURVEYS_DIR / "VP.json")
    items = []
    for scenario in scenarios:
        pid = scenario["portrait_id"]
        for output in scenario.get("outputs", []):
            oid = output["id"]
            text = (output.get("content") or "").strip()
            pvq_corrs = {display_name(n): float(v) for n, v in output.get("correlations", [])}
            bfi_corrs = {display_name(n): float(v) for n, v in output.get("bfi_correlations", [])}
            pvq_pos = [c for c, v in pvq_corrs.items() if v >= VP_POSITIVE_THRESHOLD]
            bfi_pos = [c for c, v in bfi_corrs.items() if v >= VP_POSITIVE_THRESHOLD]
            pvq_primary = max(pvq_corrs, key=pvq_corrs.get) if pvq_corrs else None
            bfi_primary = max(bfi_corrs, key=bfi_corrs.get) if bfi_corrs else None
            items.append({
                "id": f"VP_pid{pid}_out{oid}",
                "text": text,
                "pvq_correlations": pvq_corrs,
                "bfi_correlations": bfi_corrs,
                "pvq_positives": pvq_pos,
                "bfi_positives": bfi_pos,
                "pvq_primary": pvq_primary,
                "bfi_primary": bfi_primary,
            })
    return items


# ── Analysis A: Item-Construct Semantic Similarity ────────────────────


def construct_recognition(
    item_texts: list[str],
    item_constructs: list[str],
    construct_names: list[str],
    construct_target_texts: list[str],
    model: SentenceTransformer,
) -> dict:
    """
    Compute cosine similarity between each item and each construct.
    Returns per-item details and aggregate metrics.
    """
    item_embs = model.encode(item_texts, convert_to_tensor=True, show_progress_bar=False)
    const_embs = model.encode(construct_target_texts, convert_to_tensor=True, show_progress_bar=False)
    sim_matrix = cos_sim(item_embs, const_embs).cpu().numpy()

    per_item = []
    for i, correct in enumerate(item_constructs):
        sims = {c: float(sim_matrix[i, j]) for j, c in enumerate(construct_names)}
        correct_sim = sims.get(correct, 0.0)
        other_vals = [v for k, v in sims.items() if k != correct]
        mean_other = float(np.mean(other_vals)) if other_vals else 0.0
        ranked = sorted(sims, key=sims.get, reverse=True)
        rank = ranked.index(correct) + 1 if correct in ranked else len(ranked)
        per_item.append({
            "correct_construct": correct,
            "sim_to_correct": correct_sim,
            "mean_sim_to_others": mean_other,
            "discrimination": correct_sim - mean_other,
            "rank": rank,
        })

    discs = [p["discrimination"] for p in per_item]
    corr_sims = [p["sim_to_correct"] for p in per_item]
    ranks = [p["rank"] for p in per_item]
    return {
        "n_items": len(item_texts),
        "mean_sim_correct": float(np.mean(corr_sims)),
        "std_sim_correct": float(np.std(corr_sims)),
        "mean_sim_others": float(np.mean([p["mean_sim_to_others"] for p in per_item])),
        "mean_discrimination": float(np.mean(discs)),
        "std_discrimination": float(np.std(discs)),
        "top1_accuracy": float(np.mean([r == 1 for r in ranks])),
        "mean_rank": float(np.mean(ranks)),
        "per_item": per_item,
    }


def run_construct_recognition_established(
    survey_name: str, model: SentenceTransformer
) -> dict:
    items = load_established_items(survey_name)
    construct_space = "pvq_values" if survey_name.startswith("PVQ") else "bfi_traits"
    constructs = CONSTRUCT_DEFS[construct_space]
    c_names = list(constructs.keys())

    texts = [it["text"] for it in items]
    gt = [it["construct"] for it in items]

    name_result = construct_recognition(texts, gt, c_names, c_names, model)
    def_texts = [f"{n}: {constructs[n]}" for n in c_names]
    def_result = construct_recognition(texts, gt, c_names, def_texts, model)

    return {
        "survey": survey_name,
        "type": "established",
        "construct_space": construct_space,
        "by_name": _strip_per_item(name_result),
        "by_definition": _strip_per_item(def_result),
    }


def run_construct_recognition_vp(
    vp_items: list[dict], construct_space: str, model: SentenceTransformer
) -> dict:
    constructs = CONSTRUCT_DEFS[construct_space]
    c_names = list(constructs.keys())
    primary_key = "pvq_primary" if construct_space == "pvq_values" else "bfi_primary"

    filtered = [(it["text"], it[primary_key]) for it in vp_items if it[primary_key] in constructs]
    texts, gt = zip(*filtered) if filtered else ([], [])

    name_result = construct_recognition(list(texts), list(gt), c_names, c_names, model)
    def_texts = [f"{n}: {constructs[n]}" for n in c_names]
    def_result = construct_recognition(list(texts), list(gt), c_names, def_texts, model)

    label = "VP (PVQ values)" if construct_space == "pvq_values" else "VP (BFI traits)"
    return {
        "survey": label,
        "type": "ecological",
        "construct_space": construct_space,
        "by_name": _strip_per_item(name_result),
        "by_definition": _strip_per_item(def_result),
    }


def _strip_per_item(result: dict) -> dict:
    """Return aggregate-only copy (drop per_item for JSON compactness)."""
    return {k: v for k, v in result.items() if k != "per_item"}


# ── Analysis B: Lexical-Semantic Clustering ───────────────────────────


def lexical_semantic_clustering(
    item_texts: list[str],
    item_constructs: list[str],
    model: SentenceTransformer,
) -> dict:
    """Intra-construct vs inter-construct pairwise cosine similarity."""
    if len(item_texts) < 2:
        return {}

    embs = model.encode(item_texts, convert_to_tensor=True, show_progress_bar=False)
    sim_matrix = cos_sim(embs, embs).cpu().numpy()

    intra, inter = [], []
    for i, j in combinations(range(len(item_texts)), 2):
        s = float(sim_matrix[i, j])
        if item_constructs[i] == item_constructs[j]:
            intra.append(s)
        else:
            inter.append(s)

    groups = defaultdict(list)
    for idx, c in enumerate(item_constructs):
        groups[c].append(idx)

    per_construct = {}
    for c, indices in sorted(groups.items()):
        if len(indices) < 2:
            per_construct[c] = {"mean_intra": None, "n_items": len(indices)}
            continue
        pairs = [float(sim_matrix[i, j]) for i, j in combinations(indices, 2)]
        per_construct[c] = {
            "mean_intra": float(np.mean(pairs)),
            "std_intra": float(np.std(pairs)),
            "n_pairs": len(pairs),
            "n_items": len(indices),
        }

    gap = (float(np.mean(intra)) - float(np.mean(inter))) if intra and inter else None
    return {
        "n_items": len(item_texts),
        "n_constructs": len(groups),
        "n_intra_pairs": len(intra),
        "n_inter_pairs": len(inter),
        "mean_intra_sim": float(np.mean(intra)) if intra else None,
        "std_intra_sim": float(np.std(intra)) if intra else None,
        "mean_inter_sim": float(np.mean(inter)) if inter else None,
        "std_inter_sim": float(np.std(inter)) if inter else None,
        "clustering_gap": gap,
        "per_construct": per_construct,
    }


def run_lsc_established(survey_name: str, model: SentenceTransformer) -> dict:
    items = load_established_items(survey_name)
    texts = [it["text"] for it in items]
    gt = [it["construct"] for it in items]
    result = lexical_semantic_clustering(texts, gt, model)
    result["survey"] = survey_name
    result["type"] = "established"
    return result


def run_lsc_vp(
    vp_items: list[dict], construct_space: str, model: SentenceTransformer
) -> dict:
    constructs = CONSTRUCT_DEFS[construct_space]
    primary_key = "pvq_primary" if construct_space == "pvq_values" else "bfi_primary"

    filtered = [
        (it["text"], it[primary_key])
        for it in vp_items
        if it[primary_key] and it[primary_key] in constructs
    ]
    if len(filtered) < 2:
        return {"survey": f"VP ({construct_space})", "type": "ecological"}

    texts, gt = zip(*filtered)
    result = lexical_semantic_clustering(list(texts), list(gt), model)
    label = "VP (PVQ values)" if construct_space == "pvq_values" else "VP (BFI traits)"
    result["survey"] = label
    result["type"] = "ecological"
    return result


# ── Construct-level matrix computations ───────────────────────────────


def compute_construct_group_to_construct_sim(
    item_texts: list[str],
    item_constructs: list[str],
    construct_space: str,
    model: SentenceTransformer,
    use_definitions: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """
    Mean similarity from items-in-each-group to each construct target.
    Returns (n_constructs × n_constructs) matrix and ordered construct names.
    Row i = items belonging to construct i, Col j = similarity to construct j text.
    Diagonal bright → items recognisably match their own construct.
    """
    constructs = CONSTRUCT_DEFS[construct_space]
    c_names = list(constructs.keys())
    c_texts = (
        [f"{n}: {constructs[n]}" for n in c_names] if use_definitions else list(c_names)
    )

    groups: dict[str, list[str]] = defaultdict(list)
    for text, construct in zip(item_texts, item_constructs):
        if construct in c_names:
            groups[construct].append(text)

    all_texts: list[str] = []
    group_ranges: dict[str, tuple[int, int]] = {}
    for c in c_names:
        start = len(all_texts)
        all_texts.extend(groups.get(c, []))
        group_ranges[c] = (start, len(all_texts))

    if not all_texts:
        return np.zeros((len(c_names), len(c_names))), c_names

    item_embs = model.encode(all_texts, convert_to_tensor=True, show_progress_bar=False)
    const_embs = model.encode(c_texts, convert_to_tensor=True, show_progress_bar=False)
    full_sim = cos_sim(item_embs, const_embs).cpu().numpy()

    matrix = np.full((len(c_names), len(c_names)), np.nan)
    for i, c in enumerate(c_names):
        s, e = group_ranges[c]
        if s < e:
            matrix[i] = full_sim[s:e].mean(axis=0)
    return matrix, c_names


def compute_construct_pair_item_sim(
    item_texts: list[str],
    item_constructs: list[str],
    construct_space: str,
    model: SentenceTransformer,
) -> tuple[np.ndarray, list[str]]:
    """
    Mean pairwise item similarity between every pair of construct groups.
    Diagonal = mean intra-construct item similarity (excl. self-pairs).
    Off-diagonal = mean inter-construct item similarity.
    Diagonal brighter than off-diagonal → lexical-semantic clustering.
    """
    constructs = CONSTRUCT_DEFS[construct_space]
    c_names = list(constructs.keys())

    groups: dict[str, list[str]] = defaultdict(list)
    for text, construct in zip(item_texts, item_constructs):
        if construct in c_names:
            groups[construct].append(text)

    all_texts: list[str] = []
    group_ranges: dict[str, tuple[int, int]] = {}
    for c in c_names:
        start = len(all_texts)
        all_texts.extend(groups.get(c, []))
        group_ranges[c] = (start, len(all_texts))

    if not all_texts:
        return np.zeros((len(c_names), len(c_names))), c_names

    embs = model.encode(all_texts, convert_to_tensor=True, show_progress_bar=False)
    full_sim = cos_sim(embs, embs).cpu().numpy()

    matrix = np.full((len(c_names), len(c_names)), np.nan)
    for i, ci in enumerate(c_names):
        si, ei = group_ranges[ci]
        if si >= ei:
            continue
        for j, cj in enumerate(c_names):
            sj, ej = group_ranges[cj]
            if sj >= ej:
                continue
            block = full_sim[si:ei, sj:ej]
            if i == j:
                n = block.shape[0]
                if n > 1:
                    mask = ~np.eye(n, dtype=bool)
                    matrix[i, j] = block[mask].mean()
            else:
                matrix[i, j] = block.mean()
    return matrix, c_names


# ── Plotting ──────────────────────────────────────────────────────────

ESTABLISHED_COLOR = "#2166ac"
ECOLOGICAL_COLOR = "#b2182b"

_SHORT_CONSTRUCT = {
    "Self-Direction": "Self-Dir",
    "Universalism": "Univ",
    "Benevolence": "Benev",
    "Stimulation": "Stim",
    "Achievement": "Achiev",
    "Conformity": "Conf",
    "Tradition": "Trad",
    "Security": "Secur",
    "Hedonism": "Hedon",
    "Power": "Power",
    "Openness": "Open",
    "Conscientiousness": "Consc",
    "Extraversion": "Extra",
    "Agreeableness": "Agree",
    "Neuroticism": "Neuro",
}


def _short(name: str) -> str:
    return _SHORT_CONSTRUCT.get(name, name[:6])


def _heatmap_on_ax(
    ax,
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    vmin: float,
    vmax: float,
    cmap: str = "RdYlBu_r",
    annotate: bool = True,
    fontsize_annot: int = 9,
    fontsize_tick: int = 11,
):
    import matplotlib.colors as mcolors

    cm = plt.get_cmap(cmap)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(
        [_short(c) for c in col_labels],
        rotation=45, ha="right", fontsize=fontsize_tick,
    )
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(
        [_short(c) for c in row_labels],
        fontsize=fontsize_tick, va="center",
    )
    ax.set_title(title, fontsize=17, fontweight="bold", pad=12)
    if annotate:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                v = matrix[i, j]
                if np.isnan(v):
                    continue
                rgba = cm(norm(v))
                lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                color = "black" if lum > 0.5 else "white"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=fontsize_annot, color=color)
    return im


def plot_cr_heatmaps(
    matrices: list[np.ndarray],
    c_names_list: list[list[str]],
    titles: list[str],
    suptitle: str,
    filename: str,
):
    n = len(matrices)
    widths = [m.shape[1] for m in matrices]
    fig_w = max(5.0 * n, 12)
    fig, axes = plt.subplots(
        1, n, figsize=(fig_w, max(w * 0.7 for w in widths) + 2.0),
        gridspec_kw={"width_ratios": widths},
    )
    if n == 1:
        axes = [axes]

    all_vals = np.concatenate([m[~np.isnan(m)] for m in matrices])
    vmin, vmax = float(all_vals.min()), float(all_vals.max())

    im = None
    for ax, mat, c_names, title in zip(axes, matrices, c_names_list, titles):
        im = _heatmap_on_ax(ax, mat, c_names, c_names, title, vmin, vmax)
        ax.set_xlabel("Construct target", fontsize=11)
    axes[0].set_ylabel("Item group (ground truth)", fontsize=11)

    fig.colorbar(im, ax=axes, shrink=0.75, pad=0.03, label="Cosine Similarity")
    fig.subplots_adjust(bottom=0.18)
    fig.savefig(FIG_DIR / f"{filename}.pdf", bbox_inches="tight", dpi=150)
    fig.savefig(FIG_DIR / f"{filename}.png", bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_lsc_heatmaps(
    matrices: list[np.ndarray],
    c_names_list: list[list[str]],
    titles: list[str],
    filename: str,
):
    n = len(matrices)
    widths = [m.shape[1] for m in matrices]
    fig_w = max(5.0 * n, 12)
    fig, axes = plt.subplots(
        1, n, figsize=(fig_w, max(w * 0.7 for w in widths) + 2.0),
        gridspec_kw={"width_ratios": widths},
    )
    if n == 1:
        axes = [axes]

    all_vals = np.concatenate([m[~np.isnan(m)] for m in matrices])
    vmin, vmax = float(all_vals.min()), float(all_vals.max())

    im = None
    for ax, mat, c_names, title in zip(axes, matrices, c_names_list, titles):
        im = _heatmap_on_ax(ax, mat, c_names, c_names, title, vmin, vmax, cmap="YlOrRd")
        ax.set_xlabel("Construct group", fontsize=11)
    axes[0].set_ylabel("Construct group", fontsize=11)

    fig.colorbar(im, ax=axes, shrink=0.75, pad=0.03, label="Mean Pairwise Item Similarity")
    fig.subplots_adjust(bottom=0.18)
    fig.savefig(FIG_DIR / f"{filename}.pdf", bbox_inches="tight", dpi=150)
    fig.savefig(FIG_DIR / f"{filename}.png", bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_paper_figure(
    cr_matrices_pvq: list[np.ndarray],
    cr_names_pvq: list[list[str]],
    cr_matrices_bfi: list[np.ndarray],
    cr_names_bfi: list[list[str]],
    lsc_matrices_pvq: list[np.ndarray],
    lsc_names_pvq: list[list[str]],
    lsc_matrices_bfi: list[np.ndarray],
    lsc_names_bfi: list[list[str]],
):
    """1×4 paper figure: CR-Est, CR-VP, LSC-Est, LSC-VP — separate color scales."""
    from matplotlib.gridspec import GridSpec

    CMAP = "RdYlBu_r"

    def _make_figure(cr_mats, cr_nms, lsc_mats, lsc_nms, est_label, fname,
                     figw=26, figh=7):
        cr_vals = np.concatenate([m[~np.isnan(m)] for m in cr_mats])
        lsc_vals = np.concatenate([m[~np.isnan(m)] for m in lsc_mats])
        cr_vmin, cr_vmax = float(cr_vals.min()), float(cr_vals.max())
        lsc_vmin, lsc_vmax = float(lsc_vals.min()), float(lsc_vals.max())

        fig = plt.figure(figsize=(figw, figh))
        outer = GridSpec(1, 2, figure=fig, wspace=0.15)

        gs_cr = outer[0, 0].subgridspec(1, 3, width_ratios=[1, 1, 0.035], wspace=0.10)
        gs_lsc = outer[0, 1].subgridspec(1, 3, width_ratios=[1, 1, 0.035], wspace=0.10)

        ax_cr0 = fig.add_subplot(gs_cr[0, 0])
        ax_cr1 = fig.add_subplot(gs_cr[0, 1])
        cax_cr = fig.add_subplot(gs_cr[0, 2])
        ax_lsc0 = fig.add_subplot(gs_lsc[0, 0])
        ax_lsc1 = fig.add_subplot(gs_lsc[0, 1])
        cax_lsc = fig.add_subplot(gs_lsc[0, 2])

        _heatmap_on_ax(ax_cr0, cr_mats[0], cr_nms[0], cr_nms[0],
                       f"(a) Item-Def. Similarity: {est_label}", cr_vmin, cr_vmax, cmap=CMAP)
        im_cr = _heatmap_on_ax(ax_cr1, cr_mats[1], cr_nms[1], cr_nms[1],
                               "(b) Item-Def. Similarity: VP", cr_vmin, cr_vmax, cmap=CMAP)
        _heatmap_on_ax(ax_lsc0, lsc_mats[0], lsc_nms[0], lsc_nms[0],
                       f"(c) Within-Construct Item Sim.: {est_label}", lsc_vmin, lsc_vmax, cmap=CMAP)
        im_lsc = _heatmap_on_ax(ax_lsc1, lsc_mats[1], lsc_nms[1], lsc_nms[1],
                                "(d) Within-Construct Item Sim.: VP", lsc_vmin, lsc_vmax, cmap=CMAP)

        ax_cr0.set_ylabel("Item group / Construct group", fontsize=12)
        ax_cr1.set_yticklabels([])
        ax_lsc1.set_yticklabels([])

        cb1 = fig.colorbar(im_cr, cax=cax_cr)
        cb1.set_label("Cosine Sim.", fontsize=10)
        cb2 = fig.colorbar(im_lsc, cax=cax_lsc)
        cb2.set_label("Cosine Sim.", fontsize=10)

        fig.savefig(FIG_DIR / f"{fname}.pdf", bbox_inches="tight", dpi=150)
        fig.savefig(FIG_DIR / f"{fname}.png", bbox_inches="tight", dpi=150)
        plt.close(fig)

    _make_figure(cr_matrices_pvq, cr_names_pvq, lsc_matrices_pvq, lsc_names_pvq,
                 "PVQ-40", "paper_figure_pvq", figw=45, figh=10)
    _make_figure(cr_matrices_bfi, cr_names_bfi, lsc_matrices_bfi, lsc_names_bfi,
                 "BFI-44", "paper_figure_bfi", figw=30, figh=8)


def plot_summary_bars(cr_results: list[dict], lsc_results: list[dict]):
    """Clean summary bar chart with established vs ecological comparison."""
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # CR discrimination (by definition)
    labels = [r["survey"] for r in cr_results]
    discs = [r["by_definition"]["mean_discrimination"] for r in cr_results]
    types = [r["type"] for r in cr_results]
    colors = [ESTABLISHED_COLOR if t == "established" else ECOLOGICAL_COLOR for t in types]

    ax = axes[0]
    bars = ax.barh(range(len(labels)), discs, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Discrimination (Δ cosine similarity)", fontsize=9)
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.invert_yaxis()
    for bar, val in zip(bars, discs):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8)

    # LSC gap
    lsc_valid = [r for r in lsc_results if r.get("clustering_gap") is not None]
    labels_l = [r["survey"] for r in lsc_valid]
    gaps = [r["clustering_gap"] for r in lsc_valid]
    types_l = [r["type"] for r in lsc_valid]
    colors_l = [ESTABLISHED_COLOR if t == "established" else ECOLOGICAL_COLOR for t in types_l]

    ax = axes[1]
    bars = ax.barh(range(len(labels_l)), gaps, color=colors_l, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(labels_l)))
    ax.set_yticklabels(labels_l, fontsize=9)
    ax.set_xlabel("Clustering Gap (intra − inter)", fontsize=9)
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.invert_yaxis()
    for bar, val in zip(bars, gaps):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{val:+.4f}", va="center", fontsize=8)

    legend_elements = [
        Patch(facecolor=ESTABLISHED_COLOR, label="Established"),
        Patch(facecolor=ECOLOGICAL_COLOR, label="Generation (VP)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2, fontsize=10, frameon=False)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(FIG_DIR / "summary_bars.pdf", bbox_inches="tight", dpi=150)
    fig.savefig(FIG_DIR / "summary_bars.png", bbox_inches="tight", dpi=150)
    plt.close(fig)


# ── Summary CSV ───────────────────────────────────────────────────────


def write_summary_csv(cr_results: list[dict], lsc_results: list[dict]):
    import csv

    lsc_map = {r["survey"]: r for r in lsc_results}
    rows = []
    for cr in cr_results:
        lsc = lsc_map.get(cr["survey"], {})
        rows.append({
            "survey": cr["survey"],
            "type": cr["type"],
            "n_items": cr["by_name"]["n_items"],
            "cr_name_discrimination": f"{cr['by_name']['mean_discrimination']:.4f}",
            "cr_name_top1_acc": f"{cr['by_name']['top1_accuracy']:.4f}",
            "cr_name_mean_rank": f"{cr['by_name']['mean_rank']:.2f}",
            "cr_def_discrimination": f"{cr['by_definition']['mean_discrimination']:.4f}",
            "cr_def_top1_acc": f"{cr['by_definition']['top1_accuracy']:.4f}",
            "cr_def_mean_rank": f"{cr['by_definition']['mean_rank']:.2f}",
            "lsc_intra": f"{lsc['mean_intra_sim']:.4f}" if lsc.get("mean_intra_sim") else "N/A",
            "lsc_inter": f"{lsc['mean_inter_sim']:.4f}" if lsc.get("mean_inter_sim") else "N/A",
            "lsc_gap": f"{lsc['clustering_gap']:.4f}" if lsc.get("clustering_gap") else "N/A",
        })

    path = OUT_DIR / "summary.csv"
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ── Main ──────────────────────────────────────────────────────────────


def _prepare_items(survey_name: str) -> tuple[list[str], list[str]]:
    items = load_established_items(survey_name)
    return [it["text"] for it in items], [it["construct"] for it in items]


def _prepare_vp_items(
    vp_items: list[dict], construct_space: str
) -> tuple[list[str], list[str]]:
    primary_key = "pvq_primary" if construct_space == "pvq_values" else "bfi_primary"
    constructs = CONSTRUCT_DEFS[construct_space]
    pairs = [
        (it["text"], it[primary_key])
        for it in vp_items
        if it[primary_key] and it[primary_key] in constructs
    ]
    if not pairs:
        return [], []
    texts, gt = zip(*pairs)
    return list(texts), list(gt)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("Loading VP items...")
    vp_items = load_vp_items()
    print(f"  {len(vp_items)} VP outputs loaded")

    # ── Analysis A: Construct Recognition ─────────────────────────────
    print("\n=== Analysis A: Construct Recognition ===")
    cr_results = []

    for survey in ["PVQ", "PVQ21"]:
        print(f"  {survey}...")
        cr_results.append(run_construct_recognition_established(survey, model))

    for survey in ["BFI44", "BFI10"]:
        print(f"  {survey}...")
        cr_results.append(run_construct_recognition_established(survey, model))

    for space in ["pvq_values", "bfi_traits"]:
        label = "VP (PVQ values)" if space == "pvq_values" else "VP (BFI traits)"
        print(f"  {label}...")
        cr_results.append(run_construct_recognition_vp(vp_items, space, model))

    for r in cr_results:
        bn = r["by_name"]
        bd = r["by_definition"]
        print(
            f"  {r['survey']:20s}  "
            f"disc(name)={bn['mean_discrimination']:+.4f}  "
            f"top1(name)={bn['top1_accuracy']:.2f}  "
            f"disc(def)={bd['mean_discrimination']:+.4f}  "
            f"top1(def)={bd['top1_accuracy']:.2f}"
        )

    # ── Analysis B: Lexical-Semantic Clustering ───────────────────────
    print("\n=== Analysis B: Lexical-Semantic Clustering ===")
    lsc_results = []

    for survey in ["PVQ", "PVQ21", "BFI44", "BFI10"]:
        print(f"  {survey}...")
        lsc_results.append(run_lsc_established(survey, model))

    for space in ["pvq_values", "bfi_traits"]:
        label = "VP (PVQ values)" if space == "pvq_values" else "VP (BFI traits)"
        print(f"  {label}...")
        lsc_results.append(run_lsc_vp(vp_items, space, model))

    for r in lsc_results:
        if r.get("clustering_gap") is not None:
            print(
                f"  {r['survey']:20s}  "
                f"intra={r['mean_intra_sim']:.4f}  "
                f"inter={r['mean_inter_sim']:.4f}  "
                f"gap={r['clustering_gap']:+.4f}"
            )

    # ── Analysis C: Construct-level heatmap matrices ──────────────────
    print("\n=== Analysis C: Construct-level Heatmap Matrices ===")

    pvq_texts, pvq_gt = _prepare_items("PVQ")
    bfi_texts, bfi_gt = _prepare_items("BFI44")
    vp_pvq_texts, vp_pvq_gt = _prepare_vp_items(vp_items, "pvq_values")
    vp_bfi_texts, vp_bfi_gt = _prepare_vp_items(vp_items, "bfi_traits")

    print("  CR heatmaps (by definition)...")
    cr_mat_pvq, cr_n_pvq = compute_construct_group_to_construct_sim(
        pvq_texts, pvq_gt, "pvq_values", model, use_definitions=True)
    cr_mat_vp_pvq, cr_n_vp_pvq = compute_construct_group_to_construct_sim(
        vp_pvq_texts, vp_pvq_gt, "pvq_values", model, use_definitions=True)
    cr_mat_bfi, cr_n_bfi = compute_construct_group_to_construct_sim(
        bfi_texts, bfi_gt, "bfi_traits", model, use_definitions=True)
    cr_mat_vp_bfi, cr_n_vp_bfi = compute_construct_group_to_construct_sim(
        vp_bfi_texts, vp_bfi_gt, "bfi_traits", model, use_definitions=True)

    print("  LSC heatmaps...")
    lsc_mat_pvq, lsc_n_pvq = compute_construct_pair_item_sim(
        pvq_texts, pvq_gt, "pvq_values", model)
    lsc_mat_vp_pvq, lsc_n_vp_pvq = compute_construct_pair_item_sim(
        vp_pvq_texts, vp_pvq_gt, "pvq_values", model)
    lsc_mat_bfi, lsc_n_bfi = compute_construct_pair_item_sim(
        bfi_texts, bfi_gt, "bfi_traits", model)
    lsc_mat_vp_bfi, lsc_n_vp_bfi = compute_construct_pair_item_sim(
        vp_bfi_texts, vp_bfi_gt, "bfi_traits", model)

    # ── Save ──────────────────────────────────────────────────────────
    print("\nSaving results...")
    summary = {
        "embedding_model": EMBEDDING_MODEL_NAME,
        "vp_threshold": VP_POSITIVE_THRESHOLD,
        "construct_recognition": cr_results,
        "lexical_semantic_clustering": lsc_results,
    }
    (OUT_DIR / "results.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    write_summary_csv(cr_results, lsc_results)

    # ── Figures ───────────────────────────────────────────────────────
    print("Generating figures...")

    plot_cr_heatmaps(
        [cr_mat_pvq, cr_mat_vp_pvq],
        [cr_n_pvq, cr_n_vp_pvq],
        ["PVQ-40 (Established)", "VP (Generation)"],
        "",
        "cr_heatmap_pvq",
    )
    plot_cr_heatmaps(
        [cr_mat_bfi, cr_mat_vp_bfi],
        [cr_n_bfi, cr_n_vp_bfi],
        ["BFI-44 (Established)", "VP (Generation)"],
        "",
        "cr_heatmap_bfi",
    )
    plot_lsc_heatmaps(
        [lsc_mat_pvq, lsc_mat_vp_pvq],
        [lsc_n_pvq, lsc_n_vp_pvq],
        ["PVQ-40 (Established)", "VP (Generation)"],
        "lsc_heatmap_pvq",
    )
    plot_lsc_heatmaps(
        [lsc_mat_bfi, lsc_mat_vp_bfi],
        [lsc_n_bfi, lsc_n_vp_bfi],
        ["BFI-44 (Established)", "VP (Generation)"],
        "lsc_heatmap_bfi",
    )
    plot_paper_figure(
        [cr_mat_pvq, cr_mat_vp_pvq], [cr_n_pvq, cr_n_vp_pvq],
        [cr_mat_bfi, cr_mat_vp_bfi], [cr_n_bfi, cr_n_vp_bfi],
        [lsc_mat_pvq, lsc_mat_vp_pvq], [lsc_n_pvq, lsc_n_vp_pvq],
        [lsc_mat_bfi, lsc_mat_vp_bfi], [lsc_n_bfi, lsc_n_vp_bfi],
    )
    plot_summary_bars(cr_results, lsc_results)

    print(f"\nResults saved to {OUT_DIR.relative_to(ROOT)}/")
    print("Done.")


if __name__ == "__main__":
    main()
