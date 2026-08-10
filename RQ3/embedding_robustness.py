#!/usr/bin/env python3
"""
RQ3 Robustness Check: Run CR and LSC with multiple sentence encoders.

Compares results across encoders to verify that the established vs ecological
gap is model-agnostic (intrinsic textual property, not encoder artifact).

Usage:
    python RQ3/embedding_robustness.py

Outputs:
    results/RQ3/embedding_robustness.json
    results/RQ3/table_embedding_robustness.tex
    results/RQ3/embedding_robustness_cr.pdf
    results/RQ3/embedding_robustness_lsc.pdf
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results" / "RQ3"

ENCODER_MODELS = [
    "all-mpnet-base-v2",
    "all-MiniLM-L12-v2",
    "all-MiniLM-L6-v2",
    "BAAI/bge-base-en-v1.5",
    "intfloat/e5-base-v2",
]

ENCODER_SHORT = {
    "all-mpnet-base-v2": "MPNet-base",
    "all-MiniLM-L12-v2": "MiniLM-L12",
    "all-MiniLM-L6-v2": "MiniLM-L6",
    "BAAI/bge-base-en-v1.5": "BGE-base",
    "intfloat/e5-base-v2": "E5-base",
}


def _load_json(p):
    return json.loads(Path(p).read_text())


def _tex_escape(s):
    return s.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def run_analysis_for_model(model_name: str) -> dict:
    """Run CR and LSC for one encoder, reusing logic from embedding_analysis.py."""
    from sentence_transformers import SentenceTransformer

    import sys
    sys.path.insert(0, str(ROOT / "RQ3"))
    from RQ3.embedding_analysis import (
        CONSTRUCT_DEFS,
        VP_POSITIVE_THRESHOLD,
        load_established_items,
        load_vp_items,
        construct_recognition,
        lexical_semantic_clustering,
    )

    print(f"  Loading encoder: {model_name}")

    prefix = ""
    if "e5" in model_name.lower():
        prefix = "query: "

    model = SentenceTransformer(model_name)
    vp_items = load_vp_items()

    results = {"encoder": model_name, "cr": [], "lsc": []}

    # --- CR for established surveys ---
    for survey_name in ["PVQ", "PVQ21", "BFI44", "BFI10"]:
        items = load_established_items(survey_name)
        construct_space = "pvq_values" if survey_name.startswith("PVQ") else "bfi_traits"
        constructs = CONSTRUCT_DEFS[construct_space]
        c_names = list(constructs.keys())

        texts = [prefix + it["text"] for it in items]
        gt = [it["construct"] for it in items]

        def_texts = [f"{n}: {constructs[n]}" for n in c_names]
        if prefix:
            def_texts = [prefix + t for t in def_texts]

        def_result = construct_recognition(texts, gt, c_names, def_texts, model)

        results["cr"].append({
            "survey": survey_name,
            "type": "established",
            "discrimination": def_result["mean_discrimination"],
            "top1_accuracy": def_result["top1_accuracy"],
            "mean_rank": def_result["mean_rank"],
        })

    # --- CR for VP ---
    for construct_space, label in [("pvq_values", "VP (PVQ)"), ("bfi_traits", "VP (BFI)")]:
        constructs = CONSTRUCT_DEFS[construct_space]
        c_names = list(constructs.keys())
        primary_key = "pvq_primary" if construct_space == "pvq_values" else "bfi_primary"

        filtered = [(it["text"], it[primary_key]) for it in vp_items
                     if it[primary_key] in constructs]
        if not filtered:
            continue
        texts, gt = zip(*filtered)
        texts = [prefix + t for t in texts]

        def_texts = [f"{n}: {constructs[n]}" for n in c_names]
        if prefix:
            def_texts = [prefix + t for t in def_texts]

        def_result = construct_recognition(list(texts), list(gt), c_names, def_texts, model)

        results["cr"].append({
            "survey": label,
            "type": "ecological",
            "discrimination": def_result["mean_discrimination"],
            "top1_accuracy": def_result["top1_accuracy"],
            "mean_rank": def_result["mean_rank"],
        })

    # --- LSC for established surveys ---
    for survey_name in ["PVQ", "PVQ21", "BFI44", "BFI10"]:
        items = load_established_items(survey_name)
        texts = [prefix + it["text"] for it in items]
        gt = [it["construct"] for it in items]

        lsc = lexical_semantic_clustering(texts, gt, model)
        results["lsc"].append({
            "survey": survey_name,
            "type": "established",
            "mean_intra_sim": lsc.get("mean_intra_sim"),
            "mean_inter_sim": lsc.get("mean_inter_sim"),
            "clustering_gap": lsc.get("clustering_gap"),
        })

    # --- LSC for VP ---
    for construct_space, label in [("pvq_values", "VP (PVQ)"), ("bfi_traits", "VP (BFI)")]:
        constructs = CONSTRUCT_DEFS[construct_space]
        primary_key = "pvq_primary" if construct_space == "pvq_values" else "bfi_primary"

        filtered = [(it["text"], it[primary_key]) for it in vp_items
                     if it[primary_key] and it[primary_key] in constructs]
        if len(filtered) < 2:
            continue
        texts, gt = zip(*filtered)
        texts = [prefix + t for t in texts]

        lsc = lexical_semantic_clustering(list(texts), list(gt), model)
        results["lsc"].append({
            "survey": label,
            "type": "ecological",
            "mean_intra_sim": lsc.get("mean_intra_sim"),
            "mean_inter_sim": lsc.get("mean_inter_sim"),
            "clustering_gap": lsc.get("clustering_gap"),
        })

    del model
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    return results


def generate_comparison_table(all_results: list[dict]):
    """LaTeX table comparing CR discrimination and LSC gap across encoders."""
    surveys_cr = ["PVQ", "BFI44", "VP (PVQ)", "VP (BFI)"]
    surveys_lsc = ["PVQ", "BFI44", "VP (PVQ)", "VP (BFI)"]

    lines = []
    lines.append(r"% Auto-generated. Embedding robustness check.")
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Robustness check: CR discrimination (by definition) and LSC clustering gap across five sentence encoders. Established questionnaires (PVQ-40, BFI-44) consistently show higher discrimination and clustering gap than VP generation outputs, regardless of encoder choice.}")
    lines.append(r"\label{tab:app_embedding_robustness}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{l" + "rrrr" + "|" + "rrrr" + r"}")
    lines.append(r"\toprule")
    lines.append(r" & \multicolumn{4}{c}{CR Discrimination (by def.)} & \multicolumn{4}{c}{LSC Clustering Gap} \\")
    lines.append(r"\cmidrule(lr){2-5} \cmidrule(lr){6-9}")
    lines.append(r"Encoder & PVQ-40 & BFI-44 & VP (PVQ) & VP (BFI) & PVQ-40 & BFI-44 & VP (PVQ) & VP (BFI) \\")
    lines.append(r"\midrule")

    for res in all_results:
        enc = ENCODER_SHORT.get(res["encoder"], res["encoder"])
        cr_map = {r["survey"]: r for r in res["cr"]}
        lsc_map = {r["survey"]: r for r in res["lsc"]}

        cr_vals = []
        for s in surveys_cr:
            d = cr_map.get(s, {}).get("discrimination")
            cr_vals.append(f"{d:.3f}" if d is not None else "—")

        lsc_vals = []
        for s in surveys_lsc:
            g = lsc_map.get(s, {}).get("clustering_gap")
            lsc_vals.append(f"{g:+.4f}" if g is not None else "—")

        row = _tex_escape(enc) + " & " + " & ".join(cr_vals) + " & " + " & ".join(lsc_vals) + r" \\"
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def generate_figures(all_results: list[dict]):
    """Bar chart comparing CR discrimination and LSC gap across encoders."""
    surveys_cr = ["PVQ", "BFI44", "VP (PVQ)", "VP (BFI)"]
    surveys_lsc = ["PVQ", "BFI44", "VP (PVQ)", "VP (BFI)"]
    survey_labels_cr = ["PVQ-40", "BFI-44", "VP (PVQ)", "VP (BFI)"]
    survey_labels_lsc = ["PVQ-40", "BFI-44", "VP (PVQ)", "VP (BFI)"]

    enc_names = [ENCODER_SHORT.get(r["encoder"], r["encoder"]) for r in all_results]
    colors = ["#4C72B0", "#55A868", "#DD8452", "#C44E52", "#8172B3"]

    # --- CR Discrimination figure ---
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(surveys_cr))
    width = 0.15
    offsets = np.arange(len(all_results)) - (len(all_results) - 1) / 2

    for ei, res in enumerate(all_results):
        cr_map = {r["survey"]: r for r in res["cr"]}
        vals = [cr_map.get(s, {}).get("discrimination", 0) for s in surveys_cr]
        ax.bar(x + offsets[ei] * width, vals, width,
               label=enc_names[ei], color=colors[ei % len(colors)],
               alpha=0.85, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(survey_labels_cr, fontsize=10)
    ax.set_ylabel("Item–Definition Discrimination", fontsize=10)
    ax.axhline(0, color="gray", lw=0.5)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.15)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "embedding_robustness_cr.pdf", bbox_inches="tight")
    plt.close()

    # --- LSC Gap figure ---
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(surveys_lsc))

    for ei, res in enumerate(all_results):
        lsc_map = {r["survey"]: r for r in res["lsc"]}
        vals = [lsc_map.get(s, {}).get("clustering_gap", 0) or 0 for s in surveys_lsc]
        ax.bar(x + offsets[ei] * width, vals, width,
               label=enc_names[ei], color=colors[ei % len(colors)],
               alpha=0.85, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(survey_labels_lsc, fontsize=10)
    ax.set_ylabel("Within-Construct Clustering Gap", fontsize=10)
    ax.axhline(0, color="gray", lw=0.5)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.15)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "embedding_robustness_lsc.pdf", bbox_inches="tight")
    plt.close()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  RQ3 Embedding Robustness Check")
    print("=" * 60)

    all_results = []
    for enc in ENCODER_MODELS:
        print(f"\n--- Encoder: {enc} ---")
        res = run_analysis_for_model(enc)
        all_results.append(res)

        cr_map = {r["survey"]: r for r in res["cr"]}
        lsc_map = {r["survey"]: r for r in res["lsc"]}
        for s in ["PVQ", "BFI44", "VP (PVQ)", "VP (BFI)"]:
            cr_d = cr_map.get(s, {}).get("discrimination", 0)
            lsc_g = lsc_map.get(s, {}).get("clustering_gap", 0)
            print(f"    {s:12s}  CR disc={cr_d:+.4f}  LSC gap={lsc_g:+.4f}" if lsc_g else
                  f"    {s:12s}  CR disc={cr_d:+.4f}  LSC gap=N/A")

    # Save raw results
    (OUT_DIR / "embedding_robustness.json").write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False)
    )
    print(f"\n  Saved: embedding_robustness.json")

    # Generate table
    table_tex = generate_comparison_table(all_results)
    (OUT_DIR / "table_embedding_robustness.tex").write_text(table_tex)
    print(f"  Saved: table_embedding_robustness.tex")

    # Generate figures
    generate_figures(all_results)
    print(f"  Saved: embedding_robustness_cr.pdf")
    print(f"  Saved: embedding_robustness_lsc.pdf")

    # Update master tex

    print("\n" + "=" * 60)
    print("  Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
