#!/usr/bin/env python3
"""
Embedding-based cue-leak check (CPU).

Adapted from RQ3/embedding_analysis.py. For PVQ-40 and BFI-44:

  (i)  Item -> construct-definition (and -name) discrimination:
       top-1 accuracy + mean discrimination for ORIGINAL vs REWRITTEN items.
       A successful rewrite shows a large drop for rewritten items (VP reference: ~.11-.26).
  (ii) Original <-> rewritten pairwise cosine similarity per item
       (semantic preservation check): distribution should stay high.

Also emits a per-item leak table (rewritten items still ranked #1 for their own
construct) so the worst offenders can be inspected.

Usage (from the repository root):
    python RQ3/cue_reduction/scripts/check_cueless_embedding.py

Outputs:
    RQ3/cue_reduction/results/cueless_embedding_check.json
    RQ3/cue_reduction/results/cueless_embedding_per_item.csv
"""

import csv
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

CUE_DIR = Path(__file__).resolve().parent.parent
ROOT = CUE_DIR.parent.parent  # Official/ repo root
CUELESS_DIR = CUE_DIR / "surveys_cueless"
OUT_DIR = CUE_DIR / "results"

EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"

CONSTRUCT_DEFS = {
    "PVQ": {
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
    "BFI44": {
        "Openness": "The tendency to be imaginative, curious, and receptive to new ideas, experiences, and perspectives",
        "Conscientiousness": "The tendency to be organized, responsible, hardworking, and goal-directed",
        "Extraversion": "The tendency to be sociable, assertive, active, and to experience positive emotions",
        "Agreeableness": "The tendency to be compassionate, cooperative, trusting, and helpful toward others",
        "Neuroticism": "The tendency to experience negative emotions such as anxiety, depression, and emotional instability",
    },
}

CONSTRUCT_KEY = {"PVQ": "value", "BFI44": "trait"}


def load_items(path: Path, construct_key: str) -> list[dict]:
    survey = json.loads(path.read_text())
    items = []
    for item_id in sorted(survey, key=int):
        d = survey[item_id]
        items.append({
            "id": item_id,
            "text": d["en_neutral"].strip(),
            "construct": d["meta_data"][construct_key],
        })
    return items


def recognition(texts, constructs_gt, c_names, c_target_texts, model):
    item_embs = model.encode(texts, convert_to_tensor=True, show_progress_bar=False)
    const_embs = model.encode(c_target_texts, convert_to_tensor=True, show_progress_bar=False)
    sim = cos_sim(item_embs, const_embs).cpu().numpy()

    per_item = []
    for i, correct in enumerate(constructs_gt):
        sims = {c: float(sim[i, j]) for j, c in enumerate(c_names)}
        correct_sim = sims.get(correct, 0.0)
        others = [v for k, v in sims.items() if k != correct]
        ranked = sorted(sims, key=sims.get, reverse=True)
        per_item.append({
            "correct": correct,
            "sim_correct": correct_sim,
            "mean_sim_others": float(np.mean(others)),
            "discrimination": correct_sim - float(np.mean(others)),
            "rank": ranked.index(correct) + 1,
            "top_construct": ranked[0],
        })
    return {
        "n_items": len(texts),
        "top1_accuracy": float(np.mean([p["rank"] == 1 for p in per_item])),
        "mean_rank": float(np.mean([p["rank"] for p in per_item])),
        "mean_discrimination": float(np.mean([p["discrimination"] for p in per_item])),
        "mean_sim_correct": float(np.mean([p["sim_correct"] for p in per_item])),
        "per_item": per_item,
    }


def summarize(result: dict) -> dict:
    return {k: v for k, v in result.items() if k != "per_item"}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    report = {"embedding_model": EMBEDDING_MODEL_NAME, "surveys": {}}
    csv_rows = []

    for survey in ["PVQ", "BFI44"]:
        ckey = CONSTRUCT_KEY[survey]
        orig = load_items(ROOT / "surveys" / f"{survey}.json", ckey)
        cueless = load_items(CUELESS_DIR / f"{survey}.json", ckey)
        assert [o["id"] for o in orig] == [c["id"] for c in cueless]
        assert [o["construct"] for o in orig] == [c["construct"] for c in cueless]

        constructs = CONSTRUCT_DEFS[survey]
        c_names = list(constructs)
        def_texts = [f"{n}: {constructs[n]}" for n in c_names]

        res = {}
        for label, items in [("original", orig), ("rewritten", cueless)]:
            texts = [it["text"] for it in items]
            gt = [it["construct"] for it in items]
            res[label] = {
                "by_name": recognition(texts, gt, c_names, c_names, model),
                "by_definition": recognition(texts, gt, c_names, def_texts, model),
            }

        # (ii) semantic preservation: original <-> rewritten cosine per item
        o_embs = model.encode([it["text"] for it in orig], convert_to_tensor=True, show_progress_bar=False)
        c_embs = model.encode([it["text"] for it in cueless], convert_to_tensor=True, show_progress_bar=False)
        pair_sims = cos_sim(o_embs, c_embs).cpu().numpy().diagonal()

        # per-item table
        for i, (o, c) in enumerate(zip(orig, cueless)):
            r_def = res["rewritten"]["by_definition"]["per_item"][i]
            o_def = res["original"]["by_definition"]["per_item"][i]
            csv_rows.append({
                "survey": survey,
                "id": o["id"],
                "construct": o["construct"],
                "orig_rank": o_def["rank"],
                "rewr_rank": r_def["rank"],
                "rewr_top_construct": r_def["top_construct"],
                "rewr_discrimination": round(r_def["discrimination"], 4),
                "pair_cosine": round(float(pair_sims[i]), 4),
                "still_leaks": r_def["rank"] == 1,
                "rewritten": c["text"],
            })

        report["surveys"][survey] = {
            "original": {k: summarize(v) for k, v in res["original"].items()},
            "rewritten": {k: summarize(v) for k, v in res["rewritten"].items()},
            "semantic_preservation": {
                "mean_pair_cosine": float(np.mean(pair_sims)),
                "std_pair_cosine": float(np.std(pair_sims)),
                "min_pair_cosine": float(np.min(pair_sims)),
                "p10_pair_cosine": float(np.percentile(pair_sims, 10)),
                "median_pair_cosine": float(np.median(pair_sims)),
                "n_below_0.4": int(np.sum(pair_sims < 0.4)),
            },
        }

        for label in ["original", "rewritten"]:
            bd = res[label]["by_definition"]
            bn = res[label]["by_name"]
            print(
                f"{survey:6s} {label:9s}  top1(def)={bd['top1_accuracy']:.3f} "
                f"disc(def)={bd['mean_discrimination']:+.4f}  "
                f"top1(name)={bn['top1_accuracy']:.3f} "
                f"disc(name)={bn['mean_discrimination']:+.4f}"
            )
        sp = report["surveys"][survey]["semantic_preservation"]
        print(
            f"{survey:6s} pair-cos   mean={sp['mean_pair_cosine']:.3f} "
            f"median={sp['median_pair_cosine']:.3f} min={sp['min_pair_cosine']:.3f}"
        )

    (OUT_DIR / "cueless_embedding_check.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False)
    )
    with (OUT_DIR / "cueless_embedding_per_item.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)

    leaks = [r for r in csv_rows if r["still_leaks"]]
    print(f"\nRewritten items still ranked #1 for own construct (by definition): {len(leaks)}")
    for r in leaks:
        print(f"  {r['survey']} #{r['id']} ({r['construct']}, disc={r['rewr_discrimination']}): {r['rewritten'][:90]}")


if __name__ == "__main__":
    main()
