#!/usr/bin/env python3
"""
Post-processing: recognition F1 on cue-reduced items, and comparison
against the original established F1 (from RQ3 summary).

Adapted from RQ3/summarize_item_construct_f1.py.

Produces:
  results/analysis/recognition_cueless_{survey}_model_summary.csv
  results/analysis/recognition_f1_comparison.csv   (orig vs cueless vs VP)
  results/analysis/recognition_headline.json

Run (from the repository root, no GPU):
    python RQ3/cue_reduction/scripts/summarize_recognition_cueless.py
"""

import csv
import json
from pathlib import Path

CUE_DIR = Path(__file__).resolve().parent.parent
ROOT = CUE_DIR.parent.parent  # Official/ repo root
CUELESS_REC = CUE_DIR / "results" / "recognition_cueless"
OUT_DIR = CUE_DIR / "results" / "analysis"
ORIG_SUMMARY = ROOT / "results" / "RQ3" / "item_construct_recognition_summary" / "summary.json"

SURVEYS = ["PVQ", "BFI44"]


def load_json(p):
    return json.loads(p.read_text())


def f1_counts(tp, fp, fn):
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0


def metrics_for_file(path):
    payload = load_json(path)
    constructs = list(payload["construct_definitions_used"].keys())
    per_c = {}
    for c in constructs:
        tp = fp = fn = 0
        for item in payload["results"].values():
            pred = bool(item["predictions"][c]["parsed"])
            gold = item.get("ground_truth_construct") == c
            if pred and gold:
                tp += 1
            elif pred and not gold:
                fp += 1
            elif (not pred) and gold:
                fn += 1
        per_c[c] = f1_counts(tp, fp, fn)
    mean_f1 = sum(per_c.values()) / len(per_c)
    return payload["model"].split("/")[-1], per_c, mean_f1


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="Overwrite the shipped comparison files even when the "
                         "original-item F1 summary does not cover every model.")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    orig = load_json(ORIG_SUMMARY) if ORIG_SUMMARY.exists() else {}

    headline = {"per_survey": {}}
    comp_rows = []
    orig_missing = []

    for survey in SURVEYS:
        sdir = CUELESS_REC / survey
        if not sdir.exists():
            print(f"  [skip] {survey}: no cueless recognition results ({sdir})")
            continue
        paths = sorted(sdir.glob("*.json"))
        if not paths:
            print(f"  [skip] {survey}: no files")
            continue

        rows = []
        cue_model_f1 = {}
        for p in paths:
            slug, per_c, mean_f1 = metrics_for_file(p)
            cue_model_f1[slug] = mean_f1
            row = {"model": slug, "mean_f1": round(mean_f1, 4)}
            for c, v in per_c.items():
                row[c] = round(v, 4)
            rows.append(row)
        rows.sort(key=lambda r: r["mean_f1"], reverse=True)
        with (OUT_DIR / f"recognition_cueless_{survey}_model_summary.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

        # original F1 per model (mean across constructs) from RQ3 summary
        orig_models = {}
        if survey in orig:
            for mrow in orig[survey].get("model_summary", []):
                orig_models[mrow["model"]] = mrow["mean_f1_across_constructs"]
        # VP reference mean F1 (mean of construct mean_f1)
        vp_ref = None
        if "VP" in orig:
            vals = [c["mean_f1"] for c in orig["VP"]["construct_mean_f1"]]
            vp_ref = sum(vals) / len(vals) if vals else None

        for slug, cue_f1 in cue_model_f1.items():
            if slug not in orig_models:
                orig_missing.append(f"{survey}/{slug}")
                print(f"  WARNING: no original-item F1 for {slug} ({survey}) in "
                      f"{ORIG_SUMMARY}; run RQ3/item_construct_recognition.py "
                      "and RQ3/summarize_item_construct_f1.py for all models "
                      "first, otherwise f1_orig is left blank.")
            comp_rows.append({
                "survey": survey, "model": slug,
                "f1_orig": round(orig_models.get(slug), 4) if slug in orig_models else "",
                "f1_cueless": round(cue_f1, 4),
                "f1_vp_reference": round(vp_ref, 4) if vp_ref is not None else "",
            })

        cue_mean = sum(cue_model_f1.values()) / len(cue_model_f1)
        matched_orig = [orig_models[slug] for slug in cue_model_f1 if slug in orig_models]
        orig_mean = sum(matched_orig) / len(matched_orig) if matched_orig else None
        headline["per_survey"][survey] = {
            "mean_f1_cueless": round(cue_mean, 4),
            "mean_f1_orig": round(orig_mean, 4) if orig_mean is not None else None,
            "mean_f1_vp_reference": round(vp_ref, 4) if vp_ref is not None else None,
            "n_models": len(cue_model_f1),
        }

    if orig_missing and not args.force:
        print(f"\n  Original-item F1 is missing for: {', '.join(orig_missing)}.")
        print("  Keeping the shipped recognition_f1_comparison.csv and "
              "recognition_headline.json untouched (pass --force to overwrite "
              "them with the incomplete comparison).")
        return
    if comp_rows:
        with (OUT_DIR / "recognition_f1_comparison.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(comp_rows[0].keys()))
            w.writeheader()
            w.writerows(comp_rows)
    (OUT_DIR / "recognition_headline.json").write_text(json.dumps(headline, indent=2))
    print(json.dumps(headline, indent=2))
    print(f"\nWrote recognition summary to {OUT_DIR}")


if __name__ == "__main__":
    main()
