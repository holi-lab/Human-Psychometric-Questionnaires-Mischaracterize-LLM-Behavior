#!/usr/bin/env python3
"""
RQ3 summary: construct-by-model recognition F1 tables.

Reads the existing item-construct recognition outputs and computes binary
classification metrics per construct, per model, for each survey.

Usage:
    python RQ3/summarize_item_construct_f1.py

Outputs:
    results/RQ3/item_construct_recognition_summary/summary.json
    results/RQ3/item_construct_recognition_summary/{survey}_construct_by_model.csv
    results/RQ3/item_construct_recognition_summary/{survey}_model_summary.csv
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT / "results" / "RQ3" / "item_construct_recognition"
OUT_DIR = ROOT / "results" / "RQ3" / "item_construct_recognition_summary"

SURVEYS = ["VP", "PVQ", "PVQ21", "BFI44", "BFI10"]
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


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def f1_from_counts(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def compute_metrics_for_file(survey_name: str, path: Path) -> dict:
    payload = load_json(path)
    constructs = list(payload["construct_definitions_used"].keys())
    per_construct = {}

    for construct in constructs:
        tp = fp = fn = tn = 0
        for item in payload["results"].values():
            pred = bool(item["predictions"][construct]["parsed"])
            if survey_name == "VP":
                gold = construct in item.get("ground_truth_positive", [])
            else:
                gold = item.get("ground_truth_construct") == construct

            if pred and gold:
                tp += 1
            elif pred and not gold:
                fp += 1
            elif (not pred) and gold:
                fn += 1
            else:
                tn += 1

        precision, recall, f1 = f1_from_counts(tp, fp, fn)
        per_construct[construct] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    mean_f1 = sum(row["f1"] for row in per_construct.values()) / len(per_construct)
    return {
        "model": payload["model"].split("/")[-1],
        "survey": survey_name,
        "n_items": len(payload["results"]),
        "per_construct": per_construct,
        "mean_f1_across_constructs": mean_f1,
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_survey_summary(survey_name: str) -> dict:
    survey_dir = INPUT_DIR / survey_name
    model_paths = sorted(survey_dir.glob("*.json"))
    per_model = [compute_metrics_for_file(survey_name, path) for path in model_paths]
    models = [row["model"] for row in per_model]
    constructs = list(per_model[0]["per_construct"].keys()) if per_model else []

    construct_rows = []
    for construct in constructs:
        values = [row["per_construct"][construct]["f1"] for row in per_model]
        summary_row = {
            "construct": construct,
            "mean_f1": sum(values) / len(values),
        }
        for row in per_model:
            summary_row[row["model"]] = row["per_construct"][construct]["f1"]
        construct_rows.append(summary_row)

    construct_rows.sort(key=lambda row: row["mean_f1"], reverse=True)

    model_rows = []
    for row in per_model:
        model_row = {
            "model": row["model"],
            "model_short": MODEL_SHORT.get(row["model"], row["model"]),
            "mean_f1_across_constructs": row["mean_f1_across_constructs"],
        }
        for construct in constructs:
            model_row[construct] = row["per_construct"][construct]["f1"]
        model_rows.append(model_row)
    model_rows.sort(key=lambda row: row["mean_f1_across_constructs"], reverse=True)

    write_csv(
        OUT_DIR / f"{survey_name}_construct_by_model.csv",
        construct_rows,
        ["construct", "mean_f1", *models],
    )
    write_csv(
        OUT_DIR / f"{survey_name}_model_summary.csv",
        model_rows,
        ["model", "model_short", "mean_f1_across_constructs", *constructs],
    )

    return {
        "survey": survey_name,
        "models": models,
        "construct_order_by_mean_f1": [row["construct"] for row in construct_rows],
        "construct_mean_f1": [
            {"construct": row["construct"], "mean_f1": row["mean_f1"]}
            for row in construct_rows
        ],
        "construct_by_model_f1": construct_rows,
        "model_summary": model_rows,
    }


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="Overwrite the shipped summary even when the regenerated "
                         "model set differs from it.")
    args = ap.parse_args()

    if not INPUT_DIR.is_dir() or not any(INPUT_DIR.iterdir()):
        raise SystemExit(
            f"Input directory {INPUT_DIR} is missing or empty.\n"
            "The raw per-model recognition outputs are not shipped with the "
            "repository; run RQ3/item_construct_recognition.py first.\n"
            "Aborting so the shipped summary files are not overwritten."
        )
    shipped = OUT_DIR / "summary.json"
    if shipped.exists() and not args.force:
        try:
            prev_models = {
                model
                for survey in json.loads(shipped.read_text()).values()
                for model in survey.get("models", [])
            }
        except (json.JSONDecodeError, AttributeError):
            prev_models = set()
        new_models = {
            path.stem
            for survey in SURVEYS
            for path in (INPUT_DIR / survey).glob("*.json")
        }
        if prev_models and new_models != prev_models:
            raise SystemExit(
                "The regenerated model set differs from the shipped summary:\n"
                f"  shipped: {sorted(prev_models)}\n"
                f"  found:   {sorted(new_models)}\n"
                "The shipped summary covers the seven models reported in the "
                "paper's recognition tables. Pass --force to overwrite it."
            )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        survey_name: build_survey_summary(survey_name)
        for survey_name in SURVEYS
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
