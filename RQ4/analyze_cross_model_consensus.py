"""Compute the cross-model consensus statistics reported in the paper's
cross-model consensus appendix table (RQ4 supplementary materials).

For each (condition, value) pair, the seven RQ4 models cast a sign vote based
on the centered-mean delta:

    sign(delta) > 0  -> +1
    sign(delta) <= 0 -> -1     (ties / no-shift values are counted as negative)

The agreement for that pair is `max(positive_votes, negative_votes) / 7`.

Classification:
  - "Strong consensus": >= 6 of 7 models agree  (>= 85.7%)
  - "Weak  consensus":  < 5 of 7 models agree   (< 71.4%)
  - Intermediate (exactly 5/7) is neither strong nor weak.

Outputs `results/RQ4/cross_model_consensus.json` and prints a summary that
reproduces the numbers in Table tab:app_rq4_consensus.
"""

import json
from pathlib import Path

VALUE_ORDER = [
    "Achievement",
    "Benevolence",
    "Conformity",
    "Hedonism",
    "Power",
    "Security",
    "Self-Direction",
    "Stimulation",
    "Tradition",
    "Universalism",
]

SURVEYS = ["PVQ", "PVQ21", "VP"]
STRONG_K = 6  # at least 6 of 7 models agree
WEAK_K = 5  # strictly fewer than 5 of 7 agree


def sign_with_ties_negative(x: float) -> int:
    return 1 if x > 0 else -1


def compute_consensus(analysis: dict) -> dict:
    models = list(analysis["models"].keys())
    n_models = len(models)
    if n_models != 7:
        raise SystemExit(f"Expected the 7 RQ4 models in the analysis JSON, got {n_models}.")
    if not analysis["models"][models[0]].get("VP", {}).get("conditions"):
        raise SystemExit(
            "The analysis JSON has no VP condition data; regenerate it with "
            "RQ4/analysis_circumstance_vs_human.py after running "
            "RQ4/ecological_circumstance.py."
        )
    conditions = list(analysis["human"]["conditions"].keys())

    out: dict = {
        "schema_version": 1,
        "tie_handling": "ties_count_as_negative",
        "models": models,
        "conditions": conditions,
        "values": VALUE_ORDER,
        "per_survey": {},
    }

    for survey in SURVEYS:
        per_cell = []
        consensus_sum = 0.0
        strong = 0
        weak = 0
        intermediate = 0
        for cond in conditions:
            for value in VALUE_ORDER:
                votes = []
                for model in models:
                    delta = analysis["models"][model][survey]["conditions"][cond][
                        "delta"
                    ][value]
                    votes.append(sign_with_ties_negative(delta))
                pos = sum(1 for s in votes if s > 0)
                neg = sum(1 for s in votes if s < 0)
                agree_count = max(pos, neg)
                agree_frac = agree_count / n_models
                consensus_sum += agree_frac

                if agree_count >= STRONG_K:
                    bucket = "strong"
                    strong += 1
                elif agree_count < WEAK_K:
                    bucket = "weak"
                    weak += 1
                else:
                    bucket = "intermediate"
                    intermediate += 1

                per_cell.append(
                    {
                        "condition": cond,
                        "value": value,
                        "positive_votes": pos,
                        "negative_votes": neg,
                        "agree_count": agree_count,
                        "agree_frac": agree_frac,
                        "bucket": bucket,
                    }
                )

        n_total = len(per_cell)
        out["per_survey"][survey] = {
            "n_pairs": n_total,
            "mean_consensus": consensus_sum / n_total,
            "strong_count": strong,
            "strong_frac": strong / n_total,
            "weak_count": weak,
            "weak_frac": weak / n_total,
            "intermediate_count": intermediate,
            "intermediate_frac": intermediate / n_total,
            "per_cell": per_cell,
        }
    return out


def main() -> None:
    here = Path(__file__).resolve().parents[1]
    in_path = here / "results" / "RQ4" / "analysis_circumstance_vs_human.json"
    out_path = here / "results" / "RQ4" / "cross_model_consensus.json"

    analysis = json.loads(in_path.read_text())
    result = compute_consensus(analysis)
    out_path.write_text(json.dumps(result, indent=2))

    # Mirror the format of the paper's consensus appendix table.
    print(f"Wrote {out_path}")
    print()
    print(f"{'Metric':30s} {'PVQ-40':>20s} {'PVQ-21':>20s} {'VP':>20s}")
    keys = ("PVQ", "PVQ21", "VP")
    s = result["per_survey"]
    print(
        f"{'Mean consensus':30s} "
        + " ".join(f"{s[k]['mean_consensus']:>20.3f}" for k in keys)
    )
    print(
        f"{'Strong consensus (>=6/7)':30s} "
        + " ".join(
            f"{s[k]['strong_count']:>4d}/{s[k]['n_pairs']:<4d} "
            f"({s[k]['strong_frac']*100:>5.1f}%)"
            for k in keys
        )
    )
    print(
        f"{'Weak consensus (<5/7)':30s} "
        + " ".join(
            f"{s[k]['weak_count']:>4d}/{s[k]['n_pairs']:<4d} "
            f"({s[k]['weak_frac']*100:>5.1f}%)"
            for k in keys
        )
    )
    print(
        f"{'Intermediate (=5/7)':30s} "
        + " ".join(
            f"{s[k]['intermediate_count']:>4d}/{s[k]['n_pairs']:<4d} "
            f"({s[k]['intermediate_frac']*100:>5.1f}%)"
            for k in keys
        )
    )


if __name__ == "__main__":
    main()
