#!/usr/bin/env python3
"""Length-normalization ablation (Appendix, RQ1): rebuild generation-probability profiles with a selectable score field.

Replicates RQ1/ecological_profile.py exactly (two-step macro-average,
corr >= 0.3 tagging) but lets you choose the per-response score:
  - total_logprob      : paper's Eq. 1 (sum of token log-probs)  [sanity check]
  - normalized_logprob : per-token mean log-prob (length-normalized ablation)

Usage (from the repository root):
    python RQ1/length_norm/build_profiles.py \
        --score-field normalized_logprob \
        --output-dir RQ1/length_norm/profiles_normalized
"""

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OFFICIAL = HERE.parent.parent  # .../Official
VP_SURVEY = OFFICIAL / "surveys" / "VP.json"
ECOLOGICAL_DIR = OFFICIAL / "results" / "RQ1" / "ecological"

CORR_THRESHOLD = 0.3
CORRELATION_KEYS = [
    ("correlations", "pvq_values"),
    ("higher_pvq_correlations", "higher_order_values"),
    ("bfi_correlations", "bfi_traits"),
]


def load_vp_lookup() -> dict:
    scenarios = json.loads(VP_SURVEY.read_text())
    lookup = {}
    for sc in scenarios:
        outputs_map = {}
        for out in sc["outputs"]:
            outputs_map[out["id"]] = {
                corr_key: {name: val for name, val in out.get(corr_key, [])}
                for corr_key, _ in CORRELATION_KEYS
            }
        lookup[sc["portrait_id"]] = outputs_map
    return lookup


def scenario_profile(scenario_result: dict, vp_lookup: dict, score_field: str):
    pid = scenario_result["portrait_id"]
    if pid not in vp_lookup:
        return None
    valid, scores = [], []
    for out in scenario_result["outputs"]:
        s = out.get(score_field)
        if s is None:
            continue
        scores.append(float(s))
        valid.append(out)
    if not valid:
        return None
    corr_map = vp_lookup[pid]
    profile = {}
    for corr_key, label in CORRELATION_KEYS:
        acc, cnt = defaultdict(float), defaultdict(int)
        for score, out in zip(scores, valid):
            corrs = corr_map.get(out["output_id"], {}).get(corr_key, {})
            for cname, cval in corrs.items():
                if cval >= CORR_THRESHOLD:
                    acc[cname] += score
                    cnt[cname] += 1
        profile[label] = (
            {c: acc[c] / cnt[c] for c in acc} if cnt else {}
        )
    return profile


def model_profile(raw: dict, vp_lookup: dict, score_field: str):
    accum = {}
    n_valid = 0
    for sc in raw.get("results", []):
        p = scenario_profile(sc, vp_lookup, score_field)
        if p is None:
            continue
        n_valid += 1
        for label, constructs in p.items():
            accum.setdefault(label, defaultdict(list))
            for cname, val in constructs.items():
                accum[label][cname].append(val)
    profile = {
        label: {c: round(float(np.mean(v)), 6) for c, v in sorted(constructs.items())}
        for label, constructs in sorted(accum.items())
    }
    return profile, n_valid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score-field", required=True,
                    choices=["total_logprob", "normalized_logprob"])
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    out_dir = args.output_dir if args.output_dir.is_absolute() else Path.cwd() / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    vp_lookup = load_vp_lookup()
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "score_field": args.score_field,
        "corr_threshold": CORR_THRESHOLD,
        "source_dir": str(ECOLOGICAL_DIR),
        "models": {},
    }
    for path in sorted(ECOLOGICAL_DIR.glob("*.json")):
        raw = json.loads(path.read_text())
        profile, n_valid = model_profile(raw, vp_lookup, args.score_field)
        data = {
            "model": raw.get("model"),
            "survey": "VP",
            "method": "raw_prob",
            "score_field": args.score_field,
            "correlation_threshold": CORR_THRESHOLD,
            "n_scenarios_valid": n_valid,
            "n_scenarios_total": len(raw.get("results", [])),
            "gen_prob_profile": profile,
        }
        (out_dir / path.name).write_text(json.dumps(data, indent=2, ensure_ascii=False))
        summary["models"][path.stem] = {"n_valid_scenarios": n_valid}
        print(f"  {path.stem}: {n_valid} scenarios -> {out_dir / path.name}")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("done")


if __name__ == "__main__":
    main()
