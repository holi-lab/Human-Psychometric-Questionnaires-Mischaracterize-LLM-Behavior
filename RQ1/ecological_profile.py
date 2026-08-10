#!/usr/bin/env python3
"""
RQ1: Ecological Gen-Prob Profile Extraction

For each model with pre-computed log-probability results:
  - Loads RQ1 ecological results (score per output)
  - Loads VP survey for per-output value correlations
  - Tags outputs where correlation >= 0.3 for each construct
  - Uses raw total_logprob (no length-normalisation, no exp)
  - Averages total_logprob across tagged outputs per construct

Usage:
    python RQ1/ecological_profile.py

Results:
    results/RQ1/ecological_profiles/{model_slug}.json
    results/RQ1/ecological_profiles/summary.json
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

VP_SURVEY = ROOT / "surveys" / "VP.json"
ECOLOGICAL_DIR = ROOT / "results" / "RQ1" / "ecological"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "RQ1" / "ecological_profiles"

CORR_THRESHOLD = 0.3

CORRELATION_KEYS = [
    ("correlations", "pvq_values"),
    ("higher_pvq_correlations", "higher_order_values"),
    ("bfi_correlations", "bfi_traits"),
]


def log(msg: str = ""):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ───────────────────────────────────────────────────────────
#  Data Loading
# ───────────────────────────────────────────────────────────

def discover_models() -> list[str]:
    if not ECOLOGICAL_DIR.exists():
        return []
    return sorted(p.stem for p in ECOLOGICAL_DIR.glob("*.json"))


def load_ecological_result(model_slug: str) -> dict:
    path = ECOLOGICAL_DIR / f"{model_slug}.json"
    return json.loads(path.read_text())


def load_vp_survey() -> dict:
    """Load VP survey and build lookup: portrait_id → {output_id → correlations}."""
    scenarios = json.loads(VP_SURVEY.read_text())
    lookup: dict[int, dict] = {}
    for sc in scenarios:
        pid = sc["portrait_id"]
        outputs_map: dict[int, dict] = {}
        for out in sc["outputs"]:
            corr_data: dict[str, dict[str, float]] = {}
            for corr_key, _ in CORRELATION_KEYS:
                raw = out.get(corr_key, [])
                corr_data[corr_key] = {name: val for name, val in raw}
            outputs_map[out["id"]] = corr_data
        lookup[pid] = outputs_map
    return lookup


# ───────────────────────────────────────────────────────────
#  Gen-Prob Profile Computation
# ───────────────────────────────────────────────────────────

def compute_scenario_profile(
    scenario_result: dict,
    vp_lookup: dict[int, dict],
) -> dict[str, dict[str, float]] | None:
    """Compute gen-prob profile for one scenario.

    Uses raw total_logprob per output directly (no exp, no length-normalisation).
    For each construct, averages the total_logprob of tagged outputs.

    Returns {label: {construct_name: avg_logprob}} or None on error.
    """
    pid = scenario_result["portrait_id"]
    outputs = scenario_result["outputs"]

    if pid not in vp_lookup:
        return None

    valid_outputs = []
    output_scores: list[float] = []

    for out in outputs:
        total_lp = out.get("total_logprob")
        if total_lp is None:
            continue
        output_scores.append(float(total_lp))
        valid_outputs.append(out)

    if not valid_outputs:
        return None

    scenario_corr_map = vp_lookup[pid]

    profile: dict[str, dict[str, float]] = {}
    for corr_key, label in CORRELATION_KEYS:
        construct_scores: dict[str, float] = defaultdict(float)
        construct_counts: dict[str, int] = defaultdict(int)
        for score, out in zip(output_scores, valid_outputs):
            oid = out["output_id"]
            corrs = scenario_corr_map.get(oid, {}).get(corr_key, {})
            for cname, cval in corrs.items():
                if cval >= CORR_THRESHOLD:
                    construct_scores[cname] += score
                    construct_counts[cname] += 1
        if construct_counts:
            profile[label] = {
                cname: construct_scores[cname] / construct_counts[cname]
                for cname in construct_scores
            }
        else:
            profile[label] = {}

    return profile


def compute_model_profile(
    ecological_result: dict,
    vp_lookup: dict[int, dict],
) -> tuple[dict[str, dict[str, float]], int, int]:
    """Compute the aggregated gen-prob profile across all scenarios.

    Returns (profile, n_valid, n_total).
    """
    accum: dict[str, dict[str, list[float]]] = {}
    n_valid = 0
    results = ecological_result.get("results", [])

    for sc in results:
        p = compute_scenario_profile(sc, vp_lookup)
        if p is None:
            continue
        n_valid += 1
        for label, constructs in p.items():
            if label not in accum:
                accum[label] = defaultdict(list)
            for cname, val in constructs.items():
                accum[label][cname].append(val)

    profile: dict[str, dict[str, float]] = {}
    for label in sorted(accum):
        profile[label] = {
            cname: round(float(np.mean(vals)), 6)
            for cname, vals in sorted(accum[label].items())
        }

    return profile, n_valid, len(results)


# ───────────────────────────────────────────────────────────
#  I/O
# ───────────────────────────────────────────────────────────

def save_result(data: dict, model_slug: str, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{model_slug}.json"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    log(f"  => Saved: {path.relative_to(ROOT)}")


def save_summary(summary: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    log(f"  => Summary: {path.relative_to(ROOT)}")


# ───────────────────────────────────────────────────────────
#  Main
# ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for generated ecological profiles.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir or DEFAULT_OUTPUT_DIR
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    log("=" * 60)
    log("  RQ1: Ecological Gen-Prob Profile Extraction")
    log("  Score: raw total_logprob → mean per construct")
    log(f"  Correlation threshold: >= {CORR_THRESHOLD}")
    log(f"  Output dir: {output_dir.relative_to(ROOT)}")
    log("=" * 60)

    if not VP_SURVEY.exists():
        log(f"ERROR: VP survey not found at {VP_SURVEY}")
        sys.exit(1)

    log("  Loading VP survey correlations ...")
    vp_lookup = load_vp_survey()
    log(f"  Loaded {len(vp_lookup)} scenarios from VP survey")

    models = discover_models()
    if not models:
        log("  No ecological results found. Exiting.")
        sys.exit(1)
    log(f"  Found {len(models)} model(s): {models}")

    description = (
        "For each output, uses raw total_logprob directly (no exp, no length-normalisation). "
        "For each construct, averages the total_logprob of outputs "
        "tagged with correlation >= 0.3."
    )

    summary: dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "method": "raw_prob",
        "score_field": "total_logprob",
        "correlation_threshold": CORR_THRESHOLD,
        "description": description,
        "models": {},
    }

    for ms in models:
        log(f"\n{'─' * 50}")
        log(f"  Model: {ms}")
        log(f"{'─' * 50}")

        eco_result = load_ecological_result(ms)
        profile, n_valid, n_total = compute_model_profile(eco_result, vp_lookup)

        log(f"  Valid scenarios: {n_valid}/{n_total}")

        if not profile:
            log(f"  WARNING: No valid profiles for {ms}, skipping.")
            continue

        for label, constructs in profile.items():
            log(f"  {label}:")
            for cname, val in sorted(constructs.items(), key=lambda x: -x[1]):
                log(f"    {cname:25s} = {val:.6f}")

        full_output = {
            "model": eco_result.get("model", ms),
            "survey": "VP",
            "method": "raw_prob",
            "score_field": "total_logprob",
            "correlation_threshold": CORR_THRESHOLD,
            "n_scenarios_valid": n_valid,
            "n_scenarios_total": n_total,
            "gen_prob_profile": profile,
        }
        save_result(full_output, ms, output_dir)

        summary["models"][ms] = {
            "n_scenarios": n_valid,
            "profile": profile,
        }

    save_summary(summary, output_dir)

    log("\n" + "=" * 60)
    log("  Done!")
    log("=" * 60)


if __name__ == "__main__":
    main()
