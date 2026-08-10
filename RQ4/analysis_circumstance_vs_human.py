#!/usr/bin/env python3
"""
RQ4 Analysis: Per-value delta profiles — LLM persona shift vs human
demographic differences.

For each survey (PVQ, PVQ21, VP):
  LLM delta  = RQ4_conditioned_centered_mean − RQ1_baseline_centered_mean
  Human delta = condition_subgroup_centered_mean − overall_centered_mean

The human reference is read from RQ4/ess_human_aggregates.json, which contains
subgroup-level aggregates (per-respondent ipsative centering per paper Eq. 2,
then per-dimension averaging) derived from European Social Survey microdata.
The individual-level ESS records are not part of this repository.

Output: results/RQ4/analysis_circumstance_vs_human.json

Usage:
    python RQ4/analysis_circumstance_vs_human.py
"""

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# ─── Constants ──────────────────────────────────────────────

VALUE_NAMES = [
    "Achievement", "Benevolence", "Conformity", "Hedonism", "Power",
    "Security", "Self-Direction", "Stimulation", "Tradition", "Universalism",
]

VP_KEY_TO_LLM = {
    "Self_Direction": "Self-Direction",
    "Achievement": "Achievement",
    "Benevolence": "Benevolence",
    "Conformity": "Conformity",
    "Hedonism": "Hedonism",
    "Power": "Power",
    "Security": "Security",
    "Stimulation": "Stimulation",
    "Tradition": "Tradition",
    "Universalism": "Universalism",
}

MODELS = [
    "gemma-3-27b-it",
    "gpt-oss-20b",
    "gpt-oss-120b",
    "Qwen2.5-7B-Instruct",
    "Qwen2.5-72B-Instruct",
    "Qwen3-30B-A3B-Instruct-2507",
    "Qwen3-235B-A22B-Instruct-2507-FP8",
]

CONDITIONS = [
    "Gender_A", "Gender_B",
    "Age_A", "Age_B",
    "Political_A", "Political_B",
    "Education_A", "Education_B",
]

CONDITION_DESC = {
    "Gender_A": "Male",
    "Gender_B": "Female",
    "Age_A": "20–39 years old",
    "Age_B": "80+ years old",
    "Political_A": "Right-wing",
    "Political_B": "Left-wing",
    "Education_A": "Below university",
    "Education_B": "University or above",
}

# Subgroup filter definitions are documented in RQ4/ess_human_aggregates.json
# (field "filter" of each condition entry).


# ─── Helpers ────────────────────────────────────────────────

def _r(v, n=6):
    return round(v, n) if v is not None else None


def centered(raw: dict[str, float]) -> dict[str, float]:
    """Subtract the unweighted mean across the 10 value dimensions
    (paper Eq. 2, applied to a single 10-dim profile)."""
    vals = [v for v in raw.values() if v is not None]
    if not vals:
        return {k: None for k in raw}
    m = sum(vals) / len(vals)
    return {k: _r(v - m) if v is not None else None for k, v in sorted(raw.items())}


def delta(a: dict, b: dict) -> dict[str, float]:
    return {v: _r((a.get(v) or 0) - (b.get(v) or 0)) for v in VALUE_NAMES}


# ─── Human data ─────────────────────────────────────────────

def load_human():
    """Load the ESS subgroup aggregates (see module docstring).

    The per-condition centered profiles were computed from per-respondent
    records with ipsative centering (paper Eq. 2): each respondent's profile
    is centered by their grand mean across the 10 value dimensions, then
    averaged per subgroup.
    """
    return json.loads((ROOT / "RQ4" / "ess_human_aggregates.json").read_text())


# ─── LLM: Established (PVQ, PVQ21) ─────────────────────────

def load_established_centered(survey, model, base="RQ1"):
    if base == "RQ1":
        p = ROOT / "results" / "RQ1" / "established" / survey / f"{model}.json"
    else:
        p = ROOT / "results" / "RQ4" / "established_circumstance" / survey / base / f"{model}.json"
    if not p.exists():
        return None
    combined = json.loads(p.read_text()).get("construct_averages", {}).get("combined")
    return centered(combined) if combined else None


# ─── LLM: Ecological (VP) ──────────────────────────────────

CORR_THRESHOLD = 0.3
CORR_KEYS = [("correlations", "pvq_values")]


def load_vp_survey():
    scenarios = json.loads((ROOT / "surveys" / "VP.json").read_text())
    lookup = {}
    for sc in scenarios:
        pid = sc["portrait_id"]
        omap = {}
        for out in sc["outputs"]:
            cd = {}
            for ck, _ in CORR_KEYS:
                raw = out.get(ck, [])
                cd[ck] = {name: val for name, val in raw}
            omap[out["id"]] = cd
        lookup[pid] = omap
    return lookup


def compute_vp_profile(eco_result, vp_lookup):
    """Extract pvq_values gen-prob profile from ecological result."""
    accum: dict[str, list[float]] = defaultdict(list)

    for sc in eco_result.get("results", []):
        pid = sc["portrait_id"]
        if pid not in vp_lookup:
            continue
        outputs = sc["outputs"]
        scores = []
        valid = []
        for out in outputs:
            lp = out.get("total_logprob")
            if lp is not None:
                scores.append(float(lp))
                valid.append(out)

        if not valid:
            continue

        scenario_corr = vp_lookup[pid]
        scenario_profile: dict[str, list[float]] = defaultdict(list)

        for score, out in zip(scores, valid):
            oid = out["output_id"]
            corrs = scenario_corr.get(oid, {}).get("correlations", {})
            for cname, cval in corrs.items():
                if cval >= CORR_THRESHOLD:
                    scenario_profile[cname].append(score)

        for cname, vals in scenario_profile.items():
            accum[cname].append(float(np.mean(vals)))

    result = {}
    for vp_key, llm_key in VP_KEY_TO_LLM.items():
        if vp_key in accum:
            result[llm_key] = _r(float(np.mean(accum[vp_key])))
    return result if result else None


def load_vp_centered(model, vp_lookup, condition=None):
    if condition is None:
        p = ROOT / "results" / "RQ1" / "ecological" / f"{model}.json"
    else:
        p = ROOT / "results" / "RQ4" / "ecological_circumstance" / condition / f"{model}.json"
    if not p.exists():
        return None
    eco = json.loads(p.read_text())
    raw = compute_vp_profile(eco, vp_lookup)
    return centered(raw) if raw else None


# ─── Main ───────────────────────────────────────────────────

def main():
    eco_dir = ROOT / "results" / "RQ4" / "ecological_circumstance"
    if not eco_dir.is_dir():
        raise SystemExit(
            f"{eco_dir} is missing.\n"
            "The raw persona-conditioned generation probability outputs are not "
            "shipped with the repository; run RQ4/ecological_circumstance.py "
            "first.\nAborting so the shipped analysis JSON is not overwritten."
        )
    print("Loading data ...")
    human_agg = load_human()
    vp_lookup = load_vp_survey()
    print(f"  Human respondents (aggregated): {human_agg['overall']['n']}, "
          f"VP scenarios: {len(vp_lookup)}")

    # ── Human ───────────────────────────────────────────────
    # Paper Eq. (2): each respondent's profile is centered by subtracting
    # their grand mean across the 10 value dimensions, then aggregated per
    # subgroup (precomputed in ess_human_aggregates.json). The LLM side
    # applies the same operation to each model's raw 10-dim profile (one
    # profile per model, so the per-respondent step degenerates to centering
    # that single profile via `centered()`).
    h_overall = human_agg["overall"]["centered_profile"]
    h_cond = {}
    for c in CONDITIONS:
        cond_agg = human_agg["conditions"][c]
        prof = cond_agg["centered_profile"]
        d = delta(prof, h_overall)
        h_cond[c] = {"profile": prof, "delta": d, "n": cond_agg["n"]}
        print(f"  Human {c:>15} (n={cond_agg['n']:>5})  delta: {d}")

    human_out = {
        "overall_profile": h_overall,
        "overall_n": human_agg["overall"]["n"],
        "conditions": h_cond,
    }

    # ── LLM ─────────────────────────────────────────────────
    models_out = {}

    for model in MODELS:
        print(f"\n  === {model} ===")
        mdata = {}

        for survey in ["PVQ", "PVQ21"]:
            baseline = load_established_centered(survey, model, base="RQ1")
            if not baseline:
                print(f"    {survey}: no RQ1 baseline")
                continue

            conds = {}
            for c in CONDITIONS:
                prof = load_established_centered(survey, model, base=c)
                if prof:
                    d = delta(prof, baseline)
                    conds[c] = {"profile": prof, "delta": d}

            mdata[survey] = {"baseline": baseline, "conditions": conds}
            print(f"    {survey}: baseline OK, {len(conds)} conditions")

        vp_baseline = load_vp_centered(model, vp_lookup)
        if vp_baseline:
            vp_conds = {}
            for c in CONDITIONS:
                prof = load_vp_centered(model, vp_lookup, condition=c)
                if prof:
                    d = delta(prof, vp_baseline)
                    vp_conds[c] = {"profile": prof, "delta": d}
            mdata["VP"] = {"baseline": vp_baseline, "conditions": vp_conds}
            print(f"    VP: baseline OK, {len(vp_conds)} conditions")
        else:
            print(f"    VP: no RQ1 baseline")

        models_out[model] = mdata

    # ── Assemble & save ─────────────────────────────────────
    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metadata": {
            "description": (
                "Per-value delta profiles: LLM persona shift (RQ4 − RQ1) "
                "vs human demographic effect (condition − overall). "
                "All profiles are centered means (PVQ value dimensions)."
            ),
            "surveys": ["PVQ", "PVQ21", "VP"],
            "conditions": CONDITIONS,
            "condition_descriptions": CONDITION_DESC,
            "models": MODELS,
            "value_dimensions": VALUE_NAMES,
            "note": "gemma-3-4b-it excluded due to high non-response rate under circumstance prompting.",
        },
        "human": human_out,
        "models": models_out,
    }

    out = ROOT / "results" / "RQ4" / "analysis_circumstance_vs_human.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nSaved → {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
