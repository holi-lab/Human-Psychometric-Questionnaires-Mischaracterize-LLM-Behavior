#!/usr/bin/env python3
# Note: "aic" here stands for "assessing intra-construct consistency";
# unrelated to Akaike Information Criterion.
"""
RQ2: η² & WMV — Established Questionnaires

For each model × survey × prompt variant, treats questionnaire **items** as
observations grouped by construct and computes two complementary metrics:

  η²  (eta-squared)  — SS_between / SS_total on raw Likert scores.
       Scale-invariant; no z-score normalisation needed.
       Analytical baseline under random labels: (K−1)/(N−1).

  WMV (within-model variance) — mean within-construct sample variance
       on z-scored items.  Baseline under random labels ≈ 1.0.
       z-score normalisation puts all models on a common scale.

A permutation test (1 000 shuffles of construct labels, group sizes held
constant) provides one-sided empirical p-values for both metrics.

Usage:
    python RQ2/aic_established.py

Outputs:
    results/RQ2/aic_established/{survey}.json
"""

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ESTABLISHED_DIR = ROOT / "results" / "RQ1" / "established"
OUTPUT_DIR = ROOT / "results" / "RQ2" / "aic_established"

SURVEYS = ["BFI10", "BFI44", "PVQ", "PVQ21"]
PROMPT_VARIANTS = ["normal", "reversed"]

N_PERMUTATIONS = 1_000
SEED = 42


def log(msg: str = ""):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def load_json(path: Path) -> dict | list:
    return json.loads(path.read_text())


def _r6(value, ndigits=6):
    return round(float(value), ndigits) if value is not None else None


def discover_model_paths(survey_name: str) -> list[Path]:
    survey_dir = ESTABLISHED_DIR / survey_name
    if not survey_dir.exists():
        return []
    return sorted(survey_dir.glob("*.json"))


# ───────────────────────────────────────────────────────────
#  Item Extraction
# ───────────────────────────────────────────────────────────

def _build_combined_prompt_scores(result: dict) -> dict[str, dict]:
    """Average available prompt-variant scores item-wise for one model."""
    combined: dict[str, dict] = {}
    responses = result.get("responses", {})

    all_item_ids: set[str] = set()
    for pk in PROMPT_VARIANTS:
        all_item_ids.update(responses.get(pk, {}).keys())

    for item_id in sorted(all_item_ids, key=lambda x: int(x) if str(x).isdigit() else str(x)):
        entries = [
            responses.get(pk, {}).get(item_id)
            for pk in PROMPT_VARIANTS
            if responses.get(pk, {}).get(item_id) is not None
        ]
        if not entries:
            continue
        valid_scores = [
            e.get("final_score") for e in entries if e.get("final_score") is not None
        ]
        combined[item_id] = {
            "construct": entries[0].get("construct"),
            "final_score": sum(valid_scores) / len(valid_scores) if valid_scores else None,
        }
    return combined


def build_construct_groups(
    result: dict,
    prompt_key: str,
) -> dict[str, list[float]]:
    """construct → [item scores] for one model, one prompt variant."""
    if prompt_key == "combined":
        items = _build_combined_prompt_scores(result)
    else:
        items = result.get("responses", {}).get(prompt_key, {})

    groups: dict[str, list[float]] = defaultdict(list)
    for _iid, entry in items.items():
        construct = entry.get("construct")
        score = entry.get("final_score")
        if construct and score is not None:
            groups[construct].append(float(score))
    return dict(sorted(groups.items()))


# ───────────────────────────────────────────────────────────
#  η² (Eta-squared)
# ───────────────────────────────────────────────────────────

def compute_eta_squared(groups: dict[str, list[float]]) -> dict:
    """η² = SS_between / SS_total from one-way layout."""
    all_vals = [v for sc in groups.values() for v in sc]
    N = len(all_vals)
    K = len(groups)
    if N < 2 or K < 2:
        return {"eta_squared": None, "ss_between": None, "ss_total": None,
                "N": N, "K": K, "analytical_baseline": None}

    gm = sum(all_vals) / N
    ss_total = sum((x - gm) ** 2 for x in all_vals)
    if ss_total == 0:
        return {"eta_squared": None, "ss_between": 0.0, "ss_total": 0.0,
                "N": N, "K": K, "analytical_baseline": (K - 1) / (N - 1)}

    ss_between = sum(
        len(sc) * (sum(sc) / len(sc) - gm) ** 2
        for sc in groups.values() if sc
    )
    return {
        "eta_squared": ss_between / ss_total,
        "ss_between": ss_between,
        "ss_total": ss_total,
        "N": N,
        "K": K,
        "analytical_baseline": (K - 1) / (N - 1),
    }


def _eta_fast(groups: dict[str, list[float]]) -> float | None:
    """η² returning bare float (for permutation inner loop)."""
    all_vals = [v for sc in groups.values() for v in sc]
    N = len(all_vals)
    if N < 2 or len(groups) < 2:
        return None
    gm = sum(all_vals) / N
    ss_t = sum((x - gm) ** 2 for x in all_vals)
    if ss_t == 0:
        return None
    ss_b = sum(len(sc) * (sum(sc) / len(sc) - gm) ** 2
               for sc in groups.values() if sc)
    return ss_b / ss_t


# ───────────────────────────────────────────────────────────
#  WMV (Within-Model Variance)
# ───────────────────────────────────────────────────────────

def _zscore_groups(groups: dict[str, list[float]]) -> dict[str, list[float]]:
    """z-score all items within a model (population std, ddof=0)."""
    all_vals = [v for sc in groups.values() for v in sc]
    mu = sum(all_vals) / len(all_vals)
    std = (sum((x - mu) ** 2 for x in all_vals) / len(all_vals)) ** 0.5
    if std == 0:
        return {c: [0.0] * len(sc) for c, sc in groups.items()}
    return {c: [(v - mu) / std for v in sc] for c, sc in groups.items()}


def _wmv_from_zgroups(z_groups: dict[str, list[float]]) -> float | None:
    """WMV = (1/K) Σ_c [ (1/(k_c−1)) Σ_j (z_j − z̄_c)² ]."""
    K_valid = 0
    wmv_sum = 0.0
    for zs in z_groups.values():
        k_c = len(zs)
        if k_c < 2:
            continue
        z_bar = sum(zs) / k_c
        wmv_sum += sum((z - z_bar) ** 2 for z in zs) / (k_c - 1)
        K_valid += 1
    return wmv_sum / K_valid if K_valid > 0 else None


def compute_wmv(groups: dict[str, list[float]]) -> float | None:
    return _wmv_from_zgroups(_zscore_groups(groups))


# ───────────────────────────────────────────────────────────
#  Permutation Test
# ───────────────────────────────────────────────────────────

def permutation_test(
    groups: dict[str, list[float]],
    n_perm: int = N_PERMUTATIONS,
    seed: int = SEED,
) -> dict:
    obs_eta = _eta_fast(groups)
    z_groups = _zscore_groups(groups)
    obs_wmv = _wmv_from_zgroups(z_groups)

    construct_order = sorted(groups.keys())
    sizes = [(c, len(groups[c])) for c in construct_order]

    all_raw = [v for c in construct_order for v in groups[c]]
    all_z = [v for c in construct_order for v in z_groups[c]]

    rng = np.random.default_rng(seed)
    perm_etas: list[float] = []
    perm_wmvs: list[float] = []

    for _ in range(n_perm):
        perm = rng.permutation(len(all_raw))
        fake_raw: dict[str, list[float]] = {}
        fake_z: dict[str, list[float]] = {}
        offset = 0
        for c, sz in sizes:
            idx = perm[offset:offset + sz]
            fake_raw[c] = [all_raw[i] for i in idx]
            fake_z[c] = [all_z[i] for i in idx]
            offset += sz

        e = _eta_fast(fake_raw)
        w = _wmv_from_zgroups(fake_z)
        if e is not None:
            perm_etas.append(e)
        if w is not None:
            perm_wmvs.append(w)

    result: dict = {"n_permutations": n_perm}
    if perm_etas and obs_eta is not None:
        arr = np.array(perm_etas)
        result["eta_squared_p_value"] = _r6(np.mean(arr >= obs_eta - 1e-12))
        result["eta_squared_null_mean"] = _r6(arr.mean())
        result["eta_squared_null_std"] = _r6(arr.std())
    if perm_wmvs and obs_wmv is not None:
        arr = np.array(perm_wmvs)
        result["wmv_p_value"] = _r6(np.mean(arr <= obs_wmv + 1e-12))
        result["wmv_null_mean"] = _r6(arr.mean())
        result["wmv_null_std"] = _r6(arr.std())
    return result


# ───────────────────────────────────────────────────────────
#  Per-model Result Assembly
# ───────────────────────────────────────────────────────────

def compute_model_result(groups: dict[str, list[float]]) -> dict:
    eta_res = compute_eta_squared(groups)
    wmv = compute_wmv(groups)
    perm = permutation_test(groups)

    z_groups = _zscore_groups(groups)
    construct_details: dict[str, dict] = {}
    for c in sorted(groups.keys()):
        raw = groups[c]
        zs = z_groups[c]
        k_c = len(raw)
        raw_mean = sum(raw) / k_c
        z_mean = sum(zs) / k_c
        z_var = (
            sum((z - z_mean) ** 2 for z in zs) / (k_c - 1) if k_c >= 2 else None
        )
        construct_details[c] = {
            "n_items": k_c,
            "raw_mean": _r6(raw_mean),
            "z_mean": _r6(z_mean),
            "z_variance": _r6(z_var),
        }

    return {
        "eta_squared": _r6(eta_res["eta_squared"]),
        "analytical_baseline_eta_squared": _r6(eta_res["analytical_baseline"]),
        "wmv": _r6(wmv),
        "N": eta_res["N"],
        "K": eta_res["K"],
        "permutation": perm,
        "construct_details": construct_details,
    }


# ───────────────────────────────────────────────────────────
#  Survey-level Driver
# ───────────────────────────────────────────────────────────

def compute_survey(survey_name: str) -> dict | None:
    model_paths = discover_model_paths(survey_name)
    if not model_paths:
        return None

    results_by_model = {p.stem: load_json(p) for p in model_paths}
    model_slugs = sorted(results_by_model.keys())

    by_prompt: dict[str, dict] = {}
    for prompt_key in [*PROMPT_VARIANTS, "combined"]:
        log(f"    Prompt variant: {prompt_key}")
        per_model: dict[str, dict] = {}
        for slug in model_slugs:
            groups = build_construct_groups(results_by_model[slug], prompt_key)
            if len(groups) < 2:
                log(f"      {slug}: <2 constructs, skipping")
                continue

            res = compute_model_result(groups)
            eta = res["eta_squared"] or 0
            wmv = res["wmv"] or 0
            p_eta = res["permutation"].get("eta_squared_p_value", "—")
            p_wmv = res["permutation"].get("wmv_p_value", "—")
            log(f"      {slug:40s}  N={res['N']:3d}  K={res['K']:2d}"
                f"  η²={eta:.4f}  WMV={wmv:.4f}"
                f"  p(η²)={p_eta}  p(WMV)={p_wmv}")
            per_model[slug] = res

        by_prompt[prompt_key] = per_model

    return {
        "survey": survey_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_permutations": N_PERMUTATIONS,
        "seed": SEED,
        "models_used": model_slugs,
        "by_prompt_variant": by_prompt,
    }


# ───────────────────────────────────────────────────────────
#  Main
# ───────────────────────────────────────────────────────────

def main():
    log("=" * 60)
    log("  RQ2: η² & WMV — Established Questionnaires")
    log("=" * 60)

    if not ESTABLISHED_DIR.exists():
        log(f"ERROR: {ESTABLISHED_DIR} not found")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    found_any = False

    for survey in SURVEYS:
        log(f"{'─' * 50}")
        log(f"  Survey: {survey}")
        log(f"{'─' * 50}")

        result = compute_survey(survey)
        if result is None:
            log("  No model files found, skipping.")
            continue

        path = OUTPUT_DIR / f"{survey}.json"
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        log(f"  => Saved: {path.relative_to(ROOT)}")
        found_any = True

    if not found_any:
        log("No surveys processed.")
        sys.exit(1)

    log("=" * 60)
    log("  Done!")
    log("=" * 60)


if __name__ == "__main__":
    main()
