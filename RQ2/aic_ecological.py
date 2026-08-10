#!/usr/bin/env python3
# Note: "aic" here stands for "assessing intra-construct consistency";
# unrelated to Akaike Information Criterion.
"""
RQ2: η² & WMV — Ecological / VP

For each model × correlation type, assigns VP scenarios to constructs via
correlation tags (r ≥ 0.3) and computes two metrics **within each model**:

  η²  (eta-squared)  — SS_between / SS_total on raw total_logprob scores.
       Scale-invariant; directly comparable to established Likert results
       without z-score normalisation.
       Analytical baseline under random labels: (K−1)/(N−1).

  WMV (within-model variance) — mean within-construct sample variance
       on z-scored scenario scores.  Baseline ≈ 1.0.

A permutation test (1 000 shuffles of construct labels, group sizes held
constant) provides one-sided empirical p-values for both metrics.

Usage:
    python RQ2/aic_ecological.py

Outputs:
    results/RQ2/aic_ecological/{correlation_type}.json
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
DEFAULT_OUTPUT_DIR = ROOT / "results" / "RQ2" / "aic_ecological"

CORR_THRESHOLD = 0.3
CORRELATION_KEYS = [
    ("correlations", "pvq_values"),
    ("higher_pvq_correlations", "higher_order_values"),
    ("bfi_correlations", "bfi_traits"),
]

DISPLAY_NAME_MAP = {
    "Self_Direction": "Self-Direction",
    "Self_Enhancement": "Self-Enhancement",
    "Self_Transcendence": "Self-Transcendence",
    "Openness_to_Change": "Openness to Change",
}

N_PERMUTATIONS = 1_000
SEED = 42


def log(msg: str = ""):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def load_json(path: Path) -> dict | list:
    return json.loads(path.read_text())


def _r6(value, ndigits=6):
    return round(float(value), ndigits) if value is not None else None


def display_construct_name(name: str) -> str:
    return DISPLAY_NAME_MAP.get(name, name.replace("_", " "))


def discover_models() -> list[str]:
    if not ECOLOGICAL_DIR.exists():
        return []
    return sorted(p.stem for p in ECOLOGICAL_DIR.glob("*.json"))


# ───────────────────────────────────────────────────────────
#  VP Survey Lookup
# ───────────────────────────────────────────────────────────

def load_vp_survey() -> dict[int, dict[int, dict[str, dict[str, float]]]]:
    """portrait_id → output_id → corr_key → construct → corr_value."""
    scenarios = load_json(VP_SURVEY)
    lookup: dict[int, dict[int, dict[str, dict[str, float]]]] = {}
    for scenario in scenarios:
        pid = scenario["portrait_id"]
        outputs_map: dict[int, dict[str, dict[str, float]]] = {}
        for output in scenario["outputs"]:
            corr_map: dict[str, dict[str, float]] = {}
            for corr_key, _ in CORRELATION_KEYS:
                corr_map[corr_key] = {
                    display_construct_name(name): float(value)
                    for name, value in output.get(corr_key, [])
                }
            outputs_map[output["id"]] = corr_map
        lookup[pid] = outputs_map
    return lookup


# ───────────────────────────────────────────────────────────
#  Construct Group Building (Ecological)
# ───────────────────────────────────────────────────────────

def build_construct_groups_for_model(
    model_result: dict,
    vp_lookup: dict[int, dict[int, dict[str, dict[str, float]]]],
    corr_key: str,
) -> dict[str, list[float]]:
    """construct → [scenario_scores] for one model, one correlation type.

    Each scenario tagged with construct C contributes one score =
    mean total_logprob of outputs whose VP correlation with C ≥ threshold.
    """
    groups: dict[str, list[float]] = defaultdict(list)

    for scenario in model_result.get("results", []):
        pid = scenario["portrait_id"]
        if pid not in vp_lookup:
            continue

        scenario_corr_map = vp_lookup[pid]

        construct_logprobs: dict[str, list[float]] = defaultdict(list)
        for output in scenario.get("outputs", []):
            lp = output.get("total_logprob")
            if lp is None:
                continue
            output_id = output["output_id"]
            output_corrs = scenario_corr_map.get(output_id, {}).get(corr_key, {})
            for construct, corr_val in output_corrs.items():
                if corr_val >= CORR_THRESHOLD:
                    construct_logprobs[construct].append(float(lp))

        for construct, lps in construct_logprobs.items():
            groups[construct].append(sum(lps) / len(lps))

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
#  Main
# ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir or DEFAULT_OUTPUT_DIR
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    log("=" * 60)
    log("  RQ2: η² & WMV — Ecological / VP")
    log(f"  Correlation threshold: {CORR_THRESHOLD}")
    log(f"  Output: {output_dir.relative_to(ROOT)}")
    log("=" * 60)

    if not VP_SURVEY.exists():
        log(f"ERROR: VP survey not found at {VP_SURVEY}")
        sys.exit(1)
    if not ECOLOGICAL_DIR.exists():
        log(f"ERROR: ecological results not found at {ECOLOGICAL_DIR}")
        sys.exit(1)

    model_slugs = discover_models()
    if not model_slugs:
        log("ERROR: no ecological result files found.")
        sys.exit(1)

    log(f"  Models ({len(model_slugs)}): {model_slugs}")
    log("  Loading VP survey correlations ...")
    vp_lookup = load_vp_survey()
    log(f"  Loaded {len(vp_lookup)} VP scenarios")

    model_results = {
        slug: load_json(ECOLOGICAL_DIR / f"{slug}.json")
        for slug in model_slugs
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    for corr_key, correlation_type in CORRELATION_KEYS:
        log(f"{'─' * 50}")
        log(f"  Correlation type: {correlation_type}")
        log(f"{'─' * 50}")

        per_model: dict[str, dict] = {}
        for slug in model_slugs:
            groups = build_construct_groups_for_model(
                model_results[slug], vp_lookup, corr_key,
            )
            if len(groups) < 2:
                log(f"    {slug}: <2 constructs, skipping")
                continue

            res = compute_model_result(groups)
            eta = res["eta_squared"] or 0
            wmv = res["wmv"] or 0
            p_eta = res["permutation"].get("eta_squared_p_value", "—")
            p_wmv = res["permutation"].get("wmv_p_value", "—")
            log(f"    {slug:40s}  N={res['N']:3d}  K={res['K']:2d}"
                f"  η²={eta:.4f}  WMV={wmv:.4f}"
                f"  p(η²)={p_eta}  p(WMV)={p_wmv}")
            per_model[slug] = res

        output = {
            "correlation_type": correlation_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "correlation_threshold": CORR_THRESHOLD,
            "n_permutations": N_PERMUTATIONS,
            "seed": SEED,
            "models_used": model_slugs,
            "per_model": per_model,
        }

        path = output_dir / f"{correlation_type}.json"
        path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
        log(f"  => Saved: {path.relative_to(ROOT)}")

    log("=" * 60)
    log("  Done!")
    log("=" * 60)


if __name__ == "__main__":
    main()
