#!/usr/bin/env python3
"""
RQ1 Analysis: Established vs Ecological Profile Comparison

Three analyses comparing construct profiles from established questionnaires
(PVQ-40, PVQ-21, BFI-44, BFI-10) against ecological gen-prob profiles.

Analyses:
  1. Spearman ρ + Permutation test (+ established baselines)
  2. Rank comparison data (for bump charts / rank tables)
  3. NDCG — top-weighted ranking agreement (+ established baselines)

Usage:
    python RQ1/analysis.py
    python RQ1/analysis.py --ecological-dir RQ1/length_norm/profiles_normalized --output-path RQ1/length_norm/analysis_normalized.json

Results:
    results/RQ1/analysis.json
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from itertools import permutations as iterperms
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

ESTABLISHED_DIR = ROOT / "results" / "RQ1" / "established"
DEFAULT_ECOLOGICAL_DIR = ROOT / "results" / "RQ1" / "ecological_profiles"
DEFAULT_OUTPUT_PATH = ROOT / "results" / "RQ1" / "analysis.json"

N_PERMUTATIONS = 10000
SEED = 42

ALL_SURVEYS = ["PVQ", "PVQ21", "BFI44", "BFI10"]

COMPARISON_GROUPS = [
    {
        "name": "pvq_values",
        "established_surveys": ["PVQ", "PVQ21"],
        "established_key": "construct_averages",
        "ecological_key": "pvq_values",
    },
    {
        "name": "higher_order_values",
        "established_surveys": ["PVQ", "PVQ21"],
        "established_key": "higher_order_averages",
        "ecological_key": "higher_order_values",
    },
    {
        "name": "bfi_traits",
        "established_surveys": ["BFI44", "BFI10"],
        "established_key": "construct_averages",
        "ecological_key": "bfi_traits",
    },
]


def log(msg: str = ""):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ───────────────────────────────────────────────────────────
#  Construct Name Normalization
# ───────────────────────────────────────────────────────────

def _norm(name: str) -> str:
    return name.replace("-", "_").replace(" ", "_").lower()


def align_profiles(
    prof_a: dict[str, float],
    prof_b: dict[str, float],
) -> tuple[list[str], list[float], list[float]]:
    """Match constructs by normalized name. Returns (names, vals_a, vals_b)."""
    map_a = {_norm(k): (k, v) for k, v in prof_a.items()}
    map_b = {_norm(k): (k, v) for k, v in prof_b.items()}
    common = sorted(set(map_a) & set(map_b))
    if not common:
        return [], [], []
    names = [map_a[c][0] for c in common]
    return names, [map_a[c][1] for c in common], [map_b[c][1] for c in common]


# ───────────────────────────────────────────────────────────
#  Data Loading
# ───────────────────────────────────────────────────────────

def discover_models(ecological_dir: Path) -> list[str]:
    eco = set()
    if ecological_dir.exists():
        eco = {p.stem for p in ecological_dir.glob("*.json") if p.stem != "summary"}
    est: set[str] = set()
    for s in ALL_SURVEYS:
        d = ESTABLISHED_DIR / s
        if d.exists():
            est |= {p.stem for p in d.glob("*.json")}
    return sorted(eco & est)


def load_established(slug: str, survey: str) -> dict | None:
    p = ESTABLISHED_DIR / survey / f"{slug}.json"
    return json.loads(p.read_text()) if p.exists() else None


def load_ecological(slug: str, ecological_dir: Path) -> dict | None:
    p = ecological_dir / f"{slug}.json"
    return json.loads(p.read_text()) if p.exists() else None


def est_prof(data: dict, key: str) -> dict[str, float]:
    section = data.get(key, {})
    return section.get("combined", section.get("normal", {}))


# ───────────────────────────────────────────────────────────
#  Helpers
# ───────────────────────────────────────────────────────────

def _r6(x):
    return round(float(x), 6) if not (isinstance(x, float) and np.isnan(x)) else None


def _pearsonr(x, y):
    a, b = np.asarray(x, float), np.asarray(y, float)
    if len(a) < 3:
        return float("nan")
    da, db = a - a.mean(), b - b.mean()
    d = np.sqrt((da**2).sum() * (db**2).sum())
    return float("nan") if d == 0 else float((da * db).sum() / d)


def _rankdata_desc(x):
    """Rank descending (highest = 1), average ties."""
    arr = np.asarray(x, float)
    n = len(arr)
    order = np.argsort(-arr)
    ranks = np.empty(n, float)
    i = 0
    while i < n:
        j = i + 1
        while j < n and arr[order[j]] == arr[order[i]]:
            j += 1
        avg = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg
        i = j
    return ranks


def _spearmanr(x, y):
    return _pearsonr(_rankdata_desc(x), _rankdata_desc(y))


def _ndcg(ideal_vals, predicted_vals, k=None):
    """NDCG with exponential gain: (2^rel - 1) / log₂(i+2).

    Assigns relevance n, n-1, ..., 1 based on ideal ranking.
    Exponential gain heavily penalizes misranking top constructs.
    """
    n = len(ideal_vals)
    k = k or n
    ideal_arr = np.asarray(ideal_vals, float)
    pred_arr = np.asarray(predicted_vals, float)

    ideal_order = np.argsort(-ideal_arr)
    relevance = np.zeros(n)
    for rank_pos, item_idx in enumerate(ideal_order):
        relevance[item_idx] = n - rank_pos

    pred_order = np.argsort(-pred_arr)
    dcg = sum((2.0**relevance[pred_order[i]] - 1) / np.log2(i + 2) for i in range(min(k, n)))

    sorted_rel = sorted(relevance, reverse=True)
    idcg = sum((2.0**sorted_rel[i] - 1) / np.log2(i + 2) for i in range(min(k, n)))
    return float(dcg / idcg) if idcg > 0 else 0.0


# ───────────────────────────────────────────────────────────
#  Analysis 1 — Spearman ρ + Permutation Test + Baselines
# ───────────────────────────────────────────────────────────

def _perm_test(x, y, n_perm, rng):
    n = len(x)
    obs = _spearmanr(x, y)
    if obs is None or np.isnan(obs):
        return {"observed_rho": None, "p_two_sided": None, "p_one_sided": None}
    y_arr = np.array(y)

    if n <= 8:
        cnt_abs = cnt_pos = total = 0
        for perm in iterperms(range(n)):
            rho = _spearmanr(x, list(y_arr[list(perm)]))
            if not np.isnan(rho):
                total += 1
                if abs(rho) >= abs(obs) - 1e-12:
                    cnt_abs += 1
                if rho >= obs - 1e-12:
                    cnt_pos += 1
        p2 = cnt_abs / total if total else None
        p1 = cnt_pos / total if total else None
    else:
        null = np.array([_spearmanr(x, list(rng.permutation(y_arr))) for _ in range(n_perm)])
        p2 = float(np.mean(np.abs(null) >= abs(obs) - 1e-12))
        p1 = float(np.mean(null >= obs - 1e-12))

    return {
        "observed_rho": _r6(obs),
        "p_two_sided": _r6(p2) if p2 is not None else None,
        "p_one_sided": _r6(p1) if p1 is not None else None,
        "n_constructs": n,
    }


def run_analysis_1(models, all_est, all_eco, rng):
    log("  [Analysis 1] Spearman ρ + Permutation Test")
    results = {}
    for model in models:
        mr: dict = {}
        eco_data = all_eco[model]["gen_prob_profile"]

        for grp in COMPARISON_GROUPS:
            eco_p = eco_data.get(grp["ecological_key"], {})
            for survey in grp["established_surveys"]:
                ed = all_est.get(model, {}).get(survey)
                if not ed:
                    continue
                ep = est_prof(ed, grp["established_key"])
                names, ev, ecv = align_profiles(ep, eco_p)
                if len(names) < 3:
                    continue
                res = _perm_test(ev, ecv, N_PERMUTATIONS, rng)
                res["constructs"] = names
                res["established_values"] = [round(v, 4) for v in ev]
                res["ecological_values"] = [_r6(v) for v in ecv]
                mr[f"{survey}_{grp['name']}"] = res
                log(f"    {model:45s} | eco↔{survey:5s} {grp['name']:20s}  ρ={res['observed_rho']}")

        baselines: dict = {}
        for grp in COMPARISON_GROUPS:
            surveys = grp["established_surveys"]
            if len(surveys) < 2:
                continue
            sa, sb = surveys[0], surveys[1]
            eda, edb = all_est.get(model, {}).get(sa), all_est.get(model, {}).get(sb)
            if not eda or not edb:
                continue
            pa, pb = est_prof(eda, grp["established_key"]), est_prof(edb, grp["established_key"])
            eco_p = eco_data.get(grp["ecological_key"], {})
            na = {_norm(k): (k, v) for k, v in pa.items()}
            nb = {_norm(k): (k, v) for k, v in pb.items()}
            ne = {_norm(k): (k, v) for k, v in eco_p.items()}
            common = sorted(set(na) & set(nb) & set(ne))
            if len(common) < 3:
                continue
            va = [na[c][1] for c in common]
            vb = [nb[c][1] for c in common]
            rho = _spearmanr(va, vb)
            key = f"{sa}_vs_{sb}_{grp['name']}"
            baselines[key] = {"rho": _r6(rho), "n_constructs": len(common)}
            log(f"    {model:45s} | {sa:5s}↔{sb:5s} {grp['name']:20s}  ρ={_r6(rho)}")

        mr["established_baselines"] = baselines
        results[model] = mr
    return results


# ───────────────────────────────────────────────────────────
#  Analysis 2 — Rank Comparison
# ───────────────────────────────────────────────────────────

def _rank_list(profile: dict[str, float]) -> list[dict]:
    items = sorted(profile.items(), key=lambda kv: -kv[1])
    return [{"name": n, "value": round(v, 6), "rank": i + 1} for i, (n, v) in enumerate(items)]


def run_analysis_2(models, all_est, all_eco):
    log("  [Analysis 2] Rank Comparison")
    results = {}
    for model in models:
        mr = {}
        eco_data = all_eco[model]["gen_prob_profile"]
        for grp in COMPARISON_GROUPS:
            eco_p = eco_data.get(grp["ecological_key"], {})
            gd: dict = {"ecological": _rank_list(eco_p)}
            for survey in grp["established_surveys"]:
                ed = all_est.get(model, {}).get(survey)
                if not ed:
                    continue
                gd[survey] = _rank_list(est_prof(ed, grp["established_key"]))
            mr[grp["name"]] = gd
        results[model] = mr
    log(f"    Generated rank data for {len(results)} models")
    return results


# ───────────────────────────────────────────────────────────
#  Analysis 3 — NDCG + Baselines
# ───────────────────────────────────────────────────────────

def run_analysis_3(models, all_est, all_eco):
    """NDCG: top-weighted ranking agreement.

    Treats the established profile as the 'ideal' ranking and measures
    how well the ecological (or other established) ranking recovers it.
    Complements Spearman ρ by weighting top-ranked constructs more heavily.
    """
    log("  [Analysis 3] NDCG")
    results = {}
    for model in models:
        mr: dict = {}
        eco_data = all_eco[model]["gen_prob_profile"]

        for grp in COMPARISON_GROUPS:
            eco_p = eco_data.get(grp["ecological_key"], {})
            for survey in grp["established_surveys"]:
                ed = all_est.get(model, {}).get(survey)
                if not ed:
                    continue
                ep = est_prof(ed, grp["established_key"])
                names, ev, ecv = align_profiles(ep, eco_p)
                if len(names) < 2:
                    continue
                ndcg_full = _ndcg(ev, ecv)
                ndcg_at3 = _ndcg(ev, ecv, k=3)
                key = f"{survey}_{grp['name']}"
                mr[key] = {
                    "ndcg": _r6(ndcg_full),
                    "ndcg_at_3": _r6(ndcg_at3),
                    "n_constructs": len(names),
                    "constructs": names,
                    "established_values": [round(v, 4) for v in ev],
                    "ecological_values": [_r6(v) for v in ecv],
                }
                log(f"    {model:45s} | eco↔{survey:5s} {grp['name']:20s}  NDCG={ndcg_full:.4f}  NDCG@3={ndcg_at3:.4f}")

        baselines: dict = {}
        for grp in COMPARISON_GROUPS:
            surveys = grp["established_surveys"]
            if len(surveys) < 2:
                continue
            sa, sb = surveys[0], surveys[1]
            eda, edb = all_est.get(model, {}).get(sa), all_est.get(model, {}).get(sb)
            if not eda or not edb:
                continue
            pa, pb = est_prof(eda, grp["established_key"]), est_prof(edb, grp["established_key"])
            eco_p = eco_data.get(grp["ecological_key"], {})
            na = {_norm(k): (k, v) for k, v in pa.items()}
            nb = {_norm(k): (k, v) for k, v in pb.items()}
            ne = {_norm(k): (k, v) for k, v in eco_p.items()}
            common = sorted(set(na) & set(nb) & set(ne))
            if len(common) < 2:
                continue
            va = [na[c][1] for c in common]
            vb = [nb[c][1] for c in common]
            ndcg_full = _ndcg(va, vb)
            ndcg_at3 = _ndcg(va, vb, k=3)
            key = f"{sa}_vs_{sb}_{grp['name']}"
            baselines[key] = {"ndcg": _r6(ndcg_full), "ndcg_at_3": _r6(ndcg_at3), "n_constructs": len(common)}
            log(f"    {model:45s} | {sa:5s}↔{sb:5s} {grp['name']:20s}  NDCG={ndcg_full:.4f}  NDCG@3={ndcg_at3:.4f}")

        mr["established_baselines"] = baselines
        results[model] = mr
    return results


# ───────────────────────────────────────────────────────────
#  Analysis 4 — Within vs Cross-Method Significance Tests
# ───────────────────────────────────────────────────────────

WITHIN_CROSS_PAIRS = {
    "pvq_values": {
        "within_key": "PVQ_vs_PVQ21_pvq_values",
        "cross_keys": ["PVQ_pvq_values", "PVQ21_pvq_values"],
    },
    "bfi_traits": {
        "within_key": "BFI44_vs_BFI10_bfi_traits",
        "cross_keys": ["BFI44_bfi_traits", "BFI10_bfi_traits"],
    },
}

N_BOOTSTRAP = 10000


def _sign_flip_perm_test(diffs: np.ndarray) -> dict:
    """Exact sign-flip permutation test on paired differences.

    With n observations there are 2^n sign assignments.
    H0: the population mean difference is zero (each d_i equally
    likely to be positive or negative).
    """
    n = len(diffs)
    obs_mean = float(np.mean(diffs))
    total = 1 << n  # 2^n
    count_ge = 0
    count_abs_ge = 0
    for mask in range(total):
        flipped = np.array([d if not (mask >> i & 1) else -d for i, d in enumerate(diffs)])
        m = np.mean(flipped)
        if m >= obs_mean - 1e-12:
            count_ge += 1
        if abs(m) >= abs(obs_mean) - 1e-12:
            count_abs_ge += 1
    return {
        "observed_mean_diff": _r6(obs_mean),
        "p_one_sided": _r6(count_ge / total),
        "p_two_sided": _r6(count_abs_ge / total),
        "n_permutations": total,
    }


def _bootstrap_ci(diffs: np.ndarray, rng, n_boot: int = N_BOOTSTRAP, alpha: float = 0.05) -> dict:
    """Bootstrap percentile CI for the mean of paired differences."""
    n = len(diffs)
    boot_means = np.array([float(np.mean(rng.choice(diffs, size=n, replace=True))) for _ in range(n_boot)])
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return {
        "mean": _r6(np.mean(diffs)),
        "ci_lower": _r6(lo),
        "ci_upper": _r6(hi),
        "n_bootstrap": n_boot,
        "alpha": alpha,
    }


def _extract_metric_pairs(models, summary, metric_type):
    """Extract (within, cross) values per model from summary tables.

    Returns dict[group_name -> list of (model, within_val, cross_val)].
    """
    if metric_type == "spearman_rho":
        eco_rows = summary["spearman_rho"]["ecological_vs_established"]["per_model"]
        bl_rows = summary["spearman_rho"]["established_baselines"]["per_model"]
        val_key = None
    else:
        eco_rows = summary["ndcg"]["ecological_vs_established"]["per_model"]
        bl_rows = summary["ndcg"]["established_baselines"]["per_model"]
        val_key = None

    pairs_by_group: dict[str, list] = {}
    for grp_name, cfg in WITHIN_CROSS_PAIRS.items():
        pairs = []
        for i, model in enumerate(models):
            w = bl_rows[i].get(cfg["within_key"])
            cross_vals = [eco_rows[i].get(ck) for ck in cfg["cross_keys"]]
            cross_vals = [v for v in cross_vals if v is not None]
            if w is None or not cross_vals:
                continue
            c = float(np.mean(cross_vals))
            pairs.append({"model": model, "within": w, "cross": c})
        pairs_by_group[grp_name] = pairs
    return pairs_by_group


def run_analysis_4(models, summary, rng):
    """Paired significance tests: within-method vs cross-method.

    For each construct group and metric, computes per-model differences
    d_i = within_i - cross_i, then applies exact sign-flip permutation
    test and bootstrap CI.
    """
    log("  [Analysis 4] Within vs Cross-Method Significance Tests")
    results = {}

    for metric_type in ["spearman_rho", "ndcg"]:
        metric_results = {}
        pairs_by_group = _extract_metric_pairs(models, summary, metric_type)

        all_diffs_for_overall = []

        for grp_name, pairs in pairs_by_group.items():
            if len(pairs) < 3:
                log(f"    {metric_type:15s} | {grp_name:25s}  SKIPPED (n={len(pairs)})")
                continue

            diffs = np.array([p["within"] - p["cross"] for p in pairs])
            all_diffs_for_overall.extend(diffs.tolist())
            within_mean = float(np.mean([p["within"] for p in pairs]))
            cross_mean = float(np.mean([p["cross"] for p in pairs]))

            perm = _sign_flip_perm_test(diffs)
            boot = _bootstrap_ci(diffs, rng)

            metric_results[grp_name] = {
                "n_models": len(pairs),
                "within_mean": _r6(within_mean),
                "cross_mean": _r6(cross_mean),
                "per_model": [
                    {"model": p["model"], "within": _r6(p["within"]),
                     "cross": _r6(p["cross"]), "diff": _r6(p["within"] - p["cross"])}
                    for p in pairs
                ],
                "sign_flip_permutation_test": perm,
                "bootstrap_95ci": boot,
            }
            log(
                f"    {metric_type:15s} | {grp_name:25s}  "
                f"within={within_mean:.3f}  cross={cross_mean:.3f}  "
                f"Δ={perm['observed_mean_diff']:.3f}  "
                f"p(1s)={perm['p_one_sided']:.4f}  "
                f"95%CI=[{boot['ci_lower']:.3f}, {boot['ci_upper']:.3f}]"
            )

        if all_diffs_for_overall:
            overall_diffs = np.array(all_diffs_for_overall)
            overall_n = len(overall_diffs)
            if overall_n <= 20:
                overall_perm = _sign_flip_perm_test(overall_diffs)
            else:
                obs_mean = float(np.mean(overall_diffs))
                n_sim = 100000
                count_ge = count_abs = 0
                for _ in range(n_sim):
                    signs = rng.choice([-1, 1], size=overall_n)
                    m = np.mean(overall_diffs * signs)
                    if m >= obs_mean - 1e-12:
                        count_ge += 1
                    if abs(m) >= abs(obs_mean) - 1e-12:
                        count_abs += 1
                overall_perm = {
                    "observed_mean_diff": _r6(obs_mean),
                    "p_one_sided": _r6(count_ge / n_sim),
                    "p_two_sided": _r6(count_abs / n_sim),
                    "n_permutations": n_sim,
                }
            overall_boot = _bootstrap_ci(overall_diffs, rng)
            metric_results["overall"] = {
                "n_observations": overall_n,
                "sign_flip_permutation_test": overall_perm,
                "bootstrap_95ci": overall_boot,
            }
            log(
                f"    {metric_type:15s} | {'overall':25s}  "
                f"n={overall_n}  "
                f"Δ={overall_perm['observed_mean_diff']:.3f}  "
                f"p(1s)={overall_perm['p_one_sided']:.4f}  "
                f"95%CI=[{overall_boot['ci_lower']:.3f}, {overall_boot['ci_upper']:.3f}]"
            )

        results[metric_type] = metric_results

    return results


# ───────────────────────────────────────────────────────────
#  Summary Tables
# ───────────────────────────────────────────────────────────

def _avg_col(rows, col):
    vals = [r[col] for r in rows if r.get(col) is not None]
    return _r6(np.mean(vals)) if vals else None


def build_summary_tables(models, a1, a3):

    # ─── Spearman ρ ───
    rho_rows, bl_rows = [], []
    for model in models:
        row = {"model": model}
        rhos = []
        for k, v in a1.get(model, {}).items():
            if k == "established_baselines":
                continue
            r = v.get("observed_rho")
            row[k] = r
            if r is not None:
                rhos.append(r)
        row["average_rho"] = _r6(np.mean(rhos)) if rhos else None
        rho_rows.append(row)

        brow = {"model": model}
        for bk, bv in a1.get(model, {}).get("established_baselines", {}).items():
            brow[bk] = bv.get("rho")
        bl_rows.append(brow)

    eco_keys = sorted({k for r in rho_rows for k in r if k not in ("model", "average_rho")})
    bl_keys = sorted({k for r in bl_rows for k in r if k != "model"})

    # ─── NDCG ───
    ndcg_rows, ndcg_bl_rows = [], []
    for model in models:
        row = {"model": model}
        vals = []
        for k, v in a3.get(model, {}).items():
            if k == "established_baselines":
                continue
            n = v.get("ndcg")
            row[k] = n
            if n is not None:
                vals.append(n)
        row["average_ndcg"] = _r6(np.mean(vals)) if vals else None
        ndcg_rows.append(row)

        brow = {"model": model}
        for bk, bv in a3.get(model, {}).get("established_baselines", {}).items():
            brow[bk] = bv.get("ndcg")
        ndcg_bl_rows.append(brow)

    ndcg_keys = sorted({k for r in ndcg_rows for k in r if k not in ("model", "average_ndcg")})
    ndcg_bl_keys = sorted({k for r in ndcg_bl_rows for k in r if k != "model"})

    return {
        "spearman_rho": {
            "ecological_vs_established": {
                "per_model": rho_rows,
                "per_comparison_avg": {k: _avg_col(rho_rows, k) for k in eco_keys},
            },
            "established_baselines": {
                "per_model": bl_rows,
                "per_comparison_avg": {k: _avg_col(bl_rows, k) for k in bl_keys},
            },
        },
        "ndcg": {
            "ecological_vs_established": {
                "per_model": ndcg_rows,
                "per_comparison_avg": {k: _avg_col(ndcg_rows, k) for k in ndcg_keys},
            },
            "established_baselines": {
                "per_model": ndcg_bl_rows,
                "per_comparison_avg": {k: _avg_col(ndcg_bl_rows, k) for k in ndcg_bl_keys},
            },
        },
    }


# ───────────────────────────────────────────────────────────
#  Main
# ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ecological-dir",
        type=Path,
        default=DEFAULT_ECOLOGICAL_DIR,
        help="Directory containing ecological profile JSON files.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to save the analysis JSON output.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.ecological_dir.is_absolute():
        args.ecological_dir = ROOT / args.ecological_dir
    if not args.output_path.is_absolute():
        args.output_path = ROOT / args.output_path
    log("=" * 60)
    log("  RQ1 Analysis: Established vs Ecological Comparison")
    log(f"  Ecological dir: {args.ecological_dir.relative_to(ROOT)}")
    log(f"  Output path: {args.output_path.relative_to(ROOT)}")
    log("=" * 60)

    models = discover_models(args.ecological_dir)
    if not models:
        log("ERROR: No models found.")
        sys.exit(1)
    log(f"  Models ({len(models)}): {models}")

    log("  Loading data ...")
    all_est: dict[str, dict[str, dict]] = {}
    all_eco: dict[str, dict] = {}
    for model in models:
        eco = load_ecological(model, args.ecological_dir)
        if eco:
            all_eco[model] = eco
        all_est[model] = {}
        for survey in ALL_SURVEYS:
            ed = load_established(model, survey)
            if ed:
                all_est[model][survey] = ed

    rng = np.random.default_rng(SEED)

    log("")
    a1 = run_analysis_1(models, all_est, all_eco, rng)
    log("")
    a2 = run_analysis_2(models, all_est, all_eco)
    log("")
    a3 = run_analysis_3(models, all_est, all_eco)

    log("\n  Building summary tables ...")
    summary = build_summary_tables(models, a1, a3)

    log("")
    a4 = run_analysis_4(models, summary, rng)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metadata": {
            "n_permutations": N_PERMUTATIONS,
            "seed": SEED,
            "models": models,
            "ecological_dir": str(args.ecological_dir),
            "ecological_correlation_threshold": 0.3,
            "established_surveys": ALL_SURVEYS,
            "comparison_groups": [g["name"] for g in COMPARISON_GROUPS],
            "notes": {
                "spearman_rho": (
                    "Rank correlation (all positions weighted equally). "
                    "established_baselines = ρ between established surveys as reference."
                ),
                "permutation_test": (
                    "NOTE: n=10 needs ρ≥0.648 for p<0.05; n=5 needs ρ≈1.0. "
                    "p-values are underpowered; compare ρ to baselines instead."
                ),
                "ndcg": (
                    "Normalized Discounted Cumulative Gain — top-weighted ranking agreement. "
                    "Established ranking = ideal; relevance n, n-1, ..., 1. "
                    "NDCG@3 focuses on top-3 constructs only. "
                    "Range [0, 1], 1 = perfect recovery of established ranking."
                ),
                "within_vs_cross_test": (
                    "Paired significance test: d_i = within_rho_i - cross_rho_i (or NDCG). "
                    "Exact sign-flip permutation test (2^n assignments) + bootstrap 95% CI "
                    f"({N_BOOTSTRAP} resamples). Tests whether within-method agreement "
                    "systematically exceeds cross-method (ecological) agreement."
                ),
            },
        },
        "analysis_1_spearman_rho": a1,
        "analysis_2_rank_comparison": a2,
        "analysis_3_ndcg": a3,
        "analysis_4_within_vs_cross": a4,
        "summary_tables": summary,
    }

    args.output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    log(f"\n  => Saved: {args.output_path.relative_to(ROOT)}")

    # ── Print combined summary ──
    _print_summary(models, summary)

    log("\n" + "=" * 60)
    log("  Done!")
    log("=" * 60)


def _print_summary(models, summary):
    log("\n" + "=" * 70)
    log("  COMBINED TABLE: Spearman ρ / NDCG  (baseline | ecological)")
    log("=" * 70)

    rho_ee = summary["spearman_rho"]["ecological_vs_established"]
    rho_bl = summary["spearman_rho"]["established_baselines"]
    ndcg_ee = summary["ndcg"]["ecological_vs_established"]
    ndcg_bl = summary["ndcg"]["established_baselines"]

    # PVQ panel
    log("\n  Panel A: Schwartz Values (10 constructs)")
    log(f"  {'Model':30s}  {'PVQ↔PVQ21':>10s}  {'eco↔PVQ':>10s}  {'eco↔PVQ21':>10s}  │  {'PVQ↔PVQ21':>10s}  {'eco↔PVQ':>10s}  {'eco↔PVQ21':>10s}")
    log(f"  {'':30s}  {'ρ':>10s}  {'ρ':>10s}  {'ρ':>10s}  │  {'NDCG':>10s}  {'NDCG':>10s}  {'NDCG':>10s}")
    log(f"  {'─' * 30}──{'─' * 10}──{'─' * 10}──{'─' * 10}──┼──{'─' * 10}──{'─' * 10}──{'─' * 10}")

    for i, model in enumerate(models):
        rr = rho_ee["per_model"][i]
        rb = rho_bl["per_model"][i]
        nr = ndcg_ee["per_model"][i]
        nb = ndcg_bl["per_model"][i]

        def _f(v):
            return f"{v:>10.3f}" if v is not None else f"{'—':>10s}"

        short = model[:30]
        log(
            f"  {short:30s}  "
            f"{_f(rb.get('PVQ_vs_PVQ21_pvq_values'))}  "
            f"{_f(rr.get('PVQ_pvq_values'))}  "
            f"{_f(rr.get('PVQ21_pvq_values'))}  │  "
            f"{_f(nb.get('PVQ_vs_PVQ21_pvq_values'))}  "
            f"{_f(nr.get('PVQ_pvq_values'))}  "
            f"{_f(nr.get('PVQ21_pvq_values'))}"
        )

    log(f"  {'─' * 30}──{'─' * 10}──{'─' * 10}──{'─' * 10}──┼──{'─' * 10}──{'─' * 10}──{'─' * 10}")

    def _fa(d, k):
        v = d.get(k)
        return f"{v:>10.3f}" if v is not None else f"{'—':>10s}"

    log(
        f"  {'Average':30s}  "
        f"{_fa(rho_bl['per_comparison_avg'], 'PVQ_vs_PVQ21_pvq_values')}  "
        f"{_fa(rho_ee['per_comparison_avg'], 'PVQ_pvq_values')}  "
        f"{_fa(rho_ee['per_comparison_avg'], 'PVQ21_pvq_values')}  │  "
        f"{_fa(ndcg_bl['per_comparison_avg'], 'PVQ_vs_PVQ21_pvq_values')}  "
        f"{_fa(ndcg_ee['per_comparison_avg'], 'PVQ_pvq_values')}  "
        f"{_fa(ndcg_ee['per_comparison_avg'], 'PVQ21_pvq_values')}"
    )

    # BFI panel
    log("\n  Panel B: Big Five Traits (5 constructs)")
    log(f"  {'Model':30s}  {'BFI44↔10':>10s}  {'eco↔BFI44':>10s}  {'eco↔BFI10':>10s}  │  {'BFI44↔10':>10s}  {'eco↔BFI44':>10s}  {'eco↔BFI10':>10s}")
    log(f"  {'':30s}  {'ρ':>10s}  {'ρ':>10s}  {'ρ':>10s}  │  {'NDCG':>10s}  {'NDCG':>10s}  {'NDCG':>10s}")
    log(f"  {'─' * 30}──{'─' * 10}──{'─' * 10}──{'─' * 10}──┼──{'─' * 10}──{'─' * 10}──{'─' * 10}")

    for i, model in enumerate(models):
        rr = rho_ee["per_model"][i]
        rb = rho_bl["per_model"][i]
        nr = ndcg_ee["per_model"][i]
        nb = ndcg_bl["per_model"][i]

        def _f(v):
            return f"{v:>10.3f}" if v is not None else f"{'—':>10s}"

        short = model[:30]
        log(
            f"  {short:30s}  "
            f"{_f(rb.get('BFI44_vs_BFI10_bfi_traits'))}  "
            f"{_f(rr.get('BFI44_bfi_traits'))}  "
            f"{_f(rr.get('BFI10_bfi_traits'))}  │  "
            f"{_f(nb.get('BFI44_vs_BFI10_bfi_traits'))}  "
            f"{_f(nr.get('BFI44_bfi_traits'))}  "
            f"{_f(nr.get('BFI10_bfi_traits'))}"
        )

    log(f"  {'─' * 30}──{'─' * 10}──{'─' * 10}──{'─' * 10}──┼──{'─' * 10}──{'─' * 10}──{'─' * 10}")
    log(
        f"  {'Average':30s}  "
        f"{_fa(rho_bl['per_comparison_avg'], 'BFI44_vs_BFI10_bfi_traits')}  "
        f"{_fa(rho_ee['per_comparison_avg'], 'BFI44_bfi_traits')}  "
        f"{_fa(rho_ee['per_comparison_avg'], 'BFI10_bfi_traits')}  │  "
        f"{_fa(ndcg_bl['per_comparison_avg'], 'BFI44_vs_BFI10_bfi_traits')}  "
        f"{_fa(ndcg_ee['per_comparison_avg'], 'BFI44_bfi_traits')}  "
        f"{_fa(ndcg_ee['per_comparison_avg'], 'BFI10_bfi_traits')}"
    )


if __name__ == "__main__":
    main()
