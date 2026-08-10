#!/usr/bin/env python3
"""
Analysis: eta^2 / WMV / permutation on cue-reduced established
results, plus comparison tables and profile-shift (Spearman) analysis.

Adapted from RQ2/aic_established.py.

Produces:
  results/analysis/aic_cueless_{survey}.json     (full per-model eta2/WMV/perm)
  results/analysis/comparison_eta_wmv.csv        (orig vs cueless vs genprob)
  results/analysis/profile_spearman.csv          (profile shift toward genprob)
  results/analysis/headline.json                 (aggregate headline numbers)

Run (from the repository root, no GPU needed):
    python RQ3/cue_reduction/scripts/analyze_cueless.py
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

CUE_DIR = Path(__file__).resolve().parent.parent
ROOT = CUE_DIR.parent.parent  # Official/ repo root
CUELESS_RES = CUE_DIR / "results" / "established_cueless"
OUT_DIR = CUE_DIR / "results" / "analysis"

ORIG_AIC = ROOT / "results" / "RQ2" / "aic_established"          # original eta2/WMV
ORIG_EST = ROOT / "results" / "RQ1" / "established"              # original profiles
GENPROB_AIC = ROOT / "results" / "RQ2" / "aic_ecological"        # genprob eta2/WMV
GENPROB_PROF = ROOT / "results" / "RQ1" / "ecological_profiles"  # genprob profiles

SURVEYS = ["PVQ", "BFI44"]
GENPROB_SPACE = {"PVQ": "pvq_values", "BFI44": "bfi_traits"}
PROMPT_VARIANTS = ["normal", "reversed"]
N_PERMUTATIONS = 1000
SEED = 42


def load_json(p: Path):
    return json.loads(p.read_text())


def norm_construct(name: str) -> str:
    """Genprob keys use underscores (Self_Direction); established uses hyphens."""
    return name.replace("_", "-")


# ── eta2 / WMV / permutation (from aic_established.py) ────────────────────────
def _build_combined(result):
    combined = {}
    responses = result.get("responses", {})
    ids = set()
    for pk in PROMPT_VARIANTS:
        ids.update(responses.get(pk, {}).keys())
    for iid in sorted(ids, key=lambda x: int(x) if str(x).isdigit() else str(x)):
        entries = [responses.get(pk, {}).get(iid) for pk in PROMPT_VARIANTS
                   if responses.get(pk, {}).get(iid) is not None]
        if not entries:
            continue
        vals = [e.get("final_score") for e in entries if e.get("final_score") is not None]
        combined[iid] = {"construct": entries[0].get("construct"),
                         "final_score": sum(vals) / len(vals) if vals else None}
    return combined


def build_groups(result, prompt_key):
    items = _build_combined(result) if prompt_key == "combined" \
        else result.get("responses", {}).get(prompt_key, {})
    groups = defaultdict(list)
    for entry in items.values():
        c, s = entry.get("construct"), entry.get("final_score")
        if c and s is not None:
            groups[c].append(float(s))
    return dict(sorted(groups.items()))


def eta_squared(groups):
    vals = [v for sc in groups.values() for v in sc]
    N, K = len(vals), len(groups)
    if N < 2 or K < 2:
        return None, None
    gm = sum(vals) / N
    sst = sum((x - gm) ** 2 for x in vals)
    if sst == 0:
        return None, (K - 1) / (N - 1)
    ssb = sum(len(sc) * (sum(sc) / len(sc) - gm) ** 2 for sc in groups.values() if sc)
    return ssb / sst, (K - 1) / (N - 1)


def _eta_fast(groups):
    e, _ = eta_squared(groups)
    return e


def _zscore(groups):
    vals = [v for sc in groups.values() for v in sc]
    mu = sum(vals) / len(vals)
    std = (sum((x - mu) ** 2 for x in vals) / len(vals)) ** 0.5
    if std == 0:
        return {c: [0.0] * len(sc) for c, sc in groups.items()}
    return {c: [(v - mu) / std for v in sc] for c, sc in groups.items()}


def _wmv(zg):
    kv, s = 0, 0.0
    for zs in zg.values():
        k = len(zs)
        if k < 2:
            continue
        zb = sum(zs) / k
        s += sum((z - zb) ** 2 for z in zs) / (k - 1)
        kv += 1
    return s / kv if kv else None


def wmv(groups):
    return _wmv(_zscore(groups))


def permutation(groups, n_perm=N_PERMUTATIONS, seed=SEED):
    obs_eta = _eta_fast(groups)
    zg = _zscore(groups)
    obs_wmv = _wmv(zg)
    order = sorted(groups)
    sizes = [(c, len(groups[c])) for c in order]
    raw = [v for c in order for v in groups[c]]
    z = [v for c in order for v in zg[c]]
    rng = np.random.default_rng(seed)
    pe, pw = [], []
    for _ in range(n_perm):
        perm = rng.permutation(len(raw))
        fr, fz, off = {}, {}, 0
        for c, sz in sizes:
            idx = perm[off:off + sz]
            fr[c] = [raw[i] for i in idx]
            fz[c] = [z[i] for i in idx]
            off += sz
        e = _eta_fast(fr)
        w = _wmv(fz)
        if e is not None:
            pe.append(e)
        if w is not None:
            pw.append(w)
    out = {}
    if pe and obs_eta is not None:
        a = np.array(pe)
        out["eta_squared_p_value"] = float(np.mean(a >= obs_eta - 1e-12))
        out["eta_squared_null_mean"] = float(a.mean())
    if pw and obs_wmv is not None:
        a = np.array(pw)
        out["wmv_p_value"] = float(np.mean(a <= obs_wmv + 1e-12))
        out["wmv_null_mean"] = float(a.mean())
    return out


def compute_cueless_aic():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_survey = {}
    for survey in SURVEYS:
        sdir = CUELESS_RES / survey
        if not sdir.exists():
            print(f"  [skip] {survey}: no cueless results yet ({sdir})")
            continue
        model_paths = sorted(sdir.glob("*.json"))
        if not model_paths:
            print(f"  [skip] {survey}: no model files")
            continue
        by_prompt = {}
        for pk in [*PROMPT_VARIANTS, "combined"]:
            per_model = {}
            for mp in model_paths:
                res = load_json(mp)
                groups = build_groups(res, pk)
                if len(groups) < 2:
                    continue
                e, base = eta_squared(groups)
                w = wmv(groups)
                perm = permutation(groups)
                per_model[mp.stem] = {
                    "eta_squared": e, "analytical_baseline_eta_squared": base,
                    "wmv": w,
                    "N": sum(len(v) for v in groups.values()), "K": len(groups),
                    "permutation": perm,
                }
            by_prompt[pk] = per_model
        per_survey[survey] = by_prompt
        (OUT_DIR / f"aic_cueless_{survey}.json").write_text(
            json.dumps({"survey": survey, "by_prompt_variant": by_prompt},
                       indent=2, ensure_ascii=False))
    return per_survey


def load_orig_aic(survey):
    p = ORIG_AIC / f"{survey}.json"
    if not p.exists():
        return {}
    d = load_json(p)
    return d["by_prompt_variant"].get("combined", {})


def load_genprob_aic(survey):
    p = GENPROB_AIC / f"{GENPROB_SPACE[survey]}.json"
    if not p.exists():
        return {}
    return load_json(p).get("per_model", {})


def build_comparison(cueless_aic):
    rows = []
    for survey in SURVEYS:
        if survey not in cueless_aic:
            continue
        cue = cueless_aic[survey].get("combined", {})
        orig = load_orig_aic(survey)
        gen = load_genprob_aic(survey)
        models = sorted(set(cue) | set(orig) | set(gen))
        for m in models:
            c = cue.get(m, {})
            o = orig.get(m, {})
            g = gen.get(m, {})
            rows.append({
                "survey": survey, "model": m,
                "eta2_orig": round(o.get("eta_squared"), 4) if o.get("eta_squared") is not None else "",
                "eta2_cueless": round(c.get("eta_squared"), 4) if c.get("eta_squared") is not None else "",
                "eta2_genprob": round(g.get("eta_squared"), 4) if g.get("eta_squared") is not None else "",
                "eta2_baseline": round(c.get("analytical_baseline_eta_squared"), 4) if c.get("analytical_baseline_eta_squared") is not None else "",
                "wmv_orig": round(o.get("wmv"), 4) if o.get("wmv") is not None else "",
                "wmv_cueless": round(c.get("wmv"), 4) if c.get("wmv") is not None else "",
                "wmv_genprob": round(g.get("wmv"), 4) if g.get("wmv") is not None else "",
                "p_eta_orig": o.get("permutation", {}).get("eta_squared_p_value", ""),
                "p_eta_cueless": round(c.get("permutation", {}).get("eta_squared_p_value"), 4) if c.get("permutation", {}).get("eta_squared_p_value") is not None else "",
                "p_eta_genprob": g.get("permutation", {}).get("eta_squared_p_value", ""),
            })
    if rows:
        with (OUT_DIR / "comparison_eta_wmv.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    return rows


# ── Profile shift toward genprob ─────────────────────────────────────────────
def profile_from_est(path, survey):
    """construct -> combined score, from an established-style result JSON."""
    d = load_json(path)
    ca = d.get("construct_averages", {}).get("combined", {})
    return {norm_construct(k): v for k, v in ca.items() if v is not None}


def profile_from_genprob(model_slug, survey):
    p = GENPROB_PROF / f"{model_slug}.json"
    if not p.exists():
        return {}
    d = load_json(p)
    prof = d.get("gen_prob_profile", {}).get(GENPROB_SPACE[survey], {})
    return {norm_construct(k): v for k, v in prof.items() if v is not None}


def spearman(a: dict, b: dict):
    keys = sorted(set(a) & set(b))
    if len(keys) < 3:
        return None, len(keys)
    va = [a[k] for k in keys]
    vb = [b[k] for k in keys]
    rho, _ = spearmanr(va, vb)
    return float(rho) if rho == rho else None, len(keys)


def build_profile_shift():
    rows = []
    for survey in SURVEYS:
        cdir = CUELESS_RES / survey
        odir = ORIG_EST / survey
        if not cdir.exists():
            continue
        for cp in sorted(cdir.glob("*.json")):
            slug = cp.stem
            cue_prof = profile_from_est(cp, survey)
            orig_prof = profile_from_est(odir / f"{slug}.json", survey) if (odir / f"{slug}.json").exists() else {}
            gen_prof = profile_from_genprob(slug, survey)
            r_cue_orig, n1 = spearman(cue_prof, orig_prof)
            r_cue_gen, n2 = spearman(cue_prof, gen_prof)
            r_orig_gen, n3 = spearman(orig_prof, gen_prof)
            rows.append({
                "survey": survey, "model": slug,
                "spearman_cueless_vs_orig": round(r_cue_orig, 4) if r_cue_orig is not None else "",
                "spearman_cueless_vs_genprob": round(r_cue_gen, 4) if r_cue_gen is not None else "",
                "spearman_orig_vs_genprob": round(r_orig_gen, 4) if r_orig_gen is not None else "",
                "n_constructs": n1,
            })
    if rows:
        with (OUT_DIR / "profile_spearman.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    return rows


def _mean(vals):
    vals = [v for v in vals if isinstance(v, (int, float))]
    return round(sum(vals) / len(vals), 4) if vals else None


def main():
    print("=== Cue-reduction analysis: eta2/WMV on cueless established ===")
    cueless_aic = compute_cueless_aic()
    if not cueless_aic:
        print("No cueless established results found yet. "
              "Run the cue-reduction experiment scripts first; this script is ready to re-run.")
        return

    comp = build_comparison(cueless_aic)
    prof = build_profile_shift()

    # headline aggregates — averaged ONLY over models that have cueless data
    # (matched-model comparison; avoids averaging orig over 8 but cueless over fewer).
    headline = {"per_survey": {}}
    for survey in SURVEYS:
        crows = [r for r in comp if r["survey"] == survey
                 and isinstance(r["eta2_cueless"], (int, float))]
        if not crows:
            continue
        headline["per_survey"][survey] = {
            "models_matched": sorted(r["model"] for r in crows),
            "n_models_matched": len(crows),
            "mean_eta2_orig": _mean([r["eta2_orig"] for r in crows]),
            "mean_eta2_cueless": _mean([r["eta2_cueless"] for r in crows]),
            "mean_eta2_genprob": _mean([r["eta2_genprob"] for r in crows]),
            "mean_wmv_orig": _mean([r["wmv_orig"] for r in crows]),
            "mean_wmv_cueless": _mean([r["wmv_cueless"] for r in crows]),
            "mean_wmv_genprob": _mean([r["wmv_genprob"] for r in crows]),
            "n_models_p_eta_cueless_sig": sum(
                1 for r in crows if isinstance(r["p_eta_cueless"], (int, float)) and r["p_eta_cueless"] < 0.05),
            "n_models_p_eta_orig_sig": sum(
                1 for r in crows if isinstance(r["p_eta_orig"], (int, float)) and r["p_eta_orig"] < 0.05),
        }
        matched = set(r["model"] for r in crows)
        prows = [r for r in prof if r["survey"] == survey and r["model"] in matched]
        headline["per_survey"][survey]["mean_spearman_cueless_vs_orig"] = _mean(
            [r["spearman_cueless_vs_orig"] for r in prows])
        headline["per_survey"][survey]["mean_spearman_cueless_vs_genprob"] = _mean(
            [r["spearman_cueless_vs_genprob"] for r in prows])
        headline["per_survey"][survey]["mean_spearman_orig_vs_genprob"] = _mean(
            [r["spearman_orig_vs_genprob"] for r in prows])

    (OUT_DIR / "headline.json").write_text(json.dumps(headline, indent=2))
    print(json.dumps(headline, indent=2))
    print(f"\nWrote analysis to {OUT_DIR}")


if __name__ == "__main__":
    main()
