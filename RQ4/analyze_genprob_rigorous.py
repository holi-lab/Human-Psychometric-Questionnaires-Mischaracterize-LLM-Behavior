#!/usr/bin/env python3
"""
Rigorous analysis: Is the VP gen-prob delta real or a scale artifact?

Tests:
  1. Effect size: delta / baseline_spread for each source (apples-to-apples)
  2. Binomial test: is direction match better than chance?
  3. Model consistency: do models agree on VP delta direction?
  4. Bootstrap CI for cosine similarity
  5. Per-model VP delta reliability check

Usage:
    python RQ4/analyze_genprob_rigorous.py
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent

VALUES = [
    "Achievement", "Benevolence", "Conformity", "Hedonism", "Power",
    "Security", "Self-Direction", "Stimulation", "Tradition", "Universalism",
]

CONDITIONS = [
    "Gender_A", "Gender_B", "Age_A", "Age_B",
    "Political_A", "Political_B", "Education_A", "Education_B",
]
COND_LABEL = {
    "Gender_A": "Male", "Gender_B": "Female",
    "Age_A": "Young", "Age_B": "Old",
    "Political_A": "Right", "Political_B": "Left",
    "Education_A": "BelowUni", "Education_B": "Uni+",
}

SURVEYS = ["PVQ", "PVQ21", "VP"]


def load_data():
    return json.load(open(ROOT / "results" / "RQ4" / "analysis_circumstance_vs_human.json"))


def to_vec(d):
    return np.array([d.get(v, 0.0) for v in VALUES])


def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def direction_match_count(a, b):
    return sum(1 for x, y in zip(a, b) if (x > 0 and y > 0) or (x < 0 and y < 0))


def main():
    d = load_data()
    models = list(d["models"].keys())

    # ================================================================
    # TEST 1: Effect Size — delta / baseline_range (relative change)
    # ================================================================
    print("=" * 85)
    print("TEST 1: Effect Size — Is VP delta proportionally larger?")
    print("  ES = |delta_i| / IQR(baseline_values)")
    print("  If ES(VP) ≈ ES(Likert), the big numbers are just scale.")
    print("=" * 85)

    es_by_survey = defaultdict(list)

    for model in models:
        for survey in SURVEYS:
            sdata = d["models"][model].get(survey, {})
            baseline = sdata.get("baseline")
            if not baseline:
                continue
            bl_vec = to_vec(baseline)
            bl_iqr = np.percentile(bl_vec, 75) - np.percentile(bl_vec, 25)
            bl_std = np.std(bl_vec)
            bl_range = bl_vec.max() - bl_vec.min()

            for cond in CONDITIONS:
                delta = sdata.get("conditions", {}).get(cond, {}).get("delta")
                if not delta:
                    continue
                d_vec = to_vec(delta)
                if bl_std > 0:
                    es_by_survey[survey].extend(np.abs(d_vec) / bl_std)

    print(f"\n{'Survey':<8} {'Mean ES':>10} {'Median ES':>10} {'Std ES':>10} {'N':>6}")
    print("-" * 50)
    for survey in SURVEYS:
        vals = es_by_survey[survey]
        if vals:
            arr = np.array(vals)
            print(f"{survey:<8} {arr.mean():>10.4f} {np.median(arr):>10.4f} {arr.std():>10.4f} {len(arr):>6}")

    # Human effect size
    h_overall = to_vec(d["human"]["overall_profile"])
    h_std = np.std(h_overall)
    h_es = []
    for cond in CONDITIONS:
        hd = to_vec(d["human"]["conditions"][cond]["delta"])
        h_es.extend(np.abs(hd) / h_std if h_std > 0 else np.abs(hd))
    h_arr = np.array(h_es)
    print(f"{'Human':<8} {h_arr.mean():>10.4f} {np.median(h_arr):>10.4f} {h_arr.std():>10.4f} {len(h_arr):>6}")

    # ================================================================
    # TEST 2: Binomial test on direction match
    # ================================================================
    print("\n" + "=" * 85)
    print("TEST 2: Binomial test — Is direction match better than chance (50%)?")
    print("=" * 85)

    for survey in SURVEYS:
        print(f"\n--- {survey} vs Human ---")
        total_match = 0
        total_n = 0

        for cond in CONDITIONS:
            h_delta = to_vec(d["human"]["conditions"][cond]["delta"])

            per_model_deltas = []
            for model in models:
                cd = d["models"][model].get(survey, {}).get("conditions", {}).get(cond, {}).get("delta")
                if cd:
                    per_model_deltas.append(to_vec(cd))

            if not per_model_deltas:
                continue

            avg_delta = np.mean(per_model_deltas, axis=0)
            dm = direction_match_count(h_delta, avg_delta)
            total_match += dm
            total_n += 10
            binom_p = stats.binomtest(dm, 10, 0.5, alternative="greater").pvalue
            sig = "***" if binom_p < 0.001 else "**" if binom_p < 0.01 else "*" if binom_p < 0.05 else ""
            print(f"  {COND_LABEL[cond]:<12}: {dm}/10  binom p={binom_p:.4f} {sig}")

        if total_n == 0:
            raise SystemExit(
                "No VP condition deltas found; regenerate the analysis JSON "
                "after running RQ4/ecological_circumstance.py."
            )
        binom_p_all = stats.binomtest(total_match, total_n, 0.5, alternative="greater").pvalue
        print(f"  {'OVERALL':<12}: {total_match}/{total_n}  binom p={binom_p_all:.6f} "
              f"{'***' if binom_p_all < 0.001 else '**' if binom_p_all < 0.01 else '*' if binom_p_all < 0.05 else 'ns'}")

    # ================================================================
    # TEST 3: Model consistency — Do models agree on VP delta direction?
    # ================================================================
    print("\n" + "=" * 85)
    print("TEST 3: Model Consistency — Do models agree on delta direction?")
    print("  For each (condition, value), how many models have positive delta?")
    print("  If 7/7 or 0/7 → strong consensus. If 3-4/7 → no consensus.")
    print("=" * 85)

    for survey in SURVEYS:
        consensus_scores = []
        total_dims = 0
        strong_consensus = 0

        for cond in CONDITIONS:
            signs = []
            for model in models:
                cd = d["models"][model].get(survey, {}).get("conditions", {}).get(cond, {}).get("delta")
                if cd:
                    signs.append(to_vec(cd))

            if not signs:
                continue

            sign_mat = np.array(signs)  # (n_models, 10)
            n_models = sign_mat.shape[0]

            for j in range(10):
                n_positive = np.sum(sign_mat[:, j] > 0)
                consensus = max(n_positive, n_models - n_positive) / n_models
                consensus_scores.append(consensus)
                total_dims += 1
                if consensus >= 6/7:
                    strong_consensus += 1

        if consensus_scores:
            cs_arr = np.array(consensus_scores)
            print(f"\n  {survey}:")
            print(f"    Mean consensus: {cs_arr.mean():.3f} (1.0 = perfect, 0.5 = random)")
            print(f"    Strong consensus (>=6/7): {strong_consensus}/{total_dims} "
                  f"({strong_consensus/total_dims*100:.1f}%)")
            print(f"    Weak consensus (<5/7):    {sum(1 for c in cs_arr if c < 5/7)}/{total_dims} "
                  f"({sum(1 for c in cs_arr if c < 5/7)/total_dims*100:.1f}%)")

    # ================================================================
    # TEST 4: Bootstrap CI for cosine similarity
    # ================================================================
    print("\n" + "=" * 85)
    print("TEST 4: Bootstrap 95% CI for Cosine Similarity (Human vs LLM Avg)")
    print("=" * 85)

    rng = np.random.default_rng(42)
    n_boot = 10000

    for survey in SURVEYS:
        all_h = []
        all_l = []
        for cond in CONDITIONS:
            h_delta = to_vec(d["human"]["conditions"][cond]["delta"])
            per_model = []
            for model in models:
                cd = d["models"][model].get(survey, {}).get("conditions", {}).get(cond, {}).get("delta")
                if cd:
                    per_model.append(to_vec(cd))
            if per_model:
                avg = np.mean(per_model, axis=0)
                all_h.append(h_delta)
                all_l.append(avg)

        if not all_h:
            continue

        all_h = np.array(all_h)  # (n_conds, 10)
        all_l = np.array(all_l)

        observed_cos = cosine_sim(all_h.flatten(), all_l.flatten())

        boot_cos = []
        for _ in range(n_boot):
            idx = rng.choice(len(all_h), size=len(all_h), replace=True)
            h_b = all_h[idx].flatten()
            l_b = all_l[idx].flatten()
            boot_cos.append(cosine_sim(h_b, l_b))

        boot_cos = np.array(boot_cos)
        ci_lo = np.percentile(boot_cos, 2.5)
        ci_hi = np.percentile(boot_cos, 97.5)
        pct_below_zero = np.mean(boot_cos < 0) * 100

        print(f"\n  {survey}:")
        print(f"    Observed cosine: {observed_cos:.4f}")
        print(f"    95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
        print(f"    P(cos < 0): {pct_below_zero:.1f}%")

    # ================================================================
    # TEST 5: Likert vs VP — Per-model cross-survey agreement
    # ================================================================
    print("\n" + "=" * 85)
    print("TEST 5: Per-model Likert-VP agreement")
    print("  For each model: does PVQ delta direction match VP delta direction?")
    print("=" * 85)

    for model in models:
        matches = 0
        total = 0
        cos_vals = []

        for cond in CONDITIONS:
            pvq_d = d["models"][model].get("PVQ", {}).get("conditions", {}).get(cond, {}).get("delta")
            vp_d = d["models"][model].get("VP", {}).get("conditions", {}).get(cond, {}).get("delta")
            if pvq_d and vp_d:
                pvq_vec = to_vec(pvq_d)
                vp_vec = to_vec(vp_d)
                dm = direction_match_count(pvq_vec, vp_vec)
                matches += dm
                total += 10
                cos_vals.append(cosine_sim(pvq_vec, vp_vec))

        if total > 0:
            binom_p = stats.binomtest(matches, total, 0.5, alternative="greater").pvalue
            avg_cos = np.mean(cos_vals)
            sig = "***" if binom_p < 0.001 else "**" if binom_p < 0.01 else "*" if binom_p < 0.05 else "ns"
            print(f"  {model:<45}: dir={matches}/{total} ({matches/total*100:.0f}%) "
                  f"p={binom_p:.4f} {sig:3}  cos={avg_cos:+.3f}")

    # ================================================================
    # TEST 6: Permutation test — is VP-Human cosine different from zero?
    # ================================================================
    print("\n" + "=" * 85)
    print("TEST 6: Permutation test — is observed cosine(Human, LLM) ≠ 0?")
    print("  Null: randomly shuffle value labels within each condition")
    print("=" * 85)

    n_perm = 10000
    rng = np.random.default_rng(123)

    for survey in SURVEYS:
        all_h_vecs = []
        all_l_vecs = []
        for cond in CONDITIONS:
            h_delta = to_vec(d["human"]["conditions"][cond]["delta"])
            per_model = []
            for model in models:
                cd = d["models"][model].get(survey, {}).get("conditions", {}).get(cond, {}).get("delta")
                if cd:
                    per_model.append(to_vec(cd))
            if per_model:
                all_h_vecs.append(h_delta)
                all_l_vecs.append(np.mean(per_model, axis=0))

        if not all_h_vecs:
            continue

        h_flat = np.concatenate(all_h_vecs)
        l_flat = np.concatenate(all_l_vecs)
        observed = cosine_sim(h_flat, l_flat)

        null_cos = []
        for _ in range(n_perm):
            shuffled_l = l_flat.copy()
            for i in range(0, len(shuffled_l), 10):
                perm = rng.permutation(10)
                shuffled_l[i:i+10] = shuffled_l[i:i+10][perm]
            null_cos.append(cosine_sim(h_flat, shuffled_l))

        null_cos = np.array(null_cos)
        p_value = np.mean(np.abs(null_cos) >= abs(observed))

        print(f"\n  {survey}:")
        print(f"    Observed cosine: {observed:+.4f}")
        print(f"    Null distribution: mean={null_cos.mean():+.4f}, std={null_cos.std():.4f}")
        print(f"    Permutation p (two-sided): {p_value:.4f} "
              f"{'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}")


if __name__ == "__main__":
    main()
