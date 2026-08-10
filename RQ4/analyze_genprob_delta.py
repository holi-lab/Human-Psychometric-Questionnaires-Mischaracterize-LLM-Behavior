#!/usr/bin/env python3
"""
RQ4 Extended Analysis: Compare Likert delta vs Gen-Prob delta vs Human delta.

Key question: When we apply persona prompting, does the *generation behavior*
shift in the same way as the Likert responses?

Normalization approaches for cross-scale comparison:
  1. Direction match (sign agreement per value dimension)
  2. Cosine similarity between delta vectors
  3. Spearman rank correlation between delta vectors
  4. L2-normalized deltas (unit vectors) for visual comparison
  5. Effect size: delta / std(baseline_values) for each source

Usage:
    python RQ4/analyze_genprob_delta.py
"""

import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent

VALUES = [
    "Achievement", "Benevolence", "Conformity", "Hedonism", "Power",
    "Security", "Self-Direction", "Stimulation", "Tradition", "Universalism",
]
SHORT = ["Ach", "Ben", "Con", "Hed", "Pow", "Sec", "SD", "Sti", "Tra", "Uni"]

CONDITIONS = [
    "Gender_A", "Gender_B", "Age_A", "Age_B",
    "Political_A", "Political_B", "Education_A", "Education_B",
]
COND_LABEL = {
    "Gender_A": "Male", "Gender_B": "Female",
    "Age_A": "Young (20-39)", "Age_B": "Old (80+)",
    "Political_A": "Right-wing", "Political_B": "Left-wing",
    "Education_A": "Below Univ.", "Education_B": "Univ.+",
}
COND_PAIRS = [
    ("Gender", ["Gender_A", "Gender_B"]),
    ("Age", ["Age_A", "Age_B"]),
    ("Political", ["Political_A", "Political_B"]),
    ("Education", ["Education_A", "Education_B"]),
]

SURVEYS = ["PVQ", "PVQ21", "VP"]


def load_data():
    return json.load(open(ROOT / "results" / "RQ4" / "analysis_circumstance_vs_human.json"))


def to_vec(delta_dict):
    return np.array([delta_dict.get(v, 0.0) for v in VALUES])


def l2_normalize(vec):
    n = np.linalg.norm(vec)
    return vec / n if n > 1e-12 else vec


def direction_match(a, b):
    """Count how many dimensions have the same sign."""
    return int(sum(1 for x, y in zip(a, b) if (x > 0 and y > 0) or (x < 0 and y < 0)))


def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def spearman_r(a, b):
    r, p = stats.spearmanr(a, b)
    return float(r), float(p)


def compute_llm_avg_delta(d, survey, cond):
    agg = np.zeros(len(VALUES))
    n = 0
    for model in d["models"]:
        cd = d["models"][model].get(survey, {}).get("conditions", {}).get(cond, {}).get("delta")
        if cd:
            agg += to_vec(cd)
            n += 1
    return agg / n if n else agg, n


def compute_llm_avg_baseline(d, survey):
    agg = np.zeros(len(VALUES))
    n = 0
    for model in d["models"]:
        bl = d["models"][model].get(survey, {}).get("baseline")
        if bl:
            agg += to_vec(bl)
            n += 1
    return agg / n if n else agg, n


def main():
    d = load_data()
    out_dir = ROOT / "results" / "RQ4"

    print("=" * 90)
    print("RQ4 EXTENDED: Likert Delta vs Gen-Prob Delta vs Human Delta")
    print("=" * 90)

    # ── 1. Per-condition comparison ──────────────────────────────
    rows = []
    for cond in CONDITIONS:
        h_delta = to_vec(d["human"]["conditions"][cond]["delta"])
        row = {"condition": cond, "label": COND_LABEL[cond]}

        for survey in SURVEYS:
            llm_avg, n = compute_llm_avg_delta(d, survey, cond)
            if n == 0:
                continue

            dm = direction_match(h_delta, llm_avg)
            cs = cosine_sim(h_delta, llm_avg)
            sr, sp = spearman_r(h_delta, llm_avg)

            row[f"{survey}_dir_match"] = dm
            row[f"{survey}_cosine"] = cs
            row[f"{survey}_spearman_r"] = sr
            row[f"{survey}_spearman_p"] = sp
            row[f"{survey}_mean_abs_delta"] = float(np.abs(llm_avg).mean())
            row[f"{survey}_n_models"] = n

        rows.append(row)

    # ── 2. Print comparison table ────────────────────────────────
    print("\n" + "=" * 90)
    print("Per-Condition: Human vs LLM (Direction Match / Cosine Sim / Spearman r)")
    print("=" * 90)

    header = f"{'Condition':<15}"
    for s in SURVEYS:
        header += f" | {s:^30}"
    print(header)

    sub_header = f"{'':15}"
    for s in SURVEYS:
        sub_header += f" | {'Dir':>4} {'Cos':>6} {'Spr':>6} {'|d|':>6}"
    print(sub_header)
    print("-" * len(sub_header))

    for row in rows:
        line = f"{row['label']:<15}"
        for s in SURVEYS:
            dm = row.get(f"{s}_dir_match", "-")
            cs = row.get(f"{s}_cosine", float("nan"))
            sr = row.get(f"{s}_spearman_r", float("nan"))
            mad = row.get(f"{s}_mean_abs_delta", float("nan"))
            if isinstance(dm, int):
                line += f" | {dm:>3}/10 {cs:>6.3f} {sr:>6.3f} {mad:>6.3f}"
            else:
                line += f" | {'---':>30}"
        print(line)

    # ── 3. Cross-survey comparison (Likert vs GenProb) ───────────
    print("\n" + "=" * 90)
    print("Cross-Survey: PVQ Likert Delta vs VP Gen-Prob Delta (per condition)")
    print("  -> Does persona shift LLM generation behavior the same way as Likert?")
    print("=" * 90)

    header2 = f"{'Condition':<15} | {'Dir':>4} {'Cos':>6} {'Spr':>6} {'p':>7}"
    print(header2)
    print("-" * len(header2))

    all_pvq_deltas = []
    all_vp_deltas = []

    for cond in CONDITIONS:
        pvq_avg, n1 = compute_llm_avg_delta(d, "PVQ", cond)
        vp_avg, n2 = compute_llm_avg_delta(d, "VP", cond)

        if n1 == 0 or n2 == 0:
            continue

        dm = direction_match(pvq_avg, vp_avg)
        cs = cosine_sim(pvq_avg, vp_avg)
        sr, sp = spearman_r(pvq_avg, vp_avg)

        all_pvq_deltas.append(pvq_avg)
        all_vp_deltas.append(vp_avg)

        print(f"{COND_LABEL[cond]:<15} | {dm:>3}/10 {cs:>6.3f} {sr:>6.3f} {sp:>7.4f}")

    # Overall
    if all_pvq_deltas:
        all_pvq = np.concatenate(all_pvq_deltas)
        all_vp = np.concatenate(all_vp_deltas)
        dm = direction_match(all_pvq, all_vp)
        cs = cosine_sim(all_pvq, all_vp)
        sr, sp = spearman_r(all_pvq, all_vp)
        print(f"{'OVERALL':<15} | {dm:>3}/{len(all_pvq)} {cs:>6.3f} {sr:>6.3f} {sp:>7.4f}")

    # ── 4. L2-normalized delta comparison ────────────────────────
    print("\n" + "=" * 90)
    print("L2-Normalized Deltas (unit vectors) — Direction-only comparison")
    print("  -> Removes scale, keeps relative pattern")
    print("=" * 90)

    for cond in CONDITIONS:
        h_delta = to_vec(d["human"]["conditions"][cond]["delta"])
        h_norm = l2_normalize(h_delta)

        print(f"\n  {COND_LABEL[cond]:}")
        print(f"  {'Value':<16} {'Human':>8} {'PVQ':>8} {'PVQ21':>8} {'VP':>8}")
        print(f"  {'-'*52}")

        survey_norms = {}
        for s in SURVEYS:
            avg, n = compute_llm_avg_delta(d, s, cond)
            survey_norms[s] = l2_normalize(avg) if n > 0 else np.zeros(len(VALUES))

        for i, v in enumerate(VALUES):
            line = f"  {v:<16} {h_norm[i]:>+8.3f}"
            for s in SURVEYS:
                line += f" {survey_norms[s][i]:>+8.3f}"
            print(line)

    # ── 5. Per-model analysis ────────────────────────────────────
    print("\n" + "=" * 90)
    print("Per-Model: PVQ Likert vs VP Gen-Prob (Cosine Sim of delta vectors)")
    print("=" * 90)

    model_results = []
    for model in d["models"]:
        pvq_cos = []
        vp_cos = []
        pvq_vp_cos = []

        for cond in CONDITIONS:
            h_delta = to_vec(d["human"]["conditions"][cond]["delta"])

            pvq_d = d["models"][model].get("PVQ", {}).get("conditions", {}).get(cond, {}).get("delta")
            vp_d = d["models"][model].get("VP", {}).get("conditions", {}).get(cond, {}).get("delta")

            if pvq_d:
                pvq_vec = to_vec(pvq_d)
                pvq_cos.append(cosine_sim(h_delta, pvq_vec))
            if vp_d:
                vp_vec = to_vec(vp_d)
                vp_cos.append(cosine_sim(h_delta, vp_vec))
            if pvq_d and vp_d:
                pvq_vp_cos.append(cosine_sim(to_vec(pvq_d), to_vec(vp_d)))

        mr = {
            "model": model,
            "pvq_human_cos": np.mean(pvq_cos) if pvq_cos else float("nan"),
            "vp_human_cos": np.mean(vp_cos) if vp_cos else float("nan"),
            "pvq_vp_cos": np.mean(pvq_vp_cos) if pvq_vp_cos else float("nan"),
        }
        model_results.append(mr)

    print(f"\n{'Model':<45} {'PVQ-Human':>10} {'VP-Human':>10} {'PVQ-VP':>10}")
    print("-" * 80)
    for mr in model_results:
        print(f"{mr['model']:<45} {mr['pvq_human_cos']:>10.3f} {mr['vp_human_cos']:>10.3f} {mr['pvq_vp_cos']:>10.3f}")

    # Averages
    avg_ph = np.nanmean([mr["pvq_human_cos"] for mr in model_results])
    avg_vh = np.nanmean([mr["vp_human_cos"] for mr in model_results])
    avg_pv = np.nanmean([mr["pvq_vp_cos"] for mr in model_results])
    print("-" * 80)
    print(f"{'AVERAGE':<45} {avg_ph:>10.3f} {avg_vh:>10.3f} {avg_pv:>10.3f}")

    # ── 6. Amplification analysis with effect-size normalization ──
    print("\n" + "=" * 90)
    print("Amplification: How much does each source amplify human deltas?")
    print("  Metric: ratio of L2 norms (||LLM_delta|| / ||Human_delta||)")
    print("=" * 90)

    print(f"\n{'Condition':<15} {'||H||':>8} {'||PVQ||':>8} {'||VP||':>8} {'PVQ/H':>8} {'VP/H':>8} {'VP/PVQ':>8}")
    print("-" * 70)

    all_ratios = {"pvq_h": [], "vp_h": [], "vp_pvq": []}
    for cond in CONDITIONS:
        h_delta = to_vec(d["human"]["conditions"][cond]["delta"])
        h_norm = np.linalg.norm(h_delta)

        pvq_avg, n1 = compute_llm_avg_delta(d, "PVQ", cond)
        vp_avg, n2 = compute_llm_avg_delta(d, "VP", cond)

        pvq_norm = np.linalg.norm(pvq_avg)
        vp_norm = np.linalg.norm(vp_avg)

        r_pvq = pvq_norm / h_norm if h_norm > 0 else float("nan")
        r_vp = vp_norm / h_norm if h_norm > 0 else float("nan")
        r_vp_pvq = vp_norm / pvq_norm if pvq_norm > 0 else float("nan")

        all_ratios["pvq_h"].append(r_pvq)
        all_ratios["vp_h"].append(r_vp)
        all_ratios["vp_pvq"].append(r_vp_pvq)

        print(f"{COND_LABEL[cond]:<15} {h_norm:>8.3f} {pvq_norm:>8.3f} {vp_norm:>8.3f} "
              f"{r_pvq:>8.1f}x {r_vp:>8.1f}x {r_vp_pvq:>8.1f}x")

    print("-" * 70)
    print(f"{'AVERAGE':<15} {'':>8} {'':>8} {'':>8} "
          f"{np.mean(all_ratios['pvq_h']):>7.1f}x {np.mean(all_ratios['vp_h']):>7.1f}x "
          f"{np.mean(all_ratios['vp_pvq']):>7.1f}x")

    # ── 7. Save structured results ───────────────────────────────
    output = {
        "description": (
            "Extended RQ4 analysis: comparison of Likert deltas, Gen-Prob deltas, "
            "and Human deltas under persona prompting. "
            "Metrics: direction match, cosine similarity, Spearman correlation, "
            "L2-norm ratio (amplification)."
        ),
        "per_condition": [],
        "per_model": model_results,
        "cross_survey_likert_vs_genprob": [],
        "summary": {},
    }

    for cond in CONDITIONS:
        h_delta = to_vec(d["human"]["conditions"][cond]["delta"])
        entry = {"condition": cond, "label": COND_LABEL[cond]}

        for survey in SURVEYS:
            llm_avg, n = compute_llm_avg_delta(d, survey, cond)
            if n == 0:
                continue
            dm = direction_match(h_delta, llm_avg)
            cs = cosine_sim(h_delta, llm_avg)
            sr, sp = spearman_r(h_delta, llm_avg)
            entry[survey] = {
                "dir_match": dm,
                "cosine_sim": round(cs, 4),
                "spearman_r": round(sr, 4),
                "spearman_p": round(sp, 4),
                "l2_norm": round(float(np.linalg.norm(llm_avg)), 4),
                "l2_normalized_delta": {v: round(float(x), 4) for v, x in zip(VALUES, l2_normalize(llm_avg))},
            }

        entry["human"] = {
            "l2_norm": round(float(np.linalg.norm(h_delta)), 4),
            "l2_normalized_delta": {v: round(float(x), 4) for v, x in zip(VALUES, l2_normalize(h_delta))},
        }
        output["per_condition"].append(entry)

    # Cross-survey
    for cond in CONDITIONS:
        pvq_avg, n1 = compute_llm_avg_delta(d, "PVQ", cond)
        vp_avg, n2 = compute_llm_avg_delta(d, "VP", cond)
        if n1 > 0 and n2 > 0:
            output["cross_survey_likert_vs_genprob"].append({
                "condition": cond,
                "dir_match": direction_match(pvq_avg, vp_avg),
                "cosine_sim": round(cosine_sim(pvq_avg, vp_avg), 4),
                "spearman_r": round(spearman_r(pvq_avg, vp_avg)[0], 4),
            })

    output["summary"] = {
        "avg_pvq_human_cosine": round(avg_ph, 4),
        "avg_vp_human_cosine": round(avg_vh, 4),
        "avg_pvq_vp_cosine": round(avg_pv, 4),
        "avg_amplification_pvq_over_human": round(float(np.mean(all_ratios["pvq_h"])), 2),
        "avg_amplification_vp_over_human": round(float(np.mean(all_ratios["vp_h"])), 2),
        "avg_amplification_vp_over_pvq": round(float(np.mean(all_ratios["vp_pvq"])), 2),
    }

    out_path = out_dir / "analysis_genprob_delta.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nSaved -> {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
