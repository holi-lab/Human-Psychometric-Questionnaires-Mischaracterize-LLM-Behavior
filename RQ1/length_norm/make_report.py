#!/usr/bin/env python3
"""Length-normalization report: paper (sum logprob) vs length-normalized (per-token mean) RQ1 results,
plus length-bias diagnostics. Writes report.md + report_data.json."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
OFFICIAL = HERE.parent.parent
BASE = json.loads((OFFICIAL / "results/RQ1/analysis.json").read_text())
NORM = json.loads((HERE / "analysis_normalized.json").read_text())
ECO_DIR = OFFICIAL / "results/RQ1/ecological"
VP = json.loads((OFFICIAL / "surveys/VP.json").read_text())

MODELS = ["gemma-3-27b-it", "gemma-3-4b-it", "gpt-oss-120b", "gpt-oss-20b",
          "Qwen2.5-72B-Instruct", "Qwen2.5-7B-Instruct",
          "Qwen3-235B-A22B-Instruct-2507-FP8", "Qwen3-30B-A3B-Instruct-2507"]
SHORT = {m: m.replace("-Instruct", "").replace("-2507", "").replace("-A22B", "").replace("-A3B", "").replace("-FP8", "")
         for m in MODELS}
COMPS = ["PVQ_pvq_values", "PVQ21_pvq_values", "BFI44_bfi_traits", "BFI10_bfi_traits"]


def per_model_table(d, metric_block, value_key):
    rows = {}
    for comp in COMPS:
        rows[comp] = {m: d[metric_block][m][comp][value_key] for m in MODELS}
    return rows


def fmt_table(rows_base, rows_norm):
    lines = ["| comparison | " + " | ".join(SHORT[m] for m in MODELS) + " | Avg |",
             "|---" * (len(MODELS) + 2) + "|"]
    for comp in COMPS:
        for tag, rows in [("sum", rows_base), ("norm", rows_norm)]:
            vals = [rows[comp][m] for m in MODELS]
            lines.append(
                f"| {comp} ({tag}) | " + " | ".join(f"{v:.2f}" for v in vals)
                + f" | {np.mean(vals):.2f} |")
    return "\n".join(lines)


def sig_block(d):
    out = {}
    for metric in ["spearman_rho", "ndcg"]:
        for grp in ["pvq_values", "bfi_traits", "overall"]:
            blk = d["analysis_4_within_vs_cross"][metric][grp]
            perm = blk["sign_flip_permutation_test"]
            out[f"{metric}/{grp}"] = {
                "within_mean": blk.get("within_mean"),
                "cross_mean": blk.get("cross_mean"),
                "mean_diff": perm.get("observed_mean_diff"),
                "perm_p_one_sided": perm.get("p_one_sided"),
            }
    return out


# ---- length diagnostics: per-construct mean token count of tagged responses ----
def length_stats():
    # token counts come from any model's raw file (same responses); use first
    raw = json.loads(next(iter(sorted(ECO_DIR.glob("*.json")))).read_text())
    ntok = {}  # (pid, output_id) -> num_tokens
    for sc in raw["results"]:
        for out in sc["outputs"]:
            ntok[(sc["portrait_id"], out["output_id"])] = out.get("num_tokens")
    stats = defaultdict(list)
    for sc in VP:
        pid = sc["portrait_id"]
        for out in sc["outputs"]:
            for key, label in [("correlations", "pvq_values"), ("bfi_correlations", "bfi_traits")]:
                for cname, cval in out.get(key, []):
                    if cval >= 0.3 and (pid, out["id"]) in ntok and ntok[(pid, out["id"])]:
                        stats[(label, cname)].append(ntok[(pid, out["id"])])
    return {f"{lbl}/{c}": {"n": len(v), "mean_tokens": round(float(np.mean(v)), 1)}
            for (lbl, c), v in sorted(stats.items())}


def rank_shift_vs_length(lens):
    """Correlate construct mean length with rank change (sum -> norm) per model."""
    shifts = []
    for m in MODELS:
        for comp, label in [("PVQ_pvq_values", "pvq_values"), ("BFI44_bfi_traits", "bfi_traits")]:
            b = BASE["analysis_1_spearman_rho"][m][comp]
            n = NORM["analysis_1_spearman_rho"][m][comp]
            cons = b["constructs"]
            eb, en = b["ecological_values"], n["ecological_values"]
            rb = np.argsort(np.argsort(-np.asarray(eb)))
            rn = np.argsort(np.argsort(-np.asarray(en)))
            for c, dr in zip(cons, (rn - rb).tolist()):
                key = f"{label}/{c.replace('-', '_').replace(' ', '_')}"
                if key in lens:
                    shifts.append((lens[key]["mean_tokens"], dr))
    x, y = zip(*shifts)
    rho, p = spearmanr(x, y)
    return {"n_points": len(shifts), "spearman_len_vs_rankshift": round(float(rho), 3),
            "p": round(float(p), 4)}


base_rho = per_model_table(BASE, "analysis_1_spearman_rho", "observed_rho")
norm_rho = per_model_table(NORM, "analysis_1_spearman_rho", "observed_rho")
base_ndcg = per_model_table(BASE, "analysis_3_ndcg", "ndcg")
norm_ndcg = per_model_table(NORM, "analysis_3_ndcg", "ndcg")
lens = length_stats()
diag = rank_shift_vs_length(lens)

report = {
    "significance_sum": sig_block(BASE),
    "significance_normalized": sig_block(NORM),
    "per_model_rho": {"sum": base_rho, "normalized": norm_rho},
    "per_model_ndcg": {"sum": base_ndcg, "normalized": norm_ndcg},
    "construct_length_stats": lens,
    "length_vs_rankshift": diag,
}
(HERE / "report_data.json").write_text(json.dumps(report, indent=2))

md = ["# Length-normalization ablation (RQ1)", "",
      "Scoring: paper Eq.1 = **sum** of token logprobs; ablation = **per-token mean** "
      "(`normalized_logprob`, stored per response in raw results — no re-inference needed).", "",
      "## Within- vs cross-method agreement (sign-flip permutation test)", "",
      "| block | within | cross | perm p |", "|---|---|---|---|"]
for tag, sig in [("sum", report["significance_sum"]), ("normalized", report["significance_normalized"])]:
    for k, v in sig.items():
        w = f"{v['within_mean']:.3f}" if v["within_mean"] is not None else "-"
        c = f"{v['cross_mean']:.3f}" if v["cross_mean"] is not None else "-"
        md.append(f"| {k} ({tag}) | {w} | {c} | {v['perm_p_one_sided']} |")
md += ["", "## Per-model Spearman rho (Gen vs established)", "", fmt_table(base_rho, norm_rho),
       "", "## Per-model NDCG", "", fmt_table(base_ndcg, norm_ndcg),
       "", "## Length-bias diagnostics",
       f"- Construct mean tagged-response length vs rank shift (sum->norm): "
       f"Spearman = {diag['spearman_len_vs_rankshift']} (p={diag['p']}, n={diag['n_points']})", ""]
(HERE / "report.md").write_text("\n".join(md))
print("wrote report.md / report_data.json")
print(json.dumps(report["significance_normalized"], indent=1))
print("length_vs_rankshift:", diag)
