#!/usr/bin/env python3
"""
2x2 (checkpoint x item transparency) construct structure analysis.

Cells:
  base      x original     <- ../base_instruct/results/base/established
  instruct  x original     <- ../base_instruct/results/instruct_plain/established
  base      x cue-reduced  <- results/base/established
  instruct  x cue-reduced  <- results/instruct_plain/established

All four cells use the identical constrained digit-logprob scoring under plain
prompts, so the 2x2 is fully method-matched.

Key quantity: the post-training gain in construct structure,
  gain = eta2(instruct) - eta2(base),
on original vs cue-reduced items. If the convergence produced by post-training
is expressed through item transparency, the gain should shrink on cue-reduced
items (interaction = gain_original - gain_cuereduced > 0).

Run (CPU only):
    python RQ3/cueless_2x2/analysis.py
"""
import itertools
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORIG_RESULTS = HERE.parent / "base_instruct" / "results"
CUELESS_RESULTS = HERE / "results"

PAIRS = {
    "Qwen2.5-7B": ("Qwen2.5-7B", "Qwen2.5-7B-Instruct"),
    "Gemma3-4B": ("gemma-3-4b-pt", "gemma-3-4b-it"),
    "Gemma3-27B": ("gemma-3-27b-pt", "gemma-3-27b-it"),
    "Qwen3-30B-A3B": ("Qwen3-30B-A3B-Base", "Qwen3-30B-A3B-Instruct-2507"),
}
SURVEYS = ["PVQ", "BFI44"]


def load(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def eta_squared(groups):
    allv = [v for sc in groups.values() for v in sc]
    N, K = len(allv), len(groups)
    if N < 2 or K < 2:
        return None
    gm = sum(allv) / N
    sst = sum((x - gm) ** 2 for x in allv)
    if sst == 0:
        return None
    ssb = sum(len(sc) * (sum(sc) / len(sc) - gm) ** 2 for sc in groups.values() if sc)
    return ssb / sst


def wmv(groups):
    allv = [v for sc in groups.values() for v in sc]
    mu = sum(allv) / len(allv)
    std = (sum((x - mu) ** 2 for x in allv) / len(allv)) ** 0.5
    if std == 0:
        return None
    z = {c: [(v - mu) / std for v in sc] for c, sc in groups.items()}
    kv, s = 0, 0.0
    for zs in z.values():
        k = len(zs)
        if k < 2:
            continue
        zb = sum(zs) / k
        s += sum((x - zb) ** 2 for x in zs) / (k - 1)
        kv += 1
    return s / kv if kv else None


def build_groups(est_result):
    groups = defaultdict(list)
    resp = est_result["responses"]
    ids = set(resp.get("normal", {})) | set(resp.get("reversed", {}))
    for iid in ids:
        entries = [resp[v][iid] for v in ("normal", "reversed")
                   if iid in resp.get(v, {})]
        vals = [e["final_score"] for e in entries if e["final_score"] is not None]
        if vals:
            groups[entries[0]["construct"]].append(sum(vals) / len(vals))
    return dict(groups)


def mean_digit_mass(est_result):
    vals = [e["digit_prob_mass"]
            for v in ("normal", "reversed")
            for e in est_result["responses"].get(v, {}).values()
            if e.get("digit_prob_mass") is not None]
    return sum(vals) / len(vals) if vals else None


def signflip_p(deltas, direction="greater"):
    """Exact one-sided sign-flip p on the mean of deltas."""
    obs = sum(deltas)
    n = len(deltas)
    count = 0
    for signs in itertools.product([1, -1], repeat=n):
        s = sum(sg * d for sg, d in zip(signs, deltas))
        if direction == "greater" and s >= obs - 1e-12:
            count += 1
        if direction == "less" and s <= obs + 1e-12:
            count += 1
    return count / 2 ** n


def cell(root, tag, survey, slug):
    r = load(root / tag / "established" / survey / f"{slug}.json")
    if r is None:
        return None
    g = build_groups(r)
    return {"eta2": eta_squared(g), "wmv": wmv(g), "digit_mass": mean_digit_mass(r)}


def main():
    out = {"pairs": {}, "summary": {}}
    md = ["# 2x2: checkpoint (base/instruct) x items (original/cue-reduced)",
          "", "All cells: constrained digit-logprob scoring, plain prompts.", ""]
    for survey in SURVEYS:
        md.append(f"## {survey}")
        md.append("")
        md.append("| Pair | base/orig | instr/orig | base/cueless | instr/cueless | "
                  "gain(orig) | gain(cueless) | interaction |")
        md.append("|---|---|---|---|---|---|---|---|")
        gains_o, gains_c, inters = [], [], []
        for pname, (bslug, islug) in PAIRS.items():
            cells = {
                "base_orig": cell(ORIG_RESULTS, "base", survey, bslug),
                "instr_orig": cell(ORIG_RESULTS, "instruct_plain", survey, islug),
                "base_cueless": cell(CUELESS_RESULTS, "base", survey, bslug),
                "instr_cueless": cell(CUELESS_RESULTS, "instruct_plain", survey, islug),
            }
            if any(c is None for c in cells.values()):
                missing = [k for k, c in cells.items() if c is None]
                md.append(f"| {pname} | MISSING: {', '.join(missing)} | | | | | | |")
                continue
            e = {k: c["eta2"] for k, c in cells.items()}
            gain_o = e["instr_orig"] - e["base_orig"]
            gain_c = e["instr_cueless"] - e["base_cueless"]
            inter = gain_o - gain_c
            gains_o.append(gain_o)
            gains_c.append(gain_c)
            inters.append(inter)
            out["pairs"].setdefault(survey, {})[pname] = {
                "eta2": {k: round(v, 4) for k, v in e.items()},
                "wmv": {k: round(c["wmv"], 4) for k, c in cells.items()},
                "digit_mass": {k: round(c["digit_mass"], 4) for k, c in cells.items()},
                "gain_orig": round(gain_o, 4),
                "gain_cueless": round(gain_c, 4),
                "interaction": round(inter, 4),
            }
            md.append(
                f"| {pname} | {e['base_orig']:.3f} | {e['instr_orig']:.3f} | "
                f"{e['base_cueless']:.3f} | {e['instr_cueless']:.3f} | "
                f"{gain_o:+.3f} | {gain_c:+.3f} | {inter:+.3f} |")
        if inters:
            n = len(inters)
            mo, mc, mi = (sum(x) / n for x in (gains_o, gains_c, inters))
            p_inter = signflip_p(inters, "greater")
            out["summary"][survey] = {
                "n_pairs": n,
                "mean_gain_orig": round(mo, 4),
                "mean_gain_cueless": round(mc, 4),
                "mean_interaction": round(mi, 4),
                "pairs_interaction_positive": sum(1 for x in inters if x > 0),
                "signflip_p_interaction": round(p_inter, 4),
            }
            md.append("")
            md.append(
                f"Mean gain on original items {mo:+.3f}, on cue-reduced items "
                f"{mc:+.3f}; interaction {mi:+.3f} "
                f"({sum(1 for x in inters if x > 0)}/{n} pairs positive, "
                f"exact sign-flip p={p_inter:.4f}).")
            md.append("")
    (HERE / "report.json").write_text(json.dumps(out, indent=2))
    (HERE / "report.md").write_text("\n".join(md))
    print("\n".join(md))
    print("\nsaved -> report.md / report.json")


if __name__ == "__main__":
    main()
