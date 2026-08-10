#!/usr/bin/env python3
"""
Base vs instruct comparison analysis (Appendix, RQ3).

Consumes the constrained-scoring outputs written by run.py under
results/{tag}/... plus (optionally) existing instruct free-form results.

For each base<->instruct pair it computes:
  (i)   prosocial skew: PVQ/BFI construct ranking, base vs instruct, and the
        rank/value shift of prosocial constructs
        (Benevolence, Universalism, Agreeableness).
  (ii)  eta^2 & WMV construct structure (established), base vs instruct.
  (iii) recognition mean-F1, base vs instruct.
  (iv)  Within-family VP gen-prob profile Spearman(base, instruct) vs the
        matched questionnaire profile Spearman(base, instruct): Schwartz
        values vs PVQ-40 and Big Five traits vs BFI-44.

Outputs:
  results/analysis/pair_{name}.json
  results/analysis/summary.md  (headline tables)

Usage:
  python RQ3/base_instruct/analysis.py
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
OFFICIAL = HERE.parent.parent
RES = HERE / "results"
ANALYSIS = RES / "analysis"

PROSOCIAL = {"Benevolence", "Universalism", "Agreeableness"}

# base<->instruct pairs. tags are subdirs under results/.
# slug = model_slug used in filenames.
PAIRS = [
    {"name": "Qwen2.5-7B", "base_slug": "Qwen2.5-7B",
     "instr_slug": "Qwen2.5-7B-Instruct"},
    {"name": "gemma-3-4b", "base_slug": "gemma-3-4b-pt",
     "instr_slug": "gemma-3-4b-it"},
    {"name": "gemma-3-27b", "base_slug": "gemma-3-27b-pt",
     "instr_slug": "gemma-3-27b-it"},
    {"name": "Qwen3-30B-A3B", "base_slug": "Qwen3-30B-A3B-Base",
     "instr_slug": "Qwen3-30B-A3B-Instruct-2507"},
]
BASE_TAG = "base"
INSTR_TAG = "instruct_plain"   # method+format matched instruct


def load(path):
    return json.loads(Path(path).read_text()) if Path(path).exists() else None


# ---------- eta^2 / WMV ----------
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


def build_groups(est_result, prompt_key="combined"):
    groups = defaultdict(list)
    if prompt_key == "combined":
        # average normal+reversed per item
        resp = est_result["responses"]
        ids = set(resp.get("normal", {})) | set(resp.get("reversed", {}))
        for iid in ids:
            entries = [resp[v][iid] for v in ("normal", "reversed")
                       if iid in resp.get(v, {})]
            vals = [e["final_score"] for e in entries if e["final_score"] is not None]
            if vals:
                groups[entries[0]["construct"]].append(sum(vals) / len(vals))
    else:
        for e in est_result["responses"].get(prompt_key, {}).values():
            if e["final_score"] is not None:
                groups[e["construct"]].append(e["final_score"])
    return dict(groups)


# ---------- recognition F1 ----------
def recog_mean_f1(recog_result, survey):
    if not recog_result:
        return None, {}
    constructs = list(recog_result["construct_definitions_used"].keys())
    per = {}
    for c in constructs:
        tp = fp = fn = 0
        for item in recog_result["results"].values():
            pred = item["predictions"].get(c, {}).get("parsed")
            if pred is None:
                continue
            if survey == "VP":
                gold = c in item.get("ground_truth_positive", [])
            else:
                gold = item.get("ground_truth_construct") == c
            if pred and gold:
                tp += 1
            elif pred and not gold:
                fp += 1
            elif (not pred) and gold:
                fn += 1
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per[c] = round(f1, 4)
    mean = round(sum(per.values()) / len(per), 4) if per else None
    return mean, per


# ---------- ecological gen-prob profile ----------
CORR_KEYS = [("correlations", "pvq_values"),
             ("higher_pvq_correlations", "higher_order_values"),
             ("bfi_correlations", "bfi_traits")]
CORR_THRESHOLD = 0.3


def vp_lookup():
    scenarios = json.loads((OFFICIAL / "surveys" / "VP.json").read_text())
    lut = {}
    for sc in scenarios:
        m = {}
        for out in sc["outputs"]:
            cd = {}
            for ck, _ in CORR_KEYS:
                cd[ck] = {nm: val for nm, val in out.get(ck, [])}
            m[out["id"]] = cd
        lut[sc["portrait_id"]] = m
    return lut


def genprob_profiles(eco_result, lut, score_field="total_logprob"):
    """Return separate construct profiles for each VP label family."""
    accum = {label: defaultdict(list) for _, label in CORR_KEYS}
    for sc in eco_result.get("results", []):
        pid = sc["portrait_id"]
        if pid not in lut:
            continue
        cmap = lut[pid]
        for ck, label in CORR_KEYS:
            cscore = defaultdict(float)
            ccount = defaultdict(int)
            for out in sc["outputs"]:
                score = out.get(score_field)
                if score is None:
                    continue
                oid = out["output_id"]
                for cn, cv in cmap.get(oid, {}).get(ck, {}).items():
                    if cv >= CORR_THRESHOLD:
                        cscore[cn] += float(score)
                        ccount[cn] += 1
            for cn in cscore:
                accum[label][cn].append(cscore[cn] / ccount[cn])
    return {
        label: {cn: float(np.mean(values)) for cn, values in constructs.items()}
        for label, constructs in accum.items()
    }


def aligned_spearman(d1, d2):
    keys = sorted(set(d1) & set(d2))
    if len(keys) < 3:
        return None, len(keys)
    a = [d1[k] for k in keys]
    b = [d2[k] for k in keys]
    rho, _ = spearmanr(a, b)
    return (round(float(rho), 4) if rho == rho else None), len(keys)


def rank_dict(d):
    """higher value -> rank 1."""
    order = sorted(d, key=lambda k: -d[k])
    return {k: i + 1 for i, k in enumerate(order)}


def standardized_delta(base, instruct):
    """Within-profile z-score changes, avoiding raw cross-model scale effects."""
    keys = sorted(set(base) & set(instruct))
    if len(keys) < 3:
        return {}
    b = np.asarray([base[k] for k in keys], float)
    i = np.asarray([instruct[k] for k in keys], float)
    bz = (b - b.mean()) / (b.std() if b.std() > 0 else 1.0)
    iz = (i - i.mean()) / (i.std() if i.std() > 0 else 1.0)
    br, ir = rank_dict(base), rank_dict(instruct)
    return {
        k: {
            "base_rank": br[k],
            "instruct_rank": ir[k],
            "delta_z": round(float(iz[idx] - bz[idx]), 4),
        }
        for idx, k in enumerate(keys)
    }


def standardized_delta_profile(base, instruct):
    """Within-profile standardized instruction-minus-base change vector."""
    keys = sorted(set(base) & set(instruct))
    if len(keys) < 3:
        return {}
    b = np.asarray([base[k] for k in keys], float)
    i = np.asarray([instruct[k] for k in keys], float)
    bz = (b - b.mean()) / (b.std() if b.std() > 0 else 1.0)
    iz = (i - i.mean()) / (i.std() if i.std() > 0 else 1.0)
    return {k: float(iz[idx] - bz[idx]) for idx, k in enumerate(keys)}


def normalized_construct_names(profile):
    """Align VP underscore names with questionnaire hyphenated names."""
    return {name.replace("_", "-"): value for name, value in profile.items()}


def direct_gap_for_domain(
    base_ecological,
    instruct_ecological,
    lut,
    base_questionnaire,
    instruct_questionnaire,
    vp_label,
    expected_constructs,
    context,
):
    """VP-questionnaire rho at each checkpoint and between change vectors."""
    base_questionnaire = normalized_construct_names(base_questionnaire)
    instruct_questionnaire = normalized_construct_names(instruct_questionnaire)
    questionnaire_delta = standardized_delta_profile(
        base_questionnaire, instruct_questionnaire
    )
    result = {}
    for score_name, score_field in (
        ("summed", "total_logprob"),
        ("normalized", "normalized_logprob"),
    ):
        base_vp = normalized_construct_names(
            genprob_profiles(base_ecological, lut, score_field).get(vp_label, {})
        )
        instruct_vp = normalized_construct_names(
            genprob_profiles(instruct_ecological, lut, score_field).get(vp_label, {})
        )
        construct_sets = [
            set(profile) for profile in (
                base_vp,
                instruct_vp,
                base_questionnaire,
                instruct_questionnaire,
            )
        ]
        assert all(len(keys) == expected_constructs for keys in construct_sets), (
            f"{context}/{score_name}: expected {expected_constructs} constructs, "
            f"got {[len(keys) for keys in construct_sets]}"
        )
        assert all(keys == construct_sets[0] for keys in construct_sets[1:]), (
            f"{context}/{score_name}: mismatched construct names: "
            f"{[sorted(keys) for keys in construct_sets]}"
        )
        vp_delta = standardized_delta_profile(base_vp, instruct_vp)
        base_rho, base_n = aligned_spearman(base_vp, base_questionnaire)
        instruct_rho, instruct_n = aligned_spearman(
            instruct_vp, instruct_questionnaire
        )
        delta_rho, delta_n = aligned_spearman(vp_delta, questionnaire_delta)
        result[score_name] = {
            "vp_score_field": score_field,
            "base_vp_vs_questionnaire_rho": base_rho,
            "instruct_vp_vs_questionnaire_rho": instruct_rho,
            "standardized_delta_vp_vs_questionnaire_rho": delta_rho,
            "n_constructs": {
                "base": base_n,
                "instruct": instruct_n,
                "standardized_delta": delta_n,
            },
        }
    return result


def main():
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    lut = vp_lookup()
    summary_rows = []
    md = ["# Base vs instruct — analysis\n"]

    for pair in PAIRS:
        name = pair["name"]
        bslug, islug = pair["base_slug"], pair["instr_slug"]
        pair_out = {"pair": name}
        md.append(f"\n## {name}  (base={bslug}  instruct={islug})\n")

        # ---- (i)+(ii) established: PVQ + BFI44 ----
        for survey in ("PVQ", "BFI44"):
            b = load(RES / BASE_TAG / "established" / survey / f"{bslug}.json")
            i = load(RES / INSTR_TAG / "established" / survey / f"{islug}.json")
            if not b or not i:
                md.append(f"- established/{survey}: MISSING "
                          f"(base={'ok' if b else 'no'} instr={'ok' if i else 'no'})")
                continue
            bg = build_groups(b)
            ig = build_groups(i)
            b_eta, i_eta = eta_squared(bg), eta_squared(ig)
            b_wmv, i_wmv = wmv(bg), wmv(ig)
            bca = b["construct_averages"]["combined"]
            ica = i["construct_averages"]["combined"]
            main_instr = load(
                OFFICIAL / "results" / "RQ1" / "established" / survey
                / f"{islug}.json"
            )
            cross_format_rho = None
            if main_instr:
                main_ca = {
                    k: v for k, v in
                    main_instr["construct_averages"]["combined"].items()
                    if v is not None
                }
                cross_format_rho, _ = aligned_spearman(
                    {k: v for k, v in ica.items() if v is not None}, main_ca
                )
            b_rank = rank_dict({k: v for k, v in bca.items() if v is not None})
            i_rank = rank_dict({k: v for k, v in ica.items() if v is not None})
            proso = {}
            for c in sorted(set(bca) & set(ica)):
                if c in PROSOCIAL or c in ("Benevolence", "Universalism", "Agreeableness"):
                    proso[c] = {
                        "base_val": bca.get(c), "instr_val": ica.get(c),
                        "base_rank": b_rank.get(c), "instr_rank": i_rank.get(c),
                        "val_delta": (round(ica[c] - bca[c], 4)
                                      if bca.get(c) is not None and ica.get(c) is not None else None),
                        "rank_delta": (i_rank.get(c, 99) - b_rank.get(c, 99)),
                    }
            pair_out[f"established_{survey}"] = {
                "base_eta2": round(b_eta, 4) if b_eta else None,
                "instr_eta2": round(i_eta, 4) if i_eta else None,
                "base_wmv": round(b_wmv, 4) if b_wmv else None,
                "instr_wmv": round(i_wmv, 4) if i_wmv else None,
                "instr_plain_vs_main_generated_profile_rho": cross_format_rho,
                "base_ranking": sorted(bca.items(), key=lambda kv: -(kv[1] or -9)),
                "instr_ranking": sorted(ica.items(), key=lambda kv: -(kv[1] or -9)),
                "prosocial": proso,
            }
            md.append(f"\n### {survey}")
            md.append(f"- eta2: base={b_eta:.4f} instr={i_eta:.4f} "
                      f"(delta={i_eta-b_eta:+.4f})" if b_eta and i_eta else "- eta2: n/a")
            md.append(f"- WMV : base={b_wmv:.4f} instr={i_wmv:.4f} "
                      f"(delta={i_wmv-b_wmv:+.4f})" if b_wmv and i_wmv else "- WMV: n/a")
            if cross_format_rho is not None:
                md.append(
                    "- instruct plain vs main generated-answer construct-profile "
                    f"rho={cross_format_rho:.4f}"
                )
            md.append("- prosocial construct value / rank (rank 1 = highest):")
            md.append("  | construct | base val | instr val | Δval | base rank | instr rank | Δrank |")
            md.append("  |---|---|---|---|---|---|---|")
            for c, d in proso.items():
                md.append(f"  | {c} | {d['base_val']} | {d['instr_val']} | "
                          f"{d['val_delta']} | {d['base_rank']} | {d['instr_rank']} | "
                          f"{d['rank_delta']:+d} |")

        # ---- (iii) recognition mean F1 ----
        md.append("\n### recognition mean-F1")
        md.append("  | survey | base F1 | instr F1 | Δ |")
        md.append("  |---|---|---|---|")
        rec_out = {}
        for survey in ("PVQ", "BFI44", "VP"):
            b = load(RES / BASE_TAG / "recognition" / survey / f"{bslug}.json")
            i = load(RES / INSTR_TAG / "recognition" / survey / f"{islug}.json")
            bf, _ = recog_mean_f1(b, survey)
            iff, _ = recog_mean_f1(i, survey)
            rec_out[survey] = {"base": bf, "instr": iff}
            delta = f"{iff-bf:+.4f}" if (bf is not None and iff is not None) else "n/a"
            md.append(f"  | {survey} | {bf} | {iff} | {delta} |")
        pair_out["recognition_meanF1"] = rec_out

        # ---- (iv) profile shift: compare matched construct families only ----
        be = load(RES / BASE_TAG / "ecological" / f"{bslug}.json")
        ie = load(RES / INSTR_TAG / "ecological" / f"{islug}.json")
        bp, ip = {}, {}
        if be and ie:
            bp = genprob_profiles(be, lut)
            ip = genprob_profiles(ie, lut)

        domains = {
            "values": ("pvq_values", "PVQ"),
            "traits": ("bfi_traits", "BFI44"),
        }
        shift_out = {}
        md.append("\n### profile shift by matched construct family")
        md.append("  | family | VP genprob rho | questionnaire rho | VP higher? |")
        md.append("  |---|---:|---:|---| ")
        for family, (label, survey) in domains.items():
            genprob_rho = None
            delta = {}
            if label in bp and label in ip:
                genprob_rho, _ = aligned_spearman(bp[label], ip[label])
                delta = standardized_delta(bp[label], ip[label])

            bq = load(RES / BASE_TAG / "established" / survey / f"{bslug}.json")
            iq = load(RES / INSTR_TAG / "established" / survey / f"{islug}.json")
            quest_rho = None
            if bq and iq:
                bqa = {k: v for k, v in bq["construct_averages"]["combined"].items()
                       if v is not None}
                iqa = {k: v for k, v in iq["construct_averages"]["combined"].items()
                       if v is not None}
                quest_rho, _ = aligned_spearman(bqa, iqa)

            more_stable = (
                genprob_rho is not None and quest_rho is not None
                and genprob_rho > quest_rho
            )
            shift_out[family] = {
                "genprob_spearman_base_instr": genprob_rho,
                "questionnaire_spearman_base_instr": quest_rho,
                "genprob_more_stable": more_stable,
                "genprob_standardized_delta": delta,
            }
            md.append(f"  | {family} | {genprob_rho} | {quest_rho} | {more_stable} |")
        pair_out["profile_shift"] = shift_out

        # ---- (v) direct VP-questionnaire gap diagnostic by domain ----
        # Base/instruction correlations compare the two profiles at each
        # checkpoint. Delta correlations compare within-profile standardized
        # changes so VP and questionnaire scale differences do not affect it.
        gap_domains = {}
        if be and ie:
            for domain, vp_label, survey, expected_n in (
                ("values", "pvq_values", "PVQ", 10),
                ("traits", "bfi_traits", "BFI44", 5),
            ):
                bq = load(
                    RES / BASE_TAG / "established" / survey / f"{bslug}.json"
                )
                iq = load(
                    RES / INSTR_TAG / "established" / survey / f"{islug}.json"
                )
                if not bq or not iq:
                    continue
                base_questionnaire = {
                    k: v for k, v in
                    bq["construct_averages"]["combined"].items()
                    if v is not None
                }
                instruct_questionnaire = {
                    k: v for k, v in
                    iq["construct_averages"]["combined"].items()
                    if v is not None
                }
                gap_domains[domain] = direct_gap_for_domain(
                    be,
                    ie,
                    lut,
                    base_questionnaire,
                    instruct_questionnaire,
                    vp_label,
                    expected_n,
                    f"{name}/{domain}",
                )

        # Preserve the original values-only paths for existing consumers.
        value_gap = gap_domains.get("values", {})
        legacy_values = {}
        for score_name, result in value_gap.items():
            legacy_values[score_name] = {
                "vp_score_field": result["vp_score_field"],
                "base_vp_vs_pvq_rho": result[
                    "base_vp_vs_questionnaire_rho"
                ],
                "instruct_vp_vs_pvq_rho": result[
                    "instruct_vp_vs_questionnaire_rho"
                ],
                "standardized_delta_vp_vs_pvq_rho": result[
                    "standardized_delta_vp_vs_questionnaire_rho"
                ],
                "n_constructs": result["n_constructs"],
            }
        gap_out = {
            **legacy_values,
            "domains": gap_domains,
            "compatibility_note": (
                "Top-level summed/normalized remain aliases for domains.values."
            ),
        }
        pair_out["direct_gap_diagnostic"] = gap_out

        md.append("\n### direct VP-PVQ gap diagnostic (Schwartz values)")
        md.append(
            "  | VP score | base rho(VP, PVQ) | instruct rho(VP, PVQ) | "
            "rho(standardized delta VP, delta PVQ) |"
        )
        md.append("  |---|---:|---:|---:|")
        for score_name, result in legacy_values.items():
            md.append(
                f"  | {score_name} | {result['base_vp_vs_pvq_rho']} | "
                f"{result['instruct_vp_vs_pvq_rho']} | "
                f"{result['standardized_delta_vp_vs_pvq_rho']} |"
            )

        md.append("\n### direct VP-BFI44 gap diagnostic (Big Five traits)")
        md.append(
            "  | VP score | base rho(VP, BFI44) | instruct rho(VP, BFI44) | "
            "rho(standardized delta VP, delta BFI44) |"
        )
        md.append("  |---|---:|---:|---:|")
        for score_name, result in gap_domains.get("traits", {}).items():
            md.append(
                f"  | {score_name} | "
                f"{result['base_vp_vs_questionnaire_rho']} | "
                f"{result['instruct_vp_vs_questionnaire_rho']} | "
                f"{result['standardized_delta_vp_vs_questionnaire_rho']} |"
            )

        summary_rows.append(pair_out)
        (ANALYSIS / f"pair_{name}.json").write_text(
            json.dumps(pair_out, indent=2, ensure_ascii=False))

    (ANALYSIS / "summary.md").write_text("\n".join(md))
    (ANALYSIS / "all_pairs.json").write_text(
        json.dumps(summary_rows, indent=2, ensure_ascii=False))
    print("\n".join(md))
    print(f"\nWrote {ANALYSIS/'summary.md'}")


if __name__ == "__main__":
    main()
