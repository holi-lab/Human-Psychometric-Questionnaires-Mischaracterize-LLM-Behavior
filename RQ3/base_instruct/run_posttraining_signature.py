#!/usr/bin/env python3
"""Post-training signatures in VP generation probability profiles (Appendix, RQ3).

For the available matched base/instruction model pairs, this analysis compares the
base-to-instruction change in the VP profile with the corresponding change in
the questionnaire profile. Profiles are standardized within model before
computing change vectors, so comparisons concern relative construct ordering
rather than raw likelihood or response-scale differences.

Both summed sequence log-likelihood and per-token log-likelihood are reported.
"""

import importlib.util
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

# The four matched pairs reported in the paper (pairs sharing pretraining).
PAIRS = [
    ("Qwen2.5-7B", "Qwen2.5-7B", "Qwen2.5-7B-Instruct"),
    ("Gemma3-4B", "gemma-3-4b-pt", "gemma-3-4b-it"),
    ("Gemma3-27B", "gemma-3-27b-pt", "gemma-3-27b-it"),
    ("Qwen3-30B-A3B", "Qwen3-30B-A3B-Base", "Qwen3-30B-A3B-Instruct-2507"),
]

DOMAINS = {
    "values": ("pvq_values", "PVQ"),
    "traits": ("bfi_traits", "BFI44"),
}


def load_analysis_module():
    spec = importlib.util.spec_from_file_location("base_instruct_analysis", HERE / "analysis.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ANALYSIS = load_analysis_module()
VP_LOOKUP = ANALYSIS.vp_lookup()


def load_json(path):
    return json.loads(path.read_text())


def normalize(name):
    return name.lower().replace("-", "_").replace(" ", "_")


def align(a, b):
    na = {normalize(k): float(v) for k, v in a.items()}
    nb = {normalize(k): float(v) for k, v in b.items()}
    keys = sorted(set(na) & set(nb))
    return keys, np.asarray([na[k] for k in keys]), np.asarray([nb[k] for k in keys])


def zscore(values):
    sd = values.std()
    return (values - values.mean()) / (sd if sd > 0 else 1.0)


def profile_rho(a, b):
    _, av, bv = align(a, b)
    return float(spearmanr(av, bv).statistic)


def delta_vector(base, instruct):
    keys, base_values, instruct_values = align(base, instruct)
    return keys, zscore(instruct_values) - zscore(base_values)


def cosine(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else None


def rank_map(profile):
    ordered = sorted(profile.items(), key=lambda item: -item[1])
    return {normalize(name): idx + 1 for idx, (name, _) in enumerate(ordered)}


def genprob_profiles(ecological_result, score_field):
    adjusted = json.loads(json.dumps(ecological_result))
    for scenario in adjusted.get("results", []):
        for output in scenario.get("outputs", []):
            output["total_logprob"] = output.get(score_field)
    return ANALYSIS.genprob_profiles(adjusted, VP_LOOKUP)


def model_profiles(base_slug, instruct_slug, score_field, label, survey):
    base_eco = load_json(RESULTS / "base" / "ecological" / f"{base_slug}.json")
    instruct_eco = load_json(
        RESULTS / "instruct_plain" / "ecological" / f"{instruct_slug}.json"
    )
    base_vp = genprob_profiles(base_eco, score_field)[label]
    instruct_vp = genprob_profiles(instruct_eco, score_field)[label]

    base_questionnaire = load_json(
        RESULTS / "base" / "established" / survey / f"{base_slug}.json"
    )["construct_averages"]["combined"]
    instruct_questionnaire = load_json(
        RESULTS / "instruct_plain" / "established" / survey / f"{instruct_slug}.json"
    )["construct_averages"]["combined"]
    return base_vp, instruct_vp, base_questionnaire, instruct_questionnaire


def analyze(score_field):
    output = {"score_field": score_field, "domains": {}}
    for domain, (label, survey) in DOMAINS.items():
        rows = []
        vp_deltas = {}
        for pair_name, base_slug, instruct_slug in PAIRS:
            base_vp, instruct_vp, base_q, instruct_q = model_profiles(
                base_slug, instruct_slug, score_field, label, survey
            )
            keys, vp_delta = delta_vector(base_vp, instruct_vp)
            q_keys, q_delta = delta_vector(base_q, instruct_q)
            if keys != q_keys:
                q_map = dict(zip(q_keys, q_delta))
                q_delta = np.asarray([q_map[key] for key in keys])

            base_ranks = rank_map(base_vp)
            instruct_ranks = rank_map(instruct_vp)
            vp_deltas[pair_name] = dict(zip(keys, vp_delta))
            rows.append({
                "pair": pair_name,
                "vp_base_instruct_rho": round(profile_rho(base_vp, instruct_vp), 4),
                "questionnaire_base_instruct_rho": round(
                    profile_rho(base_q, instruct_q), 4
                ),
                "delta_vp_vs_questionnaire_cosine": round(cosine(vp_delta, q_delta), 4),
                "vp_delta_z": {key: round(float(value), 4) for key, value in zip(keys, vp_delta)},
                "vp_base_rank": {key: base_ranks[key] for key in keys},
                "vp_instruct_rank": {key: instruct_ranks[key] for key in keys},
            })

        pair_delta_cosines = []
        pair_names = list(vp_deltas)
        for left, right in combinations(pair_names, 2):
            keys = sorted(set(vp_deltas[left]) & set(vp_deltas[right]))
            left_delta = np.asarray([vp_deltas[left][key] for key in keys])
            right_delta = np.asarray([vp_deltas[right][key] for key in keys])
            pair_delta_cosines.append({
                "pairs": [left, right],
                "cosine": round(cosine(left_delta, right_delta), 4),
            })

        construct_directions = {}
        keys = sorted(next(iter(vp_deltas.values())))
        for key in keys:
            deltas = [vp_deltas[pair][key] for pair in pair_names]
            construct_directions[key] = {
                "n_up": int(sum(value > 0 for value in deltas)),
                "n_down": int(sum(value < 0 for value in deltas)),
                "mean_delta_z": round(float(np.mean(deltas)), 4),
            }

        output["domains"][domain] = {
            "per_pair": rows,
            "mean_vp_base_instruct_rho": round(
                float(np.mean([row["vp_base_instruct_rho"] for row in rows])), 4
            ),
            "mean_questionnaire_base_instruct_rho": round(
                float(np.mean([row["questionnaire_base_instruct_rho"] for row in rows])), 4
            ),
            "mean_delta_vp_vs_questionnaire_cosine": round(
                float(np.mean([row["delta_vp_vs_questionnaire_cosine"] for row in rows])), 4
            ),
            "pairwise_vp_delta_cosines": pair_delta_cosines,
            "mean_pairwise_vp_delta_cosine": round(
                float(np.mean([row["cosine"] for row in pair_delta_cosines])), 4
            ),
            "construct_directions": construct_directions,
        }
    return output


def make_report(results):
    n_pairs = len(PAIRS)
    lines = [
        "# Post-training-signature analysis",
        "",
        f"All comparisons use {n_pairs} matched base/instruction pairs, identical plain ",
        "prompts, and within-profile standardization. Results are descriptive because ",
        f"only {n_pairs} pairs are available.",
    ]
    for result in results:
        score = result["score_field"]
        values = result["domains"]["values"]
        lines.extend([
            "",
            f"## Schwartz values: `{score}`",
            "",
            "| Pair | VP rho | PVQ rho | cosine(delta VP, delta PVQ) | Hedonism rank | Power rank |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for row in values["per_pair"]:
            h0 = row["vp_base_rank"]["hedonism"]
            h1 = row["vp_instruct_rank"]["hedonism"]
            p0 = row["vp_base_rank"]["power"]
            p1 = row["vp_instruct_rank"]["power"]
            lines.append(
                f"| {row['pair']} | {row['vp_base_instruct_rho']:.2f} | "
                f"{row['questionnaire_base_instruct_rho']:.2f} | "
                f"{row['delta_vp_vs_questionnaire_cosine']:.2f} | "
                f"{h0}->{h1} | {p0}->{p1} |"
            )
        lines.extend([
            "",
            f"Mean pairwise cosine among VP change vectors: "
            f"{values['mean_pairwise_vp_delta_cosine']:.2f}.",
            f"Mean within-pair cosine between VP and PVQ change vectors: "
            f"{values['mean_delta_vp_vs_questionnaire_cosine']:.2f}.",
        ])
    return "\n".join(lines) + "\n"


def main():
    results = [analyze("total_logprob"), analyze("normalized_logprob")]
    (HERE / "posttraining_signature.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False)
    )
    report = make_report(results)
    (HERE / "posttraining_signature.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
