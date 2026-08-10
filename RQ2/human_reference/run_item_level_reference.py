#!/usr/bin/env python3
"""Cross-fitted, response-level held-out human reference (Appendix, RQ1/RQ2).

This analysis avoids constructing a profile from the eight scenarios seen by a
typical participant. It instead uses the 41--48 independent ratings available
for each of the 520 VP responses.

Two questions are kept separate:

1. Item-level criterion validity (the human-reference result): tags are learned
   from training raters with the original VP rule (Spearman rho >= .3). On
   held-out raters, we test whether questionnaire congruence predicts which of
   the five responses to a scenario they report as matching their thinking.
   The pooled regression removes response fixed effects and rater-by-scenario
   fixed effects. A within-scenario permutation of complete response-tag
   vectors provides the null distribution.

2. RQ2 calibration (a supplementary positive control): tags learned on one
   rater half are applied to mean response ratings from the other half. Human
   eta-squared/WMV are then computed with exactly the response -> scenario ->
   construct aggregation used for LLM generation probabilities. The same tags
   are also applied to summed and length-normalized LLM log probabilities.

The human outcome is an explicit similarity/preference rating, not observed
behavior. Results therefore must not be described as a matched behavioral
baseline or as a directly comparable human-vs-LLM effect size.

Usage:
    python RQ2/human_reference/run_item_level_reference.py
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

HERE = Path(__file__).resolve().parent
OFFICIAL = HERE.parent.parent
# Human validation-study ratings from the Value Portrait release.
# Clone https://github.com/holi-lab/ValuePortrait next to this repository,
# or set the VALUEPORTRAIT_REPO environment variable to its location.
import os  # noqa: E402

VP_REPO = (
    Path(os.environ.get("VALUEPORTRAIT_REPO", OFFICIAL.parent / "ValuePortrait"))
    / "data"
    / "prolific"
)
sys.path.insert(0, str(OFFICIAL / "RQ2"))
from RQ2.aic_ecological import compute_eta_squared, compute_wmv  # noqa: E402

SEED = 42
TAG_THRESHOLD = 0.3
MIN_RATERS = 10
N_REFERENCE_FOLDS = 5
N_HALF_SPLITS = 20
N_PERMUTATIONS = 1_000

PVQ_KEYS = [
    "Conformity", "Tradition", "Benevolence", "Universalism",
    "Self_Direction", "Stimulation", "Hedonism", "Achievement", "Power",
    "Security",
]
BFI_KEYS = [
    "Extraversion", "Agreeableness", "Conscientiousness", "Neuroticism",
    "Openness",
]
DOMAINS = [
    ("values", PVQ_KEYS, "pvq"),
    ("traits", BFI_KEYS, "bfi"),
]
MODELS = [
    "gemma-3-27b-it", "gemma-3-4b-it", "gpt-oss-120b", "gpt-oss-20b",
    "Qwen2.5-72B-Instruct", "Qwen2.5-7B-Instruct",
    "Qwen3-235B-A22B-Instruct-2507-FP8",
    "Qwen3-30B-A3B-Instruct-2507",
]


def load_data():
    survey_path = VP_REPO / "survey" / "main" / "survey.json"
    survey = json.loads(survey_path.read_text())
    value = json.loads((VP_REPO / "value" / "main" / "value.json").read_text())

    ratings = defaultdict(dict)
    item_raters = defaultdict(list)
    participant_batches = defaultdict(set)
    for batch_id, batch in enumerate(survey):
        for scenario_id, options in batch.items():
            for option in options:
                item = (int(scenario_id), int(option["option_id"]))
                for rating, participant in option["data"]:
                    if participant not in value:
                        continue
                    ratings[participant][item] = float(rating)
                    item_raters[item].append((participant, float(rating)))
                    participant_batches[participant].add(batch_id)

    participants = sorted(ratings)
    strata = {
        participant: tuple(sorted(participant_batches[participant]))
        for participant in participants
    }
    return ratings, value, item_raters, participants, strata


def load_llm_scores():
    scores = {"summed": {}, "normalized": {}}
    for model in MODELS:
        raw = json.loads(
            (OFFICIAL / "results" / "RQ1" / "ecological" / f"{model}.json")
            .read_text()
        )
        summed = {}
        normalized = {}
        for scenario in raw["results"]:
            scenario_id = int(scenario["portrait_id"])
            for output in scenario["outputs"]:
                item = (scenario_id, int(output["output_id"]))
                if output.get("total_logprob") is not None:
                    summed[item] = float(output["total_logprob"])
                if output.get("normalized_logprob") is not None:
                    normalized[item] = float(output["normalized_logprob"])
        scores["summed"][model] = summed
        scores["normalized"][model] = normalized
    return scores


def stratified_folds(participants, strata, n_folds, rng):
    by_stratum = defaultdict(list)
    for participant in participants:
        by_stratum[strata[participant]].append(participant)

    fold = {}
    for members in by_stratum.values():
        members = list(rng.permutation(members))
        for index, participant in enumerate(members):
            fold[participant] = index % n_folds
    return fold


def derive_tags(item_raters, value, pool):
    """Re-estimate VP tags with the published Spearman-rho rule."""
    pool = set(pool)
    tags = defaultdict(lambda: {"values": set(), "traits": set()})
    for item, raters in item_raters.items():
        subset = [(p, rating) for p, rating in raters if p in pool]
        if len(subset) < MIN_RATERS:
            continue
        response_ratings = [rating for _, rating in subset]
        if len(set(response_ratings)) < 2:
            continue
        for domain, constructs, section in DOMAINS:
            for construct in constructs:
                questionnaire = [value[p][section][construct] for p, _ in subset]
                if len(set(questionnaire)) < 2:
                    continue
                rho = spearmanr(response_ratings, questionnaire).statistic
                if not np.isnan(rho) and rho >= TAG_THRESHOLD:
                    tags[item][domain].add(construct)
    return tags


def questionnaire_zscores(value, participants):
    result = {domain: {} for domain, _, _ in DOMAINS}
    for domain, constructs, section in DOMAINS:
        matrix = np.array(
            [[value[p][section][construct] for construct in constructs]
             for p in participants],
            dtype=float,
        )
        means = matrix.mean(axis=0)
        stds = matrix.std(axis=0)
        stds[stds == 0] = 1.0
        matrix = (matrix - means) / stds
        for row, participant in zip(matrix, participants):
            result[domain][participant] = dict(zip(constructs, row))
    return result


def encode_groups(keys):
    lookup = {key: index for index, key in enumerate(sorted(set(keys)))}
    return np.array([lookup[key] for key in keys], dtype=int)


def two_way_residualize(values, first_group, second_group, tolerance=1e-11):
    """Alternating projections for two high-dimensional fixed effects."""
    residual = np.asarray(values, dtype=float).copy()
    groups = (first_group, second_group)
    for _ in range(200):
        largest_mean = 0.0
        for group in groups:
            counts = np.bincount(group)
            means = np.bincount(group, weights=residual) / counts
            largest_mean = max(largest_mean, float(np.max(np.abs(means))))
            residual -= means[group]
        if largest_mean < tolerance:
            break
    return residual


def build_oof_congruence(
    ratings, value, item_raters, participants, fold_ids, zscores,
    permuted_items=None, fold_tags=None,
):
    """Build one independent congruence prediction for every observed rating."""
    records = {domain: [] for domain, _, _ in DOMAINS}
    n_folds = max(fold_ids.values()) + 1
    for heldout_fold in range(n_folds):
        heldout = [p for p in participants if fold_ids[p] == heldout_fold]
        if fold_tags is None:
            train = [p for p in participants if fold_ids[p] != heldout_fold]
            tags = derive_tags(item_raters, value, train)
        else:
            tags = fold_tags[heldout_fold]

        for participant in heldout:
            for item, rating in ratings[participant].items():
                source_item = (
                    permuted_items[item] if permuted_items is not None else item
                )
                for domain, _, _ in DOMAINS:
                    selected = tags.get(source_item, {}).get(domain, ())
                    congruence = (
                        float(np.mean([
                            zscores[domain][participant][construct]
                            for construct in selected
                        ]))
                        if selected else 0.0
                    )
                    records[domain].append({
                        "participant": participant,
                        "scenario": item[0],
                        "item": item,
                        "rating": rating,
                        "congruence": congruence,
                        "tagged": bool(selected),
                    })
    return records


def derive_oof_tags(item_raters, value, participants, fold_ids):
    n_folds = max(fold_ids.values()) + 1
    return {
        heldout_fold: derive_tags(
            item_raters,
            value,
            [p for p in participants if fold_ids[p] != heldout_fold],
        )
        for heldout_fold in range(n_folds)
    }


def fixed_effect_result(records):
    ratings = np.array([row["rating"] for row in records], dtype=float)
    congruence = np.array([row["congruence"] for row in records], dtype=float)
    item_group = encode_groups([row["item"] for row in records])
    rater_scenario_group = encode_groups([
        (row["participant"], row["scenario"]) for row in records
    ])
    residual_y = two_way_residualize(ratings, item_group, rater_scenario_group)
    residual_x = two_way_residualize(
        congruence, item_group, rater_scenario_group
    )
    denom = float(np.dot(residual_x, residual_x))
    if denom == 0:
        return {"beta": None, "partial_r": None}
    beta = float(np.dot(residual_x, residual_y) / denom)
    partial_r = float(np.corrcoef(residual_x, residual_y)[0, 1])
    return {
        "beta_rating_points_per_questionnaire_sd": beta,
        "partial_r": partial_r,
        "n_ratings": len(records),
        "n_tagged_ratings": int(sum(row["tagged"] for row in records)),
    }


def prepare_fixed_effect_arrays(
    ratings, participants, fold_ids, zscores, fold_tags, item_raters,
):
    """Cache every possible within-scenario tag-vector assignment.

    Each VP scenario has five responses. For every observed rating, the cache
    stores the congruence obtained from each of those five responses' tag
    vectors. A permutation then becomes one vectorized column selection rather
    than a reconstruction of all records.
    """
    all_items = sorted(item_raters)
    item_to_code = {item: index for index, item in enumerate(all_items)}
    scenario_items = defaultdict(list)
    for item in all_items:
        scenario_items[item[0]].append(item)
    for items in scenario_items.values():
        items.sort()
        if len(items) != 5:
            raise ValueError(f"Expected five responses, found {len(items)}")

    participant_column = []
    scenario_column = []
    item_column = []
    rating_column = []
    candidate = {domain: [] for domain, _, _ in DOMAINS}
    candidate_tagged = {domain: [] for domain, _, _ in DOMAINS}
    observed_weights = {domain: [] for domain, _, _ in DOMAINS}
    construct_index = {
        domain: {construct: index for index, construct in enumerate(constructs)}
        for domain, constructs, _ in DOMAINS
    }

    for participant in participants:
        tags = fold_tags[fold_ids[participant]]
        for item, rating in sorted(ratings[participant].items()):
            participant_column.append(participant)
            scenario_column.append(item[0])
            item_column.append(item)
            rating_column.append(rating)
            for domain, _, _ in DOMAINS:
                row = []
                tagged_row = []
                for source_item in scenario_items[item[0]]:
                    selected = tags.get(source_item, {}).get(domain, ())
                    row.append(
                        float(np.mean([
                            zscores[domain][participant][construct]
                            for construct in selected
                        ]))
                        if selected else 0.0
                    )
                    tagged_row.append(bool(selected))
                candidate[domain].append(row)
                candidate_tagged[domain].append(tagged_row)
                selected = tags.get(item, {}).get(domain, ())
                weights = np.zeros(len(construct_index[domain]), dtype=float)
                if selected:
                    for construct in selected:
                        weights[construct_index[domain][construct]] = 1 / len(selected)
                observed_weights[domain].append(weights)

    item_group = encode_groups(item_column)
    rater_scenario_group = encode_groups(list(zip(
        participant_column, scenario_column
    )))
    residual_y = two_way_residualize(
        np.asarray(rating_column, dtype=float), item_group, rater_scenario_group
    )
    item_codes = np.array([item_to_code[item] for item in item_column], dtype=int)
    participant_to_code = {
        participant: index for index, participant in enumerate(participants)
    }
    participant_codes = np.array(
        [participant_to_code[p] for p in participant_column], dtype=int
    )
    observed_positions = np.zeros(len(all_items), dtype=int)
    for items in scenario_items.values():
        for position, item in enumerate(items):
            observed_positions[item_to_code[item]] = position

    return {
        "all_items": all_items,
        "item_to_code": item_to_code,
        "scenario_items": dict(scenario_items),
        "item_codes": item_codes,
        "participant_codes": participant_codes,
        "participant_to_code": participant_to_code,
        "item_group": item_group,
        "rater_scenario_group": rater_scenario_group,
        "residual_y": residual_y,
        "observed_positions": observed_positions,
        "candidate": {
            domain: np.asarray(rows, dtype=float) for domain, rows in candidate.items()
        },
        "candidate_tagged": {
            domain: np.asarray(rows, dtype=bool)
            for domain, rows in candidate_tagged.items()
        },
        "observed_weights": {
            domain: np.asarray(rows, dtype=float)
            for domain, rows in observed_weights.items()
        },
        "profile_matrix": {
            domain: np.array([
                [zscores[domain][participant][construct]
                 for construct in constructs]
                for participant in participants
            ], dtype=float)
            for domain, constructs, _ in DOMAINS
        },
    }


def fixed_effect_from_cache(cache, domain, source_positions=None):
    if source_positions is None:
        source_positions = cache["observed_positions"]
    observation_positions = source_positions[cache["item_codes"]]
    rows = np.arange(len(cache["item_codes"]))
    congruence = cache["candidate"][domain][rows, observation_positions]
    tagged = cache["candidate_tagged"][domain][rows, observation_positions]
    return fixed_effect_from_congruence(cache, congruence, tagged)


def fixed_effect_from_congruence(cache, congruence, tagged):
    residual_x = two_way_residualize(
        congruence, cache["item_group"], cache["rater_scenario_group"]
    )
    residual_y = cache["residual_y"]
    denom = float(np.dot(residual_x, residual_x))
    if denom == 0:
        return {"beta": None, "partial_r": None}
    return {
        "beta_rating_points_per_questionnaire_sd": float(
            np.dot(residual_x, residual_y) / denom
        ),
        "partial_r": float(np.corrcoef(residual_x, residual_y)[0, 1]),
        "n_ratings": int(len(residual_y)),
        "n_tagged_ratings": int(np.sum(tagged)),
    }


def permuted_profile_donors(participants, strata, fold_ids, rng):
    """Shuffle complete questionnaire profiles within batch and OOF fold."""
    by_block = defaultdict(list)
    for participant in participants:
        by_block[(strata[participant], fold_ids[participant])].append(participant)
    donor = {}
    for members in by_block.values():
        shuffled = list(rng.permutation(members))
        donor.update(zip(members, shuffled))
    participant_to_code = {
        participant: index for index, participant in enumerate(participants)
    }
    return np.array(
        [participant_to_code[donor[participant]] for participant in participants],
        dtype=int,
    )


def fixed_effect_with_profile_donors(cache, domain, donor_codes):
    observation_donors = donor_codes[cache["participant_codes"]]
    donor_profiles = cache["profile_matrix"][domain][observation_donors]
    weights = cache["observed_weights"][domain]
    congruence = np.sum(weights * donor_profiles, axis=1)
    tagged = np.sum(weights, axis=1) > 0
    return fixed_effect_from_congruence(cache, congruence, tagged)


def permuted_source_positions(cache, rng):
    positions = np.zeros(len(cache["all_items"]), dtype=int)
    for scenario_items in cache["scenario_items"].values():
        permutation = rng.permutation(len(scenario_items))
        for target_item, source_position in zip(scenario_items, permutation):
            positions[cache["item_to_code"][target_item]] = source_position
    return positions


def scenario_response_permutation(items, rng):
    by_scenario = defaultdict(list)
    for item in items:
        by_scenario[item[0]].append(item)
    mapping = {}
    for scenario_items in by_scenario.values():
        sources = [scenario_items[index]
                   for index in rng.permutation(len(scenario_items))]
        mapping.update(zip(scenario_items, sources))
    return mapping


def item_pair_diagnostics(item_raters, value, tags, heldout):
    heldout = set(heldout)
    result = {domain: {"selected": [], "unselected": [], "composite": []}
              for domain, _, _ in DOMAINS}
    for item, raters in item_raters.items():
        subset = [(p, rating) for p, rating in raters if p in heldout]
        if len(subset) < MIN_RATERS:
            continue
        response_ratings = np.array([rating for _, rating in subset], dtype=float)
        if np.std(response_ratings) == 0:
            continue
        for domain, constructs, section in DOMAINS:
            selected_constructs = tags.get(item, {}).get(domain, set())
            selected_scores = []
            for construct in constructs:
                questionnaire = np.array(
                    [value[p][section][construct] for p, _ in subset], dtype=float
                )
                rho = spearmanr(response_ratings, questionnaire).statistic
                if np.isnan(rho):
                    continue
                key = "selected" if construct in selected_constructs else "unselected"
                result[domain][key].append((item[0], float(rho)))
                if construct in selected_constructs:
                    selected_scores.append(rankdata(questionnaire))
            if selected_scores:
                composite = np.mean(np.column_stack(selected_scores), axis=1)
                rho = spearmanr(response_ratings, composite).statistic
                if not np.isnan(rho):
                    result[domain]["composite"].append((item[0], float(rho)))
    return result


def response_scores(members, ratings, scenario_center=False):
    accumulator = defaultdict(list)
    for participant in members:
        by_scenario = defaultdict(list)
        for item, rating in ratings[participant].items():
            by_scenario[item[0]].append((item, rating))
        for scenario_ratings in by_scenario.values():
            center = (
                float(np.mean([rating for _, rating in scenario_ratings]))
                if scenario_center else 0.0
            )
            for item, rating in scenario_ratings:
                accumulator[item].append(rating - center)
    return {item: float(np.mean(values)) for item, values in accumulator.items()}


def construct_groups(scores, tags, domain, item_mapping=None):
    per_scenario = defaultdict(lambda: defaultdict(list))
    for item, score in scores.items():
        tag_item = item_mapping[item] if item_mapping is not None else item
        for construct in tags.get(tag_item, {}).get(domain, ()):
            per_scenario[item[0]][construct].append(score)
    groups = defaultdict(list)
    for construct_scores in per_scenario.values():
        for construct, values in construct_scores.items():
            groups[construct].append(float(np.mean(values)))
    return dict(groups)


def structure_metrics(groups):
    eta = compute_eta_squared(groups)
    return {
        "eta_squared": eta["eta_squared"],
        "analytical_null": eta["analytical_baseline"],
        "wmv": compute_wmv(groups),
        "N": eta["N"],
        "K": eta["K"],
    }


def cluster_summary(records, rng):
    by_scenario = defaultdict(list)
    for scenario, value in records:
        by_scenario[scenario].append(value)
    scenario_means = np.array(
        [np.mean(values) for values in by_scenario.values()], dtype=float
    )
    draws = np.mean(
        rng.choice(scenario_means, size=(10_000, len(scenario_means))), axis=1
    )
    return {
        "scenario_weighted_mean_rho": float(np.mean(scenario_means)),
        "scenario_cluster_bootstrap_95ci": [
            float(np.percentile(draws, 2.5)),
            float(np.percentile(draws, 97.5)),
        ],
        "positive_fraction": float(np.mean([value > 0 for _, value in records])),
        "n_records_across_repeated_folds": len(records),
        "n_scenarios": len(scenario_means),
    }


def exact_signflip_p(differences):
    differences = np.asarray(differences, dtype=float)
    observed = float(np.mean(differences))
    exceed = 0
    for mask in range(2 ** len(differences)):
        signs = np.array([
            1.0 if (mask >> index) & 1 else -1.0
            for index in range(len(differences))
        ])
        exceed += float(np.mean(signs * differences) >= observed - 1e-12)
    return exceed / (2 ** len(differences))


def mean_sd(values):
    array = np.asarray(values, dtype=float)
    return {"mean": float(np.mean(array)), "sd_over_fits": float(np.std(array))}


def main():
    ratings, value, item_raters, participants, strata = load_data()
    llm_scores = load_llm_scores()
    zscores = questionnaire_zscores(value, participants)
    rng = np.random.default_rng(SEED)

    item_counts = np.array([len(raters) for raters in item_raters.values()])
    coverage = {
        "participants": len(participants),
        "scenarios": len({item[0] for item in item_raters}),
        "responses": len(item_raters),
        "raters_per_response": {
            "min": int(item_counts.min()),
            "median": float(np.median(item_counts)),
            "max": int(item_counts.max()),
        },
        "ratings_per_participant": dict(Counter(
            len(person_ratings) for person_ratings in ratings.values()
        )),
    }

    # A. One fixed, batch-stratified 5-fold OOF analysis.
    reference_folds = stratified_folds(
        participants, strata, N_REFERENCE_FOLDS, np.random.default_rng(SEED)
    )
    reference_tags = derive_oof_tags(
        item_raters, value, participants, reference_folds
    )
    fe_cache = prepare_fixed_effect_arrays(
        ratings, participants, reference_folds, zscores, reference_tags,
        item_raters,
    )
    item_validity = {
        domain: {"fixed_effect_model": fixed_effect_from_cache(fe_cache, domain)}
        for domain, _, _ in DOMAINS
    }

    tag_vector_null = {domain: [] for domain, _, _ in DOMAINS}
    profile_null = {domain: [] for domain, _, _ in DOMAINS}
    all_items = sorted(item_raters)
    for _ in range(N_PERMUTATIONS):
        source_positions = permuted_source_positions(fe_cache, rng)
        donor_codes = permuted_profile_donors(
            participants, strata, reference_folds, rng
        )
        for domain in tag_vector_null:
            tag_vector_null[domain].append(
                fixed_effect_from_cache(
                    fe_cache, domain, source_positions
                )["partial_r"]
            )
            profile_null[domain].append(
                fixed_effect_with_profile_donors(
                    fe_cache, domain, donor_codes
                )["partial_r"]
            )
    for domain in item_validity:
        observed = item_validity[domain]["fixed_effect_model"]["partial_r"]
        null_values = np.asarray(tag_vector_null[domain], dtype=float)
        item_validity[domain]["within_scenario_tag_vector_permutation"] = {
            "n_permutations": N_PERMUTATIONS,
            "null_mean": float(np.mean(null_values)),
            "null_95_interval": [
                float(np.percentile(null_values, 2.5)),
                float(np.percentile(null_values, 97.5)),
            ],
            "p_one_sided": float(
                (1 + np.sum(null_values >= observed - 1e-12))
                / (N_PERMUTATIONS + 1)
            ),
        }
        profile_null_values = np.asarray(profile_null[domain], dtype=float)
        item_validity[domain]["batch_blocked_profile_permutation"] = {
            "n_permutations": N_PERMUTATIONS,
            "null_mean": float(np.mean(profile_null_values)),
            "null_95_interval": [
                float(np.percentile(profile_null_values, 2.5)),
                float(np.percentile(profile_null_values, 97.5)),
            ],
            "p_one_sided": float(
                (1 + np.sum(profile_null_values >= observed - 1e-12))
                / (N_PERMUTATIONS + 1)
            ),
        }

    # Repeated 2-fold diagnostics and the RQ2-style human/LLM calibration.
    diagnostics = {
        domain: {"selected": [], "unselected": [], "composite": []}
        for domain, _, _ in DOMAINS
    }
    structure_fits = {domain: [] for domain, _, _ in DOMAINS}
    first_split = None
    split_rng = np.random.default_rng(SEED)
    for split in range(N_HALF_SPLITS):
        half_ids = stratified_folds(participants, strata, 2, split_rng)
        halves = [
            [p for p in participants if half_ids[p] == fold] for fold in range(2)
        ]
        if first_split is None:
            first_split = halves
        for train_index, eval_index in ((0, 1), (1, 0)):
            train, heldout = halves[train_index], halves[eval_index]
            tags = derive_tags(item_raters, value, train)
            diagnostic = item_pair_diagnostics(item_raters, value, tags, heldout)
            for domain in diagnostics:
                for key in diagnostics[domain]:
                    diagnostics[domain][key].extend(diagnostic[domain][key])

            human_raw = response_scores(heldout, ratings)
            human_centered = response_scores(heldout, ratings, scenario_center=True)
            for domain, _, _ in DOMAINS:
                fit = {
                    "human_raw": structure_metrics(
                        construct_groups(human_raw, tags, domain)
                    ),
                    "human_scenario_centered": structure_metrics(
                        construct_groups(human_centered, tags, domain)
                    ),
                    "llm": {score_type: {} for score_type in llm_scores},
                }
                for score_type, model_scores in llm_scores.items():
                    for model, scores in model_scores.items():
                        fit["llm"][score_type][model] = structure_metrics(
                            construct_groups(scores, tags, domain)
                        )
                structure_fits[domain].append(fit)

    for domain in item_validity:
        item_validity[domain]["pair_diagnostic_repeated_two_fold"] = {
            key: cluster_summary(values, rng)
            for key, values in diagnostics[domain].items()
        }

    rq2_reference = {}
    for domain, fits in structure_fits.items():
        human_eta = [fit["human_raw"]["eta_squared"] for fit in fits]
        human_wmv = [fit["human_raw"]["wmv"] for fit in fits]
        human_null = [fit["human_raw"]["analytical_null"] for fit in fits]
        centered_eta = [
            fit["human_scenario_centered"]["eta_squared"] for fit in fits
        ]
        centered_wmv = [fit["human_scenario_centered"]["wmv"] for fit in fits]
        domain_result = {
            "human_preference": {
                "eta_squared": mean_sd(human_eta),
                "wmv": mean_sd(human_wmv),
                "analytical_null_eta": mean_sd(human_null),
                "scenario_centered_sensitivity": {
                    "eta_squared": mean_sd(centered_eta),
                    "wmv": mean_sd(centered_wmv),
                },
            },
            "llm_generation": {},
        }
        for score_type in llm_scores:
            per_model_eta = {}
            per_model_wmv = {}
            for model in MODELS:
                per_model_eta[model] = float(np.mean([
                    fit["llm"][score_type][model]["eta_squared"] for fit in fits
                ]))
                per_model_wmv[model] = float(np.mean([
                    fit["llm"][score_type][model]["wmv"] for fit in fits
                ]))
            eta_differences = [
                np.mean(human_eta) - per_model_eta[model] for model in MODELS
            ]
            wmv_differences = [
                per_model_wmv[model] - np.mean(human_wmv) for model in MODELS
            ]
            domain_result["llm_generation"][score_type] = {
                "mean_eta_squared_across_models": float(np.mean(
                    list(per_model_eta.values())
                )),
                "mean_wmv_across_models": float(np.mean(
                    list(per_model_wmv.values())
                )),
                "per_model_eta_squared": per_model_eta,
                "per_model_wmv": per_model_wmv,
                "human_eta_above_models": int(sum(d > 0 for d in eta_differences)),
                "eta_exact_signflip_p_one_sided": exact_signflip_p(
                    eta_differences
                ),
                "human_wmv_below_models": int(sum(d > 0 for d in wmv_differences)),
                "wmv_exact_signflip_p_one_sided": exact_signflip_p(
                    wmv_differences
                ),
            }
        rq2_reference[domain] = domain_result

    # Formal conditional permutation for each direction of the fixed half split.
    fixed_split_tests = {domain: [] for domain, _, _ in DOMAINS}
    for direction, (train, heldout) in enumerate(
        ((first_split[0], first_split[1]), (first_split[1], first_split[0]))
    ):
        tags = derive_tags(item_raters, value, train)
        heldout_scores = response_scores(heldout, ratings)
        for domain, _, _ in DOMAINS:
            observed_groups = construct_groups(heldout_scores, tags, domain)
            observed = structure_metrics(observed_groups)
            perm_eta = []
            perm_wmv = []
            perm_rng = np.random.default_rng(SEED + 100 + direction)
            for _ in range(N_PERMUTATIONS):
                mapping = scenario_response_permutation(all_items, perm_rng)
                permuted = structure_metrics(construct_groups(
                    heldout_scores, tags, domain, item_mapping=mapping
                ))
                perm_eta.append(permuted["eta_squared"])
                perm_wmv.append(permuted["wmv"])
            fixed_split_tests[domain].append({
                "direction": "A_to_B" if direction == 0 else "B_to_A",
                "observed": observed,
                "permutation": {
                    "n": N_PERMUTATIONS,
                    "eta_null_mean": float(np.mean(perm_eta)),
                    "eta_p_one_sided": float(
                        (1 + np.sum(np.asarray(perm_eta) >= observed["eta_squared"]))
                        / (N_PERMUTATIONS + 1)
                    ),
                    "wmv_null_mean": float(np.mean(perm_wmv)),
                    "wmv_p_one_sided": float(
                        (1 + np.sum(np.asarray(perm_wmv) <= observed["wmv"]))
                        / (N_PERMUTATIONS + 1)
                    ),
                },
            })
    for domain in rq2_reference:
        rq2_reference[domain]["fixed_split_response_permutation"] = (
            fixed_split_tests[domain]
        )

    report = {
        "scope": (
            "Item-level human reference using explicit VP similarity ratings; "
            "not observed behavior and not a matched human-vs-LLM effect size."
        ),
        "config": {
            "tag_statistic": "Spearman rho (matches released VP tagging statistic)",
            "tag_threshold": TAG_THRESHOLD,
            "min_raters": MIN_RATERS,
            "reference_folds": N_REFERENCE_FOLDS,
            "repeated_half_splits": N_HALF_SPLITS,
            "permutations": N_PERMUTATIONS,
            "seed": SEED,
        },
        "coverage": coverage,
        "item_level_criterion_validity": item_validity,
        "rq2_item_structure_reference": rq2_reference,
    }
    output = HERE / "report_item_level_reference.json"
    output.write_text(json.dumps(report, indent=2))

    print("Coverage:", coverage)
    for domain in item_validity:
        fe = item_validity[domain]["fixed_effect_model"]
        perm = item_validity[domain]["batch_blocked_profile_permutation"]
        pair = item_validity[domain]["pair_diagnostic_repeated_two_fold"]
        print(f"[{domain}] item FE: partial r={fe['partial_r']:.3f}, "
              f"beta={fe['beta_rating_points_per_questionnaire_sd']:.3f}, "
              f"p={perm['p_one_sided']:.4f}")
        print(f"[{domain}] selected pair rho="
              f"{pair['selected']['scenario_weighted_mean_rho']:.3f}; "
              f"composite rho={pair['composite']['scenario_weighted_mean_rho']:.3f}")
        human = rq2_reference[domain]["human_preference"]
        print(f"[{domain}] human eta={human['eta_squared']['mean']:.3f}, "
              f"WMV={human['wmv']['mean']:.3f}, "
              f"null eta={human['analytical_null_eta']['mean']:.3f}")
        for score_type in ("summed", "normalized"):
            llm = rq2_reference[domain]["llm_generation"][score_type]
            print(f"[{domain}] LLM {score_type}: eta="
                  f"{llm['mean_eta_squared_across_models']:.3f}, "
                  f"WMV={llm['mean_wmv_across_models']:.3f}")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
