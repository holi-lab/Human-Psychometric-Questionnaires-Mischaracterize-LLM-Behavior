<div align="center">

# Human Psychometric Questionnaires Mischaracterize LLM Behavior

<p align="center">
  <a href="https://arxiv.org/pdf/2509.10078"><img src="https://img.shields.io/badge/Paper-PDF-b31b1b" alt="Paper"></a>
  <a href="https://arxiv.org/abs/2509.10078"><img src="https://img.shields.io/badge/arXiv-2509.10078-b31b1b" alt="arXiv"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT%20%2B%20CC--BY--4.0-green" alt="License"></a>
</p>

</div>

## Abstract

We examine whether human psychometric questionnaires can serve as reliable tools for characterizing
and predicting LLM behavior in everyday user interactions. We analyze eight open-source LLMs by
comparing their value and personality profiles derived from two different methods: Likert
self-reports on established questionnaires (PVQ-40/21 and BFI-44/10) and generation probabilities
over value-laden responses to everyday user queries. The two profiles diverge substantially.
Within-construct item consistency, often cited as evidence of stable LLM dispositions, disappears in
generation probabilities. We attribute this gap to the fact that explicit lexical cues in
established questionnaire items allow models to recognize the target construct and respond in
alignment-consistent, socially desirable ways, whereas realistic user queries provide no such cues.
In addition, demographic persona prompts shift models' responses to human questionnaires in ways
consistent with real human patterns, but no such shifts appear in the generation probabilities of
responses to realistic user queries, showing their limited ability to simulate the behaviors of
target demographics in real-world user interactions. Overall, our study shows that human
psychometric questionnaires are insufficient tools for predicting LLM behavior and suggests
generation-based profiling as a more accurate measure.

This repository contains the code, data, and experimental results for the study.

## Repository Layout

The layout mirrors the paper's research questions; appendix analyses live in subdirectories of
the RQ they support, and keep their outputs next to their scripts.

```
├── prompts/                # Prompt templates for all experiments
├── surveys/                # Survey instruments (JSON); VP.json = Value Portrait dataset
├── figures/                # Paper figures (PDF)
├── RQ1/                    # Established vs. generation probability profiles
│   ├── established.py, ecological.py, ecological_profile.py
│   ├── analysis.py, plot_rank_divergence.py
│   ├── sampling_validity.py    # appendix: sampling log-probability validity
│   └── length_norm/            # appendix: length-normalization ablation
├── RQ2/                    # Intra-construct response consistency (η², WMV)
│   ├── aic_established.py, aic_ecological.py
│   └── human_reference/        # appendix: held-out human reference
├── RQ3/                    # Item-construct recognition & its consequences
│   ├── item_construct_recognition.py, summarize_item_construct_f1.py
│   ├── embedding_analysis.py, embedding_robustness.py
│   ├── cue_reduction/          # appendix: cue-reduced questionnaire items
│   ├── base_instruct/          # appendix: base vs. instruction-tuned pairs
│   └── cueless_2x2/            # appendix: checkpoint × item-transparency 2×2
├── RQ4/                    # Persona-induced demographic shifts
│   ├── established_circumstance.py, ecological_circumstance.py
│   ├── analysis_circumstance_vs_human.py, analyze_*.py
│   └── ess_human_aggregates.json   # ESS human reference (subgroup aggregates)
└── results/                # Experimental outputs per RQ
```

## Models

Eight open-source LLMs across four families:

| Family | Small | Large |
|--------|-------|-------|
| Gemma 3 | 4B | 27B |
| GPT-OSS | 20B | 120B |
| Qwen 2.5 | 7B | 72B |
| Qwen 3 | 30B-A3B (MoE) | 235B-A22B (MoE) |

The post-training analyses additionally use the matched base checkpoints of Qwen2.5-7B,
Gemma3-4B, Gemma3-27B, and Qwen3-30B-A3B.

## Setup

- Python >= 3.10, `pip install -r requirements.txt`
- [vLLM](https://github.com/vllm-project/vllm) >= 0.16.0 for model inference
  (4× NVIDIA A100 80GB were used for the paper)

You do **not** need to start a vLLM server yourself: each GPU experiment script launches and
manages its own server on a script-specific port (8000/8001/8011/8012), and aborts with a clear
error if the port is taken. Server logs go to a gitignored `logs/` directory. The embedding
analyses download encoder checkpoints from the Hugging Face Hub on first run.

## Running Experiments

All scripts run from the repository root. Analysis scripts (no GPU) work out of the box on the
shipped results.

### RQ1 — Established vs. generation probability profiles

```bash
python RQ1/established.py            # questionnaire responses
python RQ1/ecological.py             # generation probability scores
python RQ1/ecological_profile.py     # aggregate into profiles
python RQ1/analysis.py               # Spearman ρ / NDCG comparison
python RQ1/plot_rank_divergence.py   # figures

# appendix
python RQ1/sampling_validity.py
python RQ1/length_norm/build_profiles.py --score-field normalized_logprob \
    --output-dir RQ1/length_norm/profiles_normalized
python RQ1/analysis.py --ecological-dir RQ1/length_norm/profiles_normalized \
    --output-path RQ1/length_norm/analysis_normalized.json
python RQ1/length_norm/make_report.py
```

### RQ2 — Intra-construct response consistency

```bash
python RQ2/aic_established.py
python RQ2/aic_ecological.py

# appendix: held-out human reference — clone https://github.com/holi-lab/ValuePortrait
# next to this repository (or set VALUEPORTRAIT_REPO to its location)
python RQ2/human_reference/run_item_level_reference.py
```

### RQ3 — Item-construct recognition

```bash
python RQ3/item_construct_recognition.py
python RQ3/summarize_item_construct_f1.py
python RQ3/embedding_analysis.py
python RQ3/embedding_robustness.py

# appendix: cue-reduced items
python RQ3/cue_reduction/scripts/build_cueless_surveys.py
python RQ3/cue_reduction/scripts/check_cueless_embedding.py
python RQ3/cue_reduction/scripts/run_established_cueless.py
python RQ3/cue_reduction/scripts/run_recognition_cueless.py
python RQ3/cue_reduction/scripts/analyze_cueless.py
python RQ3/cue_reduction/scripts/summarize_recognition_cueless.py

# appendix: base vs. instruct + 2×2 (one run per checkpoint; larger ones need --tp 2)
python RQ3/base_instruct/run.py --model google/gemma-3-4b-pt --format plain --out-tag base
python RQ3/base_instruct/run.py --model google/gemma-3-4b-it --format plain --out-tag instruct_plain
python RQ3/base_instruct/analysis.py
python RQ3/cueless_2x2/run.py --model google/gemma-3-4b-pt --format plain --task established --out-tag base
python RQ3/cueless_2x2/run.py --model google/gemma-3-4b-it --format plain --task established --out-tag instruct_plain
python RQ3/cueless_2x2/analysis.py   # uses the base_instruct results above
python RQ3/base_instruct/run_posttraining_signature.py
```

### RQ4 — Persona-induced demographic shifts

```bash
python RQ4/established_circumstance.py
python RQ4/ecological_circumstance.py
python RQ4/analysis_circumstance_vs_human.py
python RQ4/analyze_genprob_delta.py
python RQ4/analyze_genprob_rigorous.py
python RQ4/analyze_cross_model_consensus.py
```

## Notes

- **Shipped vs. regenerable.** Every analysis output backing the paper is shipped. Only the
  bulky raw inference dumps (`results/RQ3/item_construct_recognition/`,
  `results/RQ4/ecological_circumstance/`) are excluded; the corresponding run scripts
  regenerate them. Summarizers refuse to overwrite shipped results with missing or mismatched
  inputs unless `--force` is given.
- **Naming.** *ecological* is the in-repo name for the generation-probability measure;
  `aic` stands for *assessing intra-construct consistency* (not Akaike).
- **Coverage.** The recognition tables report seven models (GPT-OSS-20B appears only in the
  cue-reduction comparison); RQ4 uses seven models (gemma-3-4b-it excluded) and the Schwartz
  value instruments only, since the ESS reference provides no Big Five data.
- **Prompt files.** `*_reversed.txt` uses the high-to-low option order ("Variant 1" in the
  paper); the `Instruction :` / `Instruction:` spacing differences are preserved verbatim
  from data collection.

## License

- **Code** (all `*.py`): [MIT](LICENSE).
- **Data and results** (`surveys/`, `prompts/`, `results/`, `figures/`, and result files in
  the `RQ*/` directories): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
  Please cite the paper when using `surveys/VP.json` (the Value Portrait dataset).
- `RQ4/ess_human_aggregates.json` contains subgroup-level statistics derived from the
  [European Social Survey](https://www.europeansocialsurvey.org/); cite the ESS when reusing.
- The questionnaire files (`PVQ*.json`, `BFI*.json`) reproduce established psychometric
  instruments; copyright in the item text remains with their authors.

## Citation

```bibtex
@misc{song2026humanpsychometricquestionnairesmischaracterize,
      title={Human Psychometric Questionnaires Mischaracterize LLM Behavior}, 
      author={Woojung Song and Dongmin Choi and Yoonah Park and Jongwook Han and Eun-Ju Lee and Yohan Jo},
      year={2026},
      eprint={2509.10078},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.10078}, 
}
```
