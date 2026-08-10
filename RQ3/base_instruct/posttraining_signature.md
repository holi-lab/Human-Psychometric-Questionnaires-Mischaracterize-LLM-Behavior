# Post-training-signature analysis

All comparisons use 4 matched base/instruction pairs, identical plain 
prompts, and within-profile standardization. Results are descriptive because 
only 4 pairs are available.

## Schwartz values: `total_logprob`

| Pair | VP rho | PVQ rho | cosine(delta VP, delta PVQ) | Hedonism rank | Power rank |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-7B | 0.94 | 0.87 | 0.14 | 10->10 | 9->9 |
| Gemma3-4B | 0.73 | 0.15 | -0.09 | 10->10 | 8->7 |
| Gemma3-27B | 0.62 | 0.88 | -0.14 | 9->10 | 10->7 |
| Qwen3-30B-A3B | 0.85 | 0.85 | -0.30 | 10->10 | 9->8 |

Mean pairwise cosine among VP change vectors: 0.78.
Mean within-pair cosine between VP and PVQ change vectors: -0.10.

## Schwartz values: `normalized_logprob`

| Pair | VP rho | PVQ rho | cosine(delta VP, delta PVQ) | Hedonism rank | Power rank |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-7B | 0.98 | 0.87 | 0.05 | 7->7 | 8->8 |
| Gemma3-4B | 0.83 | 0.15 | -0.50 | 8->7 | 9->8 |
| Gemma3-27B | 0.70 | 0.88 | -0.15 | 7->7 | 9->8 |
| Qwen3-30B-A3B | 0.95 | 0.85 | 0.03 | 7->7 | 8->8 |

Mean pairwise cosine among VP change vectors: 0.76.
Mean within-pair cosine between VP and PVQ change vectors: -0.14.
