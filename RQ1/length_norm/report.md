# Length-normalization ablation (RQ1)

Scoring: paper Eq.1 = **sum** of token logprobs; ablation = **per-token mean** (`normalized_logprob`, stored per response in raw results — no re-inference needed).

## Within- vs cross-method agreement (sign-flip permutation test)

| block | within | cross | perm p |
|---|---|---|---|
| spearman_rho/pvq_values (sum) | 0.742 | 0.295 | 0.003906 |
| spearman_rho/bfi_traits (sum) | 0.773 | 0.189 | 0.015625 |
| spearman_rho/overall (sum) | - | - | 0.000168 |
| ndcg/pvq_values (sum) | 0.909 | 0.805 | 0.046875 |
| ndcg/bfi_traits (sum) | 0.887 | 0.726 | 0.015625 |
| ndcg/overall (sum) | - | - | 0.001694 |
| spearman_rho/pvq_values (normalized) | 0.742 | -0.099 | 0.003906 |
| spearman_rho/bfi_traits (normalized) | 0.773 | 0.154 | 0.011719 |
| spearman_rho/overall (normalized) | - | - | 4.6e-05 |
| ndcg/pvq_values (normalized) | 0.909 | 0.546 | 0.003906 |
| ndcg/bfi_traits (normalized) | 0.887 | 0.742 | 0.015625 |
| ndcg/overall (normalized) | - | - | 6.1e-05 |

## Per-model Spearman rho (Gen vs established)

| comparison | gemma-3-27b-it | gemma-3-4b-it | gpt-oss-120b | gpt-oss-20b | Qwen2.5-72B | Qwen2.5-7B | Qwen3-235B | Qwen3-30B | Avg |
|---|---|---|---|---|---|---|---|---|---|
| PVQ_pvq_values (sum) | 0.26 | 0.10 | 0.27 | 0.26 | 0.72 | 0.45 | 0.52 | -0.08 | 0.31 |
| PVQ_pvq_values (norm) | 0.15 | -0.36 | -0.08 | 0.13 | 0.55 | -0.27 | 0.14 | -0.58 | -0.04 |
| PVQ21_pvq_values (sum) | 0.34 | 0.24 | 0.36 | -0.11 | 0.64 | 0.40 | 0.46 | -0.10 | 0.28 |
| PVQ21_pvq_values (norm) | -0.06 | -0.43 | -0.20 | -0.20 | 0.44 | -0.15 | -0.08 | -0.60 | -0.16 |
| BFI44_bfi_traits (sum) | -0.50 | 0.30 | 0.20 | 0.60 | 0.70 | 0.41 | 0.90 | -0.50 | 0.26 |
| BFI44_bfi_traits (norm) | -0.80 | 0.30 | -0.50 | 0.70 | 0.70 | 0.82 | 0.40 | -0.30 | 0.17 |
| BFI10_bfi_traits (sum) | -0.80 | 0.30 | 0.10 | 0.36 | 0.90 | 0.67 | 0.05 | -0.67 | 0.11 |
| BFI10_bfi_traits (norm) | -0.50 | 0.30 | -0.30 | 0.82 | 0.90 | 0.89 | -0.31 | -0.67 | 0.14 |

## Per-model NDCG

| comparison | gemma-3-27b-it | gemma-3-4b-it | gpt-oss-120b | gpt-oss-20b | Qwen2.5-72B | Qwen2.5-7B | Qwen3-235B | Qwen3-30B | Avg |
|---|---|---|---|---|---|---|---|---|---|
| PVQ_pvq_values (sum) | 0.66 | 0.96 | 0.84 | 0.88 | 0.75 | 0.74 | 0.97 | 0.69 | 0.81 |
| PVQ_pvq_values (norm) | 0.57 | 0.45 | 0.53 | 0.61 | 0.73 | 0.52 | 0.55 | 0.42 | 0.55 |
| PVQ21_pvq_values (sum) | 0.77 | 0.96 | 0.97 | 0.64 | 0.75 | 0.66 | 0.97 | 0.68 | 0.80 |
| PVQ21_pvq_values (norm) | 0.47 | 0.44 | 0.64 | 0.48 | 0.73 | 0.64 | 0.53 | 0.41 | 0.54 |
| BFI44_bfi_traits (sum) | 0.62 | 0.72 | 0.68 | 0.95 | 0.78 | 0.76 | 0.98 | 0.61 | 0.76 |
| BFI44_bfi_traits (norm) | 0.58 | 0.78 | 0.62 | 0.80 | 0.78 | 0.80 | 0.95 | 0.68 | 0.75 |
| BFI10_bfi_traits (sum) | 0.57 | 0.72 | 0.66 | 0.68 | 0.87 | 0.76 | 0.65 | 0.57 | 0.69 |
| BFI10_bfi_traits (norm) | 0.62 | 0.78 | 0.66 | 0.87 | 0.87 | 0.80 | 0.60 | 0.68 | 0.73 |

## Length-bias diagnostics
- Construct mean tagged-response length vs rank shift (sum->norm): Spearman = -0.79 (p=0.0, n=120)
