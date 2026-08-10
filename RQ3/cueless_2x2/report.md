# 2x2: checkpoint (base/instruct) x items (original/cue-reduced)

All cells: constrained digit-logprob scoring, plain prompts.

## PVQ

| Pair | base/orig | instr/orig | base/cueless | instr/cueless | gain(orig) | gain(cueless) | interaction |
|---|---|---|---|---|---|---|---|
| Qwen2.5-7B | 0.485 | 0.642 | 0.476 | 0.367 | +0.157 | -0.109 | +0.266 |
| Gemma3-4B | 0.194 | 0.523 | 0.457 | 0.546 | +0.329 | +0.089 | +0.240 |
| Gemma3-27B | 0.490 | 0.748 | 0.463 | 0.449 | +0.258 | -0.014 | +0.272 |
| Qwen3-30B-A3B | 0.499 | 0.540 | 0.417 | 0.425 | +0.040 | +0.008 | +0.032 |

Mean gain on original items +0.196, on cue-reduced items -0.006; interaction +0.202 (4/4 pairs positive, exact sign-flip p=0.0625).

## BFI44

| Pair | base/orig | instr/orig | base/cueless | instr/cueless | gain(orig) | gain(cueless) | interaction |
|---|---|---|---|---|---|---|---|
| Qwen2.5-7B | 0.381 | 0.435 | 0.291 | 0.308 | +0.054 | +0.017 | +0.037 |
| Gemma3-4B | 0.171 | 0.131 | 0.044 | 0.087 | -0.040 | +0.043 | -0.083 |
| Gemma3-27B | 0.276 | 0.552 | 0.140 | 0.108 | +0.277 | -0.032 | +0.309 |
| Qwen3-30B-A3B | 0.686 | 0.324 | 0.294 | 0.149 | -0.362 | -0.144 | -0.217 |

Mean gain on original items -0.018, on cue-reduced items -0.029; interaction +0.011 (2/4 pairs positive, exact sign-flip p=0.4375).
