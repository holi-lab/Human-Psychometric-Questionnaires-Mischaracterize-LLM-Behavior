# Base vs instruct — analysis


## Qwen2.5-7B  (base=Qwen2.5-7B  instruct=Qwen2.5-7B-Instruct)


### PVQ
- eta2: base=0.4854 instr=0.6419 (delta=+0.1566)
- WMV : base=0.6288 instr=0.5078 (delta=-0.1210)
- instruct plain vs main generated-answer construct-profile rho=0.7378
- prosocial construct value / rank (rank 1 = highest):
  | construct | base val | instr val | Δval | base rank | instr rank | Δrank |
  |---|---|---|---|---|---|---|
  | Benevolence | 3.8536 | 3.8563 | 0.0027 | 1 | 1 | +0 |
  | Universalism | 3.7595 | 3.6646 | -0.0949 | 6 | 4 | -2 |

### BFI44
- eta2: base=0.3813 instr=0.4355 (delta=+0.0541)
- WMV : base=0.7128 instr=0.6521 (delta=-0.0607)
- instruct plain vs main generated-answer construct-profile rho=0.5643
- prosocial construct value / rank (rank 1 = highest):
  | construct | base val | instr val | Δval | base rank | instr rank | Δrank |
  |---|---|---|---|---|---|---|
  | Agreeableness | 3.288 | 3.5319 | 0.2439 | 4 | 1 | -3 |

### recognition mean-F1
  | survey | base F1 | instr F1 | Δ |
  |---|---|---|---|
  | PVQ | 0.7184 | 0.7909 | +0.0725 |
  | BFI44 | 0.5778 | 0.3122 | -0.2656 |
  | VP | 0.0893 | 0.0663 | -0.0230 |

### profile shift by matched construct family
  | family | VP genprob rho | questionnaire rho | VP higher? |
  |---|---:|---:|---| 
  | values | 0.9394 | 0.8667 | True |
  | traits | 0.9 | 0.1 | True |

### direct VP-PVQ gap diagnostic (Schwartz values)
  | VP score | base rho(VP, PVQ) | instruct rho(VP, PVQ) | rho(standardized delta VP, delta PVQ) |
  |---|---:|---:|---:|
  | summed | 0.297 | 0.4667 | 0.1758 |
  | normalized | -0.4545 | -0.2364 | 0.103 |

### direct VP-BFI44 gap diagnostic (Big Five traits)
  | VP score | base rho(VP, BFI44) | instruct rho(VP, BFI44) | rho(standardized delta VP, delta BFI44) |
  |---|---:|---:|---:|
  | summed | 0.4 | 0.4 | -0.3 |
  | normalized | 0.9 | 0.0 | -0.6 |

## gemma-3-4b  (base=gemma-3-4b-pt  instruct=gemma-3-4b-it)


### PVQ
- eta2: base=0.1941 instr=0.5230 (delta=+0.3289)
- WMV : base=1.0114 instr=0.6611 (delta=-0.3503)
- instruct plain vs main generated-answer construct-profile rho=0.8754
- prosocial construct value / rank (rank 1 = highest):
  | construct | base val | instr val | Δval | base rank | instr rank | Δrank |
  |---|---|---|---|---|---|---|
  | Benevolence | 3.6793 | 5.9003 | 2.221 | 9 | 1 | -8 |
  | Universalism | 3.7051 | 5.8458 | 2.1407 | 6 | 2 | -4 |

### BFI44
- eta2: base=0.1708 instr=0.1307 (delta=-0.0401)
- WMV : base=0.9357 instr=0.9958 (delta=+0.0601)
- instruct plain vs main generated-answer construct-profile rho=0.6000
- prosocial construct value / rank (rank 1 = highest):
  | construct | base val | instr val | Δval | base rank | instr rank | Δrank |
  |---|---|---|---|---|---|---|
  | Agreeableness | 3.0477 | 3.8107 | 0.763 | 3 | 3 | +0 |

### recognition mean-F1
  | survey | base F1 | instr F1 | Δ |
  |---|---|---|---|
  | PVQ | 0.0 | 0.3062 | +0.3062 |
  | BFI44 | 0.0 | 0.3367 | +0.3367 |
  | VP | 0.0 | 0.128 | +0.1280 |

### profile shift by matched construct family
  | family | VP genprob rho | questionnaire rho | VP higher? |
  |---|---:|---:|---| 
  | values | 0.7333 | 0.1515 | True |
  | traits | 0.9 | 1.0 | False |

### direct VP-PVQ gap diagnostic (Schwartz values)
  | VP score | base rho(VP, PVQ) | instruct rho(VP, PVQ) | rho(standardized delta VP, delta PVQ) |
  |---|---:|---:|---:|
  | summed | -0.3212 | 0.1273 | -0.2242 |
  | normalized | -0.3212 | -0.4061 | -0.4061 |

### direct VP-BFI44 gap diagnostic (Big Five traits)
  | VP score | base rho(VP, BFI44) | instruct rho(VP, BFI44) | rho(standardized delta VP, delta BFI44) |
  |---|---:|---:|---:|
  | summed | 0.6 | 0.3 | -0.1 |
  | normalized | 0.7 | -0.5 | -0.2 |

## gemma-3-27b  (base=gemma-3-27b-pt  instruct=gemma-3-27b-it)


### PVQ
- eta2: base=0.4900 instr=0.7482 (delta=+0.2582)
- WMV : base=0.6116 instr=0.2817 (delta=-0.3300)
- instruct plain vs main generated-answer construct-profile rho=0.9848
- prosocial construct value / rank (rank 1 = highest):
  | construct | base val | instr val | Δval | base rank | instr rank | Δrank |
  |---|---|---|---|---|---|---|
  | Benevolence | 3.7637 | 4.8247 | 1.061 | 2 | 2 | +0 |
  | Universalism | 3.7241 | 4.5526 | 0.8285 | 3 | 3 | +0 |

### BFI44
- eta2: base=0.2756 instr=0.5523 (delta=+0.2767)
- WMV : base=0.8294 instr=0.5176 (delta=-0.3118)
- instruct plain vs main generated-answer construct-profile rho=1.0000
- prosocial construct value / rank (rank 1 = highest):
  | construct | base val | instr val | Δval | base rank | instr rank | Δrank |
  |---|---|---|---|---|---|---|
  | Agreeableness | 3.4428 | 4.3459 | 0.9031 | 3 | 1 | -2 |

### recognition mean-F1
  | survey | base F1 | instr F1 | Δ |
  |---|---|---|---|
  | PVQ | 0.0 | 0.5229 | +0.5229 |
  | BFI44 | 0.0 | 0.5788 | +0.5788 |
  | VP | 0.0 | 0.1133 | +0.1133 |

### profile shift by matched construct family
  | family | VP genprob rho | questionnaire rho | VP higher? |
  |---|---:|---:|---| 
  | values | 0.6242 | 0.8788 | False |
  | traits | 0.7 | 0.7 | False |

### direct VP-PVQ gap diagnostic (Schwartz values)
  | VP score | base rho(VP, PVQ) | instruct rho(VP, PVQ) | rho(standardized delta VP, delta PVQ) |
  |---|---:|---:|---:|
  | summed | 0.6727 | 0.2121 | -0.1758 |
  | normalized | 0.2727 | 0.0182 | -0.2121 |

### direct VP-BFI44 gap diagnostic (Big Five traits)
  | VP score | base rho(VP, BFI44) | instruct rho(VP, BFI44) | rho(standardized delta VP, delta BFI44) |
  |---|---:|---:|---:|
  | summed | 0.6 | -0.1 | -0.2 |
  | normalized | 0.5 | -1.0 | 0.0 |

## Qwen3-30B-A3B  (base=Qwen3-30B-A3B-Base  instruct=Qwen3-30B-A3B-Instruct-2507)


### PVQ
- eta2: base=0.4993 instr=0.5396 (delta=+0.0402)
- WMV : base=0.6129 instr=0.6298 (delta=+0.0169)
- instruct plain vs main generated-answer construct-profile rho=0.5366
- prosocial construct value / rank (rank 1 = highest):
  | construct | base val | instr val | Δval | base rank | instr rank | Δrank |
  |---|---|---|---|---|---|---|
  | Benevolence | 4.5213 | 5.5727 | 1.0514 | 2 | 3 | +1 |
  | Universalism | 4.55 | 5.6124 | 1.0624 | 1 | 2 | +1 |

### BFI44
- eta2: base=0.6856 instr=0.3240 (delta=-0.3617)
- WMV : base=0.3701 instr=0.7967 (delta=+0.4265)
- instruct plain vs main generated-answer construct-profile rho=1.0000
- prosocial construct value / rank (rank 1 = highest):
  | construct | base val | instr val | Δval | base rank | instr rank | Δrank |
  |---|---|---|---|---|---|---|
  | Agreeableness | 4.0687 | 4.4127 | 0.344 | 1 | 1 | +0 |

### recognition mean-F1
  | survey | base F1 | instr F1 | Δ |
  |---|---|---|---|
  | PVQ | 0.7502 | 0.6239 | -0.1263 |
  | BFI44 | 0.2838 | 0.9282 | +0.6444 |
  | VP | 0.046 | 0.1001 | +0.0541 |

### profile shift by matched construct family
  | family | VP genprob rho | questionnaire rho | VP higher? |
  |---|---:|---:|---| 
  | values | 0.8545 | 0.8545 | False |
  | traits | 1.0 | 1.0 | False |

### direct VP-PVQ gap diagnostic (Schwartz values)
  | VP score | base rho(VP, PVQ) | instruct rho(VP, PVQ) | rho(standardized delta VP, delta PVQ) |
  |---|---:|---:|---:|
  | summed | 0.6242 | 0.5515 | -0.4788 |
  | normalized | 0.0182 | 0.2 | -0.2485 |

### direct VP-BFI44 gap diagnostic (Big Five traits)
  | VP score | base rho(VP, BFI44) | instruct rho(VP, BFI44) | rho(standardized delta VP, delta BFI44) |
  |---|---:|---:|---:|
  | summed | 0.5 | 0.5 | 0.5 |
  | normalized | 0.4 | -0.4 | 0.3 |