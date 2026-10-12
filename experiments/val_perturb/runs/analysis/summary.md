# Per-model, per-type mean accuracy drop

| type                 |   DeepSeek-R1-0528-Qwen3-8B |   DeepSeek-R1-0528-Qwen3-8B@64k |
|:---------------------|----------------------------:|--------------------------------:|
| distract             |                       0.018 |                          -0.04  |
| domain               |                      -0.012 |                           0.025 |
| expert_no_solution   |                       0.285 |                           0.405 |
| expert_perturbations |                       0.08  |                           0.025 |
| paraphrase           |                       0.08  |                           0.012 |
| rename               |                       0.01  |                          -0.015 |
| rephrase             |                       0.065 |                           0.01  |
| typos                |                      -0.004 |                           0.02  |

# max_drop per (model, problem)

| model                         | problem_id   |   base_acc |   max_drop | worst_type         | is_robust   |
|:------------------------------|:-------------|-----------:|-----------:|:-------------------|:------------|
| DeepSeek-R1-0528-Qwen3-8B     | 057f8a       |        0   |       0    | distract           | True        |
| DeepSeek-R1-0528-Qwen3-8B     | 1acac0       |        0.5 |       0.02 | expert_no_solution | True        |
| DeepSeek-R1-0528-Qwen3-8B     | 1fce4b       |        0.6 |       0.48 | expert_no_solution | False       |
| DeepSeek-R1-0528-Qwen3-8B     | 480182       |        0   |       0    | expert_no_solution | True        |
| DeepSeek-R1-0528-Qwen3-8B     | 71beb6       |        0.1 |       0.1  | distract           | True        |
| DeepSeek-R1-0528-Qwen3-8B     | 88c219       |        0   |       0    | distract           | True        |
| DeepSeek-R1-0528-Qwen3-8B     | a1d40b       |        0.9 |       0.82 | expert_no_solution | False       |
| DeepSeek-R1-0528-Qwen3-8B     | bbd91e       |        0.9 |       0.86 | expert_no_solution | False       |
| DeepSeek-R1-0528-Qwen3-8B@64k | 057f8a       |        0   |       0    | distract           | True        |
| DeepSeek-R1-0528-Qwen3-8B@64k | 1acac0       |        1   |       0.48 | expert_no_solution | False       |
| DeepSeek-R1-0528-Qwen3-8B@64k | 1fce4b       |        0.9 |       0.18 | expert_no_solution |             |
| DeepSeek-R1-0528-Qwen3-8B@64k | 480182       |        0.4 |       0.4  | expert_no_solution | False       |
| DeepSeek-R1-0528-Qwen3-8B@64k | 71beb6       |        0.7 |       0.7  | expert_no_solution | False       |
| DeepSeek-R1-0528-Qwen3-8B@64k | 88c219       |        0.1 |       0.1  | domain             | True        |
| DeepSeek-R1-0528-Qwen3-8B@64k | a1d40b       |        1   |       0.88 | expert_no_solution | False       |
| DeepSeek-R1-0528-Qwen3-8B@64k | bbd91e       |        0.9 |       0.62 | expert_no_solution | False       |

# Trace stats (correct vs incorrect)

| model                         | correct   |   n_tokens |   len_chars |   truncated |   flip |   ld_min |
|:------------------------------|:----------|-----------:|------------:|------------:|-------:|---------:|
| DeepSeek-R1-0528-Qwen3-8B     | False     |    32494   |     83355.1 |       0.95  |  0.984 |    0.773 |
| DeepSeek-R1-0528-Qwen3-8B     | True      |    25639   |     68675.9 |       0.027 |  0.029 |    0.767 |
| DeepSeek-R1-0528-Qwen3-8B@64k | False     |    54908.7 |    135604   |       0.324 |  0.758 |    0.782 |
| DeepSeek-R1-0528-Qwen3-8B@64k | True      |    32887.1 |     85388.7 |       0     |  0.036 |    0.766 |

# AUC of trace features for predicting an incorrect perturbed sample

| feature   |   DeepSeek-R1-0528-Qwen3-8B |   DeepSeek-R1-0528-Qwen3-8B@64k |
|:----------|----------------------------:|--------------------------------:|
| flip      |                       0.977 |                           0.86  |
| ld_min    |                       0.622 |                           0.801 |
| len_chars |                       0.778 |                           0.862 |
| n_tokens  |                       0.966 |                           0.889 |
| truncated |                       0.96  |                           0.661 |
