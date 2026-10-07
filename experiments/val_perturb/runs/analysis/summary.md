# Per-model, per-type mean accuracy drop

| type                 |   DeepSeek-R1-0528-Qwen3-8B |
|:---------------------|----------------------------:|
| distract             |                       0.018 |
| domain               |                      -0.012 |
| expert_no_solution   |                       0.285 |
| expert_perturbations |                       0.08  |
| paraphrase           |                       0.08  |
| rename               |                       0.01  |
| rephrase             |                       0.065 |
| typos                |                      -0.004 |

# max_drop per (model, problem)

| model                     | problem_id   |   base_acc |   max_drop | worst_type         | is_robust   |
|:--------------------------|:-------------|-----------:|-----------:|:-------------------|:------------|
| DeepSeek-R1-0528-Qwen3-8B | 057f8a       |        0   |       0    | distract           | True        |
| DeepSeek-R1-0528-Qwen3-8B | 1acac0       |        0.5 |       0.02 | expert_no_solution | True        |
| DeepSeek-R1-0528-Qwen3-8B | 1fce4b       |        0.6 |       0.48 | expert_no_solution | False       |
| DeepSeek-R1-0528-Qwen3-8B | 480182       |        0   |       0    | expert_no_solution | True        |
| DeepSeek-R1-0528-Qwen3-8B | 71beb6       |        0.1 |       0.1  | distract           | True        |
| DeepSeek-R1-0528-Qwen3-8B | 88c219       |        0   |       0    | distract           | True        |
| DeepSeek-R1-0528-Qwen3-8B | a1d40b       |        0.9 |       0.82 | expert_no_solution | False       |
| DeepSeek-R1-0528-Qwen3-8B | bbd91e       |        0.9 |       0.86 | expert_no_solution | False       |

# Trace stats (correct vs incorrect)

| model                     | correct   |   n_tokens |   len_chars |   truncated |   flip |   ld_min |
|:--------------------------|:----------|-----------:|------------:|------------:|-------:|---------:|
| DeepSeek-R1-0528-Qwen3-8B | False     |      32494 |     83355.1 |       0.95  |  0.984 |    0.773 |
| DeepSeek-R1-0528-Qwen3-8B | True      |      25639 |     68675.9 |       0.027 |  0.029 |    0.767 |

# AUC of trace features for predicting an incorrect perturbed sample

| feature   |   DeepSeek-R1-0528-Qwen3-8B |
|:----------|----------------------------:|
| flip      |                       0.977 |
| ld_min    |                       0.622 |
| len_chars |                       0.778 |
| n_tokens  |                       0.966 |
| truncated |                       0.96  |
