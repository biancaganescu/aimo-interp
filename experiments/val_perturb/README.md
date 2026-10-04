# val-sample perturbation replication (small track)

8 AIMO 2 reference problems from `aimo-interp/val-sample` x 8 perturbation types x the 4 small-track models,
then CoT analysis. Everything lives in this directory; nothing else in the repo is touched.

## Files

| file | what |
|---|---|
| `export_seed.py` | pulls the 8 problems and the 453 organiser-stored (accuracy-decaying) variants from HF -> `data/problems.json`, `data/seed_variants.json` |
| `data/answers.json` | ground-truth integer answers (solved locally; see notes below) |
| `perturb.py` | OpenRouter few-shot generator: 10 variants x 8 types per problem -> `data/variants.jsonl` (already generated with `openai/gpt-5.2`, effort low) |
| `sample.py` | vLLM sampler: 10 samples per prompt, T=1.0, top_k=40, top_p=0.95, organiser prompt (`ANSWER: <int or NaN>`) -> `runs/<model>.jsonl` |
| `hidden.py` | LPDS-style Mahalanobis drift of mean hidden state (layer ~2/3) from correct original traces -> `runs/hidden/<model>.jsonl` |
| `analyse.py` | per-type drop, `max_drop` + label, trace length, truncation, answer flips, LD_min, feature AUCs -> `runs/analysis/` |

## Run on the GPU box

```bash
# one-off: a venv with vllm (the repo venv pins transformers 5.13 / torch 2.12 which vllm may not accept)
python -m venv ~/.venv-vllm && . ~/.venv-vllm/bin/activate
pip install vllm pandas scikit-learn rapidfuzz tabulate

cd aimo-interp
# dry run: counts prompts (8 originals + 625 reviewed generated variants [+373 organiser variants with --seed-variants])
python experiments/val_perturb/sample.py --model all --dry-run --seed-variants

# sample, one model at a time (each ~6.5k-10k long generations; resumable, so Ctrl-C is safe)
python experiments/val_perturb/sample.py --model Qwen/Qwen3.5-4B --seed-variants
python experiments/val_perturb/sample.py --model Skywork/Skywork-OR1-Math-7B --seed-variants
python experiments/val_perturb/sample.py --model allenai/Olmo-3-7B-Think --seed-variants
python experiments/val_perturb/sample.py --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B --seed-variants
#   knobs: --max-tokens 32768 (default; raise to 65536/100000 for the non-Skywork models if you have the time),
#          --tp N for tensor parallel, --gpu-mem 0.9, --batch 64, --problems a,b --types rephrase,typos for a subset
#   cost:  DeepSeek-R1-0528-Qwen3-8B averages ~28k tokens/sample (38% hit the 32k cap) and an A100-80GB only fits
#          ~15-20 such sequences in KV cache (~420 tok/s), i.e. ~5 days per model at 10 variants x 10 samples.
#          Reduced budget: --variants-per-type 5 --n 5 (328 prompts x 5) and --kv-cache-dtype fp8 (~2x concurrency).

# optional: hidden-state drift (transformers, fits on one GPU)
python experiments/val_perturb/hidden.py --model Qwen/Qwen3.5-4B      # repeat per model

# analysis (CPU is fine)
python experiments/val_perturb/analyse.py          # -> runs/analysis/summary.md + CSVs
```

Then commit/push `experiments/val_perturb/runs/` (JSONL, a few hundred MB for all 4 models) or share it any other way.

## Notes

- `n=10` samples per prompt and 10 variants per type mirrors the `val-sample` setup (`n_base_predictions=10`,
  10 variants per type). Labels: robust iff `max_drop <= 0.10`, non-robust iff `>= 0.25` (train-main-v2 rule).
- `expert_no_solution` variants are scored correct when the model answers `NaN`.
- Answers (`data/answers.json`) were solved locally, not taken from Kaggle: 1acac0=50, 71beb6=891, a1d40b=201,
  bbd91e=902, 1fce4b=143, 057f8a=79, 88c219=810, 480182=751. 88c219 and 1acac0 deserve a cross-check.
- `data/review.json` is a manual (no-LLM) check of every generated variant against the original problem and
  its answer. 639 generated (one call returned 9); 14 are `drop` (change the answer, state a falsehood, or an
  `expert_no_solution` that is still solvable) and 9 are `flag` (kept, but weak: e.g. `rename` reusing `R` for
  a point and the circumradius, which the organiser variants also do). `sample.py` skips `drop` by default
  (`--include-dropped` to override) and writes `flag: true` on flagged rows so `analyse.py` consumers can
  exclude them. Drops by type: distract 3 (added a false or answer-changing condition), domain 2 (claimed
  *segment* XY meets BC; it does not), expert_perturbations 2 (false "redundant" facts), expert_no_solution 7
  (still solvable). The 5 pure surface types (rephrase, rename, typos, paraphrase) had no drops.
- Organiser-stored variants (`--seed-variants`) were *not* reviewed; they are the 373 decayed variants as published.
