"""Sample a small-track model on the original val-sample problems and their perturbed variants with vLLM.

Usage (on a GPU box):
  uv run --with vllm python experiments/val_perturb/sample.py --model Qwen/Qwen3.5-4B --n 10 \
      --variants experiments/val_perturb/data/variants.jsonl [--seed-variants] --max-tokens 32768

Writes one JSONL line per sample to experiments/val_perturb/runs/<model>.jsonl with the full response,
the reasoning part, extracted answer, correctness and token count. Re-running resumes.
"""
import argparse, json, pathlib, re, sys
from collections import Counter

HERE = pathlib.Path(__file__).parent
DATA = HERE / "data"
RUNS = HERE / "runs"

SYSTEM = ("You are a precise math problem solver. Solve the given problem step by step, then output your final "
          "answer on the last line in the exact format:\nANSWER: <your answer>\nThis result should be an integer "
          "value or NaN if the given problem does not have a solution.")

MODELS = ["Qwen/Qwen3.5-4B", "Skywork/Skywork-OR1-Math-7B", "allenai/Olmo-3-7B-Think",
          "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"]


def split_reasoning(text):
    if "</think>" in text:
        think, _, final = text.partition("</think>")
        return think.replace("<think>", "").strip(), final.strip()
    return None, text.strip()


def extract_answer(final, full):
    for src in (final, full):
        m = re.findall(r"ANSWER:\s*\$?\\?(?:boxed\{)?\s*(-?\d+|NaN|nan)\s*\}?\$?", src)
        if m:
            return m[-1]
        b = re.findall(r"\\boxed\{\s*(-?\d+|\\text\{NaN\}|NaN)\s*\}", src)
        if b:
            return b[-1].replace("\\text{NaN}", "NaN")
    return None


def is_correct(ans, expected):
    if ans is None:
        return False
    if expected == "NaN":
        return ans.lower() == "nan"
    try:
        return int(ans) == int(expected) or int(ans) % 1000 == int(expected) % 1000
    except ValueError:
        return False


def load_items(args):
    problems = json.loads((DATA / "problems.json").read_text())
    answers = json.loads((DATA / "answers.json").read_text())
    items = []
    for p in problems:
        items.append({"problem_id": p["problem_id"], "type": "original", "variant_idx": 0, "source": "original",
                      "question": p["problem"], "expected": answers[p["problem_id"]]})
    review = {(r["problem_id"], r["type"], r["variant_idx"]): r["status"]
              for r in json.loads((DATA / "review.json").read_text())["items"]}
    taken = Counter()
    if args.variants:
        for line in open(args.variants):
            r = json.loads(line)
            status = review.get((r["problem_id"], r["type"], r["variant_idx"]), "ok")
            if status == "drop" and not args.include_dropped:
                continue
            if args.variants_per_type and taken[(r["problem_id"], r["type"])] >= args.variants_per_type:
                continue
            taken[(r["problem_id"], r["type"])] += 1
            items.append({**{k: r[k] for k in ("problem_id", "type", "variant_idx", "question")},
                          "source": r.get("generator", "llm"), "flag": status == "flag",
                          "expected": "NaN" if r["type"] == "expert_no_solution" else answers[r["problem_id"]]})
    if args.seed_variants:
        for i, r in enumerate(json.loads((DATA / "seed_variants.json").read_text())):
            if r["problem_id"] in answers:
                items.append({"problem_id": r["problem_id"], "type": r["type"], "variant_idx": 1000 + i,
                              "source": "organiser", "question": r["question"],
                              "expected": "NaN" if r["type"] == "expert_no_solution" else answers[r["problem_id"]]})
    if args.problems:
        keep = set(args.problems.split(","))
        items = [it for it in items if it["problem_id"] in keep]
    if args.types:
        keep = set(args.types.split(",")) | {"original"}
        items = [it for it in items if it["type"] in keep]
    for it in items:
        it["id"] = f"{it['problem_id']}/{it['type']}/{it['variant_idx']}/{it['source']}"
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS + ["all"])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--variants", default=str(DATA / "variants.jsonl"))
    ap.add_argument("--seed-variants", action="store_true", help="also run the organiser-stored (decayed) variants")
    ap.add_argument("--variants-per-type", type=int, default=None,
                    help="use only the first K accepted generated variants per (problem, type); default all (10)")
    ap.add_argument("--kv-cache-dtype", default="auto",
                    help="vLLM kv_cache_dtype; 'fp8' roughly doubles concurrent sequences")
    ap.add_argument("--include-dropped", action="store_true",
                    help="also run variants marked 'drop' in data/review.json (manual review rejects)")
    ap.add_argument("--problems", default=None)
    ap.add_argument("--types", default=None)
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--max-model-len", type=int, default=None)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu-mem", type=float, default=0.9)
    ap.add_argument("--batch", type=int, default=64, help="prompts per llm.chat call (x n samples)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    items = load_items(args)
    print(f"{len(items)} prompts x {args.n} samples", file=sys.stderr)
    if args.dry_run:
        print(Counter(it["type"] for it in items))
        return

    from vllm import LLM, SamplingParams
    models = MODELS if args.model == "all" else [args.model]
    RUNS.mkdir(exist_ok=True)
    for model in models:
        out = RUNS / (model.replace("/", "__") + ".jsonl")
        done = set()
        if out.exists():
            for line in out.open():
                done.add(json.loads(line)["id"])
        todo = [it for it in items if it["id"] not in done]
        print(f"{model}: {len(todo)} prompts to run ({len(done)} done)", file=sys.stderr)
        if not todo:
            continue
        llm = LLM(model=model, tensor_parallel_size=args.tp, gpu_memory_utilization=args.gpu_mem,
                  max_model_len=args.max_model_len, trust_remote_code=True, enable_prefix_caching=True,
                  kv_cache_dtype=args.kv_cache_dtype)
        sp = SamplingParams(n=args.n, temperature=1.0, top_k=40, top_p=0.95, max_tokens=args.max_tokens, seed=0)
        with out.open("a") as f:
            for i in range(0, len(todo), args.batch):
                chunk = todo[i:i + args.batch]
                convs = [[{"role": "system", "content": SYSTEM}, {"role": "user", "content": it["question"]}]
                         for it in chunk]
                results = llm.chat(convs, sp, use_tqdm=True)
                for it, res in zip(chunk, results):
                    for k, o in enumerate(res.outputs):
                        reasoning, final = split_reasoning(o.text)
                        ans = extract_answer(final, o.text)
                        f.write(json.dumps({**it, "model": model, "sample_idx": k, "response": o.text,
                                            "reasoning": reasoning, "final": final, "answer": ans,
                                            "correct": is_correct(ans, it["expected"]),
                                            "n_tokens": len(o.token_ids), "finish_reason": o.finish_reason},
                                           ensure_ascii=False) + "\n")
                f.flush()
        del llm
        import gc, torch
        gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
