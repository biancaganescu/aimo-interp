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


def _byte_decoder():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1)) + list(range(ord("\xae"), ord("\xff") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for c, b in zip(cs, bs)}


BYTE_DECODER = _byte_decoder()


def fix_bytelevel(text):
    """Some vLLM/transformers combinations return the raw byte-level BPE surface forms ('Ġ' for space,
    'Ċ' for newline) instead of decoded text. Map them back through the GPT-2 byte table."""
    if "Ġ" not in text and "Ċ" not in text:
        return text
    out = bytearray()
    for ch in text:
        b = BYTE_DECODER.get(ch)
        if b is None:
            out += ch.encode("utf-8")
        else:
            out.append(b)
    return out.decode("utf-8", errors="replace")


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


def read_base_rows(model):
    """Rows of the untagged run: runs/<model>.jsonl, else the split .partNN(.gz) pieces; de-duplicated."""
    import gzip
    stem = model.replace("/", "__")
    files = sorted(RUNS.glob(f"{stem}.jsonl")) or sorted(RUNS.glob(f"{stem}.part*"))
    seen, rows = set(), []
    for f in files:
        with (gzip.open if f.suffix == ".gz" else open)(f, "rt") as fh:
            for line in fh:
                r = json.loads(line)
                if (r["id"], r["sample_idx"]) not in seen:
                    seen.add((r["id"], r["sample_idx"]))
                    rows.append(r)
    return rows


def continue_truncated(args, items, model):
    tagged = f"{model}@{args.tag}"
    out = RUNS / (tagged.replace("/", "__").replace("@", "__") + ".jsonl")
    ids = {it["id"] for it in items}
    done = set()
    if out.exists():
        for line in out.open():
            r = json.loads(line)
            done.add((r["id"], r["sample_idx"]))
    base = read_base_rows(model)
    todo = [r for r in base if r["finish_reason"] == "length" and r["id"] in ids and (r["id"], r["sample_idx"]) not in done]
    longest = max((r["n_tokens"] for r in todo), default=0)
    print(f"{model}: {len(base)} base samples, {len(todo)} truncated to extend by {args.max_tokens} tokens "
          f"({len(done)} done); longest prefix {longest} tokens", file=sys.stderr)
    if args.dry_run or not todo:
        return
    from vllm import LLM, SamplingParams
    max_len = args.max_model_len or (longest + args.max_tokens + 2048)
    llm = LLM(model=model, tensor_parallel_size=args.tp, gpu_memory_utilization=args.gpu_mem,
              max_model_len=max_len, trust_remote_code=True, enable_prefix_caching=True,
              kv_cache_dtype=args.kv_cache_dtype)
    tok = llm.get_tokenizer()
    sp = SamplingParams(n=1, temperature=1.0, top_k=40, top_p=0.95, max_tokens=args.max_tokens, seed=0)
    step = args.batch * args.n  # same number of concurrent sequences as a normal batch
    with out.open("a") as f:
        for i in range(0, len(todo), step):
            chunk = todo[i:i + step]
            prompts = []
            for r in chunk:
                prefix = tok.apply_chat_template(
                    [{"role": "system", "content": SYSTEM}, {"role": "user", "content": r["question"]}],
                    tokenize=False, add_generation_prompt=True)
                partial = fix_bytelevel(r["response"])
                # the generation prompt of thinking models already ends with "<think>"; don't emit it twice
                if prefix.rstrip().endswith("<think>") and partial.lstrip().startswith("<think>"):
                    partial = partial.lstrip()[len("<think>"):]
                prompts.append(prefix + partial)
            results = llm.generate(prompts, sp, use_tqdm=True)
            for r, res in zip(chunk, results):
                o = res.outputs[0]
                text = fix_bytelevel(r["response"]) + fix_bytelevel(o.text)
                reasoning, final = split_reasoning(text)
                ans = extract_answer(final, text)
                keep = {k: r[k] for k in ("problem_id", "type", "variant_idx", "question", "source", "flag", "expected", "id")
                        if k in r}
                f.write(json.dumps({**keep, "model": tagged, "sample_idx": r["sample_idx"], "response": text,
                                    "reasoning": reasoning, "final": final, "answer": ans,
                                    "correct": is_correct(ans, r["expected"]),
                                    "n_tokens": r["n_tokens"] + len(o.token_ids), "finish_reason": o.finish_reason,
                                    "continued_from": r["n_tokens"]}, ensure_ascii=False) + "\n")
            f.flush()


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
    ap.add_argument("--tag", default=None,
                    help="write to runs/<model>__<tag>.jsonl and record model as <model>@<tag> (e.g. a 64k-cap rerun), "
                         "so it resumes independently and is analysed as a separate column")
    ap.add_argument("--continue-truncated", action="store_true",
                    help="instead of sampling afresh, extend every sample of this model that hit the cap "
                         "(finish_reason=length) by --max-tokens more tokens, feeding prompt + partial trace back as "
                         "the prefix. Requires --tag; analyse.py shows <model>@<tag> = finished originals + extended traces")
    ap.add_argument("--max-model-len", type=int, default=None)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu-mem", type=float, default=0.9)
    ap.add_argument("--batch", type=int, default=64, help="prompts per llm.chat call (x n samples)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    items = load_items(args)
    if args.continue_truncated:
        if not args.tag:
            sys.exit("--continue-truncated needs --tag (e.g. --tag 64k)")
        for model in (MODELS if args.model == "all" else [args.model]):
            continue_truncated(args, items, model)
        return
    print(f"{len(items)} prompts x {args.n} samples", file=sys.stderr)
    if args.dry_run:
        print(Counter(it["type"] for it in items))
        return

    from vllm import LLM, SamplingParams
    models = MODELS if args.model == "all" else [args.model]
    RUNS.mkdir(exist_ok=True)
    for model in models:
        tagged = f"{model}@{args.tag}" if args.tag else model
        out = RUNS / (tagged.replace("/", "__").replace("@", "__") + ".jsonl")
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
                        text = fix_bytelevel(o.text)
                        reasoning, final = split_reasoning(text)
                        ans = extract_answer(final, text)
                        f.write(json.dumps({**it, "model": tagged, "sample_idx": k, "response": text,
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
