"""Generate perturbed variants of the val-sample problems with an LLM via OpenRouter.

Usage:
  OPENROUTER_API_KEY=... uv run python experiments/val_perturb/perturb.py \
      --model openai/gpt-5.2 --n 10 --types all --out experiments/val_perturb/data/variants.jsonl

Each (problem, type) call asks for N distinct variants in one JSON response, few-shot
on stored organiser variants of *other* problems from seed_variants.json.
"""
import argparse, json, os, pathlib, random, re, sys, time, urllib.request

HERE = pathlib.Path(__file__).parent
DATA = HERE / "data"

TYPES = {
    "rephrase": "Rewrite the problem statement in different words. Keep every mathematical object, "
                "every number, every condition and the question exactly equivalent. Do not add or remove information.",
    "rename": "Rename the variables, points, people, places or objects (e.g. ABC -> PQR, Alice -> Mara, n -> k). "
              "Keep all numbers, conditions and the question identical. Avoid name clashes with existing symbols "
              "(e.g. do not rename a point to R if R already denotes the circumradius). Change nothing else.",
    "domain": "Transplant the problem into a different real-world setting or story (e.g. geometry -> radio beacons, "
              "airlines -> cargo ships, digits -> ticket codes) so that the mathematical structure, all numbers, "
              "and the answer are unchanged. The surface domain must change substantially; the maths must not.",
    "distract": "Insert one or two irrelevant but plausible sentences or clauses (extra context, an unused quantity, "
                "a remark) that do not change the answer and are not needed to solve the problem. Keep the original "
                "text otherwise intact.",
    "typos": "Introduce harmless typing noise: misspellings of ordinary words (e.g. 'lenght', 'perpendicluar'), "
             "doubled or missing letters, odd spacing or punctuation, abbreviations like 'w/out'. Never touch any "
             "number, variable name or mathematical symbol, and never change the meaning.",
    "paraphrase": "Produce a thorough paraphrase: restructure sentences, reorder clauses, and optionally recast the "
                  "setting, while keeping every number, condition and the question logically identical.",
    "expert_perturbations": "Act as an expert mathematician. Add a true but redundant or misleading-looking "
                            "statement derived from the problem (e.g. a bound on a related quantity, an unusual "
                            "convention made explicit, a reformulated hypothesis) that an expert knows does not "
                            "change the answer. The answer must remain exactly the same. For each variant also give "
                            "a one-sentence justification of why the answer is unchanged.",
    "expert_no_solution": "Act as an expert mathematician. Modify the problem so that it has NO valid answer: make "
                          "the premises contradictory, remove a premise so the answer is undetermined, or ask for an "
                          "object that cannot exist. The change should look innocuous (e.g. alter one number or one "
                          "condition). For each variant also give a one-sentence justification of why no solution exists.",
}
EXPERT = {"expert_perturbations", "expert_no_solution"}

SYSTEM = ("You generate perturbed versions of competition mathematics problems for a robustness study. "
          "You answer with a single JSON object and nothing else.")


def build_prompt(problem, ptype, n, shots, answer=None):
    shot_txt = "\n\n".join(f"Original:\n{s['original']}\nVariant ({ptype}):\n{s['question']}" for s in shots)
    extra = ""
    if ptype in EXPERT:
        extra = (f"\nThe correct answer to the original problem is {answer}.\n" if answer is not None else "") + \
                'Return {"variants": [{"question": ..., "justification": ...}, ...]}.'
    else:
        extra = 'Return {"variants": [{"question": ...}, ...]}.'
    return (f"Perturbation type: {ptype}\nInstructions: {TYPES[ptype]}\n\n"
            f"Here are examples of this perturbation type applied to other problems:\n\n{shot_txt}\n\n"
            f"Now produce {n} distinct variants of the following problem. The variants must differ from each other "
            f"substantially (different wording/choices), use LaTeX with $...$ like the original, and be self-contained.\n\n"
            f"Original:\n{problem}\n\n{extra}")


def call_openrouter(model, messages, key, temperature=1.0, retries=5, effort=None):
    payload = {"model": model, "messages": messages, "temperature": temperature,
               "response_format": {"type": "json_object"}}
    if effort:
        payload["reasoning"] = {"effort": effort}
    body = json.dumps(payload).encode()
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=body, headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/aimo-interp", "X-Title": "aimo-interp perturb"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                out = json.load(r)
            return out["choices"][0]["message"]["content"], out.get("usage", {})
        except Exception as e:  # noqa: BLE001
            wait = 2 ** attempt
            print(f"  request failed ({e}); retry in {wait}s", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError("OpenRouter call failed")


def parse_variants(text):
    m = re.search(r"\{.*\}", text, re.S)
    obj = json.loads(m.group(0) if m else text)
    vs = obj["variants"] if isinstance(obj, dict) else obj
    return [v if isinstance(v, dict) else {"question": v} for v in vs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-5.2")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--types", default="all")
    ap.add_argument("--problems", default=None, help="comma-separated problem_ids (default all)")
    ap.add_argument("--shots", type=int, default=4)
    ap.add_argument("--out", default=str(DATA / "variants.jsonl"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--effort", default="low", help="reasoning effort (organisers used :low)")
    args = ap.parse_args()

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        sys.exit("OPENROUTER_API_KEY not set")
    rng = random.Random(args.seed)
    problems = json.loads((DATA / "problems.json").read_text())
    answers = json.loads((DATA / "answers.json").read_text()) if (DATA / "answers.json").exists() else {}
    seeds = json.loads((DATA / "seed_variants.json").read_text())
    orig = {p["problem_id"]: p["problem"] for p in problems}
    # seed variants also cover 2 problems not in val-sample; recover their originals from the full sample
    for s in seeds:
        s["original"] = orig.get(s["problem_id"])
    types = list(TYPES) if args.types == "all" else args.types.split(",")
    if args.problems:
        problems = [p for p in problems if p["problem_id"] in args.problems.split(",")]

    out = pathlib.Path(args.out)
    done = set()
    if out.exists():
        for line in out.open():
            r = json.loads(line)
            done.add((r["problem_id"], r["type"]))
    with out.open("a") as f:
        for p in problems:
            for t in types:
                if (p["problem_id"], t) in done:
                    continue
                pool = [s for s in seeds if s["type"] == t and s["problem_id"] != p["problem_id"] and s["original"]]
                if t == "paraphrase" and len(pool) < 2:  # only 3 stored; pad with domain/rephrase examples
                    pool += [s for s in seeds if s["type"] in ("rephrase", "domain") and s["problem_id"] != p["problem_id"] and s["original"]]
                shots = rng.sample(pool, min(args.shots, len(pool)))
                prompt = build_prompt(p["problem"], t, args.n, shots, answers.get(p["problem_id"]))
                print(f"{p['problem_id']} {t}: {len(shots)} shots", flush=True)
                text, usage = call_openrouter(args.model, [{"role": "system", "content": SYSTEM},
                                                           {"role": "user", "content": prompt}], key, effort=args.effort)
                try:
                    vs = parse_variants(text)
                except Exception as e:  # noqa: BLE001
                    print(f"  parse failed: {e}\n{text[:500]}", file=sys.stderr)
                    continue
                for i, v in enumerate(vs[: args.n]):
                    f.write(json.dumps({"problem_id": p["problem_id"], "type": t, "variant_idx": i,
                                        "question": v["question"].strip(),
                                        "justification": v.get("justification"),
                                        "generator": args.model, "usage": usage}, ensure_ascii=False) + "\n")
                f.flush()
                print(f"  -> {len(vs)} variants, usage={usage}", flush=True)


if __name__ == "__main__":
    main()
