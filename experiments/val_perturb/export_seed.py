"""Export the 8 val-sample problems and every stored perturbed variant
(from aimo-interp-challenge-sample-full + val-sample) grouped by type."""
import glob, json, pathlib
import pandas as pd
from huggingface_hub import snapshot_download

OUT = pathlib.Path(__file__).parent / "data"

def load(repo):
    root = snapshot_download(repo, repo_type="dataset")
    return pd.concat(pd.read_parquet(f) for f in glob.glob(f"{root}/**/*.parquet", recursive=True))

val = load("aimo-interp/val-sample")
full = load("aimo-interp/aimo-interp-challenge-sample-full")

problems = (val.drop_duplicates("problem_id")[["problem_id", "original_problem"]]
            .rename(columns={"original_problem": "problem"}).to_dict("records"))
(OUT / "problems.json").write_text(json.dumps(problems, indent=1, ensure_ascii=False))

variants = {}
for df in (full, val):
    for _, r in df.iterrows():
        if not r["permutations_causing_decay"]:
            continue
        for v in json.loads(r["permutations_causing_decay"]):
            key = (r["problem_id"], v["permutation_type"], v["question"].strip())
            variants[key] = {"problem_id": r["problem_id"], "type": v["permutation_type"],
                             "source": v.get("permutation_source"), "question": v["question"].strip(),
                             "variant_accuracy": v.get("variant_accuracy")}
vl = list(variants.values())
(OUT / "seed_variants.json").write_text(json.dumps(vl, indent=1, ensure_ascii=False))
pids = {p["problem_id"] for p in problems}
print(len(problems), "problems;", len(vl), "unique variants;", sum(v["problem_id"] in pids for v in vl), "for val problems")
print(pd.DataFrame(vl).assign(inval=lambda d: d.problem_id.isin(pids)).groupby(["type", "inval"]).size().unstack(fill_value=0))
