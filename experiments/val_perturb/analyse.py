"""Analyse sampled traces: per-type accuracy drop, max_drop, trace length, answer flips, LD_min drift.

Usage: uv run --with rapidfuzz python experiments/val_perturb/analyse.py [--runs experiments/val_perturb/runs]
Writes CSVs + summary.md to <runs>/analysis/. If <runs>/hidden/<model>.jsonl exists (from hidden.py),
Mahalanobis distances are merged in.
"""
import argparse, json, pathlib
from collections import Counter
import numpy as np, pandas as pd

HERE = pathlib.Path(__file__).parent
ROBUST_MAX, NONROBUST_MIN = 0.10, 0.25


def iter_run_files(runs, model=None):
    """runs/<model>.jsonl plus split+gzipped parts runs/<model>.partNN.gz (GitHub's 100 MB limit)."""
    stem = model.replace("/", "__") if model else "*"
    return sorted(runs.glob(f"{stem}.jsonl")) + sorted(runs.glob(f"{stem}.part*.gz")) + sorted(runs.glob(f"{stem}.part*"))


def load_rows(runs, model=None):
    import gzip
    seen, rows = set(), []
    for f in iter_run_files(runs, model):
        if f.suffix == ".gz" or f.suffix == ".jsonl" or ".part" in f.name:
            opener = gzip.open if f.suffix == ".gz" else open
            with opener(f, "rt") as fh:
                for line in fh:
                    r = json.loads(line)
                    key = (r["model"], r["id"], r["sample_idx"])
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append(r)
    return rows


def load_runs(runs):
    rows = load_rows(runs)
    for r in rows:
        r["len_chars"] = len(r["response"])
        r["truncated"] = r["finish_reason"] == "length"
    return pd.DataFrame(rows)


def ld_min(df, max_chars):
    """Normalised Levenshtein distance of each trace to the nearest *correct* trace on the original problem."""
    from rapidfuzz.distance import Levenshtein
    out = np.full(len(df), np.nan)
    for (m, pid), g in df.groupby(["model", "problem_id"]):
        refs = g[(g.type == "original") & g.correct]
        ref_txt = [(r["reasoning"] or r["response"])[:max_chars] for _, r in refs.iterrows()]
        if not ref_txt:
            continue
        for i, r in g.iterrows():
            txt = (r["reasoning"] or r["response"])[:max_chars]
            ds = [Levenshtein.normalized_distance(txt, t) for j, t in zip(refs.index, ref_txt) if j != i]
            if ds:
                out[df.index.get_loc(i)] = min(ds)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(HERE / "runs"))
    ap.add_argument("--ld-chars", type=int, default=20000, help="truncate traces for LD_min (speed)")
    ap.add_argument("--no-ld", action="store_true")
    ap.add_argument("--drop-flagged", action="store_true",
                    help="exclude variants marked 'flag' in data/review.json (sample.py writes flag=true)")
    args = ap.parse_args()
    runs = pathlib.Path(args.runs)
    out = runs / "analysis"; out.mkdir(exist_ok=True)
    df = load_runs(runs)
    if df.empty:
        raise SystemExit("no runs found")
    if args.drop_flagged and "flag" in df:
        df = df[df["flag"].fillna(False) != True]
    df["model"] = df["model"].str.split("/").str[-1]

    # answer flip: answer differs from the majority answer on the original problem (same model)
    maj = (df[df.type == "original"].groupby(["model", "problem_id"])["answer"]
           .agg(lambda s: Counter(s.dropna()).most_common(1)[0][0] if s.notna().any() else None))
    df["orig_majority"] = [maj.get((m, p)) for m, p in zip(df.model, df.problem_id)]
    df["flip"] = (df.answer != df.orig_majority)

    if not args.no_ld:
        df["ld_min"] = ld_min(df, args.ld_chars)
    hid = runs / "hidden"
    if hid.exists():
        h = pd.concat(pd.read_json(f, lines=True) for f in hid.glob("*.jsonl"))
        h["model"] = h["model"].str.split("/").str[-1]
        df = df.merge(h[["model", "id", "sample_idx", "md_h", "md_prompt"]], on=["model", "id", "sample_idx"], how="left")

    # per (model, problem, type) accuracy; drop vs original
    acc = (df.groupby(["model", "problem_id", "type", "source"])
             .agg(acc=("correct", "mean"), n=("correct", "size"), tokens=("n_tokens", "mean"),
                  truncated=("truncated", "mean"), flip=("flip", "mean"),
                  **({"ld_min": ("ld_min", "mean")} if "ld_min" in df else {}),
                  **({"md_h": ("md_h", "mean")} if "md_h" in df else {})).reset_index())
    base = acc[acc.type == "original"].set_index(["model", "problem_id"])["acc"]
    acc["base_acc"] = [base.get((m, p), np.nan) for m, p in zip(acc.model, acc.problem_id)]
    acc["drop"] = acc.base_acc - acc.acc
    acc.to_csv(out / "per_type.csv", index=False)

    pert = acc[acc.type != "original"]
    md = (pert.groupby(["model", "problem_id"]).agg(base_acc=("base_acc", "first"), max_drop=("drop", "max"),
                                                    worst_type=("drop", lambda s: pert.loc[s.idxmax(), "type"]))
              .reset_index())
    md["is_robust"] = np.where(md.max_drop <= ROBUST_MAX, True, np.where(md.max_drop >= NONROBUST_MIN, False, None))
    md.to_csv(out / "max_drop.csv", index=False)

    by_type = (pert.groupby(["model", "type"]).agg(mean_drop=("drop", "mean"), max_drop=("drop", "max"),
                                                   acc=("acc", "mean"), tokens=("tokens", "mean"),
                                                   flip=("flip", "mean"), n_variants=("n", "size")).reset_index())
    by_type.to_csv(out / "by_type.csv", index=False)

    # trace-level: correct vs incorrect traces
    cols = ["n_tokens", "len_chars", "truncated", "flip"] + [c for c in ("ld_min", "md_h") if c in df]
    tr = df.groupby(["model", "correct"])[cols].mean().reset_index()
    tr.to_csv(out / "trace_stats.csv", index=False)

    # AUC of each trace feature for predicting incorrectness (LPDS-style)
    from sklearn.metrics import roc_auc_score
    aucs = []
    for m, g in df[df.type != "original"].groupby("model"):
        for c in cols:
            s = g[c].astype(float)
            ok = s.notna() & g.correct.notna()
            if ok.sum() > 10 and g.correct[ok].nunique() == 2:
                aucs.append({"model": m, "feature": c, "auc_incorrect": roc_auc_score(~g.correct[ok], s[ok])})
    aucs = pd.DataFrame(aucs); aucs.to_csv(out / "feature_auc.csv", index=False)

    with (out / "summary.md").open("w") as f:
        f.write("# Per-model, per-type mean accuracy drop\n\n")
        f.write(by_type.pivot(index="type", columns="model", values="mean_drop").round(3).to_markdown() + "\n\n")
        f.write("# max_drop per (model, problem)\n\n" + md.round(3).to_markdown(index=False) + "\n\n")
        f.write("# Trace stats (correct vs incorrect)\n\n" + tr.round(3).to_markdown(index=False) + "\n\n")
        if not aucs.empty:
            f.write("# AUC of trace features for predicting an incorrect perturbed sample\n\n")
            f.write(aucs.pivot(index="feature", columns="model", values="auc_incorrect").round(3).to_markdown() + "\n")
    print((out / "summary.md").read_text())


if __name__ == "__main__":
    main()
