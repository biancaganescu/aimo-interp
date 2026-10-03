"""LPDS-style hidden-state drift: Mahalanobis distance of each trace's mean hidden state (layer ~2/3 depth)
from the distribution of correct original-problem traces of the same (model, problem).

Usage (GPU): uv run python experiments/val_perturb/hidden.py --model Qwen/Qwen3.5-4B [--layer-frac 0.66] [--max-tokens 8192]
Reads runs/<model>.jsonl, writes runs/hidden/<model>.jsonl with md_h (prompt+trace) and md_prompt (prompt only).
"""
import argparse, json, pathlib
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.covariance import LedoitWolf

HERE = pathlib.Path(__file__).parent
RUNS = HERE / "runs"
from sample import SYSTEM  # noqa: E402


def mean_hidden(model, tok, texts, layer, max_tokens, device):
    feats = []
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True, max_length=max_tokens).to(device)
        with torch.no_grad():
            hs = model(**ids, output_hidden_states=True).hidden_states[layer][0]
        feats.append(hs.float().mean(0).cpu().numpy())
    return np.stack(feats)


def mahal(ref, x):
    lw = LedoitWolf().fit(ref)
    d = x - ref.mean(0)
    return np.sqrt(np.einsum("ij,jk,ik->i", d, lw.precision_, d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer-frac", type=float, default=0.66)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()
    src = RUNS / (args.model.replace("/", "__") + ".jsonl")
    rows = [json.loads(l) for l in src.open()]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=getattr(torch, args.dtype),
                                                 trust_remote_code=True).to(device).eval()
    n_layers = model.config.num_hidden_layers
    layer = int(round(n_layers * args.layer_frac))
    print(f"{args.model}: layer {layer}/{n_layers}")

    def prompt(q):
        return tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}],
                                       tokenize=False, add_generation_prompt=True)

    full = mean_hidden(model, tok, [prompt(r["question"]) + r["response"] for r in rows], layer, args.max_tokens, device)
    uniq = {r["id"]: r["question"] for r in rows}
    pq = mean_hidden(model, tok, [prompt(q) for q in uniq.values()], layer, args.max_tokens, device)
    pq = dict(zip(uniq, pq))
    md_h = np.full(len(rows), np.nan); md_p = np.full(len(rows), np.nan)
    pids = {r["problem_id"] for r in rows}
    for pid in pids:
        idx = [i for i, r in enumerate(rows) if r["problem_id"] == pid]
        ref = [i for i in idx if rows[i]["type"] == "original" and rows[i]["correct"]]
        if len(ref) < 2:
            continue
        md_h[idx] = mahal(full[ref], full[idx])
        ref_p = np.stack([pq[rows[i]["id"]] for i in ref])
        md_p[idx] = mahal(ref_p, np.stack([pq[rows[i]["id"]] for i in idx]))
    out = RUNS / "hidden"; out.mkdir(exist_ok=True)
    with (out / src.name).open("w") as f:
        for r, a, b in zip(rows, md_h, md_p):
            f.write(json.dumps({"model": r["model"], "id": r["id"], "sample_idx": r["sample_idx"],
                                "md_h": None if np.isnan(a) else float(a), "md_prompt": None if np.isnan(b) else float(b)}) + "\n")


if __name__ == "__main__":
    main()
