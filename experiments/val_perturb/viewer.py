"""Build the side-by-side trace viewer body (HTML fragment) from runs/.

  uv run python experiments/val_perturb/viewer.py [--head 3000 --tail 3000] [--out runs/analysis/viewer_body.html]

The fragment embeds a gzip+base64 JSON blob (questions, per-sample answers, trace head/tail) and the
JS that renders it; it is compiled into a standalone page by the Devin artifact kit (runs/analysis/viewer.html).
"""
import argparse, base64, gzip, html, json, pathlib, sys
import pandas as pd

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
from analyse import load_runs

TYPES = ["rephrase", "rename", "domain", "distract", "typos", "paraphrase", "expert_perturbations", "expert_no_solution"]


def short(model):
    return model.split("/")[-1]


def build_data(df, head, tail):
    problems = {p["problem_id"]: p["problem"] for p in json.load(open(HERE / "data/problems.json"))}
    order = sorted(problems, key=lambda p: -df[(df.problem_id == p) & (df.type == "original")].correct.mean())
    problems = {p: problems[p] for p in order}  # easiest first so the viewer opens on finished traces
    answers = json.load(open(HERE / "data/answers.json"))
    prompts = {}
    for _, r in df.drop_duplicates("id").iterrows():
        prompts[r.id] = dict(pid=r.problem_id, type=r.type, v=int(r.variant_idx), q=r.question, flag=bool(r.flag),
                             src=r.source, exp=str(r.expected))
    samples = []
    for _, r in df.iterrows():
        rs = r.reasoning or (r.response if r.finish_reason == "length" else "")
        cut = len(rs) > head + tail
        samples.append(dict(id=r.id, m=short(r.model), k=int(r.sample_idx), ans=None if pd.isna(r.answer) else str(r.answer),
                            ok=bool(r.correct), fin=r.finish_reason, tok=int(r.n_tokens), chars=len(rs),
                            head=rs[:head], tail=rs[-tail:] if cut else "", cut=cut,
                            final="" if r.finish_reason == "length" else (r.final or "")[:4000]))
    models = sorted(df.model.map(short).unique())
    # aggregates
    df = df.assign(m=df.model.map(short))
    per_prompt = df.groupby(["m", "id"]).agg(acc=("correct", "mean"), trunc=("truncated", "mean"), n=("correct", "size")).reset_index()
    per_prompt = per_prompt.merge(df.drop_duplicates("id")[["id", "problem_id", "type"]], on="id")
    base = per_prompt[per_prompt.type == "original"].set_index(["m", "problem_id"]).acc
    pt = per_prompt[per_prompt.type != "original"].groupby(["m", "problem_id", "type"]).acc.mean()
    by_type, max_drop = {}, {}
    for m in models:
        by_type[m] = {}
        for t in TYPES:
            drops = [base.get((m, p), float("nan")) - pt.get((m, p, t), float("nan")) for p in problems]
            s = pd.Series(drops).dropna()
            acc_t = df[(df.m == m) & (df.type == t)]
            by_type[m][t] = dict(drop=round(float(s.mean()), 3) if len(s) else None, acc=round(float(acc_t.correct.mean()), 3),
                                 trunc=round(float(acc_t.truncated.mean()), 3), tok=int(acc_t.n_tokens.mean()))
        max_drop[m] = {}
        for p in problems:
            d = {t: base.get((m, p), float("nan")) - pt.get((m, p, t), float("nan")) for t in TYPES}
            d = {t: v for t, v in d.items() if v == v}
            if not d:
                continue
            worst = max(d, key=d.get)
            o = df[(df.m == m) & (df.problem_id == p) & (df.type == "original")]
            max_drop[m][p] = dict(base=round(float(base.get((m, p), float("nan"))), 2), max_drop=round(d[worst], 2), worst=worst,
                                  trunc=round(float(o.truncated.mean()), 2), per_type={t: round(v, 2) for t, v in d.items()})
    overall = {m: dict(n=int((df.m == m).sum()), acc=round(float(df[df.m == m].correct.mean()), 3),
                       trunc=round(float(df[df.m == m].truncated.mean()), 3), tok=int(df[df.m == m].n_tokens.mean()),
                       acc_fin=round(float(df[(df.m == m) & ~df.truncated].correct.mean()), 3),
                       prompts=int(df[df.m == m].id.nunique())) for m in models}
    return dict(problems=problems, order=list(problems), answers=answers, prompts=prompts, samples=samples, models=models, types=TYPES,
                by_type=by_type, max_drop=max_drop, overall=overall, head=head, tail=tail)


def esc(s):
    return html.escape(str(s))


def pct(x):
    return "–" if x is None or x != x else f"{100 * x:.0f}%"


def summary_tables(d):
    out = []
    out.append('<div class="a-grid">')
    for m in d["models"]:
        o = d["overall"][m]
        out.append(f'''<div class="a-metric"><div class="a-metric__label">{esc(m)}</div>
<div class="a-metric__value">{pct(o["acc"])}</div>
<div class="a-metric__delta">{o["n"]:,} samples · {o["prompts"]} prompts · {pct(o["trunc"])} hit the token cap · {pct(o["acc_fin"])} correct when finished · mean {o["tok"]:,} tokens</div></div>''')
    out.append('</div>')
    # per type accuracy chart
    out.append('<figure class="a-panel" data-a-chart="bar" data-a-chart-unit="%" data-a-chart-max="100"><figcaption class="a-panel__title">Accuracy by perturbation type (all samples, truncated counted wrong)</figcaption><div class="a-table-scroll"><table class="a-table"><caption class="a-visually-hidden">Accuracy per perturbation type per model</caption><thead><tr><th scope="col">Type</th>')
    out += [f'<th scope="col" data-numeric>{esc(m)}</th>' for m in d["models"]]
    out.append('</tr></thead><tbody>')
    for t in ["original"] + d["types"]:
        out.append(f'<tr><th scope="row">{t}</th>')
        for m in d["models"]:
            a = d["overall"][m]["acc"] if t == "original" else d["by_type"][m][t]["acc"]
            if t == "original":
                a = sum(v["base"] for v in d["max_drop"][m].values()) / max(1, len(d["max_drop"][m]))
            out.append(f'<td data-numeric>{100 * a:.1f}%</td>')
        out.append('</tr>')
    out.append('</tbody></table></div></figure>')
    # per type drop / trunc table
    out.append('<div class="a-table-scroll"><table class="a-table" data-a-sticky-columns="1"><caption>Per type: mean accuracy drop vs. the original (positive = perturbation hurts), truncation rate, mean generated tokens</caption><thead><tr><th scope="col">Type</th>')
    out += [f'<th scope="col" data-numeric>{esc(m)}<br><small>drop · trunc · tokens</small></th>' for m in d["models"]]
    out.append('</tr></thead><tbody>')
    for t in d["types"]:
        out.append(f'<tr><th scope="row">{t}</th>')
        for m in d["models"]:
            b = d["by_type"][m][t]
            dr = "–" if b["drop"] is None else f'{b["drop"]:+.2f}'
            out.append(f'<td data-numeric>{dr} · {pct(b["trunc"])} · {b["tok"]:,}</td>')
        out.append('</tr>')
    out.append('</tbody></table></div>')
    # max_drop table
    out.append('<div class="a-table-scroll"><table class="a-table" data-a-sticky-columns="1"><caption>Per problem: original accuracy → max_drop over types (worst type); robust if max_drop ≤ 0.10, non-robust if ≥ 0.25</caption><thead><tr><th scope="col">Problem</th><th scope="col" data-numeric>Answer</th>')
    out += [f'<th scope="col">{esc(m)}</th>' for m in d["models"]]
    out.append('</tr></thead><tbody>')
    for p in d["problems"]:
        out.append(f'<tr><th scope="row"><button type="button" class="v-link" data-goto="{p}">{p}</button></th><td data-numeric>{d["answers"][p]}</td>')
        for m in d["models"]:
            x = d["max_drop"][m].get(p)
            if not x:
                out.append('<td>–</td>')
                continue
            lab = "robust" if x["max_drop"] <= 0.10 else ("non-robust" if x["max_drop"] >= 0.25 else "unlabelled")
            out.append(f'<td>base {x["base"]:.1f} → drop {x["max_drop"]:.2f} ({esc(x["worst"])}) · {lab}; {pct(x["trunc"])} truncated</td>')
        out.append('</tr>')
    out.append('</tbody></table></div>')
    return "\n".join(out)


CSS = """
<style>
.v-controls{display:flex;flex-wrap:wrap;gap:.75rem 1.25rem;align-items:end;margin:.5rem 0 1rem}
.v-controls label{display:flex;flex-direction:column;font-size:.85rem;gap:.25rem}
.v-controls select,.v-controls input{font:inherit;padding:.3rem .5rem;border:1px solid currentColor;border-radius:4px;background:transparent;color:inherit;min-width:9rem}
.v-cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,22rem),1fr));gap:1rem;align-items:start}
.v-q{white-space:pre-wrap;font-size:.92rem;line-height:1.45;padding:.75rem;border:1px solid rgba(128,128,128,.35);border-radius:6px}
.v-q ins{text-decoration:none;background:rgba(46,160,67,.25)}
.v-q del{background:rgba(248,81,73,.25)}
.v-badge{display:inline-block;padding:.05rem .45rem;border-radius:999px;font-size:.75rem;border:1px solid currentColor;margin-right:.3rem}
.v-badge[data-k="ok"]{color:#1a7f37}.v-badge[data-k="wrong"]{color:#cf222e}.v-badge[data-k="len"]{color:#9a6700}
.v-samples{display:flex;flex-wrap:wrap;gap:.3rem;margin:.4rem 0}
.v-sample{font:inherit;font-size:.8rem;padding:.2rem .5rem;border:1px solid rgba(128,128,128,.5);border-radius:4px;background:transparent;color:inherit;cursor:pointer}
.v-sample[aria-pressed="true"]{outline:2px solid currentColor}
.v-trace{white-space:pre-wrap;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.78rem;line-height:1.4;max-height:34rem;overflow:auto;padding:.6rem;border:1px solid rgba(128,128,128,.35);border-radius:6px}
.v-trace .v-gap{display:block;text-align:center;opacity:.7;margin:.5rem 0;font-style:italic}
.v-trace mark{background:rgba(255,212,0,.45);color:inherit}
.v-meta{font-size:.8rem;opacity:.85;margin:.2rem 0 .4rem}
.v-link{font:inherit;background:none;border:0;padding:0;color:inherit;text-decoration:underline;cursor:pointer}
.v-stat{font-size:.85rem;margin:.2rem 0}
.v-mini{display:grid;grid-template-columns:repeat(auto-fill,minmax(5.5rem,1fr));gap:.25rem;margin:.4rem 0}
.v-mini button{font:inherit;font-size:.72rem;padding:.25rem .3rem;border:1px solid rgba(128,128,128,.5);border-radius:4px;background:transparent;color:inherit;cursor:pointer;text-align:left}
.v-mini button[aria-pressed="true"]{outline:2px solid currentColor}
.v-mini b{display:block;font-weight:600}
@media print{.v-trace{max-height:none}}
</style>
"""

JS = r"""
<script>
(async function(){
  const el = (t, a={}, ...kids) => { const e=document.createElement(t); for (const [k,v] of Object.entries(a)) { if (k==='text') e.textContent=v; else if (k==='html') e.innerHTML=v; else e.setAttribute(k,v);} for (const k of kids) if (k!=null) e.append(k); return e; };
  const b64 = document.getElementById('v-data').textContent.trim();
  let D;
  try {
    const bytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
    const ds = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
    D = JSON.parse(await new Response(ds).text());
  } catch (e) { document.getElementById('v-app').innerHTML = '<div class="a-error">Could not decode the embedded data (needs a browser with DecompressionStream, e.g. Chrome 80+, Firefox 113+, Safari 16.4+): ' + e + '</div>'; return; }
  const byPrompt = {}; for (const s of D.samples) (byPrompt[s.id] ||= []).push(s);
  const promptsOf = (pid, type) => Object.entries(D.prompts).filter(([id,p]) => p.pid===pid && p.type===type).sort((a,b)=>a[1].v-b[1].v);
  const origId = pid => Object.keys(D.prompts).find(id => D.prompts[id].pid===pid && D.prompts[id].type==='original');
  const acc = (id, m) => { const ss=(byPrompt[id]||[]).filter(s=>s.m===m); return ss.length ? ss.filter(s=>s.ok).length/ss.length : null; };
  const trunc = (id, m) => { const ss=(byPrompt[id]||[]).filter(s=>s.m===m); return ss.length ? ss.filter(s=>s.fin==='length').length/ss.length : null; };
  const pct = x => x==null ? '–' : Math.round(100*x)+'%';

  // word diff (LCS) for the question pane
  function diffWords(a, b) {
    const A=a.split(/(\s+)/), B=b.split(/(\s+)/); const n=A.length, m=B.length;
    if (n*m > 4e6) return [[0,b]];
    const L=Array.from({length:n+1},()=>new Uint16Array(m+1));
    for (let i=n-1;i>=0;i--) for (let j=m-1;j>=0;j--) L[i][j] = A[i]===B[j] ? L[i+1][j+1]+1 : Math.max(L[i+1][j], L[i][j+1]);
    const out=[]; let i=0,j=0;
    while (i<n && j<m) { if (A[i]===B[j]) { out.push([0,A[i]]); i++; j++; } else if (L[i+1][j] >= L[i][j+1]) { out.push([-1,A[i]]); i++; } else { out.push([1,B[j]]); j++; } }
    while (i<n) out.push([-1,A[i++]]); while (j<m) out.push([1,B[j++]]);
    return out;
  }
  function renderQ(text, ref, mode) {
    const box = el('div', {class:'v-q'});
    if (!ref || ref===text) { box.textContent = text; return box; }
    for (const [op, w] of diffWords(ref, text)) {
      if (op===0) box.append(w); else if (op===1) box.append(el('ins',{text:w})); else if (mode==='full') box.append(el('del',{text:w}));
    }
    return box;
  }
  function highlight(txt, needle) {
    const frag = document.createDocumentFragment();
    if (!needle) { frag.append(txt); return frag; }
    const re = new RegExp(needle.replace(/[.*+?^${}()|[\]\\]/g,'\\$&'), 'gi'); let last=0, m;
    while ((m = re.exec(txt))) { frag.append(txt.slice(last, m.index)); frag.append(el('mark',{text:m[0]})); last = m.index+m[0].length; if (m[0].length===0) re.lastIndex++; }
    frag.append(txt.slice(last)); return frag;
  }
  function badge(s) {
    const k = s.fin==='length' ? 'len' : (s.ok ? 'ok' : 'wrong');
    return el('span', {class:'v-badge', 'data-k':k, text: s.fin==='length' ? 'truncated' : (s.ok ? 'correct' : 'wrong')});
  }
  function tracePane(s, needle) {
    const wrap = el('div');
    wrap.append(el('div', {class:'v-meta'}, badge(s), ` answer: ${s.ans ?? '—'} · expected ${D.prompts[s.id].exp} · ${s.tok.toLocaleString()} tokens · ${s.chars.toLocaleString()} chars of reasoning`));
    const t = el('div', {class:'v-trace', tabindex:'0', 'aria-label':'reasoning trace'});
    t.append(highlight(s.head, needle));
    if (s.cut) { t.append(el('span',{class:'v-gap', text:`… ${(s.chars - D.head - D.tail).toLocaleString()} chars omitted (viewer keeps the first ${D.head.toLocaleString()} and last ${D.tail.toLocaleString()}; full text in runs/) …`})); t.append(highlight(s.tail, needle)); }
    wrap.append(t);
    if (s.final) { const f = el('details'); f.append(el('summary', {text:'final answer section'})); f.append(el('div', {class:'v-trace'}, highlight(s.final, needle))); wrap.append(f); }
    return wrap;
  }

  const app = document.getElementById('v-app');
  const ctl = el('div', {class:'v-controls'});
  const selP = el('select', {id:'v-problem', 'aria-label':'problem'}); for (const p of D.order) selP.append(el('option',{value:p, text:`${p} (answer ${D.answers[p]})`}));
  const selT = el('select', {id:'v-type', 'aria-label':'perturbation type'}); for (const t of D.types) selT.append(el('option',{value:t, text:t}));
  const selV = el('select', {id:'v-variant', 'aria-label':'variant'});
  const selMode = el('select', {id:'v-mode', 'aria-label':'layout'}); selMode.append(el('option',{value:'models', text:'models side by side'}), el('option',{value:'orig', text:'original vs perturbed (one model)'}));
  const selM = el('select', {id:'v-model', 'aria-label':'model'}); for (const m of D.models) selM.append(el('option',{value:m, text:m}));
  const search = el('input', {id:'v-search', type:'search', placeholder:'highlight in traces…', 'aria-label':'highlight text in traces'});
  const chkDiff = el('input', {type:'checkbox', id:'v-diff'}); chkDiff.checked = false;
  ctl.append(el('label',{}, 'Problem', selP), el('label',{}, 'Perturbation', selT), el('label',{}, 'Variant', selV), el('label',{}, 'Layout', selMode), el('label',{}, 'Model', selM), el('label',{}, 'Highlight', search), el('label',{}, el('span',{text:'Show removed words'}), chkDiff));
  app.append(ctl);
  const qRow = el('div', {class:'v-cols'}); app.append(qRow);
  const grid = el('div', {class:'v-mini'}); app.append(el('p',{class:'v-stat', text:'Variants of this problem and type (accuracy per model; click to open):'}), grid);
  const cols = el('div', {class:'v-cols'}); app.append(cols);
  const picked = {};  // model -> sample idx

  function fillVariants() {
    const pid=selP.value, t=selT.value; selV.innerHTML='';
    for (const [id,p] of promptsOf(pid,t)) selV.append(el('option',{value:id, text:`#${p.v}${p.flag?' (flagged)':''} · ${D.models.map(m=>pct(acc(id,m))).join(' / ')}`}));
  }
  function render() {
    const pid=selP.value, id=selV.value, p=D.prompts[id]; if (!p) { qRow.innerHTML=''; cols.innerHTML=''; return; }
    const oid = origId(pid), needle = search.value.trim();
    qRow.innerHTML='';
    const left = el('div'); left.append(el('h4',{text:'Original'}), renderQ(D.problems[pid]), el('p',{class:'v-stat', html: D.models.map(m=>`${m}: <b>${pct(acc(oid,m))}</b> correct, ${pct(trunc(oid,m))} truncated`).join(' · ')}));
    const right = el('div'); right.append(el('h4',{text:`${p.type} #${p.v}${p.flag?' · flagged in manual review':''} · expected ${p.exp}`}), renderQ(p.q, D.problems[pid], chkDiff.checked?'full':'ins'), el('p',{class:'v-stat', html: D.models.map(m=>`${m}: <b>${pct(acc(id,m))}</b> correct, ${pct(trunc(id,m))} truncated`).join(' · ')}));
    qRow.append(left, right);
    grid.innerHTML='';
    for (const [vid,vp] of promptsOf(pid, selT.value)) { const b = el('button',{type:'button','aria-pressed': String(vid===id)}); b.append(el('b',{text:`#${vp.v}${vp.flag?'⚑':''}`}), D.models.map(m=>pct(acc(vid,m))).join(' / ')); b.onclick=()=>{selV.value=vid; render();}; grid.append(b); }
    cols.innerHTML='';
    const mode = selMode.value;
    const panes = mode==='models' ? D.models.map(m=>[m, id, `${m} · perturbed`]) : [[selM.value, oid, `${selM.value} · original`], [selM.value, id, `${selM.value} · ${p.type} #${p.v}`]];
    for (const [m, sid, title] of panes) {
      const col = el('div'); col.append(el('h4',{text:title}));
      const ss = (byPrompt[sid]||[]).filter(s=>s.m===m).sort((a,b)=>a.k-b.k);
      if (!ss.length) { col.append(el('div',{class:'a-empty', text:'No samples for this model yet.'})); cols.append(col); continue; }
      const key = m+'|'+sid; if (!(key in picked)) picked[key] = 0;
      const row = el('div',{class:'v-samples'});
      ss.forEach((s,i)=>{ const b=el('button',{type:'button', class:'v-sample','aria-pressed':String(i===picked[key])}); b.append(badge(s), `#${s.k} → ${s.ans ?? '—'}`); b.onclick=()=>{picked[key]=i; render();}; row.append(b); });
      col.append(row, tracePane(ss[picked[key]], needle)); cols.append(col);
    }
  }
  selP.onchange = () => { fillVariants(); render(); };
  selT.onchange = () => { fillVariants(); render(); };
  for (const c of [selV, selMode, selM, chkDiff]) c.onchange = render;
  let tm; search.oninput = () => { clearTimeout(tm); tm = setTimeout(render, 250); };
  document.querySelectorAll('[data-goto]').forEach(b => b.onclick = () => { selP.value = b.dataset.goto; fillVariants(); render(); document.getElementById('v-app').scrollIntoView({behavior:'smooth'}); });
  fillVariants(); render();
})();
</script>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(HERE / "runs"))
    ap.add_argument("--head", type=int, default=3000)
    ap.add_argument("--tail", type=int, default=3000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    runs = pathlib.Path(args.runs)
    df = load_runs(runs)
    d = build_data(df, args.head, args.tail)
    blob = base64.b64encode(gzip.compress(json.dumps(d, ensure_ascii=False).encode("utf-8"), 9)).decode()
    body = f"""
<section class="a-section">
  <h2 class="a-section__title">Overview</h2>
  <p class="a-section__note">8 AIMO 2 reference problems (val-sample) × 8 perturbation types × 5 variants (first 5 accepted after manual review), T=1.0, top-k 40, top-p 0.95, max 32,768 generated tokens. A sample is correct if its last <code>ANSWER:</code> matches the reference (mod 1000) or is <code>NaN</code> for <code>expert_no_solution</code>; samples that hit the token cap count as wrong.</p>
  {summary_tables(d)}
</section>
<section class="a-section">
  <h2 class="a-section__title">Traces side by side</h2>
  <p class="a-section__note">Pick a problem, perturbation type and variant. The perturbed question is diffed against the original (<ins style="background:rgba(46,160,67,.25);text-decoration:none">added</ins>, <del style="background:rgba(248,81,73,.25)">removed</del>). Each column shows one model's samples: click a sample to open its reasoning; the viewer keeps the first {args.head:,} and last {args.tail:,} characters of each trace.</p>
  {CSS}
  <div id="v-app"></div>
  <script type="text/plain" id="v-data">{blob}</script>
  {JS}
</section>
"""
    out = pathlib.Path(args.out) if args.out else runs / "analysis" / "viewer_body.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body)
    print(f"{out}: {out.stat().st_size / 1e6:.1f} MB body, {len(d['samples'])} samples, {len(d['prompts'])} prompts, models {d['models']}", file=sys.stderr)


if __name__ == "__main__":
    main()
