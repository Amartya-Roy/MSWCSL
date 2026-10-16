#!/usr/bin/env python3
"""LaTeX tables of the second JMLR revision, from stored results only (no fitting).

Usage: build_native_tables.py OUT_DIR
Inputs: results/jmlr/{bench_review3,review3_*,synh,causeme_review3}.jsonl, sortability.json and the stored
baseline runs ../../results/final_results_10th_october/merged_latest.jsonl (methods other than MSWCSL).
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
RES = os.path.join(ROOT, "results", "reference")
MERGED = os.path.join(RES, "baselines.jsonl")

DS = ["fMRI d=5", "fMRI d=10", "fMRI d=15", "Finance 20-1A", "Finance 20-1B", "DREAM3 yeast1",
      "DREAM3 yeast2", "Syn-6"]
DISPLAY_DS = [d for d in DS if d not in ("fMRI d=5", "fMRI d=15")]
DISPLAY_LABELS = {"fMRI d=10": "fMRI", "Finance 20-1A": "Fin. A", "Finance 20-1B": "Fin. B",
                  "DREAM3 yeast1": "yeast1", "DREAM3 yeast2": "yeast2", "Syn-6": "Syn-6"}


def load_aggregate_overrides():
    path = Path(HERE)/'results/author_rerun/aggregate_overrides.json'
    if not path.exists():
        return {}
    return {(r['target'], r['method'], r['dataset']): r
            for r in json.loads(path.read_text())['overrides']}


def reported_stats(sc, target, method, dataset, overrides):
    rec = overrides.get((target, method, dataset))
    if rec is not None:
        return rec['mean'], rec['halfwidth'], rec['n']
    return mean_hw(list(sc[target][method][dataset].values()))


GROUPS = {
    "both": ["PCMCI+", "DYNOTEARS", "MS-CASTLE", "VarLiNGAM", "STIC", "CDANs"],
    "lagged": ["PCMCI", "TCDF", "GVAR", "NAVAR", "ACD", "CUTS", "CUTS+"],
    "summary": ["NBCB", "PCTMI", "oCSE"],
}
LINEAR = {"both": [],
          "lagged": [("var_threshold", "VAR-threshold"), ("lasso_var", "Lasso-VAR")],
          "summary": []}
TITLE = {"both": "Both", "lagged": "Lagged", "summary": "Summary"}
OURS = ["MSWCSL"]


def fit_width(path):
    """Scale the tabular of a written table to the text width."""
    t = open(path).read()
    i = t.index("\\begin{tabular}")
    j = t.rindex("\\end{tabular}") + len("\\end{tabular}")
    open(path, "w").write(t[:i] + "\\resizebox{\\linewidth}{!}{" + t[i:j] + "}" + t[j:])


def jl(name):
    path = os.path.join(RES, name)
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


def mean_hw(v):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    if len(v) == 0:
        return np.nan, np.nan, 0
    if len(v) == 1:
        return float(v[0]), np.nan, 1
    return float(v.mean()), float(stats.t.ppf(.975, len(v) - 1) * v.std(ddof=1) / math.sqrt(len(v))), len(v)


def f3(x):
    return "--" if not np.isfinite(x) else f"{x:.3f}"[1:] if 0 <= x < 1 else f"{x:.3f}" if x >= 0 else f"$-${abs(x):.3f}"[0:]


def cell(m, hw, bold=False, under=False):
    if not np.isfinite(m):
        return "NC"
    s = f"{m:.3f}"[1:] if m < 1 else "1.00"
    s = f"\\mathbf{{{s}}}" if bold else f"\\underline{{{s}}}" if under else s
    return f"${s}{{\\scriptstyle\\pm{hw:.3f}}}$".replace("0.", ".") if np.isfinite(hw) else f"${s}$"


def top_two(values):
    """Distinct mean levels, with numerical ties sharing the same emphasis."""
    levels = []
    for value in sorted((float(v) for v in values if np.isfinite(v)), reverse=True):
        if not levels or not np.isclose(value, levels[-1], rtol=0, atol=1e-12):
            levels.append(value)
    return levels[:2]


# ------------------------------------------------------------------ benchmark data

def bench_scores():
    """scores[target][method][dataset] = {draw: F1}; extra[target][dataset][metric][method] for TPR/FDR."""
    sc = {t: defaultdict(lambda: defaultdict(dict)) for t in TITLE}
    full = {t: defaultdict(lambda: defaultdict(dict)) for t in TITLE}
    status = defaultdict(lambda: defaultdict(list))
    for r in (json.loads(l) for l in open(MERGED)):
        if r.get("dataset") not in DS or r["method"].startswith("MSWCSL") or r["method"] in ("SS-CASTLE", "VAR-threshold"):
            continue
        status[r["method"]][r["dataset"]].append(r["status"])
        if r["status"] != "ok" or "metrics" not in r:
            continue
        for t in TITLE:
            if t in r["metrics"]:
                sc[t][r["method"]][r["dataset"]][r["draw"]] = r["metrics"][t]["F1"]
                full[t][r["method"]][r["dataset"]][r["draw"]] = r["metrics"][t]
    for r in jl("bench_review3.jsonl"):
        if "error" in r or r["estimator"] == "random":
            continue
        name = "MSWCSL" if r["estimator"] == "mswcsl" else r["estimator"]
        for t in TITLE:
            sc[t][name][r["dataset"]][r["draw"]] = r["metrics"][t]["F1"]
            full[t][name][r["dataset"]][r["draw"]] = r["metrics"][t]
    rnd = defaultdict(lambda: defaultdict(list))
    for r in jl("bench_review3.jsonl"):
        if r.get("estimator") == "random":
            for t in TITLE:
                rnd[t][r["dataset"]].append(r["metrics_f1"][t])
    return sc, full, status, rnd


def holm(pvals):
    """Holm step-down adjusted p-values for a dict key -> p."""
    keys = sorted(pvals, key=lambda k: pvals[k])
    m = len(keys)
    adj, run = {}, 0.0
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (m - i) * pvals[k]))
        adj[k] = run
    return adj


def table_benchmark(out, score_data=None, record_dir=None, datasets=None, overrides=None):
    """Reported aggregates and native metrics with explicit rerun provenance."""
    sc, full, status, rnd = bench_scores() if score_data is None else score_data
    datasets = DISPLAY_DS if datasets is None else datasets
    overrides = load_aggregate_overrides() if overrides is None else overrides
    record_dir = RES if record_dir is None else record_dir
    os.makedirs(record_dir, exist_ok=True)
    stats = lambda t, m, ds: reported_stats(sc, t, m, ds, overrides)
    caption = (r'Benchmark-label agreement: mean F1 $\pm$ 95\% Student interval. '
               r'The benchmark protocol uses five draws per setting; fMRI denotes the ten-node simulations. '
               r'Incomplete baseline coverage is detailed in Appendix~\ref{app:exp-config}; NC means no completed draw. '
               r'MSWCSL uses one configuration and retains native fitted lags. '
               r'Bold marks the highest mean and underline the second-highest distinct mean within each target and dataset; ties share the marking. '
               r'These are descriptive development comparisons, not significance tests. '
               r'Finance and DREAM3 intervals describe resampling of the same source data.')
    opening = r'\resizebox{\linewidth}{!}{\begin{tabular}{@{}l*{' + str(len(datasets)) + r'}{c}@{}}'
    header = 'Method & ' + ' & '.join(DISPLAY_LABELS[ds] for ds in datasets) + r'\\'
    L = [r'\begin{table}[t]', r'\centering', r'\caption{'+caption+'}',
         r'\label{tab:real-benchmark-compact}', r'\scriptsize', r'\setlength{\tabcolsep}{3pt}',
         opening, r'\toprule', header]
    paired, coverage = {}, []
    for t in TITLE:
        methods = GROUPS[t]+[m for m,_ in LINEAR[t]]+['MSWCSL']
        names = dict(LINEAR[t])
        levels = {ds: top_two(stats(t,m,ds)[0] for m in methods) for ds in datasets}
        L += [r'\midrule', r'\multicolumn{'+str(len(datasets)+1)+r'}{@{}l}{\textit{'+TITLE[t]+r'}}\\']
        for m in methods:
            row = [(r'\rowcolor{black!6}' if m=='MSWCSL' else '')+names.get(m,m)]
            for ds in datasets:
                mu,hw,n = stats(t,m,ds)
                best = bool(levels[ds]) and np.isclose(mu,levels[ds][0],rtol=0,atol=1e-12)
                second = len(levels[ds])>1 and np.isclose(mu,levels[ds][1],rtol=0,atol=1e-12)
                row.append(cell(mu,hw,bold=best,under=second))
                if 0<n<5:
                    coverage.append(f"{m} on {DISPLAY_LABELS[ds]} ({n} of 5)")
            L.append(' & '.join(row)+r'\\')
        for m in GROUPS[t]+[m for m,_ in LINEAR[t]]:
            for ds in datasets:
                key=f'{t}|{ds}|MSWCSL|{m}'
                if (t,'MSWCSL',ds) in overrides or (t,m,ds) in overrides:
                    paired[key]=dict(status='unavailable: rerun per-draw records not supplied')
                    continue
                aa,bb=sc[t]['MSWCSL'][ds],sc[t][m][ds]
                common=sorted(set(aa)&set(bb));mean,hw,n=mean_hw([aa[k]-bb[k] for k in common])
                paired[key]=dict(draws=common,n=n,mean_difference=mean if n else None,
                                 interval_halfwidth=hw if n>1 else None)
        A=[r'\begin{table}[tbp]',r'\centering',
           r'\caption{'+TITLE[t]+r' target: mean F1 / TPR / FDR / SHD. F1 matches Table~\ref{tab:real-benchmark-compact}, including author-supplied rerun updates. For updated F1 aggregates without matching per-draw rerun records, the secondary metrics are shown as --. Bold and underline rank F1 only. Coverage exceptions are in Appendix~\ref{app:exp-config}.}',
           r'\label{tab:real-benchmark-'+t+'}',r'\scriptsize',opening,r'\toprule',header,r'\midrule']
        for m in methods:
            row=[names.get(m,m)]
            for ds in datasets:
                mu,hw,n=stats(t,m,ds)
                if not n:row.append('NC');continue
                best=bool(levels[ds]) and np.isclose(mu,levels[ds][0],rtol=0,atol=1e-12)
                second=len(levels[ds])>1 and np.isclose(mu,levels[ds][1],rtol=0,atol=1e-12)
                first=cell(mu,float('nan'),bold=best,under=second)
                vals=list(full[t][m][ds].values())
                secondary='--/--/--' if (t,m,ds) in overrides else '/'.join(f'{np.mean([v[k] for v in vals]):.2f}' for k in ['TPR','FDR','SHD'])
                row.append(first+'/'+secondary)
            A.append(' & '.join(row)+r'\\')
        A += [r'\bottomrule',r'\end{tabular}}',r'\end{table}']
        Path(out,f'real_benchmark_{t}.tex').write_text('\n'.join(A)+'\n')
    L += [r'\bottomrule',r'\end{tabular}}',r'\end{table}']
    Path(out,'real_benchmark_compact.tex').write_text('\n'.join(L)+'\n')
    cov = (r'The benchmark protocol uses five draws per setting: five simulations for fMRI, '
           r'five bootstrap seeds for each Finance and DREAM3 dataset, and five generating seeds for Syn-6. '
           r'Among reported baseline cells, incomplete runs are '+ '; '.join(coverage)+r'. '
           r'All remaining non-NC cells use five draws. Separate structural validation uses 20 seeds per regime; historical study sizes are stated individually.')
    Path(out,'benchmark_coverage.tex').write_text(cov+'\n')
    summary=dict(description='Paired differences only where matching per-draw records exist; no pooled ranking.',paired=paired,
                 aggregates=[dict(target=t,method=m,dataset=ds,mean=stats(t,m,ds)[0],halfwidth=stats(t,m,ds)[1],n=stats(t,m,ds)[2]) for t in TITLE for m in GROUPS[t]+[m for m,_ in LINEAR[t]]+['MSWCSL'] for ds in datasets])
    Path(record_dir,'benchmark_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def tabular(path, caption, label, columns, body):
    compact = len(columns) <= 4
    opening = r'\begin{tabular}{@{}l' + 'c'*(len(columns)-1) + '@{}}'
    if not compact:
        opening = r'\resizebox{\linewidth}{!}{' + opening
    text = [r'\begin{table}[tbp]', r'\centering', r'\caption{' + caption + '}',
            r'\label{' + label + '}', r'\small' if compact else r'\scriptsize',
            r'\setlength{\tabcolsep}{6pt}' if compact else r'\setlength{\tabcolsep}{3pt}',
            opening, r'\toprule', ' & '.join(columns) + r'\\', r'\midrule', *body,
            r'\bottomrule', r'\end{tabular}' + ('' if compact else '}'), r'\end{table}', '']
    path.write_text('\n'.join(text))
