"""Tables and descriptive alpha sweeps; never choose a best alpha from evaluation data."""
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/mswcsl_matplotlib')
import csv,hashlib,json
from pathlib import Path
from collections import defaultdict
import numpy as np
from build_native_tables import tabular
from build_review_tables import NAMES
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results/review_diagnostics'

def main():
    man=json.loads((OUT/'manifest.json').read_text())
    for name,h in man['sources'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
    for a in man['artifacts']:
        assert hashlib.sha256((OUT/a['artifact']).read_bytes()).hexdigest()==a['sha256']
    rows=[json.loads(x) for x in (OUT/'runs.jsonl').read_text().splitlines()]
    assert len(rows)==640*6*6 and len({(r['id'],r['variant'],r['alpha']) for r in rows})==len(rows)
    groups=defaultdict(list)
    for r in rows:
        regime='High spike' if r['a']==.9 and r['shape']=='spike' else 'High distributed' if r['a']==.9 else 'Moderate'
        groups[regime,r['variant'],r['alpha']].append(r)
    summaries=[]
    for (regime,v,a),rr in groups.items():
        summaries.append(dict(regime=regime,variant=v,alpha=a,n=len(rr),TPR=float(np.mean([r['metrics']['TPR'] for r in rr])),FDR=float(np.mean([r['metrics']['FDR'] for r in rr])),F1=float(np.mean([r['metrics']['F1'] for r in rr])),parent_recall=float(np.mean([r['parent_recall'] for r in rr])),candidates_per_node=float(np.mean([r['candidates_per_node'] for r in rr]))))
    with (OUT/'summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
    body=[]
    for regime in ('High spike','High distributed','Moderate'):
        body.append(r'\multicolumn{6}{@{}l}{\textit{'+regime+r'}}\\')
        for v,label in NAMES.items():
            s=next(q for q in summaries if q['regime']==regime and q['variant']==v and q['alpha']==.05)
            body.append(' & '.join([label,*[f'{s[k]:.3f}' for k in ('parent_recall','candidates_per_node','TPR','FDR','F1')]])+r'\\')
    tabular(ROOT.parent/'tex/tables/screen_diagnostics.tex',r'Post-review diagnostics on the same 640 persistence series, estimated orders, $\alpha=0.05$. High spike and high distributed average five cells each; moderate averages six cells. Recall is the per-series fraction of structural parent pairs retained at Stage 3; candidate size is the mean number per response. No-screen candidate sets contain every other variable. TPR, FDR (mean false-discovery proportion), and F1 refer to final non-self triples. Every nominal-level graph was checked against its saved counterpart.','tab:screen-diagnostics',['Variant','Parent recall','Candidates/node','TPR','FDR','F1'],body)
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,3,figsize=(11,3.8),layout='constrained')
    for ax,regime in zip(axs,('High spike','High distributed','Moderate')):
        for v,label in NAMES.items():
            ss=sorted([s for s in summaries if s['regime']==regime and s['variant']==v],key=lambda q:q['alpha'])
            xx=[s['FDR'] for s in ss];yy=[s['TPR'] for s in ss]
            line,=ax.plot(xx,yy,'o-',ms=3,lw=1,label=label.replace('$c_\\lambda=0.35$','half penalty'))
            q=next(s for s in ss if s['alpha']==.05);ax.scatter(q['FDR'],q['TPR'],s=65,facecolors='none',edgecolors=line.get_color())
        ax.set_title(regime);ax.set_xlabel('Mean false-discovery proportion');ax.set_ylim(0,1.02);ax.grid(alpha=.2)
    axs[0].set_ylabel('Mean true-positive rate');axs[-1].legend(fontsize=6.5,loc='lower right')
    fig.savefig(ROOT.parent/'figures/review_power_fdr.pdf');plt.close(fig)
    print(f'Verified {len(rows)} alpha-sweep metric rows from 3840 fits; wrote diagnostics and power/FDR figure.')
if __name__=='__main__':main()
