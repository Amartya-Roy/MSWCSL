"""Prespecified summary of the selection-aware study (SELECTION_AWARE_PROTOCOL.md)."""
import csv,hashlib,json,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from summarize_independent_validation import wilson,interval
OUT=ROOT/'results/selection_aware_v1'
VARIANTS=('current','split','noscreen','diff','prewhite','rawlow')

def main():
    man=json.loads((OUT/'manifest.json').read_text());rows=[json.loads(x) for x in (OUT/'runs.jsonl').read_text().splitlines()]
    for name,h in man['sources'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
    ids={r['id'] for r in rows};assert ids==set(man['jobs']) and len(rows)==9*len(ids)
    assert all(r['status']=='ok' for r in rows)
    g=defaultdict(list)
    for r in rows:g[(r['cell'],r['order'],r['variant'])].append(r)
    summary=[]
    for (cell,order,v),rr in sorted(g.items()):
        r0=rr[0];fp=sum(r['metrics']['both']['FP']>0 for r in rr)
        summary.append(dict(cell=cell,study=r0['study'],kind=r0.get('kind'),T=r0['T'],a=r0['a'],shape=r0.get('shape'),s=r0.get('s'),order=order,variant=v,n=len(rr),
            FWER=fp/len(rr),FWER_CI=wilson(fp,len(rr)),TPR=float(np.mean([r['metrics']['both']['TPR'] for r in rr])),
            F1=float(np.mean([r['metrics']['both']['F1'] for r in rr])),reversed_fp=float(np.mean([r['reversed_fp'] for r in rr])),
            valid_order=float(np.mean([r['valid_order'] for r in rr]))))
    paired=[]
    for cell in range(18,34):
        base={r['rep']:r['metrics']['both']['F1'] for r in g[(cell,'estimated','current')]}
        for v in VARIANTS[1:]:
            diff=[r['metrics']['both']['F1']-base[r['rep']] for r in g[(cell,'estimated',v)]]
            m=float(np.mean(diff));lo,hi=interval(diff,.05/80)
            paired.append(dict(cell=cell,variant=v,mean=m,CI=interval(diff),simultaneous_CI=[lo,hi],
                verdict='improve' if m>=.02 and lo>0 else 'loss' if m<=-.02 and hi<0 else 'none'))
    near={v:all(s['FWER_CI'][1]<=.08 for s in summary if s['variant']==v and s['kind'] and s['kind'].startswith('null') and s['order'] in ('known','estimated')) for v in ('current','split','noscreen')}
    rep=dict(series=len(ids),rows=len(rows),summary=summary,paired=paired,near_nominal=near)
    (OUT/'summary.json').write_text(json.dumps(rep,indent=2)+'\n')
    with (OUT/'cells.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    plot(rep);table(rep);print(json.dumps(near))

def plot(rep):
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    S=rep['summary'];col={'current':'#2166ac','split':'#b2182b','noscreen':'#238b45'}
    fig,axs=plt.subplots(1,2,figsize=(11,4.6),layout='constrained')
    kinds=[('null_iid','null iid'),('null_persistent','null persistent'),('sparse_moderate','sparse moderate'),('sparse_persistent','sparse persistent')]
    labels=[f'{lab}, T={T}' for k,lab in kinds for T in (200,500)]
    for ax,order in zip(axs,('known','estimated')):
        for off,v in zip((-.2,0,.2),('current','split','noscreen')):
            pts=[next(s for s in S if s['kind']==k and s['T']==T and s['order']==order and s['variant']==v) for k,_ in kinds for T in (200,500)]
            m=np.array([s['FWER'] for s in pts]);ci=np.array([s['FWER_CI'] for s in pts])
            ax.errorbar(m,np.arange(len(pts))+off,xerr=np.maximum(0,[m-ci[:,0],ci[:,1]-m]),fmt='o',ms=4,color=col[v],label={'current':'Current refit','split':'Sample split','noscreen':'No screen'}[v])
        ax.axvline(.05,color='.4',ls='--',lw=.8);ax.set_yticks(range(len(labels)),labels,fontsize=10);ax.invert_yaxis();ax.set_xlim(0,1)
        ax.set_xlabel('Probability of any false edge (95% Wilson CI)',fontsize=10);ax.set_title(f'{order.capitalize()} order',fontsize=11);ax.grid(alpha=.15)
    axs[0].legend(fontsize=9,loc='lower right')
    fig.savefig(ROOT.parent/'figures/selection_aware_fwer.pdf');plt.close(fig)
def table(rep):
    S=rep['summary'];P=rep['paired']
    rng=lambda xs,f='{:.2f}':(f+'--'+f).format(min(xs),max(xs))
    names=[('current','Current refit'),('split','Sample-split refit'),('noscreen','No-screen refit'),
           ('diff','First-difference screen'),('prewhite','AR-prewhitened screen'),('rawlow','Raw screen, $c_\\lambda=0.35$')]
    lines=[]
    for k,(v,lab) in enumerate(names):
        sel=lambda **c:[s for s in S if s['variant']==v and all(s.get(a)==b for a,b in c.items())]
        nk=[s['FWER'] for s in sel(order='known',study='fwer') if s['kind'].startswith('null')]
        ne=[s['FWER'] for s in sel(order='estimated',study='fwer') if s['kind'].startswith('null')]
        sk=[s for s in sel(order='known',study='fwer') if s['kind'].startswith('sparse')]
        pe=sel(order='estimated',study='persistence')
        f1=lambda a,sh:np.mean([s['F1'] for s in pe if s['a']==a and (sh is None or s['shape']==sh)])
        losses=sum(p['verdict']=='loss' for p in P if p['variant']==v)
        cells=[lab,rng(nk) if nk else '--',rng(ne),
               (rng([s['FWER'] for s in sk])+' / '+rng([s['TPR'] for s in sk])) if sk else '--',
               f"{f1(.9,'spike'):.3f}",f"{f1(.9,'distributed'):.3f}",f"{f1(.6,None):.3f}",'--' if v=='current' else str(losses)]
        if k==3:lines.append('\\midrule')
        lines.append(' & '.join(cells)+'\\\\')
    tex=r"""\begin{table}[t]
\centering
\caption{Selection-aware refits and simpler screens on the archived validation series. False-edge columns give the range over the four null cells (500 series each) of the probability of at least one false non-self edge at nominal $\alpha=0.05$; the sparse column gives false-edge probability / Both TPR under known order. Persistence columns give mean Both F1 under estimated order (40 series per cell). Losses count cells, out of 16, where the paired F1 difference from the current refit is at most $-0.02$ with a simultaneous 95\% interval below zero. Screen variants use the current refit and were run with estimated orders only.}
\label{tab:selection-aware}
\scriptsize
\setlength{\tabcolsep}{3pt}
\resizebox{\linewidth}{!}{\begin{tabular}{@{}lccccccc@{}}
\toprule
& \multicolumn{3}{c}{False edges} & \multicolumn{3}{c}{Persistence Both F1} & \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
Variant & Null, known & Null, est. & Sparse, known & $a{=}0.9$ spike & $a{=}0.9$ distr. & $a{=}0.6$ & Losses\\
\midrule
"""+'\n'.join(lines)+r"""
\bottomrule
\end{tabular}}
\end{table}
"""
    (ROOT.parent/'tex/tables/selection_aware_table.tex').write_text(tex)
if __name__=='__main__':main()
