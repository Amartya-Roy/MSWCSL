"""Finance failure analysis from saved Bonferroni runs and archival comparators."""
import json,hashlib
from pathlib import Path
import numpy as np
from scipy.stats import t
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/artifacts/review_finance'
GROUPS={'Both':['PCMCI+','DYNOTEARS','MS-CASTLE','VarLiNGAM','STIC','CDANs'],
'Lagged':['PCMCI','TCDF','GVAR','NAVAR','ACD','CUTS','CUTS+','Lasso-VAR'],
'Summary':['NBCB','PCTMI','oCSE']}
def read(name):return [json.loads(l) for l in (ROOT/name).read_text().splitlines()]
def interval(v):
 v=np.asarray(v,float);return [float(v.mean()),float(t.ppf(.975,len(v)-1)*v.std(ddof=1)/np.sqrt(len(v))) if len(v)>1 else None]
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 ours=read('experiments/results/bonf_v1/runs.jsonl');others=read('experiments/results/reference/baselines.jsonl')
 export=[];metrics=[];cov=[]
 for ds in ['Finance 20-1A','Finance 20-1B']:
  rows=sorted([r for r in ours if r['dataset']==ds],key=lambda r:r['draw']);assert len(rows)==5
  for row in rows:
   a=np.load(ROOT/'experiments/results/bonf_v1'/row['artifact'])
   for target in GROUPS:
    truth=a['truth'].copy();pred=a['MSWCSL'].copy()
    for lag in range(truth.shape[2]):np.fill_diagonal(truth[:,:,lag],0);np.fill_diagonal(pred[:,:,lag],0)
    if target=='Lagged':truth=truth[:,:,1:];pred=pred[:,:,1:]
    if target=='Summary':truth=truth.any(axis=2);pred=pred.any(axis=2)
    tp=int(np.sum((truth!=0)&(pred!=0)));fp=int(np.sum((truth==0)&(pred!=0)));fn=int(np.sum((truth!=0)&(pred==0)))
    m=row['methods']['MSWCSL']['metrics'][target.lower()]
    assert (tp,fp,fn)==(m['TP'],m['FP'],m['FN'])
  for target,methods in GROUPS.items():
   vals=[r['methods']['MSWCSL']['metrics'][target.lower()] for r in rows]
   full={k:float(np.mean([m[k] for m in vals])) for k in vals[0]}
   candidates=[]
   for method in methods:
    rr=[r for r in others if r['dataset']==ds and r['method']==method and r.get('status')=='ok' and target.lower() in r.get('metrics',{})]
    if rr:
     assert len({r['draw'] for r in rr})==len(rr)
     candidates.append({'method':method,'n':len(rr),'F1':interval([r['metrics'][target.lower()]['F1'] for r in rr])})
   best=max(candidates,key=lambda r:r['F1'][0])
   result={'dataset':ds,'target':target,'ours':interval([m['F1'] for m in vals]),'n':5,'metrics':full,
           'strongest_completed':[best['method'],best['F1']],'comparator_n':best['n'],'comparators':candidates}
   export.append(result)
  # Original data diagnostics, not bootstrap draws; preserve exact source provenance.
  file=next(Path(f) for f in rows[0]['meta']['files'] if 'returns' in f)
  X=np.loadtxt(file,delimiter=',',skiprows=1);C=np.cov(X,rowvar=False);e=np.linalg.eigvalsh(C)[::-1]
  R=np.corrcoef(X,rowvar=False)
  cov.append({'dataset':ds,'rows':len(X),'d':X.shape[1],'pc1':float(e[0]/e.sum()),'pc12':float(e[:2].sum()/e.sum()),
              'mean_abs_correlation':float(np.abs(R[np.triu_indices(len(R),1)]).mean()),'source':str(file),
              'sha256':hashlib.sha256(file.read_bytes()).hexdigest()})
 (OUT/'finance_values.json').write_text(json.dumps({'results':export,'covariance':cov},indent=2)+'\n')
 lines=[r'\begin{table}[tbp]',r'\centering\small',
 r'\caption{Finance failure analysis: MSWCSL mean F1 $\pm$ 95\% Student interval, precision (P), recall (R), and SHD over five moving-block bootstrap draws per series. Best comparator is selected by available-draw mean F1 within each target; brackets give completed draws. These intervals resample two fixed series, not independent networks.}',
 r'\label{tab:finance-failure}',r'\setlength{\tabcolsep}{3pt}',r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}llcccc l@{}}',r'\toprule',r'Series & Target & F1 & P & R & SHD & Best comparator F1 [$n$]\\',r'\midrule']
 for q in export:
  mu,hw=q['ours'];m=q['metrics'];name,v=q['strongest_completed']
  lines.append(f"{q['dataset'][-1]} & {q['target']} & ${mu:.3f}\\pm{hw:.3f}$ & {m['Precision']:.3f} & {m['TPR']:.3f} & {m['SHD']:.1f} & {name} {v[0]:.3f} [{q['comparator_n']}]"+r'\\')
 lines += [r'\bottomrule\end{tabular*}',r'\end{table}']
 (ROOT/'tex/tables/finance_failure.tex').write_text('\n'.join(lines)+'\n')
 print(json.dumps({'finance_cells':len(export),'covariance':cov},indent=2))
if __name__=='__main__':main()
