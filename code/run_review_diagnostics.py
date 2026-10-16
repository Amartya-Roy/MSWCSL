"""Post-review descriptive diagnostics; archived observations, no retuning.

Recompute all six variants on the 640 persistence series, verify alpha=.05
against saved graphs, record candidates and p-values, then vary alpha only.
These are exploratory power/FDR curves, not independent null calibration.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[k]='1'
import json,hashlib,sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
OUT=ROOT/'results/review_diagnostics'
ALPHAS=(.001,.005,.01,.025,.05,.1)

def work(job):
    import torch
    from scipy.stats import t
    from threadpoolctl import threadpool_limits
    from mswcsl import MSWCSL,MSWCSLConfig
    from mswcsl.refit import refit
    from mswcsl.prep import center,as_replicates
    from mswcsl.metrics import score_all
    from run_selection_aware import prewhiten
    torch.set_num_threads(1)
    source=ROOT/'results/independent_v5'/f"draw_{job['id']}.npz"
    saved=ROOT/'results/selection_aware_v1'/f"sel_{job['id']}.npz"
    with np.load(source) as z: X,truth=z['X'],z['truth']
    with np.load(saved) as z: graphs={k:z[k] for k in z.files}
    p=job['p'];d=X.shape[1];c=MSWCSLConfig(p=p)
    rows=[];arrays={}
    with threadpool_limits(limits=1):
        base=MSWCSL(c).fit(X)
        split=MSWCSL(c).fit(X[:len(X)//2])
        diff=MSWCSL(MSWCSLConfig(p=p,S=0,include_raw_view=False,finest_band='detail')).fit(X)
        rawlow=MSWCSL(MSWCSLConfig(p=p,S=0,finest_band='none',c_lambda=.35)).fit(X)
        pre=MSWCSL(MSWCSLConfig(p=p,order=base.order.tolist(),S=0,finest_band='none')).fit(prewhiten(X,p))
        variants=[('current',X,base.candidates,base.pos),('noscreen',X,[np.array([i for i in range(d) if i!=j]) for j in range(d)],base.pos),('split',X[len(X)//2:],split.candidates,split.pos),('diff',X,diff.candidates,diff.pos),('rawlow',X,rawlow.candidates,rawlow.pos),('prewhite',X,pre.candidates,base.pos)]
        parents=np.any(truth!=0,axis=2);np.fill_diagonal(parents,False)
        for name,x,cands,pos in variants:
            g,scores,info=refit(center(as_replicates(x),c.center),cands,pos,p,c.c_raw,c.c_lambda,'tstat',c.refit_wide_ratio,rule='bonferroni',alpha=.05)
            assert np.array_equal(g,graphs['estimated_'+name+'_graph']),(job['id'],name)
            cm=np.zeros((d,d),bool)
            for j,cs in enumerate(cands):cm[cs,j]=True
            arrays[name+'_candidates']=cm;arrays[name+'_pos']=pos
            total=sum(sum(i!=rec['node'] for i,r in rec['slots']) for rec in info)
            pv=np.ones_like(scores)
            for j,rec in enumerate(info):
                if 'crit' in rec:
                    pv[:,j,:]=2*t.sf(scores[:,j,:]*rec['crit'],max(rec['n']-rec['K_eff'],1))
            arrays[name+'_pvalues']=pv
            for alpha in ALPHAS:
                pred=(pv<alpha/max(total,1)).astype('int8')
                assert alpha!=.05 or np.array_equal(pred,g)
                rows.append(dict(id=job['id'],cell=job['cell'],a=job['a'],shape=job['shape'],T=job['T'],s=job['s'],variant=name,alpha=alpha,metrics=score_all(truth,pred)['both'],parent_recall=float((cm&parents).sum()/max(parents.sum(),1)),candidates_per_node=float(cm.sum()/d)))
    archive=OUT/f"diagnostics_{job['id']}.npz";np.savez_compressed(archive,**arrays)
    return rows,dict(id=job['id'],source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),reference_sha256=hashlib.sha256(saved.read_bytes()).hexdigest(),artifact=archive.name,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())

def main():
    OUT.mkdir(exist_ok=False)
    jobs=[j for j in json.loads((ROOT/'results/independent_v5/manifest.json').read_text())['jobs'] if j['study']=='persistence']
    sources=[Path(__file__),ROOT/'run_selection_aware.py',*sorted((ROOT/'src/mswcsl').rglob('*.py'))]
    manifest=dict(purpose=__doc__,alphas=ALPHAS,jobs=jobs,sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},artifacts=[])
    with ProcessPoolExecutor(max_workers=4) as pool,(OUT/'runs.jsonl').open('w') as f:
        for k,(rows,artifact) in enumerate(pool.map(work,jobs),1):
            for row in rows:f.write(json.dumps(row)+'\n')
            f.flush();manifest['artifacts'].append(artifact)
            if k%40==0:print(f'{k}/{len(jobs)} series verified',flush=True)
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':main()
