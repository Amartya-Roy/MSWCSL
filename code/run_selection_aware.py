"""Selection-aware refit and screening baselines on the archived v5 series (SELECTION_AWARE_PROTOCOL.md)."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/mswcsl_matplotlib')
import argparse, hashlib, json, sys, warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
V5=ROOT/'results/independent_v5'

def prewhiten(X,p):
    T,d=X.shape;out=np.empty((T-p,d))
    for k in range(d):
        D=np.column_stack([np.ones(T-p)]+[X[p-r:T-r,k] for r in range(1,p+1)])
        beta,*_=np.linalg.lstsq(D,X[p:,k],rcond=None);out[:,k]=X[p:,k]-D@beta
    return out

def run_one(j,out):
    import torch
    from threadpoolctl import threadpool_limits
    from mswcsl import MSWCSL,MSWCSLConfig
    from mswcsl.metrics import score_all
    from mswcsl.prep import as_replicates,center
    from mswcsl.refit import refit
    torch.set_num_threads(1)
    z=np.load(V5/f"draw_{j['id']}.npz");X,truth,topo=z['X'],z['truth'],z['topo'];d,p,T=X.shape[1],j['p'],len(X)
    arrays={};rows=[]
    def cfg(**kw):return MSWCSLConfig(p=p,**kw)
    def do_refit(Xr,cands,pos,c):
        g,_,_=refit(center(as_replicates(Xr),c.center),cands,pos,p,c.c_raw,c.c_lambda,'tstat',c.refit_wide_ratio,rule='bonferroni',alpha=c.refit_alpha)
        return g
    def record(order,variant,g,pos):
        fp0=(g[:,:,0]!=0)&(truth[:,:,0]==0);np.fill_diagonal(fp0,False)
        rows.append(dict(j,order=order,variant=variant,status='ok',metrics=score_all(truth,g),
            reversed_fp=int((fp0&(truth[:,:,0].T!=0)).sum()),
            valid_order=bool(all(pos[a]<pos[b] for a,b in zip(*np.nonzero(truth[:,:,0]))))))
        arrays[order+'_'+variant+'_graph']=g
    with threadpool_limits(limits=1),warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for order in ('known','estimated'):
            okw=dict(order=topo.tolist()) if order=='known' else dict(order='peel')
            c=cfg(**okw);res=MSWCSL(c).fit(X)
            assert np.array_equal(res.graph,z[order+'_full_graph']),'archive mismatch'
            assert np.array_equal(do_refit(X,res.candidates,res.pos,c),res.graph),'refit mismatch'
            record(order,'current',res.graph,res.pos)
            allc=[np.array([i for i in range(d) if i!=k]) for k in range(d)]
            record(order,'noscreen',do_refit(X,allc,res.pos,c),res.pos)
            h=T//2;r1=MSWCSL(c).fit(X[:h])
            record(order,'split',do_refit(X[h:],r1.candidates,r1.pos,c),r1.pos)
            if order=='estimated':
                est=res.order.tolist()
                for name,kw in (('diff',dict(S=0,include_raw_view=False,finest_band='detail')),('rawlow',dict(S=0,finest_band='none',c_lambda=.35))):
                    r=MSWCSL(cfg(order='peel',**kw)).fit(X);assert np.array_equal(r.order,res.order)
                    record(order,name,r.graph,r.pos)
                rp=MSWCSL(cfg(order=est,S=0,finest_band='none')).fit(prewhiten(X,p))
                record(order,'prewhite',do_refit(X,rp.candidates,res.pos,c),res.pos)
    path=Path(out)/f"sel_{j['id']}.npz";np.savez_compressed(path,**arrays)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    for r in rows:r.update(archive=path.name,sha256=digest)
    return rows

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=ROOT/'results/selection_aware_v1');ap.add_argument('--workers',type=int,default=6);ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    man=json.loads((V5/'manifest.json').read_text())
    work=[j for j in man['jobs'] if j['study'] in ('fwer','persistence')]
    if args.smoke:work=[next(j for j in work if j['study']==s) for s in ('fwer','persistence')]
    args.out.mkdir(parents=True,exist_ok=args.smoke)
    files=[Path(__file__),ROOT/'SELECTION_AWARE_PROTOCOL.md',*sorted((ROOT/'src/mswcsl').rglob('*.py'))]
    (args.out/'manifest.json').write_text(json.dumps(dict(jobs=[j['id'] for j in work],sources={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},smoke=args.smoke),indent=2)+'\n')
    with ProcessPoolExecutor(max_workers=args.workers) as pool,(args.out/'runs.jsonl').open('w') as fh:
        futures=[pool.submit(run_one,j,str(args.out)) for j in work]
        for k,f in enumerate(as_completed(futures),1):
            for row in f.result():fh.write(json.dumps(row)+'\n')
            fh.flush()
            if k%100==0 or k==len(work):print(f'{k}/{len(work)} series done',flush=True)
if __name__=='__main__':main()
