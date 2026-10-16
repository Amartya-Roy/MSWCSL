"""Frozen independent validation of the unmodified estimator; no tuning or replacement."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/mswcsl_matplotlib')
import argparse, hashlib, importlib.metadata, itertools, json, sys, warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))

def jobs():
    cells=[]
    for T,a,b in itertools.product((200,1000),(.9,.98,'local'),(.25,.5,1.)):
        cells.append(dict(study='cancellation',T=T,a=a,b=b,d=4,p=1,reps=40))
    for T,a,shape in itertools.product((200,500,1000),(.6,.9),('spike','distributed')):
        cells.append(dict(study='persistence',T=T,a=a,shape=shape,s=.6,d=10,p=4,reps=40))
    for s,shape in itertools.product((.3,.9),('spike','distributed')):
        cells.append(dict(study='persistence',T=500,a=.9,shape=shape,s=s,d=10,p=4,reps=40))
    for T,kind in itertools.product((200,500),('null_iid','null_persistent','sparse_moderate','sparse_persistent')):
        cells.append(dict(study='fwer',T=T,kind=kind,a=.9 if 'persistent' in kind else .3 if 'sparse' in kind else 0.,d=6,p=2,reps=500))
    return [dict(c,cell=i,rep=r,seed=80260000+i*10000+r,id=f'{i:02d}_{r:04d}') for i,c in enumerate(cells) for r in range(c['reps'])]

def generate(j):
    rng=np.random.default_rng(j['seed']); d,p,T=j['d'],j['p'],j['T']
    B=np.zeros((d,d,p+1)); a=1-(T-2)**(-.75) if j['a']=='local' else j['a']
    if j['study']=='cancellation':
        b=j['b']; B[0,0,1]=a; B[0,1,0]=B[0,2,0]=1;B[1,3,1]=b;B[2,3,1]=-b
        f=np.empty(T+1);f[0]=rng.normal()/np.sqrt(1-a*a)
        for t in range(1,T+1): f[t]=a*f[t-1]+rng.normal()
        x=f+rng.normal(size=T+1);z=f+rng.normal(size=T+1)
        y=b*(x[:-1]-z[:-1])+rng.normal(size=T)
        return np.column_stack((f[1:],x[1:],z[1:],y)),B,np.arange(4)
    topo=rng.permutation(d);pos=np.argsort(topo)
    if j['study']=='persistence':
        for i,k in itertools.product(range(d),repeat=2):
            if pos[i]<pos[k] and rng.random()<.15: B[i,k,0]=rng.choice([-1,1])*.5*rng.uniform(.6,1.2)
        for i,k in itertools.product(range(d),repeat=2):
            if pos[i]<pos[k] and rng.random()<.3:
                tot=rng.choice([-1,1])*j['s']*rng.uniform(.6,1.2)
                if j['shape']=='spike': B[i,k,rng.integers(1,p+1)]=tot
                else: B[i,k,1:]=tot/p
    elif 'sparse' in j['kind']:
        for k in range(d-1):
            B[topo[k],topo[k+1],0]=.3
            B[topo[k],topo[k+1],1+k%p]=.3*(-1)**k
    B[np.arange(d),np.arange(d),1]=a
    # Same 500-observation burn-in and random-graph law as historical persistence study.
    burn=500;E=rng.normal(size=(T+burn,d));X=np.zeros_like(E);inv=np.linalg.inv(np.eye(d)-B[:,:,0])
    for t in range(T+burn):
        drive=E[t].copy()
        for r in range(1,p+1):
            if t>=r:drive+=X[t-r]@B[:,:,r]
        X[t]=drive@inv
    return X[burn:],B,topo

def matched_screen(X):
    from sklearn.linear_model import Lasso
    n=len(X)-2;lam=n**(-.25);out={};saved={}
    for name,Z in [('raw',X),('haar',np.diff(X,axis=0)/np.sqrt(2))]:
        start=2 if name=='raw' else 1
        D=np.column_stack((Z[start:,:3],Z[start-1:-1,:]));y=Z[start:,3]
        D=(D-D.mean(0))/D.std(0);y=(y-y.mean())/y.std()
        co=Lasso(alpha=lam,fit_intercept=False,tol=1e-12,max_iter=1000000).fit(D,y).coef_
        cand=set(np.array([0,1,2,0,1,2,3])[abs(co)>1.5*lam])-{3}
        out[name]=dict(candidates=sorted(map(int,cand)),exact=cand=={1,2},empty=not cand,recall=len(cand&{1,2})/2)
        saved['matched_'+name]=co
    return out,saved

def run_one(j,out):
    import torch
    from threadpoolctl import threadpool_limits
    from mswcsl import MSWCSL,MSWCSLConfig
    from mswcsl.metrics import score_all
    torch.set_num_threads(1)
    X,B,topo=generate(j);truth=(B!=0).astype(np.int8);truth[np.arange(j['d']),np.arange(j['d']),:]=0
    arrays=dict(X=X,B=B,truth=truth,topo=topo);rows=[]
    variants=['full'] if j['study']=='fwer' else ['full','raw','haar']
    with threadpool_limits(limits=1):
        for order in ('known','estimated'):
            expected_order=None
            for v in variants:
                row=dict(j,order=order,variant=v)
                try:
                    kw=dict(p=j['p'],order=topo.tolist() if order=='known' else 'peel')
                    if v!='full':kw.update(S=0,finest_band='none' if v=='raw' else 'detail')
                    with warnings.catch_warnings(record=True) as warns:
                        warnings.simplefilter('always');res=MSWCSL(MSWCSLConfig(**kw)).fit(X)
                    if expected_order is None:expected_order=res.order.copy()
                    assert np.array_equal(expected_order,res.order)
                    c=np.zeros((j['d'],j['d']),dtype=np.int8)
                    for target,cands in enumerate(res.candidates):c[cands,target]=1
                    np.fill_diagonal(c,0);pa=truth.any(2)
                    valid=all(res.pos[i]<res.pos[k] for i,k in zip(*np.nonzero(truth[:,:,0])))
                    row.update(status='ok',metrics=score_all(truth,res.graph),parent_recall=float((c.astype(bool)&pa).sum()/max(pa.sum(),1)),false_candidates=float((c.astype(bool)&~pa).sum()/j['d']),valid_order=bool(valid),warnings=[str(w.message) for w in warns])
                    if j['study']=='cancellation':row.update(y_recall=float(c[1:3,3].mean()),y_exact=bool(np.array_equal(c[:,3],[0,1,1,0])))
                    tag=order+'_'+v;arrays[tag+'_graph']=res.graph;arrays[tag+'_candidates']=c;arrays[tag+'_order']=res.order
                except Exception as exc:row.update(status='error',error=repr(exc))
                rows.append(row)
        if j['study']=='cancellation':
            try:
                matched,co=matched_screen(X);arrays.update(co)
                rows.append(dict(j,order='known',variant='matched_theory',status='ok',matched=matched))
            except Exception as exc:rows.append(dict(j,order='known',variant='matched_theory',status='error',error=repr(exc)))
    path=Path(out)/('draw_'+j['id']+'.npz');np.savez_compressed(path,**arrays)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    for r in rows:r['archive']=path.name;r['sha256']=digest
    return rows

def main():
    from mswcsl import MSWCSLConfig
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=ROOT/'results/independent_v5');ap.add_argument('--workers',type=int,default=6);ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    work=jobs()
    if args.smoke:
        work=[next(j for j in work if j['study']==s) for s in ('cancellation','persistence','fwer')]
        work=[dict(j,seed=90260000+i,id=f'smoke_{i}') for i,j in enumerate(work)]
    args.out.mkdir(parents=True,exist_ok=False)
    files=[Path(__file__),ROOT/'INDEPENDENT_VALIDATION_PROTOCOL.md',*sorted((ROOT/'src/mswcsl').rglob('*.py'))]
    manifest=dict(jobs=work,config=MSWCSLConfig().to_dict(),versions={k:importlib.metadata.version(k) for k in ('numpy','scipy','scikit-learn','torch','PyWavelets')},sources={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},smoke=args.smoke)
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with ProcessPoolExecutor(max_workers=args.workers) as pool,(args.out/'runs.jsonl').open('w') as fh:
        futures=[pool.submit(run_one,j,str(args.out)) for j in work]
        for k,f in enumerate(as_completed(futures),1):
            for row in f.result():fh.write(json.dumps(row)+'\n')
            fh.flush()
            if k%25==0 or k==len(work):print(f'{k}/{len(work)} draws saved',flush=True)
if __name__=='__main__':main()
