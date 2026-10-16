"""Audit archived predictions and produce every prespecified cell summary."""
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/mswcsl_matplotlib')
import csv,hashlib,json,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import norm,t
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
from mswcsl.metrics import score_all
OUT=ROOT/'results/independent_v5'
def wilson(k,n):
    z=norm.ppf(.975);q=k/n;den=1+z*z/n;half=z*np.sqrt(q*(1-q)/n+z*z/(4*n*n))/den
    mid=(q+z*z/(2*n))/den
    return [float(mid-half),float(mid+half)]
def interval(v,alpha=.05):
    a=np.asarray(v);mean=a.mean();half=t.ppf(1-alpha/2,len(a)-1)*a.std(ddof=1)/np.sqrt(len(a))
    return [float(mean-half),float(mean+half)]
def csvout(name,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def main():
    manifest=json.loads((OUT/'manifest.json').read_text());rows=[json.loads(x) for x in (OUT/'runs.jsonl').read_text().splitlines()]
    assert len({j['id'] for j in manifest['jobs']})==5360
    for name,h in manifest['sources'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
    byid=defaultdict(list)
    for r in rows:byid[r['id']].append(r)
    assert set(byid)=={j['id'] for j in manifest['jobs']}
    verified=0;warnings_count=0;fail=[]
    for j in manifest['jobs']:
        rr=byid[j['id']];assert len(rr)==(7 if j['study']=='cancellation' else 6 if j['study']=='persistence' else 2)
        path=OUT/rr[0]['archive'];digest=hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as data:
            for r in rr:
                assert digest==r['sha256']
                if r['status']!='ok':fail.append(r);continue
                if r['variant']=='matched_theory':continue
                tag=r['order']+'_'+r['variant'];assert score_all(data['truth'],data[tag+'_graph'])==r['metrics'];verified+=1
                warnings_count+=len(r['warnings'])
    groups=defaultdict(list)
    for r in rows:groups[(r['cell'],r['order'],r['variant'])].append(r)
    summary=[];paired=[];matched=[]
    for (cell,order,v),group in sorted(groups.items()):
        ok=[r for r in group if r['status']=='ok'];base={k:group[0][k] for k in ('cell','study','T','a','b','shape','s','kind') if k in group[0]}
        base.update(order=order,variant=v,n=len(ok),failed=len(group)-len(ok))
        if not ok:summary.append(base);continue
        if v=='matched_theory':
            for view in ('raw','haar'):
                hits=sum(r['matched'][view]['exact'] for r in ok)
                matched.append(dict(base,view=view,exact=hits/len(ok),exact_CI=wilson(hits,len(ok)),recall=float(np.mean([r['matched'][view]['recall'] for r in ok])),empty=float(np.mean([r['matched'][view]['empty'] for r in ok]))))
            continue
        for target in ('both','lagged','summary'):
            for metric in ('F1','TPR','Precision','FDR','SHD'):
                values=[r['metrics'][target][metric] for r in ok];base[target+'_'+metric]=float(np.mean(values))
        for metric in ('parent_recall','false_candidates','valid_order'):
            base[metric]=float(np.mean([r[metric] for r in ok]))
        fp=sum(r['metrics']['both']['FP']>0 for r in ok);base.update(FWER=fp/len(ok),FWER_CI=wilson(fp,len(ok)),false_edge_draws=fp)
        if group[0]['study']=='cancellation':
            base['y_recall']=float(np.mean([r['y_recall'] for r in ok]));exact=sum(r['y_exact'] for r in ok);base['y_exact']=exact/len(ok);base['y_exact_CI']=wilson(exact,len(ok))
        summary.append(base)
        if v=='full' and group[0]['study']!='fwer':
            for other in ('raw','haar'):
                comp={r['rep']:r for r in groups[cell,order,other] if r['status']=='ok'}
                pairs=[(r,comp[r['rep']]) for r in ok if r['rep'] in comp]
                for target in ('both','lagged'):
                    diff=[x['metrics'][target]['F1']-y['metrics'][target]['F1'] for x,y in pairs]
                    paired.append(dict(cell=cell,study=group[0]['study'],order=order,comparison='full-'+other,target=target,n=len(diff),mean=float(np.mean(diff)),CI=interval(diff),simultaneous_CI=interval(diff,.05/32) if group[0]['study']=='persistence' and order=='estimated' and target=='both' else None))
    report=dict(protocol='INDEPENDENT_VALIDATION_PROTOCOL.md',draws=len(byid),predictions_verified=verified,warnings=warnings_count,failures=fail,summary=summary,paired=paired,matched=matched)
    (OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n');csvout('cells.csv',summary);csvout('paired.csv',paired);csvout('matched.csv',matched)
    plot(report)
    print(json.dumps({k:report[k] for k in ('draws','predictions_verified','warnings','failures')},indent=2))
    for s in summary:
        if s['study']=='fwer':print('FWER',s['kind'],s['T'],s['order'],s['FWER'],s['FWER_CI'],'power',s['both_TPR'])
    for s in paired:
        if s['study']=='persistence' and s['order']=='estimated' and s['target']=='both':print('PAIRED',s)

def plot(rep):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10})
    sums=rep['summary'];colors={'full':'#2166ac','raw':'#b2182b','haar':'#238b45'}
    fig,axs=plt.subplots(1,2,figsize=(12,5.6),layout='constrained');ax=axs[0]
    cells=sorted({s['cell'] for s in sums if s['study']=='persistence'});labels=[]
    for c in cells:
        s=next(s for s in sums if s['cell']==c);labels.append(f"a={s['a']}, T={s['T']}, {s['shape']}, s={s['s']}")
    for other,offset in [('raw',-.15),('haar',.15)]:
        pts=[next(p for p in rep['paired'] if p['cell']==c and p['comparison']=='full-'+other and p['order']=='estimated' and p['target']=='both') for c in cells]
        y=np.arange(len(cells))+offset;m=np.array([p['mean'] for p in pts]);ci=np.array([p['simultaneous_CI'] for p in pts])
        ax.errorbar(m,y,xerr=np.maximum(0,np.array([m-ci[:,0],ci[:,1]-m])),fmt='o',ms=4,color=colors[other],label='Full minus '+('raw' if other=='raw' else 'raw+Haar'))
    ax.axvline(0,color='.4',lw=.8);ax.set_yticks(range(len(cells)),labels,fontsize=9);ax.invert_yaxis()
    ax.set_xlabel('Paired Both-F1 difference (simultaneous 95% CI)');ax.set_title('(a) Persistence design, estimated order');ax.legend(fontsize=9)
    ax=axs[1];fw=sorted({s['cell'] for s in sums if s['study']=='fwer'});labels=[]
    for c in fw:
        s=next(s for s in sums if s['cell']==c);labels.append(s['kind'].replace('_',' ')+f", T={s['T']}")
    for order,offset,color in [('known',-.12,'#2166ac'),('estimated',.12,'#b2182b')]:
        pts=[next(s for s in sums if s['cell']==c and s['order']==order and s['variant']=='full') for c in fw];m=np.array([s['FWER'] for s in pts]);ci=np.array([s['FWER_CI'] for s in pts])
        ax.errorbar(m,np.arange(len(fw))+offset,xerr=np.maximum(0,np.array([m-ci[:,0],ci[:,1]-m])),fmt='o',ms=5,color=color,label=order+' order')
    ax.axvline(.05,color='.4',ls='--');ax.set_yticks(range(len(fw)),labels,fontsize=9);ax.invert_yaxis();ax.set_xlim(0,1)
    ax.set_xlabel('Probability of any false edge (95% Wilson CI)');ax.set_title('(b) Current refit at nominal alpha = .05');ax.legend(fontsize=9)
    for a in axs:a.grid(alpha=.15)
    fig.savefig(ROOT.parent/'figures/independent_validation_v5a.pdf');plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(12,7),layout='constrained',sharey=True)
    for col,T in enumerate((200,1000)):
        cc=sorted({s['cell'] for s in sums if s['study']=='cancellation' and s['T']==T})
        xl=[]
        for c in cc:
            s=next(s for s in sums if s['cell']==c);xl.append(('a=LU' if s['a']=='local' else f"a={s['a']}")+f"\nb={s['b']}")
        x=np.arange(len(cc));ax=axs[0,col]
        for v,off in (('raw',-.2),('haar',0),('full',.2)):
            pts=[next(s for s in sums if s['cell']==c and s['order']=='known' and s['variant']==v) for c in cc]
            ax.plot(x+off,[s['y_recall'] for s in pts],'o',ms=6,color=colors[v],label={'full':'Full','raw':'Raw','haar':'Raw+Haar'}[v])
        ax.set_title(f'(c) Implemented screen, T={T}');ax.set_xticks(x,xl,fontsize=9)
        ax=axs[1,col]
        for view,off in (('raw',-.12),('haar',.12)):
            pts=[next(s for s in rep['matched'] if s['cell']==c and s['view']==view) for c in cc];m=np.array([s['exact'] for s in pts]);ci=np.array([s['exact_CI'] for s in pts])
            ax.errorbar(x+off,m,yerr=np.maximum(0,np.array([m-ci[:,0],ci[:,1]-m])),fmt='o',ms=6,color=colors[view],label='Raw' if view=='raw' else 'Haar')
        ax.set_title(f'(d) Theorem calibration, T={T}');ax.set_xticks(x,xl,fontsize=9)
    axs[0,0].set_ylabel('Recall of the two parents');axs[1,0].set_ylabel('P(exact parent set), 95% CI')
    axs[0,0].legend(fontsize=9);axs[1,0].legend(fontsize=9)
    for a in axs.flat:a.set_ylim(-.05,1.05);a.grid(alpha=.15)
    fig.savefig(ROOT.parent/'figures/independent_validation_v5b.pdf');plt.close(fig)
if __name__=='__main__':main()
