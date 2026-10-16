"""Build the regime figure from the current main tables, without fitting models."""
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/mswcsl_matplotlib')
import json,re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[1]
DATASETS=['fMRI','DREAM3 yeast1','DREAM3 yeast2','Syn-6','CauseMe N=5','CauseMe N=10','Finance 20-1A','Finance 20-1B']
def parsed(cell):
    values=re.findall(r'\d*\.\d+|\d+',cell)
    return tuple(map(float,values[:2])) if cell.strip()!='NC' else None

def source_rows():
    result={}
    for target in ['Both','Lagged','Summary']:
        rows=[]
        for line in (ROOT/f'tex/tables/benchmark_main_{target.lower()}.tex').read_text().splitlines():
            if not line.endswith(r'\\') or ' & ' not in line:continue
            cs=line[:-2].split(' & ')
            if len(cs)!=7 or not any(r'\pm' in c for c in cs):continue
            name=re.sub(r'\\rowcolor\{[^}]+\}','',cs[0])
            rows.append((name,[parsed(c) for c in cs[1:]]))
        assert rows and rows[-1][0]=='MSWCSL',target
        result[target]=rows
    return result

def mechanism_plot(rep):
    cells={q['cell']:q for q in rep['summary'] if q['study']=='persistence'}
    groups=[('High persistence, spike ($a=0.9$)',[c for c,q in cells.items() if q['a']==.9 and q['shape']=='spike']),
            ('High persistence, distributed ($a=0.9$)',[c for c,q in cells.items() if q['a']==.9 and q['shape']!='spike']),
            ('Moderate persistence ($a=0.6$)',[c for c,q in cells.items() if q['a']==.6])]
    fig,axs=plt.subplots(1,2,figsize=(9.4,5.6),sharey=True)
    fig.subplots_adjust(left=.28,right=.985,top=.90,bottom=.10,wspace=.15)
    rows=[];heads=[];y=0
    for title,cs in groups:
        heads.append((y,title));y+=1.1
        for c in sorted(cs,key=lambda c:(cells[c]['T'],cells[c]['s'],cells[c]['shape'])):
            rows.append((y,c));y+=1
        y+=.55
    export=[]
    for ax,comparison,title in zip(axs,['full-raw','full-haar'],['(a) Full $-$ raw','(b) Full $-$ (raw + Haar)']):
        for yy,c in rows:
            q=next(q for q in rep['paired'] if q['cell']==c and q['comparison']==comparison and q['order']=='estimated' and q['target']=='both')
            lo,hi=q['simultaneous_CI'];m=q['mean']
            color,face=('black','black') if lo>0 else ('black','white') if hi<0 else ('#8c8c8c','#8c8c8c')
            ax.errorbar(m,yy,xerr=[[max(0,m-lo)],[max(0,hi-m)]],fmt='o',ms=4.3,color=color,mfc=face,mec=color,mew=.9,capsize=2,lw=.9)
            export.append(q)
        for (yy,title_group),end in zip(heads,[heads[1][0]-.3,heads[2][0]-.3,y-.5]):
            if yy>0:ax.axhline(yy-.75,color='#bdbdbd',lw=.5)
            ax.text(.02,yy,title_group,transform=ax.get_yaxis_transform(),fontsize=10,va='center',style='italic',bbox=dict(fc='white',ec='none',pad=1.2),zorder=5)
        ax.axvline(0,color='black',lw=.7);ax.set_title(title,fontsize=12);ax.grid(axis='x',color='#e3e3e3',lw=.5)
        ax.set_xlabel('Paired difference in Both F1',fontsize=11)
        ax.set_ylim(y-.3,-.8)
    labels=[]
    for yy,c in rows:
        q=cells[c];labels.append(f"$T={q['T']}$, $s={q['s']}$"+(', '+q['shape'] if q['a']==.6 else ''))
    axs[0].set_yticks([yy for yy,c in rows],labels,fontsize=10.5)
    from matplotlib.lines import Line2D as L
    fig.legend(handles=[L([],[],marker='o',color='black',mfc='black',ls='none',label='interval above zero'),
                        L([],[],marker='o',color='#8c8c8c',mfc='#8c8c8c',ls='none',label='interval contains zero'),
                        L([],[],marker='o',color='black',mfc='white',ls='none',label='interval below zero')],
               loc='lower center',bbox_to_anchor=(.6,-.035),ncol=3,frameon=False,fontsize=10)
    for ext in ['pdf','png']:fig.savefig(ROOT/f'figures/persistence_mechanism.{ext}',dpi=200,bbox_inches='tight')
    plt.close(fig)
    return export

def cancellation_plot(rep):
    fig,axs=plt.subplots(2,2,figsize=(9.4,5.5),sharey=True)
    fig.subplots_adjust(left=.08,right=.99,top=.92,bottom=.10,wspace=.10,hspace=.35)
    style={'raw':dict(marker='o',color='black',mfc='white'),'haar':dict(marker='D',color='#6e6e6e',mfc='#6e6e6e'),
           'full':dict(marker='o',color='black',mfc='black')}
    export=[]
    for col,T in enumerate([200,1000]):
        cc=sorted({q['cell'] for q in rep['summary'] if q['study']=='cancellation' and q['T']==T})
        for row in [0,1]:
            ax=axs[row,col]
            for g,label in enumerate(['$a=0.9$','$a=0.98$','$a=$ MD']):
                ax.axvspan(3*g-.5,3*g+2.5,color=['#f2f2f2','#fff','#f2f2f2'][g],zorder=-2,lw=0)
                ax.text(3*g+1,1.15,label,ha='center',fontsize=10)
            ax.set_xlim(-.5,8.5);ax.set_ylim(-.04,1.26);ax.set_yticks([0,.5,1]);ax.set_xticks(range(9),['.25','.5','1']*3,fontsize=10)
            ax.grid(axis='y',color='#e3e3e3',lw=.5);ax.set_title(f"{'Practical penalty' if row==0 else 'Matched theorem penalty'}, $T={T}$",fontsize=11)
            if row==1:ax.set_xlabel('effect strength $b$ (within persistence group)',fontsize=10)
        for v,off in [('raw',-.2),('haar',0),('full',.2)]:
            pts=[next(q for q in rep['summary'] if q['cell']==c and q['order']=='known' and q['variant']==v) for c in cc]
            axs[0,col].plot(np.arange(9)+off,[q['y_recall'] for q in pts],ls='none',ms=4.5,mew=.9,**style[v],label={'raw':'Raw','haar':'Raw+Haar','full':'Full'}[v]);export.extend(pts)
        for v,off,marker in [('raw',-.12,'o'),('haar',.12,'D')]:
            pts=[next(q for q in rep['matched'] if q['cell']==c and q['view']==v) for c in cc]
            m=np.array([q['exact'] for q in pts]);ci=np.array([q['exact_CI'] for q in pts])
            axs[1,col].errorbar(np.arange(9)+off,m,yerr=np.maximum(0,np.array([m-ci[:,0],ci[:,1]-m])),fmt=style[v]['marker'],ms=4.2,capsize=2,lw=.9,mew=.9,color=style[v]['color'],mfc=style[v]['mfc'],mec=style[v]['color'],label='Raw' if v=='raw' else 'Haar');export.extend(pts)
    axs[0,0].set_ylabel('Recall of two parents',fontsize=11)
    axs[1,0].set_ylabel('Exact-set probability',fontsize=11)
    h0,l0=axs[0,0].get_legend_handles_labels();h1,l1=axs[1,0].get_legend_handles_labels()
    fig.legend(h0,['Raw screen','Raw+Haar screen','Full library'],loc='upper center',bbox_to_anchor=(.53,1.03),
               ncol=3,frameon=False,fontsize=10)
    fig.legend(h1,['Raw view','Haar view'],loc='lower center',bbox_to_anchor=(.53,-.045),
               ncol=2,frameon=False,fontsize=10,title=None)
    for ext in ['pdf','png']:fig.savefig(ROOT/f'figures/independent_validation_v5b.{ext}',dpi=200,bbox_inches='tight')
    plt.close(fig)
    return export

def main():
    plt.rcParams.update({'font.family':'serif','font.serif':['cmr10'],'mathtext.fontset':'cm',
                         'axes.formatter.use_mathtext':True,'axes.unicode_minus':False,
                         'font.size':11,'pdf.fonttype':42,'axes.edgecolor':'black','axes.linewidth':.7,
                         'axes.spines.top':False,'axes.spines.right':False})
    from build_finance_analysis import main as finance_assets
    finance_assets()
    finance=json.loads((ROOT/'experiments/artifacts/review_finance/finance_values.json').read_text())['results']
    S=source_rows();export=[]
    fig,axs=plt.subplots(3,4,figsize=(9.4,5.8),sharey=True,
        gridspec_kw={'width_ratios':[3,1.35,2,2]})
    fig.subplots_adjust(left=.08,right=.99,bottom=.09,top=.80,hspace=.38,wspace=.16)
    groups=[([3,4,5],['Syn-6','V5-5','V5-10'],'Linear VARs','model-compatible',None),
            ([0],['fMRI'],'fMRI','hemodynamic',None),
            ([1,2],['Yeast 1','Yeast 2'],'Gene expression','replicated courses',None),
            ([6,7],['A','B'],'Finance','dense dependence',None)]
    for row,(target,rows) in enumerate(S.items()):
        for col,(indices,labels,title,sub,bg) in enumerate(groups):
            ax=axs[row,col];ax.set_ylim(-.03,1.09)
            ax.set_xlim(-.55,len(indices)-.45);ax.set_xticks(range(len(indices)),labels)
            ax.set_yticks([0,.5,1]);ax.grid(axis='y',color='#d9d9d9',linewidth=.6);ax.set_axisbelow(True)
            ax.tick_params(length=2.5,width=.6,pad=3)
            if col==0:ax.set_ylabel(target+' F1')
            if row==0:
                ax.set_title(title+'\n('+sub+')',fontsize=10.5,pad=9,linespacing=1.4)
            for pos,j in enumerate(indices):
                if j<6:
                    ours=next(cs[j] for name,cs in rows if name=='MSWCSL')
                    others=[(name,cs[j]) for name,cs in rows if name!='MSWCSL' and cs[j] is not None]
                    best=max(others,key=lambda x:x[1][0]) if others else None
                else:
                    f=next(q for q in finance if q['dataset']==DATASETS[j] and q['target']==target)
                    ours=f['ours'];best=f['strongest_completed']
                for dx,vals,face,marker in [(-.13,ours,'black','o'),(.13,best[1] if best else None,'white','s')]:
                    if vals is None:continue
                    mu,hw=vals
                    ax.errorbar(pos+dx,mu,yerr=hw,fmt=marker,color='black',mfc=face,mec='black',mew=.9,
                                ms=4.8,capsize=2.5,lw=.9,clip_on=False)
                # Direct labels preserve readability for the low-F1 DREAM3 group.
                ax.annotate(f'{ours[0]:.3f}',(pos-.13,ours[0]+ours[1]),xytext=(0,5),
                            textcoords='offset points',ha='center',fontsize=9.5,color='black')
                if not best:ax.text(pos,.46,'no\ncomparator',ha='center',va='center',fontsize=9,color='#555555')
                export.append({'dataset':DATASETS[j],'target':target,'ours':ours,'strongest_completed':best})
    legend=[Line2D([],[],marker='o',color='black',mfc='black',linestyle='none',label='MSWCSL (mean labeled)'),
            Line2D([],[],marker='s',color='black',mfc='white',linestyle='none',label='Highest-mean completed comparator')]
    fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.53,.995),ncol=2,frameon=False,fontsize=11)
    fig.text(.53,.925,'Dataset groups describe observation regimes; all panels use the same F1 scale.',ha='center',fontsize=10,color='#444444')
    for ext in ['pdf','png']:fig.savefig(ROOT/f'figures/benchmark_regimes.{ext}',dpi=200,bbox_inches='tight')
    plt.close(fig)
    out=ROOT/'experiments/artifacts/final_pass';out.mkdir(parents=True,exist_ok=True)
    (out/'benchmark_plot_values.json').write_text(json.dumps(export,indent=2)+'\n')
    rep=json.loads((ROOT/'experiments/results/independent_v5/summary.json').read_text())
    values={'persistence':mechanism_plot(rep),'cancellation':cancellation_plot(rep)}
    (out/'mechanism_plot_values.json').write_text(json.dumps(values,indent=2)+'\n')
    print(f'Plotted {len(export)} benchmark comparisons, 32 paired intervals, and all cancellation cells.')
if __name__=='__main__':main()
