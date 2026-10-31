"""Readable simulation figures, example QC and data-driven paper tables."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import chi2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from simulation import generate
from eqtl import analyse,nested,fit
ROOT=Path(__file__).resolve().parents[1]
SC=['null','shared_eqtl','subtype_eqtl','abundance_only']
SL=['Null','Shared eQTL','Subtype eQTL','Abundance only']


def save(fig,name):
    fig.savefig(ROOT/f'figures/{name}.png',dpi=180,bbox_inches='tight',facecolor='white')
    p=ROOT/f'figures/{name}.svg';fig.savefig(p,bbox_inches='tight',facecolor='white');plt.close(fig)
    p.write_text('\n'.join(x.rstrip() for x in p.read_text().splitlines())+'\n',encoding='utf-8',newline='\n')


def main():
    plt.rcParams.update({'font.size':10,'axes.spines.right':False,'axes.spines.top':False})
    i=pd.read_csv(ROOT/'results/integration_summary.csv',keep_default_na=False,na_values=['']);q=pd.read_csv(ROOT/'results/eqtl_summary.csv',keep_default_na=False,na_values=[''])
    example=generate(240,805240000,'subtype_eqtl');rows=analyse(example)
    table=pd.DataFrame(rows);table.to_csv(ROOT/'results/example_eqtl.csv',index=False)
    # Genotype QC is descriptive; HWE is evaluated within synthetic ancestry strata.
    g=example['genotype'];freq=g.mean(0)/2;maf=np.minimum(freq,1-freq);hwe=[]
    for anc in range(3):
        for j in range(24):
            x=g[example['ancestry']==anc,j];p=x.mean()/2;expected=len(x)*np.array([(1-p)**2,2*p*(1-p),p*p]);obs=np.bincount(x.astype(int),minlength=3)
            reliable=bool(np.all(expected>=5));pv=float(chi2.sf(np.sum((obs-expected)**2/np.maximum(expected,1e-12)),1)) if reliable else None
            hwe.append(dict(ancestry=anc,variant=j,samples=len(x),minimum_expected=float(expected.min()),approximation_reliable=reliable,p=pv))
    pd.DataFrame(hwe).to_csv(ROOT/'results/example_HWE.csv',index=False)
    qc=dict(status='passed',samples=240,genotype_variants=24,RNA_genes=12,genotype_call_rate=float(np.isfinite(g).mean()),minimum_MAF=float(maf.min()),genotype_values_valid=bool(np.isin(g,[0,1,2]).all()),finite_RNA=bool(np.isfinite(example['RNA']).all()),nonnegative_integer_counts=bool(np.all(example['counts']>=0) and np.all(example['counts']==np.floor(example['counts']))),positive_libraries=bool(np.all(example['library']>0)),TNBC=int(example['y'].sum()),non_TNBC=int((1-example['y']).sum()),protein_features=0,somatic_features=0,synthetic_ancestry_counts=np.bincount(example['ancestry'],minlength=3).tolist(),hwe_reliable_cells=sum(x['approximation_reliable'] for x in hwe),hwe_policy='Pearson approximation only where all expected genotype counts >=5, within ancestry; descriptive, no phenotype-dependent filtering; no real genotype QC validation implied.')
    assert qc['genotype_values_valid'] and qc['finite_RNA'] and qc['nonnegative_integer_counts'] and qc['positive_libraries'] and qc['minimum_MAF']>.05
    (ROOT/'results/example_QC.json').write_text(json.dumps(qc,indent=2))
    fig,axes=plt.subplots(2,2,figsize=(7.2,6.1),sharex=True)
    for col,n in enumerate([240,480]):
        for j,(family,label,color) in enumerate([('omnibus','Any cis association','#2166ac'),('interaction','Subtype interaction','#b2182b')]):
            r=q[(q.n==n)&(q.family==family)].set_index('scenario').loc[SC]
            for row,(metric,ylabel) in enumerate([('sensitivity','Sensitivity among true genes'),('FDP','Mean false discovery proportion')]):
                mu=r[metric].to_numpy();err=1.96*r[metric+'_mcse'].to_numpy()
                axes[row,col].errorbar(np.arange(4)+(j-.5)*.12,mu,yerr=err,fmt='o',capsize=2,color=color,label=label)
                axes[row,col].set(xticks=np.arange(4),xticklabels=SL,ylabel=ylabel,title=f'N={n}',ylim=(-.025,1.025) if row==0 else (-.01,max(.12,float(q.FDP.max()+2*q.FDP_mcse.max()))))
        axes[1,col].axhline(.05,color='grey',ls=':')
    h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='upper center',ncol=2,bbox_to_anchor=(.5,1.04),fontsize=9)
    for ax in axes[1]:ax.tick_params(axis='x',rotation=25)
    fig.tight_layout();save(fig,'eqtl_performance')
    robust=pd.read_csv(ROOT/'results/robustness_summary.csv',keep_default_na=False,na_values=[''])
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.7),sharey=True)
    for ax,n in zip(axes,[240,480]):
        for j,(bank,method,color,marker) in enumerate([('original','Ordinary F','#2166ac','o'),('original','HC3 Wald F','#b2182b','o'),('fresh','Ordinary F','#2166ac','s'),('fresh','HC3 Wald F','#b2182b','s')]):
            r=robust[(robust.n==n)&(robust.bank==bank)&(robust.method==method)&(robust.family=='omnibus')].set_index('scenario').loc[['null','abundance_only']]
            ax.errorbar(np.arange(2)+(j-1.5)*.1,r.FDP,yerr=1.96*r.FDP_mcse,fmt=marker,color=color,ms=4,capsize=2,label=f'{method}, {bank}')
        ax.axhline(.05,color='grey',ls=':');ax.set(title=f'No genetic expression effects, N={n}',xticks=[0,1],xticklabels=['Null','Abundance only'],ylabel='Any false cis gene',ylim=(-.01,max(.18,float(robust.FDP.max()+2*robust.FDP_mcse.max()))))
    h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,ncol=2,loc='upper center',bbox_to_anchor=(.5,1.09),fontsize=8.5)
    fig.tight_layout();save(fig,'eqtl_robustness')
    y=example['y']
    base=np.c_[np.ones(240),example['covariates'],y];gg=example['genotype'][:,:4];full=np.c_[base,gg,gg*y[:,None]]
    _,_,coef=fit(full,example['RNA'][:,0]);adjusted=example['RNA'][:,0]-example['covariates']@coef[1:7]
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.5));rng=np.random.default_rng(951)
    for group,color,label in [(0,'#2166ac','Non TNBC'),(1,'#b2182b','TNBC')]:
        idx=y==group;axes[0].scatter(gg[idx,0]+rng.normal(0,.035,idx.sum()),adjusted[idx],color=color,s=10,alpha=.5,label=label)
        means=[np.mean(adjusted[(y==group)&(gg[:,0]==v)]) if np.any((y==group)&(gg[:,0]==v)) else np.nan for v in range(3)]
        axes[0].plot(range(3),means,'o-',color=color)
    axes[0].set(title='Illustrative genotype expression pattern',xlabel='SIM SNP 01 alternate allele dosage',ylabel='Covariate adjusted log2 CPM',xticks=[0,1,2]);axes[0].legend(fontsize=8)
    # Marginal cis tag associations: distinct from gene-level joint/interaction tests.
    ps=[nested(base,np.c_[base,gg[:,j]],example['RNA'][:,0])[0] for j in range(4)]
    axes[1].plot([100,110,120,130],-np.log10(np.maximum(ps,1e-300)),'o-',color='#2166ac');axes[1].set(title='SIM GENE 01 local association',xlabel='Synthetic position in kb',ylabel='Negative log10 nominal P')
    fig.tight_layout();save(fig,'example_eqtl')
    fig,ax=plt.subplots(figsize=(5.2,4.7));matrix=-np.log10(np.maximum(table[['omnibus_q','interaction_q']].to_numpy(),1e-15));im=ax.imshow(matrix,aspect='auto',cmap='Blues',vmin=0,vmax=max(3,float(matrix.max())))
    ax.set(xticks=[0,1],xticklabels=['Any cis association','Subtype interaction'],yticks=np.arange(12),yticklabels=[f'SIM GENE {j+1:02}' for j in range(12)])
    for row in range(12):
        for col,name in enumerate(['omnibus_q','interaction_q']):
            if table.iloc[row][name]<.05:ax.text(col,row,'*',ha='center',va='center',color='white' if matrix[row,col]>matrix.max()*.6 else 'black',fontsize=14)
    fig.colorbar(im,ax=ax,label='Negative log10 BH adjusted P');fig.tight_layout();save(fig,'example_qtl_map')
    complementary_figures(example)
    print('Eight complementary analysis figures generated')


def complementary_figures(example):
    from integration import prep_fit
    fig,ax=plt.subplots(figsize=(7.4,4));ax.axis('off');ax.set_xlim(0,10);ax.set_ylim(0,5)
    ax.text(5,4.5,'Matched germline genotype and RNA\nAll participants and molecular features are simulated',ha='center',va='center',fontsize=12)
    blocks=[(1.5,'MOFA+','Molecular discovery','Factors and loadings\nReconstruction diagnostics'),(5,'DIABLO','Subtype prediction','Independent test accuracy\nSignature reproducibility'),(8.5,'cis eQTL','Genetic association','Local genotype effects\nSubtype interactions')]
    for x,method,question,outcome in blocks:
        ax.annotate('',xy=(x,3.5),xytext=(5,4),arrowprops=dict(arrowstyle='->',color='#777777'))
        ax.text(x,3.1,method,ha='center',fontsize=12,fontweight='bold');ax.text(x,2.4,question,ha='center',fontsize=10);ax.text(x,1.3,outcome,ha='center',fontsize=9.5)
    ax.text(5,.25,'Distinct questions and endpoints   |   No cross method winner is defined',ha='center',fontsize=10)
    save(fig,'study_design')
    read=lambda n:pd.read_csv(ROOT/f'results/{n}.csv',keep_default_na=False,na_values=[''])
    fixture=np.load(ROOT/'results/integration/subtype_eqtl_240_00.npz');z=fixture['scores'];y=example['y'];t=read('mofa_discovery_tuning');m=read('mofa_discovery_examples')
    fig,axes=plt.subplots(2,2,figsize=(7.4,6.2))
    for group,color,label in [(0,'#2166ac','Non TNBC'),(1,'#b2182b','TNBC')]:
        ix=y==group;axes[0,0].scatter(z[ix,0],z[ix,1],s=9,alpha=.6,color=color,label=label)
    axes[0,0].set(title='Unsupervised example factor scores',xlabel='Factor 1',ylabel='Factor 2');axes[0,0].legend(fontsize=8)
    loading=fixture['weights_RNA'];im=axes[0,1].imshow(loading,aspect='auto',cmap='RdBu_r',vmin=-np.abs(loading).max(),vmax=np.abs(loading).max())
    axes[0,1].set(title='Example RNA loadings',xlabel='Factor',yticks=np.arange(12),yticklabels=[f'G{j+1:02}' for j in range(12)],xticks=np.arange(loading.shape[1]),xticklabels=np.arange(1,loading.shape[1]+1));axes[0,1].tick_params(axis='y',labelsize=8);fig.colorbar(im,ax=axes[0,1],label='Loading')
    counts=t.selected_k.value_counts().reindex([2,4,6],fill_value=0);axes[1,0].bar([2,4,6],counts,color='#4393c3');axes[1,0].set(title='Factor budgets across 80 fits',xticks=[2,4,6],xlabel='Selected factor budget',ylabel='Number of fits')
    for n,color,offset in [(240,'#2166ac',-.06),(480,'#b2182b',.06)]:
        a=m[m.n==n].set_index('scenario').loc[SC];axes[1,1].plot(np.arange(4)+offset,a.residual_programme_capture,'o',color=color,label=f'N={n}')
    axes[1,1].set(title='Known residual programme capture',xticks=np.arange(4),xticklabels=['Null','Shared','Subtype','Mean'],ylabel='Subspace capture',ylim=(0,1));axes[1,1].legend(fontsize=8)
    fig.tight_layout();save(fig,'mofa_discovery')
    pred=read('supervised_prediction_summary');methods=['DIABLO','Combined ridge','RNA ridge','Genotype ridge','Clinical only'];colors=['#b2182b','#1b7837','#4393c3','#e08214','#777777']
    fig,axes=plt.subplots(2,1,figsize=(7.4,6),sharex=True)
    for ax,n in zip(axes,[240,480]):
        for j,(method,color) in enumerate(zip(methods,colors)):
            a=pred[(pred.n==n)&(pred.method==method)].set_index('scenario').loc[SC]
            ax.errorbar(np.arange(4)+(j-2)*.09,a.balanced_accuracy,yerr=1.96*a.balanced_accuracy_mcse,fmt='o',ms=3,capsize=2,color=color,label=method)
        ax.axhline(.5,color='grey',ls=':');ax.set(title=f'Independent subtype prediction, training N={n}',ylabel='Balanced accuracy',ylim=(.4,1.02),xticks=np.arange(4),xticklabels=SL)
    h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,ncol=3,loc='upper center',bbox_to_anchor=(.5,1.06),fontsize=9);fig.tight_layout();save(fig,'diablo_prediction')
    signature=read('diablo_signature_frequency');a=signature[(signature.n==480)&(signature.block=='RNA')]
    matrix=a.pivot(index='feature',columns='scenario',values='selection_frequency').reindex(columns=SC)
    fig,ax=plt.subplots(figsize=(6.1,4.5));im=ax.imshow(matrix,aspect='auto',cmap='Blues',vmin=0,vmax=1)
    ax.set(xticks=np.arange(4),xticklabels=SL,yticks=np.arange(12),yticklabels=[f'SIM GENE {j+1:02}' for j in range(12)]);ax.tick_params(axis='x',rotation=20)
    fig.colorbar(im,ax=ax,label='Selection frequency across 10 simulated cohorts');fig.tight_layout();save(fig,'diablo_signature')


if __name__=='__main__':main()
