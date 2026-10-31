"""Cohort-level uncertainty, null-response ablations and preserved-bank forensics."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import pandas as pd
from scipy.stats import binomtest,norm
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parents[1]
METHODS=['Ordinary F','HC3 Wald F','Restricted HC2 wild F']
RESPONSES=['Observed log CPM plus one','Latent Gaussian','Log CPM without offset']
FAMILIES=['omnibus','interaction']


def wilson(k,n,alpha=.05):
    z=norm.ppf(1-alpha/2);p=k/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den;radius=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return float(center-radius),float(center+radius)


def full_null(sc,family):
    return sc in ['null','abundance_only'] or (sc=='shared_eqtl' and family==1)


def true_count(sc,family):
    if sc in ['null','abundance_only']:return 0
    if sc=='shared_eqtl':return 6 if family==0 else 0
    return 9 if family==0 else 6


def holm(p):
    p=np.asarray(p);order=np.argsort(p);q=np.empty(len(p))
    q[order]=np.minimum(1,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
    return q


def main():
    inherited=pd.read_csv(BASE/'results/robustness_evaluation.csv',keep_default_na=False,na_values=[''])
    historic=[]
    for key,g in inherited.groupby(['bank','scenario','n','method','family']):
        bank,sc,n,method,family=key
        if not full_null(sc,FAMILIES.index(family)):continue
        k=int((g.FDP>0).sum());total=len(g);lo,hi=wilson(k,total)
        historic.append(dict(bank=bank,scenario=sc,n=n,method=method,family=family,rejections=k,cohorts=total,rate=k/total,lower=lo,upper=hi,inflation_p=float(binomtest(k,total,.05,alternative='greater').pvalue)))
    h=pd.DataFrame(historic);h['holm_p']=holm(h.inflation_p);h.to_csv(ROOT/'results/historical_null_cells.csv',index=False)
    raw={};checks=[];seen={}
    for phase,expected in [('diagnostic',4000),('validation',4400),('conditional_null',2000)]:
        seeds=set();groups={}
        for path in sorted((ROOT/f'results/{phase}').glob('*.npz')):
            meta=json.loads(path.with_suffix('.json').read_text());assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['sha256']
            data=np.load(path);s=set(data['seeds'].tolist());assert len(s)==len(data['seeds']) and not seeds&s;seeds.update(s)
            for key in ['metrics','p_values','diagnostics']:assert np.isfinite(data[key]).all(),path
            assert np.all((data['p_values']>=0)&(data['p_values']<=1))
            groups.setdefault((meta['scenario'],meta['n']),[]).append({k:data[k].copy() for k in ['metrics','p_values','diagnostics','seeds']})
        assert len(seeds)==expected,(phase,len(seeds));seen[phase]=seeds
        raw[phase]={key:{k:np.concatenate([d[k] for d in v]) for k in ['metrics','p_values','diagnostics','seeds']} for key,v in groups.items()}
        checks.append(dict(phase=phase,cohorts=len(seeds)))
    phases=list(seen)
    for i,a in enumerate(phases):
        for b in phases[i+1:]:assert not seen[a]&seen[b]
    old_seeds=set()
    for bank,sc,n,rep,seed in inherited[['bank','scenario','n','replicate','seed']].drop_duplicates().itertuples(index=False,name=None):old_seeds.add(int(seed))
    assert not set().union(*seen.values())&old_seeds
    rows=[];contrasts=[];diag=[];tails=[]
    for phase,groups in raw.items():
        for (sc,n),data in groups.items():
            metrics=data['metrics'];probs=data['p_values'];diagnostics=data['diagnostics']
            if phase!='diagnostic':
                metrics=metrics[:,None];probs=probs[:,None];diagnostics=diagnostics[:,None]
            for response in range(metrics.shape[1]):
                for method in range(metrics.shape[2]):
                    for family in range(2):
                        a=metrics[:,response,method,family];p=probs[:,response,method,:,family]
                        total=len(a);k=int(np.sum(a[:,4]));null=full_null(sc,family)
                        row=dict(phase=phase,scenario=sc,n=n,response=RESPONSES[response],method=METHODS[method],family=FAMILIES[family],cohorts=total,rejections=k,mean_FDP=float(a[:,2].mean()),FDP_mcse=float(a[:,2].std(ddof=1)/np.sqrt(total)),any_false=float(a[:,4].mean()),sensitivity=float(a[:,3].mean()) if true_count(sc,family) else None,full_null=null)
                        if null:row['lower'],row['upper']=wilson(k,total)
                        else:row['lower'],row['upper']=None,None
                        row['nominal_gene_rejection']=float(np.mean(p<.05));row['bonferroni_rejection']=float(np.mean(np.any(p<.05/12,axis=1)))
                        ranked=np.sort(p,axis=1)
                        row['BY_rejection']=float(np.mean(np.any(ranked<(.05/np.sum(1/np.arange(1,13)))*np.arange(1,13)/12,axis=1)))
                        rows.append(row)
                        for threshold in [.001,.005,.01,.025,.05]:
                            rate=np.mean(p<threshold,axis=1)
                            tails.append(dict(phase=phase,scenario=sc,n=n,response=RESPONSES[response],method=METHODS[method],family=FAMILIES[family],threshold=threshold,rejection=float(rate.mean()),mcse=float(rate.std(ddof=1)/np.sqrt(total))))
                # Paired covariance-corrected method contrasts within cohorts.
                for family in range(2):
                    for other in range(1,metrics.shape[2]):
                        delta=metrics[:,response,other,family,2]-metrics[:,response,0,family,2]
                        se=delta.std(ddof=1)/np.sqrt(len(delta));mean=delta.mean()
                        contrasts.append(dict(phase=phase,scenario=sc,n=n,response=RESPONSES[response],family=FAMILIES[family],comparison=METHODS[other]+' minus Ordinary F',difference=float(mean),lower=float(mean-1.96*se),upper=float(mean+1.96*se)))
                for j,name in enumerate(['maximum_leverage','condition_number','rank','subtype_variance_ratio','skewness','excess_kurtosis']):
                    values=diagnostics[:,response,:,j]
                    diag.append(dict(phase=phase,scenario=sc,n=n,response=RESPONSES[response],quantity=name,median=float(np.median(values)),p05=float(np.quantile(values,.05)),p95=float(np.quantile(values,.95))))
            if phase=='diagnostic':
                for method in range(2):
                    for family in range(2):
                        for response in [1,2]:
                            delta=metrics[:,0,method,family,2]-metrics[:,response,method,family,2]
                            se=delta.std(ddof=1)/np.sqrt(len(delta));mean=delta.mean()
                            contrasts.append(dict(phase=phase,scenario=sc,n=n,response='Observed minus '+RESPONSES[response],family=FAMILIES[family],comparison=METHODS[method],difference=float(mean),lower=float(mean-1.96*se),upper=float(mean+1.96*se)))
    summary=pd.DataFrame(rows);summary.to_csv(ROOT/'results/error_summary.csv',index=False)
    pooled=[]
    for sc in ['null','abundance_only']:
        for n in [240,480]:
            for method in range(2):
                for family in range(2):
                    a=raw['diagnostic'][(sc,n)]['metrics'][:,0,method,family,4]
                    b=raw['validation'][(sc,n)]['metrics'][:,method,family,4]
                    values=np.r_[a,b];k=int(values.sum());lo,hi=wilson(k,len(values))
                    pooled.append(dict(scenario=sc,n=n,method=METHODS[method],family=FAMILIES[family],cohorts=len(values),rejections=k,rate=float(values.mean()),lower=lo,upper=hi))
    pd.DataFrame(pooled).to_csv(ROOT/'results/pooled_confirmation.csv',index=False)
    pd.DataFrame(contrasts).to_csv(ROOT/'results/paired_contrasts.csv',index=False);pd.DataFrame(diag).to_csv(ROOT/'results/design_diagnostics.csv',index=False);pd.DataFrame(tails).to_csv(ROOT/'results/pvalue_tails.csv',index=False)
    lock=json.loads((ROOT/'source_lock.json').read_text())
    for name,digest in lock['sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    for name,digest in lock['original_sources'].items():assert hashlib.sha256((BASE/name).read_bytes()).hexdigest()==digest,name
    amendment=json.loads((ROOT/'conditional_null_amendment.json').read_text())
    assert hashlib.sha256((ROOT/'src/conditional_null.py').read_bytes()).hexdigest()==amendment['source_sha256']
    for file in ['implementation_validation.json','R_validation.json']:assert json.loads((ROOT/'results'/file).read_text())['status']=='passed'
    result=dict(status='passed',fresh_diagnostic_cohorts=4000,independent_validation_cohorts=4400,conditional_interaction_null_cohorts=2000,total_new_cohorts=10400,summary_cells=len(summary),seed_banks_disjoint=True,source_locks_verified=True,historical_complete_null_cells=len(h),previous_results_unchanged=True,limitations='Intervals and exploratory paired contrasts are cell-specific, not simultaneous guarantees. Wild bootstrap validity remains model- and design-dependent.')
    (ROOT/'results/investigation_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8',newline='\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
