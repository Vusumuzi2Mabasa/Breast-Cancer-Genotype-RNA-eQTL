"""Descriptive discovery and signature summaries from preserved fitted models.

These are post hoc summaries, not additional prospectively selected benchmarks.
The original model fits, test participants and numerical records are unchanged.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from simulation import generate
from integration import prep_fit,prep_apply
ROOT=Path(__file__).resolve().parents[1]


def basis(x):
    u,s,_=np.linalg.svd(x-x.mean(0),full_matrices=False)
    return u[:,s>max(s[0],1)*1e-10]


def capture(z,h):
    q=basis(z);truth=basis(h)
    return float(np.linalg.norm(q.T@truth,'fro')**2/truth.shape[1])


def reconstruction_r2(x,z,w):
    # A descriptive projection reconstruction, not the official MOFA posterior R2.
    return float(1-np.sum((x-z@w.T)**2)/np.sum((x-x.mean(0))**2))


def main():
    tuning=[];diagnostics=[];signatures=[];examples={}
    for path in sorted((ROOT/'results/integration').glob('*.json')):
        d=json.loads(path.read_text());selected=next(c for c in d['candidates'] if c['k']==d['selected_k'])
        row=dict(scenario=d['scenario'],n=d['n'],replicate=d['replicate'],selected_k=d['selected_k'],masked_validation_mse=selected['validation_mse'],genotype_validation_mse=selected['view_errors'][0],RNA_validation_mse=selected['view_errors'][1],diagnostic_pass=next(r['diagnostic_pass'] for r in d['records'] if r['method']=='MOFA+'))
        tuning.append(row)
        if d['replicate']!=0:continue
        fixture=np.load(path.with_suffix('.npz'))
        train=generate(d['n'],d['seed'],d['scenario']);test=generate(360,d['test_seed'],d['scenario'])
        a,params=prep_fit([train['genotype'],train['RNA']],train['covariates'])
        b=prep_apply([test['genotype'],test['RNA']],test['covariates'],params)
        zg=fixture['scores'];wg=fixture['weights_genotype'];wr=fixture['weights_RNA']
        tg=fixture['precision_genotype']
        # Genotype-only score projection for entirely unobserved RNA in new cases.
        ztest=(b[0]*tg)@wg@np.linalg.inv(np.eye(wg.shape[1])+(wg.T*tg)@wg)
        prediction=ztest@wr.T
        mse=float(np.mean((b[1]-prediction)**2));baseline=float(np.mean(b[1]**2))
        diagnostics.append(dict(scenario=d['scenario'],n=d['n'],replicate=0,seed=d['seed'],selected_k=d['selected_k'],genotype_projection_R2=reconstruction_r2(a[0],zg,wg),RNA_projection_R2=reconstruction_r2(a[1],zg,wr),residual_programme_capture=capture(zg,train['h']),test_RNA_from_genotype_mse=mse,test_training_mean_baseline_mse=baseline))
        assert np.array_equal(fixture['train_y'],train['y'])
    for path in sorted((ROOT/'results/diablo').glob('*.json')):
        d=json.loads(path.read_text());assert d['status']=='ok'
        # Map selected indices to original feature identifiers after confirming
        # preparation removes no constant molecular features in this cohort.
        train=generate(d['n'],d['seed'],d['scenario'])
        prepared,_=prep_fit([train['genotype'],train['RNA']],train['covariates'])
        assert [np.sum(np.std(x,axis=0)>1e-7) for x in prepared]==[24,12]
        for block,size in [('genotype',24),('RNA',12)]:
            selected=set(int(v) for v in np.atleast_1d(d['selected'][block]))
            assert selected<=set(range(1,size+1))
            for j in range(1,size+1):
                signatures.append(dict(scenario=d['scenario'],n=d['n'],replicate=d['replicate'],block=block,feature=j,selected=int(j in selected)))
    t=pd.DataFrame(tuning);m=pd.DataFrame(diagnostics);s=pd.DataFrame(signatures)
    t.to_csv(ROOT/'results/mofa_discovery_tuning.csv',index=False);m.to_csv(ROOT/'results/mofa_discovery_examples.csv',index=False)
    s.to_csv(ROOT/'results/diablo_signature_records.csv',index=False)
    s.groupby(['scenario','n','block','feature'],as_index=False).agg(selection_frequency=('selected','mean'),replicates=('selected','count')).to_csv(ROOT/'results/diablo_signature_frequency.csv',index=False)
    methods=['DIABLO','Combined ridge','RNA ridge','Genotype ridge','Clinical only']
    records=pd.read_csv(ROOT/'results/integration_evaluation.csv',keep_default_na=False,na_values=[''])
    records[records.method.isin(methods)].to_csv(ROOT/'results/supervised_prediction_evaluation.csv',index=False)
    # Predictive MOFA and PCA records stay in the inherited numerical archive.
    summary=pd.read_csv(ROOT/'results/integration_summary.csv',keep_default_na=False,na_values=[''])
    summary[summary.method.isin(methods)].to_csv(ROOT/'results/supervised_prediction_summary.csv',index=False)
    audit=dict(status='passed',mofa_selection_records=len(t),mofa_diagnostic_fixtures=len(m),DIABLO_fits=80,signature_records=len(s),post_hoc=True,new_model_fits=0,source_data_unchanged=True)
    (ROOT/'results/complementary_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8',newline='\n')
    print(json.dumps(audit))


if __name__=='__main__':main()
