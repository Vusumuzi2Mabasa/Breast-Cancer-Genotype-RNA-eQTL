"""Frozen, resumable fresh-bank error-rate investigation."""
from pathlib import Path
import argparse,json,hashlib,sys,time
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from methods import instrumented,evaluate,wild,decisions,BASE
ROOT=Path(__file__).resolve().parents[1]
SC=['null','shared_eqtl','subtype_eqtl','abundance_only']


def lock():
    paths=['protocol.json','src/methods.py','src/run.py']
    hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}
    file=ROOT/'source_lock.json'
    original={p:hashlib.sha256((BASE/p).read_bytes()).hexdigest() for p in ['src/simulation.py','src/eqtl.py','src/robustness.py']}
    value=dict(sha256=hashes,original_sources=original)
    if file.exists():assert json.loads(file.read_text())==value
    else:file.write_text(json.dumps(value,indent=2),encoding='utf-8',newline='\n')


def chunk(task):
    began=time.perf_counter()
    phase,scenario,n,start,count=task;out=ROOT/f'results/{phase}/{scenario}_{n}_{start:04}.npz'
    if out.exists():return
    out.parent.mkdir(parents=True,exist_ok=True)
    base=1803000000 if phase=='diagnostic' else 1903000000
    records=[];pv=[];diags=[];seeds=[];fixtures=[]
    for rep in range(start,start+count):
        seed=base+SC.index(scenario)*1000000+n*1000+rep
        d=instrumented(n,seed,scenario);truth=np.c_[(d['beta']!=0)|(d['delta']!=0),d['delta']!=0]
        if phase=='diagnostic':
            assert (d['counts']>0).all()
            responses=[d['RNA'],d['latent'],np.log2(d['counts']/d['library'][:,None]*1e6)]
            values=[];probs=[];diagnostics=[]
            for response in responses:
                p,diag=evaluate(d,response);values.append(decisions(p,truth));probs.append(p);diagnostics.append(diag)
            records.append(values);pv.append(probs);diags.append(diagnostics)
        else:
            p,diag=evaluate(d,d['RNA']);boot,fixture=wild(d,999,seed+4000000000,return_fixture=(rep==0))
            p=np.concatenate([p,boot[None]],axis=0);records.append(decisions(p,truth));pv.append(p);diags.append(diag)
            if fixture is not None:fixtures.append(dict(scenario=scenario,n=n,seed=seed,**fixture))
        seeds.append(seed)
    np.savez_compressed(out,metrics=np.asarray(records),p_values=np.asarray(pv),diagnostics=np.asarray(diags),seeds=np.asarray(seeds))
    out.with_suffix('.json').write_text(json.dumps(dict(phase=phase,scenario=scenario,n=n,first=start,count=count,sha256=hashlib.sha256(out.read_bytes()).hexdigest(),seconds_per_cohort=(time.perf_counter()-began)/count),indent=2),encoding='utf-8',newline='\n')
    if fixtures:out.with_suffix('.fixture.json').write_text(json.dumps(fixtures),encoding='utf-8',newline='\n')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['diagnostic','validation'],required=True);ap.add_argument('--workers',type=int,default=2);a=ap.parse_args()
    lock();tasks=[]
    scenarios=['null','abundance_only'] if a.phase=='diagnostic' else SC
    for sc in scenarios:
        count=1000 if a.phase=='diagnostic' else (1000 if sc in ['null','abundance_only'] else 100)
        for n in [240,480]:
            for start in range(0,count,25):tasks.append((a.phase,sc,n,start,min(25,count-start)))
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        fs=[pool.submit(chunk,t) for t in tasks]
        for k,f in enumerate(as_completed(fs),1):f.result();print(a.phase,k,'/',len(tasks),flush=True)


if __name__=='__main__':main()
