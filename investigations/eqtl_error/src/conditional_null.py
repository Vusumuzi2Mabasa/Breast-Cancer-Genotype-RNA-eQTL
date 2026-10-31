"""Fresh interaction-null confirmation with nonzero common eQTL effects."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import json,hashlib
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run import chunk,lock
ROOT=Path(__file__).resolve().parents[1]


def main():
    lock()
    value=dict(reason='The original validation plan allocated only 100 shared-eQTL power cohorts per sample size. Its interaction-null family is important and imprecise (8/100 for the candidate at N480), so add independent confirmation without altering any test.',scenario='shared_eqtl',sample_sizes=[240,480],cohorts_per_cell=1000,replicate_offset=10000,seed='1903000000 + 1000000 + N*1000 + 10000 + r',methods='Same ordinary F, HC3 Wald F and restricted HC2 wild F with 999 draws; unchanged BH families.',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    path=ROOT/'conditional_null_amendment.json'
    if path.exists():assert json.loads(path.read_text())==value
    else:path.write_text(json.dumps(value,indent=2),encoding='utf-8',newline='\n')
    tasks=[('conditional_null','shared_eqtl',n,10000+r,25) for n in [240,480] for r in range(0,1000,25)]
    with ProcessPoolExecutor(max_workers=2) as pool:
        fs=[pool.submit(chunk,t) for t in tasks]
        for k,f in enumerate(as_completed(fs),1):f.result();print('Conditional-null',k,'/',len(tasks),flush=True)


if __name__=='__main__':main()
