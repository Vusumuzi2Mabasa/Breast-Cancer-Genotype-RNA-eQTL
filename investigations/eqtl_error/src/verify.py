from pathlib import Path
import json,time
import numpy as np
from methods import instrumented,evaluate,wild,simulation
from eqtl import analyse,nested
from robustness import hc3
ROOT=Path(__file__).resolve().parents[1]


def main():
    errors=[];fixtures=[];bootstrap_errors=[];timings=[]
    for scenario in ['null','abundance_only','subtype_eqtl']:
        seed=2903000001+['null','abundance_only','subtype_eqtl'].index(scenario)
        d=instrumented(240,seed,scenario);original=simulation.generate(240,seed,scenario)
        for key,value in original.items():
            if isinstance(value,np.ndarray):np.testing.assert_array_equal(d[key],value)
            else:assert d[key]==value
        p,_=evaluate(d,d['RNA']);ordinary=analyse(d);robust=hc3(d)
        for j in range(12):
            for family,name in enumerate(['omnibus','interaction']):
                errors.extend([abs(p[0,j,family]-ordinary[j][name+'_p']),abs(p[1,j,family]-robust[j][name+'_p'])])
        began=time.perf_counter();pboot,fixture=wild(d,999,seed+4000000000,True);timings.append(time.perf_counter()-began)
        assert np.all((pboot>=.001)&(pboot<=1));fixtures.append(fixture)
        x=np.array(fixture['x']);p0=fixture['p0'];y=np.array(fixture['y']);weights=np.array(fixture['multipliers'])
        x0=x[:,:p0];b=np.linalg.lstsq(x0,y,rcond=None)[0];res=y-x0@b;h=np.sum(np.linalg.qr(x0,mode='reduced')[0]**2,axis=1)
        adjusted=res/np.sqrt(1-h)
        for k in range(weights.shape[1]):
            ys=x0@b+adjusted*weights[:,k]
            _,stat,_=nested(x0,x,ys);bootstrap_errors.append(abs(stat-fixture['star_F'][k]))
    assert max(errors)<1e-9 and max(bootstrap_errors)<1e-8
    (ROOT/'results/pilot_fixtures.json').write_text(json.dumps(fixtures),encoding='utf-8',newline='\n')
    result=dict(status='passed',generator_equivalence_cohorts=3,maximum_original_p_discrepancy=max(errors),naive_bootstrap_refits=len(bootstrap_errors),maximum_bootstrap_F_discrepancy=max(bootstrap_errors),wild_seconds_per_cohort=timings)
    (ROOT/'results/implementation_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8',newline='\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
