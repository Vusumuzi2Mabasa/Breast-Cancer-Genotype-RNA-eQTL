"""Regression diagnostics and restricted wild bootstrap without altering old models."""
from pathlib import Path
import inspect,sys
import numpy as np
from scipy.stats import f
BASE=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(BASE/'src'))
import simulation
from eqtl import bh


def instrumented(n,seed,scenario):
    source=inspect.getsource(simulation.generate)
    needle='return dict(genotype=g,RNA=rna,'
    assert source.count(needle)==1
    source=source.replace(needle,'return dict(latent=latent,mu=mu,genotype=g,RNA=rna,')
    namespace=dict(vars(simulation));exec(compile(source,'<instrumented original generator>','exec'),namespace)
    return namespace['generate'](n,seed,scenario)


def designs(d):
    n=len(d['y']);base=np.c_[np.ones(n),d['covariates'],d['y']];out=[]
    for locus in range(6):
        g=d['genotype'][:,locus*4:locus*4+4];x=np.c_[base,g,g*d['y'][:,None]]
        rank=int(np.linalg.matrix_rank(x));assert rank==x.shape[1]
        q,r=np.linalg.qr(x,mode='reduced');bread=np.linalg.solve(r,np.eye(len(r)))
        h=np.sum(q*q,axis=1)
        out.append(dict(x=x,q=q,r=r,rinv=bread,h=h,rank=rank,p0=base.shape[1],condition=float(np.linalg.cond(x))))
    return out


def classical(y,model,p0):
    q=model['q'];theta=q.T@y;res=y-q@theta
    rss=np.sum(res*res,axis=0);extra=np.sum(theta[p0:]**2,axis=0)
    statistic=(extra/(q.shape[1]-p0))/(rss/(len(y)-q.shape[1]))
    return f.sf(statistic,q.shape[1]-p0,len(y)-q.shape[1]),statistic,res,theta


def hc3(y,model,p0,res=None,theta=None):
    if res is None or theta is None:_,_,res,theta=classical(y,model,p0)
    q=model['q'];r=model['r'];ri=model['rinv'];coef=np.linalg.solve(r,theta);idx=np.arange(p0,len(r))
    weighted=q[:,:,None]*(res/(1-model['h'])[:,None])[:,None,:]
    # Q coordinates first, then transformation back to raw-design coefficients.
    meat=np.einsum('nig,njg->gij',weighted,weighted)
    covariance=np.einsum('ij,gjk,lk->gil',ri,meat,ri)
    b=coef[idx].T;v=covariance[:,idx][:,:,idx]
    w=np.einsum('gi,gi->g',b,np.linalg.solve(v,b[...,None])[...,0])/len(idx)
    return f.sf(np.maximum(w,0),len(idx),len(y)-len(r))


def evaluate(d,response):
    ds=designs(d);p=np.empty((2,12,2));diagnostics=np.empty((12,6))
    for locus,model in enumerate(ds):
        y=response[:,locus*2:locus*2+2];pbase=model['p0']
        for family,p0 in enumerate([pbase,pbase+4]):
            ordinary,stat,res,theta=classical(y,model,p0)
            p[0,locus*2:locus*2+2,family]=ordinary
            p[1,locus*2:locus*2+2,family]=hc3(y,model,p0,res,theta)
        for j in range(2):
            e=res[:,j];v=e/np.sqrt(1-model['h']);a=np.mean(v[d['y']==1]**2);b=np.mean(v[d['y']==0]**2)
            z=(e-e.mean())/e.std()
            diagnostics[locus*2+j]=[model['h'].max(),model['condition'],model['rank'],a/b,float(np.mean(z**3)),float(np.mean(z**4)-3)]
    return p,diagnostics


def wild(d,draws,seed,return_fixture=False):
    rng=np.random.default_rng(seed);n=len(d['y'])
    multipliers=rng.choice(np.array([-1.,1.]),size=(n,draws))
    p=np.empty((12,2));fixture=None
    for locus,model in enumerate(designs(d)):
        q=model['q'];y=d['RNA'][:,locus*2:locus*2+2];rank=model['rank'];df=n-rank
        for family,p0 in enumerate([model['p0'],model['p0']+4]):
            _,observed,_,_=classical(y,model,p0)
            q0=q[:,:p0];e0=y-q0@(q0.T@y);h0=np.sum(q0*q0,axis=1)
            adjusted=e0/np.sqrt(1-h0)[:,None]
            for j in range(2):
                e=adjusted[:,j,None]*multipliers;projected=q.T@e
                total=np.sum(adjusted[:,j]**2)
                denominator=np.maximum(total-np.sum(projected*projected,axis=0),1e-12)/df
                stars=(np.sum(projected[p0:]**2,axis=0)/(rank-p0))/denominator
                p[locus*2+j,family]=(1+int(np.sum(stars>=observed[j])))/(draws+1)
                if return_fixture and locus==0 and family==0 and j==0:
                    fixture=dict(x=model['x'].tolist(),p0=p0,y=y[:,j].tolist(),multipliers=multipliers[:,:31].tolist(),observed_F=float(observed[j]),star_F=stars[:31].tolist())
    return p,fixture


def decisions(p,truth):
    # p: method x gene x family. Truth: gene x family.
    out=np.empty((len(p),2,5))
    for method in range(len(p)):
        for family in range(2):
            s=bh(p[method,:,family])<.05;t=truth[:,family]
            tp=int((s&t).sum());fp=int((s&~t).sum())
            out[method,family]=[s.sum(),fp,fp/max(s.sum(),1),tp/max(t.sum(),1),float(fp>0)]
    return out
