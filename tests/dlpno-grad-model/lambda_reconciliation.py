"""Audit the production Toth and reverse Lambda functions at fixed local spaces.

Build build_lambda_reconciliation_probe.py first, then pass its build directory.
The adapter executes extracted production C++ functions, including the actual
overlap table/cache code. It does not run SCF, solve Lambda, or test a nuclear
gradient. The ordinary (T) source is deliberately zero in this CCSD audit.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('probe',type=Path)
parser.add_argument('--output',type=Path)
args=parser.parse_args()
sys.path.insert(0,str(args.probe.resolve()))
from lambda_reconciliation_fixture import fixture,screened,evaluate,zero,perturb,maxdiff

rows=[]
graphs=[(3,[]),(3,[(0,2),(2,0)]),(4,[(0,2),(2,0),(1,3),(3,1)]),
        (4,[(0,3),(3,0)])]
for seed in (17,91):
    for graph,(no,missing) in enumerate(graphs):
        for truncated in (False,True):
            for local in (False,True):
                # Exercise both global and pair-specific PAO parents. Local
                # metric inverses are recomputed on each auxiliary subset.
                for local_pao in (False,True):
                    x=list(screened(fixture(seed=seed,no=no,truncated=truncated,
                                           local_fit=local,local_pao=local_pao,fov=True),missing))
                    rng=np.random.default_rng(seed+1)
                    foo=rng.normal(scale=.02,size=(no,no))
                    x[0]['F_lmo_']+=(foo+foo.T)/2
                    for sector in ('source','lambda1','lambda2','both'):
                        y=list(x)
                        if sector in ('source','lambda2'):y[3]=zero(y[3])
                        if sector in ('source','lambda1'):y[4]=zero(y[4])
                        ref=evaluate(y,'adjoint');shared=evaluate(y,'corrected');old=evaluate(y,'published')
                        error=maxdiff(ref,shared)
                        assert error<2.e-12,(seed,graph,truncated,local,local_pao,sector,error)
                        split=maxdiff(shared,evaluate(y,'corrected_split',4))
                        assert split<2.e-12,split
                        fd=[]
                        for rank in (1,2):
                            direction=x[rank+2];h=1.e-5
                            plus=evaluate(perturb(y,rank,direction,h),'lagrangian')
                            minus=evaluate(perturb(y,rank,direction,-h),'lagrangian')
                            analytic=sum(np.vdot(a,b) for a,b in zip(shared[rank-1],direction))*(1 if rank==1 else .5)
                            fd.append(float(abs((plus-minus)/(2*h)-analytic)))
                        assert max(fd)<2.e-10,(seed,graph,sector,fd)
                        threaded=maxdiff(shared,evaluate(y,'corrected',4))
                        assert threaded<2.e-12,threaded
                        cached=0.
                        for budget in (0,256,1048576):
                            data=dict(y[0],low_memory_overlap=True,cache_bytes=budget)
                            cached=max(cached,maxdiff(shared,evaluate([data,*y[1:]],'corrected',4)))
                        assert cached<2.e-12,cached
                        rows.append(dict(seed=seed,graph=graph,truncated=truncated,local_fit=local,
                                         local_pao=local_pao,sector=sector,shared_reverse=error,
                                         shared_fd=fd,threads=threaded,cache=cached,
                                         fixed_source_split=split,
                                         published_difference=maxdiff(old,ref)))
    print(f'Completed seed {seed}: {len(rows)} independent sector fixtures.',flush=True)

# Complete coordinate basis for both amplitude ranks at an independent trial
# multiplier on a screened graph. Doubles variations respect ij/ji and ii symmetry.
x=list(screened(fixture(seed=43,truncated=True,local_fit=True,local_pao=True,fov=True)))
shared=evaluate(x,'corrected');coordinate_errors=[]
for rank in (1,2):
    for p,block in enumerate(x[rank]):
        if rank==2:
            i,j=x[0]['ij_to_i_j_'][p]
            if i>j:continue
        for a,b in np.ndindex(block.shape):
            if rank==2 and i==j and a>b:continue
            d=zero(x[rank]);d[p][a,b]=1.
            if rank==2:d[x[0]['ij_to_ji_'][p]][b,a]=1.
            h=1.e-5
            numeric=(evaluate(perturb(x,rank,d,h),'lagrangian')-evaluate(perturb(x,rank,d,-h),'lagrangian'))/(2*h)
            analytic=sum(np.vdot(u,v) for u,v in zip(shared[rank-1],d))*(1 if rank==1 else .5)
            coordinate_errors.append(abs(numeric-analytic))
assert max(coordinate_errors)<2.e-10,max(coordinate_errors)
result={'scope':'Fixed-space C++ amplitude algebra; no molecular gradient validation',
        'fixtures':len(rows),'coordinate_probes':len(coordinate_errors),
        'max_shared_reverse':max(r['shared_reverse'] for r in rows),
        'max_shared_fd':max(max(r['shared_fd']) for r in rows),
        'max_coordinate_fd':float(max(coordinate_errors)),
        'max_thread_difference':max(r['threads'] for r in rows),
        'max_cache_difference':max(r['cache'] for r in rows),
        'max_fixed_source_split':max(r['fixed_source_split'] for r in rows),
        'rows':rows}
if args.output:args.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
