"""Count matrix work in actual projector-sensitive C++ expressions.

Build the probe with --count-projectors. Counts include Matrix products,
including their overlap fallback when invoked, not total CPU time or every
scalar addition. Triplet association follows Psi4's C1 cost model.
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
from lambda_reconciliation_fixture import fixture,screened,evaluate
import lambda_probe

labels=['shared_GD','retained_L1','published_L1','retained_L2','published_L2','published_L2_setup']
rows=[]
def count(no,nv,fragments=1):
    missing=[(i,j) for i in range(no) for j in range(no) if i//(no//fragments)!=j//(no//fragments)]
    x=list(screened(fixture(no=no,nv=nv,truncated=False,fov=True),missing))
    x[0]['count_flops']=True
    results={}
    for kind in ('corrected','published'):
        evaluate(x,kind)
        counts=lambda_probe.work_counts()
        results[kind]={k:dict(zip(labels,v)) for k,v in counts.items()}
    # The retained chains must use the energy's shared-index and pair/diagonal
    # tables even when general four-index overlaps are not stored. No new
    # PAO-sized contraction may be hidden in these projector-sensitive blocks.
    x[0]['low_memory_overlap']=True
    evaluate(x,'corrected')
    assert lambda_probe.work_counts()['flops']==[results['corrected']['flops'][k] for k in labels]
    evaluate(x,'corrected',4)
    assert lambda_probe.work_counts()['flops']==[results['corrected']['flops'][k] for k in labels]
    assert results['corrected']['flops']['shared_GD']>0,'Build with --count-projectors'
    row=dict(occupied=no,rank=nv,pairs=len(x[2]),neighbors=max(map(len,x[0]['lmopair_to_lmos_'])),
             fragments=fragments,counts=results)
    rows.append(row)
    return row

def phase(row,mode,key):return row['counts'][mode]['flops'][key]

# Fixed local domains: replicate disconnected three-occupied components.
local=[count(3*m,4,m) for m in (1,2,4)]
for row in local:
    m=row['fragments']
    for mode in ('corrected','published'):
        for key in labels:
            assert phase(row,mode,key)==m*phase(local[0],mode,key),(row,mode,key)

# Independent PNO-rank scaling, including the same triplet association as Matrix.
rank=[count(3,r) for r in (2,4,8)]
for a,b in zip(rank,rank[1:]):
    for mode,key in [('corrected','shared_GD'),('corrected','retained_L2'),
                     ('published','published_L2'),('published','published_L2_setup')]:
        assert phase(b,mode,key)==8*phase(a,mode,key),(mode,key,a,b)
    for mode,key in [('corrected','retained_L1'),('published','published_L1')]:
        assert phase(b,mode,key)<=8*phase(a,mode,key),(mode,key,a,b)

# Dense graph: P~No^2 and d~No. Restored matrices and higher L2 remain cubic
# in No at fixed PNO rank; the original fixed higher tensors are quartic.
dense=[count(n,3) for n in (3,6,12)]
for a,b in zip(dense,dense[1:]):
    for mode,key,factor in [('corrected','shared_GD',8),('corrected','retained_L2',8),
                            ('published','published_L2',8),('published','published_L2_setup',16)]:
        assert phase(b,mode,key)==factor*phase(a,mode,key),(mode,key,a,b)
    assert phase(b,'corrected','retained_L1')<=8*phase(a,'corrected','retained_L1')

# Isolate projector choices: complete pair graph, common PAO parent, common
# source. Different local auxiliary domains are allowed, but no sparse-bound
# correction enters this comparison. These are synthetic residuals, not forces.
differences=[]
for seed in (17,91):
    for local_fit in (False,True):
        for truncated in (False,True):
            x=fixture(seed=seed,local_fit=local_fit,truncated=truncated,fov=True)
            exact=evaluate(x,'corrected');published=evaluate(x,'published')
            err=[float(max(np.max(np.abs(a-b)) for a,b in zip(u,v))) for u,v in zip(exact,published)]
            if not truncated:assert max(err)<2e-14
            differences.append(dict(seed=seed,local_fit=local_fit,truncated=truncated,
                                    ranks=x[0]['n_pno_'],max_L1_difference=err[0],max_L2_difference=err[1]))

result={'scope':'Matrix-operation scaling and synthetic fixed-space residual changes; not molecular gradients or timings',
        'cost_cases':rows,'projector_only_residual_examples':differences}
if args.output:args.output.write_text(json.dumps(result,indent=2)+'\n')
print('PASS: fixed-domain replication is linear; rank growth is at most cubic; restored higher terms add no occupied-index power.')
print(json.dumps(differences,indent=2))
