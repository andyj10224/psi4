"""Time isolated mixed contractions; this is not a molecular speedup claim.

Use the newly compiled Psi4 Python. For a serial comparison set
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 before starting.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import psi4

from psi4.driver.procrouting.response.dlpno_triples_curvature import (
    MixedBlock,mixed_product,_expanded_mixed_product)
from psi4.driver.procrouting.response.dlpno_triples_response import (
    _TriplesOverlapCurvature,_mixed_transfer)


def run(rank=24,auxiliary=60,directions=9,repeats=3):
    if not hasattr(psi4.core,'dlpno_mixed_gemm'):
        raise RuntimeError('Rebuild Psi4: the native mixed GEMM binding is missing.')
    rng=np.random.default_rng(733)
    draw=lambda shape:rng.normal(scale=.1,size=shape)
    records=[]
    def compare(label,old,new):
        expected=np.asarray(old());actual=np.asarray(new())
        error=float(np.max(abs(actual-expected),initial=0.))
        np.testing.assert_allclose(actual,expected,atol=2e-10,rtol=2e-12)
        def elapsed(operation):
            times=[]
            for _ in range(repeats):
                start=perf_counter();operation();times.append(perf_counter()-start)
            return min(times)
        before,after=elapsed(old),elapsed(new)
        records.append(dict(stage=label,old_seconds=before,new_seconds=after,
                            speedup=before/after,max_error=error))
    for label,equation,shapes in (
            ('mixed AO projection','ma,Qmn,nb->Qab',[(rank+5,rank),(auxiliary,rank+5,rank+5),(rank+5,rank)]),
            ('mixed DF pair contraction','Qac,cd,Rbd,QR->ab',[(auxiliary,rank,rank),(rank,rank),
                                                           (auxiliary,rank,rank),(auxiliary,auxiliary)])):
        blocks=[MixedBlock(*(draw(shape) for _ in range(4))) for shape in shapes]
        compare(label,lambda:_expanded_mixed_product(equation,*blocks),lambda:mixed_product(equation,*blocks))
    z,t=draw((rank,)*3),draw((rank,)*3)
    s,sv=draw((rank,rank)),draw((rank,rank))
    ev=SimpleNamespace(overlap=s,doverlap=sv,fock=.3,dfock=.2)
    rows=[(SimpleNamespace(overlap=s,doverlap=draw(s.shape),fock=.3,dfock=.1),draw(s.shape),-.2)
          for _ in range(directions)]
    def old():return [np.vdot(z,_mixed_transfer(t,eu,ev,suv,fuv)) for eu,suv,fuv in rows]
    def new():
        # Include preparation in the measured column cost.
        prepared=_TriplesOverlapCurvature(z,t,s,sv)
        return [prepared.element(eu,ev,suv,fuv) for eu,suv,fuv in rows]
    compare('overlap curvature column',old,new)
    return dict(rank=rank,auxiliary=auxiliary,directions=directions,records=records,
                scope='Isolated algebra kernels only; includes overlap column preparation, excludes molecular solves.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rank',type=int,default=24)
    parser.add_argument('--auxiliary',type=int,default=60)
    parser.add_argument('--directions',type=int,default=9)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--output',help='Optional JSON destination')
    args=parser.parse_args()
    if min(args.rank,args.auxiliary,args.directions,args.repeats,args.threads)<1:
        parser.error('All sizes and counts must be positive.')
    psi4.set_num_threads(args.threads)
    result=run(args.rank,args.auxiliary,args.directions,args.repeats)
    print(json.dumps(result,indent=2))
    if args.output:Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
