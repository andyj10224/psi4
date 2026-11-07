"""Measure native screened-parent span stability across Cartesian displacements.

This diagnostic preserves the existing 2e-8 physical-span acceptance. It also
reports normalization errors to distinguish actual pivot-span changes from
loss of orthonormality; normalizing the native span does not waive the guard.
"""
import argparse,json
from pathlib import Path
import numpy as np
import psi4
from scipy.optimize import linear_sum_assignment
from screened_molecules import fixture,SYSTEM_SIZES
from screened_first_pno import align_snapshot
from screened_corrections import normalized_paos


def run(args):
    mol,w=fixture(args.system,args.localization);xyz=mol.geometry().np.copy()
    d0,b0=psi4.core.dlpno_screened_pair_state(w);r0=w.reference_wavefunction()
    L0=w.variable('DLPNO RESPONSE ACTIVE LMOS').np;result=dict(system=args.system,localization=args.localization,
        coordinate=args.coordinate,step_bohr=args.step,coordinates_bohr=xyz.tolist(),samples=[],passed=False)
    for delta in (args.step,-args.step,args.step/2,-args.step/2):
        moved=mol.clone();coords=xyz.copy();coords.flat[args.coordinate]+=delta
        moved.set_geometry(psi4.core.Matrix.from_array(coords));moved.update_geometry()
        _,other=psi4.energy('dlpno-ccsd(t)',molecule=moved,return_wfn=True)
        r=other.reference_wavefunction();L=other.variable('DLPNO RESPONSE ACTIVE LMOS').np
        match=L0.T@psi4.core.MintsHelper(r0).ao_overlap(r0.basisset(),r.basisset()).np@L
        _,order=linear_sum_assignment(-abs(match));_,blocks=psi4.core.dlpno_screened_pair_state(other)
        N,S=normalized_paos(r),r.S().np;parents=[]
        for p,fields in b0.items():
            q=tuple(sorted(order[list(p)]));new=blocks[q]
            ids=d0['DIPOLE_PAO'][p[0]] if p[0]==p[1] else fields['PAOS'].np[:,0].astype(int)
            seed=N[:,ids]@fields['PARENT'].np;actual=N[:,ids]@new['PARENT'].np
            gs,ga=seed.T@S@seed,actual.T@S@actual
            expected=seed@np.linalg.solve(gs,seed.T)
            normalized=actual@np.linalg.solve(ga,actual.T)
            parents.append(dict(pair=p,shape=actual.shape,
                reference_pivots=np.flatnonzero(np.any(fields['PARENT'].np!=0.,axis=1)).tolist(),
                displaced_pivots=np.flatnonzero(np.any(new['PARENT'].np!=0.,axis=1)).tolist(),
                gram_error=float(np.max(abs(ga-np.eye(len(ga))))),
                raw_span_error=float(np.max(abs(actual@actual.T-expected))),
                normalized_span_error=float(np.max(abs(normalized-expected)))))
        result['samples'].append(dict(displacement_bohr=delta,parents=parents))
        Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
        align_snapshot(other,w)
        print('native parent span',delta,max(x['raw_span_error'] for x in parents),flush=True)
    result['passed']=True;Path(args.output).write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--system',choices=SYSTEM_SIZES,default='ammonia_h2')
    p.add_argument('--localization',choices=('BOYS','PIPEK_MEZEY'),default='BOYS')
    p.add_argument('--coordinate',type=int,default=5)
    p.add_argument('--step',type=float,default=.004)
    p.add_argument('--output',default='screened_parent_spans_results.json')
    args=p.parse_args()
    if not 0<=args.coordinate<SYSTEM_SIZES[args.system] or not np.isfinite(args.step) or args.step<=0:p.error('Invalid coordinate or step.')
    psi4.set_output_file('screened_parent_spans.out',False);run(args)
