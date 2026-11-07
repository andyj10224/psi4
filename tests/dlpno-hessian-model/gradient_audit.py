"""Gradient-first audit: independent native energies, branch checks and threads.

Use a fresh working/scratch directory per invocation. This deliberately tests
an off-equilibrium geometry. No numerical derivative enters production code.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import psi4

from screened_first_pno import align_snapshot


def run(args):
    psi4.set_num_threads(args.threads)
    psi4.set_memory(args.memory)
    psi4.set_options(dict(basis=args.basis,freeze_core=not args.all_electron,scf_type='df',
        pno_convergence='tight',dlpno_local_orbitals=args.localization,save_jk=True,
        dlpno_gradient_model=True,dlpno_keep_lmp2_response=True,dlpno_keep_triples_response=True,
        e_convergence=1.e-14,d_convergence=1.e-14,r_convergence=1.e-11,
        solver_convergence=1.e-14,solver_maxiter=200,ints_tolerance=1.e-14,
        local_convergence=1.e-14,local_gradient_convergence=1.e-13,dlpno_maxiter=600,
        dlpno_ao_ints_tol=0.,f_cut=0.,f_cut_t=0.,t_cut_do_uv=0.))
    mol=psi4.geometry('''0 1
O  0.03 -0.02 -0.05
H  0.10 -0.77  0.55
H -0.06  0.79  0.58
units angstrom
symmetry c1
no_com
no_reorient
''')
    xyz=mol.geometry().np.copy()
    start=perf_counter()
    g,w=psi4.gradient('dlpno-ccsd(t)',molecule=mol,dertype=1,return_wfn=True)
    gradient=g.np.copy()
    sources=('CCSD','(T)','PNO PROJECTOR','PNO TRUNCATION','LMP2 RESPONSE','SC-LMP2 PAIRS')
    parts={f'{source} {term}':w.variable(f'DLPNO {source} {term} GRADIENT').np.tolist()
           for source in sources for term in ('AUXILIARY METRIC','THREE-CENTER')}
    parts['DIPOLE PAIRS EXPLICIT']=w.variable('DLPNO DIPOLE PAIRS EXPLICIT GRADIENT').np.tolist()
    settings=dict(basis=args.basis,localization=args.localization,all_electron=args.all_electron)
    result=dict(passed=False,version=psi4.__version__,threads=args.threads,settings=settings,
        gradient_seconds=perf_counter()-start,coordinates_bohr=xyz.tolist(),energy=w.energy(),
        gradient=gradient.tolist(),explicit_components=parts,finite_differences=[])
    path=Path(args.output)
    def save(): path.write_text(json.dumps(result,indent=2)+'\n')
    save()
    result['force_error']=float(np.max(abs(gradient.sum(axis=0))))
    result['torque_error']=float(np.max(abs(np.cross(xyz,gradient).sum(axis=0))))
    save()
    assert max(result['force_error'],result['torque_error'])<2.e-8
    # Test the new batched API against an independent directional native call,
    # including the frozen-active orbital gauge when a core is frozen.
    ref=w.reference_wavefunction()
    direction=np.sin(np.arange(xyz.size)+.3).reshape(xyz.shape)
    batch=psi4.core.rhf_occupied_tangents(ref)
    dc=psi4.core.rhf_occupied_tangent(ref,psi4.core.Matrix.from_array(direction)).np
    replay=sum(a*m.np for a,m in zip(direction.ravel(),batch))
    result['occupied_batch_error']=float(np.max(abs(replay-dc)))
    save()
    np.testing.assert_allclose(replay,dc,atol=2e-10,rtol=0.)
    # Parallel shell ownership must also work for non-contiguous/reversed IDs.
    old_threads=psi4.get_num_threads()
    try:
        for name in ('DF_BASIS_SCF','DF_BASIS_CC'):
            aux=list(range(ref.get_basisset(name).nbf()-1,-1,-2))
            aos=list(range(ref.basisset().nbf()-1,-1,-2))
            for u,v in ((2,2),(1,5)):
                psi4.set_num_threads(1)
                serial=psi4.core.dlpno_df_mixed_integrals(ref,name,aux,aos,u,v)
                psi4.set_num_threads(4)
                parallel=psi4.core.dlpno_df_mixed_integrals(ref,name,aux,aos,u,v)
                for a,b in zip(serial,parallel):
                    np.testing.assert_allclose(a.np,b.np,atol=2e-12,rtol=0.)
    finally:
        psi4.set_num_threads(old_threads)
    if args.compare:
        previous=json.loads(Path(args.compare).read_text())
        assert previous['settings']==settings
        np.testing.assert_allclose(previous['coordinates_bohr'],xyz,atol=1e-12,rtol=0.)
        np.testing.assert_allclose(previous['energy'],w.energy(),atol=2e-10,rtol=0.)
        result['thread_gradient_error']=float(np.max(abs(gradient-np.array(previous['gradient']))))
        save()
        np.testing.assert_allclose(gradient,previous['gradient'],atol=2e-8,rtol=0.)
        for key in parts:
            np.testing.assert_allclose(parts[key],previous['explicit_components'][key],atol=2e-8,rtol=0.)
    coordinates=list(range(xyz.size)) if args.coordinates=='all' else [int(x) for x in args.coordinates.split(',')]
    if len(set(coordinates))!=len(coordinates) or any(k<0 or k>=xyz.size for k in coordinates):
        raise ValueError('Coordinates must be distinct indices in 0..8, or all.')
    for k in coordinates:
        record=dict(coordinate=k,steps=[],analytic=float(gradient.ravel()[k]))
        result['finite_differences'].append(record)
        save()
        for step in (.004,.002,.001):
            energies=[]
            for sign in (1,-1):
                displaced=mol.clone();coords=xyz.copy();coords.flat[k]+=sign*step
                displaced.set_geometry(psi4.core.Matrix.from_array(coords));displaced.update_geometry()
                # The same gradient-model energy and discrete selection must
                # be used. This is an energy call, not a gradient difference.
                energy,other=psi4.energy('dlpno-ccsd(t)',molecule=displaced,return_wfn=True)
                align_snapshot(other,w)
                energies.append(energy)
            central=(energies[0]-energies[1])/(2*step)
            record['steps'].append(dict(step_bohr=step,energies=energies,central=central,
                                        error=abs(central-record['analytic'])))
            save()
        values=[x['central'] for x in record['steps']]
        estimates=[(4*values[i+1]-values[i])/3 for i in (0,1)]
        record['richardson']=estimates
        record['error']=abs(estimates[-1]-record['analytic'])
        record['extrapolation_change']=abs(estimates[1]-estimates[0])
        save()
        assert record['error']<2e-7,record
        assert record['extrapolation_change']<2e-7,record
    result['passed']=True
    save()
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--basis',choices=('cc-pvdz','cc-pvtz'),default='cc-pvdz')
    parser.add_argument('--localization',choices=('BOYS','PIPEK_MEZEY'),default='BOYS')
    parser.add_argument('--all-electron',action='store_true')
    parser.add_argument('--memory',default='28 GB')
    parser.add_argument('--coordinates',default='all',help='all or comma-separated Cartesian indices 0..8')
    parser.add_argument('--compare',help='JSON from another thread count or the preceding build')
    parser.add_argument('--output',default='gradient_audit.json')
    args=parser.parse_args()
    psi4.core.set_output_file(str(Path(args.output).with_suffix('.out')),False)
    run(args)
