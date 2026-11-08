"""Molecular audit of the production-triples gradient (requires compiled Psi4).

Run in a fresh scratch directory. The default cc-pVDZ calculation retains the
existing all-strong CCSD reference; it exercises different tight/loose TNO spaces
and a configurable screened-triples correction. Finite differences hold every exported
stage selection fixed and use two step lengths. No numerical derivative enters
the analytic gradient implementation.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import psi4

STAGES = ('SCREENED T0', 'TIGHT T0', 'LOOSE T0', 'ITERATIVE T')
WEIGHTS = (1., 1., -1., 1.)


def run(args):
    psi4.set_memory(args.memory)
    psi4.set_num_threads(args.threads)
    psi4.set_options(dict(
        basis=args.basis, freeze_core=True, scf_type='df', pno_convergence='tight',
        dlpno_gradient_model=True, dlpno_gradient_full_triples=True, dlpno_keep_lmp2_response=True, t_cut_iter=1.e-5,
        t_cut_pairs=0., dlpno_ao_ints_tol=0., f_cut=0., f_cut_t=0., t_cut_do_uv=0.,
        t_cut_tno=1.e-8, t_cut_tno_strong=1.e-6, t_cut_tno_weak=1.e-5,
        t_cut_triples_weak=args.screen, e_convergence=1.e-13, r_convergence=1.e-11,
        d_convergence=1.e-13, local_convergence=1.e-14, local_gradient_convergence=1.e-13,
        dlpno_maxiter=600, maxiter=300))
    mol=psi4.geometry('''0 1
O  0.03 -0.02 -0.05
H  0.10 -0.77  0.55
H -0.06  0.79  0.58
units angstrom
symmetry c1
no_com
no_reorient
''')
    def pair_branch(wfn):
        domains,blocks=psi4.core.dlpno_first_pno_state(wfn)
        final=psi4.core.dlpno_response_domain_state(wfn)
        ranks={p:(v['PARENT'].np.shape,v['FRAME'].np.shape) for p,v in blocks.items()}
        final_ranks={p:x.np.shape for p,x in psi4.core.dlpno_lmp2_blocks(wfn,'FINAL_FRAME').items()}
        return domains,final,ranks,final_ranks
    g,w=psi4.gradient('dlpno-ccsd(t)',molecule=mol,dertype=1,return_wfn=True)
    assert psi4.core.get_option('DLPNO','T_CUT_ITER')==1.e-5
    pair_selection=pair_branch(w)
    xyz=mol.geometry().np.copy()
    branches={s:w.variable(f'DLPNO GRADIENT {s} BRANCH').np.copy() for s in STAGES}
    energies={s:float(w.variable(f'DLPNO GRADIENT {s} ENERGY')) for s in STAGES}
    result=dict(passed=False,version=psi4.__version__,threads=args.threads,basis=args.basis,
                screened_cutoff=args.screen,energy=w.energy(),gradient=g.np.tolist(),
                stage_energies=energies,checks=[])
    dest=Path(args.output)
    def save():dest.write_text(json.dumps(result,indent=2)+'\n')
    save()
    np.testing.assert_allclose(sum(c*energies[s] for c,s in zip(WEIGHTS,STAGES)),
                               w.variable('(T) CORRECTION ENERGY'),atol=2.e-11,rtol=0.)
    np.testing.assert_allclose(w.variable('DLPNO (T) ADJOINT ENERGY'),
                               w.variable('(T) CORRECTION ENERGY'),atol=2.e-10,rtol=0.)
    # Independent legacy production energy, with every retained pair strong and
    # every iterative triples equation active. This does not call the adjoint route.
    psi4.set_options(dict(dlpno_gradient_model=False,dlpno_gradient_full_triples=False,t_cut_iter=0.))
    legacy,legacy_wfn=psi4.energy('dlpno-ccsd(t)',molecule=mol,return_wfn=True)
    result['legacy_energy']=float(legacy)
    result['legacy_triples_correction']=float(legacy_wfn.variable('(T) CORRECTION ENERGY'))
    save()
    np.testing.assert_allclose(legacy,w.energy(),atol=2.e-9,rtol=0.)
    np.testing.assert_allclose(legacy_wfn.variable('(T) CORRECTION ENERGY'),
                               w.variable('(T) CORRECTION ENERGY'),atol=2.e-10,rtol=0.)
    psi4.set_options(dict(dlpno_gradient_model=True,dlpno_gradient_full_triples=True,t_cut_iter=1.e-5))
    direction=np.sin(np.arange(xyz.size)+.3).reshape(xyz.shape)
    direction/=np.linalg.norm(direction)
    exact=float(np.vdot(direction,g.np))
    central=[]
    for h in (.002,.001):
        values=[]
        for sign in (1,-1):
            other=mol.clone()
            other.set_geometry(psi4.core.Matrix.from_array(xyz+sign*h*direction));other.update_geometry()
            value,ow=psi4.energy('dlpno-ccsd(t)',molecule=other,return_wfn=True)
            assert pair_branch(ow)==pair_selection,"PNO/DF branch changed; choose a smaller displacement."
            for s in STAGES:
                np.testing.assert_array_equal(ow.variable(f'DLPNO GRADIENT {s} BRANCH').np,branches[s],
                                              err_msg=f'{s}: selection changed; this is not a smooth-branch derivative test.')
            values.append(value)
        central.append((values[0]-values[1])/(2*h))
        result['checks'].append(dict(step_bohr=h,energies=values,central=central[-1],analytic=exact))
        save()
    extrapolated=(4*central[1]-central[0])/3
    result['directional_error']=abs(extrapolated-exact)
    result['force_error']=float(np.max(abs(g.np.sum(axis=0))))
    result['torque_error']=float(np.max(abs(np.cross(xyz,g.np).sum(axis=0))))
    save()
    assert result['directional_error']<3.e-7,result
    assert max(result['force_error'],result['torque_error'])<3.e-8,result
    if args.compare:
        old=json.loads(Path(args.compare).read_text())
        assert old['basis']==args.basis and old['screened_cutoff']==args.screen
        np.testing.assert_allclose(g.np,old['gradient'],atol=2.e-8,rtol=0.)
    result['passed']=True;save()
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--basis',default='cc-pvdz',choices=('cc-pvdz','cc-pvtz'))
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--screen',type=float,default=1.e-5,help='Also test 0 and 1 (all triplets screened).')
    parser.add_argument('--memory',default='28 GB')
    parser.add_argument('--output',default='production_triples_gradient.json')
    parser.add_argument('--compare')
    args=parser.parse_args()
    psi4.core.set_output_file(str(Path(args.output).with_suffix('.out')),False)
    run(args)
