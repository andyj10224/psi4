"""Canonical-limit checks after analytic mixed occupied/localization response.

Run in a built Psi4 environment. The remaining mixed CCSD residual is sampled;
--coarse-step tests its numerical sensitivity, not a step in the analytic
RHF/localization or first/final-PNO construction. Units are hartree/bohr^2.
"""
import argparse
import numpy as np
import psi4
from triples_curvature import benchmark, canonical_correlation_columns
from water_hvp import BASE, molecule


def configure():
    psi4.set_num_threads(1)
    psi4.set_options(dict(
        scf__diis=False, scf_initial_accelerator='NONE', basis='cc-pvdz',
        reference='rhf', scf_type='df', df_basis_cc='cc-pvdz-ri',
        df_basis_scf='cc-pvdz-jkfit', e_convergence=1.e-13, d_convergence=1.e-13,
        r_convergence=1.e-11, solver_convergence=1.e-12, dlpno_maxiter=600,
        local_convergence=1.e-14, local_gradient_convergence=1.e-12,
        dlpno_keep_lmp2_response=True, dlpno_gradient_model=True,
        dlpno_local_orbitals='BOYS', t_cut_pno_mp2=0., t_cut_pno=0.,
        t_cut_trace_mp2=0., t_cut_energy_mp2=0., t_cut_trace=0., t_cut_energy=0.,
        t_cut_pairs=1.e-9, t_cut_pairs_mp2=1.e-9, dlpno_keep_triples_response=True,
        min_tnos=3, t_cut_tno=0., t_cut_do_triples=0., t_cut_mkn_triples=0.,
        f_cut=0., f_cut_t=0.))
    psi4.set_options({k:0. for k in (
        't_cut_do t_cut_do_pre t_cut_do_ij t_cut_do_uv t_cut_mkn '
        't_cut_pre t_cut_clmo t_cut_cpao dlpno_ao_ints_tol').split()})


def verify(frozen, coordinates=(1,5), coarse_step=.008):
    result = benchmark(frozen, coordinates, complete_spaces=True,
                       coarse_step=coarse_step)
    target = canonical_correlation_columns(frozen).reshape(3,9)[:,coordinates].T
    hf = psi4.hessian('scf', molecule=molecule(BASE), dertype=2).np[list(coordinates),5]
    # complete_spaces verifies zero PNO loss and other screened corrections at
    # every sampled point; there is no PNO-loss curvature on this full-rank branch.
    total = hf + result[:,1] + result[:,3]
    errors = np.column_stack((result[:,1]-target[:,0], result[:,3]-target[:,1],
                              total-target[:,2]))
    print('Mixed localization integration', frozen, coarse_step, list(coordinates),
          'triples/CCSD/total errors', errors, 'total', total, flush=True)
    np.testing.assert_allclose(errors, 0., atol=2.e-7, rtol=0.)
    return errors


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core', choices=('both','frozen','all'), default='both')
    parser.add_argument('--coarse-step', type=float, default=.008)
    parser.add_argument('--coordinates', type=int, nargs='+', default=[1,5])
    args = parser.parse_args()
    if not args.coarse_step > 0 or any(not 0 <= u < 9 for u in args.coordinates):
        parser.error('Expected positive spacing and Cartesian indices 0 through 8.')
    configure()
    psi4.set_output_file('localization_integration.out', False)
    cores = (True,False) if args.core == 'both' else (args.core == 'frozen',)
    for frozen in cores:
        verify(frozen, tuple(args.coordinates), args.coarse_step)
