"""Small-system dense JKFIT oracle for the assembled Boys/PM water gradient.

Run with a development Psi4 installation. The Cartesian derivative uses a
separate reference JKFIT and correlation RIFIT. No nuclear displacements.
"""
import numpy as np
import psi4
from psi4.driver.p4util.exceptions import ValidationError
from psi4.driver.procrouting.response.dlpno_orbital import assemble_cartesian_gradient


def fixed_density_fock(ref):
    core = psi4.core
    basis = ref.basisset()
    mints = core.MintsHelper(ref)
    aux = ref.get_basisset('DF_BASIS_SCF')
    zero = core.BasisSet.zero_ao_basis_set()
    q = mints.ao_eri(aux, zero, basis, basis).np.reshape(aux.nbf(), -1)
    j = mints.ao_eri(aux, zero, aux, zero).np.reshape(aux.nbf(), aux.nbf())
    b = np.linalg.solve(j, q)
    D = ref.Da().np
    def derivatives(atom):
        dm = mints.ao_metric_deriv1(atom, 'DF_BASIS_SCF')
        dq = mints.ao_3center_deriv1(atom, 'DF_BASIS_SCF')
        dt = mints.ao_oei_deriv1('KINETIC', atom)
        dv = mints.ao_oei_deriv1('POTENTIAL', atom)
        result = []
        for x in range(3):
            a = dq[x].np.reshape(q.shape).T @ b
            eri = (a + a.T - b.T @ dm[x].np @ b).reshape((basis.nbf(),)*4)
            result.append(dt[x].np + dv[x].np + 2*np.einsum('pqrs,rs->pq',eri,D)
                          - np.einsum('prqs,rs->pq',eri,D))
        return result
    return derivatives


def check(frozen, localizer='BOYS', screened=False):
    psi4.core.clean_options()
    psi4.core.clean()
    psi4.geometry('O\nH 1 0.958\nH 1 0.958 2 104.4776\nsymmetry c1')
    psi4.set_options(dict(basis='cc-pvdz', df_basis_scf='cc-pvdz-jkfit', df_basis_cc='cc-pvdz-ri',
                          scf_type='df', cc_type='df', freeze_core=frozen, save_jk=True,
                          e_convergence=1.e-11, d_convergence=1.e-11, r_convergence=1.e-10,
                          dlpno_gradient_model=True, dlpno_local_orbitals=localizer, dlpno_maxiter=600))
    reference = psi4.gradient('ccsd(t)', dertype=1).np.copy()
    hf = psi4.gradient('scf', dertype=1).np.copy()
    if not screened:
        psi4.set_options({k: 0. for k in ('t_cut_do t_cut_do_pre t_cut_do_ij t_cut_do_uv t_cut_mkn '
        't_cut_pairs t_cut_pairs_mp2 t_cut_pre t_cut_clmo t_cut_cpao t_cut_pno t_cut_pno_mp2 '
        'f_cut t_cut_tno t_cut_do_triples t_cut_mkn_triples f_cut_t').split()})
    _, w = psi4.energy('dlpno-ccsd(t)', return_wfn=True)
    ref = w.reference_wavefunction()
    dense = assemble_cartesian_gradient(w, hf, fixed_density_fock(ref))
    # Exercise public dispatch with the energy-only settings at their defaults.
    psi4.set_options(dict(dlpno_gradient_model=False, save_jk=False))
    matrix, result = psi4.gradient('dlpno-ccsd(t)', return_wfn=True)
    grad = matrix.np
    np.testing.assert_array_equal(grad, result.gradient().np)
    assert not psi4.core.get_option('DLPNO', 'DLPNO_GRADIENT_MODEL')
    assert not psi4.core.get_option('SCF', 'SAVE_JK')
    print('screened JK versus dense', np.max(abs(grad-dense)), flush=True)
    np.testing.assert_allclose(grad, dense, atol=1.e-9, rtol=0)
    print(localizer, 'screened', screened, 'freeze_core', frozen, 'assembled', grad, 'canonical', reference,
          'max error', np.max(abs(grad-reference)), flush=True)
    np.testing.assert_allclose(grad.sum(axis=0), np.zeros(3), atol=1.e-8, rtol=0)
    if not screened:
        np.testing.assert_allclose(grad, reference, atol=1.e-7, rtol=0)


if __name__ == '__main__':
    psi4.set_num_threads(1)
    for frozen in (True, False):
        for method in ('BOYS', 'PIPEK_MEZEY'):
            for screened in (False, True):
                check(frozen, method, screened)


    # Exercise OptKing on the same triples-bearing molecule.
    psi4.set_options(dict(dlpno_local_orbitals='BOYS', freeze_core=True))
    try:
        psi4.optimize('dlpno-ccsd(t)', dertype=0)
    except ValidationError as exc:
        assert 'must be analytic' in str(exc)
    else:
        raise AssertionError('Optimizer accepted a numerical DLPNO gradient.')
    optimized, wfn = psi4.optimize('dlpno-ccsd(t)', return_wfn=True)
    np.testing.assert_allclose(optimized, wfn.energy(), rtol=0, atol=1.e-10)
    assert np.max(np.abs(wfn.gradient().np)) < 3.e-4
