"""PAO coefficient response, occupied-gauge cancellation and normalized-column checks."""
from unittest.mock import patch

import numpy as np
import psi4
from psi4.driver.procrouting.response import dlpno_orbital as response
from water_hvp import BASE, STRETCH, molecule


def coefficients(ref):
    C = ref.Ca_subset('AO', 'OCC').np
    S = ref.S().np
    B = np.eye(S.shape[0]) - C @ C.T @ S
    n = np.diag(B.T @ S @ B)
    scale = np.zeros_like(n)
    keep = n > 1.e-20
    scale[keep] = n[keep]**-.5
    return np.stack((B, B*scale))


def displaced(coords):
    _, ref = psi4.energy('scf', molecule=molecule(coords), return_wfn=True)
    return coefficients(ref)


def check(frozen):
    psi4.set_options({'freeze_core': frozen})
    _, w = psi4.energy('dlpno-ccsd(t)', molecule=molecule(BASE), return_wfn=True)
    ref = w.reference_wavefunction()
    out = response.pao_nuclear_tangent(w, STRETCH)
    estimates = []
    for h in (.004, .002):
        estimates.append((displaced(BASE+h*STRETCH)-displaced(BASE-h*STRETCH))/(2*h))
    fd = (4*estimates[1]-estimates[0])/3
    errors = np.max(abs(np.stack((out.raw, out.normalized))-fd), axis=(1,2))
    print('frozen', frozen, 'rotation error', out.rotation_error, 'displaced errors', errors, flush=True)
    np.testing.assert_allclose(errors, 0., atol=2.e-8)
    B, N = coefficients(ref)
    S = ref.S().np
    ds = psi4.core.MintsHelper(ref).ao_oei_deriv1('OVERLAP', 1)[2].np
    np.testing.assert_allclose(out.raw @ B + B @ out.raw, out.raw, atol=2.e-10)
    np.testing.assert_allclose(np.diag(2*out.normalized.T @ S @ N + N.T @ ds @ N), 0., atol=2.e-10)

    # Challenge the actual diagnostic with a pure gauge change, then an invalid one.
    original = response._boys_lmo_tangent
    L = w.variable('DLPNO RESPONSE ACTIVE LMOS').np
    K = np.zeros((L.shape[1], L.shape[1]))
    K[0,1], K[1,0] = 1., -1.
    def modified(*args):
        return original(*args) + L @ K
    with patch.object(response, '_boys_lmo_tangent', modified):
        shifted = response.pao_nuclear_tangent(w, STRETCH)
        np.testing.assert_allclose(shifted.raw, out.raw, atol=1.e-12, rtol=0)
        K[1,0] = 1.
        with np.testing.assert_raises_regex(RuntimeError, 'occupied-rotation cancellation'):
            response.pao_nuclear_tangent(w, STRETCH)


if __name__ == '__main__':
    psi4.set_num_threads(1)
    psi4.set_options(dict(basis='cc-pvdz', reference='rhf', scf_type='df', df_basis_scf='cc-pvdz-jkfit',
                          df_basis_cc='cc-pvdz-ri', e_convergence=1.e-12, d_convergence=1.e-12,
                          r_convergence=1.e-10, solver_convergence=1.e-11, dlpno_maxiter=600,
                          local_convergence=1.e-14, local_gradient_convergence=1.e-8,
                          dlpno_gradient_model=True, dlpno_local_orbitals='BOYS'))
    for frozen in (True, False):
        check(frozen)
