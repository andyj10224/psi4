"""Verify the native RHF occupied response before passing it to localization."""
import numpy as np
import psi4
from water_hvp import BASE, STRETCH, molecule


def scf(coords):
    _, w = psi4.energy('scf', molecule=molecule(coords), return_wfn=True)
    return w


def projectors(w):
    occupied = w.Ca_subset('AO', 'OCC').np
    core = w.Ca_subset('AO', 'FROZEN_OCC').np
    return np.array([occupied @ occupied.T, core @ core.T])


def check(frozen):
    psi4.set_options({'freeze_core': frozen, 'save_jk': frozen})
    w = scf(BASE)
    C = w.Ca_subset('AO', 'OCC').np
    V = w.Ca_subset('AO', 'VIR').np
    S = w.S().np
    F = w.Fa().np
    nc = w.Ca_subset('AO', 'FROZEN_OCC').np.shape[1]
    assert nc == int(frozen)
    original_jk = w.jk()
    dC = psi4.core.rhf_occupied_tangent(w, psi4.core.Matrix.from_array(STRETCH)).np
    assert w.jk() is original_jk
    dS = psi4.core.MintsHelper(w).ao_oei_deriv1('OVERLAP', 1)[2].np
    orth = dC.T @ S @ C + C.T @ S @ dC + C.T @ dS @ C
    dP = dC @ C.T
    dPc = dC[:,:nc] @ C[:,:nc].T
    expected = np.array([dP+dP.T, dPc+dPc.T])
    estimates = []
    for h in (0.002, 0.001):
        plus, minus = scf(BASE+h*STRETCH), scf(BASE-h*STRETCH)
        estimates.append(((projectors(plus)-projectors(minus))/(2*h),
                          (plus.Fa().np-minus.Fa().np)/(2*h)))
    tangent, dF = [(4*fine-coarse)/3 for coarse, fine in zip(*estimates)]
    eps = w.epsilon_a().np[:C.shape[1]]
    brillouin = V.T @ (dF @ C + F @ dC - (dS @ C + S @ dC)*eps)
    print('frozen', frozen, 'orthonormality', np.max(abs(orth)), 'Brillouin', np.max(abs(brillouin)),
          'projector errors', np.max(abs(expected-tangent), axis=(1,2)), flush=True)
    np.testing.assert_allclose(orth, 0, atol=1.e-10)
    np.testing.assert_allclose(brillouin, 0, atol=2.e-7)
    np.testing.assert_allclose(expected, tangent, atol=2.e-7, rtol=0)


if __name__ == '__main__':
    psi4.set_num_threads(1)
    psi4.set_options(dict(basis='cc-pvdz', df_basis_scf='cc-pvdz-jkfit', scf_type='df',
                          e_convergence=1.e-12, d_convergence=1.e-12,
                          solver_convergence=1.e-11, solver_maxiter=200))
    for frozen in (True, False):
        check(frozen)
