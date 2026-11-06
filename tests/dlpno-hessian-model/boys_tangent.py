"""Fixed-frame Boys orbital-projector derivatives for the water O-H stretch."""
import numpy as np
import psi4
from scipy.optimize import linear_sum_assignment
from psi4.driver.procrouting.response.dlpno_orbital import boys_nuclear_tangent
from water_hvp import BASE, STRETCH, molecule


def displaced(coords, L, basis):
    _, ref = psi4.energy('scf', molecule=molecule(coords), return_wfn=True)
    active = ref.Ca_subset('AO', 'ACTIVE_OCC').np
    overlap = psi4.core.MintsHelper(ref).ao_overlap(ref.basisset(), basis).np
    # Start on the reference localization branch, then freely relocalize.
    u, _, vt = np.linalg.svd(active.T @ overlap @ L)
    localizer = psi4.core.Localizer.build('BOYS', ref.basisset(), psi4.core.Matrix.from_array(active @ (u @ vt)))
    localizer.localize()
    assert localizer.converged
    aligned = localizer.L.np
    _, order = linear_sum_assignment(-abs(L.T @ overlap.T @ aligned))
    aligned = aligned[:, order]
    return np.einsum('mi,ni->imn', aligned, aligned)


def check(frozen):
    psi4.set_options({'freeze_core': frozen})
    _, w = psi4.energy('dlpno-ccsd(t)', molecule=molecule(BASE), return_wfn=True)
    ref = w.reference_wavefunction()
    L = w.variable('DLPNO RESPONSE ACTIVE LMOS').np
    tangent = boys_nuclear_tangent(w, STRETCH)
    t = np.einsum('mi,ni->imn', tangent, L)
    prediction = t + t.swapaxes(1,2)
    ds = psi4.core.MintsHelper(ref).ao_oei_deriv1('OVERLAP', 1)[2].np
    metric = tangent.T @ ref.S().np @ L
    np.testing.assert_allclose(metric + metric.T + L.T @ ds @ L, 0., atol=2.e-10)
    mints = psi4.core.MintsHelper(ref)
    dd = mints.ao_elec_dip_deriv1(1)
    i, j = np.triu_indices(L.shape[1], 1)
    dg = np.zeros(i.size)
    for k, dip in enumerate(mints.ao_dipole()):
        a = L.T @ dip.np @ L
        motion = tangent.T @ dip.np @ L
        da = motion + motion.T + L.T @ dd[3*k+2].np @ L
        dg += 4*((da.diagonal()[i]-da.diagonal()[j])*a[i,j]
                 + (a.diagonal()[i]-a.diagonal()[j])*da[i,j])
    print('Boys stationarity derivative', np.max(abs(dg)), flush=True)
    np.testing.assert_allclose(dg, 0., atol=2.e-8)
    estimates = []
    for h in (.004, .002):
        plus = displaced(BASE+h*STRETCH, L, ref.basisset())
        minus = displaced(BASE-h*STRETCH, L, ref.basisset())
        estimates.append((plus-minus)/(2*h))
    target = (4*estimates[1]-estimates[0])/3
    print('frozen', frozen, 'LMO projector max errors', np.max(abs(target-prediction), axis=(1,2)), flush=True)
    np.testing.assert_allclose(prediction, target, atol=2.e-8, rtol=0)


if __name__ == '__main__':
    psi4.set_num_threads(1)
    psi4.set_options(dict(basis='cc-pvdz', reference='rhf', scf_type='df', df_basis_scf='cc-pvdz-jkfit',
                          df_basis_cc='cc-pvdz-ri', e_convergence=1.e-12, d_convergence=1.e-12,
                          r_convergence=1.e-10, solver_convergence=1.e-11, dlpno_maxiter=600,
                          local_convergence=1.e-14, local_gradient_convergence=1.e-8,
                          dlpno_gradient_model=True, dlpno_local_orbitals='BOYS'))
    for frozen in (True, False):
        check(frozen)
