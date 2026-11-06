"""Fixed-geometry RHS replay and canonical nuclear references on water/cc-pVDZ.

Run with the development Psi4 Python environment. The nuclear comparison against
DLPNO can be enabled once the full Cartesian derivative is implemented.
"""
import numpy as np
import psi4
from scipy.linalg import expm
from psi4.driver.procrouting.response.dlpno_orbital import (
    constrained_rhf_adjoint, localization_adjoint, molecular_rhf_source)


def check(frozen, method='BOYS'):
    from psi4 import core
    psi4.core.clean()
    psi4.geometry('O\nH 1 0.958\nH 1 0.958 2 104.4776\nsymmetry c1')
    psi4.set_options(dict(basis='cc-pvdz', df_basis_scf='cc-pvdz-jkfit', df_basis_cc='cc-pvdz-ri',
                          scf_type='df', cc_type='df', freeze_core=frozen, save_jk=True,
                          e_convergence=1.e-11, d_convergence=1.e-11, r_convergence=1.e-10,
                          dlpno_gradient_model=True, dlpno_local_orbitals=method, dlpno_maxiter=600))
    grad = psi4.gradient('ccsd(t)', dertype=1).np
    oz, hy = (0.0144076115, 0.0033644851) if frozen else (0.0137443429, 0.0029508710)
    np.testing.assert_allclose(grad, [[0, 0, oz], [0, hy, -oz/2], [0, -hy, -oz/2]], atol=1.e-7)
    _, w = psi4.energy('dlpno-ccsd(t)', return_wfn=True)
    assert abs(w.variable('(T) CORRECTION ENERGY')) > 0.003
    if not frozen and method == 'IBO':
        # Three oxygen-local rotations have zero curvature and a nonzero
        # correlation source at these finite cutoffs. Refuse an arbitrary gauge.
        try:
            molecular_rhf_source(w)
        except RuntimeError as exc:
            if 'Localization adjoint residual' not in str(exc):
                raise
            return
    rhs, _ = molecular_rhf_source(w)
    ref = w.reference_wavefunction()
    constrained_rhf_adjoint(ref, rhs)
    C = ref.Ca().np
    S = ref.S().np
    F = ref.Fa().np
    no = ref.nalpha()
    nc = ref.Ca_subset('AO', 'FROZEN_OCC').np.shape[1]
    L = w.variable('DLPNO RESPONSE ACTIVE LMOS').np
    U = C[:, nc:no].T @ S @ L
    def source(label, n):
        return core.dlpno_orbital_response_action(w, label, core.Matrix.from_array(np.eye(n))).np.copy()
    gL = source('ACTIVE', L.shape[1])
    gF = source('FOCK', S.shape[0])
    gc = source('FROZEN', nc) if nc else np.zeros((S.shape[0], 0))
    mints = core.MintsHelper(ref.basisset())
    def invsqrt(M):
        d, v = np.linalg.eigh(M)
        return (v / np.sqrt(d)) @ v.T
    if method == 'IBO':
        minimal = w.get_basisset('MINAO')
        idx = w.variable('DLPNO RESPONSE IAO INDICES').np[:,0].astype(int)
        cross = mints.ao_overlap(ref.basisset(), minimal).np[:,idx]
        S22 = mints.ao_overlap(minimal, minimal).np[np.ix_(idx,idx)]
        centers = np.array([minimal.function_to_center(int(mu)) for mu in idx])
        def operators(O):
            X = np.linalg.solve(S, cross @ np.linalg.solve(S22, cross.T @ O))
            dep = X @ invsqrt(X.T @ S @ X)
            Dt = dep @ dep.T
            D = O @ O.T
            raw = (np.linalg.inv(S) + 2*D @ S @ Dt-D-Dt) @ cross
            proj = S @ raw @ invsqrt(raw.T @ S @ raw)
            return [proj[:,centers==a] @ proj[:,centers==a].T for a in np.unique(centers)]
        power = int(w.variable('DLPNO RESPONSE IAO POWER'))
    else:
        power = 2
        if method == 'BOYS':
            fixed = [m.np.copy() for m in mints.ao_dipole()]
        else:
            centers = np.array([ref.basisset().function_to_center(i) for i in range(S.shape[0])])
            fixed = [((centers==a)[:,None]*S + S*(centers==a)[None,:])/2 for a in np.unique(centers)]
        def operators(O):
            return fixed
    ops = [L.T @ d @ L for d in operators(C[:,:no])]
    z, _ = localization_adjoint(ops, power, L, gL)
    # Independent dense four-index JKFIT reference, only for this small molecule.
    aux = ref.get_basisset('DF_BASIS_SCF')
    zero = core.BasisSet.zero_ao_basis_set()
    Q = mints.ao_eri(aux, zero, ref.basisset(), ref.basisset()).np.reshape(aux.nbf(), -1)
    J = mints.ao_eri(aux, zero, aux, zero).np.reshape(aux.nbf(), aux.nbf())
    eri = (Q.T @ np.linalg.solve(J, Q)).reshape((S.shape[0],)*4)
    x = np.random.default_rng(71).normal(size=rhs.shape)
    x /= np.linalg.norm(x)
    K = np.zeros((C.shape[1],)*2)
    K[:no,no:], K[no:,:no] = x, -x.T
    i,j = np.triu_indices(L.shape[1], 1)
    def value(t):
        O = (C @ expm(t*K))[:, :no]
        density = O @ O.T - C[:,:no] @ C[:,:no].T
        df = (2*np.einsum('pqrs,rs->pq', eri, density)
              - np.einsum('prqs,rs->pq', eri, density))
        if nc:
            _, ev = np.linalg.eigh(O.T @ (F+df) @ O)
            # Transport the active/core subspaces without internal rotations.
            for block in (slice(0,nc), slice(nc,no)):
                u, _, vt = np.linalg.svd(ev[block,block])
                ev[:,block] = ev[:,block] @ (u @ vt).T
            O = O @ ev
        l = O[:,nc:] @ U
        scalar = np.sum(gL*l) + np.sum(gc*O[:,:nc]) + np.sum(gF*df)
        for d in operators(O):
            a = l.T @ d @ l
            scalar += z @ (2*power*(a.diagonal()[i]**(power-1)-a.diagonal()[j]**(power-1))*a[i,j])
        return scalar
    h = 1.e-5
    tangent = (value(h)-value(-h))/(2*h)
    print(method, 'freeze_core', frozen, 'orbital direction', tangent, 'assembled', np.sum(rhs*x))
    np.testing.assert_allclose(tangent, np.sum(rhs*x), atol=1.e-8, rtol=0)


if __name__ == '__main__':
    psi4.set_num_threads(1)
    for frozen in (True, False):
        for method in ('BOYS', 'PIPEK_MEZEY', 'IBO'):
            check(frozen, method)
