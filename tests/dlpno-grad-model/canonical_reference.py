"""Regenerate the Cartesian components and amplitude response references in input.dat (optional PySCF + SciPy).

At one RHF geometry, solve canonical CCSD once. In the canonical (T) functional,
vary either its two-center metric J or its three-center integrals Q by +/- h
times the corresponding analytic nuclear integral derivative. Keep the other
integrals, the orbital coefficients, orbital energies, Fock matrix and CCSD
amplitudes fixed. This isolates exactly one explicit fitting contribution.
"""

import numpy as np
import psi4
from pyscf import ao2mo, cc, gto, lib, scf
from pyscf.cc import ccsd_t_lambda
from scipy.linalg import solve


def main():
    lib.num_threads(1)
    mol = psi4.geometry("""
        O
        H 1 0.958
        H 1 0.958 2 104.4776
        symmetry c1
    """)
    psi4.set_options(dict(basis="cc-pvdz", scf_type="df", df_basis_scf="cc-pvdz-jkfit",
                          freeze_core=True, e_convergence=1e-11, d_convergence=1e-11))
    e_scf, w = psi4.energy("hf", return_wfn=True)
    aux = psi4.core.BasisSet.build(mol, "DF_BASIS_CC", "cc-pvdz-ri", "RIFIT", "cc-pvdz")
    w.set_basisset("DF_BASIS_CC", aux)
    mints = psi4.core.MintsHelper(w)
    C, eps = w.Ca().np, w.epsilon_a().np
    zero = psi4.core.BasisSet.zero_ao_basis_set()
    n, nq, no = len(eps), aux.nbf(), w.nalpha()

    def to_mo(raw):
        return np.einsum("Qmn,mp,nq->Qpq", raw.reshape(nq, C.shape[0], C.shape[0]), C, C, optimize=True)

    J = mints.ao_eri(aux, zero, aux, zero).np.reshape(nq, nq)
    Q = to_mo(mints.ao_eri(aux, zero, w.basisset(), w.basisset()).np)
    # PySCF's RHF object operates in the Psi4 MO basis. All its Hamiltonian
    # integrals and orbital energies are supplied explicitly.
    shell = gto.M(atom="O 0 0 0; H 0 -.75 .58; H 0 .75 .58", basis="cc-pvdz", verbose=0)
    mf = scf.RHF(shell)
    mf.mo_coeff = np.eye(n)
    mf.mo_occ = np.r_[2*np.ones(no), np.zeros(n-no)]
    mf.mo_energy, mf.e_tot, mf.converged = eps, e_scf, True
    mf.get_ovlp = lambda *args: np.eye(n)

    def set_integrals(metric, raw):
        q = raw.reshape(nq, -1)
        eri = (q.T @ solve(metric, q, assume_a="pos")).reshape(n, n, n, n)
        mean_field = (2*np.einsum("pqii->pq", eri[:, :, :no, :no])
                      - np.einsum("piiq->pq", eri[:, :no, :no, :]))
        h = np.diag(eps) - mean_field  # hold the JKFIT reference Fock matrix fixed
        mf.get_hcore = lambda *args: h
        mf._eri = ao2mo.restore(8, eri, n)

    set_integrals(J, Q)
    ccw = cc.CCSD(mf, frozen=1)
    ccw.conv_tol, ccw.conv_tol_normt = 1e-13, 1e-11
    ccw.max_cycle = 200
    _, t1, t2 = ccw.kernel()
    assert ccw.converged

    # PySCF uses spin-free amplitudes. In the local solver the singles
    # multiplier is 2*l1; the ordered doubles sum contracts (2*l2-l2^T).
    eris = ccw.ao2mo()
    ok, l1, l2 = ccsd_t_lambda.kernel(ccw, eris, t1, t2, max_cycle=200, tol=1e-12)
    assert ok
    print("Lambda1.T1", 2*np.vdot(l1, t1))
    print("Lambda2.T2 (ordered half-weight)", np.vdot(2*l2 - l2.swapaxes(2, 3), t2))
    h = 1.e-4
    for rank in (1, 2):
        def rescale(step):
            u1 = (1+step)*t1 if rank == 1 else t1
            u2 = (1+step)*t2 if rank == 2 else t2
            return ccw.ccsd_t(u1, u2, eris=eris)
        print("Triples scale derivative, rank", rank, (rescale(h)-rescale(-h))/(2*h))

    def triples(metric, raw):
        set_integrals(metric, raw)
        return ccw.ccsd_t(t1, t2, eris=ccw.ao2mo())

    print("Canonical (T)", triples(J, Q))
    for h in (5e-5, 1e-5):
        metric_grad, three_grad = [], []
        for atom in range(mol.natom()):
            dJ = mints.ao_metric_deriv1(atom, "DF_BASIS_CC")
            dQ = mints.ao_3center_deriv1(atom, "DF_BASIS_CC")
            metric_grad.append([(triples(J + h*d.np, Q) - triples(J - h*d.np, Q))/(2*h) for d in dJ])
            three_grad.append([(triples(J, Q + h*to_mo(d.np)) - triples(J, Q - h*to_mo(d.np)))/(2*h) for d in dQ])
        print("h =", h, "\nMetric\n", np.array(metric_grad), "\nThree-center\n", np.array(three_grad))


if __name__ == "__main__":
    main()
