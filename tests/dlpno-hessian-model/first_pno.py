"""One actual truncated water pair; full fitting/local AO domains for this nuclear test."""
import numpy as np
import psi4
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment
from psi4.driver.procrouting.response.dlpno_orbital import (
    boys_nuclear_tangent, pao_nuclear_tangent, pair_exchange_tangent, pair_three_center_tangent, first_pno_tangent)
from water_hvp import BASE, STRETCH, molecule


def raw(ref, aux):
    basis = ref.basisset()
    mints = psi4.core.MintsHelper(ref)
    zero = psi4.core.BasisSet.zero_ao_basis_set()
    J = mints.ao_eri(aux,zero,aux,zero).np.reshape(aux.nbf(),aux.nbf())
    Q = mints.ao_eri(aux,zero,basis,basis).np.reshape(aux.nbf(),basis.nbf(),basis.nbf())
    return J, Q


def paos(ref, domain):
    C = ref.Ca_subset('AO','OCC').np
    S = ref.S().np
    B = (np.eye(len(S))-C @ C.T @ S)[:,domain]
    return B/np.sqrt(np.diag(B.T @ S @ B))


def relax_fock(ref, dC):
    # Independent small-system DF-SCF derivative provider for the verification.
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'dlpno-grad-model'))
    from cartesian_water import fixed_density_fock
    J,Q = raw(ref,ref.get_basisset('DF_BASIS_SCF'))
    n = ref.basisset().nbf()
    q = Q.reshape(len(J),-1)
    eri = (q.T @ np.linalg.solve(J,q)).reshape((n,)*4)
    half = dC @ ref.Ca_subset('AO','OCC').np.T
    dD = half+half.T
    return (fixed_density_fock(ref)(1)[2] + 2*np.einsum('pqrs,rs->pq',eri,dD)
            - np.einsum('prqs,rs->pq',eri,dD))


def check(frozen):
    psi4.set_options({'freeze_core': frozen})
    _,w = psi4.energy('dlpno-ccsd(t)', molecule=molecule(BASE), return_wfn=True)
    ref = w.reference_wavefunction()
    L = w.variable('DLPNO RESPONSE ACTIVE LMOS').np
    i,j = [int(w.variable('DLPNO FIRST PNO PROBE '+key)) for key in ('I','J')]
    domain = w.variable('DLPNO FIRST PNO PROBE DOMAIN').np[:,0].astype(int)
    X = w.variable('DLPNO FIRST PNO PROBE PARENT').np
    kept = w.variable('DLPNO FIRST PNO PROBE RETAINED').np
    rank = kept.shape[1]
    N = paos(ref,domain)
    dN = pao_nuclear_tangent(w,STRETCH).normalized[:,domain]
    dL = boys_nuclear_tangent(w,STRETCH)
    mints = psi4.core.MintsHelper(ref)
    J,Q = raw(ref, ref.get_basisset('DF_BASIS_CC'))
    dJ = mints.ao_metric_deriv1(1,'DF_BASIS_CC')[2].np.reshape(J.shape)
    dQ = mints.ao_3center_deriv1(1,'DF_BASIS_CC')[2].np.reshape(Q.shape)
    ds = mints.ao_oei_deriv1('OVERLAP',1)[2].np
    all_ao = np.arange(ref.basisset().nbf())
    args = []
    for k in (i,j):
        args.extend(pair_three_center_tangent(ref.S().np, ds, L[:,k], dL[:,k], N, dN,
                                              all_ao, all_ao, Q, dQ))
    # Verify a genuinely shortened auxiliary-center AO refit separately.
    left, right = all_ao[::2], all_ao[1::2]
    local_q = Q[:,left][:,:,right]
    local_dq = dQ[:,left][:,:,right]
    _, probe = pair_three_center_tangent(ref.S().np, ds, L[:,i], dL[:,i], N, dN,
                                         left, right, local_q, local_dq)
    def refit(step):
        S = ref.S().np + step*ds
        c = np.linalg.solve(S[np.ix_(left,left)], S[left] @ (L[:,i]+step*dL[:,i]))
        return c @ (local_q+step*local_dq) @ (N+step*dN)[right]
    h = 1.e-5
    np.testing.assert_allclose(probe,(refit(h)-refit(-h))/(2*h),atol=2.e-8,rtol=0)
    exchange, dexchange = pair_exchange_tangent(J,dJ,*args)
    F = ref.Fa().np
    dC = psi4.core.rhf_occupied_tangent(ref,psi4.core.Matrix.from_array(STRETCH)).np
    dF = relax_fock(ref,dC)
    pair = L[:,[i,j]]
    dsum = np.sum(2*dL[:,[i,j]]*(F @ pair) + pair*(dF @ pair))
    out = first_pno_tangent(N,dN,ref.S().np,ds,
                            F,dF,exchange,dexchange,np.sum(pair*(F @ pair)),dsum,X,rank)
    native = N @ kept
    np.testing.assert_allclose(out.retained_density, native @ native.T, atol=2.e-8, rtol=0)
    P, dP = out.retained_density, out.retained_density_tangent
    S = ref.S().np
    np.testing.assert_allclose(dP @ S @ P + P @ ds @ P + P @ S @ dP, dP, atol=2.e-8, rtol=0)
    assert rank < X.shape[1] and out.boundary_gap > 1.e-10

    def sample(coords):
        _,scf = psi4.energy('scf',molecule=molecule(coords),return_wfn=True)
        c = scf.Ca_subset('AO','ACTIVE_OCC').np
        cross = psi4.core.MintsHelper(scf).ao_overlap(scf.basisset(),ref.basisset()).np
        u,_,vt = np.linalg.svd(c.T @ cross @ L)
        loc = psi4.core.Localizer.build('BOYS',scf.basisset(),psi4.core.Matrix.from_array(c @ (u @ vt)))
        loc.localize()
        assert loc.converged
        l = loc.L.np
        _,order = linear_sum_assignment(-abs(L.T @ cross.T @ l))
        l = l[:,order]
        n = paos(scf,domain)
        aux = psi4.core.BasisSet.build(scf.molecule(),'DF_BASIS_CC','cc-pvdz-ri','RIFIT','cc-pvdz')
        metric,q = raw(scf,aux)
        y = n @ X
        e,z = eigh(y.T @ scf.Fa().np @ y,y.T @ scf.S().np @ y)
        y = y @ z
        qb = l[:,i] @ q @ y
        kc = qb.T @ np.linalg.solve(metric,l[:,j] @ q @ y)
        pair = l[:,[i,j]]
        t = kc/(np.sum(pair*(scf.Fa().np @ pair))-e[:,None]-e[None,:])
        tt = 2*t-t.T
        _,p = np.linalg.eigh(tt @ t.T + tt.T @ t)
        retained = y @ p[:,-rank:]
        return retained @ retained.T
    estimates = [(sample(BASE+h*STRETCH)-sample(BASE-h*STRETCH))/(2*h) for h in (.002,.001)]
    fd = (4*estimates[1]-estimates[0])/3
    error = np.max(abs(fd-out.retained_density_tangent))
    print('first PNO',frozen,i,j,rank,X.shape[1],out.boundary_gap,error,flush=True)
    np.testing.assert_allclose(fd,out.retained_density_tangent,atol=1.e-7,rtol=0)


if __name__ == '__main__':
    psi4.set_num_threads(1)
    psi4.set_options(dict(basis='cc-pvdz',reference='rhf',scf_type='df',df_basis_cc='cc-pvdz-ri',
                          df_basis_scf='cc-pvdz-jkfit',e_convergence=1.e-12,d_convergence=1.e-12,
                          r_convergence=1.e-10,solver_convergence=1.e-11,dlpno_maxiter=600,
                          local_convergence=1.e-14,local_gradient_convergence=1.e-8,dlpno__debug=1,
                          dlpno_gradient_model=True,dlpno_local_orbitals='BOYS',t_cut_pno_mp2=1.e-5))
    psi4.set_options({k:0. for k in ('t_cut_do t_cut_do_pre t_cut_do_ij t_cut_do_uv t_cut_mkn '
                                      't_cut_pairs t_cut_pairs_mp2 t_cut_pre t_cut_clmo t_cut_cpao '
                                      'dlpno_ao_ints_tol').split()})
    for frozen in (True, False):
        check(frozen)
