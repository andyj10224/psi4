"""Water first-space LMP2 amplitude derivatives, checked against reconverged pair tensors."""
import numpy as np
import psi4
from scipy.optimize import linear_sum_assignment
from scipy.sparse.linalg import LinearOperator, gmres
from psi4.driver.procrouting.response.dlpno_orbital import (
    boys_nuclear_tangent, first_pno_tangent, coupled_lmp2_tangent, pair_exchange_tangent)
from first_pno import raw, relax_fock
from water_hvp import BASE, STRETCH, molecule


def calculation(coords):
    _, w = psi4.energy('dlpno-ccsd(t)', molecule=molecule(coords), return_wfn=True)
    return w


def blocks(w, name):
    return {ij: m.np.copy() for ij,m in psi4.core.dlpno_lmp2_blocks(w,name).items()}


def check(frozen):
    psi4.set_options({'freeze_core': frozen})
    w = calculation(BASE)
    ref = w.reference_wavefunction()
    S, F = ref.S().np, ref.Fa().np
    C, V = [ref.Ca_subset('AO',s).np for s in ('OCC','VIR')]
    L = w.variable('DLPNO RESPONSE ACTIVE LMOS').np
    dL = boys_nuclear_tangent(w,STRETCH)
    dC = psi4.core.rhf_occupied_tangent(ref,psi4.core.Matrix.from_array(STRETCH)).np
    mints = psi4.core.MintsHelper(ref)
    ds = mints.ao_oei_deriv1('OVERLAP',1)[2].np
    df = relax_fock(ref,dC)
    dV = -C @ (dC.T @ S @ V + C.T @ ds @ V) - .5*V @ (V.T @ ds @ V)
    J,Q = raw(ref,ref.get_basisset('DF_BASIS_CC'))
    dJ = mints.ao_metric_deriv1(1,'DF_BASIS_CC')[2].np.reshape(J.shape)
    dQ = mints.ao_3center_deriv1(1,'DF_BASIS_CC')[2].np.reshape(Q.shape)
    # With full AO and auxiliary domains the canonical virtuals span each parent.
    qall = [col @ Q for col in L.T]
    dqall = [dL[:,i] @ Q + L[:,i] @ dQ for i in range(L.shape[1])]
    def exchange(ij,y,dy):
        args = []
        for i in ij:
            args.extend((qall[i] @ y, dqall[i] @ y + qall[i] @ dy))
        return pair_exchange_tangent(J,dJ,*args)
    frames, amplitudes = blocks(w,'FRAME'), blocks(w,'AMPLITUDE')
    foo = L.T @ F @ L
    half = dL.T @ F @ L
    dfoo = half + half.T + L.T @ df @ L
    first = {}
    for (i,j),y in frames.items():
        k,dk = exchange((i,j),V,dV)
        first[i,j] = first_pno_tangent(V,dV,S,ds,F,df,k,dk,foo[i,i]+foo[j,j],
                                       dfoo[i,i]+dfoo[j,j],np.eye(V.shape[1]),y.shape[1],i==j)
    out = coupled_lmp2_tangent(w,first,lambda ij,y,dy: exchange(ij,y,dy)[1],ds,df,dfoo)
    dphysical = {}
    for ij, y in frames.items():
        dy, t = out.frames[ij], amplitudes[ij]
        dphysical[ij] = dy @ t @ y.T + y @ out.amplitudes[ij] @ y.T + y @ t @ dy.T
    cutoff = psi4.core.get_option('DLPNO','F_CUT')
    def sample(coords):
        _, scf = psi4.energy('scf',molecule=molecule(coords),return_wfn=True)
        active = scf.Ca_subset('AO','ACTIVE_OCC').np
        cross = psi4.core.MintsHelper(scf).ao_overlap(scf.basisset(),ref.basisset()).np
        u,_,vt = np.linalg.svd(active.T @ cross @ L)
        loc = psi4.core.Localizer.build('BOYS',scf.basisset(),psi4.core.Matrix.from_array(active @ (u @ vt)))
        loc.localize()
        assert loc.converged
        ov = L.T @ cross.T @ loc.L.np
        _, order = linear_sum_assignment(-abs(ov))
        l = loc.L.np[:,order]*np.sign(ov[np.arange(len(order)),order])
        virtual = scf.Ca_subset('AO','VIR').np
        eps = scf.epsilon_a().np[scf.nalpha():]
        f = l.T @ scf.Fa().np @ l
        aux = psi4.core.BasisSet.build(scf.molecule(),'DF_BASIS_CC','cc-pvdz-ri','RIFIT','cc-pvdz')
        metric,q = raw(scf,aux)
        qb = [c @ q @ virtual for c in l.T]
        y, kpair, fp = {}, {}, {}
        for ij in frames:
            i,j = ij
            k = qb[i].T @ np.linalg.solve(metric,qb[j])
            t = k/(f[i,i]+f[j,j]-eps[:,None]-eps[None,:])
            tt = 2*t-t.T
            _, rot = np.linalg.eigh((tt @ t.T + tt.T @ t)/(1+(i==j)))
            rot = rot[:,-frames[ij].shape[1]:]
            y[ij] = virtual @ rot
            kpair[ij] = rot.T @ k @ rot
            fp[ij] = rot.T @ (eps[:,None]*rot)
        edges = []
        for i,j in frames:
            for k in range(l.shape[1]):
                for idx,a,b in (((k,j),i,k), ((i,k),k,j)):
                    owner = tuple(sorted(idx))
                    if a != b and abs(foo[a,b]) > cutoff and owner in y:
                        overlap = y[i,j].T @ scf.S().np @ y[owner]
                        edges.append(((i,j), owner, idx!=owner, f[a,b], overlap))
        assert len(edges) == int(w.variable('DLPNO LMP2 RESPONSE EDGES'))
        def pack(t):
            return np.concatenate([t[ij].ravel() for ij in frames])
        def unpack(flat):
            start, t = 0, {}
            for ij in frames:
                n = frames[ij].shape[1]
                t[ij] = flat[start:start+n*n].reshape(n,n)
                start += n*n
            return t
        def apply(flat):
            t = unpack(flat)
            r = {ij:fp[ij]@tp+tp@fp[ij]-(f[ij[0],ij[0]]+f[ij[1],ij[1]])*tp for ij,tp in t.items()}
            for ij,owner,transpose,fij,s in edges:
                r[ij] -= fij*s @ (t[owner].T if transpose else t[owner]) @ s.T
            return pack(r)
        rhs = -pack(kpair)
        op = LinearOperator((rhs.size,rhs.size),matvec=apply,dtype=float)
        flat, info = gmres(op,rhs,rtol=2.e-13,atol=0.)
        assert info == 0 and np.max(abs(apply(flat)-rhs)) < 1.e-11
        return {ij:y[ij] @ t @ y[ij].T for ij,t in unpack(flat).items()}
    at_origin = sample(BASE)
    for ij,y in frames.items():
        np.testing.assert_allclose(at_origin[ij],y @ amplitudes[ij] @ y.T,atol=2.e-8,rtol=0)
    estimates = []
    for h in (.001,.0005):
        plus, minus = sample(BASE+h*STRETCH), sample(BASE-h*STRETCH)
        estimates.append({ij:(plus[ij]-minus[ij])/(2*h) for ij in frames})
    errors = [np.max(abs(dphysical[ij]-(4*estimates[1][ij]-estimates[0][ij])/3)) for ij in frames]
    rms = w.variable('DLPNO LMP2 NUCLEAR TANGENT RESIDUAL')
    print('frozen',frozen,'unique pairs',len(frames),'edges',w.variable('DLPNO LMP2 RESPONSE EDGES'),
          'rectangular',w.variable('DLPNO LMP2 RECTANGULAR RESPONSE EDGES'),'residual',rms,
          'AO pair tangent error',max(errors),flush=True)
    assert w.variable('DLPNO LMP2 RECTANGULAR RESPONSE EDGES') > 0
    assert rms < 1.e-12 and max(errors) < 2.e-8


if __name__ == '__main__':
    psi4.set_num_threads(1)
    psi4.set_options(dict(basis='cc-pvdz',reference='rhf',scf_type='df',df_basis_cc='cc-pvdz-ri',
                          df_basis_scf='cc-pvdz-jkfit',e_convergence=1.e-12,d_convergence=1.e-12,
                          r_convergence=1.e-10,solver_convergence=1.e-11,dlpno_maxiter=600,
                          local_convergence=1.e-14,local_gradient_convergence=1.e-8,
                          dlpno_keep_lmp2_response=True, dlpno_gradient_model=True,
                          dlpno_local_orbitals='BOYS',t_cut_pno_mp2=1.e-5))
    psi4.set_options({k:0. for k in ('t_cut_do t_cut_do_pre t_cut_do_ij t_cut_do_uv t_cut_mkn '
                                      't_cut_pairs t_cut_pairs_mp2 t_cut_pre t_cut_clmo t_cut_cpao '
                                      'dlpno_ao_ints_tol').split()})
    for frozen in (True, False):
        check(frozen)
