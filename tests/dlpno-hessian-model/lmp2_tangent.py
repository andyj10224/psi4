"""Water first-space LMP2 amplitude derivatives, checked against reconverged pair tensors."""
import numpy as np
import psi4
from scipy.optimize import linear_sum_assignment
from scipy.sparse.linalg import LinearOperator, gmres
from psi4.driver.procrouting.response.dlpno_orbital import (
    boys_nuclear_tangent, first_pno_tangent, coupled_lmp2_tangent, pair_exchange_tangent, final_pno_tangent,
    ccsd_pair_frames, ccsd_projected_fock_source, ccsd_direct_df_source, ccsd_pair_df_factors, PairDFFactors)
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
    domains = blocks(w, 'OCC_DOMAIN')
    foo = L.T @ F @ L
    half = dL.T @ F @ L
    dfoo = half + half.T + L.T @ df @ L
    first = {}
    for (i,j),y in frames.items():
        k,dk = exchange((i,j),V,dV)
        first[i,j] = first_pno_tangent(V,dV,S,ds,F,df,k,dk,foo[i,i]+foo[j,j],
                                       dfoo[i,i]+dfoo[j,j],np.eye(V.shape[1]),y.shape[1],i==j)
    out = coupled_lmp2_tangent(w,first,lambda ij,y,dy: exchange(ij,y,dy)[1],ds,df,dfoo)
    final = final_pno_tangent(w,out)
    ccsd_frames = ccsd_pair_frames(w,final,ds)
    b1,b2 = ccsd_projected_fock_source(w,ccsd_frames,dL,df)
    edge = max(((p,q) for p in frames for q in frames if p != q and set(p) & set(q)),
               key=lambda pq: abs(ccsd_frames.frames[pq[0]].shape[1]-ccsd_frames.frames[pq[1]].shape[1]))
    left_pair,right_pair = edge
    # One off-diagonal target exercises the two direct DF doubles terms.
    target = next(ij for ij in frames if ij[0]!=ij[1])
    dom,ts,state = psi4.core.dlpno_ccsd_df_pair_state(w,*target)
    aux_idx = state['AUX'].np[:,0].astype(int)
    Js, dJs = J[np.ix_(aux_idx,aux_idx)], dJ[np.ix_(aux_idx,aux_idx)]
    target_t = state['T2'].np.copy()
    def projected(left,dleft,right,dright):
        value = left.T @ Q[aux_idx] @ right
        derivative = (dleft.T @ Q[aux_idx] @ right + left.T @ dQ[aux_idx] @ right +
                      left.T @ Q[aux_idx] @ dright)
        return value,derivative
    occ, docc = L[:,dom], dL[:,dom]
    z, dz = ccsd_frames.frames[target], ccsd_frames.tangents[target]
    blocks0,blocks1 = zip(projected(occ,docc,occ,docc),projected(occ,docc,z,dz),projected(z,dz,z,dz))
    b,db = ccsd_pair_df_factors(Js,dJs,PairDFFactors(*blocks0),PairDFFactors(*blocks1))
    df_direct = ccsd_direct_df_source(w,target,ccsd_frames,b,db)
    analytic = np.concatenate([ccsd_frames.overlap_tangent(left_pair,right_pair).ravel()] +
                              [block.ravel() for block in b1] + [b2[ij].ravel() for ij in frames] +
                              [df_direct.ravel()])
    final_frames = blocks(w,'FINAL_FRAME')
    # Rotate each parent frame internally and compensate the amplitude derivative.
    shifted = type(out)({}, {})
    for ij,y in frames.items():
        rot = np.zeros((y.shape[1],)*2)
        if len(rot) > 1:
            rot[1,0], rot[0,1] = .23, -.23
        shifted.frames[ij] = out.frames[ij] + y @ rot
        t = amplitudes[ij]
        shifted.amplitudes[ij] = out.amplitudes[ij] + t @ rot - rot @ t
    rotated = final_pno_tangent(w, shifted)
    for ij in frames:
        np.testing.assert_allclose(rotated[ij].retained_density_tangent,
                                   final[ij].retained_density_tangent, atol=2.e-8, rtol=0)
    truncated = [ij for ij in frames if final_frames[ij].shape[1] < frames[ij].shape[1]]
    assert truncated
    for ij, selection in final.items():
        p, dp = selection.retained_density, selection.retained_density_tangent
        np.testing.assert_allclose(dp @ S @ p + p @ ds @ p + p @ S @ dp, dp, atol=2.e-8, rtol=0)
        # The spectral projector and density commute in the moving parent.
        y, dy, t, dt = frames[ij], out.frames[ij], amplitudes[ij], out.amplitudes[ij]
        tt, dtt = 2*t-t.T, 2*dt-dt.T
        den = tt @ t.T + tt.T @ t
        dden = dtt @ t.T + tt @ dt.T + dtt.T @ t + tt.T @ dt
        proj = y.T @ S @ p @ S @ y
        half = dy @ proj @ y.T
        dproj = y.T @ S @ (dp-half-half.T) @ S @ y
        np.testing.assert_allclose(dden @ proj + den @ dproj, dproj @ den + proj @ dden,
                                   atol=2.e-10, rtol=0)
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
        flat, info = gmres(op,rhs,rtol=3.e-14,atol=0.)
        assert info == 0 and np.max(abs(apply(flat)-rhs)) < 1.e-11
        physical, final_densities, ov, vv, aligned = {}, {}, {}, {}, {}
        for (i,j),t in unpack(flat).items():
            parent = y[i,j]
            physical[i,j] = parent @ t @ parent.T
            tt = 2*t - t.T
            density = (tt @ t.T + tt.T @ t)/(1+(i==j))
            _, u = np.linalg.eigh(density)
            keep = parent @ u[:,-final_frames[i,j].shape[1]:]
            final_densities[i,j] = keep @ keep.T
            # Positive polar overlap fixes parallel transport to first order at the origin.
            a,_,bt = np.linalg.svd(keep.T @ S @ final_frames[i,j])
            z = keep @ (a @ bt)
            aligned[i,j] = z
            ov[i,j] = psi4.core.Matrix.from_array(l[:,domains[i,j][:,0].astype(int)].T @ scf.Fa().np @ z)
            vv[i,j] = psi4.core.Matrix.from_array(z.T @ scf.Fa().np @ z)
        val1,val2 = psi4.core.dlpno_ccsd_fock_source(w,psi4.core.Matrix.from_array(f),ov,vv)
        # Independent forward evaluation, including the instantaneous auxiliary metric.
        vals,U = np.linalg.eigh(metric[np.ix_(aux_idx,aux_idx)])
        W = (U/np.sqrt(vals)) @ U.T
        bq = (W @ q[aux_idx].reshape(len(vals),-1)).reshape(len(vals),q.shape[1],q.shape[2])
        lm, z = l[:,dom], aligned[target]
        oo, ovf, vvf = lm.T @ bq @ lm, lm.T @ bq @ z, z.T @ bq @ z
        tn = np.stack([(z.T @ scf.S().np @ aligned[m,m] @ s.np).ravel() for m,s in zip(dom,ts)])
        def via(m):
            n = dom.index(m)
            return ovf[:,n,:] - (oo[:,n,:]+ovf@tn[n]) @ tn + vvf @ tn[n]
        dressed = vvf - tn.T @ ovf
        direct = via(target[0]).T @ via(target[1]) + (dressed @ target_t @ dressed.transpose(0,2,1)).sum(axis=0)
        samples = np.concatenate([(aligned[left_pair].T @ scf.S().np @ aligned[right_pair]).ravel()]
                                 + [m.np.ravel() for m in val1] + [val2[ij].np.ravel() for ij in frames]
                                 + [direct.ravel()])
        return physical, final_densities, samples
    at_origin, final_origin, _ = sample(BASE)
    for ij,y in frames.items():
        np.testing.assert_allclose(at_origin[ij],y @ amplitudes[ij] @ y.T,atol=2.e-8,rtol=0)
        np.testing.assert_allclose(final_origin[ij],final[ij].retained_density,atol=2.e-8,rtol=0)
    estimates, final_estimates, fock_estimates = [], [], []
    for h in (.001,.0005,.00025,.000125):
        plus, final_plus, plus_source = sample(BASE+h*STRETCH)
        minus, final_minus, minus_source = sample(BASE-h*STRETCH)
        fock_estimates.append((plus_source-minus_source)/(2*h))
        final_estimates.append({ij:(final_plus[ij]-final_minus[ij])/(2*h) for ij in frames})
        estimates.append({ij:(plus[ij]-minus[ij])/(2*h) for ij in frames})
    errors = [np.max(abs(dphysical[ij]-(4*estimates[-1][ij]-estimates[-2][ij])/3)) for ij in frames]
    discrepancy = np.abs(analytic-(4*fock_estimates[-1]-fock_estimates[-2])/3)
    nedge = ccsd_frames.overlap_tangent(left_pair,right_pair).size
    nsingles = sum(x.size for x in b1)
    print('projected CCSD errors',frozen, [part.max() for part in
          np.split(discrepancy,[nedge,nedge+nsingles,analytic.size-df_direct.size])],flush=True)
    np.testing.assert_allclose(discrepancy,0.,atol=3.e-8)
    final_errors = [np.max(abs(final[ij].retained_density_tangent -
                               (4*final_estimates[-1][ij]-final_estimates[-2][ij])/3)) for ij in frames]
    print('final PNO', frozen, 'truncated',len(truncated),'of',len(frames),'smallest gap',
          min(final[ij].boundary_gap for ij in truncated),'max error',max(final_errors),flush=True)
    for k in range(1,len(final_estimates)):
        errs = {ij:np.max(abs(final[ij].retained_density_tangent -
                              (4*final_estimates[k][ij]-final_estimates[k-1][ij])/3)) for ij in frames}
        print('extrapolation',k,'worst',max(errs,key=errs.get),max(errs.values()),flush=True)
    scaled = [err/max(1.,np.max(abs(final[ij].retained_density_tangent)))
              for ij,err in zip(frames, final_errors)]
    print('maximum scaled final PNO error',max(scaled),flush=True)
    assert max(scaled) < 2.e-7 and max(final_errors) < 1.e-6
    rms = w.variable('DLPNO LMP2 NUCLEAR TANGENT RESIDUAL')
    print('frozen',frozen,'unique pairs',len(frames),'edges',w.variable('DLPNO LMP2 RESPONSE EDGES'),
          'rectangular',w.variable('DLPNO LMP2 RECTANGULAR RESPONSE EDGES'),'residual',rms,
          'AO pair tangent error',max(errors),flush=True)
    assert w.variable('DLPNO LMP2 RECTANGULAR RESPONSE EDGES') > 0
    assert rms < 1.e-12 and max(errors) < 2.e-8


if __name__ == '__main__':
    psi4.set_num_threads(1)
    psi4.set_options(dict(basis='cc-pvdz',reference='rhf',scf_type='df',df_basis_cc='cc-pvdz-ri',
                          df_basis_scf='cc-pvdz-jkfit',e_convergence=1.e-13,d_convergence=1.e-13,
                          r_convergence=1.e-11,solver_convergence=1.e-12,dlpno_maxiter=600,
                          local_convergence=1.e-14,local_gradient_convergence=1.e-8,
                          dlpno_keep_lmp2_response=True, dlpno_gradient_model=True,
                          dlpno_local_orbitals='BOYS',t_cut_pno_mp2=1.e-5,t_cut_pno=1.e-4))
    psi4.set_options({k:0. for k in ('t_cut_do t_cut_do_pre t_cut_do_ij t_cut_do_uv t_cut_mkn '
                                      't_cut_pairs t_cut_pairs_mp2 t_cut_pre t_cut_clmo t_cut_cpao '
                                      'dlpno_ao_ints_tol').split()})
    for frozen in (True, False):
        check(frozen)
