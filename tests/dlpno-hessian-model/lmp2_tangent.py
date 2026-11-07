"""Water first-space LMP2 amplitude derivatives, checked against reconverged pair tensors."""
from functools import lru_cache
import numpy as np
import psi4
from scipy.optimize import linear_sum_assignment
from scipy.sparse.linalg import LinearOperator, gmres
from psi4.driver.procrouting.response.dlpno_orbital import (
    boys_nuclear_tangent, first_pno_tangent, coupled_lmp2_tangent, pair_exchange_tangent, final_pno_tangent,
    ccsd_pair_frames, ccsd_projected_fock_source, ccsd_direct_df_source, ccsd_pair_df_factors, PairDFFactors)
from psi4.driver.procrouting.response.dlpno_ccsd_response import (
    ccsd_pair_intermediate_tangent, ccsd_beta_source, ccsd_mixed_doubles_source, ccsd_singles_source, ccsd_doubles_fock_source, ccsd_residual_source, coupled_ccsd_tangent)
from first_pno import raw, relax_fock
from water_hvp import BASE, STRETCH, molecule


def calculation(coords):
    _, w = psi4.energy('dlpno-ccsd(t)', molecule=molecule(coords), return_wfn=True)
    return w


def blocks(w, name):
    return {ij: m.np.copy() for ij,m in psi4.core.dlpno_lmp2_blocks(w,name).items()}


def check_ccsd_tangent(w, source1, source2, frozen, dL, frames):
    """Check the weighted transpose and transport reconverged displaced amplitudes."""
    matrix = psi4.core.Matrix.from_array
    rng = np.random.default_rng(122)
    def probe():
        one = [rng.normal(size=x.shape) for x in source1]
        two = {p: rng.normal(size=x.shape) for p,x in source2.items()}
        for (i,j),x in two.items():
            if i == j:
                two[i,j] = (x + x.T)/2
        return one,two
    def action(x, transpose=False):
        a,b = psi4.core.dlpno_ccsd_amplitude_action(
            w, [matrix(v[:,None]) for v in x[0]], {p: matrix(v) for p,v in x[1].items()}, transpose)
        return [v.np.ravel() for v in a], {p: v.np for p,v in b.items()}
    def dot(x,y):
        return (sum(np.vdot(a,b) for a,b in zip(x[0],y[0])) +
                sum((.5 if i == j else 1.)*np.vdot(v,y[1][i,j]) for (i,j),v in x[1].items()))
    x,z = probe(),probe()
    lhs,rhs = dot(z,action(x)),dot(action(z,True),x)
    print('CCSD weighted transpose', frozen, lhs, rhs, abs(lhs-rhs), flush=True)
    np.testing.assert_allclose(lhs,rhs,rtol=2.e-12,atol=2.e-11)
    tangent = coupled_ccsd_tangent(w, source1, source2)
    jt = action(tangent)
    for j,b in zip(jt[0],source1):
        np.testing.assert_allclose(j,-b,rtol=0.,atol=1.e-10)
    for p,j in jt[1].items():
        np.testing.assert_allclose(j,-source2[p],rtol=0.,atol=1.e-10)

    original = blocks(w, 'FINAL_FRAME')
    parent = blocks(w,'FRAME')
    ref = w.reference_wavefunction()
    L = w.variable('DLPNO RESPONSE ACTIVE LMOS').np
    def displaced(h):
        other = calculation(BASE+h*STRETCH)
        scf = other.reference_wavefunction()
        moved = other.variable('DLPNO RESPONSE ACTIVE LMOS').np
        cross = psi4.core.MintsHelper(scf).ao_overlap(ref.basisset(),scf.basisset()).np
        ov = L.T @ cross @ moved
        _,order = linear_sum_assignment(-abs(ov))
        signs = np.sign(ov[np.arange(len(order)),order])
        local = moved[:,order]*signs
        foo0 = L.T @ ref.Fa().np @ L
        fooh = local.T @ scf.Fa().np @ local
        cut = psi4.core.get_option('DLPNO','F_CUT')
        np.testing.assert_array_equal(abs(foo0) > cut, abs(fooh) > cut)
        assert min(abs(ov[np.arange(len(order)),order])) > .999
        virtual = blocks(other,'FINAL_FRAME')
        doubles = blocks(other,'CCSD_DOUBLES')
        par = blocks(other,'FRAME')
        assert len(virtual) == len(original)
        one,two = {},{}
        dy = []
        for (i,j),y in original.items():
            a,b = order[i],order[j]
            p = tuple(sorted((int(a),int(b))))
            assert virtual[p].shape == y.shape
            assert par[p].shape == parent[i,j].shape, (h, (i,j), 'first PNO rank changed')
            left,_,right = np.linalg.svd(virtual[p].T @ ref.S().np @ y)
            rotation = left @ right
            dy.append((virtual[p] @ rotation).ravel())
            t = doubles[p] if a <= b else doubles[p].T
            two[i,j] = signs[i]*signs[j]*(rotation.T @ t @ rotation)
            if i == j:
                dom,t1,_ = psi4.core.dlpno_ccsd_df_pair_state(other,int(a),int(a))
                one[i] = signs[i]*(rotation.T @ t1[dom.index(a)].np).ravel()
        return (np.concatenate([one[i] for i in range(len(one))]),
                np.concatenate([two[p].ravel() for p in original]), local.ravel(), np.concatenate(dy))
    expected = (np.concatenate(tangent[0]), np.concatenate([tangent[1][p].ravel() for p in original]),
                dL.ravel(), np.concatenate([frames.tangents[p].ravel() for p in original]))
    estimates = []
    for h in (.002,.001,.0005):
        plus,minus = displaced(h),displaced(-h)
        estimates.append([(a-b)/(2*h) for a,b in zip(plus,minus)])
    for coarse,fine in zip(estimates,estimates[1:]):
        errors = [np.max(abs(v-(4*f-c)/3)) for v,c,f in zip(expected,coarse,fine)]
        print('CCSD reconverged errors (T1, T2, LMO, PNO)', frozen, errors, flush=True)
        assert max(errors[:2]) < 1.e-6
        assert errors[2] < 1.e-5 and errors[3] < 5.e-5


def check(frozen, *, amplitude_response=False):
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
    doubles = blocks(w,'CCSD_DOUBLES')
    def oriented(tensors,i,j):
        key = tuple(sorted((i,j)))
        return tensors[key] if i<=j else tensors[key].T
    exchange_blocks = {ij:exchange(ij,ccsd_frames.frames[ij],ccsd_frames.tangents[ij]) for ij in frames}
    def ordered_exchange(i,j):
        k,dk = exchange_blocks[tuple(sorted((i,j)))]
        return (k,dk) if i<=j else (k.T,dk.T)
    @lru_cache(None)
    def factors_for(i,j):
        pair = i,j
        dl,_,st = psi4.core.dlpno_ccsd_df_pair_state(w,*pair)
        assert np.array_equal(st['AUX'].np[:,0],aux_idx)
        lm,dm = L[:,dl], dL[:,dl]
        y,dy = (tab[tuple(sorted(pair))] for tab in (ccsd_frames.frames,ccsd_frames.tangents))
        vals,dirs = zip(projected(lm,dm,lm,dm),projected(lm,dm,y,dy),projected(y,dy,y,dy))
        return ccsd_pair_df_factors(Js,dJs,PairDFFactors(*vals),PairDFFactors(*dirs))
    @lru_cache(None)
    def interm(i,j):
        pair = i,j
        fac,dfac = factors_for(*pair)
        return ccsd_pair_intermediate_tangent(w,pair,ccsd_frames,fac,dfac,doubles,ordered_exchange)
    def mixed_derivative(i,j,k):
        yp,dyp = (tab[tuple(sorted((i,j)))] for tab in (ccsd_frames.frames,ccsd_frames.tangents))
        yq,dyq = (tab[tuple(sorted((k,j)))] for tab in (ccsd_frames.frames,ccsd_frames.tangents))
        occ = lambda m: (L[:,m:m+1], dL[:,m:m+1])
        def packed(*args):
            return [t.reshape(len(Js),-1) for t in projected(*args)]
        _,dj = pair_exchange_tangent(Js,dJs,*packed(*occ(i),*occ(k)),*packed(yp,dyp,yq,dyq))
        _,dk = pair_exchange_tangent(Js,dJs,*packed(*occ(i),yp,dyp),*packed(*occ(k),yq,dyq))
        return dj.reshape(yp.shape[1],yq.shape[1]),dk
    intermediates = interm(*target)
    beta_source = ccsd_beta_source(w,target,ccsd_frames,intermediates.beta,doubles)
    cd_source = ccsd_mixed_doubles_source(w,target,ccsd_frames,interm,doubles,mixed_derivative)
    singles_source = ccsd_singles_source(w,ccsd_frames,factors_for,doubles,ordered_exchange,L,dL,F,df)
    eg = ccsd_doubles_fock_source(w,ccsd_frames,factors_for,doubles,ordered_exchange,L,dL,F,df)
    total1,total2 = ccsd_residual_source(w,ccsd_frames,factors_for,ordered_exchange,mixed_derivative,dL,df)
    if amplitude_response:
        check_ccsd_tangent(w, total1, total2, frozen, dL, ccsd_frames)
        return
    for got,expected in zip(total1,singles_source):
        np.testing.assert_allclose(got,expected,atol=1.e-12,rtol=0)
    np.testing.assert_allclose(total2[target],df_direct+beta_source+cd_source+eg[target],atol=1.e-12,rtol=0)
    analytic = np.concatenate([ccsd_frames.overlap_tangent(left_pair,right_pair).ravel()] +
                              [block.ravel() for block in b1] + [b2[ij].ravel() for ij in frames] +
                              [df_direct.ravel(),*(v.ravel() for v in intermediates),beta_source.ravel(),cd_source.ravel(),*singles_source,eg[target].ravel(),*(total2[ij].ravel() for ij in frames)])
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
        dom,ts,_ = psi4.core.dlpno_ccsd_df_pair_state(w,*target)
        lm, z = l[:,dom], aligned[target]
        oo, ovf, vvf = lm.T @ bq @ lm, lm.T @ bq @ z, z.T @ bq @ z
        tn = np.stack([(z.T @ scf.S().np @ aligned[m,m] @ s.np).ravel() for m,s in zip(dom,ts)])
        def via(m):
            n = dom.index(m)
            return ovf[:,n,:] - (oo[:,n,:]+ovf@tn[n]) @ tn + vvf @ tn[n]
        dressed = vvf - tn.T @ ovf
        direct = via(target[0]).T @ via(target[1]) + (dressed @ target_t @ dressed.transpose(0,2,1)).sum(axis=0)
        def sij(p,q):
            return aligned[tuple(sorted(p))].T @ scf.S().np @ aligned[tuple(sorted(q))]
        @lru_cache(None)
        def forward_intermediates(pair):
            dom,ts,_ = psi4.core.dlpno_ccsd_df_pair_state(w,*pair)
            lm,z = l[:,dom],aligned[tuple(sorted(pair))]
            oo,ovf,vvf = lm.T @ bq @ lm,lm.T @ bq @ z,z.T @ bq @ z
            tn = np.stack([(sij(pair,(m,m)) @ t.np).ravel() for m,t in zip(dom,ts)])
            i,j = pair
            ki, kj = dom.index(i),dom.index(j)
            oi = oo[:,ki,:] + ovf @ tn[ki]
            oj = oo[:,kj,:] + ovf @ tn[kj]
            beta = oi.T @ oj + np.einsum('Qkc,cd,Qld->kl',ovf,oriented(doubles,i,j),ovf)
            coul = np.einsum('Q,Qlc->lc',oo[:,ki,kj],ovf)
            exch = oo[:,ki,:].T @ ovf[:,kj,:]
            ivvv = np.einsum('Qb,Qac->bac',ovf[:,ki,:],vvf)
            jvvv = np.einsum('Qb,Qac->bac',ovf[:,kj,:],vvf)
            gamma = -tn.T @ coul + np.einsum('b,bac->ac',tn[kj],ivvv)
            delta = -tn.T @ (2*exch-coul) + np.einsum('b,cab->ac',tn[ki],2*jvvv)-np.einsum('b,bca->ac',tn[ki],jvvv)
            Sm = scf.S().np
            def overlap(p,q):
                return aligned[tuple(sorted(p))].T @ Sm @ aligned[tuple(sorted(q))]
            def t1(m,p):
                return (overlap(p,(m,m)) @ ts[dom.index(m)].np).ravel()
            def pair_k(p):
                m,n = p
                yp = aligned[tuple(sorted(p))]
                return (l[:,m] @ bq @ yp).T @ (l[:,n] @ bq @ yp)
            beta_part = np.zeros_like(oriented(doubles,i,j))
            for m in dom:
                im,mj = (i,m),(m,j)
                K = pair_k(im)
                vec = overlap(pair,im) @ K.T @ t1(j,im)
                gamma -= np.outer(tn[dom.index(m)],vec)
                gamma -= .5*overlap(pair,mj) @ oriented(doubles,m,j) @ overlap(mj,im) @ K @ overlap(im,pair)
                K = pair_k(mj)
                ex = 2*K - K.T
                vec = t1(i,mj) @ ex @ overlap(mj,pair)
                delta -= np.outer(tn[dom.index(m)],vec)
                t = oriented(doubles,i,m)
                delta += .5*overlap(pair,im) @ (2*t-t.T) @ overlap(im,mj) @ ex @ overlap(mj,pair)
                for n in dom:
                    if tuple(sorted((m,n))) not in doubles:
                        continue
                    ovmn = overlap(pair,(m,n))
                    beta_part += beta[dom.index(m),dom.index(n)] * (ovmn @ oriented(doubles,m,n) @ ovmn.T)
            return beta,gamma,delta,beta_part
        beta,gamma,delta,beta_part = forward_intermediates(target)
        def cd_order(i,j):
            dom,_,_ = psi4.core.dlpno_ccsd_df_pair_state(w,i,j)
            pair = i,j
            yp = aligned[tuple(sorted(pair))]
            c = np.zeros((yp.shape[1],)*2)
            d = np.zeros_like(c)
            for k in dom:
                kj,ki,ik,jk = (k,j),(k,i),(i,k),(j,k)
                yq = aligned[tuple(sorted(kj))]
                Jik = np.einsum('Q,Qac->ac',l[:,i] @ bq @ l[:,k],yp.T @ bq @ yq)
                Kik = (l[:,i] @ bq @ yp).T @ (l[:,k] @ bq @ yq)
                g = sij(pair,ki) @ forward_intermediates(ki)[1] @ sij(ki,kj)
                c -= (Jik+g) @ oriented(doubles,*kj).T @ sij(kj,pair)
                g = sij(pair,ik) @ forward_intermediates(ik)[2] @ sij(ik,jk)
                t = oriented(doubles,*jk)
                d += .5*(2*Kik-Jik+g) @ (2*t-t.T).T @ sij(jk,pair)
            return .5*c+c.T+d
        cd = cd_order(*target)+cd_order(*target[::-1]).T
        # Independent forward singles residual. All amplitudes are held at the reference.
        @lru_cache(None)
        def inputs(i,k):
            dom,singles,_ = psi4.core.dlpno_ccsd_df_pair_state(w,i,k)
            y = aligned[tuple(sorted((i,k)))]
            lm = l[:,dom]
            oo,ov,vv = lm.T @ bq @ lm,lm.T @ bq @ y,y.T @ bq @ y
            tn = np.stack([(sij((i,k),(m,m)) @ t.np).ravel() for m,t in zip(dom,singles)])
            return dom,singles,oo,ov,vv,tn
        @lru_cache(None)
        def K(i,k):
            y = aligned[tuple(sorted((i,k)))]
            return (l[:,i] @ bq @ y).T @ (l[:,k] @ bq @ y)
        fbar = f.copy()
        orders = [p for i,k in frames for p in (((i,k),) if i==k else ((i,k),(k,i)))]
        for i,k in orders:
            dom,_,oo,ov,vv,tn = inputs(i,k)
            a,b = dom.index(i),dom.index(k)
            Jik = np.einsum('Q,Qlc->lc',oo[:,a,b],ov)
            fbar[i,k] += np.sum(tn*(2*Jik - oo[:,b,:].T @ ov[:,a,:]))
        residual = [np.zeros(aligned[i,i].shape[1]) for i in range(l.shape[1])]
        for i,k in orders:
            pair = i,k
            dom,singles,oo,ov,vv,tn = inputs(*pair)
            y = aligned[tuple(sorted(pair))]
            a,b = dom.index(i),dom.index(k)
            if i==k:
                coeff = np.einsum('Qme,me->Q',ov,tn)
                ai = l[:,i] @ scf.Fa().np @ y + 2*coeff @ ov[:,a,:] - np.einsum('Qae,me,Qm->a',vv,tn,oo[:,a,:])
                ab = y.T @ scf.Fa().np @ y + np.einsum('Q,Qab->ab',2*coeff,vv) - np.einsum('Qae,me,Qmb->ab',vv,tn,ov)
                mb = l[:,dom].T @ scf.Fa().np @ y + np.einsum('Q,Qmb->mb',2*coeff,ov) - np.einsum('Qme,le,Qlb->mb',ov,tn,ov)
                ti = singles[a].np.ravel()
                residual[i] += ai + ab @ ti - tn.T @ (mb @ ti + fbar[dom,i])
            t = oriented(doubles,i,k)
            u = 2*t-t.T
            tmp = np.einsum('Qc,Qda,cd->a',ov[:,b,:],vv,u.T)
            residual[i] += sij(pair,(i,i)).T @ tmp
            fk = l[:,k] @ scf.Fa().np @ y
            for m,t1 in zip(dom,singles):
                km = k,m
                s = sij(km,pair)
                residual[i] -= np.sum((s @ u.T @ s.T)*K(*km))*(sij((i,i),(m,m)) @ t1.np).ravel()
                fk += sij(pair,km) @ (2*K(*km)-K(*km).T) @ (sij(km,(m,m)) @ t1.np).ravel()
            residual[i] += sij(pair,(i,i)).T @ u @ fk
            B = (oo[:,a,:].T @ ov[:,b,:] + tn @ K(i,k)) @ u.T
            for row,m in enumerate(dom):
                residual[m] -= B[row] @ sij(pair,(m,m))
        if np.array_equal(coords, BASE):
            assert max(np.max(abs(r)) for r in residual) < 1.e-8
        # E and G from the original doubly dressed Fock intermediates.
        fkj = fbar.copy()
        for i,j in orders:
            dom0,ts0,_,hh,_,tn0 = inputs(j,j)
            gq = np.einsum('Qme,me->Q',hh,tn0)
            kk = dom0.index(i)
            hbar = l[:,i] @ scf.Fa().np @ aligned[j,j] + 2*gq @ hh[:,kk,:]
            hbar -= np.einsum('Qe,me,Qmc->c',hh[:,kk,:],tn0,hh)
            fkj[i,j] += hbar @ ts0[dom0.index(j)].np.ravel()
            for m in inputs(i,j)[0]:
                sop = sij((m,i),(m,j))
                t = oriented(doubles,m,j)
                fkj[i,j] += np.sum(K(m,i)*(sop @ (2*t-t.T) @ sop.T))
        def eg_value(pair):
            i,j = pair
            dom,ts,_,hh,vv,tn = inputs(*pair)
            gq = np.einsum('Qme,me->Q',hh,tn)
            y = aligned[pair]
            ff = y.T @ scf.Fa().np @ y + np.einsum('Q,Qab->ab',2*gq,vv)
            ff -= np.einsum('Qae,me,Qmb->ab',vv,tn,hh)
            hhbar = l[:,dom].T @ scf.Fa().np @ y + np.einsum('Q,Qma->ma',2*gq,hh)
            hhbar -= np.einsum('Qme,le,Qlb->mb',hh,tn,hh)
            ff -= tn.T @ hhbar
            for k in dom:
                for m in dom:
                    if tuple(sorted((k,m))) not in doubles:
                        continue
                    s = sij(pair,(k,m))
                    t = oriented(doubles,k,m)
                    ff -= s @ (2*t-t.T) @ K(k,m).T @ s.T
            eg_forward = oriented(doubles,i,j) @ ff.T + ff @ oriented(doubles,i,j)
            for perm,(i,j) in enumerate((pair,pair[::-1])):
                for k in domains[i,i].astype(int).ravel():
                    s = sij(pair,(i,k))
                    term = fkj[k,j]*(s @ oriented(doubles,i,k) @ s.T)
                    eg_forward -= term.T if perm else term
            return eg_forward
        eg_forward = eg_value(target)
        complete = {}
        for pair in frames:
            i,j = pair
            locdom,_,qo,qh,qv,proj_singles = inputs(i,j)
            def dressed_virtual(m):
                n = locdom.index(m)
                return qh[:,n,:] - (qo[:,n,:]+qh @ proj_singles[n]) @ proj_singles + qv @ proj_singles[n]
            vv_t = qv - proj_singles.T @ qh
            t = doubles[pair]
            driving = (dressed_virtual(i).T @ dressed_virtual(j) +
                       (vv_t @ t @ vv_t.transpose(0,2,1)).sum(axis=0))
            complete[pair] = (driving + forward_intermediates(pair)[3] +
                              cd_order(*pair) + cd_order(*pair[::-1]).T + eg_value(pair))
        if np.array_equal(coords, BASE):
            assert max(np.max(abs(v)) for v in complete.values()) < 1.e-8
        samples = np.concatenate([(aligned[left_pair].T @ scf.S().np @ aligned[right_pair]).ravel()]
                                 + [m.np.ravel() for m in val1] + [val2[ij].np.ravel() for ij in frames]
                                 + [direct.ravel(),beta.ravel(),gamma.ravel(),delta.ravel(),beta_part.ravel(),cd.ravel(),*residual,eg_forward.ravel(),*(complete[ij].ravel() for ij in frames)])
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
    sizes = [nedge, nsingles, sum(x.size for x in b2.values()), df_direct.size,
             *(v.size for v in intermediates), beta_source.size, cd_source.size, nsingles, eg[target].size, sum(t.size for t in total2.values())]
    print('projected CCSD errors',frozen,[part.max() for part in
          np.split(discrepancy,np.cumsum(sizes[:-1]))],flush=True)
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
    # The trace/energy rank selectors give different first-space ranks at
    # displaced geometries on the original fixture. Check the native solve
    # on an occupation-selected branch, asserting both ranks at each point.
    psi4.set_options(dict(t_cut_trace_mp2=0., t_cut_energy_mp2=0., t_cut_trace=0., t_cut_energy=0.,
                          t_cut_pairs=1.e-9, t_cut_pairs_mp2=1.e-9))
    for frozen in (True, False):
        check(frozen, amplitude_response=True)
