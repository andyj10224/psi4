"""Fixed-amplitude nuclear sources for the RHF t1-transformed CCSD residual."""
from functools import lru_cache
from typing import NamedTuple

import numpy as np

from .dlpno_orbital import ccsd_direct_df_source


def _contract_direction(subscripts, *terms):
    """Value and product-rule derivative of a pair-local tensor contraction."""
    values, tangents = zip(*terms)
    primal = np.einsum(subscripts, *values, optimize=True)
    derivative = np.zeros_like(primal)
    for k, tangent in enumerate(tangents):
        if tangent is not None:
            derivative += np.einsum(subscripts, *values[:k], tangent, *values[k+1:], optimize=True)
    return primal, derivative


class CCSDPairIntermediates(NamedTuple):
    beta: np.ndarray
    gamma: np.ndarray
    delta: np.ndarray


def ccsd_pair_intermediate_tangent(wfn, ij, frames, factors, tangent, doubles, exchange):
    """Partial nuclear derivatives of ordered-pair beta, gamma and delta (Jiang 82--84).

    doubles holds native unique CCSD T2 blocks. exchange(i,j) supplies K_ij
    and its derivative in that pair's own frame and fitting domain. factors
    belong to the target pair; occupied axes follow its native domain order.
    Amplitudes stay fixed in their own moving frames.
    """
    from psi4 import core
    i,j = ij
    dom,singles,state = core.dlpno_ccsd_df_pair_state(wfn,i,j)
    tn = state['TN'].np
    dtn = np.stack([(frames.overlap_tangent(ij,(m,m)) @ s.np).ravel() for m,s in zip(dom,singles)])
    h,dh = factors.ov,tangent.ov
    f,df = factors.vv,tangent.vv
    pos = {m:k for k,m in enumerate(dom)}
    a,b = pos[i],pos[j]

    def overlap(p,q):
        y = frames.frames[tuple(sorted(p))]
        z = frames.frames[tuple(sorted(q))]
        return y.T @ frames.overlap @ z, frames.overlap_tangent(p,q)

    def amp(p, spin=False):
        t = doubles[tuple(sorted(p))]
        if p[0] > p[1]:
            t = t.T
        return (2*t-t.T if spin else t), None

    def sing(m,p):
        return _contract_direction('ab,b->a',overlap(p,(m,m)),(singles[pos[m]].np.ravel(),None))

    def oi(m,side):
        k = pos[m]
        o = state[side].np
        if not np.allclose(o,factors.oo[:,k,:]+h @ tn[k],rtol=1.e-8,atol=1.e-9):
            raise ValueError(f"Target-pair occupied DF dressing disagrees for {ij}.")
        return o,tangent.oo[:,k,:] + dh @ tn[k] + h @ dtn[k]

    def add(dest, term, scale=1.):
        for target, value in zip(dest,term):
            target += scale*value

    beta = _contract_direction('Qk,Ql->kl',oi(i,'OI'),oi(j,'OJ'))
    add(beta,_contract_direction('Qkc,cd,Qld->kl',(h,dh),amp(ij),(h,dh)))

    # Native OOOV and OVVV blocks in the target pair's own fit.
    oo = factors.oo
    doo = tangent.oo
    coul = _contract_direction('Q,Qlc->lc',(oo[:,a,b],doo[:,a,b]),(h,dh))
    exch = _contract_direction('Ql,Qc->lc',(oo[:,a,:],doo[:,a,:]),(h[:,b,:],dh[:,b,:]))
    ivvv = _contract_direction('Qb,Qac->bac',(h[:,a,:],dh[:,a,:]),(f,df))
    jvvv = _contract_direction('Qb,Qac->bac',(h[:,b,:],dh[:,b,:]),(f,df))
    gamma = _contract_direction('la,lc->ac',(tn,dtn),coul)
    gamma = tuple(-v for v in gamma)
    add(gamma,_contract_direction('b,bac->ac',sing(j,ij),ivvv))
    delta = _contract_direction('la,lc->ac',(tn,dtn),coul)
    add(delta,_contract_direction('la,lc->ac',(tn,dtn),exch),-2.)
    add(delta,_contract_direction('b,cab->ac',sing(i,ij),jvvv),2.)
    add(delta,_contract_direction('b,bca->ac',sing(i,ij),jvvv),-1.)

    for l in dom:
        il, lj = (i,l),(l,j)
        kil = exchange(*il)
        klj = exchange(*lj)
        llj = tuple(2*k-k.T for k in klj)
        # Eqs. 83c,d: rectangular transfers follow the forward projection order.
        column = _contract_direction('ca,ba,b->c',overlap(ij,il),kil,sing(j,il))
        add(gamma,_contract_direction('a,c->ac',sing(l,ij),column),-1.)
        middle = _contract_direction('ad,de,ec->ac',amp(lj),overlap(lj,il),kil)
        add(gamma,_contract_direction('xa,ac,cy->xy',overlap(ij,lj),middle,overlap(il,ij)),-.5)
        # Eqs. 84c,d use spin-adapted doubles and exchange.
        column = _contract_direction('b,ba,ac->c',sing(i,lj),llj,overlap(lj,ij))
        add(delta,_contract_direction('a,c->ac',sing(l,ij),column),-1.)
        middle = _contract_direction('ad,de,ec->ac',amp(il,True),overlap(il,lj),llj)
        add(delta,_contract_direction('xa,ac,cy->xy',overlap(ij,il),middle,overlap(lj,ij)),.5)
    for key, (value, _) in zip(('BETA','GAMMA','DELTA'),(beta,gamma,delta)):
        if not np.allclose(value,state[key].np,rtol=1.e-8,atol=1.e-9):
            raise ValueError(f"Reconstructed CCSD {key} disagrees with the native ordered pair {ij}.")
    return CCSDPairIntermediates(beta[1],gamma[1],delta[1])


def ccsd_beta_source(wfn, ij, frames, dbeta, doubles):
    """Doubles source through beta and both explicit overlap legs (Jiang Eq. 77)."""
    from psi4 import core
    dom,_,state = core.dlpno_ccsd_df_pair_state(wfn,*ij)
    beta = state['BETA'].np
    yp = frames.frames[ij]
    result = np.zeros((yp.shape[1],)*2)
    for a,k in enumerate(dom):
        for b,l in enumerate(dom):
            key = tuple(sorted((k,l)))
            if key not in frames.frames:
                continue
            t = doubles[key] if k<=l else doubles[key].T
            s = yp.T @ frames.overlap @ frames.frames[key]
            ds = frames.overlap_tangent(ij,key)
            result += dbeta[a,b]*(s @ t @ s.T) + beta[a,b]*(ds @ t @ s.T + s @ t @ ds.T)
    return result


def ccsd_mixed_doubles_source(wfn, ij, frames, intermediate, doubles, mixed_integrals):
    """Jiang Eqs. 78--79, including the ordered residual permutation.

    intermediate(i,j) supplies the ordered beta/gamma/delta tangents.
    mixed_integrals(i,j,k) supplies dJ,dK on target ij's local fit,
    with virtual columns in the actual kj frame, as in the forward non-projected blocks.
    """
    from psi4 import core
    if ij != tuple(sorted(ij)):
        raise ValueError("Expected a unique CCSD doubles target.")

    def overlap(p,q):
        yp,yq = (frames.frames[tuple(sorted(pair))] for pair in (p,q))
        return yp.T @ frames.overlap @ yq, frames.overlap_tangent(p,q)

    def oriented(i,j):
        t = doubles[tuple(sorted((i,j)))]
        return t if i<=j else t.T

    def one_order(i,j):
        p = (i,j)
        dom,_,_ = core.dlpno_ccsd_df_pair_state(wfn,*p)
        n = frames.frames[tuple(sorted(p))].shape[1]
        c, d = np.zeros((n,n)), np.zeros((n,n))
        for k in dom:
            ki,kj,ik,jk = (k,i),(k,j),(i,k),(j,k)
            j0,k0,g0,d0 = core.dlpno_ccsd_mixed_state(wfn,i,j,k)
            dj,dk = mixed_integrals(i,j,k)
            J,K = (j0.np,dj), (k0.np,dk)
            gamma = (g0.np,intermediate(*ki).gamma)
            g = _contract_direction('xa,ac,cy->xy',overlap(p,ki),gamma,overlap(ki,kj))
            vertex = tuple(a+b for a,b in zip(J,g))
            t = oriented(*kj)
            c -= _contract_direction('ac,bc,by->ay',vertex,(t,None),overlap(kj,p))[1]

            delta = (d0.np,intermediate(*ik).delta)
            g = _contract_direction('xa,ac,cy->xy',overlap(p,ik),delta,overlap(ik,jk))
            vertex = tuple(2*a-b+c0 for a,b,c0 in zip(K,J,g))
            t = oriented(*jk)
            d += .5*_contract_direction('ac,bc,by->ay',vertex,(2*t-t.T,None),overlap(jk,p))[1]
        return .5*c+c.T+d

    i,j = ij
    forward = one_order(i,j)
    reverse = forward if i==j else one_order(j,i)
    return forward+reverse.T


def _ccsd_occupied_bar(wfn,frames,factors_for,lmo,dlmo,fock,dfock):
    from psi4 import core
    foo = lmo.T @ fock @ lmo
    half = dlmo.T @ fock @ lmo
    fij,dfij = foo.copy(), half + half.T + lmo.T @ dfock @ lmo
    for i,j in frames.frames:
        h,dh = factors_for(i,j)
        for a,b in (((i,j),) if i==j else ((i,j),(j,i))):
            dom,ts,state = core.dlpno_ccsd_df_pair_state(wfn,a,b)
            tn = state['TN'].np
            dtn = np.stack([(frames.overlap_tangent((a,b),(l,l)) @ t.np).ravel() for l,t in zip(dom,ts)])
            ma,mb = dom.index(a),dom.index(b)
            coul = _contract_direction('Q,Qlc->lc',(h.oo[:,ma,mb],dh.oo[:,ma,mb]),(h.ov,dh.ov))
            ex = _contract_direction('Ql,Qc->lc',(h.oo[:,mb,:],dh.oo[:,mb,:]),
                                     (h.ov[:,ma,:],dh.ov[:,ma,:]))
            corr,dcorr = _contract_direction('lc,lc->',(tn,dtn),tuple(2*x-y for x,y in zip(coul,ex)))
            fij[a,b] += corr
            dfij[a,b] += dcorr
    return fij,dfij


def _ccsd_partial_fock_blocks(factors, direction, tn, dtn, y, dy, lm, dlm, fock, dfock):
    h,dh = factors,direction
    gram = _contract_direction('Qmc,mc->Q',(h.ov,dh.ov),(tn,dtn))
    def bar(block,dblock,left,dleft):
        value = left.T @ fock @ y
        deriv = dleft.T @ fock @ y + left.T @ dfock @ y + left.T @ fock @ dy
        jj = _contract_direction('Q,Qrc->rc',gram,(block,dblock))
        kk = _contract_direction('Qre,me,Qmc->rc',(block,dblock),(tn,dtn),(h.ov,dh.ov))
        return value+2*jj[0]-kk[0], deriv+2*jj[1]-kk[1]
    return gram,bar(h.vv,dh.vv,y,dy),bar(h.ov,dh.ov,lm,dlm)


def ccsd_singles_source(wfn, frames, factors_for, doubles, exchange, lmo, dlmo, fock, dfock):
    """Full R1 source (Jiang 87--90 and dressed Fai), including the Fock terms.

    This already contains the singles part of ccsd_projected_fock_source.
    factors_for(i,j) gives pair-local factors and
    their tangents. exchange(i,j) uses the ordered pair's own fit.
    """
    from psi4 import core
    def ov(p,q):
        yp,yq = (frames.frames[tuple(sorted(pair))] for pair in (p,q))
        return yp.T @ frames.overlap @ yq, frames.overlap_tangent(p,q)

    def singles(p,m,t):
        return _contract_direction('ab,b->a',ov(p,(m,m)),(t.np.ravel(),None))

    sizes = [frames.frames[i,i].shape[1] for i in range(lmo.shape[1])]
    result = [np.zeros(n) for n in sizes]
    fij,dfij = _ccsd_occupied_bar(wfn,frames,factors_for,lmo,dlmo,fock,dfock)
    for p in frames.frames:
        h,dh = factors_for(*p)
        y,dy = frames.frames[p],frames.tangents[p]
        orders = (p,) if p[0]==p[1] else (p,p[::-1])
        for i,k in orders:
            dom,ts,state = core.dlpno_ccsd_df_pair_state(wfn,i,k)
            tn = state['TN'].np
            dtn = np.stack([(ov(p,(l,l))[1] @ t.np).ravel() for l,t in zip(dom,ts)])
            t = doubles[p] if i<=k else doubles[p].T
            u = 2*t-t.T
            mi,mk = dom.index(i),dom.index(k)
            if i == k:
                # Eqs. 96, 98--101: first dress the contracted, then the free indices.
                gram,fvv,fov = _ccsd_partial_fock_blocks(
                    h,dh,tn,dtn,y,dy,lmo[:,dom],dlmo[:,dom],fock,dfock)
                ai = dlmo[:,i] @ fock @ y + lmo[:,i] @ dfock @ y + lmo[:,i] @ fock @ dy
                ai += 2*_contract_direction('Q,Qa->a',gram,(h.ov[:,mi,:],dh.ov[:,mi,:]))[1]
                ai -= _contract_direction('Qae,me,Qm->a',(h.vv,dh.vv),(tn,dtn),
                                          (h.oo[:,mi,:],dh.oo[:,mi,:]))[1]
                ti = ts[mi].np.ravel()
                ai += fvv[1] @ ti
                ai -= _contract_direction('ma,mb,b->a',(tn,dtn),fov,(ti,None))[1]
                ai -= _contract_direction('ma,m->a',(tn,dtn),(fij[dom,i],dfij[dom,i]))[1]
                result[i] += ai
            # A (88), including the ordered ki amplitude.
            ovvv = _contract_direction('Qc,Qda->cda',(h.ov[:,mk,:],dh.ov[:,mk,:]),(h.vv,dh.vv))
            dest = ov(p,(i,i))
            result[i] += _contract_direction('ax,cda,cd->x',dest,ovvv,(u.T,None))[1]
            # Start the Fkc(ki) used by C (90) with its bare k-virtual row.
            fk = lmo[:,k] @ fock @ y
            dfk = dlmo[:,k] @ fock @ y + lmo[:,k] @ dfock @ y + lmo[:,k] @ fock @ dy
            for l,t1 in zip(dom,ts):
                kl = k,l
                K = exchange(*kl)
                s = ov(kl,p)
                coefficient = _contract_direction('ac,cd,bd,ab->',s,(u.T,None),s,K)
                result[i] -= _contract_direction(',a->a',coefficient,singles((i,i),l,t1))[1]
                spin = tuple(2*b-b.T for b in K)
                g,dg = _contract_direction('ac,cb,b->a',ov(p,kl),spin,singles(kl,l,t1))
                fk += g
                dfk += dg
            result[i] += _contract_direction('ax,ac,c->x',dest,(u,None),(fk,dfk))[1]
            # B (89): flush each row into its diagonal singles space.
            M,dM = _contract_direction('Ql,Qc->lc',(h.oo[:,mi,:],dh.oo[:,mi,:]),
                                       (h.ov[:,mk,:],dh.ov[:,mk,:]))
            corr,dcorr = _contract_direction('lb,bc->lc',(tn,dtn),exchange(i,k))
            row,drow = (M+corr) @ u.T, (dM+dcorr) @ u.T
            for lidx,l in enumerate(dom):
                result[l] -= _contract_direction('a,ax->x',(row[lidx],drow[lidx]),ov(p,(l,l)))[1]
    return result


def ccsd_doubles_fock_source(wfn, frames, factors_for, doubles, exchange, lmo, dlmo, fock, dfock):
    """Full E/G contributions, Jiang 80--81, including the bare Fock response."""
    from psi4 import core
    def ov(p,q):
        yp,yq = (frames.frames[tuple(sorted(pair))] for pair in (p,q))
        return yp.T @ frames.overlap @ yq, frames.overlap_tangent(p,q)
    def ordered(i,j):
        t = doubles[tuple(sorted((i,j)))]
        return t if i<=j else t.T
    fij,dfij = _ccsd_occupied_bar(wfn,frames,factors_for,lmo,dlmo,fock,dfock)
    domains = {ij:v.np[:,0].astype(int) for ij,v in core.dlpno_lmp2_blocks(wfn,'OCC_DOMAIN').items()}
    result = {}
    for ij,y in frames.frames.items():
        i,j = ij
        dom,ts,state = core.dlpno_ccsd_df_pair_state(wfn,i,j)
        tn = state['TN'].np
        dtn = np.stack([(frames.overlap_tangent(ij,(m,m)) @ s.np).ravel() for m,s in zip(dom,ts)])
        h,dh = factors_for(i,j)
        _,fvv,fov = _ccsd_partial_fock_blocks(h,dh,tn,dtn,y,frames.tangents[ij],
                                              lmo[:,dom],dlmo[:,dom],fock,dfock)
        dvirtual = fvv[1] - dtn.T @ fov[0] - tn.T @ fov[1]
        if i==j:
            t1 = ts[dom.index(j)].np.ravel()
            fij[dom,j] += fov[0] @ t1
            dfij[dom,j] += fov[1] @ t1
        for k in dom:
            for l in dom:
                if tuple(sorted((k,l))) not in frames.frames:
                    continue
                t = ordered(k,l)
                s = ov((k,l),ij)
                dvirtual -= _contract_direction('bx,bd,cd,cy->xy',s,(2*t-t.T,None),exchange(k,l),s)[1]
        t = doubles[ij]
        result[ij] = t @ dvirtual.T + dvirtual @ t
    # Complete F_kj'' on the native ordered pair list before using it in G.
    for i,j in [(a,b) for a,b in frames.frames]+[(b,a) for a,b in frames.frames if a!=b]:
        for l in domains[tuple(sorted((i,j)))]:
            s = ov((l,i),(l,j))
            t = ordered(l,j)
            g,dg = _contract_direction('cd,ca,ab,db->',exchange(l,i),s,(2*t-t.T,None),s)
            fij[i,j] += g
            dfij[i,j] += dg
    for ij in frames.frames:
        i,j = ij
        for perm,(a,b) in enumerate(((i,j),(j,i))):
            g = np.zeros_like(result[ij])
            for k in domains[a,a]:
                s = ov(ij,(a,k))
                g -= _contract_direction(',uc,cd,vd->uv',(fij[k,b],dfij[k,b]),
                                         s,(ordered(a,k),None),s)[1]
            result[ij] += g.T if perm else g
    return result

def ccsd_residual_source(wfn, frames, factors_for, exchange, mixed_integrals, dlmo, dfock):
    """Assemble b^v at fixed t1,t2 in the molecular final PNO frames.

    factors_for(i,j) returns PairDFFactors and their tangents in the unique
    target pair's native local ordering; exchange(i,j) returns ordered K,dK.
    mixed_integrals(i,j,k) returns ordered derivatives of native mixed J,K
    (the unfiltered metric solve). These are distinct from the whitened
    factors in the t1 dressing. No residual contribution is counted twice.
    """
    from psi4 import core
    lmo = wfn.variable('DLPNO RESPONSE ACTIVE LMOS').np
    fock = wfn.reference_wavefunction().Fa().np
    doubles = {ij:block.np for ij,block in core.dlpno_lmp2_blocks(wfn,'CCSD_DOUBLES').items()}
    factors = lru_cache(None)(factors_for)
    pair_exchange = lru_cache(None)(exchange)

    @lru_cache(None)
    def intermediate(i,j):
        h,dh = factors(*sorted((i,j)))
        return ccsd_pair_intermediate_tangent(wfn,(i,j),frames,h,dh,doubles,pair_exchange)

    singles = ccsd_singles_source(wfn,frames,factors,doubles,pair_exchange,lmo,dlmo,fock,dfock)
    doubles_source = ccsd_doubles_fock_source(wfn,frames,factors,doubles,pair_exchange,lmo,dlmo,fock,dfock)
    for ij in frames.frames:
        h,dh = factors(*ij)
        doubles_source[ij] += ccsd_direct_df_source(wfn,ij,frames,h,dh)
        doubles_source[ij] += ccsd_beta_source(wfn,ij,frames,intermediate(*ij).beta,doubles)
        doubles_source[ij] += ccsd_mixed_doubles_source(wfn,ij,frames,intermediate,doubles,mixed_integrals)
    return singles,doubles_source
