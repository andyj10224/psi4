"""Analytic mixed RHF transformed-CCSD residual in moving final PNO frames.

This diagnostic replays Jiang 75--101 with mixed tensors at one geometry.
The molecular caller supplies the orbital/domain response; no geometry stencil
or mixed CCSD amplitude is constructed here. Discrete pair/domain lists stay fixed.
"""
from functools import lru_cache
import numpy as np
from .dlpno_triples_curvature import MixedBlock, mixed_product, _symmetric_inverse as _inverse, _filtered_inverse


def _sum(*terms):
    return MixedBlock(*(sum(parts) for parts in zip(*terms)))


def _scale(a, x):
    return MixedBlock(*(a*p for p in x))


def _transpose(x):
    return MixedBlock(*(p.T for p in x))


def _stack(rows):
    return MixedBlock(*(np.stack(parts) for parts in zip(*rows)))


def mixed_ccsd_residual(wfn, electronic, overlap, local_df, *, fitted=None):
    """Return full R1/R2 MixedBlocks on the affine t1/t2 path.

    electronic supplies lmo, fock, ('pno',pair), ('t1',i), ('t2',pair).
    local_df(aux_indices) supplies the local Coulomb metric and raw AO integrals
    in exactly that auxiliary order. AO projections here use full occupied/AO
    domains, as in the Hessian model; screened-domain refits need a separate
    molecular provider. Fail closed if the model uses screened orbital domains.
    A supplied fitted provider replaces projected OO/OV/VV and cross-pair J/K
    with the native-domain local fit. Native state supplies fixed metadata.
    """
    from psi4 import core
    for option in (() if fitted is not None else ('T_CUT_DO','T_CUT_DO_PRE','T_CUT_DO_IJ','T_CUT_DO_UV',
                   'T_CUT_MKN','T_CUT_PRE','T_CUT_CLMO','T_CUT_CPAO','DLPNO_AO_INTS_TOL')):
        if core.get_option('DLPNO', option) != 0.:
            raise ValueError('Mixed CCSD molecular replay currently requires full orbital/AO domains.')
    C = mixed_product
    lmo, fock = electronic('lmo'), electronic('fock')
    pairs = tuple(core.dlpno_lmp2_blocks(wfn,'CCSD_DOUBLES'))
    nocc = lmo.value.shape[1]
    singles = [electronic(('t1',i)) for i in range(nocc)]
    doubles = {p:electronic(('t2',p)) for p in pairs}
    if any(np.any(x.uv) for x in [*singles,*doubles.values()]):
        raise ValueError('The mixed CCSD source requires an affine amplitude path (t_uv = 0).')
    domains, auxiliaries = {}, {}
    for p in pairs:
        dom,_,state = core.dlpno_ccsd_df_pair_state(wfn,*p)
        domains[p] = tuple(dom)
        auxiliaries[p] = tuple(state['AUX'].np[:,0].astype(int))
    key = lambda p:tuple(sorted(p))
    frame = lambda p:electronic(('pno',key(p)))
    def amp(p):
        t = doubles[key(p)]
        return t if p[0] <= p[1] else _transpose(t)
    def spin(t): return _sum(_scale(2.,t),_scale(-1.,_transpose(t)))
    @lru_cache(None)
    def ov(p,q): return C('ma,mn,nb->ab',frame(p),overlap,frame(q))
    @lru_cache(None)
    def sing(p,m): return C('ab,b->a',ov(p,(m,m)),singles[m])
    @lru_cache(None)
    def projected(p):
        p = key(p)
        if fitted is not None: return fitted.projected(p)
        metric, raw = local_df(auxiliaries[p])
        g = _inverse(metric)
        gf = _filtered_inverse(metric,g)
        lm,y = lmo.slice((slice(None),list(domains[p]))),frame(p)
        proj = lambda a,b:C('ma,Qmn,nb->Qab',a,raw,b)
        return proj(lm,lm),proj(lm,y),proj(y,y),g,gf,raw
    @lru_cache(None)
    def tn(p): return _stack([sing(p,m) for m in domains[key(p)]])
    def df(p,subscripts,*args):
        # Every fitted term contains exactly two half-inverse factors. Contract
        # their product G_f directly, retaining the native spectral cutoff.
        lhs,rhs = subscripts.split('->')
        axes = lhs.split(',')
        at = [i for i,s in enumerate(axes) if 'Q' in s]
        if len(at) != 2: raise ValueError('Expected two fitted factors.')
        axes[at[1]] = axes[at[1]].replace('Q','R')
        return C(','.join(axes)+',QR->'+rhs,*args,projected(p)[4])
    @lru_cache(None)
    def exchange(i,j):
        p = key((i,j))
        _,h,_,g,_,_ = projected(p)
        dom = domains[p]
        return C('Qa,QR,Rb->ab',h.slice((slice(None),dom.index(i),slice(None))),g,
                 h.slice((slice(None),dom.index(j),slice(None))))
    @lru_cache(None)
    def dressed(p,m):
        oo,h,f,_,_,_ = projected(p)
        row = domains[key(p)].index(m)
        t = tn(p)
        o = _sum(oo.slice((slice(None),row,slice(None))),C('Qkc,c->Qk',h,t.slice(row)))
        v = _sum(h.slice((slice(None),row,slice(None))),
                 _scale(-1.,C('Qk,ka->Qa',o,t)),C('Qac,c->Qa',f,t.slice(row)))
        return o,v
    @lru_cache(None)
    def intermediate(i,j):
        p = (i,j)
        dom = domains[key(p)]
        a,b = dom.index(i),dom.index(j)
        oo,h,f,_,_,_ = projected(p)
        t = tn(p)
        oi,vi = dressed(p,i)
        oj,vj = dressed(p,j)
        beta = _sum(df(p,'Qk,Ql->kl',oi,oj),df(p,'Qkc,cd,Qld->kl',h,amp(p),h))
        coul = df(p,'Q,Qlc->lc',oo.slice((slice(None),a,b)),h)
        exch = df(p,'Ql,Qc->lc',oo.slice((slice(None),a,slice(None))),h.slice((slice(None),b,slice(None))))
        ivvv = df(p,'Qb,Qac->bac',h.slice((slice(None),a,slice(None))),f)
        jvvv = df(p,'Qb,Qac->bac',h.slice((slice(None),b,slice(None))),f)
        gamma = _sum(_scale(-1.,C('la,lc->ac',t,coul)),C('b,bac->ac',sing(p,j),ivvv))
        delta = _sum(C('la,lc->ac',t,coul),_scale(-2.,C('la,lc->ac',t,exch)),
                     _scale(2.,C('b,cab->ac',sing(p,i),jvvv)),_scale(-1.,C('b,bca->ac',sing(p,i),jvvv)))
        for l in dom:
            il,lj = (i,l),(l,j)
            kil,llj = exchange(*il),spin(exchange(*lj))
            column = C('ca,ba,b->c',ov(p,il),kil,sing(il,j))
            middle = C('ad,de,ec->ac',amp(lj),ov(lj,il),kil)
            gamma = _sum(gamma,_scale(-1.,C('a,c->ac',sing(p,l),column)),
                         _scale(-.5,C('xa,ac,cy->xy',ov(p,lj),middle,ov(il,p))))
            column = C('b,ba,ac->c',sing(lj,i),llj,ov(lj,p))
            middle = C('ad,de,ec->ac',spin(amp(il)),ov(il,lj),llj)
            delta = _sum(delta,_scale(-1.,C('a,c->ac',sing(p,l),column)),
                         _scale(.5,C('xa,ac,cy->xy',ov(p,il),middle,ov(lj,p))))
        return beta,gamma,delta
    @lru_cache(None)
    def partial_fock(p):
        _,h,f,_,_,_ = projected(p)
        t = tn(p)
        gram = C('Qmc,mc->Q',h,t)
        def bar(block,left):
            return _sum(C('mr,mn,nc->rc',left,fock,frame(p)),
                        _scale(2.,df(p,'Q,Qrc->rc',gram,block)),
                        _scale(-1.,df(p,'Qre,me,Qmc->rc',block,t,h)))
        return gram,bar(f,frame(p)),bar(h,lmo.slice((slice(None),list(domains[key(p)]))))
    foo = C('mi,mn,nj->ij',lmo,fock,lmo)
    fij = [[foo.slice((i,j)) for j in range(nocc)] for i in range(nocc)]
    orders = [p for pair in pairs for p in ((pair,) if pair[0]==pair[1] else (pair,pair[::-1]))]
    for p in orders:
        i,j = p
        oo,h,_,_,_,_ = projected(p)
        dom = domains[key(p)]
        a,b = dom.index(i),dom.index(j)
        coul = df(p,'Q,Qlc->lc',oo.slice((slice(None),a,b)),h)
        ex = df(p,'Ql,Qc->lc',oo.slice((slice(None),b,slice(None))),h.slice((slice(None),a,slice(None))))
        fij[i][j] = _sum(fij[i][j],C('lc,lc->',tn(p),_sum(_scale(2.,coul),_scale(-1.,ex))))
    one = [_scale(0.,t) for t in singles]
    two = {}
    # Singles, Jiang 87--90, with the complete free/contracted-index Fai dressing.
    for p in orders:
        i,k = p
        dom = domains[key(p)]
        mi,mk = dom.index(i),dom.index(k)
        oo,h,f,_,_,_ = projected(p)
        t,u = tn(p),spin(amp(p))
        if i == k:
            gram,fvv,fov = partial_fock(p)
            ti = singles[i]
            ai = _sum(C('m,mn,na->a',lmo.slice((slice(None),i)),fock,frame(p)),
                 _scale(2.,df(p,'Q,Qa->a',gram,h.slice((slice(None),mi,slice(None))))),
                 _scale(-1.,df(p,'Qae,me,Qm->a',f,t,oo.slice((slice(None),mi,slice(None))))),
                 C('ab,b->a',fvv,ti),_scale(-1.,C('ma,mb,b->a',t,fov,ti)),
                 _scale(-1.,C('ma,m->a',t,_stack([fij[m][i] for m in dom]))))
            one[i] = _sum(one[i],ai)
        ovvv = df(p,'Qc,Qda->cda',h.slice((slice(None),mk,slice(None))),f)
        dest = ov(p,(i,i))
        one[i] = _sum(one[i],C('ax,cda,cd->x',dest,ovvv,_transpose(u)))
        fk = C('m,mn,na->a',lmo.slice((slice(None),k)),fock,frame(p))
        for l in dom:
            kl = (k,l)
            K = exchange(*kl)
            s = ov(kl,p)
            coefficient = C('ac,cd,bd,ab->',s,_transpose(u),s,K)
            one[i] = _sum(one[i],_scale(-1.,C(',a->a',coefficient,sing((i,i),l))))
            fk = _sum(fk,C('ac,cb,b->a',ov(p,kl),spin(K),sing(kl,l)))
        one[i] = _sum(one[i],C('ax,ac,c->x',dest,u,fk))
        M = df(p,'Ql,Qc->lc',oo.slice((slice(None),mi,slice(None))),h.slice((slice(None),mk,slice(None))))
        row = C('lb,cb->lc',_sum(M,C('lb,bc->lc',t,exchange(i,k))),u)
        for idx,l in enumerate(dom):
            one[l] = _sum(one[l],_scale(-1.,C('a,ax->x',row.slice(idx),ov(p,(l,l)))))
    # F'' occupied: finish all rows before using the G term.
    for j in range(nocc):
        p = (j,j)
        fov = partial_fock(p)[2]
        corr = C('ma,a->m',fov,singles[j])
        for m,i in enumerate(domains[p]): fij[i][j] = _sum(fij[i][j],corr.slice(m))
    for i,j in orders:
        for l in domains[key((i,j))]:
            s = ov((l,i),(l,j))
            fij[i][j] = _sum(fij[i][j],C('cd,ca,ab,db->',exchange(l,i),s,spin(amp((l,j))),s))
    @lru_cache(None)
    def mixed_integrals(i,j,k):
        if fitted is not None: return fitted.mixed_integrals(i,j,k)
        p = (i,j)
        _,_,_,g,_,raw = projected(p)
        li,lk = (lmo.slice((slice(None),m)) for m in (i,k))
        yp,yq = frame(p),frame((k,j))
        a = C('m,Qmn,n->Q',li,raw,lk)
        b = C('ma,Qmn,nc->Qac',yp,raw,yq)
        c = C('m,Qmn,na->Qa',li,raw,yp)
        d = C('m,Qmn,nc->Qc',lk,raw,yq)
        return C('Q,QR,Rac->ac',a,g,b),C('Qa,QR,Rc->ac',c,g,d)
    def ordered_cd(i,j):
        p = (i,j)
        z = _scale(0.,amp(p))
        c,d = z,z
        for k in domains[key(p)]:
            ki,kj,ik,jk = (k,i),(k,j),(i,k),(j,k)
            J,K = mixed_integrals(i,j,k)
            g = C('xa,ac,cy->xy',ov(p,ki),intermediate(*ki)[1],ov(ki,kj))
            c = _sum(c,_scale(-1.,C('ac,bc,by->ay',_sum(J,g),amp(kj),ov(kj,p))))
            g = C('xa,ac,cy->xy',ov(p,ik),intermediate(*ik)[2],ov(ik,jk))
            vertex = _sum(_scale(2.,K),_scale(-1.,J),g)
            d = _sum(d,_scale(.5,C('ac,bc,by->ay',vertex,spin(amp(jk)),ov(jk,p))))
        return _sum(_scale(.5,c),_transpose(c),d)
    # Doubles, Jiang 75--81 (unique pair with both ordered permutations).
    for p in pairs:
        i,j = p
        dom = domains[p]
        _,h,f,_,_,_ = projected(p)
        t = amp(p)
        vv = _sum(f,_scale(-1.,C('ka,Qkc->Qac',tn(p),h)))
        result = _sum(df(p,'Qa,Qb->ab',dressed(p,i)[1],dressed(p,j)[1]),
                      df(p,'Qac,cd,Qbd->ab',vv,t,vv))
        _,fvv,fov = partial_fock(p)
        virtual = _sum(fvv,_scale(-1.,C('ma,mb->ab',tn(p),fov)))
        beta = intermediate(*p)[0]
        for a,k in enumerate(dom):
            for b,l in enumerate(dom):
                q = (k,l)
                if key(q) not in doubles: continue
                s = ov(p,q)
                result = _sum(result,C(',ac,cd,bd->ab',beta.slice((a,b)),s,amp(q),s))
                s = ov(q,p)
                virtual = _sum(virtual,_scale(-1.,C('bx,bd,cd,cy->xy',s,spin(amp(q)),exchange(k,l),s)))
        result = _sum(result,C('ac,bc->ab',t,virtual),C('ac,cb->ab',virtual,t),
                      ordered_cd(i,j),_transpose(ordered_cd(j,i)))
        for perm,(a,b) in enumerate(((i,j),(j,i))):
            g = _scale(0.,t)
            for k in domains[a,a]:
                s = ov(p,(a,k))
                g = _sum(g,_scale(-1.,C(',uc,cd,vd->uv',fij[k][b],s,amp((a,k)),s)))
            result = _sum(result,_transpose(g) if perm else g)
        two[p] = result
    return one,two
