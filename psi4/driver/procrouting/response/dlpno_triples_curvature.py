"""Mixed local projections and Jiang (T) moment derivatives on a fixed-rank branch."""
from functools import lru_cache
from itertools import permutations
from typing import NamedTuple

import numpy as np

from .dlpno_triples_response import TriplesMixedRow


class MixedBlock(NamedTuple):
    """Value, two first derivatives and the mixed derivative in one common frame."""
    value: np.ndarray
    u: np.ndarray
    v: np.ndarray
    uv: np.ndarray

    def slice(self, index):
        return MixedBlock(*(part[index] for part in self))


def mixed_product(subscripts, *blocks):
    """Differentiate a local multilinear contraction in two directions."""
    def contraction(*parts):
        return np.einsum(subscripts, *(block[i] for block,i in zip(blocks,parts)), optimize=True)
    indices = [0]*len(blocks)
    value = contraction(*indices)
    u,v,uv = (np.zeros_like(value) for _ in range(3))
    for i in range(len(blocks)):
        indices[i] = 1
        u += contraction(*indices)
        for j in range(len(blocks)):
            if j != i:
                indices[j] = 2
                uv += contraction(*indices)
                indices[j] = 0
        indices[i] = 2
        v += contraction(*indices)
        indices[i] = 3
        uv += contraction(*indices)
        indices[i] = 0
    return MixedBlock(value,u,v,uv)


def _inverse(metric):
    """Full local Coulomb inverse from differentiating M G = I."""
    m,mu,mv,muv = metric
    g = np.linalg.solve(m,np.eye(len(m)))
    gu,gv = -g@mu@g, -g@mv@g
    return MixedBlock(g,gu,gv,-g@(muv@g + mu@gv + mv@gu))


def _filtered_inverse(metric, full):
    """Fixed spectral branch of the product of the two native half-inverses."""
    m,mu,mv,muv = metric
    if not len(m):
        return full
    eig,rot = np.linalg.eigh(m)
    active = (eig > 0.) & (eig >= 1.e-14 * max(abs(eig[0]),abs(eig[-1])))
    if active.all():
        return full
    g = np.zeros_like(eig)
    g[active] = 1./eig[active]
    a,b = eig[:,None],eig[None,:]
    both = active[:,None] & active[None,:]
    split = active[:,None] != active[None,:]
    dd1 = -g[:,None]*g[None,:]
    dd1[split] = ((g[:,None]-g[None,:])[split] / (a-b)[split])
    eu,ev = rot.T@mu@rot,rot.T@mv@rot
    mixed = dd1*(rot.T@muv@rot)
    for k,e in enumerate(eig):
        # Symmetric second divided difference; use a cross-branch denominator
        # when possible, and analytic same-branch limits at repeated eigenvalues.
        dd2 = np.zeros_like(dd1)
        if active[k]:
            dd2[both] = (g[:,None]*g[k]*g[None,:])[both]
        # For (a,k,b) with endpoints on different branches.
        dd2[split] = ((dd1[:,k,None]-dd1[k,None,:])[split] / (a-b)[split])
        # If the endpoints share a branch opposite to k, interchange k and b.
        opposite = ~split & (active[:,None] != active[k])
        dd2[opposite] = ((dd1-dd1[:,k,None]).T[opposite] / np.broadcast_to(a-e,dd1.shape)[opposite])
        mixed += dd2 * (eu[:,k,None]*ev[k,None,:] + ev[:,k,None]*eu[k,None,:])
    return MixedBlock((rot*g)@rot.T,rot@(dd1*eu)@rot.T,rot@(dd1*ev)@rot.T,rot@mixed@rot.T)


def mixed_triples_moments(triplet, domain, metric, oo, ov, vv, transfer, single, double, fov, virtual_fock):
    """Contract the mixed W/V moments from their local inputs, with all product terms.

    All array inputs and callback results are MixedBlocks. OO/OV/VV are raw
    local three-center blocks in native domain order. transfer(i,j) gives
    the pair-by-triplet overlap. single(i), double(i,j) give ordered amplitudes
    in the moving pair frames. fov(i) and virtual_fock are projected on the TNOs.
    """
    pos = {i:n for n,i in enumerate(domain)}
    n = vv.value.shape[-1]
    full = _inverse(metric)
    filtered = _filtered_inverse(metric,full)

    @lru_cache(None)
    def pair(a,b):
        s = transfer(a,b)
        return mixed_product('ia,ij,jb->ab',s,double(a,b),s)

    @lru_cache(None)
    def ovvv(a):
        return mixed_product('Qa,QR,Rbd->abd',ov.slice((slice(None),pos[a])),full,vv)

    @lru_cache(None)
    def ooov(b,c):
        return mixed_product('Ql,QR,Rc->lc',oo.slice((slice(None),pos[b])),filtered,
                             ov.slice((slice(None),pos[c])))

    w = np.zeros((n,n,n))
    for perm in permutations(range(3)):
        a,b,c = (triplet[k] for k in perm)
        term = mixed_product('abd,cd->abc',ovvv(a),pair(c,b)).uv
        integral = ooov(b,c)
        for m,l in enumerate(domain):
            term -= mixed_product('ab,c->abc',pair(a,l),integral.slice(m)).uv
        w += term.transpose(np.argsort(perm))
    v = w.copy()
    for slot,a in enumerate(triplet):
        b,c = (triplet[i] for i in range(3) if i != slot)
        projected_single = mixed_product('ia,i->a',transfer(a,a),single(a))
        exchange = mixed_product('Qb,QR,Rc->bc',ov.slice((slice(None),pos[b])),filtered,
                                 ov.slice((slice(None),pos[c])))
        term = mixed_product('a,bc->abc',projected_single,exchange).uv
        term += mixed_product('a,bc->abc',fov(a),pair(b,c)).uv
        v += np.moveaxis(term,0,slot)
    return TriplesMixedRow(w,v,virtual_fock.uv)


def project_mixed_triples_row(triplet, domain, metric, three_center, overlap, fock, lmos, tnos,
                              pair_frame, single, double):
    """Project molecular mixed inputs into one triplet and evaluate W_uv, V_uv, Fvv_uv.

    Inputs must describe the same forward fit. In the full AO fit the three-center
    block is the raw AO tensor; screened/refitted domains must supply the matching
    refitted blocks to mixed_triples_moments when their fits differ by vertex.
    All coefficient panels use the AO ordering of the supplied integral blocks.
    """
    local = lmos.slice((slice(None),list(domain)))
    oo = mixed_product('ma,Qmn,nb->Qab',local,three_center,local)
    ov = mixed_product('ma,Qmn,nb->Qab',local,three_center,tnos)
    vv = mixed_product('ma,Qmn,nb->Qab',tnos,three_center,tnos)
    vf = mixed_product('ma,mn,nb->ab',tnos,fock,tnos)

    @lru_cache(None)
    def transfer(i,j):
        return mixed_product('ma,mn,nb->ab',pair_frame(i,j),overlap,tnos)

    @lru_cache(None)
    def fov(i):
        return mixed_product('m,mn,nb->b',lmos.slice((slice(None),i)),fock,tnos)

    return mixed_triples_moments(triplet,domain,metric,oo,ov,vv,transfer,single,double,fov,vf)
