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


@lru_cache(maxsize=1024)
def _mixed_contraction_path(subscripts, shapes):
    """Reuse planning by shape; broadcast placeholders allocate no tensors."""
    operands=[np.broadcast_to(np.array(0.),shape) for shape in shapes]
    return tuple(np.einsum_path(subscripts,*operands,optimize=True)[0])


def mixed_product(subscripts, *blocks):
    """Differentiate a local multilinear contraction in two directions."""
    path=_mixed_contraction_path(subscripts,tuple(np.shape(block[0]) for block in blocks))
    def contraction(*parts):
        return np.einsum(subscripts, *(block[i] for block,i in zip(blocks,parts)), optimize=path)
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
    """Full matrix inverse from differentiating M G = I."""
    m,mu,mv,muv = metric
    g = np.linalg.solve(m,np.eye(len(m)))
    gu,gv = -g@mu@g, -g@mv@g
    return MixedBlock(g,gu,gv,-g@(muv@g + mu@gv + mv@gu))


def _symmetric_inverse(metric):
    """Differentiate M G = I while preserving the real metric's symmetry.

    A general LU inverse can acquire an antisymmetric roundoff component on
    conditioned fitting metrics. Its derivatives can then leave the physical
    symmetric diagonal-pair CCSD space. Preserve this known tensor symmetry
    at the inverse itself; do not project residual sources or Hessian entries.
    The full Coulomb inverse remains unfiltered; signed spectra are supported.
    """
    symmetric=lambda x:(x+x.T)*.5
    if any(not np.allclose(x,x.T,rtol=1.e-12,atol=1.e-12) for x in metric):
        raise ValueError('Expected symmetric metric and metric derivatives.')
    m,mu,mv,muv = (symmetric(x) for x in metric)
    eig,rot=np.linalg.eigh(m)
    if np.any(eig==0.):raise np.linalg.LinAlgError('Singular metric.')
    inverse=1./eig
    # Work in the reference spectral basis. Dense products of several full
    # inverses amplify cancellation on conditioned metrics even after their
    # antisymmetric roundoff has been removed. No eigenvalue is filtered here.
    eu,ev,euv=(symmetric(rot.T@x@rot) for x in (mu,mv,muv))
    scale=inverse[:,None]*inverse[None,:]
    cross=(eu*inverse)@ev
    back=lambda x:symmetric(rot@x@rot.T)
    return MixedBlock(symmetric((rot*inverse)@rot.T),back(-scale*eu),back(-scale*ev),
                      back(scale*((cross+cross.T)-euv)))


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


def triples_moment_blocks(triplet, domain, metric, oo, ov, vv, transfer, single, double, fov, virtual_fock):
    """Contract W/V/Fvv MixedBlocks from local inputs, with all product terms.

    All array inputs and callback results are MixedBlocks. OO/OV/VV are raw
    local three-center blocks in native domain order. transfer(i,j) gives
    the pair-by-triplet overlap. single(i), double(i,j) give ordered amplitudes
    in the moving pair frames. fov(i) and virtual_fock are projected on the TNOs.
    """
    pos = {i:n for n,i in enumerate(domain)}
    n = vv.value.shape[-1]
    full = _symmetric_inverse(metric)
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

    w = MixedBlock(*(np.zeros((n,n,n)) for _ in range(4)))
    for perm in permutations(range(3)):
        a,b,c = (triplet[k] for k in perm)
        term = mixed_product('abd,cd->abc',ovvv(a),pair(c,b))
        integral = ooov(b,c)
        for m,l in enumerate(domain):
            extra = mixed_product('ab,c->abc',pair(a,l),integral.slice(m))
            term = MixedBlock(*(x-y for x,y in zip(term,extra)))
        for target,x in zip(w,term): target += x.transpose(np.argsort(perm))
    v = MixedBlock(*(x.copy() for x in w))
    for slot,a in enumerate(triplet):
        b,c = (triplet[i] for i in range(3) if i != slot)
        projected_single = mixed_product('ia,i->a',transfer(a,a),single(a))
        exchange = mixed_product('Qb,QR,Rc->bc',ov.slice((slice(None),pos[b])),filtered,
                                 ov.slice((slice(None),pos[c])))
        term = mixed_product('a,bc->abc',projected_single,exchange)
        extra = mixed_product('a,bc->abc',fov(a),pair(b,c))
        for target,x,y in zip(v,term,extra): target += np.moveaxis(x+y,0,slot)
    return w,v,virtual_fock


def mixed_triples_moments(*args, **kwargs):
    """Mixed W/V/Fvv moments; preserves the Hessian-row interface."""
    return TriplesMixedRow(*(x.uv for x in triples_moment_blocks(*args, **kwargs)))


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


def _mixed_projector(density, kept):
    """Differentiate P²=P and [D,P]=0 in an orthonormal parent."""
    eig,rot = np.linalg.eigh(density.value)
    n = len(eig)
    if not 0 < kept <= n:
        raise ValueError("Invalid natural-orbital selection rank.")
    if kept == n:
        zero = np.zeros((n,n))
        return MixedBlock(np.eye(n),zero,zero,zero)
    cut = n-kept
    gaps = eig[None,cut:] - eig[:cut,None]
    if np.min(gaps) <= 64*np.finfo(float).eps * max(abs(eig)):
        raise ValueError("Unresolved retained/discarded natural-orbital boundary.")
    du,dv,duv = (rot.T@x@rot for x in density[1:])
    def first(x):
        p = np.zeros_like(x)
        p[:cut,cut:] = x[:cut,cut:] / gaps
        p[cut:,:cut] = p[:cut,cut:].T
        return p
    pu,pv = first(du),first(dv)
    puv = first(duv + du@pv-pv@du + dv@pu-pu@dv)
    diagonal = pu@pv+pv@pu
    puv[:cut,:cut] = diagonal[:cut,:cut]
    puv[cut:,cut:] = -diagonal[cut:,cut:]
    return MixedBlock(rot[:,cut:]@rot[:,cut:].T,rot@pu@rot.T,rot@pv@rot.T,rot@puv@rot.T)


def mixed_tno_frame(triplet, parent, metric, pair_frames, doubles, reference_frame):
    """Mixed TNO coefficients in the reference polar gauge, including parent motion.

    For the triples-only CCSD adjoint, supply t_u, t_v and zero t_uv. Each
    repeated pair slot contributes separately to the density.
    """
    i,j,k = triplet
    n = parent.value.shape[1]
    density = MixedBlock(*(np.zeros((n,n)) for _ in range(4)))
    for p in ((i,j),(j,k),(i,k)):
        t = doubles[p]
        spin = MixedBlock(*(2*x-x.T for x in t))
        direct = mixed_product('ac,bc->ab',spin,t)
        exchange = mixed_product('ca,cb->ab',spin,t)
        d = MixedBlock(*((x+y)/(3*(1+(p[0]==p[1]))) for x,y in zip(direct,exchange)))
        transfer = mixed_product('ma,mn,nb->ab',pair_frames[p],metric,parent)
        contribution = mixed_product('ia,ij,jb->ab',transfer,d,transfer)
        for target,x in zip(density,contribution):
            target += x
    proj = _mixed_projector(density,reference_frame.shape[1])
    ao = mixed_product('ma,ab,nb->mn',parent,proj,parent)
    ref = metric.value@reference_frame
    unnormalized = MixedBlock(*(x@ref for x in ao))
    gram = mixed_product('ma,mn,nb->ab',unnormalized,metric,unnormalized)
    eye = np.eye(reference_frame.shape[1])
    if (not np.allclose(gram.value,eye,atol=2.e-8,rtol=0.) or
            not np.allclose(unnormalized.value,reference_frame,atol=2.e-8,rtol=0.)):
        raise ValueError(f"Mixed TNO selection disagrees with the native frame for {triplet}.")
    from .dlpno_pno_curvature import _polar_normalization
    polar = _polar_normalization(gram)
    return mixed_product('ma,ab->mb',unnormalized,polar)
