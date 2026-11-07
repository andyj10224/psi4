"""Amplitude, adjoint and energy response of ordinary single-space RHF iterative triples."""
from functools import lru_cache
from dataclasses import dataclass
from itertools import permutations
from typing import NamedTuple

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

from .dlpno_orbital import ccsd_pair_df_factors, _transport_pno_frame, _density_selection_tangent


class TriplesTangent(NamedTuple):
    """Directional derivatives in transported frames, and true right/left residual norms."""
    frames: dict
    amplitudes: dict
    adjoints: dict
    energy_tangent: float
    residual: float
    adjoint_residual: float


class _TriplesEdge(NamedTuple):
    source: tuple
    axes: tuple
    occupied: tuple
    fock: float
    dfock: float
    overlap: np.ndarray
    doverlap: np.ndarray


@dataclass
class TriplesGradientDirection:
    """First derivatives and residual sources in one reference triplet ordering."""
    wavefunction: object
    frame_tangents: dict
    amplitudes: dict
    adjoints: dict
    gaps: dict
    energy_seeds: dict
    energy_seed_tangents: dict
    right_source: dict
    left_source: dict
    edges: dict


def _transfer(y, dy, s, ds, z, dz):
    return y.T @ s @ z, dy.T @ s @ z + y.T @ ds @ z + y.T @ s @ dz


def _project(t, dt, u, du):
    return u.T @ t @ u, du.T @ t @ u + u.T @ dt @ u + u.T @ t @ du


def _transform(t, a, b, c):
    # Apply one rectangular transfer per leg, avoiding an explicit Kronecker matrix.
    return np.einsum('ia,jb,kc,abc->ijk', a,b,c,t,optimize=True)


def triples_moment_integrals(domain, metric, dmetric, raw, draw):
    """Build Jiang Eq. 109 vertices and their derivatives from local raw DF blocks.

    OO and OV follow the supplied occupied domain. VV follows the triplet frame.
    """
    positions = {i: k for k,i in enumerate(domain)}
    n = raw.vv.shape[1]
    if len(metric):
        fit,dfit = ccsd_pair_df_factors(metric,dmetric,raw,draw)

    @lru_cache(None)
    def ovvv(i):
        if not len(metric):
            return np.zeros((n,n,n)), np.zeros((n,n,n))
        h = raw.ov[:,positions[i],:]
        dh = draw.ov[:,positions[i],:]
        x = np.linalg.solve(metric,h)
        dx = np.linalg.solve(metric,dh-dmetric@x)
        vv = raw.vv.reshape(len(metric),-1)
        dvv = draw.vv.reshape(vv.shape)
        return (x.T@vv).reshape(n,n,n), (dx.T@vv+x.T@dvv).reshape(n,n,n)

    @lru_cache(None)
    def ooov(j,k):
        if not len(metric):
            return np.zeros((len(domain),n)), np.zeros((len(domain),n))
        a,b = positions[j],positions[k]
        oo,ov = fit.oo[:,a,:],fit.ov[:,b,:]
        return oo.T@ov, dfit.oo[:,a,:].T@ov + oo.T@dfit.ov[:,b,:]
    @lru_cache(None)
    def ovov(j,k):
        if not len(metric):
            return np.zeros((n,n)), np.zeros((n,n))
        a,b = positions[j],positions[k]
        left,right = fit.ov[:,a,:],fit.ov[:,b,:]
        return left.T@right, dfit.ov[:,a,:].T@right + left.T@dfit.ov[:,b,:]
    return ovvv,ooov,ovov


def _energy_moment(triplet,v):
    i,j,k = triplet
    result = 8*v
    for order,weight in (((k,j,i),-4), ((i,k,j),-4), ((j,i,k),-4), ((j,k,i),2), ((k,i,j),2)):
        axes = np.argsort(np.argsort(order,kind='stable'))
        result = result + weight*v.transpose(axes)
    return result / (6 if i == k else 2 if i == j or j == k else 1)


def _local_fock_action(dv,t,occ):
    return (np.einsum('ad,dbc->abc',dv,t) + np.einsum('bd,adc->abc',dv,t)
            + np.einsum('cd,abd->abc',dv,t) - occ*t)


def triples_gradient_direction(wfn, pair_frames, dt1, dt2, dlmo, dfock, parent_for, integrals_for):
    """Prepare the stationary gradient and both sources without solving triples response.

    parent_for(ijk) returns the orthonormal triplet PAO parent and its tangent.
    integrals_for(ijk, domain, aux, Y, dY) returns ovvv(i), ooov(j,k), ovov(j,k) callables.
    Each returns (value, tangent) in the native triplet fit, including LMO and
    TNO motion. ovvv uses the full inverse metric, ooov the filtered inverse.
    dt1 and dt2 are the coupled CCSD tangents in pair_frames.
    """
    from psi4 import core

    rows = core.dlpno_triples_response_state(wfn)
    doubles = {p: v.np for p,v in core.dlpno_lmp2_blocks(wfn,'CCSD_DOUBLES').items()}
    s,ds = pair_frames.overlap,pair_frames.doverlap
    lmo = wfn.variable('DLPNO RESPONSE ACTIVE LMOS').np
    f = wfn.reference_wavefunction().Fa().np
    foo = lmo.T @ f @ lmo
    half = dlmo.T @ f @ lmo
    dfoo = half + half.T + lmo.T @ dfock @ lmo

    if not rows:
        return TriplesGradientDirection(wfn,{}, {}, {}, {}, {}, {}, {}, {}, {})

    def projected(i,j, y,dy):
        p = tuple(sorted((i,j)))
        u,du = _transfer(pair_frames.frames[p],pair_frames.tangents[p],s,ds,y,dy)
        t,dt = doubles[p],dt2[p]
        if i > j:
            t,dt = t.T,dt.T
        return _project(t,dt,u,du)

    dy3, tensors, gaps, sources, frames = {}, {}, {}, {}, {}
    singles = {}
    for i in range(lmo.shape[1]):
        domain,ts,_ = core.dlpno_ccsd_df_pair_state(wfn,i,i)
        singles[i] = ts[domain.index(i)].np.ravel()
    energy_seeds,dseeds,z3,left_source = {},{},{},{}
    for triplet,(domain,aux,frame,amplitude,moment,_,adjoint,native_v) in rows.items():
        y = frames[triplet] = frame.np
        n = y.shape[1]
        parent,dp = parent_for(triplet)
        metric = parent.T @ s @ parent
        motion = dp.T @ s @ parent
        if (not np.allclose(metric,np.eye(metric.shape[0]),rtol=0.,atol=2.e-8)
                or not np.allclose(motion+motion.T+parent.T@ds@parent,0.,rtol=0.,atol=2.e-8)):
            raise ValueError(f"Inconsistent moving PAO parent metric for triplet {triplet}.")
        density = np.zeros((parent.shape[1],)*2)
        ddensity = np.zeros_like(density)
        i,j,k = triplet
        for a,b in ((i,j),(j,k),(i,k)):
            p = (a,b)
            t,dt = doubles[p],dt2[p]
            u,du = 2*t-t.T,2*dt-dt.T
            d = u@t.T + u.T@t
            dd = du@t.T + u@dt.T + du.T@t + u.T@dt
            transfer,dtransfer = _transfer(pair_frames.frames[p],pair_frames.tangents[p],s,ds,parent,dp)
            block,dblock = _project(d,dd,transfer,dtransfer)
            weight = 1./(3*(1+(a==b)))
            density += weight*block
            ddensity += weight*dblock
        selection = _density_selection_tangent(parent,dp,density,ddensity,n)
        dy = dy3[triplet] = _transport_pno_frame(y,selection,s,ds,triplet)
        t = tensors[triplet] = amplitude.np.reshape((n,)*3)
        w,dw = np.zeros_like(t), np.zeros_like(t)
        z = z3[triplet] = adjoint.np.reshape(t.shape)
        ovvv,ooov,ovov = integrals_for(triplet,domain,aux,y,dy)
        pair_cache = {}
        def pair(a,b):
            if (a,b) not in pair_cache:
                pair_cache[a,b] = projected(a,b,y,dy)
            return pair_cache[a,b]
        for perm in permutations(range(3)):
            a,b,c = (triplet[v] for v in perm)
            ov,dov = ovvv(a)
            tcb,dtcb = pair(c,b)
            part = np.einsum('abd,cd->abc',ov,tcb,optimize=True)
            dpart = (np.einsum('abd,cd->abc',dov,tcb,optimize=True)
                     + np.einsum('abd,cd->abc',ov,dtcb,optimize=True))
            oo,doo = ooov(b,c)
            for m,l in enumerate(domain):
                tal,dtal = pair(a,l)
                part -= tal[:,:,None] * oo[m]
                dpart -= dtal[:,:,None] * oo[m] + tal[:,:,None] * doo[m]
            axes = np.argsort(perm)
            w += part.transpose(axes)
            dw += dpart.transpose(axes)
        if not np.allclose(w,moment.np.reshape(t.shape),atol=2.e-10,rtol=1.e-8):
            raise ValueError(f"Reconstructed W disagrees with native triples moment for {triplet}.")
        v,dv_moment = w.copy(),dw.copy()
        for slot,a in enumerate(triplet):
            other = [m for m in range(3) if m != slot]
            b,c = (triplet[m] for m in other)
            p = (a,a)
            u,du = _transfer(pair_frames.frames[p],pair_frames.tangents[p],s,ds,y,dy)
            single = u.T@singles[a]
            dsingle = du.T@singles[a] + u.T@dt1[a]
            g,dg = ovov(b,c)
            tbc,dtbc = pair(b,c)
            fa = lmo[:,a]@f@y
            dfa = dlmo[:,a]@f@y + lmo[:,a]@dfock@y + lmo[:,a]@f@dy
            # Put the distinguished singles/Fock leg back into its original slot.
            def outer(a,bc):
                return np.moveaxis(a[:,None,None]*bc,0,slot)
            v += outer(single,g) + outer(fa,tbc)
            dv_moment += outer(dsingle,g) + outer(single,dg) + outer(dfa,tbc) + outer(fa,dtbc)
        if not np.allclose(v,native_v.np.reshape(t.shape),atol=2.e-10,rtol=1.e-8):
            raise ValueError(f"Reconstructed V disagrees with native triples moment for {triplet}.")
        energy_seeds[triplet] = _energy_moment(triplet,v)
        dseeds[triplet] = _energy_moment(triplet,dv_moment)
        vv = y.T @ f @ y
        eps = vv.diagonal()
        if not np.allclose(vv,np.diag(eps),atol=1.e-8,rtol=0.):
            raise ValueError("Expected the native semicanonical reference TNOs.")
        gaps[triplet] = (eps[:,None,None]+eps[None,:,None]+eps[None,None,:]
                         -sum(foo[a,a] for a in triplet))
        half = dy.T @ f @ y
        dv = half + half.T + y.T @ dfock @ y
        docc = sum(dfoo[a,a] for a in triplet)
        sources[triplet] = dw + _local_fock_action(dv,t,docc)
        left_source[triplet] = dseeds[triplet] + _local_fock_action(dv,z,docc)

    edges = {r: [] for r in rows}
    left_residual = {r: energy_seeds[r] + gaps[r]*z3[r] for r in rows}
    for r,(_,_,_,_,w,row_edges,_,_) in rows.items():
        residual = w.np.reshape(tensors[r].shape) + gaps[r]*tensors[r]
        for l,m,i,j,k,_ in row_edges:
            order = (i,j,k)
            q = tuple(sorted(order))
            axes = np.argsort(np.argsort(order,kind='stable'))
            u,du = _transfer(frames[r],dy3[r],s,ds,frames[q],dy3[q])
            tq = tensors[q].transpose(axes)
            transferred = _transform(tq,u,u,u)
            residual -= foo[l,m]*transferred
            sources[r] -= dfoo[l,m]*transferred + foo[l,m]*(
                _transform(tq,du,u,u) + _transform(tq,u,du,u) + _transform(tq,u,u,du))
            z = z3[r]
            back = np.argsort(axes)
            zq = _transform(z,u.T,u.T,u.T).transpose(back)
            left_residual[q] -= foo[l,m]*zq
            moved = (_transform(z,du.T,u.T,u.T) + _transform(z,u.T,du.T,u.T)
                     + _transform(z,u.T,u.T,du.T)).transpose(back)
            left_source[q] -= dfoo[l,m]*zq + foo[l,m]*moved
            edges[r].append(_TriplesEdge(q,tuple(axes),(l,m),foo[l,m],dfoo[l,m],u,du))
        if np.max(abs(residual)) > max(2.e-10, 1.01*core.get_option('DLPNO','R_CONVERGENCE')):
            raise ValueError(f"Reconstructed iterative triples residual disagrees for {r}.")

    threshold = max(2.e-10, 1.01*core.get_option('DLPNO','R_CONVERGENCE'))
    if any(np.max(abs(r)) > threshold for r in left_residual.values()):
        raise ValueError("Reconstructed ordinary triples adjoint residual disagrees.")
    energy = sum(np.vdot(energy_seeds[r],tensors[r]) for r in rows)
    if not np.isclose(energy,wfn.variable('(T) CORRECTION ENERGY'),rtol=1.e-9,atol=1.e-11):
        raise ValueError("Reconstructed ordinary triples energy disagrees.")
    return TriplesGradientDirection(wfn,dy3,tensors,z3,gaps,energy_seeds,dseeds,sources,left_source,edges)


def solve_triples_tangent(direction, *, tolerance=1.e-10):
    """Solve both tangent equations using the prepared gradient direction."""
    keys = list(direction.amplitudes)
    if not keys:
        return TriplesTangent({}, {}, {}, 0., 0., 0.)
    tensors, z3 = direction.amplitudes, direction.adjoints
    gaps, edges = direction.gaps, direction.edges
    energy_seeds, dseeds = direction.energy_seeds, direction.energy_seed_tangents
    sources, left_source = direction.right_source, direction.left_source
    cuts = np.cumsum([tensors[r].size for r in keys])[:-1]
    def pack(values):
        return np.concatenate([values[r].ravel() for r in keys])
    def unpack(flat):
        return {r: block.reshape(tensors[r].shape) for r,block in zip(keys,np.split(flat,cuts))}
    def apply(flat,left):
        x = unpack(flat)
        result = {r: gaps[r]*x[r] for r in keys}
        for r in keys:
            for edge in edges[r]:
                q,axes,fij,u = edge.source,edge.axes,edge.fock,edge.overlap
                if left:
                    result[q] -= fij*_transform(x[r],u.T,u.T,u.T).transpose(np.argsort(axes))
                else:
                    result[r] -= fij*_transform(x[q].transpose(axes),u,u,u)
        return pack(result)
    diag = pack(gaps)
    if not np.isfinite(diag).all() or np.any(abs(diag)<1.e-12):
        raise ValueError("Singular triples tangent denominator.")
    def solve(source,left):
        rhs = -pack(source)
        if not np.isfinite(rhs).all():
            raise ValueError("Invalid triples tangent source.")
        operator = LinearOperator((rhs.size,rhs.size),matvec=lambda x:apply(x,left),dtype=float)
        inverse = LinearOperator(operator.shape,matvec=lambda x:x/diag,dtype=float)
        x,info = gmres(operator,rhs,M=inverse,rtol=tolerance,atol=0.)
        err = np.linalg.norm(operator@x-rhs)
        if info or not np.isfinite(err) or err > tolerance*max(np.linalg.norm(rhs),1.e-14):
            raise RuntimeError(f"Iterative triples tangent residual {err:.3e} (info={info}).")
        return unpack(x),err
    dt,right_error = solve(sources,False)
    dz,left_error = solve(left_source,True)
    de = sum(np.vdot(dseeds[r],tensors[r])+np.vdot(energy_seeds[r],dt[r]) for r in keys)
    dual_de = sum(np.vdot(dseeds[r],tensors[r])+np.vdot(z3[r],sources[r]) for r in keys)
    if not np.isclose(de,dual_de,atol=max(1.e-10,tolerance),rtol=1.e-8):
        raise RuntimeError("Triples energy tangent disagrees with its adjoint contraction.")
    return TriplesTangent(direction.frame_tangents,dt,dz,de,right_error,left_error)


def iterative_triples_tangent(wfn, pair_frames, dt1, dt2, dlmo, dfock, parent_for, integrals_for, *, tolerance=1.e-10):
    """Differentiate the TNO frames, right and adjoint triples, and (T) energy."""
    direction = triples_gradient_direction(wfn,pair_frames,dt1,dt2,dlmo,dfock,parent_for,integrals_for)
    return solve_triples_tangent(direction,tolerance=tolerance)


class TriplesMixedRow(NamedTuple):
    """Mixed derivatives W_uv, V_uv, and the projected virtual Fock block in this triplet."""
    right_moment: np.ndarray
    energy_moment: np.ndarray
    virtual_fock: np.ndarray


def _mixed_transfer(t, edge_u, edge_v, suv, fuv):
    """Mixed derivative of F_lm S_rs^(tensor 3) acting on a fixed, permuted source."""
    s, su, sv = edge_u.overlap, edge_u.doverlap, edge_v.doverlap
    if np.shape(suv) != s.shape:
        raise ValueError("Mixed intertriplet overlap has incompatible dimensions.")
    value = fuv * _transform(t,s,s,s)
    for leg in range(3):
        factors = [s,s,s]
        factors[leg] = su
        value += edge_v.dfock * _transform(t,*factors)
        factors[leg] = sv
        value += edge_u.dfock * _transform(t,*factors)
        factors[leg] = suv
        value += edge_u.fock * _transform(t,*factors)
        factors[leg] = su
        for other in range(3):
            if other != leg:
                factors[other] = sv
                value += edge_u.fock * _transform(t,*factors)
                factors[other] = s
    return value


class TriplesHessianColumn:
    """Stationary curvature of ordinary (T) for one fixed column direction.

    Prepare the row and column with triples_gradient_direction in one common
    transported frame. The row needs no triples solve. Mixed inputs must use
    the same frame and include the lower-order model's response.
    """
    def __init__(self, direction, *, tolerance=1.e-10):
        self.direction = direction
        self.tangent = solve_triples_tangent(direction,tolerance=tolerance)

    def relaxation(self, row):
        """Terms in d_v L_u from relaxing the right and left triples only."""
        if row.wavefunction is not self.direction.wavefunction or row.edges.keys() != self.direction.edges.keys():
            raise ValueError("Triples Hessian directions require the same reference wavefunction and triplets.")
        return sum(np.vdot(row.left_source[r],self.tangent.amplitudes[r])
                   + np.vdot(self.tangent.adjoints[r],row.right_source[r]) for r in row.edges)

    def element(self, row, occupied_fock_uv, mixed_row, mixed_overlap):
        """Contract H_uv. Providers supply projected mixed inputs only on native rows/edges.

        mixed_row(r) returns TriplesMixedRow. mixed_overlap(r,s) returns S_rs^uv.
        occupied_fock_uv is an active LMO matrix or a callable (i,j) -> F_ij^uv.
        """
        result = self.relaxation(row)
        column = self.direction
        no = column.wavefunction.variable('DLPNO RESPONSE ACTIVE LMOS').np.shape[1]
        if callable(occupied_fock_uv):
            occupied = occupied_fock_uv
        else:
            if np.shape(occupied_fock_uv) != (no,no):
                raise ValueError("Mixed occupied Fock block has incompatible dimensions.")
            occupied = lambda i,j: occupied_fock_uv[i,j]
        for r in row.edges:
            t,z = row.amplitudes[r],row.adjoints[r]
            wuv,vuv,fuv = mixed_row(r)
            if np.shape(wuv) != t.shape or np.shape(vuv) != t.shape or np.shape(fuv) != (len(t),len(t)):
                raise ValueError(f"Mixed moment or virtual Fock block has incompatible dimensions for {r}.")
            result += np.vdot(_energy_moment(r,vuv),t)
            residual = wuv + _local_fock_action(fuv,t,sum(occupied(a,a) for a in r))
            if len(row.edges[r]) != len(column.edges[r]):
                raise ValueError(f"Triples Hessian directions use different edges for {r}.")
            for eu,ev in zip(row.edges[r],column.edges[r]):
                if (eu.source,eu.axes,eu.occupied) != (ev.source,ev.axes,ev.occupied):
                    raise ValueError(f"Inconsistent occupied-slot replacement in triplet {r}.")
                q = eu.source
                residual -= _mixed_transfer(row.amplitudes[q].transpose(eu.axes),eu,ev,
                                            mixed_overlap(r,q),occupied(*eu.occupied))
            result += np.vdot(z,residual)
        if not np.isfinite(result):
            raise ValueError("Nonfinite stationary triples curvature.")
        return result

    def ccsd_stationary_element(self, row, occupied_fock_uv, mixed_row, mixed_overlap,
                                ccsd_adjoint, mixed_residual):
        """Eliminate mixed CCSD amplitudes from the isolated triples curvature.

        All mixed inputs follow the same affine CCSD trial path in moving pair
        coordinates. Rebuild TNO projectors along this path: their amplitude
        dependence is included in the native triples source. mixed_residual is
        the pair of transformed right residual derivatives on that trial path.
        """
        if ccsd_adjoint.wavefunction is not self.direction.wavefunction:
            raise ValueError("The CCSD adjoint and triples column require the same reference.")
        return (self.element(row,occupied_fock_uv,mixed_row,mixed_overlap)
                + ccsd_adjoint.correction(*mixed_residual))
