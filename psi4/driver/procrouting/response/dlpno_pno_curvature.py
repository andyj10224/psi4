"""Two-stage RHF PNO truncation-energy curvature on a fixed selection branch.

Local fitting and first-PNO kernels propagate total mixed derivatives into
orthonormal parent charts. The direct loss contraction accepts mixed LMP2
amplitudes; the stationary route replaces them with a native loss adjoint and
a mixed residual source. Lower-order molecular inputs are supplied by the caller.
"""
from typing import NamedTuple

import numpy as np

from .dlpno_frame_curvature import mixed_metric_solve
from .dlpno_triples_curvature import MixedBlock, _inverse, _mixed_projector, mixed_product


def mixed_pair_three_center(overlap, lmo, pao, left, right, eri):
    """Differentiate a fixed-domain occupied refit and its raw DF panel.

    eri has indices (Q, left AO, right AO), with the forward shell mask held
    fixed. pao contains AO columns for the pair domain. Auxiliary-center
    callers may concatenate panels; this kernel does not select domains.
    """
    left, right = np.asarray(left), np.asarray(right)
    n = overlap.value.shape[0]
    for domain in (left, right):
        if (domain.ndim != 1 or domain.dtype.kind not in 'iu' or
                np.any(domain < 0) or np.any(domain >= n) or len(np.unique(domain)) != len(domain)):
            raise ValueError("Expected unique AO indices for a fixed fitting domain.")
    if any(x.shape[1:] != (len(left), len(right)) for x in eri):
        raise ValueError("Three-center panel disagrees with its AO fitting domains.")
    rhs = mixed_product('mn,n->m', overlap.slice((left, slice(None))), lmo)
    fitted = mixed_metric_solve(overlap.slice(np.ix_(left, left)), rhs)
    return mixed_product('m,Qmn,na->Qa', fitted, eri, pao.slice((right, slice(None))))


def mixed_pair_exchange(metric, qi, qj):
    """Mixed fitted exchange K = qi^T J^-1 qj, using a metric panel solve."""
    return mixed_product('Qa,Qb->ab', qi, mixed_metric_solve(metric, qj))


def _polar_normalization(gram):
    """Mixed inverse square root of a numerically near-identity Gram matrix.

    Solve the square-root Sylvester equations at the actual reference Gram.
    Replacing it by exact identity can amplify small parent conditioning errors
    in a large tangent and violate differentiated metric orthonormality.
    """
    eye = np.eye(len(gram.value))
    if not np.allclose(gram.value, eye, atol=2.e-8, rtol=0):
        raise ValueError("PNO reference frame must be metric orthonormal.")
    eig, rotation = np.linalg.eigh(gram.value)
    roots = np.sqrt(eig)
    root = (rotation*roots) @ rotation.T
    def solve(rhs):
        return rotation @ ((rotation.T @ rhs @ rotation)/(roots[:,None]+roots[None,:])) @ rotation.T
    u,v = solve(gram.u),solve(gram.v)
    uv = solve(gram.uv-u@v-v@u)
    return _inverse(MixedBlock(root,u,v,uv))


class PNOSubspaceCurvature(NamedTuple):
    frame: MixedBlock
    projector: MixedBlock
    occupations: np.ndarray
    boundary_gap: float


def mixed_pno_selection(parent, overlap, amplitudes, reference_frame, *, diagonal=False):
    """Retained projector and fixed-reference polar frame in a moving parent.

    The reference frame fixes the actual forward rank and chart. Only the
    retained/discarded occupation gap is divided by; degeneracies within either
    space are harmless. Projectors are stored in the small parent, not AO space.
    """
    n, rank = parent.value.shape[1], reference_frame.shape[1]
    if (not 0 < rank <= n or reference_frame.shape[0] != parent.value.shape[0] or
            any(x.shape != (n, n) for x in amplitudes)):
        raise ValueError("Invalid moving PNO parent, amplitudes or retained frame.")
    gram = mixed_product('ma,mn,nb->ab', parent, overlap, parent)
    if (not np.allclose(gram.value, np.eye(n), atol=2.e-8, rtol=0) or
            any(not np.allclose(x, 0., atol=2.e-8, rtol=0) for x in gram[1:])):
        raise ValueError("PNO parent response is not metric orthonormal.")
    spin = MixedBlock(*(2*x-x.T for x in amplitudes))
    left = mixed_product('ac,bc->ab', spin, amplitudes)
    right = mixed_product('ca,cb->ab', spin, amplitudes)
    density = MixedBlock(*((a+b)/(1+diagonal) for a, b in zip(left, right)))
    projector = _mixed_projector(density, rank)
    occupations = np.linalg.eigvalsh(density.value)[::-1]
    gap = occupations[rank-1]-occupations[rank] if rank < n else np.inf
    # P_AO S0 Y0 using only rectangular AO/parent panels.
    anchor = overlap.value @ reference_frame
    projected = mixed_product('ma,ab,bc->mc', parent, projector,
                              MixedBlock(*(x.T @ anchor for x in parent)))
    if not np.allclose(projected.value, reference_frame, atol=2.e-8, rtol=0):
        raise ValueError("Selected PNO space disagrees with the forward reference frame.")
    norm = _polar_normalization(mixed_product('ma,mn,nb->ab', projected, overlap, projected))
    frame = mixed_product('ma,ab->mb', projected, norm)
    if any(not np.isfinite(x).all() for x in (*frame, *projector)):
        raise ValueError("Nonfinite mixed PNO selection response.")
    return PNOSubspaceCurvature(frame, projector, occupations, gap)


class FirstPNOCurvature(NamedTuple):
    parent: MixedBlock
    exchange: MixedBlock
    fock: MixedBlock
    amplitudes: MixedBlock
    selection: PNOSubspaceCurvature


def _moving_pao_parent(pao, overlap, transform):
    """Fixed pivot span in a chart with no first-order internal rotation.

    Large, arbitrary within-parent rotations in a conditioned local PAO span
    cause cancellation in second derivatives. Choose the chart at the base
    point by differentiated orthonormality, applying the SAME coefficients to
    exchange and Fock blocks. The final retained PNO chart remains fixed by its
    own reference projector, so this internal choice changes no observable.
    """
    B0 = pao.value @ transform
    metric = B0.T @ overlap.value @ B0
    if not np.allclose(metric, np.eye(len(metric)), atol=2.e-8, rtol=0.):
        raise ValueError('First-PNO reference parent is not metric orthonormal.')
    eig,rotation = np.linalg.eigh(metric)
    X = transform @ ((rotation*eig**(-.5)) @ rotation.T)
    B,Bu,Bv,Buv = (x @ X for x in pao)
    S,Su,Sv,Suv = overlap
    metric = B.T @ S @ B
    cu = -np.linalg.solve(metric, B.T @ S @ Bu + .5*B.T @ Su @ B)
    cv = -np.linalg.solve(metric, B.T @ S @ Bv + .5*B.T @ Sv @ B)
    Yu,Yv = Bu+B@cu,Bv+B@cv
    bare = Buv+Bu@cv+Bv@cu
    residual = (bare.T@S@B + B.T@S@bare + Yu.T@S@Yv + Yv.T@S@Yu +
                Yu.T@Sv@B + B.T@Sv@Yu + Yv.T@Su@B + B.T@Su@Yv + B.T@Suv@B)
    cuv = -.5*np.linalg.solve(metric, residual)
    return MixedBlock(B,Yu,Yv,bare+B@cuv), MixedBlock(X,X@cu,X@cv,X@cuv)


def first_pno_curvature(pao, overlap, fock, exchange, occupied_sum, parent,
                        reference_frame, *, diagonal=False):
    """Mixed first-PNO response on a fixed full-rank Cholesky-pivot branch.

    parent is the forward PAO-to-orthonormal-parent transform, including its
    selected pivot rows. A metric-orthonormal chart transports their fixed
    span; a Sylvester solve differentiates MP2
    without semicanonical eigenvector derivatives or within-parent energy gaps.
    exchange is expressed in the normalized PAO domain, before this transform.
    """
    X = np.asarray(parent)
    if X.ndim != 2 or X.shape[0] != pao.value.shape[1] or X.shape[1] == 0:
        raise ValueError("Invalid first-PNO parent transform.")
    if np.count_nonzero(np.any(X != 0., axis=1)) != X.shape[1]:
        raise ValueError("First-PNO curvature requires a full-rank selected pivot span.")
    moving, coefficients = _moving_pao_parent(pao,overlap,X)
    F = mixed_product('ma,mn,nb->ab', moving, fock, moving)
    K = mixed_product('ma,mn,nb->ab', coefficients, exchange, coefficients)
    t = semicanonical_mp2(K, F, occupied_sum)
    selection = mixed_pno_selection(moving, overlap, t, reference_frame, diagonal=diagonal)
    return FirstPNOCurvature(moving, K, F, t, selection)


def semicanonical_mp2(exchange, fock, occupied_sum):
    """Differentiate (o_i+o_j) t - F t - t F = K without a pair Kronecker matrix."""
    eps, rotation = np.linalg.eigh(fock.value)
    gaps = occupied_sum.value - eps[:, None] - eps[None, :]
    if np.any(abs(gaps) <= 1.e-12):
        raise ValueError("Singular semicanonical MP2 denominator in PNO curvature.")

    def solve(rhs):
        return rotation @ ((rotation.T @ rhs @ rotation) / gaps) @ rotation.T

    def action(index, t):
        return occupied_sum[index] * t - fock[index] @ t - t @ fock[index]

    t = solve(exchange.value)
    tu = solve(exchange.u - action(1, t))
    tv = solve(exchange.v - action(2, t))
    tuv = solve(exchange.uv - action(3, t) - action(1, tv) - action(2, tu))
    return MixedBlock(t, tu, tv, tuv)


def pno_truncation_loss(exchange, amplitudes, fock, fov, occupied_energy, rank, *, diagonal):
    """One ordered orientation of the native full-minus-retained pair energy.

    Includes the diagonal-pair singles loss. The denominator mask must be fully
    active; near-zero denominators are rejected rather than silently changing
    the native screened inverse. No unoccupied/unretained rank is guessed.
    """
    n = amplitudes.value.shape[0]
    if not isinstance(rank, (int, np.integer)) or not 0 < rank <= n:
        raise ValueError("Invalid retained PNO rank.")
    if any(not np.allclose(x,x.T,atol=1.e-10,rtol=0.) for x in fock):
        raise ValueError("PNO Fock inputs must be symmetric in an orthonormal chart.")
    for block in (exchange, amplitudes, fock):
        if any(np.shape(x) != (n, n) or not np.isfinite(x).all() for x in block):
            raise ValueError("Invalid square PNO curvature input.")
    spin = MixedBlock(*(2*x-x.T for x in amplitudes))
    left = mixed_product('ac,bc->ab', spin, amplitudes)
    right = mixed_product('ca,cb->ab', spin, amplitudes)
    density = MixedBlock(*((a+b)/(1+diagonal) for a,b in zip(left,right)))
    projector = _mixed_projector(density, rank)
    full = mixed_product('ab,ab->', exchange, spin)
    kept = mixed_product('ab,ac,cd,db->', exchange, projector, spin, projector)
    loss = MixedBlock(*(a-b for a,b in zip(full,kept)))
    if diagonal:
        # Polar-align the selected eigenvectors to the reference selected frame.
        _, eig = np.linalg.eigh(density.value)
        reference = eig[:, n-rank:]
        selected = MixedBlock(*(p @ reference for p in projector))
        gram = mixed_product('ma,mb->ab', selected, selected)
        eye = np.eye(rank)
        norm = MixedBlock(eye, -.5*gram.u, -.5*gram.v,
                          -.5*gram.uv + .375*(gram.u@gram.v+gram.v@gram.u))
        selected = mixed_product('ma,ab->mb', selected, norm)
        small_fock = mixed_product('ma,mn,nb->ab', selected, fock, selected)
        small_fov = mixed_product('ma,m->a', selected, fov)

        def singles(F, f):
            gap = MixedBlock(*(o*np.eye(len(F.value))-a for o,a in zip(occupied_energy,F)))
            if np.min(abs(np.linalg.eigvalsh(gap.value))) <= 1.e-12:
                raise ValueError("Screened singles denominator requires spectral-mask response.")
            return mixed_product('a,ab,b->', f, _inverse(gap), f)

        large, small = singles(fock,fov), singles(small_fock,small_fov)
        loss = MixedBlock(*(a+b-c for a,b,c in zip(loss,large,small)))
    if not np.isfinite(loss).all():
        raise ValueError("Nonfinite PNO truncation curvature.")
    return loss


def pno_truncation_energy(initial, final):
    """Sum both native selection losses over unique pairs with multiplicities 1 and 2.

    Each mapping entry holds keyword arguments for pno_truncation_loss except
    diagonal. Initial amplitudes should come from semicanonical_mp2; final ones
    are the converged coupled LMP2 tensor. Parent motion belongs in the input
    derivatives at both stages. Omitted stages have zero loss.
    """
    total = np.zeros(4)
    for stage in (initial, final):
        for (i,j), inputs in stage.items():
            if i > j:
                raise ValueError("Expected unique ordered PNO pairs i <= j.")
            total += (1 if i == j else 2)*np.asarray(
                pno_truncation_loss(**inputs, diagonal=(i == j)))
    return MixedBlock(*total)


def lmp2_residual(final, occupied_fock, edges, overlap, *, affine_amplitudes=False):
    """LMP2 residual and its derivatives in native first-PNO frames.

    edges[p] contains ordered (k,l,f1,f2) rows from RESPONSE_EDGES. overlap(p,q)
    supplies a MixedBlock of first-PNO overlaps for unique q. No edge screen is
    recomputed here. The residual convention is K + F t + t F - o t - couplings.
    """
    def affine(t):
        return MixedBlock(t.value,t.u,t.v,np.zeros_like(t.value)) if affine_amplitudes else t
    result = {}
    for p, inputs in final.items():
        i,j = p
        t = affine(inputs['amplitudes'])
        F = inputs['fock']
        occ = MixedBlock(*(x[i,i]+x[j,j] for x in occupied_fock))
        left = mixed_product('ac,cb->ab',F,t)
        right = mixed_product('ac,cb->ab',t,F)
        diagonal = mixed_product(',ab->ab',occ,t)
        residual = MixedBlock(*(k+a+b-c for k,a,b,c in zip(inputs['exchange'],left,right,diagonal)))
        for k,l,f1,f2 in edges[p]:
            q = min(k,l),max(k,l)
            tq = affine(final[q]['amplitudes'])
            if k > l:
                tq = MixedBlock(*(x.T for x in tq))
            S = overlap(p,q)
            term = mixed_product(',ac,cd,bd->ab',occupied_fock.slice((f1,f2)),S,tq,S)
            residual = MixedBlock(*(a-b for a,b in zip(residual,term)))
        if i==j:
            residual = MixedBlock(*(.5*(x+x.T) for x in residual))
        result[p] = residual
    return result


def lmp2_mixed_residual(final, occupied_fock, edges, overlap):
    """Mixed residual source with t_uv=0, for the stationary PNO-loss curvature."""
    return {p:r.uv for p,r in lmp2_residual(
        final,occupied_fock,edges,overlap,affine_amplitudes=True).items()}


def coupled_lmp2_curvature(wfn, final, occupied_fock, edges, overlap):
    """Complete the mixed amplitudes with the native fixed-graph right solver.

    All inputs must use the native reference first-PNO frames and their smooth
    transport. The supplied amplitude uv fields are ignored. Verify the result
    with the independently assembled moving-frame residual, including both
    overlap legs and reversed source-pair orientations.
    """
    from psi4 import core
    source = lmp2_mixed_residual(final, occupied_fock, edges, overlap)
    solved = core.dlpno_lmp2_mixed_tangent(wfn, {p:core.Matrix.from_array(x) for p,x in source.items()})
    if solved.keys() != final.keys():
        raise ValueError('Mixed LMP2 response pair graph disagrees with the reference.')
    amplitudes = {p:inputs['amplitudes']._replace(uv=solved[p].np.copy()) for p,inputs in final.items()}
    complete = {p:dict(inputs,amplitudes=amplitudes[p]) for p,inputs in final.items()}
    residual = lmp2_residual(complete, occupied_fock, edges, overlap)
    error = max((np.max(abs(x.uv), initial=0.) for x in residual.values()), default=0.)
    if not np.isfinite(error) or error > 2.e-9:
        raise ValueError('Mixed LMP2 amplitudes fail the moving-frame residual check.')
    return amplitudes


class PNOTruncationCurvature:
    """Eliminate final-stage mixed LMP2 amplitudes using the native loss adjoint.

    Native sources already include ordered-pair energy multiplicities. Their
    dual uses an unweighted Frobenius sum over unique pairs, unlike CCSD Lambda.
    All inputs must be in the reference native first-PNO chart.
    """
    def __init__(self,wfn):
        from psi4 import core
        self.adjoint = {p:x.np.copy() for p,x in core.dlpno_pno_truncation_lmp2_response(wfn,True).items()}
        self.reference_energy = wfn.variable('DLPNO PNO TRUNCATION ERROR')

    def element(self,initial,final,mixed_residual):
        if final.keys()!=self.adjoint.keys() or mixed_residual.keys()!=self.adjoint.keys():
            raise ValueError('PNO-loss response pair graph disagrees with the reference.')
        affine = {}
        correction = 0.
        for p,inputs in final.items():
            t = inputs['amplitudes']
            if t.value.shape!=self.adjoint[p].shape or mixed_residual[p].shape!=t.value.shape:
                raise ValueError('PNO-loss response pair dimensions disagree with the reference.')
            affine[p] = dict(inputs,amplitudes=MixedBlock(t.value,t.u,t.v,np.zeros_like(t.value)))
            correction += np.vdot(self.adjoint[p],mixed_residual[p])
        trial = pno_truncation_energy(initial,affine)
        if not np.isclose(trial.value,self.reference_energy,rtol=1.e-9,atol=1.e-11):
            raise ValueError('Reconstructed PNO loss disagrees with the native correction.')
        value = trial.uv+correction
        if not np.isfinite(value):
            raise ValueError('Nonfinite stationary PNO-loss curvature.')
        return value
