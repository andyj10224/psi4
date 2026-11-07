"""Two-stage RHF PNO truncation-energy curvature on a fixed selection branch.

Inputs are total mixed derivatives in an orthonormal parent chart. The direct
contraction accepts mixed LMP2 amplitudes; the stationary route replaces them
with a native loss adjoint and a mixed residual source. Lower-order molecular
response inputs are still supplied by the caller.
"""
import numpy as np

from .dlpno_triples_curvature import MixedBlock, _inverse, _mixed_projector, mixed_product


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
