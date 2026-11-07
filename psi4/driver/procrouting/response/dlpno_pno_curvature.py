"""Two-stage RHF PNO truncation-energy curvature on a fixed selection branch.

Inputs are total mixed derivatives in an orthonormal parent chart. In particular,
final-stage coupled-LMP2 amplitudes require their mixed derivative. This module
is a local contraction kernel, not a molecular response solver.
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
