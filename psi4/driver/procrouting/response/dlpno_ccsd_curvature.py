"""Stationary retained-pair RHF CCSD correlation-energy curvature (Jiang Eq. 45)."""

import numpy as np

from .dlpno_ccsd_response import TriplesCCSDAdjoint, ccsd_weighted_dot
from .dlpno_triples_curvature import MixedBlock, _symmetric_inverse as _inverse, mixed_product


def retained_ccsd_energy(singles, doubles, transfer, exchange, fov):
    """Return the value, u, v and uv derivatives of the native retained-pair energy.

    transfer(pair, i) maps the diagonal-pair singles into the target pair.
    exchange(pair) supplies its ordered bare exchange block, and fov(i) the
    bare occupied-virtual Fock row in its diagonal PNO frame.
    """
    energy = np.zeros(4)
    for i, single in enumerate(singles):
        energy += 2 * np.asarray(mixed_product('a,a->', fov(i), single))

    for pair, double in doubles.items():
        i, j = pair
        if i > j:
            raise ValueError("Expected unique RHF doubles pairs.")
        ti = mixed_product('ab,b->a', transfer(pair, i), singles[i])
        tj = mixed_product('ab,b->a', transfer(pair, j), singles[j])
        outer = mixed_product('a,b->ab', ti, tj)
        tau = MixedBlock(*(a + b for a, b in zip(double, outer)))
        spin_exchange = MixedBlock(*(2 * k - k.T for k in exchange(pair)))
        multiplicity = 1 if i == j else 2
        energy += multiplicity * np.asarray(mixed_product('ab,ab->', tau, spin_exchange))
    return MixedBlock(*energy)


def project_retained_ccsd_energy(lmos, pair_frames, overlap, fock, singles, doubles, pair_fit):
    """Project molecular panels and evaluate Eq. 45 in each retained local fit.

    pair_fit(pair) returns mixed blocks for the metric and raw (Q|i a), (Q|j b)
    in native pair order. A screened occupied refit must include its response.
    """
    def transfer(pair, i):
        return mixed_product('ma,mn,nb->ab', pair_frames[pair], overlap, pair_frames[i, i])

    def exchange(pair):
        metric, qi, qj = pair_fit(pair)
        return mixed_product('Qa,QR,Rb->ab', qi, _inverse(metric), qj)

    def fov(i):
        return mixed_product('m,mn,na->a', lmos.slice((slice(None), i)), fock, pair_frames[i, i])

    return retained_ccsd_energy(singles, doubles, transfer, exchange, fov)


class RetainedCCSDCurvature:
    """Isolate the retained-energy multiplier from the converged gradient Lambda."""

    def __init__(self, wfn, triples_adjoint=None):
        from psi4 import core

        if triples_adjoint is None:
            triples_adjoint = TriplesCCSDAdjoint(wfn)
        if triples_adjoint.wavefunction is not wfn:
            raise ValueError("Expected the triples adjoint from the same CCSD reference.")
        one, two = core.dlpno_gradient_ccsd_lambda(wfn)
        if len(one) != len(triples_adjoint.singles) or two.keys() != triples_adjoint.doubles.keys():
            raise ValueError("The two CCSD multipliers use different amplitude spaces.")
        self.singles = [x.np.ravel() - y for x, y in zip(one, triples_adjoint.singles)]
        self.doubles = {p: x.np - triples_adjoint.doubles[p] for p, x in two.items()}
        self.reference_energy = wfn.variable('DLPNO RETAINED CCSD CORRELATION ENERGY')

    def element(self, lmos, pair_frames, overlap, fock, singles, doubles, pair_fit, mixed_residual):
        """Evaluate a Hessian entry on the same affine CCSD path as mixed_residual.

        Each amplitude supplies its value and its u and v tangents. Other
        local inputs carry all four components of MixedBlock.
        """
        if len(singles) != len(self.singles) or doubles.keys() != self.doubles.keys():
            raise ValueError("Retained-energy inputs disagree with the reference occupied/pair list.")

        def affine(t):
            return MixedBlock(t[0], t[1], t[2], np.zeros_like(t[0]))

        trial = project_retained_ccsd_energy(
            lmos, pair_frames, overlap, fock, [affine(t) for t in singles],
            {p: affine(t) for p, t in doubles.items()}, pair_fit)
        if not np.isclose(trial.value, self.reference_energy, rtol=1.e-9, atol=1.e-11):
            raise ValueError("Reconstructed retained-pair energy disagrees with the native CCSD state.")
        value = trial.uv + ccsd_weighted_dot(self.singles, self.doubles, *mixed_residual)
        if not np.isfinite(value):
            raise ValueError("Nonfinite retained-pair CCSD curvature.")
        return value
