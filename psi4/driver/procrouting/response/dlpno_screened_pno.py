"""Analytic first-PNO response with the native retained-pair local domains.

Inputs are total analytic value/u/v/uv blocks at one geometry. This supplies
first-PNO frames and local fitting data; it does not assemble final PNOs or a
screened CCSD(T) Hessian. The AO Schwarz screen must be off because the native
snapshot does not export its shell mask. No cutoff or rank is inferred here.
"""
from functools import lru_cache
from typing import NamedTuple
import numpy as np

from .dlpno_frame_curvature import mixed_paos
from .dlpno_pno_curvature import first_pno_curvature, mixed_pair_exchange, mixed_pair_three_center
from .dlpno_triples_curvature import MixedBlock, mixed_product


class ScreenedFirstPNOResponse(NamedTuple):
    pairs: dict
    paos: dict
    panels: dict
    exchange: dict
    occupied_fock: MixedBlock
    replay_errors: dict


def local_pair_panels(overlap, lmo, pao, pair, pao_ids, auxiliary_centers, domains, eri):
    """Differentiate the native OV refit on each auxiliary center's AO lists.

    eri is a raw AO three-center MixedBlock with the pair's auxiliary order.
    Missing occupied or PAO labels are structural zeros on this fixed branch.
    The OO/VV PAO-pair mask is not part of an OV first-PNO exchange panel.
    """
    centers = np.asarray(auxiliary_centers, dtype=int)
    naux, npao = len(centers), len(pao_ids)
    if any(x.shape != (naux, len(overlap.value), len(overlap.value)) for x in eri):
        raise ValueError('First-PNO raw three-center input has inconsistent AO/auxiliary dimensions.')
    result = []
    for label in pair:
        parts = [np.zeros((naux, npao)) for _ in range(4)]
        for atom in sorted(set(centers)):
            if label not in domains['OCC'][atom]:
                continue
            rows = np.flatnonzero(centers == atom)
            left, right = (np.asarray(domains[key][atom], dtype=int) for key in ('AO_OCC','AO_VIR'))
            panel = mixed_pair_three_center(overlap, lmo.slice((slice(None),label)), pao,
                                           left, right, eri.slice(np.ix_(rows,left,right)))
            present = np.isin(pao_ids, domains['PAO'][atom])
            for target, value in zip(parts, panel):
                target[rows] = value * present
        result.append(MixedBlock(*parts))
    return tuple(result)


def screened_first_pno_response(wfn, overlap, fock, lmo, occupied_action, local_df):
    """Construct analytic retained-pair first-PNO frames from native selections.

    local_df(auxiliary_ids) returns (metric, raw AO three-center integrals),
    both MixedBlocks in that exact auxiliary order. occupied_action(panel)
    applies the total spatial occupied density and its derivatives, including
    frozen core. LMO columns must have the native active occupied labeling.
    DLPNO_KEEP_LMP2_RESPONSE must have been enabled in the forward calculation.
    """
    from psi4 import core
    domains, native = core.dlpno_first_pno_state(wfn)
    if wfn.variable('DLPNO FIRST PNO AO INTS TOL') != 0.:
        raise ValueError('Screened first-PNO molecular inputs require the AO Schwarz screen off.')
    reference = wfn.reference_wavefunction()
    np.testing.assert_allclose(overlap.value, reference.S().np, atol=2.e-10, rtol=0.)
    np.testing.assert_allclose(fock.value, reference.Fa().np, atol=2.e-10, rtol=0.)
    np.testing.assert_allclose(lmo.value, wfn.variable('DLPNO RESPONSE ACTIVE LMOS').np,
                               atol=2.e-10, rtol=0.)
    basis = reference.get_basisset('DF_BASIS_CC')
    foo = mixed_product('mi,mn,nj->ij', lmo, fock, lmo)
    @lru_cache(None)
    def paos(ids):
        return mixed_paos(occupied_action, overlap, np.asarray(ids, dtype=int))[1]
    pairs, panels, exchange, normalized = {}, {}, {}, {}
    errors = dict(panel=0., amplitude=0., occupations=0., frame=0.)
    def verify(label, value, expected, tolerance=2.e-10):
        error = float(np.max(abs(value-expected), initial=0.))
        errors[label] = max(errors[label], error)
        np.testing.assert_allclose(value, expected, atol=tolerance, rtol=0.,
                                   err_msg='Native first-PNO '+label+' replay')
    if len(domains['PAIRS']) != len(native):
        raise ValueError('First-PNO native snapshot has an inconsistent pair graph.')
    for pair, ids, aux in zip(domains['PAIRS'], domains['PAIR_PAO'], domains['PAIR_AUX']):
        i,j = p = tuple(pair)
        if i > j or p in pairs:
            raise ValueError('First-PNO state must contain unique pairs i <= j.')
        saved = native[p]
        N = paos(tuple(ids))
        J,Q = local_df(tuple(aux))
        centers = [basis.function_to_center(q) for q in aux]
        qi,qj = local_pair_panels(overlap,lmo,N,p,ids,centers,domains,Q)
        verify('panel', qi.value, saved['QI'].np)
        verify('panel', qj.value, saved['QJ'].np)
        K = mixed_pair_exchange(J,qi,qj)
        occ = MixedBlock(*(x[i,i]+x[j,j] for x in foo))
        out = first_pno_curvature(N,overlap,fock,K,occ,saved['PARENT'].np,
                                  saved['FRAME'].np,diagonal=(i==j))
        verify('amplitude', out.amplitudes.value, saved['AMPLITUDE'].np)
        verify('occupations', out.selection.occupations, saved['OCCUPATIONS'].np[:,0])
        # Same reference-frame criterion as mixed_pno_selection. Normalizing
        # a numerically conditioned native parent can change its last digits.
        verify('frame', out.selection.frame.value, saved['FRAME'].np, 2.e-8)
        pairs[p], normalized[p], panels[p], exchange[p] = out, N, (qi,qj), K
    return ScreenedFirstPNOResponse(pairs,normalized,panels,exchange,foo,errors)
