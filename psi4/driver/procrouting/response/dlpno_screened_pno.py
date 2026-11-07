"""Analytic two-stage PNO response with native retained-pair local domains.

Inputs are total analytic value/u/v/uv blocks at one geometry. Providers supply
local first/final PNO frames, coupled LMP2 response and both PNO-loss curvatures.
They do not assemble a screened CCSD(T) Hessian. The AO Schwarz screen must be
off because the native snapshot does not export its shell mask. No cutoff or
rank is inferred here.
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


class ScreenedPNOResponse(NamedTuple):
    first_response: ScreenedFirstPNOResponse
    initial: dict
    final: dict
    final_selection: dict
    edges: dict
    pair_overlap: object
    mixed_source: dict
    residual_errors: tuple
    loss: MixedBlock
    stationary_loss: float


def screened_pno_response(wfn, overlap, fock, lmo, occupied_action, local_df):
    """Propagate local first PNOs through coupled LMP2 and final selection.

    Inputs and the fixed-branch contract are those of screened_first_pno_response.
    The native first-space frames and screened response edges define the LMP2
    chart and graph. Both first amplitudes and the mixed amplitude use native
    right solves; the independent moving-frame residual verifies all orders.
    Includes both PNO-loss curvatures, with a separate stationary adjoint check.
    This supplies PNO molecular blocks, not a complete screened CCSD(T) Hessian.
    """
    from psi4 import core
    from .dlpno_pno_curvature import (coupled_lmp2_curvature, lmp2_residual,
        lmp2_mixed_residual, mixed_pno_selection, pno_truncation_energy, PNOTruncationCurvature)
    first = screened_first_pno_response(wfn,overlap,fock,lmo,occupied_action,local_df)
    def native(key):
        return {p:x.np.copy() for p,x in core.dlpno_lmp2_blocks(wfn,key).items()}
    frames, final_frames, amplitudes = (native(k) for k in ('FRAME','FINAL_FRAME','AMPLITUDE'))
    if not (first.pairs.keys() == frames.keys() == final_frames.keys() == amplitudes.keys()):
        raise ValueError('First-PNO snapshot and retained LMP2 graph disagree.')
    foo = first.occupied_fock
    initial, final = {}, {}
    for p,out in first.pairs.items():
        i,j = p
        Y = out.selection.frame
        np.testing.assert_allclose(Y.value,frames[p],atol=2.e-8,rtol=0.,
                                   err_msg='Native first-space LMP2 chart replay')
        # Project the already refitted/masked PAO exchange via its moving
        # orthonormal parent. A new full-AO L-Q-Y contraction would lose masks.
        C = mixed_product('ma,mn,nb->ab',out.parent,overlap,Y)
        initial[p] = dict(exchange=out.exchange,amplitudes=out.amplitudes,fock=out.fock,
            fov=mixed_product('m,mn,na->a',lmo.slice((slice(None),i)),fock,out.parent),
            occupied_energy=foo.slice((i,i)),rank=Y.value.shape[1])
        final[p] = dict(exchange=mixed_product('ma,mn,nb->ab',C,out.exchange,C),
            fock=mixed_product('ma,mn,nb->ab',Y,fock,Y),
            fov=mixed_product('m,mn,na->a',lmo.slice((slice(None),i)),fock,Y),
            occupied_energy=foo.slice((i,i)),rank=final_frames[p].shape[1])
    matrix = core.Matrix.from_array
    def tangent(k):
        return {p:x.np.copy() for p,x in core.dlpno_lmp2_tangent(wfn,
            {p:matrix(x.selection.frame[k]) for p,x in first.pairs.items()},
            {p:matrix(x['exchange'][k]) for p,x in final.items()},
            matrix(overlap[k]),matrix(fock[k]),matrix(foo[k])).items()}
    tu,tv = tangent(1),tangent(2)
    for p,t in amplitudes.items():
        final[p]['amplitudes'] = MixedBlock(t,tu[p],tv[p],np.zeros_like(t))
    edges = {p:x.astype(int) for p,x in native('RESPONSE_EDGES').items()}
    @lru_cache(None)
    def pair_overlap(p,q):
        return mixed_product('ma,mn,nb->ab',first.pairs[p].selection.frame,overlap,
                             first.pairs[q].selection.frame)
    residual = lmp2_residual(final,foo,edges,pair_overlap)
    errors = [max(np.max(abs(r[k]),initial=0.) for r in residual.values()) for k in (0,1,2)]
    if not np.isfinite(errors).all() or max(errors)>2.e-9:
        raise ValueError('Screened LMP2 value/first response fails the moving-frame residual.')
    source = lmp2_mixed_residual(final,foo,edges,pair_overlap)
    mixed = coupled_lmp2_curvature(wfn,final,foo,edges,pair_overlap)
    final = {p:dict(x,amplitudes=mixed[p]) for p,x in final.items()}
    residual = lmp2_residual(final,foo,edges,pair_overlap)
    errors.append(max(np.max(abs(r.uv),initial=0.) for r in residual.values()))
    selection = {p:mixed_pno_selection(first.pairs[p].selection.frame,overlap,mixed[p],Y,
                                       diagonal=(p[0]==p[1])) for p,Y in final_frames.items()}
    loss = pno_truncation_energy(initial,final)
    stationary = PNOTruncationCurvature(wfn).element(initial,final,source)
    if not np.isclose(loss.uv,stationary,atol=2.e-10,rtol=0.):
        raise ValueError('Screened direct and stationary PNO-loss curvatures disagree.')
    return ScreenedPNOResponse(first,initial,final,selection,edges,pair_overlap,source,
                               tuple(float(x) for x in errors),loss,float(stationary))
