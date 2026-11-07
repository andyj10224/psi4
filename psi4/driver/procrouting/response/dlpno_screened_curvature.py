"""Analytic dipole and excluded semicanonical-MP2 energy curvature.

The caller supplies analytic molecular inputs at one geometry. Actual native
crude-stage domains and Cholesky parents are retained; no displaced quantities
are used. The mixed AO integral delivery currently requires the Schwarz screen
off. This restriction does not apply to the native analytic gradient.
"""
from functools import lru_cache
import numpy as np
from .dlpno_triples_curvature import MixedBlock, mixed_product
from .dlpno_frame_curvature import mixed_paos
from .dlpno_pno_curvature import (
    _polar_normalization, mixed_pair_three_center, mixed_pair_exchange, semicanonical_mp2)


def _sum(*blocks):
    return MixedBlock(*(sum(parts) for parts in zip(*blocks)))


def _scale(a, block):
    return MixedBlock(*(a*x for x in block))


def _power(block, power):
    x, u, v, uv = block
    if np.any(x <= 0.):
        raise ValueError('Expected a positive squared centroid distance.')
    return MixedBlock(x**power, power*x**(power-1)*u, power*x**(power-1)*v,
                      power*x**(power-1)*uv + power*(power-1)*x**(power-2)*u*v)


def mixed_local_parent(pao, overlap, parent):
    """Transport the native fixed, full-rank Cholesky pivot span."""
    X = np.asarray(parent)
    if (X.ndim != 2 or X.shape[0] != pao.value.shape[1] or not X.shape[1] or
            np.count_nonzero(np.any(X != 0., axis=1)) != X.shape[1]):
        raise ValueError('Screened curvature requires a full-rank selected PAO pivot block.')
    Y = MixedBlock(*(x @ X for x in pao))
    norm = _polar_normalization(mixed_product('ma,mn,nb->ab', Y, overlap, Y))
    coefficients = MixedBlock(*(X @ x for x in norm))
    return mixed_product('ma,ab->mb', pao, coefficients), coefficients


def rectangular_mp2(exchange, left_fock, right_fock, occupied_sum):
    """Mixed rectangular Sylvester solve: o t - F_i t - t F_j = K."""
    ei, Ui = np.linalg.eigh(left_fock.value)
    ej, Uj = np.linalg.eigh(right_fock.value)
    gaps = occupied_sum.value - ei[:, None] - ej[None, :]
    if np.any(abs(gaps) <= 1.e-12):
        raise ValueError('Singular dipole pair denominator.')
    def solve(rhs):
        return Ui @ ((Ui.T @ rhs @ Uj)/gaps) @ Uj.T
    def action(k, t):
        return occupied_sum[k]*t - left_fock[k]@t - t@right_fock[k]
    t = solve(exchange.value)
    u = solve(exchange.u-action(1,t))
    v = solve(exchange.v-action(2,t))
    return MixedBlock(t, u, v, solve(exchange.uv-action(3,t)-action(1,v)-action(2,u)))


def screened_pair_curvature(wfn, overlap, fock, lmo, occupied_action, dipoles, local_df):
    """Return scalar MixedBlocks for both correction energies, gradients and H_uv.

    local_df(aux) returns (Coulomb metric, raw AO ERIs) in that auxiliary order.
    It is called only for eliminated SC-MP2 pairs. dipoles contains the three
    electronic AO dipole MixedBlocks in the same convention as the forward code.
    occupied_action applies the total occupied density, including frozen core.
    """
    from psi4 import core
    if core.get_option('DLPNO', 'DLPNO_AO_INTS_TOL') != 0.:
        raise ValueError('Mixed screened-pair AO delivery requires the Schwarz screen off.')
    domains, state = core.dlpno_screened_pair_state(wfn)
    C = mixed_product
    foo = C('mi,mn,nj->ij', lmo, fock, lmo)
    basis = wfn.reference_wavefunction().get_basisset('DF_BASIS_CC')
    @lru_cache(None)
    def paos(ids):
        return mixed_paos(occupied_action, overlap, np.array(ids, dtype=int))[1]
    def occ(i,j):
        return _sum(foo.slice((i,i)), foo.slice((j,j)))

    sc = MixedBlock(*(np.array(0.) for _ in range(4)))
    for i,j in domains['SCMP2_PAIRS']:
        if i >= j:
            raise ValueError('Expected each excluded SC-MP2 pair exactly once, i < j.')
        p = state[i,j]
        ids = tuple(p['PAOS'].np[:,0].astype(int))
        aux = tuple(p['AUX'].np[:,0].astype(int))
        N = paos(ids)
        J, Q = local_df(aux)
        panels = []
        for label in (i,j):
            parts = [np.zeros((len(aux), len(ids))) for _ in range(4)]
            for atom in sorted({basis.function_to_center(int(q)) for q in aux}):
                rows = np.array([k for k,q in enumerate(aux) if basis.function_to_center(int(q)) == atom])
                left, right = (np.array(domains[key][atom], dtype=int) for key in ('AO_OCC','AO_VIR'))
                panel = mixed_pair_three_center(overlap, lmo.slice((slice(None),label)), N,
                                               left, right, Q.slice(np.ix_(rows,left,right)))
                for target, value in zip(parts, panel):
                    target[rows] = value
            panels.append(MixedBlock(*parts))
        for key, panel in zip(('QI','QJ'), panels):
            np.testing.assert_allclose(panel.value, p[key].np, atol=2.e-10, rtol=0.)
        K0 = mixed_pair_exchange(J, *panels)
        Y, X = mixed_local_parent(N, overlap, p['PARENT'].np)
        F = C('ma,mn,nb->ab', Y, fock, Y)
        K = C('ma,mn,nb->ab', X, K0, X)
        t = semicanonical_mp2(K, F, occ(i,j))
        spin = MixedBlock(*(2*x-x.T for x in t))
        sc = _sum(sc, _scale(2., C('ab,ab->', K, spin)))

    @lru_cache(None)
    def dipole_orbital(i):
        N = paos(tuple(domains['DIPOLE_PAO'][i]))
        Y, _ = mixed_local_parent(N, overlap, state[i,i]['PARENT'].np)
        L = lmo.slice((slice(None),i))
        center = [C('m,mn,n->',L,op,L) for op in dipoles]
        moment = [C('m,mn,na->a',L,op,Y) for op in dipoles]
        stack = lambda xs: MixedBlock(*(np.stack(parts) for parts in zip(*xs)))
        return stack(center), stack(moment), C('ma,mn,nb->ab', Y, fock, Y)
    dip = MixedBlock(*(np.array(0.) for _ in range(4)))
    eye = MixedBlock(np.eye(3), *[np.zeros((3,3)) for _ in range(3)])
    for i,j in domains['DIPOLE_PAIRS']:
        if i >= j:
            raise ValueError('Expected each excluded dipole pair exactly once, i < j.')
        ri, di, Fi = dipole_orbital(i)
        rj, dj, Fj = dipole_orbital(j)
        R = _sum(ri, _scale(-1.,rj))
        r2 = C('k,k->', R, R)
        A = _sum(C(',kl->kl', _power(r2,-1.5), eye),
                 _scale(-3., C(',k,l->kl',_power(r2,-2.5),R,R)))
        K = C('ka,kl,lb->ab',di,A,dj)
        t = rectangular_mp2(K,Fi,Fj,occ(i,j))
        dip = _sum(dip, _scale(4.,C('ab,ab->',K,t)))
    for label, value in (('SC-LMP2 PAIR',sc), ('DIPOLE',dip)):
        np.testing.assert_allclose(value.value, wfn.variable(f'DLPNO {label} ENERGY'), atol=2.e-10, rtol=0.)
        if not np.isfinite(value).all():
            raise ValueError('Nonfinite screened-pair curvature.')
    return {'SCMP2':sc, 'DIPOLE':dip}
