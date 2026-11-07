"""Audit screened-domain Hessian readiness on an interacting four-H2 chain.

The positive checks cover the sparse CCSD graph and the existing gradient.
Expected mixed-nuclear provider rejections are recorded as UNSUPPORTED, never
as a validated Hessian. Exit zero means the audit completed; --require-production
returns 2 while any production blocker remains. Run in a built Psi4 environment.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import psi4
from scipy.optimize import linear_sum_assignment

from ccsd_mixed import pack
from first_pno import raw
from first_pno_mixed import molecular_inputs
from lmp2_tangent import blocks
from localization_integration import configure
from psi4.driver.p4util.exceptions import MissingMethodError
from psi4.driver.procrouting.response.dlpno_ccsd_mixed import mixed_ccsd_residual
from psi4.driver.procrouting.response.dlpno_ccsd_response import ccsd_amplitude_curvature
from psi4.driver.procrouting.response.dlpno_orbital import assemble_cartesian_gradient
from psi4.driver.procrouting.response.dlpno_triples_curvature import MixedBlock


SCREENING = dict(t_cut_do=.01, t_cut_mkn=.001, t_cut_clmo=.01, t_cut_cpao=.01,
                 t_cut_do_ij=1.e-5, t_cut_do_uv=1.e-5, t_cut_do_pre=.03)


def constant(x):
    return MixedBlock(x, *[np.zeros_like(x) for _ in range(3)])


def canonical_branch(w, inverse_order=None):
    """Compare actual native selections after relabeling the occupied orbitals.

    The AO Schwarz mask is deliberately off, and triplet AO/fitting domains
    are complete. All local pair/refit lists and retained ranks are checked.
    """
    assert psi4.core.get_option('DLPNO', 'DLPNO_AO_INTS_TOL') == 0.
    assert psi4.core.get_option('DLPNO', 'T_CUT_DO_TRIPLES') == 0.
    assert psi4.core.get_option('DLPNO', 'T_CUT_MKN_TRIPLES') == 0.
    nocc = w.variable('DLPNO RESPONSE ACTIVE LMOS').np.shape[1]
    order = list(range(nocc)) if inverse_order is None else inverse_order
    relabel = lambda ids: tuple(sorted(int(order[i]) for i in ids))
    state = psi4.core.dlpno_response_domain_state(w)
    result = {key: tuple(tuple(row) for row in state[key])
              for key in ('AO_OCC', 'AO_VIR', 'PAO', 'PAO_PAIRS')}
    result['OCC'] = tuple(relabel(row) for row in state['OCC'])
    pairs = state['PAIRS']
    for key in ('PAIR_OCC', 'PAIR_PAO', 'PAIR_AUX'):
        result[key] = {relabel(p): relabel(row) if key == 'PAIR_OCC' else tuple(row)
                       for p, row in zip(pairs, state[key])}
    for name in ('FRAME', 'FINAL_FRAME'):
        result[name] = {relabel(p): y.shape[1] for p, y in blocks(w, name).items()}
    result['LMP2_EDGES'] = {
        relabel(p): tuple(sorted((relabel(row[:2]), relabel(row[2:]))
                                for row in edges.astype(int)))
        for p, edges in blocks(w, 'RESPONSE_EDGES').items()}
    result['TRIPLES'] = {}
    for r, row in psi4.core.dlpno_triples_response_state(w).items():
        # Repeated slots retain their multiplicity after relabeling.
        edges = tuple(sorted((int(order[l]), int(order[m]), relabel((a,b,c)))
                             for l,m,a,b,c,_ in row[5]))
        result['TRIPLES'][relabel(r)] = relabel(row[0]), tuple(row[1]), row[2].np.shape[1], edges
    return result


def residual_inputs(w):
    ref = w.reference_wavefunction()
    L = w.variable('DLPNO RESPONSE ACTIVE LMOS').np.copy()
    frames, doubles = blocks(w, 'FINAL_FRAME'), blocks(w, 'CCSD_DOUBLES')
    singles = [None] * L.shape[1]
    for p in doubles:
        domain, values, _ = psi4.core.dlpno_ccsd_df_pair_state(w, *p)
        for i, value in zip(domain, values):
            singles[i] = value.np.ravel().copy()
    base = {'lmo': L, 'fock': ref.Fa().np.copy()}
    base.update({('pno', p): x for p, x in frames.items()})
    base.update({('t1', i): x for i, x in enumerate(singles)})
    base.update({('t2', p): x for p, x in doubles.items()})
    J, Q = raw(ref, ref.get_basisset('DF_BASIS_CC'))
    inputs = (lambda key: constant(base[key]), constant(ref.S().np.copy()),
              lambda aux: (constant(J[np.ix_(aux, aux)]), constant(Q[list(aux)])))

    def evaluate(one, two):
        r1, r2 = psi4.core.dlpno_ccsd_residual_at(
            w, [psi4.core.Matrix.from_array(x[:, None]) for x in one],
            {p: psi4.core.Matrix.from_array(x) for p, x in two.items()})
        return [x.np.ravel().copy() for x in r1], {p: x.np.copy() for p, x in r2.items()}
    return base, (singles, doubles), inputs, evaluate


def check_graph(w):
    base, amplitudes, inputs, evaluate = residual_inputs(w)
    native = evaluate(*amplitudes)
    replay = mixed_ccsd_residual(w, *inputs)
    value_error = float(np.max(abs(pack(replay, 0) - pack(native))))
    np.testing.assert_allclose(pack(replay, 0), pack(native), atol=2.e-10, rtol=0.)
    # Cross-check arbitrary mixed amplitude directions against native quartic
    # extraction. Nuclear orbital/frame derivatives are not claimed by this test.
    rng = np.random.default_rng(3982)
    trial = {}
    for key, value in base.items():
        if isinstance(key, tuple) and key[0] in ('t1', 't2'):
            u, v = (rng.normal(size=value.shape)*.01 for _ in range(2))
            if key[0] == 't2' and key[1][0] == key[1][1]:
                u, v = (u+u.T)/2, (v+v.T)/2
            trial[key] = MixedBlock(value, u, v, np.zeros_like(value))
        else:
            trial[key] = constant(value)
    mixed = mixed_ccsd_residual(w, trial.__getitem__, *inputs[1:])
    def direction(k):
        return ([trial['t1', i][k] for i in range(len(amplitudes[0]))],
                {p: trial['t2', p][k] for p in amplitudes[1]})
    exact = ccsd_amplitude_curvature(evaluate, amplitudes, direction(1), direction(2))
    mixed_error = float(np.max(abs(pack(mixed, 3) - pack(exact))))
    np.testing.assert_allclose(pack(mixed, 3), pack(exact), atol=2.e-11, rtol=0.)
    # Replays restore the native converged amplitudes.
    np.testing.assert_allclose(pack(evaluate(*amplitudes)), pack(native), atol=1.e-13, rtol=0.)
    return dict(residual_value_error=value_error, amplitude_mixed_error=mixed_error)


def describe(w):
    state = psi4.core.dlpno_response_domain_state(w)
    nocc = w.variable('DLPNO RESPONSE ACTIVE LMOS').np.shape[1]
    counts = {key: [len(row) for row in state[key]] for key in
              ('AO_OCC', 'AO_VIR', 'OCC', 'PAO', 'PAIR_OCC', 'PAIR_PAO', 'PAIR_AUX')}
    counts.update(nbf=w.basisset().nbf(), naux=w.get_basisset('DF_BASIS_CC').nbf(),
                  nocc=nocc, unique_pairs=len(state['PAIRS']), complete_pairs=nocc*(nocc+1)//2)
    counts['native_domains'] = state
    assert counts['unique_pairs'] < counts['complete_pairs']
    graph = set(map(tuple, state['PAIRS']))
    # Exercise the case of an ij/ik coupling whose kj pair was discarded.
    counts['missing_kj_couplings'] = sum(
        tuple(sorted((k, j))) not in graph
        for i, j in graph for k in range(nocc) if tuple(sorted((i, k))) in graph)
    assert counts['missing_kj_couplings'] > 0
    return counts


def check_local(w, mol):
    counts = describe(w)
    assert min(counts['AO_OCC']) < counts['nbf'] and min(counts['AO_VIR']) < counts['nbf']
    assert max(counts['PAIR_AUX']) < counts['naux']
    assert max(counts['PAIR_PAO']) < counts['nbf']
    failures = {}
    def forbidden(*args):
        raise AssertionError('Unsupported mixed provider consumed full-domain inputs.')
    for label, operation in (
        ('ccsd_mixed', lambda: mixed_ccsd_residual(w, forbidden, None, forbidden)),
        ('pno_molecular', lambda: molecular_inputs(w, 2, 18))):
        try:
            operation()
        except ValueError as exc:
            assert 'domain' in str(exc).lower()
            failures[label] = str(exc)
        else:
            raise AssertionError('Screened mixed nuclear path needs validation before accepting it.')

    gradient = assemble_cartesian_gradient(w)
    reference_branch = canonical_branch(w)
    ref = w.reference_wavefunction()
    L = w.variable('DLPNO RESPONSE ACTIVE LMOS').np
    xyz = mol.geometry().np.copy()
    comparisons = []
    for coordinate in (2, 18):
        estimates = []
        for step in (.004, .002):
            energies = []
            for sign in (-1, 1):
                trial = xyz.copy(); trial.flat[coordinate] += sign*step
                displaced = mol.clone()
                displaced.set_geometry(psi4.core.Matrix.from_array(trial))
                displaced.update_geometry()
                energy, other = psi4.energy('dlpno-ccsd(t)', molecule=displaced, return_wfn=True)
                cross = psi4.core.MintsHelper(ref).ao_overlap(ref.basisset(), other.basisset()).np
                match = L.T @ cross @ other.variable('DLPNO RESPONSE ACTIVE LMOS').np
                _, order = linear_sum_assignment(-abs(match))
                assert min(abs(match[np.arange(len(order)), order])) > .999
                assert canonical_branch(other, np.argsort(order)) == reference_branch
                energies.append(energy)
            estimates.append((energies[1]-energies[0])/(2*step))
        target = (4*estimates[1]-estimates[0])/3
        error = float(gradient.flat[coordinate]-target)
        np.testing.assert_allclose(error, 0., atol=2.e-8, rtol=0.)
        comparisons.append(dict(coordinate=coordinate, analytic=float(gradient.flat[coordinate]),
                                centered_estimates=estimates, richardson_error=error))

    probes = {key: float(value) for key, value in w.variables().items()
              if isinstance(value, float) and ('DIRECTION ERROR' in key or 'REPLAY ERROR' in key)}
    # Preserve the old acceptance threshold. The native projector diagnostic
    # now uses Richardson; the old central estimate is retained separately.
    probe_failures = {key: value for key, value in probes.items() if value > 1.e-8}
    assert probes['DLPNO DF ORBITAL REPLAY ERROR'] < 1.e-10
    assert probes['DLPNO DF LMO REFIT DIRECTION ERROR'] < 1.e-8
    return dict(domains=counts, mixed_nuclear_status='complete_screened_hessian_unsupported', provider_rejections=failures,
                gradient_comparisons=comparisons, native_debug_probes=probes,
                debug_probes_above_1e_minus8=probe_failures, gradient=gradient.tolist())


def run(path):
    configure()
    psi4.set_options(dict(freeze_core=False, save_jk=True, t_cut_pairs_mp2=1.e-6,
                          t_cut_pairs=1.e-6, t_cut_pno_mp2=1.e-5, t_cut_pno=1.e-4,
                          t_cut_tno=1.e-5, dlpno__debug=1))
    mol = psi4.geometry('0 1\n' + '\n'.join(
        f'H {x} 0 {z}' for x in (0, 3.5, 7, 10.5) for z in (-.37, .37)) +
        '\nunits angstrom\nsymmetry c1\nno_com\nno_reorient')
    _, graph_wfn = psi4.energy('dlpno-ccsd(t)', molecule=mol, return_wfn=True)
    graph = dict(domains=describe(graph_wfn), **check_graph(graph_wfn))
    assert min(graph['domains']['AO_OCC']) == graph['domains']['nbf']
    assert min(graph['domains']['PAIR_AUX']) == graph['domains']['naux']
    psi4.set_options(SCREENING)
    energy, w = psi4.energy('dlpno-ccsd(t)', molecule=mol, return_wfn=True)
    local = check_local(w, mol)
    keys = ('DLPNO LMP2 WEAK PAIR ENERGY', 'DLPNO SC-LMP2 PAIR ENERGY', 'DLPNO DIPOLE ENERGY')
    model_corrections = {key: w.variable(key) for key in keys}
    assert abs(model_corrections['DLPNO SC-LMP2 PAIR ENERGY']) > 1.e-7
    psi4.set_options(dict(dlpno_gradient_model=False, t_cut_triples_weak=0., t_cut_iter=0.,
                          t_cut_tno_strong=1.e-5, t_cut_tno_weak=1.e-5))
    ordinary, other = psi4.energy('dlpno-ccsd(t)', molecule=mol, return_wfn=True)
    corrections = {key: other.variable(key) for key in keys}
    assert abs(corrections['DLPNO SC-LMP2 PAIR ENERGY']) > 1.e-7
    np.testing.assert_allclose(list(model_corrections.values()), list(corrections.values()), atol=1.e-12, rtol=0.)
    np.testing.assert_allclose(ordinary, energy, atol=1.e-10, rtol=0.)
    # Confirm that an explicit analytic request cannot silently become an FD Hessian.
    try:
        psi4.hessian('dlpno-ccsd(t)', molecule=mol, dertype=2, return_plan=True)
    except MissingMethodError as exc:
        routing = str(exc)
    else:
        raise AssertionError('A new analytic driver must pass screened Hessian validation first.')
    blockers = ['Screened coupled-LMP2/final-PNO and CCSD molecular delivery is incomplete.',
                'No screened nuclear Hessian was assembled or compared to gradients.']
    if local['debug_probes_above_1e_minus8']:
        blockers.append('A native synthetic PNO operator probe exceeds its old tolerance.')
    result = dict(graph_only=graph, local_domains=local, coordinates_bohr=mol.geometry().np.tolist(),
                  screening=SCREENING, model_energy=energy, ordinary_energy=ordinary,
                  model_corrections=model_corrections, ordinary_corrections=corrections,
                  analytic_driver_rejection=routing, screened_hessian_validated=False,
                  production_ready=False, blockers=blockers,
                  functional_scope='The gradient model now includes eliminated SC-MP2 and dipole estimates '
                                   'with analytic gradients. Their mixed correction kernels are tested in '
                                   'screened_corrections.py. Screened first-PNO response is tested separately in '
                                   'screened_first_pno.py; complete screened PNO/CCSD Hessians remain unsupported.')
    Path(path).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ('screened_hessian_validated', 'production_ready', 'blockers')}, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='screened_domains_results.json')
    parser.add_argument('--require-production', action='store_true')
    args = parser.parse_args()
    psi4.set_output_file('screened_domains.out', False)
    result = run(args.output)
    if args.require_production and not result['production_ready']:
        raise SystemExit(2)
