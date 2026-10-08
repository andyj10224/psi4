"""Independent determinant check of the non-HF (Q) spin trace and tuple weights.

This checks the algebra used by compute_quadruplet_fock_energy, including
repeated occupied indices. It does not substitute for molecular C++ tests.
It can run without a compiled Psi4: pytest --noconftest test_dlpno_fock_algebra.py.
"""

import itertools
import math

import numpy as np
import pytest


def _one_body(state, p, q):
    result = np.zeros_like(state)
    for determinant in np.flatnonzero(state):
        occupation = int(determinant)
        if not (occupation >> q) & 1:
            continue
        sign = (-1) ** ((occupation & ((1 << q) - 1)).bit_count())
        occupation ^= 1 << q
        if (occupation >> p) & 1:
            continue
        sign *= (-1) ** ((occupation & ((1 << p) - 1)).bit_count())
        occupation ^= 1 << p
        result[occupation] += sign * state[determinant]
    return result


def _cluster_state(amplitudes, rank, nocc, nvirt):
    # T_n = (1/n!) sum t_(i...)(a...) E_ai ... E_bn, E_ai = sum_spin a+_a a_i.
    reference = np.zeros(1 << (2 * (nocc + nvirt)))
    reference[(1 << (2 * nocc)) - 1] = 1.0
    result = np.zeros_like(reference)
    for occupied in itertools.product(range(nocc), repeat=rank):
        for virtual in itertools.product(range(nvirt), repeat=rank):
            state = reference.copy()
            for i, a in zip(occupied, virtual):
                state = sum((_one_body(state, 2 * (nocc + a) + spin, 2 * i + spin)
                             for spin in (0, 1)), np.zeros_like(state))
            result += amplitudes[occupied + virtual] * state / math.factorial(rank)
    return result


def _spin_trace(t4):
    result = np.zeros_like(t4)
    for permutation in itertools.permutations(range(4)):
        visited = set()
        cycles = 0
        for start in range(4):
            if start in visited:
                continue
            cycles += 1
            p = start
            while p not in visited:
                visited.add(p)
                p = permutation[p]
        weight = (-1) ** (4 - cycles) * 2 ** cycles
        result += weight * t4.transpose(tuple(range(4)) + tuple(p + 4 for p in permutation))
    return result


@pytest.mark.parametrize("nocc", [2, 3, 4])
def test_non_hf_quadruples_left_moment(nocc):
    nvirt = 2
    rng = np.random.default_rng(819 + nocc)
    amplitudes = {}
    for rank in (3, 4):
        tensor = rng.normal(size=(nocc,) * rank + (nvirt,) * rank)
        # Impose simultaneous occupied/virtual column symmetry only; the
        # determinant construction, independently, handles fermion signs.
        amplitudes[rank] = sum(
            tensor.transpose(p + tuple(i + rank for i in p))
            for p in itertools.permutations(range(rank))
        ) / math.factorial(rank)
    fov = rng.normal(size=(nocc, nvirt))
    t3_state = _cluster_state(amplitudes[3], 3, nocc, nvirt)
    t4_state = _cluster_state(amplitudes[4], 4, nocc, nvirt)
    f_t4 = np.zeros_like(t4_state)
    for i, a, spin in itertools.product(range(nocc), range(nvirt), (0, 1)):
        f_t4 += fov[i, a] * _one_body(t4_state, 2 * i + spin, 2 * (nocc + a) + spin)
    determinant_energy = t3_state @ f_t4

    traced = _spin_trace(amplitudes[4])
    ordered_energy = np.einsum("ijkabc,ld,ijklabcd->", amplitudes[3], fov, traced) / 6
    np.testing.assert_allclose(ordered_energy, determinant_energy, rtol=1e-12, atol=1e-12)

    # Fold the ordered occupied sum onto the stored i <= j <= k <= l
    # quadruplets. Each distinct omitted orbital occurs once; the remaining
    # triple carries the inverse factorial of its repeated-index counts.
    folded_energy = 0.0
    for occupied in itertools.combinations_with_replacement(range(nocc), 4):
        if any(occupied.count(i) > 2 for i in set(occupied)):
            continue  # Three electrons cannot occupy one spatial orbital.
        for omitted in sorted(set(occupied)):
            triple = list(occupied)
            triple.remove(omitted)
            triple = tuple(triple)
            weight = 1 / math.prod(math.factorial(triple.count(i)) for i in set(triple))
            folded_energy += weight * np.einsum(
                "abc,d,abcd->", amplitudes[3][triple], fov[omitted], traced[triple + (omitted,)])
    np.testing.assert_allclose(folded_energy, determinant_energy, rtol=1e-12, atol=1e-12)
