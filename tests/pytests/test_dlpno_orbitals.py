"""Molden export of final DLPNO domains in the AO metric.

These integration tests require a compiled Psi4. In particular they check the
ROHF SOMO augmentation and final Brueckner state, not just file existence.
"""

from pathlib import Path

import numpy as np
import pytest
import psi4

from addons import uusing

pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.dlpno]


def _read_molden(path, wfn):
    text = path.read_text()
    assert text.startswith('[Molden Format]\n')
    assert '[Atoms] (AU)' in text and '[GTO]' in text and '[MO]' in text
    nbf = wfn.basisset().nbf()
    columns, occupations, energies = [], [], []
    for block in text.split('[MO]', 1)[1].split('Sym=')[1:]:
        lines = block.splitlines()
        coeffs = np.zeros(nbf)
        seen = []
        for line in lines[1:]:
            if '=' in line:
                key, value = (part.strip() for part in line.split('=', 1))
                if key == 'Occup':
                    occupations.append(float(value))
                elif key == 'Ene':
                    energies.append(float(value))
                elif key == 'Spin':
                    assert value in ('Alpha', 'Beta')
            elif line.strip():
                i, value = line.split()
                seen.append(int(i))
                coeffs[int(i) - 1] = float(value)
        assert seen == list(range(1, nbf + 1)), path
        columns.append(coeffs)
    C = np.column_stack(columns)
    assert np.isfinite(C).all() and np.isfinite(energies).all()
    assert len(occupations) == len(energies) == C.shape[1]
    return C, np.asarray(occupations), np.asarray(energies)


def _psi4_coefficients(C_molden, wfn):
    """Undo Molden ordering using the documented Cartesian exponent lists."""
    molden_powers = {
        2: [(2, 0, 0), (0, 2, 0), (0, 0, 2), (1, 1, 0), (1, 0, 1), (0, 1, 1)],
        3: [(3, 0, 0), (0, 3, 0), (0, 0, 3), (1, 2, 0), (2, 1, 0),
            (2, 0, 1), (1, 0, 2), (0, 1, 2), (0, 2, 1), (1, 1, 1)],
        4: [(4, 0, 0), (0, 4, 0), (0, 0, 4), (3, 1, 0), (3, 0, 1),
            (1, 3, 0), (0, 3, 1), (1, 0, 3), (0, 1, 3), (2, 2, 0),
            (2, 0, 2), (0, 2, 2), (2, 1, 1), (1, 2, 1), (1, 1, 2)],
    }
    C = C_molden.copy()
    basis = wfn.basisset()
    offset = 0
    for s in range(basis.nshell()):
        shell = basis.shell(s)
        am, size = shell.am, shell.nfunction
        if am == 1 and basis.has_puream():
            # Psi4 spherical p: z,x,y; Molden: x,y,z.
            C[offset:offset + size] = C_molden[offset + np.array([2, 0, 1])]
        elif am >= 2 and shell.is_cartesian():
            powers = [(x, y, am - x - y) for x in range(am, -1, -1)
                      for y in range(am - x, -1, -1)]
            positions = [molden_powers[am].index(power) for power in powers]
            C[offset:offset + size] = C_molden[offset + np.asarray(positions)]
        offset += size
    S = psi4.core.MintsHelper(basis).ao_overlap().np
    return C / np.sqrt(np.diag(S))[:, None]


def _check_export(root, wfn, kinds, brueckner=False):
    index = [line.split('\t') for line in (root / 'index.tsv').read_text().splitlines()]
    assert index[0] == ['file', 'orbitals']
    files = {name: int(count) for name, count in index[1:]}
    assert len(files) == len(index) - 1
    assert {Path(name).parts[0] for name in files} == set(kinds)
    S = psi4.core.MintsHelper(wfn.basisset()).ao_overlap().np
    L_molden, occupation, _ = _read_molden(root / 'LMO/lmo.molden', wfn)
    L = _psi4_coefficients(L_molden, wfn)
    naocc, nbocc = wfn.nalpha() - wfn.nfrzc(), wfn.nbeta() - wfn.nfrzc()
    nsomo = naocc - nbocc
    assert L.shape[1] == naocc
    np.testing.assert_array_equal(occupation, [2.0] * nbocc + [1.0] * nsomo)
    # The frozen space is unchanged even for Brueckner optimization.
    core = wfn.reference_wavefunction().Ca_subset('AO', 'FROZEN_OCC').np
    docc = np.column_stack((core, L[:, :nbocc]))
    for name, count in files.items():
        raw, occ, _ = _read_molden(root / name, wfn)
        C = _psi4_coefficients(raw, wfn)
        assert C.shape[1] == count
        np.testing.assert_allclose(C.T @ S @ C, np.eye(count), atol=2.e-8, rtol=0)
        kind = Path(name).parts[0]
        if kind in ('PNO', 'TNO', 'QNO'):
            np.testing.assert_array_equal(occ, np.zeros(count))
            np.testing.assert_allclose(docc.T @ S @ C, 0, atol=2.e-8, rtol=0)
            indices = [int(i) for i in Path(name).stem.split('_')[1:]]
            assert len(indices) == {'PNO': 2, 'TNO': 3, 'QNO': 4}[kind]
            assert all(1 <= i <= naocc for i in indices)
            if kind == 'PNO' or not nsomo:
                assert indices == sorted(indices)
            if nsomo:
                # SOMOs are explicit trailing columns, not accidentally dropped
                # when transforming the augmented local PAO domain into AOs.
                np.testing.assert_allclose(C[:, -nsomo:], L[:, -nsomo:], atol=2.e-8, rtol=0)
                np.testing.assert_allclose(L.T @ S @ C[:, :-nsomo], 0, atol=2.e-8, rtol=0)
    if brueckner:
        C, occ, eps = _read_molden(root / 'Brueckner/brueckner.molden', wfn)
        C = _psi4_coefficients(C, wfn)
        np.testing.assert_allclose(C, wfn.Ca().np, atol=2.e-12, rtol=0)
        np.testing.assert_allclose(C, wfn.Cb().np, atol=2.e-12, rtol=0)
        expected_occ = np.zeros(wfn.nmo())
        expected_occ[:wfn.nalpha()] += 1
        expected_occ[:wfn.nbeta()] += 1
        np.testing.assert_array_equal(occ, expected_occ)
        np.testing.assert_allclose(eps, np.diag(C.T @ (wfn.Fa().np + wfn.Fb().np) @ C) / 2,
                                   atol=2.e-10, rtol=0)
        # Compare against Psi4's independent public Molden writer, which sorts
        # orbitals by energy and, for ROHF, writes both spin channels.
        standard = root / 'standard.molden'
        wfn.write_molden(str(standard), do_virtual=True)
        C_standard, _, _ = _read_molden(standard, wfn)
        C_standard = _psi4_coefficients(C_standard, wfn)
        order = np.argsort(wfn.epsilon_a().np, kind='stable')
        np.testing.assert_allclose(C_standard[:, :wfn.nmo()], C[:, order], atol=2.e-8, rtol=0)
    return files


def _options():
    psi4.set_options({'basis': 'cc-pvdz', 'scf_type': 'df', 'freeze_core': True,
                      'df_basis_scf': 'cc-pvdz-jkfit', 'df_basis_cc': 'cc-pvdz-ri',
                      'e_convergence': 1.e-10, 'd_convergence': 1.e-10,
                      'r_convergence': 1.e-9, 'dlpno_maxiter': 100})


@pytest.mark.parametrize('puream', [True, False], ids=['spherical', 'cartesian'])
def test_dlpno_molden_mp2(puream, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mol = psi4.geometry('''0 1
O
H 1 0.96
H 1 0.96 2 104.5
symmetry c1
''')
    _options()
    psi4.set_options({'reference': 'rhf', 'puream': puream})
    _, ref = psi4.energy('scf', molecule=mol, return_wfn=True)
    off = psi4.energy('dlpno-mp2', ref_wfn=ref)
    assert not Path('dlpno_info').exists()
    psi4.set_options({'dlpno_write_orbitals': True})
    on, wfn = psi4.energy('dlpno-mp2', ref_wfn=ref, return_wfn=True)
    assert abs(on - off) < 1.e-12
    files = _check_export(Path('dlpno_info'), wfn, ['LMO', 'PNO'])
    saved = {name: (Path('dlpno_info') / name).read_bytes() for name in files}
    psi4.energy('dlpno-mp2', ref_wfn=ref)
    _check_export(Path('dlpno_info/run_2'), wfn, ['LMO', 'PNO'])
    for name, contents in saved.items():
        assert (Path('dlpno_info') / name).read_bytes() == contents


def test_dlpno_molden_rohf_brueckner(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mol = psi4.geometry('''0 3
C
H 1 1.08
H 1 1.08 2 133.0
symmetry c1
''')
    _options()
    psi4.set_options({'reference': 'rohf', 'dlpno_write_orbitals': True,
                      'brueckner_orbs_r_convergence': 1.e-7, 'brueckner_maxiter': 80,
                      't_cut_pno': 0.0, 't_cut_pno_mp2': 0.0, 't_cut_pairs': 0.0,
                      't_cut_tno': 0.0, 't_cut_tno_pre': 0.0})
    _, ref = psi4.energy('scf', molecule=mol, return_wfn=True)
    _, wfn = psi4.energy('dlpno-bccd(t)', ref_wfn=ref, return_wfn=True)
    assert wfn.variable('BRUECKNER ITERATIONS') > 0
    assert not np.allclose(wfn.Ca().np, ref.Ca().np, atol=1.e-7, rtol=0)
    _check_export(Path('dlpno_info'), wfn, ['LMO', 'PNO', 'TNO', 'Brueckner'], brueckner=True)
    # Inner orbital macroiterations must not create additional snapshots.
    assert not list(Path('dlpno_info').glob('run_*'))


@uusing('einsums')
def test_dlpno_molden_quadruples(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mol = psi4.geometry('''0 1
B
H 1 1.232
symmetry c1
''')
    _options()
    psi4.set_options({'basis': '6-31g', 'reference': 'rhf', 'dlpno_write_orbitals': True,
                      't_cut_pno': 0.0, 't_cut_pno_mp2': 0.0, 't_cut_pairs': 0.0,
                      't_cut_tno': 0.0, 't_cut_tno_pre': 0.0, 't_cut_tno_full': 0.0,
                      't_cut_qno': 0.0, 't_cut_qno_pre': 0.0, 't_cut_qno_full': 0.0,
                      't_cut_triples_weak': 0.0, 't_cut_quads_weak': 0.0})
    _, wfn = psi4.energy('dlpno-ccsdtq', molecule=mol, return_wfn=True)
    _check_export(Path('dlpno_info'), wfn, ['LMO', 'PNO', 'TNO', 'QNO'])
