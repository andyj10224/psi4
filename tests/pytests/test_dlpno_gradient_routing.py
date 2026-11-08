"""Gradient stopping policy and option lifetime without a molecular solve."""
from types import SimpleNamespace

import numpy as np
import pytest
from psi4 import core
from psi4.driver import p4util
from psi4.driver.procrouting import proc
from psi4.driver.procrouting.response import dlpno_orbital


@pytest.mark.parametrize('full_triples', [False, True])
@pytest.mark.parametrize('cutoff_scope', ['default', 'global', 'local'])
@pytest.mark.parametrize('failure', [None, 'energy', 'assembly'])
def test_gradient_disables_triplet_stopping_and_restores_options(monkeypatch, full_triples, cutoff_scope, failure):
    # Exercise the real OptionsState save/restore, including the changed flags.
    settings = p4util.OptionsState(
        ['REFERENCE'], ['SCF_TYPE'], ['DLPNO', 'DLPNO_LOCAL_ORBITALS'],
        ['DLPNO', 'DLPNO_GRADIENT_MODEL'], ['DLPNO', 'DLPNO_GRADIENT_FULL_TRIPLES'],
        ['DLPNO', 'DLPNO_KEEP_TRIPLES_RESPONSE'], ['DLPNO', 'T_CUT_ITER'], ['SCF', 'SAVE_JK'])
    try:
        core.set_global_option('REFERENCE', 'RHF')
        core.set_global_option('SCF_TYPE', 'DF')
        core.set_local_option('DLPNO', 'DLPNO_LOCAL_ORBITALS', 'BOYS')
        core.set_local_option('DLPNO', 'DLPNO_GRADIENT_MODEL', False)
        core.set_local_option('DLPNO', 'DLPNO_GRADIENT_FULL_TRIPLES', full_triples)
        core.set_local_option('DLPNO', 'DLPNO_KEEP_TRIPLES_RESPONSE', False)
        core.set_global_option('T_CUT_ITER', 1.e-5)
        core.set_local_option('DLPNO', 'T_CUT_ITER', 1.e-5)
        core.revoke_global_option_changed('T_CUT_ITER')
        core.revoke_local_option_changed('DLPNO', 'T_CUT_ITER')
        if cutoff_scope == 'global':
            core.set_global_option('T_CUT_ITER', .2)
        elif cutoff_scope == 'local':
            core.set_local_option('DLPNO', 'T_CUT_ITER', .3)

        def snapshot():
            return (core.get_global_option('T_CUT_ITER'), core.get_local_option('DLPNO', 'T_CUT_ITER'),
                    core.get_option('DLPNO', 'T_CUT_ITER'), core.has_global_option_changed('T_CUT_ITER'),
                    core.has_local_option_changed('DLPNO', 'T_CUT_ITER'))

        before = snapshot()
        called = []
        wfn = SimpleNamespace(set_gradient=lambda x: None, set_variable=lambda *args: None, energy=lambda: -75.)
        mol = SimpleNamespace(schoenflies_symbol=lambda: 'c1')

        def energy(*args, **kwargs):
            called.append('energy')
            assert core.get_option('DLPNO', 'T_CUT_ITER') == 0.
            assert core.get_option('DLPNO', 'DLPNO_GRADIENT_MODEL')
            if failure == 'energy':
                raise RuntimeError('test energy failure')
            return wfn

        def assemble(reference):
            called.append('assembly')
            assert reference is wfn
            assert core.get_option('DLPNO', 'T_CUT_ITER') == 0.
            if failure == 'assembly':
                raise RuntimeError('test assembly failure')
            return np.zeros((1, 3))

        monkeypatch.setattr(proc, 'run_dlpno', energy, raising=False)
        monkeypatch.setattr(dlpno_orbital, 'assemble_cartesian_gradient', assemble)
        monkeypatch.setattr(core, 'Matrix', SimpleNamespace(from_array=lambda x: x))
        monkeypatch.setattr(core, 'set_variable', lambda *args: None)
        if failure:
            with pytest.raises(RuntimeError, match=f'test {failure} failure'):
                proc.run_dlpno_gradient('dlpno-ccsd(t)', molecule=mol)
        else:
            assert proc.run_dlpno_gradient('dlpno-ccsd(t)', molecule=mol) is wfn
        assert called == (['energy'] if failure == 'energy' else ['energy', 'assembly'])
        assert snapshot() == before
        assert not core.get_option('DLPNO', 'DLPNO_GRADIENT_MODEL')
    finally:
        settings.restore()
