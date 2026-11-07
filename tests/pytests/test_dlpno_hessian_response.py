"""Response-cache lifetime and actionable failures, without molecular solves."""
from types import SimpleNamespace

import numpy as np
import pytest

from psi4.driver.procrouting.response import dlpno_hessian_inputs as inputs
from psi4.driver.procrouting.response.dlpno_screened_hessian import _check_column_direction
from psi4.driver.procrouting.response.dlpno_triples_curvature import MixedBlock


def test_density_cache_is_shared_across_rows_and_columns(monkeypatch):
    ref=SimpleNamespace(molecule=lambda:SimpleNamespace(natom=lambda:1))
    wfn=SimpleNamespace(reference_wavefunction=lambda:ref)
    calls=[]
    def tangents(reference):
        assert reference is ref
        calls.append(reference)
        # Deliberately non-repeatable source: duplicate solves must not leak
        # into different entries of the same Hessian.
        return tuple(np.full((2,2),k+len(calls)*1.e-8) for k in range(3))
    contexts=[]
    def context(reference):
        value=object();contexts.append(value);return value
    monkeypatch.setattr(inputs,'_occupied_density_tangents',tangents)
    monkeypatch.setattr(inputs,'_MolecularIntegrals',context)
    monkeypatch.setattr(inputs,'molecular_inputs',lambda w,u,v,*,Du,Dv,integrals:(Du,Dv,integrals))
    provider=inputs.molecular_input_provider(wfn)
    entries={(u,v):provider(u,v) for v in range(3) for u in range(3)}
    assert len(calls)==1
    for (u,v),(du,dv,integrals) in entries.items():
        assert du is entries[u,u][0]
        assert dv is entries[v,v][0]
        assert integrals is contexts[0]
        assert not du.flags.writeable and not dv.flags.writeable
    # A new assembly must never inherit a previous geometry's cache.
    fresh=inputs.molecular_input_provider(wfn)(0,0)
    assert len(calls)==2
    assert fresh[0] is fresh[1] and fresh[0] is not entries[0,0][0]
    assert fresh[2] is not contexts[0]
    with pytest.raises(ValueError,match='Invalid nuclear response coordinate'):
        provider(3,0)
    with pytest.raises(ValueError,match='Invalid nuclear response coordinate'):
        provider(0,-1)


def test_integral_cache_is_bounded_and_reference_scoped(monkeypatch):
    class Mints:
        def __init__(self,ref,print): self.calls=[]; self.ref=ref
        def set_basisset(self,*args): pass
        def __getattr__(self,name):
            def evaluate(*args):
                value=object();self.calls.append((name,args));return value
            return evaluate
    monkeypatch.setattr(inputs.core,'MintsHelper',Mints,raising=False)
    ref=SimpleNamespace(basisset=lambda:object(),get_basisset=lambda name:object())
    cache=inputs._MolecularIntegrals(ref)
    first=cache.ao_3center_deriv1(0,'DF_BASIS_SCF')
    assert cache.ao_3center_deriv1(0,'DF_BASIS_SCF') is first
    for atom in range(1,10): cache.ao_3center_deriv1(atom,'DF_BASIS_SCF')
    assert cache.ao_3center_deriv1.cache_info().currsize==4
    assert cache.ao_3center_deriv1(0,'DF_BASIS_SCF') is not first
    fresh=inputs._MolecularIntegrals(ref)
    assert fresh.ao_3center_deriv1(0,'DF_BASIS_SCF') is not first


def direction_data(delta=0.):
    p=(0,0)
    def block(value=0.):
        zero=np.zeros((2,2))
        return MixedBlock(zero.copy(),zero.copy(),np.full((2,2),value),zero.copy())
    first=SimpleNamespace(frame=block(),boundary_gap=3.e-7)
    final=SimpleNamespace(frame=block(delta),boundary_gap=2.e-8)
    electronic={'lmo':block(),'fock':block(),('t1',0):block(),('t2',p):block(),('pno',p):final.frame}
    pno=SimpleNamespace(first_response=SimpleNamespace(pairs={p:SimpleNamespace(selection=first)},paos={p:block()}),
                        final_selection={p:final},final={p:{'amplitudes':block()}})
    return dict(response=dict(base=([None],{p:None}),electronic=electronic.__getitem__,
                              fitting=SimpleNamespace(S=block())),pno=pno)


def test_pno_repeatability_reports_pair_and_stages():
    saved=direction_data()
    _check_column_direction(direction_data(1.9e-9),saved,1,0)
    with pytest.raises(RuntimeError) as failure:
        _check_column_direction(direction_data(3.4e-9),saved,1,0)
    message=str(failure.value)
    for text in ("block=('pno', (0, 0))",'row=1, column=0','max_abs=3.400000e-09',
                 'atol=2.000000e-09, rtol=0','first_occupation_gap=3.000000e-07',
                 'final_occupation_gap=2.000000e-08','normalized_pao_column_delta=',
                 'first_pno_column_delta=','lmp2_amplitude_column_delta='):
        assert text in message


@pytest.mark.parametrize('value',[np.nan,np.inf])
def test_column_repeatability_rejects_nonfinite(value):
    with pytest.raises(RuntimeError,match='nonfinite response'):
        _check_column_direction(direction_data(value),direction_data(value),1,0)


def test_column_repeatability_rejects_shape_change():
    saved=direction_data()
    current=direction_data()
    previous=current['response']['electronic']
    current['response']['electronic']=lambda key:previous(key)._replace(v=np.zeros((3,2)))
    with pytest.raises(RuntimeError,match='shapes'):
        _check_column_direction(current,saved,1,0)


def test_reference_adjoints_are_shared_only_with_the_same_wavefunction(monkeypatch):
    from psi4.driver.procrouting.response import dlpno_screened_hessian as h
    counts=dict(gradient=0,adjoint=0,retained=0)
    w=SimpleNamespace(molecule=lambda:SimpleNamespace(natom=lambda:1),
                      reference_wavefunction=lambda:object(),variable=lambda key:0.)
    def gradient(ref):
        counts['gradient']+=1
        return SimpleNamespace(np=np.zeros((1,3)))
    def adjoint(wfn):
        counts['adjoint']+=1
        return object()
    def retained(wfn,source):
        counts['retained']+=1
        return object()
    monkeypatch.setattr(inputs.core,'scfgrad',gradient,raising=False)
    monkeypatch.setattr(h,'TriplesCCSDAdjoint',adjoint)
    monkeypatch.setattr(h,'RetainedCCSDCurvature',retained)
    monkeypatch.setattr(h,'TriplesHessianColumn',lambda direction:object())
    monkeypatch.setattr(h.ScreenedHessianColumn,'_prepare',
                        lambda self,k:{'triples':SimpleNamespace(first_direction=lambda order:None)})
    ref=h.ScreenedHessianReference(w)
    columns=[h.ScreenedHessianColumn(w,k,None,np.zeros(3),reference=ref) for k in range(3)]
    assert counts==dict(gradient=1,adjoint=1,retained=1)
    assert all(c.adjoint is ref.adjoint and c.retained is ref.retained for c in columns)
    other=SimpleNamespace(**w.__dict__)
    with pytest.raises(ValueError,match='different wavefunction'):
        h.ScreenedHessianColumn(other,0,None,np.zeros(3),reference=ref)
