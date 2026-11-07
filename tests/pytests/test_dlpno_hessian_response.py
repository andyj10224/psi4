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


def test_ccsd_first_response_cache_reuses_directions_and_checks_residuals(monkeypatch):
    from psi4.driver.procrouting.response import dlpno_screened_ccsd as cc
    from psi4.driver.procrouting.response import dlpno_ccsd_mixed as equations
    from psi4.driver.procrouting.response import dlpno_ccsd_response as solver
    p=(0,0)
    w=SimpleNamespace(molecule=lambda:SimpleNamespace(natom=lambda:1))
    matrix=lambda x:SimpleNamespace(np=np.asarray(x))
    monkeypatch.setattr(inputs.core,'dlpno_lmp2_blocks',lambda *a:{p:matrix([[0.]])},raising=False)
    monkeypatch.setattr(inputs.core,'dlpno_ccsd_df_pair_state',
                        lambda *a:([0],[matrix([0.])],None),raising=False)
    monkeypatch.setattr(cc,'ScreenedFitting',lambda *a:None)
    calls=dict(residual=0,solve=0)
    from psi4.driver.procrouting.response.dlpno_triples_curvature import mixed_product
    def residual(wfn,electronic,*args,**kwargs):
        calls['residual']+=1
        x=electronic('lmo')
        one=electronic(('t1',0));two=electronic(('t2',p))
        # R1=(1+x)t1+2x+x^2 and R2=2t2+3x at reference x=t=0.
        # The known first solutions are t1'=-2x', t2'=-3x'/2.
        xt=mixed_product('ij,j->i',x,one)
        xx=mixed_product('ij,ij->i',x,x)
        r1=MixedBlock(*(t+a+2*b.ravel()+c for t,a,b,c in zip(one,xt,x,xx)))
        r2=MixedBlock(*(2*t+3*b for t,b in zip(two,x)))
        return [r1],{p:r2}
    def solve(wfn,one,two):
        calls['solve']+=1
        return [-one[0]],{p:-two[p]/2}
    monkeypatch.setattr(equations,'mixed_ccsd_residual',residual)
    monkeypatch.setattr(solver,'coupled_ccsd_tangent',solve)
    def block(base,u=0.,v=0.,uv=0.):
        return MixedBlock(*(np.array([[x]]) for x in (base,u,v,uv)))
    overlap=block(1.)
    pnos=SimpleNamespace(final_selection={p:SimpleNamespace(frame=block(1.))})
    def evaluate(u,v,cache=None):
        x=block(0.,u+1.,v+1.,.1*(u+v))
        return cc.screened_ccsd_response(w,pnos,overlap,x,x,None,None,
            **({} if cache is None else dict(first_cache=cache,coordinates=(u,v))))
    expected={(u,v):evaluate(u,v)['residual'] for v in range(3) for u in range(3)}
    assert calls==dict(residual=18,solve=18)
    calls.update(residual=0,solve=0)
    cache=cc.CCSDFirstResponseCache(w)
    for v in range(3):
        for u in range(3):
            result=evaluate(u,v,cache)
            for actual,reference in zip([*result['residual'][0],*result['residual'][1].values()],
                                        [*expected[u,v][0],*expected[u,v][1].values()]):
                np.testing.assert_allclose(actual,reference,atol=1e-12,rtol=0.)
            # Independently known mixed residual on the affine amplitude path.
            np.testing.assert_allclose(result['residual'][0][0].uv,
                .2*(u+v)-2*(u+1)*(v+1),atol=1e-12,rtol=0.)
    assert calls==dict(residual=12,solve=3)
    assert cache.solves==3 and cache.source_replays==3
    assert not cache.entries[0]['tangent'][0][0].flags.writeable
    # Same coordinates with a changed first chart must never reuse amplitudes.
    old=pnos.final_selection[p].frame
    pnos.final_selection[p].frame=old._replace(u=np.ones((1,1)))
    with pytest.raises(ValueError,match='chart changed'):evaluate(0,1,cache)
    pnos.final_selection[p].frame=old
    # Even a matching frame must pass the independent current residual replay.
    wrong_cache=cc.CCSDFirstResponseCache(w)
    evaluate(0,0,wrong_cache)
    wrong_cache.entries[0]['tangent']=([np.array([7.])],{p:np.array([[0.]])})
    with pytest.raises(ValueError,match='value/first residual'):evaluate(0,0,wrong_cache)
