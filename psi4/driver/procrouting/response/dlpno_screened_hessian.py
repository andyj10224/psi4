"""Assemble a screened RHF DLPNO-CCSD(T) Hessian column on a fixed branch.

This backend consumes analytic molecular inputs at one geometry and a native
analytic RHF column. It performs no coordinate sampling. The guarded public
driver in dlpno_hessian supplies installed molecular inputs and assembles all
columns. Dense input delivery does not provide local memory scaling. All
retained pairs are strong, with ordinary single-space T.
"""
import numpy as np

from .dlpno_ccsd_curvature import RetainedCCSDCurvature, project_retained_ccsd_energy
from .dlpno_ccsd_response import TriplesCCSDAdjoint
from .dlpno_pno_curvature import pno_truncation_energy
from .dlpno_screened_pno import screened_pno_response
from .dlpno_screened_ccsd import screened_ccsd_response, CCSDFirstResponseCache
from time import perf_counter
from .dlpno_screened_triples import ScreenedTriplesInputs,TriplesFirstResponseCache
from .dlpno_screened_curvature import screened_pair_curvature
from .dlpno_triples_response import TriplesHessianColumn


COMPONENTS=('rhf','retained_ccsd','iterative_triples','initial_pno_loss',
            'final_pno_loss','screened_scmp2','dipole_pairs')


def retained_inputs(response):
    """Retained energy uses the exact same local OV factors as the residual."""
    fit=response['fitting'];electronic=response['electronic']
    pairs={p:electronic(('pno',p)) for p in response['base'][1]}
    one=[electronic(('t1',i)) for i in range(len(response['base'][0]))]
    two={p:electronic(('t2',p)) for p in pairs}
    def pair_fit(p):
        _,ov,*_=fit.projected(p);domain=fit.pair_occ[p]
        return (fit.df(fit.pair_aux[p])[0],
                ov.slice((slice(None),domain.index(p[0]),slice(None))),
                ov.slice((slice(None),domain.index(p[1]),slice(None))))
    return electronic('lmo'),pairs,fit.S,electronic('fock'),one,two,pair_fit


def triples_first_energy(direction):
    return sum(np.vdot(direction.energy_seed_tangents[r],direction.amplitudes[r])+
               np.vdot(direction.adjoints[r],direction.right_source[r]) for r in direction.edges)


def _check_column_direction(data,reference,row,column):
    """Check repeatability without weakening the original absolute tolerance."""
    response=data['response'];saved=reference['response']
    keys=('lmo','fock',*[('t1',i) for i in range(len(response['base'][0]))],
          *[('t2',p) for p in response['base'][1]],*[('pno',p) for p in response['base'][1]])
    for key in keys:
        actual=response['electronic'](key).v
        expected=saved['electronic'](key).v
        context=f'block={key!r}, row={row}, column={column} (zero-based Cartesian coordinates)'
        if actual.shape!=expected.shape:
            raise RuntimeError(f'Inconsistent cached column direction: {context}; shapes {actual.shape} and {expected.shape}.')
        if not np.isfinite(actual).all() or not np.isfinite(expected).all():
            raise RuntimeError(f'Inconsistent cached column direction: {context}; nonfinite response.')
        delta=np.abs(actual-expected)
        error=float(np.max(delta,initial=0.))
        if error<=2.e-9:
            continue
        index=np.unravel_index(np.argmax(delta),delta.shape)
        details=[f'Inconsistent cached column direction: {context}',
                 f'max_abs={error:.6e}, atol=2.000000e-09, rtol=0',
                 f'index={index}, actual={actual[index]:.16e}, cached={expected[index]:.16e}']
        if isinstance(key,tuple) and key[0]=='pno':
            p=key[1];current,previous=data['pno'],reference['pno']
            for label,selection,old in (
                ('first',current.first_response.pairs[p].selection,previous.first_response.pairs[p].selection),
                ('final',current.final_selection[p],previous.final_selection[p])):
                details.append(f'{label}_occupation_gap={selection.boundary_gap:.6e} '
                               f'(cached {old.boundary_gap:.6e})')
            stages=(('overlap',response['fitting'].S,saved['fitting'].S),
                    ('lmo',response['electronic']('lmo'),saved['electronic']('lmo')),
                    ('fock',response['electronic']('fock'),saved['electronic']('fock')),
                    ('normalized_pao',current.first_response.paos[p],previous.first_response.paos[p]),
                    ('first_pno',current.first_response.pairs[p].selection.frame,
                                 previous.first_response.pairs[p].selection.frame),
                    ('lmp2_amplitude',current.final[p]['amplitudes'],previous.final[p]['amplitudes']))
            details.extend(f'{label}_column_delta={np.max(abs(a.v-b.v),initial=0.):.6e}' for label,a,b in stages)
        raise RuntimeError('; '.join(details)+'.')


class ScreenedHessianReference:
    """Reference-only gradient and CCSD adjoints shared by all columns."""
    def __init__(self, wfn):
        from psi4 import core
        self.wfn = wfn
        self.ccsd_first = CCSDFirstResponseCache(wfn)
        self.triples_first = TriplesFirstResponseCache(wfn)
        self.rhf_gradient = core.scfgrad(wfn.reference_wavefunction()).np.ravel().copy()
        self.adjoint = TriplesCCSDAdjoint(wfn)
        self.retained = RetainedCCSDCurvature(wfn, self.adjoint)


class ScreenedHessianColumn:
    """Cache the reference CCSD adjoints and one right/left triples tangent.

    inputs(u,v) supplies (S,F,L,occupied_action,dipoles,local_df), all analytic
    MixedBlocks/actions on this reference branch. rhf_column includes nuclear
    repulsion and follows the same fixed Cartesian order, in hartree/bohr^2.
    Separate row evaluations are returned without symmetrizing or projection.
    """
    def __init__(self,wfn,column,inputs,rhf_column,reference=None):
        self.wfn,self.coordinate,self.inputs=wfn,column,inputs
        self.size=3*wfn.molecule().natom()
        if not 0<=column<self.size:
            raise ValueError('Invalid Hessian column coordinate.')
        self.rhf=np.asarray(rhf_column,dtype=float).copy()
        if self.rhf.shape!=(self.size,) or not np.isfinite(self.rhf).all():
            raise ValueError('Expected a finite analytic RHF Hessian column.')
        if abs(wfn.variable('DLPNO LMP2 WEAK PAIR ENERGY'))>1.e-12:
            raise ValueError('Screened Hessian column requires the all-strong gradient model.')
        if reference is None: reference=ScreenedHessianReference(wfn)
        if reference.wfn is not wfn:
            raise ValueError('Hessian reference belongs to a different wavefunction.')
        self.rhf_gradient=reference.rhf_gradient
        self.adjoint=reference.adjoint
        self.retained=reference.retained
        self.ccsd_first=reference.ccsd_first
        self.triples_first=reference.triples_first
        self.timings={}
        self.reference=self._prepare(column)
        direction=self._timed('triples first directions',self.reference['triples'].first_direction,2,column)
        self.column=self._timed('triples tangent solve',TriplesHessianColumn,direction)

    def _timed(self,name,operation,*args,**kwargs):
        start=perf_counter()
        try:return operation(*args,**kwargs)
        finally:self.timings[name]=self.timings.get(name,0.)+perf_counter()-start

    def _prepare(self,u):
        S,F,L,D,dip,df=self._timed('molecular inputs',self.inputs,u,self.coordinate)
        pno=self._timed('PNO/LMP2 response',screened_pno_response,self.wfn,S,F,L,D,df)
        response=self._timed('CCSD mixed response',screened_ccsd_response,self.wfn,pno,S,F,L,D,df,
                             first_cache=self.ccsd_first,coordinates=(u,self.coordinate))
        triples=ScreenedTriplesInputs(self.wfn,response['fitting'],self.triples_first)
        corrections=self._timed('screened pair curvature',screened_pair_curvature,self.wfn,S,F,L,D,dip,df)
        initial=pno_truncation_energy(pno.initial,{})
        final=pno_truncation_energy({},pno.final)
        np.testing.assert_allclose(initial.uv+final.uv,pno.stationary_loss,atol=2.e-10,rtol=0.)
        args=retained_inputs(response)
        retained=self._timed('retained energy replay',project_retained_ccsd_energy,*args)
        values=[retained.value,initial.value,final.value,corrections['SCMP2'].value,corrections['DIPOLE'].value]
        names=['DLPNO RETAINED CCSD CORRELATION ENERGY','DLPNO INITIAL PNO TRUNCATION ENERGY',
               'DLPNO FINAL PNO TRUNCATION ENERGY','DLPNO SC-LMP2 PAIR ENERGY','DLPNO DIPOLE ENERGY']
        np.testing.assert_allclose(values,[self.wfn.variable(k) for k in names],atol=2.e-10,rtol=0.)
        np.testing.assert_allclose(self.wfn.reference_wavefunction().energy()+sum(values)+
            self.wfn.variable('(T) CORRECTION ENERGY'),self.wfn.energy(),atol=2.e-10,rtol=0.)
        return dict(response=response,triples=triples,corrections=corrections,pno=pno,
                    initial=initial,final=final,retained=retained,args=args)

    def element(self,u):
        if not 0<=u<self.size:
            raise ValueError('Invalid Hessian row coordinate.')
        data=self.reference if u==self.coordinate else self._prepare(u)
        response=data['response'];triples=data['triples']
        # A cached column solve is valid only for the same first molecular and
        # amplitude direction. Check it independently of this row's mixed data.
        _check_column_direction(data,self.reference,u,self.coordinate)
        row=(self.column.direction if u==self.coordinate else
             self._timed('triples first directions',triples.first_direction,1,u))
        residual=([x.uv for x in response['residual'][0]],{p:x.uv for p,x in response['residual'][1].items()})
        retained=self._timed('retained CCSD curvature',self.retained.element,*data['args'],residual)
        triple=self._timed('triples mixed curvature',self.column.ccsd_stationary_element,
            row,lambda i,j:triples.occupied(i,j).uv,
            triples.mixed_row,lambda r,q:triples.edge(r,q).uv,self.adjoint,residual)
        values=np.array([self.rhf[u],retained,triple,data['initial'].uv,data['final'].uv,
                         data['corrections']['SCMP2'].uv,data['corrections']['DIPOLE'].uv],dtype=float)
        first=np.array([self.rhf_gradient[u],data['retained'].u,triples_first_energy(row),
                        data['initial'].u,data['final'].u,data['corrections']['SCMP2'].u,
                        data['corrections']['DIPOLE'].u],dtype=float)
        if not np.isfinite(values).all() or not np.isfinite(first).all():
            raise ValueError('Nonfinite screened Hessian or first-energy contribution.')
        return dict(components=values,gradient_components=first,
                    cc_residual_errors=response['residual_errors'],pno_residual_errors=data['pno'].residual_errors,
                    triples_right_residual=self.column.tangent.residual,
                    triples_left_residual=self.column.tangent.adjoint_residual)
