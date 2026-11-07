"""Water/cc-pVDZ full Cartesian analytic Hessian and independent invariance checks.

All ordered entries are evaluated independently. No symmetrization or rigid-motion
projection is applied to the reported Hessian. This is a small-system diagnostic,
not the public screened-domain Hessian driver. Run in a built Psi4 environment.
"""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import time
import numpy as np
import psi4
from localization_integration import configure
from lmp2_tangent import calculation, check
from final_pno_mixed import analytic_final_pno_inputs
from triples_curvature import aligned_inputs, molecular_inputs, retained_inputs
from water_hvp import BASE, molecule
from psi4.driver.procrouting.response.dlpno_triples_curvature import MixedBlock, mixed_tno_frame
from psi4.driver.procrouting.response.dlpno_ccsd_mixed import mixed_ccsd_residual
from psi4.driver.procrouting.response.dlpno_ccsd_response import TriplesCCSDAdjoint
from psi4.driver.procrouting.response.dlpno_ccsd_curvature import RetainedCCSDCurvature
from psi4.driver.procrouting.response.dlpno_triples_response import TriplesHessianColumn
from psi4.driver.procrouting.response.dlpno_pno_curvature import PNOTruncationCurvature
from psi4.driver.procrouting.response.dlpno_orbital import assemble_cartesian_gradient


def reject_sample(key):
    raise AssertionError(f'Analytic Hessian requested a sampled input: {key}')


class CartesianModel:
    def __init__(self,frozen):
        self.w = calculation(BASE)
        self.rows = psi4.core.dlpno_triples_response_state(self.w)
        self.base = aligned_inputs(self.w,self.w,self.rows).electronic
        self.adjoint = TriplesCCSDAdjoint(self.w)
        self.retained = RetainedCCSDCurvature(self.w,self.adjoint)
        self.pno = PNOTruncationCurvature(self.w)
        self.lower,self.first,self.columns = [],[],[]
        for u in range(BASE.size):
            lower = {}
            row = check(frozen,row_at=divmod(u,3),reference=self.w,lower_order=lower)
            self.lower.append(lower)
            self.first.append(row)
            self.columns.append(TriplesHessianColumn(row))
            print('first response',u,flush=True)

    def providers(self,u,v,data):
        """Share the already analytic molecular inputs of the PNO construction."""
        base,du,dv = self.base,self.lower[u],self.lower[v]
        @lru_cache(None)
        def electronic(k):
            if k in ('lmo','fock','parent','occupied_density'):
                x = data['density' if k=='occupied_density' else k]
            elif k[0]=='pno':
                x = data['final_selection'][k[1]].frame
            elif k[0] in ('t1','t2'):
                x = MixedBlock(base[k],du[k],dv[k],np.zeros_like(base[k]))
            elif k[0]=='tno':
                i,j,l = r = k[1]
                pairs = ((i,j),(j,l),(i,l))
                x = mixed_tno_frame(r,electronic('parent'),data['overlap'],
                    {p:electronic(('pno',p)) for p in pairs},
                    {p:electronic(('t2',p)) for p in pairs},base[k])
            else: return reject_sample(k)
            for part,expected in zip(x[:3],(base[k],du[k],dv[k])):
                np.testing.assert_allclose(part,expected,atol=2.e-8,rtol=0.,err_msg=str(k))
            return x
        @lru_cache(None)
        def local_df(aux):
            ids = list(aux)
            return data['metric'].slice(np.ix_(ids,ids)),data['eri'].slice(ids)
        return electronic,data['overlap'],local_df

    def element(self,u,v):
        data = analytic_final_pno_inputs(self.w,u,v)
        provider = self.providers(u,v,data)
        replay = mixed_ccsd_residual(self.w,*provider)
        for component in (1,2):
            first = np.concatenate([x[component].ravel() for x in [*replay[0],*replay[1].values()]])
            np.testing.assert_allclose(first,0.,atol=2.e-9,rtol=0.)
        source = [x.uv for x in replay[0]],{p:x.uv for p,x in replay[1].items()}
        args = self.w,u,v,self.base,self.lower[u],self.lower[v],reject_sample
        triples = molecular_inputs(self.w,self.rows,u,v,self.base,self.lower[u],self.lower[v],
                                   reject_sample,trial_ccsd=True,pno_data=data,providers=provider)
        t = self.columns[v].ccsd_stationary_element(self.first[u],*triples,self.adjoint,source)
        cc = self.retained.element(*retained_inputs(*args,True,pno_data=data,providers=provider),source)
        pno = self.pno.element(data['initial'],data['final'],data['source'])
        return np.array([cc,t,pno])


def assert_branch(model,reference,rows,full):
    # Align solely to check labels, pair/triplet ranks and edges. The public
    # Cartesian gradient needs no orbital alignment.
    aligned_inputs(model,reference,rows)
    for name in ('DLPNO LMP2 WEAK PAIR ENERGY','DLPNO SC-LMP2 PAIR ENERGY','DLPNO DIPOLE ENERGY'):
        np.testing.assert_allclose(model.variable(name),0.,atol=1.e-12,rtol=0.,err_msg=name)
    if full:
        np.testing.assert_allclose(model.variable('DLPNO PNO TRUNCATION ERROR'),0.,atol=1.e-10,rtol=0.)


def invariants(hessian,gradient):
    """Rigid rotations obey H (Omega R) = Omega g away from stationarity."""
    translations = np.tile(np.eye(3),(len(BASE),1))
    rotations = np.stack([np.cross(np.broadcast_to(axis,BASE.shape),BASE).ravel() for axis in np.eye(3)],axis=1)
    rotated_grad = np.stack([np.cross(np.broadcast_to(axis,gradient.shape),gradient).ravel() for axis in np.eye(3)],axis=1)
    errors = dict(symmetry=float(np.max(abs(hessian-hessian.T))),
                  translations=float(np.max(abs(hessian@translations))),
                  rotations=float(np.max(abs(hessian@rotations-rotated_grad))),
                  net_force=float(np.max(abs(gradient.sum(axis=0)))),
                  torque=float(np.max(abs(np.cross(BASE,gradient).sum(axis=0)))))
    print('Cartesian invariants',errors,flush=True)
    for name in ('symmetry','translations','rotations'):
        assert errors[name] < 2.e-7,(name,errors[name])
    for name in ('net_force','torque'): assert errors[name] < 2.e-8,(name,errors[name])
    return errors


def run(frozen,full,output,stage='all',entries=None):
    psi4.set_options({'freeze_core':frozen})
    path = Path(output)
    if stage in ('analytic','all'):
        start = time.monotonic()
        model = CartesianModel(frozen)
        w = model.w
        assert_branch(w,w,model.rows,full)
        gradient = assemble_cartesian_gradient(w)
        hf = psi4.hessian('scf',molecule=molecule(BASE),dertype=2).np.copy()
        parts = np.full((3,9,9),np.nan)
        pairs = entries if entries is not None else [(u,v) for u in range(9) for v in range(9)]
        for u,v in pairs:
            tick = time.monotonic()
            parts[:,u,v] = model.element(u,v)
            np.savez(path,parts=parts,rhf=hf,gradient=gradient,coords=BASE,frozen=frozen,full=full)
            print('analytic entry',frozen,full,u,v,parts[:,u,v],time.monotonic()-tick,flush=True)
        print('analytic elapsed',time.monotonic()-start,flush=True)
        if entries is not None: return
    saved = np.load(path)
    assert bool(saved['frozen'])==frozen and bool(saved['full'])==full
    np.testing.assert_array_equal(saved['coords'],BASE)
    H = saved['rhf']+saved['parts'].sum(axis=0)
    assert np.isfinite(H).all()
    errors = invariants(H,saved['gradient'])
    if stage=='analytic': return
    # The independent finite-PNO gradient includes both loss stages; complete
    # spaces use canonical DF-CCSD(T). Difference without symmetrization at two
    # spacings before extrapolating.
    reference = calculation(BASE)
    rows = psi4.core.dlpno_triples_response_state(reference)
    numerical = []
    for h in (.004,.002):
        matrix = np.empty_like(H)
        for v in range(9):
            gradients = []
            for sign in (1,-1):
                moved = BASE.copy()
                moved.flat[v] += sign*h
                w = calculation(moved)
                assert_branch(w,reference,rows,full)
                gradients.append(psi4.gradient('ccsd(t)',molecule=molecule(moved),dertype=1).np.copy()
                                 if full else assemble_cartesian_gradient(w))
            matrix[:,v] = ((gradients[0]-gradients[1])/(2*h)).ravel()
            print('gradient column',frozen,full,h,v,flush=True)
        numerical.append(matrix)
    target = (4*numerical[1]-numerical[0])/3
    errors['gradient_difference'] = float(np.max(abs(H-target)))
    errors['gradient_difference_symmetry'] = float(np.max(abs(target-target.T)))
    print('full Hessian gradient comparison',errors,flush=True)
    np.testing.assert_allclose(H,target,atol=2.e-7,rtol=0.)
    result = dict(frozen_core=frozen,full_virtual_spaces=full,units='hartree/bohr^2',
                  basis='cc-pVDZ',scf_fit='cc-pVDZ-JKFIT',correlation_fit='cc-pVDZ-RI',
                  localization='BOYS',verification_steps_bohr=[.004,.002],
                  component_order=['retained_ccsd','iterative_triples','pno_truncation'],
                  coordinates_bohr=BASE.tolist(),gradient=saved['gradient'].tolist(),rhf_hessian=saved['rhf'].tolist(),
                  hessian=H.tolist(),correlation_components=saved['parts'].tolist(),
                  gradient_difference=target.tolist(),reference_method='DF-CCSD(T)' if full else 'DLPNO-CCSD(T)',errors=errors)
    path.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core',choices=('frozen','all'),required=True)
    parser.add_argument('--spaces',choices=('finite','full'),default='finite')
    parser.add_argument('--stage',choices=('all','analytic','verify'),default='all')
    parser.add_argument('--output',default='cartesian.npz')
    parser.add_argument('--entry',type=int,nargs=2,action='append')
    args = parser.parse_args()
    if Path(args.output).suffix != '.npz': parser.error('--output must end in .npz')
    if args.entry and any(not 0<=i<9 for pair in args.entry for i in pair):
        parser.error('Cartesian indices must be between 0 and 8.')
    configure()
    psi4.set_options(dict(save_jk=True,cc_type='df',scf__ints_tolerance=1.e-14,e_convergence=1.e-14,d_convergence=1.e-14,
                          solver_convergence=1.e-14,local_gradient_convergence=1.e-13))
    if args.spaces=='finite':
        psi4.set_options(dict(t_cut_pno_mp2=1.e-5,t_cut_pno=1.e-4,t_cut_tno=1.e-5))
    psi4.set_output_file('cartesian_hessian.out',False)
    run(args.core=='frozen',args.spaces=='full',args.output,args.stage,args.entry)
