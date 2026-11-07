"""Analytic mixed coupled-LMP2 and final-PNO response in the full-domain model."""
import argparse
from functools import lru_cache
import numpy as np
import psi4
from scipy.optimize import linear_sum_assignment
from first_pno_mixed import analytic_pno_inputs, first_projector_derivative
from lmp2_tangent import blocks, calculation
from water_hvp import BASE
from psi4.driver.procrouting.response.dlpno_orbital import LMP2Tangent, final_pno_tangent
from psi4.driver.procrouting.response.dlpno_pno_curvature import (
    coupled_lmp2_curvature, mixed_pno_selection, lmp2_residual,
    pno_truncation_energy, PNOTruncationCurvature)
from psi4.driver.procrouting.response.dlpno_triples_curvature import MixedBlock, mixed_product


def analytic_final_pno_inputs(w,u,v,lower=None):
    """Complete both PNO selections with analytic inputs at one geometry."""
    data = analytic_pno_inputs(w,u,v) if lower is None else lower
    amplitudes = coupled_lmp2_curvature(w,data['final'],data['foo'],data['edges'],data['pair_overlap'])
    complete = {p:dict(inputs,amplitudes=amplitudes[p]) for p,inputs in data['final'].items()}
    selection = {p:mixed_pno_selection(data['first'][p],data['overlap'],amplitudes[p],y,
                                      diagonal=(p[0]==p[1])) for p,y in blocks(w,'FINAL_FRAME').items()}
    residual = lmp2_residual(complete,data['foo'],data['edges'],data['pair_overlap'])
    return dict(data,final=complete,final_selection=selection,
                mixed_residual_error=max(np.max(abs(r.uv)) for r in residual.values()))


def independent_tangent(w,u,reference):
    """Original first-order orbitals/PNOs plus native first LMP2 response."""
    lower = {}
    first_projector_derivative(w,u,w,response=lower)
    matrix = psi4.core.Matrix.from_array
    dt = psi4.core.dlpno_lmp2_tangent(w,
             {p:matrix(x) for p,x in lower['frame_derivatives'].items()},
             {p:matrix(x) for p,x in lower['exchange_derivatives'].items()},
             *(matrix(lower[k]) for k in ('overlap_derivative','fock_derivative','occupied_fock_derivative')))
    dt = {p:x.np.copy() for p,x in dt.items()}
    selected = final_pno_tangent(w,LMP2Tangent(lower['frame_derivatives'],dt))
    t = blocks(w,'AMPLITUDE')
    tensors = {}
    for p,y in lower['frames'].items():
        dy = lower['frame_derivatives'][p]
        tensors[p] = dy@t[p]@y.T+y@dt[p]@y.T+y@t[p]@dy.T
    ref,ref0 = w.reference_wavefunction(),reference.reference_wavefunction()
    cross = psi4.core.MintsHelper(ref).ao_overlap(ref0.basisset(),ref.basisset()).np
    match = reference.variable('DLPNO RESPONSE ACTIVE LMOS').np.T@cross@w.variable('DLPNO RESPONSE ACTIVE LMOS').np
    _,order = linear_sum_assignment(-abs(match))
    phases = np.sign(match[np.arange(len(order)),order])
    assert min(abs(match[np.arange(len(order)),order])) > .999
    amp,proj = {},{}
    first0,final0,final = blocks(reference,'FRAME'),blocks(reference,'FINAL_FRAME'),blocks(w,'FINAL_FRAME')
    for p in first0:
        i,j = p
        a,b = order[i],order[j]
        q = min(a,b),max(a,b)
        assert lower['frames'][q].shape == first0[p].shape and final[q].shape == final0[p].shape
        amp[p] = phases[i]*phases[j]*(tensors[q] if a<=b else tensors[q].T)
        proj[p] = selected[q].retained_density_tangent
    return amp,proj


def check_source_contract(w,data):
    """Reject malformed native sources; preserve the forward state and input."""
    matrix = psi4.core.Matrix.from_array
    original = blocks(w,'AMPLITUDE')
    source = {p:x.copy() for p,x in data['source'].items()}
    packed = lambda x:{p:matrix(a) for p,a in x.items()}
    raw = packed(source)
    psi4.core.dlpno_lmp2_mixed_tangent(w,raw)
    for p in source: np.testing.assert_array_equal(raw[p].np,source[p])
    diagonal = next(p for p,x in source.items() if p[0]==p[1] and len(x)>1)
    failures = [dict(source)]
    failures[0].pop(diagonal)
    for mode in ('size','nan','asymmetric'):
        bad = dict(source)
        bad[diagonal] = source[diagonal].copy()
        if mode=='size': bad[diagonal]=bad[diagonal][:-1,:-1]
        elif mode=='nan': bad[diagonal][0,0]=np.nan
        else: bad[diagonal][0,1]+=1.
        failures.append(bad)
    for bad in failures:
        try: psi4.core.dlpno_lmp2_mixed_tangent(w,packed(bad))
        except RuntimeError: pass
        else: raise AssertionError('Invalid mixed LMP2 source was accepted')
    zero = psi4.core.dlpno_lmp2_mixed_tangent(w,packed({p:np.zeros_like(x) for p,x in source.items()}))
    assert all(np.count_nonzero(x.np)==0 for x in zero.values())
    assert w.variable('DLPNO LMP2 MIXED TANGENT ITERATIONS') == 0
    for p,x in blocks(w,'AMPLITUDE').items(): np.testing.assert_array_equal(x,original[p])


def check(frozen,u,v,coarse_step=.001):
    psi4.set_options({'freeze_core':frozen})
    w = calculation(BASE)
    data = analytic_final_pno_inputs(w,u,v)
    native_residual = w.variable('DLPNO LMP2 MIXED TANGENT RESIDUAL')
    ao_t = {p:mixed_product('ma,ab,nb->mn',data['first'][p],entry['amplitudes'],data['first'][p])
            for p,entry in data['final'].items()}
    ao_p = {p:mixed_product('ma,na->mn',s.frame,s.frame) for p,s in data['final_selection'].items()}
    for p,selection in data['final_selection'].items():
        gram = mixed_product('ma,mn,nb->ab',selection.frame,data['overlap'],selection.frame)
        for x in gram[1:]: np.testing.assert_allclose(x,0,atol=2.e-8,rtol=0)
        squared = mixed_product('ma,ab,bn->mn',ao_p[p],data['overlap'],ao_p[p])
        for x,y in zip(squared,ao_p[p]): np.testing.assert_allclose(x,y,atol=2.e-8,rtol=0)
    for k,coordinate in enumerate((u,v),1):
        amplitude,projector = independent_tangent(w,coordinate,w)
        for p in ao_t:
            np.testing.assert_allclose(ao_t[p][k],amplitude[p],atol=2.e-9,rtol=0)
            np.testing.assert_allclose(ao_p[p][k],projector[p],atol=2.e-8,rtol=0)
    @lru_cache(None)
    def moved(h):
        coords = BASE.copy(); coords.flat[v]+=h
        return independent_tangent(calculation(coords),u,w)
    errors = []
    for index,actual in enumerate((ao_t,ao_p)):
        estimates = []
        for h in (coarse_step,coarse_step/2):
            plus,minus = moved(h)[index],moved(-h)[index]
            estimates.append({p:(plus[p]-minus[p])/(2*h) for p in actual})
        differences = {p:x.uv-(4*estimates[1][p]-estimates[0][p])/3 for p,x in actual.items()}
        worst = max(differences,key=lambda p:np.max(abs(differences[p])))
        error = np.max(abs(differences[worst]))
        print('Mixed response comparison',index,'worst pair',worst,'maximum analytic component',
              np.max(abs(actual[worst].uv)),'maximum error',error,flush=True)
        errors.append(error)
    direct = pno_truncation_energy(data['initial'],data['final']).uv
    stationary = PNOTruncationCurvature(w).element(data['initial'],data['final'],data['source'])
    print('Mixed LMP2/final PNO',frozen,u,v,coarse_step,'AO amplitude/projector errors',errors,
          'native/moving residuals',native_residual,data['mixed_residual_error'],
          'direct/stationary PNO-loss difference',direct-stationary,flush=True)
    np.testing.assert_allclose(errors[0],0,atol=2.e-7,rtol=0)
    np.testing.assert_allclose(errors[1],0,atol=2.e-6,rtol=0)
    np.testing.assert_allclose(direct,stationary,atol=2.e-10,rtol=0)
    if u!=v:
        reverse = analytic_final_pno_inputs(w,v,u)
        for p in ao_t:
            np.testing.assert_allclose(data['final'][p]['amplitudes'].uv,reverse['final'][p]['amplitudes'].uv,atol=2.e-9,rtol=0)
            np.testing.assert_allclose(data['final_selection'][p].frame.uv,reverse['final_selection'][p].frame.uv,atol=2.e-8,rtol=0)
    check_source_contract(w,data)


if __name__=='__main__':
    from localization_integration import configure
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core',choices=('both','frozen','all'),default='both')
    parser.add_argument('--coordinates',type=int,nargs='+',default=[1,5])
    parser.add_argument('--coarse-step',type=float,default=.001)
    args = parser.parse_args()
    if not args.coarse_step>0 or any(not 0<=u<9 for u in args.coordinates):
        parser.error('Expected positive spacing and Cartesian indices 0 through 8.')
    configure()
    # Small final-PNO occupation gaps amplify noise in displaced first
    # derivatives. Tighten the reference/orbital solves, not the error criterion.
    psi4.set_options(dict(t_cut_pno_mp2=1.e-5,t_cut_pno=1.e-4,t_cut_tno=1.e-5,
                          scf__ints_tolerance=1.e-14,e_convergence=1.e-14,d_convergence=1.e-14,
                          solver_convergence=1.e-14,local_gradient_convergence=1.e-13))
    psi4.set_output_file('final_pno_mixed.out',False)
    for frozen in ((True,False) if args.core=='both' else (args.core=='frozen',)):
        for u in args.coordinates: check(frozen,u,5,args.coarse_step)
