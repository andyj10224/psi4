"""Analytic first-PNO and stationary PNO-loss nuclear curvature diagnostic.

Full AO/fitting domains, fixed pair graph/ranks, inverse-metric DF and Boys
localization only. Displacements appear exclusively in the verification code.
This small-system provider is not a screened local Hessian driver.
"""
from functools import lru_cache
import argparse
import numpy as np
import psi4
from scipy.optimize import linear_sum_assignment
from first_pno import raw, relax_fock
from lmp2_tangent import blocks, calculation
from rhf_mixed import analytic_inputs
from localization_mixed import molecular_localization
from water_hvp import BASE
from psi4.driver.procrouting.response.dlpno_triples_curvature import MixedBlock, mixed_product
from psi4.driver.procrouting.response.dlpno_frame_curvature import mixed_virtual_parent, mixed_paos
from psi4.driver.procrouting.response.dlpno_orbital import (
    first_pno_tangent, _transport_pno_frame, boys_nuclear_tangent, pair_exchange_tangent)
from psi4.driver.procrouting.response.dlpno_pno_curvature import (
    first_pno_curvature, mixed_pair_three_center, mixed_pair_exchange,
    PNOTruncationCurvature, lmp2_residual, lmp2_mixed_residual)


def molecular_inputs(w, u, v):
    """All lower-order electronic inputs are analytic at one geometry."""
    for key in ('T_CUT_DO T_CUT_DO_PRE T_CUT_DO_IJ T_CUT_DO_UV T_CUT_MKN '
                'T_CUT_PRE T_CUT_CLMO T_CUT_CPAO DLPNO_AO_INTS_TOL').split():
        if psi4.core.get_option('DLPNO',key) != 0.:
            raise ValueError('Full-domain mixed PNO diagnostic requires '+key+' = 0.')
    if not w.variable('DLPNO RESPONSE BOYS'):
        raise ValueError('This molecular PNO diagnostic requires Boys localization.')
    ref = w.reference_wavefunction()
    data = {}
    _, _, active = analytic_inputs(ref,u,v,partition=True,block_output=data)
    S, D = data['overlap'], data['density']
    L = molecular_localization(ref,w.variable('DLPNO RESPONSE ACTIVE LMOS').np,'BOYS',u,v,active)
    action = lambda panel: mixed_product('mn,na->ma',D,panel)
    V = mixed_virtual_parent(action,S,ref.Ca_subset('AO','VIR').np)
    mints = psi4.core.MintsHelper(ref)
    J, Q = raw(ref,ref.get_basisset('DF_BASIS_CC'))
    au,xu = divmod(u,3); av,xv = divmod(v,3)
    j2,q2 = psi4.core.dlpno_df_mixed_integrals(ref,'DF_BASIS_CC',list(range(len(J))),
                                             list(range(len(S.value))),u,v)
    metric = MixedBlock(J,mints.ao_metric_deriv1(au,'DF_BASIS_CC')[xu].np.reshape(J.shape),
                         mints.ao_metric_deriv1(av,'DF_BASIS_CC')[xv].np.reshape(J.shape),j2.np.reshape(J.shape))
    eri = MixedBlock(Q,mints.ao_3center_deriv1(au,'DF_BASIS_CC')[xu].np.reshape(Q.shape),
                      mints.ao_3center_deriv1(av,'DF_BASIS_CC')[xv].np.reshape(Q.shape),q2.np.reshape(Q.shape))
    data.update(lmo=L,parent=V,metric=metric,eri=eri,occupied_action=action)
    return data


def analytic_pno_inputs(w,u,v,verify_first=False):
    """Build both loss stages without displaced models or mixed LMP2 amplitudes."""
    data = molecular_inputs(w,u,v)
    S,F,L,V,J,Q = (data[k] for k in ('overlap','fock','lmo','parent','metric','eri'))
    foo = mixed_product('mi,mn,nj->ij',L,F,L)
    frames, final_frames, amps = (blocks(w,k) for k in ('FRAME','FINAL_FRAME','AMPLITUDE'))
    all_ao = np.arange(len(S.value))
    qparent = {i:mixed_pair_three_center(S,L.slice((slice(None),i)),V,all_ao,all_ao,Q)
               for i in range(L.value.shape[1])}
    initial, final, first = {}, {}, {}
    for p, reference in frames.items():
        i,j = p
        occ = MixedBlock(*(x[i,i]+x[j,j] for x in foo))
        exchange = mixed_pair_exchange(J,qparent[i],qparent[j])
        out = first_pno_curvature(V,S,F,exchange,occ,np.eye(V.value.shape[1]),reference,diagonal=(i==j))
        first[p] = out.selection.frame
        initial[p] = dict(exchange=out.exchange,amplitudes=out.amplitudes,fock=out.fock,
                          fov=mixed_product('m,mn,na->a',L.slice((slice(None),i)),F,out.parent),
                          occupied_energy=foo.slice((i,i)),rank=reference.shape[1])
        if verify_first:
            gram = mixed_product('ma,mn,nb->ab',first[p],S,first[p])
            for x in gram[1:]:
                np.testing.assert_allclose(x,0,atol=2.e-8,rtol=0)
            for k in (1,2):
                old = first_pno_tangent(V.value,V[k],S.value,S[k],F.value,F[k],exchange.value,exchange[k],
                                        occ.value,occ[k],np.eye(V.value.shape[1]),reference.shape[1],i==j)
                expected = _transport_pno_frame(reference,old,S.value,S[k],p)
                np.testing.assert_allclose(first[p][k],expected,atol=2.e-9,rtol=0)
        qi,qj = (mixed_product('m,Qmn,na->Qa',L.slice((slice(None),k)),Q,first[p]) for k in p)
        final[p] = dict(exchange=mixed_pair_exchange(J,qi,qj),
                        fock=mixed_product('ma,mn,nb->ab',first[p],F,first[p]),
                        fov=mixed_product('m,mn,na->a',L.slice((slice(None),i)),F,first[p]),
                        occupied_energy=foo.slice((i,i)),rank=final_frames[p].shape[1])
    matrix = psi4.core.Matrix.from_array
    def tangent(k):
        result = psi4.core.dlpno_lmp2_tangent(w,{p:matrix(x[k]) for p,x in first.items()},
                    {p:matrix(x['exchange'][k]) for p,x in final.items()},matrix(S[k]),matrix(F[k]),matrix(foo[k]))
        return {p:x.np.copy() for p,x in result.items()}
    tu = tangent(1)
    tv = tu if u==v else tangent(2)
    for p,t in amps.items():
        final[p]['amplitudes'] = MixedBlock(t,tu[p],tv[p],np.zeros_like(t))
    edges = {p:x.astype(int) for p,x in blocks(w,'RESPONSE_EDGES').items()}
    @lru_cache(None)
    def overlap(p,q):
        return mixed_product('ma,mn,nb->ab',first[p],S,first[q])
    residual = lmp2_residual(final,foo,edges,overlap)
    errors = [max(np.max(abs(r[k])) for r in residual.values()) for k in (0,1,2)]
    np.testing.assert_allclose(errors,0,atol=2.e-9,rtol=0)
    source = lmp2_mixed_residual(final,foo,edges,overlap)
    data.update(initial=initial,final=final,first=first,foo=foo,source=source,residual_errors=errors)
    return data


def analytic_pno_entry(w,u,v,adjoint=None):
    data = analytic_pno_inputs(w,u,v)
    if adjoint is None: adjoint = PNOTruncationCurvature(w)
    return adjoint.element(data['initial'],data['final'],data['source'])


def first_projector_derivative(w,u,reference):
    """Independent pre-existing first-order route, used only as a mixed oracle."""
    ref,ref0 = w.reference_wavefunction(),reference.reference_wavefunction()
    S,F = ref.S().np,ref.Fa().np
    C,V = (ref.Ca_subset('AO',space).np for space in ('OCC','VIR'))
    L = w.variable('DLPNO RESPONSE ACTIVE LMOS').np
    direction = np.zeros((ref.molecule().natom(),3)); direction.flat[u]=1.
    dC = psi4.core.rhf_occupied_tangent(ref,psi4.core.Matrix.from_array(direction)).np
    dL = boys_nuclear_tangent(w,direction,tol=1.e-11)
    atom,xyz = divmod(u,3)
    mints = psi4.core.MintsHelper(ref)
    dS = mints.ao_oei_deriv1('OVERLAP',atom)[xyz].np
    dF = relax_fock(ref,dC,atom,xyz)
    dV = -C@(dC.T@S@V+C.T@dS@V)-.5*V@(V.T@dS@V)
    J,Q = raw(ref,ref.get_basisset('DF_BASIS_CC'))
    dJ = mints.ao_metric_deriv1(atom,'DF_BASIS_CC')[xyz].np.reshape(J.shape)
    dQ = mints.ao_3center_deriv1(atom,'DF_BASIS_CC')[xyz].np.reshape(Q.shape)
    panels = {i:(L[:,i]@Q@V,dL[:,i]@Q@V+L[:,i]@dQ@V+L[:,i]@Q@dV) for i in range(L.shape[1])}
    foo = L.T@F@L
    dfoo = dL.T@F@L+L.T@dF@L+L.T@F@dL
    cross = mints.ao_overlap(ref0.basisset(),ref.basisset()).np
    match = reference.variable('DLPNO RESPONSE ACTIVE LMOS').np.T@cross@L
    _,order = linear_sum_assignment(-abs(match))
    assert min(abs(match[np.arange(len(order)),order])) > .999
    native,old = blocks(w,'FRAME'),blocks(reference,'FRAME')
    result = {}
    for p,y in old.items():
        i,j = sorted(order[list(p)])
        assert native[i,j].shape == y.shape
        k,dk = pair_exchange_tangent(J,dJ,*panels[i],*panels[j])
        selected = first_pno_tangent(V,dV,S,dS,F,dF,k,dk,foo[i,i]+foo[j,j],dfoo[i,i]+dfoo[j,j],
                                     np.eye(V.shape[1]),y.shape[1],i==j)
        result[p] = selected.retained_density_tangent
    return result


def check(frozen,u,v,coarse_step=.004):
    from pno_curvature_water import snapshot, sampled_entry
    psi4.set_options({'freeze_core':frozen,'dlpno__debug':1})
    w = calculation(BASE)
    data = analytic_pno_inputs(w,u,v,verify_first=True)
    adjoint = PNOTruncationCurvature(w)
    value = adjoint.element(data['initial'],data['final'],data['source'])
    @lru_cache(None)
    def model_at(delta):
        return calculation(BASE+np.asarray(delta).reshape(3,3)) if any(delta) else w
    @lru_cache(None)
    def selected(delta):
        response = {}
        snapshot(model_at(delta),w,response)
        return response['frames']
    def stencil(h,projector):
        nodes = [(1,0,1),(-1,0,1),(0,0,-2)] if u==v else [(a,b,a*b/4) for a in (-1,1) for b in (-1,1)]
        result = {p:np.zeros((len(data['overlap'].value),)*2) if projector else np.zeros_like(y.value)
                  for p,y in data['first'].items()}
        for a,b,weight in nodes:
            delta = np.zeros(9); delta[u]+=a*h; delta[v]+=b*h
            for p,y in selected(tuple(delta)).items():
                result[p] += weight/(h*h)*(y@y.T if projector else y)
        return result
    maxima = []
    for projector in (False,True):
        coarse,fine = stencil(coarse_step,projector),stencil(coarse_step/2,projector)
        maximum = 0.
        for p,y in data['first'].items():
            target = (4*fine[p]-coarse[p])/3
            actual = mixed_product('ma,na->mn',y,y).uv if projector else y.uv
            maximum = max(maximum,np.max(abs(actual-target)))
        maxima.append(maximum)
    print('Mixed first-PNO frame/projector comparison',frozen,u,v,coarse_step,maxima,flush=True)
    np.testing.assert_allclose(maxima[0],0,atol=3.e-6,rtol=0)
    estimates = [sampled_entry(w,model_at,u,v,h,adjoint) for h in (coarse_step,coarse_step/2)]
    target = (4*np.asarray(estimates[1])-np.asarray(estimates[0]))/3
    def scalar_difference(h):
        nodes = [(1,0,1),(-1,0,1),(0,0,-2)] if u==v else [(a,b,a*b/4) for a in (-1,1) for b in (-1,1)]
        result = 0.
        for a,b,weight in nodes:
            delta = np.zeros(9); delta[u]+=a*h; delta[v]+=b*h
            result += weight/(h*h)*model_at(tuple(delta)).variable('DLPNO PNO TRUNCATION ERROR')
        return result
    native = (4*scalar_difference(coarse_step/2)-scalar_difference(coarse_step))/3
    print('Analytic PNO nuclear curvature',frozen,u,v,'value',value,'sampled stationary/direct',target,
          'errors',value-target,'native scalar error',value-native,
          'frame/projector errors',maxima,'LMP2 residuals',data['residual_errors'],flush=True)
    np.testing.assert_allclose(value,target,atol=2.e-8,rtol=0)
    np.testing.assert_allclose(value,native,atol=2.e-8,rtol=0)
    # Differencing analytic first projectors avoids cancellation proportional
    # to h^-2 in the AO-density value oracle. Keep the stricter derivative test.
    estimates = []
    for h in (coarse_step,coarse_step/2):
        delta = np.zeros(9); delta[v]=h
        plus = first_projector_derivative(model_at(tuple(delta)),u,w)
        minus = first_projector_derivative(model_at(tuple(-delta)),u,w)
        estimates.append({p:(plus[p]-minus[p])/(2*h) for p in plus})
    error = max(np.max(abs(mixed_product('ma,na->mn',y,y).uv-(4*estimates[1][p]-estimates[0][p])/3))
                for p,y in data['first'].items())
    print('Mixed PNO projector from independent first response',frozen,u,v,error,flush=True)
    np.testing.assert_allclose(error,0,atol=5.e-7,rtol=0)
    # Exercise the actual selected Cholesky-pivot parent and normalized PAOs.
    i,j = [int(w.variable('DLPNO FIRST PNO PROBE '+key)) for key in ('I','J')]
    domain = w.variable('DLPNO FIRST PNO PROBE DOMAIN').np[:,0].astype(int)
    X = w.variable('DLPNO FIRST PNO PROBE PARENT').np
    S,F,L,J,Q = (data[k] for k in ('overlap','fock','lmo','metric','eri'))
    _,N = mixed_paos(data['occupied_action'],S,domain)
    all_ao = np.arange(len(S.value))
    qi,qj = (mixed_pair_three_center(S,L.slice((slice(None),k)),N,all_ao,all_ao,Q) for k in (i,j))
    occ = MixedBlock(*(x[i,i]+x[j,j] for x in data['foo']))
    probe = first_pno_curvature(N,S,F,mixed_pair_exchange(J,qi,qj),occ,X,
                                data['first'][i,j].value,diagonal=(i==j))
    np.testing.assert_allclose(probe.selection.frame.uv,data['first'][i,j].uv,atol=2.e-8,rtol=0)
    print('Selected PAO pivot mixed frame error',np.max(abs(probe.selection.frame.uv-data['first'][i,j].uv)),
          'pair/rank/gap',(i,j),probe.selection.frame.value.shape[1],probe.selection.boundary_gap,flush=True)
    if u!=v:
        reverse = analytic_pno_inputs(w,v,u)
        opposite = adjoint.element(reverse['initial'],reverse['final'],reverse['source'])
        np.testing.assert_allclose(value,opposite,atol=2.e-10,rtol=0)
        for p,y in data['first'].items():
            np.testing.assert_allclose(y.uv,reverse['first'][p].uv,atol=2.e-8,rtol=0)


def check_full_rank(frozen):
    psi4.set_options(dict(freeze_core=frozen,t_cut_pno_mp2=0.,t_cut_pno=0.,t_cut_tno=0.))
    w = calculation(BASE)
    data = analytic_pno_inputs(w,1,5,verify_first=True)
    adjoint = PNOTruncationCurvature(w)
    value = adjoint.element(data['initial'],data['final'],data['source'])
    np.testing.assert_allclose(value,0,atol=1.e-12,rtol=0)
    assert max(np.max(abs(y.uv)) for y in data['first'].values()) > 1.e-3
    print('Full-rank analytic PNO-loss curvature',frozen,value,flush=True)


if __name__ == '__main__':
    from localization_integration import configure
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core',choices=('both','frozen','all'),default='both')
    parser.add_argument('--coordinates',type=int,nargs='+',default=[1,5])
    parser.add_argument('--coarse-step',type=float,default=.004)
    args = parser.parse_args()
    if not args.coarse_step > 0 or any(not 0 <= u < 9 for u in args.coordinates):
        parser.error('Expected positive spacing and Cartesian indices 0 through 8.')
    configure()
    psi4.set_output_file('first_pno_mixed.out',False)
    for frozen in ((True,False) if args.core=='both' else (args.core=='frozen',)):
        psi4.set_options(dict(t_cut_pno_mp2=1.e-5,t_cut_pno=1.e-4,t_cut_tno=1.e-5))
        for u in args.coordinates: check(frozen,u,5,args.coarse_step)
        check_full_rank(frozen)
