"""Full-domain water diagnostic with sampled total mixed lower-order inputs.

Reconstruct both native PNO losses at every fixed-rank displaced geometry.
Scalar differences validate the contraction only, not a production Hessian.
"""
import numpy as np
import psi4
from scipy.optimize import linear_sum_assignment
from first_pno import raw
from lmp2_tangent import calculation, blocks
from water_hvp import BASE
from psi4.driver.procrouting.response.dlpno_pno_curvature import semicanonical_mp2, pno_truncation_energy, PNOTruncationCurvature, lmp2_mixed_residual, lmp2_residual
from psi4.driver.procrouting.response.dlpno_triples_curvature import MixedBlock


def snapshot(w, reference, response=None):
    ref,ref0=w.reference_wavefunction(),reference.reference_wavefunction()
    lm=w.variable('DLPNO RESPONSE ACTIVE LMOS').np
    l0=reference.variable('DLPNO RESPONSE ACTIVE LMOS').np
    cross=psi4.core.MintsHelper(ref).ao_overlap(ref0.basisset(),ref.basisset()).np
    match=l0.T@cross@lm
    _,order=linear_sum_assignment(-abs(match))
    assert min(abs(match[np.arange(len(order)),order])) > .999
    signs=np.sign(match[np.arange(len(order)),order])
    lm=lm[:,order]*signs
    J,Q=raw(ref,ref.get_basisset('DF_BASIS_CC'))
    frame,final,t=(blocks(w,k) for k in ('FRAME','FINAL_FRAME','AMPLITUDE'))
    old,oldfinal=(blocks(reference,k) for k in ('FRAME','FINAL_FRAME'))
    parent,parent0=ref.Ca_subset('AO','VIR').np,ref0.Ca_subset('AO','VIR').np
    a,_,b=np.linalg.svd(parent.T@ref0.S().np@parent0)
    parent=parent@(a@b)
    stages=[{},{}]
    frames={}
    for (i,j),first0 in old.items():
        ai,aj=order[i],order[j]
        pair=tuple(sorted((ai,aj)))
        assert frame[pair].shape==first0.shape
        assert final[pair].shape==oldfinal[i,j].shape
        a,_,b=np.linalg.svd(frame[pair].T@ref0.S().np@first0)
        rot=a@b
        first=frame[pair]@rot
        frames[i,j]=first
        amp=t[pair] if ai<=aj else t[pair].T
        amp=signs[i]*signs[j]*(rot.T@amp@rot)
        oi=lm[:,i]@ref.Fa().np@lm[:,i]
        oj=lm[:,j]@ref.Fa().np@lm[:,j]
        for stage,A,rank in ((0,parent,first.shape[1]),(1,first,final[pair].shape[1])):
            qi=np.einsum('Qmn,m,na->Qa',Q,lm[:,i],A,optimize=True)
            qj=np.einsum('Qmn,m,na->Qa',Q,lm[:,j],A,optimize=True)
            data=dict(exchange=qi.T@np.linalg.solve(J,qj),fock=A.T@ref.Fa().np@A,
                      fov=lm[:,i]@ref.Fa().np@A,occupied_energy=np.asarray(oi),
                      occupied_sum=np.asarray(oi+oj),rank=rank)
            # Remove only roundoff antisymmetry before second differencing.
            data['fock']=(data['fock']+data['fock'].T)/2
            if stage:
                data['amplitudes']=amp
            stages[stage][i,j]=data
    if response is not None:
        response['foo']=lm.T@ref.Fa().np@lm
        response['foo']=(response['foo']+response['foo'].T)/2
        edges=blocks(reference,'RESPONSE_EDGES')
        moved=blocks(w,'RESPONSE_EDGES')
        for p,entries in edges.items():
            native_pair=tuple(sorted(order[list(p)]))
            expected=sorted(tuple(order[int(x)] for x in row) for row in entries)
            actual=sorted(tuple(int(x) for x in row) for row in moved[native_pair])
            # Reversing a target pair swaps ordered source-pair orientation.
            if order[p[0]]>order[p[1]]:
                expected=sorted((l,k,f1,f2) for k,l,f1,f2 in expected)
            # F is symmetric; its stored index orientation can change on relabeling.
            canonical=lambda rows: sorted((k,l,min(a,b),max(a,b)) for k,l,a,b in rows)
            assert canonical(expected)==canonical(actual),(p,expected,actual)
            for k,l,_,_ in entries.astype(int):
                q=min(k,l),max(k,l)
                response[p,q]=frames[p].T@ref.S().np@frames[q]
    return stages


def mixed_inputs(samples,h):
    stages=[{},{}]
    for stage in (0,1):
        for pair,values in samples[0][stage].items():
            args={}
            for key,val in values.items():
                if key=='rank':
                    assert all(s[stage][pair][key]==val for s in samples.values())
                    args[key]=val
                    continue
                x={s:v[stage][pair][key] for s,v in samples.items()}
                first=(x[-2]-8*x[-1]+8*x[1]-x[2])/(12*h)
                second=(-x[2]+16*x[1]-30*x[0]+16*x[-1]-x[-2])/(12*h*h)
                args[key]=MixedBlock(val,first,first,second)
            osum=args.pop('occupied_sum')
            if stage==0:
                args['amplitudes']=semicanonical_mp2(args['exchange'],args['fock'],osum)
            stages[stage][pair]=args
    return stages


def primal(stages):
    converted=[]
    for stage,entries in enumerate(stages):
        result={}
        for pair,data in entries.items():
            args={k:(v if k=='rank' else MixedBlock(v,np.zeros_like(v),np.zeros_like(v),np.zeros_like(v)))
                  for k,v in data.items()}
            osum=args.pop('occupied_sum')
            if stage==0:
                args['amplitudes']=semicanonical_mp2(args['exchange'],args['fock'],osum)
            result[pair]=args
        converted.append(result)
    return pno_truncation_energy(*converted).value


def check(frozen,h=.001):
    psi4.set_options({'freeze_core':frozen})
    w=calculation(BASE)
    responses={0:{}}
    samples={0:snapshot(w,w,responses[0])}
    native={0:w.variable('DLPNO PNO TRUNCATION ERROR')}
    for shift in (-2,-1,1,2):
        coords=BASE.copy()
        coords[1,2]+=shift*h
        other=calculation(coords)
        responses[shift]={}
        samples[shift]=snapshot(other,w,responses[shift])
        native[shift]=other.variable('DLPNO PNO TRUNCATION ERROR')
    for shift,data in samples.items():
        np.testing.assert_allclose(primal(data),native[shift],rtol=0,atol=2.e-10)
    initial,final=mixed_inputs(samples,h)
    result=pno_truncation_energy(initial,final)
    def block(key):
        x={s:v[key] for s,v in responses.items()}
        first=(x[-2]-8*x[-1]+8*x[1]-x[2])/(12*h)
        second=(-x[2]+16*x[1]-30*x[0]+16*x[-1]-x[-2])/(12*h*h)
        return MixedBlock(x[0],first,first,second)
    edges={p:x.astype(int) for p,x in blocks(w,'RESPONSE_EDGES').items()}
    foo=block('foo')
    overlap=lambda p,q: block((p,q))
    stationary=PNOTruncationCurvature(w)
    source=lmp2_mixed_residual(final,foo,edges,overlap)
    value=stationary.element(initial,final,source)
    # Independent analytic energy directional derivative verifies the native source.
    eta={p:x.np for p,x in psi4.core.dlpno_pno_truncation_lmp2_response(w,False).items()}
    rng=np.random.default_rng(731)
    trial={}
    const=lambda x: MixedBlock(x,np.zeros_like(x),np.zeros_like(x),np.zeros_like(x))
    for p,inputs in final.items():
        trial[p]={k:(const(x.value) if isinstance(x,MixedBlock) else x) for k,x in inputs.items()}
        dt=rng.normal(size=inputs['amplitudes'].value.shape)*.01
        if p[0]==p[1]: dt=(dt+dt.T)/2
        trial[p]['amplitudes']=trial[p]['amplitudes']._replace(u=dt)
    derivative=pno_truncation_energy({},trial).u
    dual=sum(np.vdot(eta[p],trial[p]['amplitudes'].u) for p in eta)
    np.testing.assert_allclose(derivative,dual,rtol=0,atol=2.e-10)
    action=lmp2_residual(trial,const(foo.value),edges,lambda p,q:const(overlap(p,q).value))
    primal_error=max(np.max(abs(r.value)) for r in action.values())
    assert primal_error<1.e-9,primal_error
    adjoint_error=dual+sum(np.vdot(stationary.adjoint[p],r.u) for p,r in action.items())
    assert abs(adjoint_error)<2.e-10,adjoint_error
    print('PNO stationary curvature',frozen,h,value,'direct difference',value-result.uv,
          'source error',derivative-dual,'adjoint identity',adjoint_error,flush=True)
    np.testing.assert_allclose(value,result.uv,rtol=0,atol=3.e-7)
    # The stationary API must not read an eliminated mixed amplitude.
    poisoned={p:dict(inputs,amplitudes=inputs['amplitudes']._replace(
        uv=np.full_like(inputs['amplitudes'].value,np.nan))) for p,inputs in final.items()}
    np.testing.assert_allclose(stationary.element(initial,poisoned,source),value,rtol=0,atol=0)
    reference=(-native[2]+16*native[1]-30*native[0]+16*native[-1]-native[-2])/(12*h*h)
    print('PNO curvature water',frozen,'initial',pno_truncation_energy(initial,{}).uv,
          'final',pno_truncation_energy({},final).uv,'total',result.uv,
          'fixed-branch scalar diagnostic',reference,'error',result.uv-reference,flush=True)
    np.testing.assert_allclose(result.uv,reference,rtol=0,atol=2.e-8)


def sampled_entry(w,model_at,u,v,h,adjoint):
    """Supply a stationary PNO entry to the shared semi-analytic column diagnostic.

    Sample lower-order local inputs, never use a sampled mixed LMP2 amplitude
    in the stationary contraction. Keep a direct route only for verification.
    """
    from functools import lru_cache
    @lru_cache(None)
    def sample(a,b):
        delta=np.zeros(9)
        delta[u]+=a*h
        delta[v]+=b*h
        model=model_at(tuple(delta))
        response={}
        data=snapshot(model,w,response)
        np.testing.assert_allclose(primal(data),model.variable('DLPNO PNO TRUNCATION ERROR'),rtol=0,atol=2.e-10)
        return data,response
    def mixed(get):
        base=get(sample(0,0))
        du=(get(sample(1,0))-get(sample(-1,0)))/(2*h)
        dv=(get(sample(0,1))-get(sample(0,-1)))/(2*h)
        if u==v:
            uv=(get(sample(1,0))-2*base+get(sample(-1,0)))/(h*h)
        else:
            uv=sum(a*b*get(sample(a,b)) for a in (-1,1) for b in (-1,1))/(4*h*h)
        return MixedBlock(base,du,dv,uv)
    stages=[{},{}]
    for stage in (0,1):
        for p,values in sample(0,0)[0][stage].items():
            args={key:(value if key=='rank' else mixed(lambda data:data[0][stage][p][key]))
                  for key,value in values.items()}
            osum=args.pop('occupied_sum')
            if stage==0:
                args['amplitudes']=semicanonical_mp2(args['exchange'],args['fock'],osum)
            stages[stage][p]=args
    edges={p:x.astype(int) for p,x in blocks(w,'RESPONSE_EDGES').items()}
    source=lmp2_mixed_residual(stages[1],mixed(lambda data:data[1]['foo']),edges,
                               lambda p,q:mixed(lambda data:data[1][p,q]))
    return adjoint.element(*stages,source),pno_truncation_energy(*stages).uv


def check_mixed(frozen):
    from functools import lru_cache
    psi4.set_options({'freeze_core':frozen})
    w=calculation(BASE)
    adjoint=PNOTruncationCurvature(w)
    @lru_cache(None)
    def model_at(delta):
        return calculation(BASE+np.asarray(delta).reshape(3,3)) if any(delta) else w
    values=[sampled_entry(w,model_at,1,5,h,adjoint) for h in (.002,.001)]
    result=(4*np.asarray(values[1])-np.asarray(values[0]))/3
    print('Mixed O_y/H1_z PNO curvature',frozen,result,'difference',result[0]-result[1],flush=True)
    np.testing.assert_allclose(result[0],result[1],atol=3.e-7,rtol=0)


def check_full_rank(frozen):
    # Called after setting both PNO cutoffs to zero.
    psi4.set_options({'freeze_core':frozen})
    adjoint=PNOTruncationCurvature(calculation(BASE))
    maximum=max(np.max(abs(x)) for x in adjoint.adjoint.values())
    print('Full-rank PNO adjoint',frozen,maximum,flush=True)
    assert maximum<1.e-13


if __name__=='__main__':
    psi4.set_num_threads(1)
    psi4.set_output_file('pno_curvature_water.out',False)
    psi4.set_options(dict(scf__diis=False,scf_initial_accelerator='NONE',basis='cc-pvdz',reference='rhf',scf_type='df',df_basis_cc='cc-pvdz-ri',
                          df_basis_scf='cc-pvdz-jkfit',e_convergence=1.e-13,d_convergence=1.e-13,
                          r_convergence=1.e-11,solver_convergence=1.e-12,dlpno_maxiter=600,
                          local_convergence=1.e-14,local_gradient_convergence=1.e-8,
                          dlpno_keep_lmp2_response=True,dlpno_gradient_model=True,
                          dlpno_local_orbitals='BOYS',t_cut_pno_mp2=1.e-5,t_cut_pno=1.e-4,
                          t_cut_trace_mp2=0.,t_cut_energy_mp2=0.,t_cut_trace=0.,t_cut_energy=0.,
                          t_cut_pairs=1.e-9,t_cut_pairs_mp2=1.e-9,dlpno_keep_triples_response=True,
                          min_tnos=3,t_cut_tno=1.e-5))
    psi4.set_options({k:0. for k in ('t_cut_do t_cut_do_pre t_cut_do_ij t_cut_do_uv t_cut_mkn '
                                      't_cut_pre t_cut_clmo t_cut_cpao dlpno_ao_ints_tol').split()})
    for frozen in (True,False):
        check(frozen)
        check(frozen,.0005)

    for frozen in (True,False):
        check_mixed(frozen)
    psi4.set_options(dict(t_cut_pno=0.,t_cut_pno_mp2=0.,t_cut_tno=0.,t_cut_do_triples=0.,
                          t_cut_mkn_triples=0.,f_cut=0.,f_cut_t=0.))
    for frozen in (True,False):
        check_full_rank(frozen)
