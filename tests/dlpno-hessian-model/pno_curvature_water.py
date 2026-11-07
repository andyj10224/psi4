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
from psi4.driver.procrouting.response.dlpno_pno_curvature import semicanonical_mp2, pno_truncation_energy
from psi4.driver.procrouting.response.dlpno_triples_curvature import MixedBlock


def snapshot(w, reference):
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
    for (i,j),first0 in old.items():
        ai,aj=order[i],order[j]
        pair=tuple(sorted((ai,aj)))
        assert frame[pair].shape==first0.shape
        assert final[pair].shape==oldfinal[i,j].shape
        a,_,b=np.linalg.svd(frame[pair].T@ref0.S().np@first0)
        rot=a@b
        first=frame[pair]@rot
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
    samples={0:snapshot(w,w)}
    native={0:w.variable('DLPNO PNO TRUNCATION ERROR')}
    for shift in (-2,-1,1,2):
        coords=BASE.copy()
        coords[1,2]+=shift*h
        other=calculation(coords)
        samples[shift]=snapshot(other,w)
        native[shift]=other.variable('DLPNO PNO TRUNCATION ERROR')
    for shift,data in samples.items():
        np.testing.assert_allclose(primal(data),native[shift],rtol=0,atol=2.e-10)
    initial,final=mixed_inputs(samples,h)
    result=pno_truncation_energy(initial,final)
    reference=(-native[2]+16*native[1]-30*native[0]+16*native[-1]-native[-2])/(12*h*h)
    print('PNO curvature water',frozen,'initial',pno_truncation_energy(initial,{}).uv,
          'final',pno_truncation_energy({},final).uv,'total',result.uv,
          'fixed-branch scalar diagnostic',reference,'error',result.uv-reference,flush=True)
    np.testing.assert_allclose(result.uv,reference,rtol=0,atol=2.e-8)


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
