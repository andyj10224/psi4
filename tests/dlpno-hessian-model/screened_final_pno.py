"""Screened coupled-LMP2/final-PNO response and both PNO-loss curvatures.

Independent first-order local refits/projectors plus the native first LMP2
solve provide derivative references; displaced native energies independently
check the PNO-loss curvature. No complete screened CCSD(T) Hessian is assembled.
"""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import numpy as np
import psi4
from screened_first_pno import fixture, independent_first, align_snapshot
from screened_corrections import mixed_inputs
from lmp2_tangent import blocks
from final_pno_mixed import check_source_contract
from psi4.driver.procrouting.response.dlpno_orbital import LMP2Tangent, final_pno_tangent
from psi4.driver.procrouting.response.dlpno_screened_pno import screened_pno_response
from psi4.driver.procrouting.response.dlpno_pno_curvature import pno_truncation_energy
from psi4.driver.procrouting.response.dlpno_triples_curvature import mixed_product


def response(w,u,v):
    S,F,L,occupied,_,df=mixed_inputs(w,u,v)
    return screened_pno_response(w,S,F,L,occupied,df),S


def independent_tangent(w,u):
    lower={}
    independent_first(w,u,lower)
    matrix=psi4.core.Matrix.from_array
    dt=psi4.core.dlpno_lmp2_tangent(w,
        {p:matrix(x) for p,x in lower['frame_derivatives'].items()},
        {p:matrix(x) for p,x in lower['exchange_derivatives'].items()},
        *(matrix(lower[k]) for k in ('overlap_derivative','fock_derivative','occupied_fock_derivative')))
    dt={p:x.np.copy() for p,x in dt.items()}
    selected=final_pno_tangent(w,LMP2Tangent(lower['frame_derivatives'],dt))
    tensors={}
    for p,t in blocks(w,'AMPLITUDE').items():
        y,dy=lower['frames'][p],lower['frame_derivatives'][p]
        tensors[p]=dy@t@y.T+y@dt[p]@y.T+y@t@dy.T
    return tensors,{p:x.retained_density_tangent for p,x in selected.items()}


def run(args):
    mol,w=fixture(args.system,args.localization,args.full_rank,final_cutoff=args.final_cutoff,tight=True)
    xyz=mol.geometry().np.copy()
    first,final,amps=(blocks(w,k) for k in ('FRAME','FINAL_FRAME','AMPLITUDE'))
    ranks=[dict(pair=list(p),first=y.shape[1],final=final[p].shape[1]) for p,y in first.items()]
    assert bool(any(x['final']<x['first'] for x in ranks)) != args.full_rank
    domains,_=psi4.core.dlpno_first_pno_state(w)
    nbf=w.basisset().nbf(); naux=w.get_basisset('DF_BASIS_CC').nbf()
    assert min(map(len,domains['AO_OCC']))<nbf and min(map(len,domains['AO_VIR']))<nbf
    assert max(map(len,domains['PAIR_AUX']))<naux and max(map(len,domains['PAIR_PAO']))<nbf
    edges={p:x.astype(int) for p,x in blocks(w,'RESPONSE_EDGES').items()}
    graph=dict(edges=sum(len(x) for x in edges.values()),
               reversed_sources=sum(k>l for x in edges.values() for k,l,_,_ in x),
               rectangular_edges=sum(first[p].shape[1]!=first[min(k,l),max(k,l)].shape[1]
                                     for p,x in edges.items() for k,l,_,_ in x))
    graph={k:int(v) for k,v in graph.items()}
    if args.system=='chain': assert all(graph.values())
    result=dict(graph=graph,edges={str(p):x.tolist() for p,x in edges.items()},system=args.system,localization=args.localization,full_rank=args.full_rank,
                frozen_orbitals=w.reference_wavefunction().nfrzc(),step_bohr=args.step,
                first_cutoff=0. if args.full_rank else 1.e-5,
                final_cutoff=0. if args.full_rank else args.final_cutoff,coordinates_bohr=xyz.tolist(),domains=domains,ranks=ranks,
                scope='Screened coupled LMP2, final PNOs and PNO loss only.',checks=[])
    print('ranks',ranks,flush=True)
    # Persist a partial record too if an assertion fails later.
    def save(): Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    save()
    psi4.set_options({'dlpno__debug':0})
    u,v=(2,18) if args.system=='chain' else (5,12)
    coordinates=[(u,v)] if args.stage=='replay' or args.full_rank else [(u,u),(u,v),(v,v)]
    @lru_cache(None)
    def model(delta):
        if not any(delta): return w
        m=mol.clone();m.set_geometry(psi4.core.Matrix.from_array(xyz+np.array(delta).reshape(xyz.shape)));m.update_geometry()
        _,other=psi4.energy('dlpno-ccsd(t)',molecule=m,return_wfn=True)
        align_snapshot(other,w)
        return other
    def delta(k,h):
        d=np.zeros(xyz.size);d[k]=h;return tuple(d)
    @lru_cache(None)
    def oracle(displacement,k):
        other=model(displacement)
        _,order=align_snapshot(other,w)
        ref,ref0=other.reference_wavefunction(),w.reference_wavefunction()
        cross=psi4.core.MintsHelper(ref).ao_overlap(ref0.basisset(),ref.basisset()).np
        match=w.variable('DLPNO RESPONSE ACTIVE LMOS').np.T@cross@other.variable('DLPNO RESPONSE ACTIVE LMOS').np
        signs=np.sign(match[np.arange(len(order)),order])
        tensor,projector=independent_tangent(other,k)
        amp,proj={},{}
        for p in first:
            a,b=order[list(p)];q=min(a,b),max(a,b)
            amp[p]=signs[p[0]]*signs[p[1]]*(tensor[q] if a<=b else tensor[q].T)
            proj[p]=projector[q]
        return amp,proj
    def scalar(h,a,c):
        nodes=[(1,0,1),(-1,0,1),(0,0,-2)] if a==c else [(i,j,i*j/4) for i in (-1,1) for j in (-1,1)]
        total=0.
        for i,j,weight in nodes:
            d=np.zeros(xyz.size);d[a]+=i*h;d[c]+=j*h
            total+=weight*model(tuple(d)).variable('DLPNO PNO TRUNCATION ERROR')/h**2
        return total
    for a,c in coordinates:
        out,S=response(w,a,c)
        metrics=dict(coordinates=[a,c],residual_errors=out.residual_errors,
                     native_mixed_residual=w.variable('DLPNO LMP2 MIXED TANGENT RESIDUAL'),
                     first_replay=out.first_response.replay_errors,
                     loss=list(out.loss),stationary_loss=out.stationary_loss,
                     first_loss=list(pno_truncation_energy(out.initial,{})),
                     final_loss=list(pno_truncation_energy({},out.final)))
        result['checks'].append(metrics);save()
        print('replay',a,c,metrics,flush=True)
        if not args.full_rank: assert abs(metrics['final_loss'][0])>1.e-10
        else: np.testing.assert_allclose(out.loss,0.,atol=1.e-12,rtol=0.)
        ao_t={p:mixed_product('ma,ab,nb->mn',out.first_response.pairs[p].selection.frame,
                              x['amplitudes'],out.first_response.pairs[p].selection.frame) for p,x in out.final.items()}
        ao_p={p:mixed_product('ma,na->mn',x.frame,x.frame) for p,x in out.final_selection.items()}
        gram_error=max(np.max(abs(x)) for sel in out.final_selection.values()
                       for x in mixed_product('ma,mn,nb->ab',sel.frame,S,sel.frame)[1:])
        idem_error=max(np.max(abs(x-y)) for p,P in ao_p.items()
                       for x,y in zip(mixed_product('ma,ab,bn->mn',P,S,P),P))
        metrics.update(metric_error=float(gram_error),projector_idempotency_error=float(idem_error),
                       final_boundary_gaps={str(p):float(x.boundary_gap) if np.isfinite(x.boundary_gap) else None
                                            for p,x in out.final_selection.items()})
        np.testing.assert_allclose([gram_error,idem_error],0.,atol=2.e-8,rtol=0.)
        check_source_contract(w,dict(source=out.mixed_source))
        if args.stage=='replay':save();continue
        for index,k in enumerate((a,c),1):
            expected=oracle(delta(k,0.),k)
            errors=[max(np.max(abs(x[index]-expected[n][p])) for p,x in actual.items())
                    for n,actual in enumerate((ao_t,ao_p))]
            metrics['independent_first_'+str(index)]=[float(x) for x in errors]
            print('independent first',k,errors,flush=True)
            np.testing.assert_allclose(errors[0],0.,atol=2.e-9,rtol=0.)
            np.testing.assert_allclose(errors[1],0.,atol=2.e-8,rtol=0.)
        for n,(key,actual,tol) in enumerate((('ao_amplitude',ao_t,2.e-7),('final_projector',ao_p,2.e-6))):
            estimates=[]
            for h in (args.step,args.step/2):
                plus,minus=oracle(delta(c,h),a)[n],oracle(delta(c,-h),a)[n]
                estimates.append({p:(plus[p]-minus[p])/(2*h) for p in actual})
            error=max(np.max(abs(x.uv-(4*estimates[1][p]-estimates[0][p])/3)) for p,x in actual.items())
            metrics['mixed_'+key+'_error']=float(error);save()
            print('mixed',a,c,key,error,flush=True)
            np.testing.assert_allclose(error,0.,atol=tol,rtol=0.)
        for index,k in enumerate((a,c),1):
            estimates=[(model(delta(k,h)).variable('DLPNO PNO TRUNCATION ERROR')-
                         model(delta(k,-h)).variable('DLPNO PNO TRUNCATION ERROR'))/(2*h)
                       for h in (args.step,args.step/2)]
            error=out.loss[index]-(4*estimates[1]-estimates[0])/3
            metrics['native_first_loss_error_'+str(index)]=float(error)
            np.testing.assert_allclose(error,0.,atol=2.e-9,rtol=0.)
        estimates=[scalar(h,a,c) for h in (args.step,args.step/2)]
        error=out.loss.uv-(4*estimates[1]-estimates[0])/3
        metrics['native_mixed_loss_error']=float(error);save()
        print('mixed loss',a,c,error,flush=True)
        np.testing.assert_allclose(error,0.,atol=2.e-8,rtol=0.)
        if a!=c:
            rev,_=response(w,c,a)
            errors=[max(np.max(abs(x['amplitudes'].uv-rev.final[p]['amplitudes'].uv)) for p,x in out.final.items()),
                    max(np.max(abs(x.frame.uv-rev.final_selection[p].frame.uv)) for p,x in out.final_selection.items()),
                    abs(out.loss.uv-rev.loss.uv)]
            metrics['ordered_errors']=[float(x) for x in errors]
            for e,tol in zip(errors,(2.e-9,2.e-8,2.e-10)):np.testing.assert_allclose(e,0.,atol=tol,rtol=0.)
        for p,t in blocks(w,'AMPLITUDE').items(): np.testing.assert_array_equal(t,amps[p])
        metrics['passed']=True;save()
    result['validation_stage']=args.stage;result['passed']=True;save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--system',choices=('chain','lih'),default='chain')
    p.add_argument('--localization',choices=('BOYS','PIPEK_MEZEY'),default='BOYS')
    p.add_argument('--full-rank',action='store_true')
    p.add_argument('--final-cutoff',type=float,default=1.e-3)
    p.add_argument('--step',type=float,default=.004)
    p.add_argument('--stage',choices=('all','replay'),default='all')
    p.add_argument('--output',default='screened_final_pno_results.json')
    args=p.parse_args()
    if args.step<=0. or args.final_cutoff<=0.:p.error('Expected positive displacement spacing and cutoff.')
    psi4.set_output_file('screened_final_pno.out',False)
    run(args)
