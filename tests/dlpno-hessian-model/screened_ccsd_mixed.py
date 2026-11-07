"""Native-domain mixed CCSD molecular response and local triples delivery."""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import numpy as np
import psi4
from scipy.linalg import solve_sylvester
from screened_first_pno import fixture,align_snapshot
from screened_final_pno import independent_tangent
from screened_corrections import mixed_inputs
from lmp2_tangent import blocks
from triples_curvature import aligned_inputs
from ccsd_mixed import pack
from psi4.driver.procrouting.response.dlpno_screened_triples import ScreenedTriplesInputs
from psi4.driver.procrouting.response.dlpno_screened_pno import screened_pno_response
from psi4.driver.procrouting.response.dlpno_screened_ccsd import ScreenedFitting,screened_ccsd_response
from psi4.driver.procrouting.response.dlpno_ccsd_mixed import mixed_ccsd_residual
from psi4.driver.procrouting.response.dlpno_orbital import ccsd_pair_df_factors
from psi4.driver.procrouting.response.dlpno_ccsd_response import (
    ccsd_residual_source,coupled_ccsd_tangent,transported_ccsd_residual,ccsd_amplitude_curvature)
from psi4.driver.procrouting.response.dlpno_triples_curvature import MixedBlock


def analytic(w,u,v):
    S,F,L,D,_,df=mixed_inputs(w,u,v)
    pno=screened_pno_response(w,S,F,L,D,df)
    out=screened_ccsd_response(w,pno,S,F,L,D,df)
    return out,pno,(S,F,L,D,df)


def first(w,u):
    # Old first-PNO, native first LMP2 and old final-PNO selection kernels.
    lower={};independent_tangent(w,u,lower)
    S,F,L,D,_,df=mixed_inputs(w,u,u)
    frame=lower['ccsd_frames']
    def block(x,dx):return MixedBlock(x,dx,np.zeros_like(x),np.zeros_like(x))
    def electronic(key):
        if key=='lmo':return block(L.value,L.u)
        if key=='fock':return block(F.value,F.u)
        return block(frame.frames[key[1]],frame.tangents[key[1]])
    fitting=ScreenedFitting(w,electronic,S,D,df)
    @lru_cache(None)
    def factors(i,j):
        oo,ov,vv,*_=fitting.projected((i,j));metric,_=df(fitting.pair_aux[i,j])
        return ccsd_pair_df_factors(metric.value,metric.u,[x.value for x in (oo,ov,vv)],[x.u for x in (oo,ov,vv)])
    def exchange(i,j):
        p=tuple(sorted((i,j)));_,ov,_,g,_,_=fitting.projected(p);dom=fitting.pair_occ[p]
        from psi4.driver.procrouting.response.dlpno_triples_curvature import mixed_product as C
        K=C('Qa,QR,Rb->ab',ov.slice((slice(None),dom.index(i),slice(None))),g,
            ov.slice((slice(None),dom.index(j),slice(None))))
        return K.value,K.u
    source=ccsd_residual_source(w,frame,factors,exchange,
        lambda i,j,k:tuple(x.u for x in fitting.mixed_integrals(i,j,k)),L.u,F.u)
    return coupled_ccsd_tangent(w,*source),frame,source


def transported_first(w,ref,rows,u):
    (one,two),frames,_=first(w,u)
    state=aligned_inputs(w,ref,rows)
    native=blocks(w,'CCSD_DOUBLES');y0=blocks(ref,'FINAL_FRAME');S0=ref.reference_wavefunction().S().np
    out1=[None]*len(state.order);out2={}
    for p,Q in state.pair_rotations.items():
        i,j=p;a,b=state.order[i],state.order[j];q=min(a,b),max(a,b)
        A=frames.frames[q].T@S0@y0[p];dA=frames.tangents[q].T@S0@y0[p]
        H=Q.T@A;H=(H+H.T)/2
        omega=solve_sylvester(H,H,Q.T@dA-dA.T@Q)
        np.testing.assert_allclose(omega+omega.T,0.,atol=2.e-12,rtol=0.)
        t,dt=native[q],two[q]
        if a>b:t,dt=t.T,dt.T
        aligned=Q.T@t@Q
        out2[p]=state.signs[i]*state.signs[j]*(Q.T@dt@Q+omega.T@aligned+aligned@omega)
        if i==j:
            dom,ts,_=psi4.core.dlpno_ccsd_df_pair_state(w,int(a),int(a));ti=ts[dom.index(a)].np.ravel()
            out1[i]=state.signs[i]*(Q.T@one[a]+omega.T@(Q.T@ti))
    return out1,out2


def amplitude_check(w,lower,electronic,base):
    S,F,L,D,df=lower
    constant=lambda x:MixedBlock(x.value,*[np.zeros_like(x.value) for _ in range(3)])
    rng=np.random.default_rng(3982);directions={}
    def trial(key):
        x=electronic(key)
        if isinstance(key,tuple) and key[0] in ('t1','t2'):
            if key not in directions:
                u,v=(rng.normal(size=x.value.shape)*.01 for _ in range(2))
                if key[0]=='t2' and key[1][0]==key[1][1]:u,v=(u+u.T)/2,(v+v.T)/2
                directions[key]=MixedBlock(x.value,u,v,np.zeros_like(x.value))
            return directions[key]
        return constant(x)
    cdf=lambda a:tuple(constant(x) for x in df(a))
    occupied=w.reference_wavefunction().Ca_subset('AO','OCC').np
    density=occupied@occupied.T
    fit=ScreenedFitting(w,trial,constant(S),lambda p:MixedBlock(*(density@x for x in p)),cdf)
    result=mixed_ccsd_residual(w,trial,constant(S),cdf,fitted=fit)
    def direction(k):return [directions['t1',i][k] for i in range(len(base[0]))],{p:directions['t2',p][k] for p in base[1]}
    def evaluator(one,two):
        return transported_ccsd_residual(w,one,two,tuple(range(len(one))),np.ones(len(one)),
                                        {p:np.eye(len(x)) for p,x in two.items()})
    exact=ccsd_amplitude_curvature(evaluator,base,direction(1),direction(2))
    error=float(np.max(abs(pack(result,3)-pack(exact))))
    np.testing.assert_allclose(error,0.,atol=2.e-11,rtol=0.)
    def invalid(key):
        x=trial(key)
        return MixedBlock(x.value,x.u,x.v,np.ones_like(x.value)) if key==('t1',0) else x
    try:mixed_ccsd_residual(w,invalid,constant(S),cdf,fitted=fit)
    except ValueError as exc:assert 'affine' in str(exc)
    else:raise AssertionError('Accepted mixed CCSD amplitudes.')
    after=blocks(w,'CCSD_DOUBLES')
    for p in after:np.testing.assert_array_equal(after[p],base[1][p])
    return error


def run(args):
    mol,w=fixture(args.system,args.localization,False,final_cutoff=1.e-3,tight=True,screened_triplets=args.screened_triplets,tno_cutoff=0. if args.full_tno else 1.e-5)
    xyz=mol.geometry().np.copy();rows=psi4.core.dlpno_triples_response_state(w)
    psi4.set_options({'dlpno__debug':0})
    u,v=(2,18) if args.system=='chain' else (5,12)
    result=dict(system=args.system,localization=args.localization,screened_triplets=args.screened_triplets,full_tno=args.full_tno,step_bohr=args.step,checks=[])
    def save():Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    @lru_cache(None)
    def model(k,h):
        if h==0:return w
        m=mol.clone();d=xyz.copy();d.flat[k]+=h;m.set_geometry(psi4.core.Matrix.from_array(d));m.update_geometry()
        _,other=psi4.energy('dlpno-ccsd(t)',molecule=m,return_wfn=True);align_snapshot(other,w)
        return other
    for a,c in ([(u,v)] if args.stage=='replay' else [(u,u),(u,v),(v,v)]):
        out,pno,lower=analytic(w,a,c)
        replay=out['residual'];base=out['base'];fit=out['fitting'];electronic=out['electronic'];S,F,L,D,df=lower
        native=transported_ccsd_residual(w,*base,tuple(range(len(base[0]))),np.ones(len(base[0])),
                                         {p:np.eye(len(t)) for p,t in base[1].items()})
        value=float(np.max(abs(pack(replay,0)-pack(native))))
        np.testing.assert_allclose(value,0.,atol=2.e-10,rtol=0.)
        mixed_jk=0.
        for p,dom in fit.pair_occ.items():
            for i,j in (p,p[::-1]):
                for k in dom:
                    J,K,*_=psi4.core.dlpno_ccsd_mixed_state(w,int(i),int(j),int(k))
                    actual=fit.mixed_integrals(i,j,k)
                    mixed_jk=max(mixed_jk,np.max(abs(actual[0].value-J.np)),np.max(abs(actual[1].value-K.np)))
        np.testing.assert_allclose(mixed_jk,0.,atol=2.e-10,rtol=0.)
        metrics=dict(coordinates=[a,c],residual_errors=out['residual_errors'],native_replay_error=value,mixed_integral_replay_error=float(mixed_jk))
        result['checks'].append(metrics);save();print('replay',metrics,flush=True)
        triples=ScreenedTriplesInputs(w,fit)
        te=0.;gram=0.
        for r,row in triples.rows.items():
            bs=triples.blocks(r);n=row[2].np.shape[1]
            te=max(te,np.max(abs(bs[0].value-row[4].np.reshape((n,)*3))),
                   np.max(abs(bs[1].value-row[7].np.reshape((n,)*3))))
            from psi4.driver.procrouting.response.dlpno_triples_curvature import mixed_product as C
            norm=C('ma,mn,nb->ab',triples.frame(r),S,triples.frame(r))
            gram=max(gram,np.max(abs(norm.value-np.eye(n))),*[np.max(abs(x)) for x in norm[1:]])
        metrics.update(triples_native_replay_error=float(te),tno_metric_error=float(gram),
            triplets=[dict(labels=r,paos=triples.basis[r]['PAOS'].np.size,
                parent_rank=triples.parent(r).value.shape[1],tno_rank=row[2].np.shape[1],aux=len(row[1]))
                for r,row in triples.rows.items()])
        save();print('triples replay',te,gram,flush=True)
        np.testing.assert_allclose(te,0.,atol=2.e-10,rtol=0.)
        np.testing.assert_allclose(gram,0.,atol=2.e-8,rtol=0.)
        def swap(x):return MixedBlock(x.value,x.v,x.u,x.uv)
        reverse=lambda k:swap(electronic(k))
        rdf=lambda aux:tuple(swap(x) for x in df(aux))
        other=ScreenedFitting(w,reverse,swap(S),lambda p:swap(D(swap(p))),rdf)
        rev=mixed_ccsd_residual(w,reverse,swap(S),rdf,fitted=other)
        metrics['ordered_swap_error']=float(np.max(abs(pack(replay,3)-pack(rev,3))))
        np.testing.assert_allclose(metrics['ordered_swap_error'],0.,atol=2.e-11,rtol=0.)
        reversed_triples=ScreenedTriplesInputs(w,other)
        ordered_triples=0.
        for r,row in triples.rows.items():
            ordered_triples=max(ordered_triples,float(np.max(abs(triples.frame(r).uv-reversed_triples.frame(r).uv))),
                *[float(np.max(abs(x.uv-y.uv))) for x,y in zip(triples.blocks(r),reversed_triples.blocks(r))])
            for _,_,i,j,k,_ in row[5]:
                q=tuple(sorted((i,j,k)))
                ordered_triples=max(ordered_triples,float(np.max(abs(triples.edge(r,q).uv-reversed_triples.edge(r,q).uv))))
        metrics['triples_ordered_swap_error']=ordered_triples
        np.testing.assert_allclose(ordered_triples,0.,atol=2.e-10,rtol=0.)
        if a==c==u or args.stage=='replay':
            metrics['amplitude_algebra_error']=amplitude_check(w,lower,electronic,base)
        save()
        if args.stage=='replay':continue
        for k,tangent in ((a,out['tangent_u']),(c,out['tangent_v'])):
            old,_,source=first(w,k)
            error=float(np.max(abs(pack(old)-pack(tangent))))
            np.testing.assert_allclose(error,0.,atol=2.e-9,rtol=0.)
            metrics['independent_first_'+str(k)]=error
        estimates=[]
        for h in (args.step,args.step/2):
            plus,minus=transported_first(model(c,h),w,rows,a),transported_first(model(c,-h),w,rows,a)
            estimates.append(([(x-y)/(2*h) for x,y in zip(plus[0],minus[0])],
                              {p:(x-minus[1][p])/(2*h) for p,x in plus[1].items()}))
        tuv=([(4*x-y)/3 for x,y in zip(estimates[1][0],estimates[0][0])],
             {p:(4*x-estimates[0][1][p])/3 for p,x in estimates[1][1].items()})
        matrix=psi4.core.Matrix.from_array
        expected=psi4.core.dlpno_ccsd_amplitude_action(w,[matrix(x[:,None]) for x in tuv[0]],{p:matrix(x) for p,x in tuv[1].items()})
        expected=([x.np.ravel() for x in expected[0]],{p:x.np for p,x in expected[1].items()})
        error=float(np.max(abs(pack(replay,3)+pack(expected))))
        metrics['mixed_source_error']=error;save();print('mixed source',a,c,error,flush=True)
        np.testing.assert_allclose(error,0.,atol=2.e-7,rtol=0.)
    result['passed']=True;save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--system',choices=('chain','lih'),default='chain')
    p.add_argument('--localization',choices=('BOYS','PIPEK_MEZEY'),default='BOYS')
    p.add_argument('--step',type=float,default=.004);p.add_argument('--stage',choices=('all','replay'),default='all')
    p.add_argument('--full-tno',action='store_true')
    p.add_argument('--screened-triplets',action='store_true')
    p.add_argument('--output',default='screened_ccsd_results.json');args=p.parse_args()
    psi4.set_output_file('screened_ccsd.out',False);run(args)
