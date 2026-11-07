"""Screened TNO/moment mixed inputs versus an independent displaced affine-CCSD path.

Native orbitals/final PNOs and raw integrals are recomputed only in this oracle.
Trial TNO densities use fixed reference CCSD amplitudes plus their first tangents;
no relaxed mixed CCSD amplitudes contaminate the stationary source comparison.
"""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import numpy as np
import psi4
from first_pno import raw
from screened_corrections import normalized_paos
from screened_first_pno import fixture,align_snapshot
from screened_ccsd_mixed import analytic
from triples_curvature import aligned_inputs,trial_tno_frame
from lmp2_tangent import blocks
from psi4.driver.procrouting.response.dlpno_screened_ccsd import ScreenedFitting
from psi4.driver.procrouting.response.dlpno_screened_triples import ScreenedTriplesInputs
from psi4.driver.procrouting.response.dlpno_triples_curvature import MixedBlock


def constant(x):return MixedBlock(x,*[np.zeros_like(x) for _ in range(3)])


def sample(model,ref,reference,base,du,dv):
    """Pure values on a trial amplitude path, in fixed reference polar frames."""
    state=aligned_inputs(model,ref,reference.rows)
    other=model.reference_wavefunction();S=other.S().np;F=other.Fa().np
    native_basis=psi4.core.dlpno_triples_basis_state(model)
    N=normalized_paos(other);native_frames=blocks(model,'FINAL_FRAME')
    data={'lmo':model.variable('DLPNO RESPONSE ACTIVE LMOS').np,'fock':F}
    for p,Q in state.pair_rotations.items():
        a,b=state.order[p[0]],state.order[p[1]];q=tuple(sorted((a,b)))
        data['pno',q]=native_frames[q]
        t=base(('t2',p));x=t.value+du*t.u+dv*t.v
        x=state.signs[p[0]]*state.signs[p[1]]*(Q@x@Q.T)
        data['t2',q]=x if a<=b else x.T
        if a==b:
            t=base(('t1',p[0]));data['t1',a]=state.signs[p[0]]*(Q@(t.value+du*t.u+dv*t.v))
    occupied=other.Ca_subset('AO','OCC').np;density=occupied@occupied.T
    J,Q=raw(other,other.get_basisset('DF_BASIS_CC'))
    def df(aux):return constant(J[np.ix_(aux,aux)]),constant(Q[list(aux)])
    fit=ScreenedFitting(model,lambda k:constant(data[k]),constant(S),
                        lambda p:MixedBlock(*(density@x for x in p)),df)
    triples=ScreenedTriplesInputs(model,fit)
    frames={};mapping={}
    for r,fields in reference.basis.items():
        q=tuple(sorted(np.asarray(state.order)[list(r)]));mapping[r]=q
        np.testing.assert_array_equal(fields['PAOS'].np,native_basis[q]['PAOS'].np)
        ids=fields['PAOS'].np[:,0].astype(int)
        B=N[:,ids]@fields['PARENT'].np
        eig,U=np.linalg.eigh(B.T@S@B);parent=B@((U*eig**(-.5))@U.T)
        native=N[:,ids]@native_basis[q]['PARENT'].np
        np.testing.assert_allclose(parent@parent.T,native@native.T,atol=2.e-8,rtol=0.)
        pairs=((r[0],r[1]),(r[1],r[2]),(r[0],r[2]))
        t2={p:base(('t2',p)).value+du*base(('t2',p)).u+dv*base(('t2',p)).v for p in pairs}
        frames[q]=trial_tno_frame(r,parent,S,{p:state.electronic['pno',p] for p in pairs},t2,
                                 reference.rows[r][2].np,ref.reference_wavefunction().S().np)
    triples.frame=lambda q:constant(frames[q])
    result={}
    for r,q in mapping.items():
        axes=np.argsort(np.argsort(np.asarray(state.order)[list(r)],kind='stable'))
        W,V,vf=triples.blocks(q)
        phase=np.prod(state.signs[list(r)])
        result['frame',r]=frames[q]
        result['W',r]=phase*W.value.transpose(axes)
        result['V',r]=phase*V.value.transpose(axes)
        result['F',r]=vf.value
        for _,_,i,j,k,_ in reference.rows[r][5]:
            s=tuple(sorted((i,j,k)))
            result['edge',r,s]=frames[q].T@S@frames[mapping[s]]
    lm=state.electronic['lmo'];result['occupied']=lm.T@F@lm
    return result


def run(args):
    mol,w=fixture(args.system,args.localization,False,final_cutoff=1.e-3,tight=True,screened_triplets=True)
    psi4.set_options({'dlpno__debug':0})
    xyz=mol.geometry().np.copy();u,v=(2,18) if args.system=='chain' else (5,12)
    results=dict(system=args.system,localization=args.localization,step_bohr=args.step,checks=[])
    def save():Path(args.output).write_text(json.dumps(results,indent=2)+'\n')
    @lru_cache(None)
    def model(delta):
        if not any(delta):return w
        m=mol.clone();m.set_geometry(psi4.core.Matrix.from_array(xyz+np.array(delta).reshape(xyz.shape)));m.update_geometry()
        _,other=psi4.energy('dlpno-ccsd(t)',molecule=m,return_wfn=True);align_snapshot(other,w)
        return other
    for a,c in ((u,u),(u,v),(v,v)):
        out,pno,lower=analytic(w,a,c);triples=ScreenedTriplesInputs(w,out['fitting']);expected={}
        if not triples.rows:raise AssertionError('No triples in this test.')
        for r,row in triples.rows.items():
            expected['frame',r]=triples.frame(r).uv
            for key,x in zip(('W','V','F'),triples.blocks(r)):expected[key,r]=x.uv
            for _,_,i,j,k,_ in row[5]:
                s=tuple(sorted((i,j,k)));expected['edge',r,s]=triples.edge(r,s).uv
        n=out['electronic']('lmo').value.shape[1]
        expected['occupied']=np.array([[triples.occupied(i,j).uv for j in range(n)] for i in range(n)])
        base=sample(w,w,triples,out['electronic'],0.,0.)
        for r,row in triples.rows.items():
            np.testing.assert_allclose(base['frame',r],row[2].np,atol=2.e-8,rtol=0.)
            for key,index in (('W',4),('V',7)):
                np.testing.assert_allclose(base[key,r],row[index].np.reshape(base[key,r].shape),atol=2.e-10,rtol=0.)
        estimates=[]
        for h in (args.step,args.step/2):
            nodes=((1,0,1),(0,0,-2),(-1,0,1)) if a==c else ((1,1,1),(1,-1,-1),(-1,1,-1),(-1,-1,1))
            estimate={k:np.zeros_like(x) for k,x in expected.items()}
            for da,dc,weight in nodes:
                delta=np.zeros(xyz.size);delta[a]+=da*h;delta[c]+=dc*h
                data=sample(model(tuple(delta)),w,triples,out['electronic'],da*h,dc*h)
                for k in estimate:estimate[k]+=data[k]*(weight/(h*h*(1 if a==c else 4)))
            estimates.append(estimate)
        errors={kind:0. for kind in ('frame','W','V','F','edge','occupied')}
        raw_errors=errors.copy()
        for k,x in expected.items():
            kind=k if isinstance(k,str) else k[0]
            errors[kind]=max(errors[kind],float(np.max(abs(x-(4*estimates[1][k]-estimates[0][k])/3))))
            raw_errors[kind]=max(raw_errors[kind],float(np.max(abs(x-estimates[1][k]))))
        record=dict(coordinates=[a,c],mixed_errors=errors,raw_half_step_errors=raw_errors,
                    triplets=len(triples.rows),edges=sum(k[0]=='edge' for k in expected if isinstance(k,tuple)))
        results['checks'].append(record);save();print(record,flush=True)
        for kind,error in errors.items():
            np.testing.assert_allclose(error,0.,atol=2.e-6 if kind in ('frame','F','edge') else 2.e-7,rtol=0.)
    results['passed']=True;save()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--system',choices=('chain','lih'),default='chain')
    parser.add_argument('--localization',choices=('BOYS','PIPEK_MEZEY'),default='BOYS')
    parser.add_argument('--step',type=float,default=.008)
    parser.add_argument('--output',default='screened_triples_results.json')
    args=parser.parse_args();psi4.set_output_file('screened_triples.out',False);run(args)
