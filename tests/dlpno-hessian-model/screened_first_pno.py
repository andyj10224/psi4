"""Validate screened first-PNO molecular response on the native fixed branch.

Analytic inputs are evaluated at one geometry. Independent native projector
values and the older first-order response supply the numerical references.
This test does not assemble a final-PNO, CCSD or total molecular Hessian.
"""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import numpy as np
import psi4
from scipy.optimize import linear_sum_assignment

from localization_integration import configure
from screened_corrections import assert_branch, mixed_inputs, normalized_paos
from psi4.driver.procrouting.response.dlpno_frame_curvature import mixed_paos
from psi4.driver.procrouting.response.dlpno_orbital import (
    first_pno_tangent, pair_three_center_tangent, pair_exchange_tangent)
from psi4.driver.procrouting.response.dlpno_screened_pno import screened_first_pno_response
from psi4.driver.procrouting.response.dlpno_triples_curvature import mixed_product


def fixture(system,localization,full_rank):
    configure()
    frozen = system == 'lih'
    psi4.set_options(dict(freeze_core=frozen,save_jk=True,t_cut_pairs_mp2=1.e-6,t_cut_pairs=1.e-6,
        t_cut_pre=1.e-6,t_cut_do_ij=1.e-5,t_cut_do=.01,t_cut_do_pre=.03,t_cut_mkn=.001,
        t_cut_clmo=.01,t_cut_cpao=.01,t_cut_do_uv=1.e-5,dlpno__debug=2,dlpno_local_orbitals=localization,
        t_cut_pno_mp2=0. if full_rank else 1.e-5,t_cut_pno=0. if full_rank else 1.e-4,t_cut_tno=1.e-5))
    if frozen:
        psi4.set_options(dict(basis='def2-svp',df_basis_scf='def2-universal-jkfit',df_basis_cc='def2-svp-ri',
            t_cut_pairs_mp2=1.e-4,t_cut_pairs=1.e-4,t_cut_pre=1.e-5,t_cut_do_ij=1.e-3))
    atoms = ([('H',x,0,z) for x in (0,3.5,7,10.5) for z in (-.37,.37)] if not frozen else
             [(a,x,0,z) for x in (0,6,12) for a,z in (('Li',0),('H',1.6))])
    mol=psi4.geometry('0 1\n'+'\n'.join(f'{a} {x} {y} {z}' for a,x,y,z in atoms)+
                     '\nunits angstrom\nsymmetry c1\nno_com\nno_reorient')
    _,w=psi4.energy('dlpno-ccsd(t)',molecule=mol,return_wfn=True)
    return mol,w


def response(w,u,v):
    S,F,L,occupied,_,df=mixed_inputs(w,u,v)
    return screened_first_pno_response(w,S,F,L,occupied,df),S


def independent_first(w,u):
    """Older first-order refit, metric solve and spectral projector response."""
    S,F,L,occupied,_,df=mixed_inputs(w,u,u)
    d,b=psi4.core.dlpno_first_pno_state(w)
    basis=w.reference_wavefunction().get_basisset('DF_BASIS_CC')
    foo=L.value.T@F.value@L.value
    dfoo=L.u.T@F.value@L.value+L.value.T@F.u@L.value+L.value.T@F.value@L.u
    result={}
    for p,ids,aux in zip(d['PAIRS'],d['PAIR_PAO'],d['PAIR_AUX']):
        p=tuple(p); i,j=p
        N=mixed_paos(occupied,S,np.array(ids,dtype=int))[1]
        J,Q=df(tuple(aux))
        centers=np.array([basis.function_to_center(q) for q in aux])
        panels=[]
        for label in p:
            q,dq=np.zeros((len(aux),len(ids))),np.zeros((len(aux),len(ids)))
            for atom in sorted(set(centers)):
                if label not in d['OCC'][atom]: continue
                rows=np.flatnonzero(centers==atom)
                left,right=(np.array(d[key][atom],dtype=int) for key in ('AO_OCC','AO_VIR'))
                local=np.ix_(rows,left,right)
                a,da=pair_three_center_tangent(S.value,S.u,L.value[:,label],L.u[:,label],N.value,N.u,
                                               left,right,Q.value[local],Q.u[local])
                present=np.isin(ids,d['PAO'][atom])
                q[rows],dq[rows]=a*present,da*present
            panels.append((q,dq))
        K,dK=pair_exchange_tangent(J.value,J.u,*panels[0],*panels[1])
        # Honor the older first-order routine's stricter semicanonical input
        # contract, correcting only roundoff within the native pivot span.
        X=b[p]['PARENT'].np
        Y=N.value@X
        eig,U=np.linalg.eigh(Y.T@S.value@Y)
        X=X@((U*eig**(-.5))@U.T)
        _,U=np.linalg.eigh(X.T@N.value.T@F.value@N.value@X)
        X=X@U
        out=first_pno_tangent(N.value,N.u,S.value,S.u,F.value,F.u,K,dK,foo[i,i]+foo[j,j],
                              dfoo[i,i]+dfoo[j,j],X,b[p]['FRAME'].np.shape[1],i==j)
        result[p]=out.retained_density_tangent
    return result


def full_rank_first(w,u,reference):
    """Independent derivative of B(B^T S B)^-1 B^T in a fixed pivot chart.

    At full rank there is no retained/discarded PNO boundary. The old spectral
    oracle unnecessarily rediagonalizes a complete space at every geometry;
    this defining span formula avoids those internal rotations altogether.
    """
    S,_,_,occupied,_,_=mixed_inputs(w,u,u)
    d0,b0=psi4.core.dlpno_first_pno_state(reference)
    result={}
    for p,ids in zip(d0['PAIRS'],d0['PAIR_PAO']):
        p=tuple(p)
        N=mixed_paos(occupied,S,np.array(ids,dtype=int))[1]
        X=b0[p]['PARENT'].np
        B,Bu=N.value@X,N.u@X
        G=B.T@S.value@B
        Gu=Bu.T@S.value@B+B.T@S.u@B+B.T@S.value@Bu
        Z=np.linalg.solve(G,B.T)
        result[p]=Bu@Z+B@np.linalg.solve(G,Bu.T-Gu@Z)
    return result


def align_snapshot(w,reference):
    """Native values aligned only for occupied labeling and internal PNO gauge."""
    assert_branch(w,reference)
    r0,r=reference.reference_wavefunction(),w.reference_wavefunction()
    L0,L=(x.variable('DLPNO RESPONSE ACTIVE LMOS').np for x in (reference,w))
    overlap=psi4.core.MintsHelper(r0).ao_overlap(r0.basisset(),r.basisset()).np
    match=L0.T@overlap@L
    _,order=linear_sum_assignment(-abs(match)); inverse=np.argsort(order)
    signs=np.sign(match[np.arange(len(order)),order])
    d0,b0=psi4.core.dlpno_first_pno_state(reference)
    d,b=psi4.core.dlpno_first_pno_state(w)
    for key in ('AO_OCC','AO_VIR','PAO'): assert d0[key]==d[key]
    assert [sorted(inverse[x]) for x in d['OCC']]==d0['OCC']
    meta={tuple(p):(a,q) for p,a,q in zip(d['PAIRS'],d['PAIR_PAO'],d['PAIR_AUX'])}
    assert {tuple(sorted(inverse[p])) for p in d['PAIRS']}==set(map(tuple,d0['PAIRS']))
    N,S=normalized_paos(r),r.S().np
    result={}
    for p,ids,aux in zip(d0['PAIRS'],d0['PAIR_PAO'],d0['PAIR_AUX']):
        p=tuple(p); mapped=order[list(p)]; q=tuple(sorted(mapped))
        assert meta[q]==(ids,aux)
        old,new=b0[p],b[q]
        assert old['PARENT'].np.shape==new['PARENT'].np.shape
        seed=N[:,ids]@old['PARENT'].np
        actual=N[:,ids]@new['PARENT'].np
        np.testing.assert_allclose(actual@actual.T,seed@np.linalg.solve(seed.T@S@seed,seed.T),atol=2.e-8,rtol=0.)
        Y,Y0=new['FRAME'].np,old['FRAME'].np
        assert Y.shape==Y0.shape
        projector=Y@Y.T
        if Y.shape[1] == seed.shape[1]:
            # At full rank the selected projector IS the parent-span projector.
            # Use its defining metric solve in the fixed reference pivot chart;
            # re-diagonalizing all PNOs introduces needless, noisy rotations.
            # The native transported-span equality was independently checked above.
            projector=seed@np.linalg.solve(seed.T@S@seed,seed.T)
        projected=projector@r0.S().np@Y0
        eig,U=np.linalg.eigh(projected.T@S@projected)
        assert min(eig)>.9
        frame=projected@(U*eig**(-.5))@U.T
        qi,qj=(new[k].np for k in (('QI','QJ') if mapped[0]<=mapped[1] else ('QJ','QI')))
        result[p]=dict(frame=frame,projector=projector,qi=signs[p[0]]*qi,qj=signs[p[1]]*qj,pao=N[:,ids])
    return result,order


def run(args):
    mol,w=fixture(args.system,args.localization,args.full_rank)
    xyz=mol.geometry().np.copy()
    d,b=psi4.core.dlpno_first_pno_state(w)
    nbf=w.basisset().nbf(); naux=w.get_basisset('DF_BASIS_CC').nbf()
    assert min(map(len,d['AO_OCC']))<nbf and min(map(len,d['AO_VIR']))<nbf
    assert max(map(len,d['PAIR_AUX']))<naux and max(map(len,d['PAIR_PAO']))<nbf
    ranks=[dict(pair=list(p),parent=x['PARENT'].np.shape[1],retained=x['FRAME'].np.shape[1]) for p,x in b.items()]
    truncated=[x for x in ranks if x['retained']<x['parent']]
    assert bool(truncated)!=args.full_rank
    if args.system == 'chain' and not args.full_rank:
        assert any(x['pair'][0]!=x['pair'][1] for x in truncated)
    result=dict(system=args.system,localization=args.localization,full_rank=args.full_rank,
                frozen_orbitals=w.reference_wavefunction().nfrzc(),step_bohr=args.step,coordinates_bohr=xyz.tolist(),
                domains=d,ranks=ranks,checks=[],scope='Screened first-PNO molecular response only.',
                value_oracle=('Transported native parent-span metric solve at full rank.' if args.full_rank else
                              'Native selected PNO coefficients and projector.'),
                mixed_oracle=('Differences of independent full-rank parent-span metric derivative.' if args.full_rank else
                              'Differences of older independent first-order spectral projector response.'),
                raw_second_value_status='Recorded separately, may be unresolved because of h^-2 cancellation.')
    result['operator_error']=w.variable('DLPNO PNO PROJECTOR OPERATOR DIRECTION ERROR')
    result['operator_central_error']=w.variable('DLPNO PNO PROJECTOR OPERATOR CENTRAL ERROR')
    np.testing.assert_allclose(result['operator_error'],0.,atol=1.e-8,rtol=0.)
    if w.has_variable('DLPNO PNO PROJECTOR OPERATOR STEP STUDY'):
        study=w.variable('DLPNO PNO PROJECTOR OPERATOR STEP STUDY').np
        result['operator_step_study']=study.tolist()
        print('operator study',{float(h):float(max(abs(row[3]-row[4]) for row in study if row[2]==h))
                                 for h in sorted(set(study[:,2]))},flush=True)
    # Snapshot access is read-only and returned matrices must be independent.
    p=next(iter(b)); saved=b[p]['PARENT'].np.copy(); b[p]['PARENT'].np[:]=0.
    np.testing.assert_array_equal(psi4.core.dlpno_first_pno_state(w)[1][p]['PARENT'].np,saved)
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
    @lru_cache(None)
    def snapshot(delta):
        return align_snapshot(model(delta),w)[0]
    def delta(a,h,b=None,k=0.):
        x=np.zeros(xyz.size);x[a]+=h
        if b is not None: x[b]+=k
        return tuple(x)
    for a,c in coordinates:
        out,S=response(w,a,c)
        print('replay',a,c,out.replay_errors,'ranks',ranks,flush=True)
        metrics=dict(coordinates=[a,c],replay=out.replay_errors)
        projections={p:mixed_product('ma,na->mn',x.selection.frame,x.selection.frame) for p,x in out.pairs.items()}
        gram_error=max(np.max(abs(part)) for x in out.pairs.values()
                       for part in mixed_product('ma,mn,nb->ab',x.selection.frame,S,x.selection.frame)[1:])
        np.testing.assert_allclose(gram_error,0.,atol=2.e-8,rtol=0.)
        metrics['metric_orthonormality_error']=float(gram_error)
        if args.stage=='replay':
            result['checks'].append(metrics);continue
        for index,k in enumerate((a,c),1):
            independent=independent_first(w,k)
            error=max(np.max(abs(projections[p][index]-independent[p])) for p in projections)
            np.testing.assert_allclose(error,0.,atol=2.e-9,rtol=0.)
            metrics['independent_first_'+str(index)]=float(error)
            for key in ('frame','projector'):
                estimates=[]
                for h in (args.step,args.step/2):
                    plus,minus=(snapshot(delta(k,sign*h)) for sign in (1,-1))
                    estimates.append({p:(plus[p][key]-minus[p][key])/(2*h) for p in projections})
                expected={p:x.selection.frame[index] if key=='frame' else projections[p][index]
                          for p,x in out.pairs.items()}
                error=max(np.max(abs(expected[p]-(4*estimates[1][p]-estimates[0][p])/3)) for p in expected)
                metrics['native_first_'+key+'_'+str(index)]=float(error)
                print('native first',k,key,error,flush=True)
                np.testing.assert_allclose(error,0.,atol=5.e-7,rtol=0.)
        def stencil(h,key):
            nodes=[(1,0,1),(-1,0,1),(0,0,-2)] if a==c else [(i,j,i*j/4) for i in (-1,1) for j in (-1,1)]
            total={p:np.zeros_like(x[key]) for p,x in snapshot(delta(a,0.)).items()}
            for i,j,weight in nodes:
                for p,x in snapshot(delta(a,i*h,c,j*h)).items(): total[p]+=weight*x[key]/h**2
            return total
        actual=dict(frame={p:x.selection.frame.uv for p,x in out.pairs.items()},
                    projector={p:x.uv for p,x in projections.items()},
                    pao={p:x.uv for p,x in out.paos.items()},
                    qi={p:x[0].uv for p,x in out.panels.items()},qj={p:x[1].uv for p,x in out.panels.items()})
        for key,values in actual.items():
            coarse,fine=(stencil(h,key) for h in (args.step,args.step/2))
            error=max(np.max(abs(values[p]-(4*fine[p]-coarse[p])/3)) for p in values)
            metrics['native_mixed_'+key]=float(error)
            print('native mixed',a,c,key,error,flush=True)
            if key in ('frame','projector'):
                # Do not turn a cancellation-limited value oracle into a pass.
                # The stricter independent first-response difference below is
                # the mixed-projector acceptance test, as in first_pno_mixed.py.
                metrics['native_mixed_'+key+'_resolved']=bool(error<=3.e-6)
            else:
                np.testing.assert_allclose(error,0.,atol=5.e-7,rtol=0.)
        estimates=[]
        for h in (args.step,args.step/2):
            rows=[]
            for sign in (-1,1):
                other=model(delta(c,sign*h)); _,order=align_snapshot(other,w)
                if args.full_rank:
                    rows.append(full_rank_first(other,a,w))
                else:
                    first=independent_first(other,a)
                    rows.append({p:first[tuple(sorted(order[list(p)]))] for p in projections})
            estimates.append({p:(rows[1][p]-rows[0][p])/(2*h) for p in projections})
        error=max(np.max(abs(x.uv-(4*estimates[1][p]-estimates[0][p])/3)) for p,x in projections.items())
        metrics['mixed_from_independent_first']=float(error)
        print('mixed from independent first',a,c,error,flush=True)
        np.testing.assert_allclose(error,0.,atol=5.e-7,rtol=0.)
        if a!=c:
            reverse,_=response(w,c,a)
            error=max(np.max(abs(x.selection.frame.uv-reverse.pairs[p].selection.frame.uv)) for p,x in out.pairs.items())
            metrics['ordered_uv_vu_error']=float(error)
            np.testing.assert_allclose(error,0.,atol=2.e-8,rtol=0.)
        result['checks'].append(metrics)
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--system',choices=('chain','lih'),default='chain')
    p.add_argument('--localization',choices=('BOYS','PIPEK_MEZEY'),default='BOYS')
    p.add_argument('--full-rank',action='store_true')
    p.add_argument('--stage',choices=('all','replay'),default='all')
    p.add_argument('--step',type=float,default=.004)
    p.add_argument('--output',default='screened_first_pno_results.json')
    args=p.parse_args()
    if args.step<=0.: p.error('Expected positive displacement spacing.')
    psi4.set_output_file('screened_first_pno.out',False)
    run(args)
