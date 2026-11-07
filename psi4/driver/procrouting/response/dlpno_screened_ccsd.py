"""Native-domain mixed fitting factors for RHF CCSD and ordinary triples.

All inputs are analytic blocks at one geometry. Discrete domains, PAO-pair
masks and parent pivots come from the forward model; AO Schwarz screening must
be off. This is molecular response delivery, not a public Hessian driver.
"""
from functools import lru_cache
import numpy as np
from .dlpno_frame_curvature import mixed_paos, mixed_metric_solve
from .dlpno_triples_curvature import MixedBlock, mixed_product as C, _symmetric_inverse as _inverse, _filtered_inverse


class ScreenedFitting:
    """Differentiate the actual raw OO/OV/VV fit and cross-pair J/K vertices."""
    def __init__(self,wfn,electronic,overlap,occupied_action,local_df):
        from psi4 import core
        if core.get_option('DLPNO','F_CUT') != 0.:
            raise ValueError('Screened mixed fitting currently requires F_CUT=0.')
        self.wfn,self.electronic,self.S = wfn,electronic,overlap
        self.occupied_action,self.df = occupied_action,lru_cache(None)(local_df)
        # Instance-owned caches release raw panels with this response direction.
        for name in ('paos','refit','pair_coefficients','raw','projected','mixed_integrals'):
            setattr(self,name,lru_cache(None)(getattr(self,name)))
        self.domains=core.dlpno_response_domain_state(wfn)
        first,self.parents=core.dlpno_first_pno_state(wfn)
        if wfn.variable('DLPNO FIRST PNO AO INTS TOL') != 0.:
            raise ValueError('Screened mixed fitting requires the forward AO Schwarz screen off.')
        self.pair_paos={tuple(p):tuple(ids) for p,ids in zip(first['PAIRS'],first['PAIR_PAO'])}
        self.pair_occ={tuple(p):tuple(ids) for p,ids in zip(self.domains['PAIRS'],self.domains['PAIR_OCC'])}
        self.pair_aux={tuple(p):tuple(ids) for p,ids in zip(self.domains['PAIRS'],self.domains['PAIR_AUX'])}
        for p,ids in zip(self.domains['PAIRS'],self.domains['PAIR_PAO']):
            if self.pair_paos[tuple(p)]!=tuple(ids):raise ValueError('First/final fitting PAO domains disagree.')
        self.L=electronic('lmo')
        basis=wfn.get_basisset('DF_BASIS_CC')
        self.centers=tuple(basis.function_to_center(q) for q in range(basis.nbf()))
        self.masks=[]
        for row in self.domains['PAO_PAIRS']:
            pairs=set(zip(row[::2],row[1::2]));self.masks.append(pairs|{(v,u) for u,v in pairs})
        np.testing.assert_allclose(overlap.value,wfn.reference_wavefunction().S().np,atol=2.e-10,rtol=0.)
        np.testing.assert_allclose(self.L.value,wfn.variable('DLPNO RESPONSE ACTIVE LMOS').np,atol=2.e-10,rtol=0.)

    def paos(self,ids):
        return mixed_paos(self.occupied_action,self.S,np.asarray(ids,dtype=int))[1]

    def refit(self,atom,occ):
        rows=np.asarray(self.domains['AO_OCC'][atom],dtype=int)
        rhs=C('mn,ni->mi',self.S.slice((rows,slice(None))),self.L.slice((slice(None),list(occ))))
        out=mixed_metric_solve(self.S.slice(np.ix_(rows,rows)),rhs)
        present=np.isin(occ,self.domains['OCC'][atom])
        return MixedBlock(*(x*present for x in out))

    def coefficients(self,ids,parent,frame):
        """PAO coefficients in the fixed native pivot span, including its motion."""
        N=self.paos(ids)
        B=MixedBlock(*(n@parent for n in N))
        Z=mixed_metric_solve(C('ma,mn,nb->ab',B,self.S,B),C('ma,mn,nb->ab',B,self.S,frame))
        return MixedBlock(*(parent@z for z in Z))

    def pair_coefficients(self,p):
        return self.coefficients(self.pair_paos[p],self.parents[p]['PARENT'].np,self.electronic(('pno',p)))

    def raw(self,kind,aux,left,right):
        """OO uses two refits; OV extended-label zeros; VV the PAO-pair mask."""
        _,Q=self.df(aux)
        if any(x.shape!=(len(aux),len(self.S.value),len(self.S.value)) for x in Q):
            raise ValueError('Expected raw AO three-center blocks in the requested auxiliary order.')
        parts=[np.zeros((len(aux),len(left),len(right))) for _ in range(4)]
        centers=np.asarray([self.centers[q] for q in aux])
        for atom in sorted(set(centers)):
            rows=np.flatnonzero(centers==atom)
            A=np.asarray(self.domains['AO_OCC'][atom],dtype=int)
            B=np.asarray(self.domains['AO_VIR'][atom],dtype=int)
            if kind=='oo':
                block=C('ma,Qmn,nb->Qab',self.refit(atom,left),Q.slice(np.ix_(rows,A,A)),self.refit(atom,right))
                mask=1.
            elif kind=='ov':
                block=C('ma,Qmn,nb->Qab',self.refit(atom,left),Q.slice(np.ix_(rows,A,B)),
                        self.paos(right).slice((B,slice(None))))
                mask=np.isin(right,self.domains['PAO'][atom])[None,:]
            elif kind=='vv':
                block=C('ma,Qmn,nb->Qab',self.paos(left).slice((B,slice(None))),
                        Q.slice(np.ix_(rows,B,B)),self.paos(right).slice((B,slice(None))))
                mask=np.array([[(a,b) in self.masks[atom] for b in right] for a in left])
            else:raise ValueError('Unknown screened fitting block.')
            for target,x in zip(parts,block):target[rows]=x*mask
        return MixedBlock(*parts)

    def projected(self,p):
        p=tuple(sorted(p));aux,occ,ids=self.pair_aux[p],self.pair_occ[p],self.pair_paos[p]
        X=self.pair_coefficients(p)
        oo=self.raw('oo',aux,occ,occ)
        ov=C('Qma,ab->Qmb',self.raw('ov',aux,occ,ids),X)
        vv=C('ma,Qmn,nb->Qab',X,self.raw('vv',aux,ids,ids),X)
        metric,_=self.df(aux);g=_inverse(metric)
        return oo,ov,vv,g,_filtered_inverse(metric,g),None

    def mixed_integrals(self,i,j,k):
        p=tuple(sorted((i,j)));q=tuple(sorted((k,j)))
        aux,ids,other=self.pair_aux[p],self.pair_paos[p],self.pair_paos[q]
        X,Y=self.pair_coefficients(p),self.pair_coefficients(q)
        a=self.raw('oo',aux,(i,),(k,)).slice((slice(None),0,0))
        b=C('ma,Qmn,nb->Qab',X,self.raw('vv',aux,ids,other),Y)
        c=C('Qa,ab->Qb',self.raw('ov',aux,(i,),ids).slice((slice(None),0,slice(None))),X)
        d=C('Qa,ab->Qb',self.raw('ov',aux,(k,),other).slice((slice(None),0,slice(None))),Y)
        g=self.projected(p)[3]
        return C('Q,QR,Rab->ab',a,g,b),C('Qa,QR,Rb->ab',c,g,d)


def screened_ccsd_residual(wfn,electronic,overlap,occupied_action,local_df):
    """Analytic affine-amplitude mixed CCSD residual using actual local fits."""
    from .dlpno_ccsd_mixed import mixed_ccsd_residual
    fit=ScreenedFitting(wfn,electronic,overlap,occupied_action,local_df)
    return mixed_ccsd_residual(wfn,electronic,overlap,local_df,fitted=fit)


def screened_ccsd_response(wfn,pnos,overlap,fock,lmo,occupied_action,local_df):
    """Solve first CCSD response and form its analytic affine mixed source."""
    from psi4 import core
    from .dlpno_ccsd_mixed import mixed_ccsd_residual
    from .dlpno_ccsd_response import coupled_ccsd_tangent
    t2={p:x.np.copy() for p,x in core.dlpno_lmp2_blocks(wfn,'CCSD_DOUBLES').items()}
    t1=[]
    for i in range(lmo.value.shape[1]):
        dom,one,_=core.dlpno_ccsd_df_pair_state(wfn,i,i)
        t1.append(one[dom.index(i)].np.ravel().copy())
    constant=lambda x:MixedBlock(x,*[np.zeros_like(x) for _ in range(3)])
    amplitudes={('t1',i):constant(x) for i,x in enumerate(t1)}
    amplitudes.update({('t2',p):constant(x) for p,x in t2.items()})
    def electronic(key):
        if key=='lmo':return lmo
        if key=='fock':return fock
        if isinstance(key,tuple) and key[0]=='pno':return pnos.final_selection[key[1]].frame
        return amplitudes[key]
    fit=ScreenedFitting(wfn,electronic,overlap,occupied_action,local_df)
    fixed=mixed_ccsd_residual(wfn,electronic,overlap,local_df,fitted=fit)
    tangent=[]
    for k in (1,2):
        tangent.append(coupled_ccsd_tangent(wfn,[x[k] for x in fixed[0]],{p:x[k] for p,x in fixed[1].items()}))
    for i,x in enumerate(t1):amplitudes['t1',i]=MixedBlock(x,tangent[0][0][i],tangent[1][0][i],np.zeros_like(x))
    for p,x in t2.items():amplitudes['t2',p]=MixedBlock(x,tangent[0][1][p],tangent[1][1][p],np.zeros_like(x))
    result=mixed_ccsd_residual(wfn,electronic,overlap,local_df,fitted=fit)
    errors=[max(np.max(abs(x[k]),initial=0.) for x in [*result[0],*result[1].values()]) for k in (0,1,2)]
    if not np.isfinite(errors).all() or max(errors)>2.e-9:
        raise ValueError('Screened CCSD value/first residual response failed.')
    return dict(electronic=electronic,fitting=fit,residual=result,fixed_residual=fixed,
                tangent_u=tangent[0],tangent_v=tangent[1],base=(t1,t2),residual_errors=errors)
