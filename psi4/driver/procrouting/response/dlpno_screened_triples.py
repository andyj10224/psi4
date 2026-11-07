"""Screened fixed-branch TNO and ordinary-triples molecular mixed inputs."""
from functools import lru_cache
import numpy as np

from .dlpno_pno_curvature import _moving_pao_parent
from .dlpno_triples_curvature import MixedBlock, mixed_product as C, mixed_tno_frame, triples_moment_blocks
from .dlpno_triples_response import TriplesMixedRow


class ScreenedTriplesInputs:
    """Deliver TNOs, W/V/Fvv, occupied Fock and edge-overlap mixed blocks.

    Consume the SAME affine CCSD path and screened fitting provider as the
    residual. Mixed CCSD amplitudes remain eliminated by the existing adjoint.
    Native triplet PAO lists and full parent pivots are retained explicitly,
    including for full-rank TNO selections. No geometry samples enter here.
    """
    def __init__(self,wfn,fitting):
        from psi4 import core
        if core.get_option('DLPNO','F_CUT_T') != 0.:
            raise ValueError('Screened mixed triples currently requires F_CUT_T=0.')
        self.fit=fitting
        for name in ('parent','frame','projected','blocks','edge','occupied'):
            setattr(self,name,lru_cache(None)(getattr(self,name)))
        self.rows=core.dlpno_triples_response_state(wfn)
        self.basis=core.dlpno_triples_basis_state(wfn)
        self.electronic,self.S=fitting.electronic,fitting.S
        self.F,self.L=self.electronic('fock'),self.electronic('lmo')
        if self.rows.keys()!=self.basis.keys():
            raise ValueError('Missing native triples parent snapshot.')
        pairs=tuple(core.dlpno_lmp2_blocks(wfn,'CCSD_DOUBLES'))
        if any(np.any(self.electronic(('t2',p)).uv) for p in pairs) or any(
                np.any(self.electronic(('t1',i)).uv) for i in range(self.L.value.shape[1])):
            raise ValueError('Screened triples inputs require affine CCSD amplitudes.')

    def parent(self,r):
        ids=tuple(self.basis[r]['PAOS'].np[:,0].astype(int))
        X=self.basis[r]['PARENT'].np
        if np.count_nonzero(np.any(X!=0.,axis=1))!=X.shape[1]:
            raise ValueError('Expected a full-rank native triplet pivot span.')
        return _moving_pao_parent(self.fit.paos(ids),self.S,X)[0]

    def frame(self,r):
        pairs=((r[0],r[1]),(r[1],r[2]),(r[0],r[2]))
        return mixed_tno_frame(r,self.parent(r),self.S,
            {p:self.electronic(('pno',p)) for p in pairs},
            {p:self.electronic(('t2',p)) for p in pairs},self.rows[r][2].np)

    def projected(self,r):
        domain,aux=tuple(self.rows[r][0]),tuple(self.rows[r][1])
        ids=tuple(self.basis[r]['PAOS'].np[:,0].astype(int))
        X=self.fit.coefficients(ids,self.basis[r]['PARENT'].np,self.frame(r))
        oo=self.fit.raw('oo',aux,domain,domain)
        ov=C('Qma,ab->Qmb',self.fit.raw('ov',aux,domain,ids),X)
        vv=C('ma,Qmn,nb->Qab',X,self.fit.raw('vv',aux,ids,ids),X)
        return self.fit.df(aux)[0],oo,ov,vv

    def blocks(self,r):
        frame=self.frame(r)
        def ordered(i,j):
            t=self.electronic(('t2',tuple(sorted((i,j)))))
            return t if i<=j else MixedBlock(*(x.T for x in t))
        def transfer(i,j):
            return C('ma,mn,nb->ab',self.electronic(('pno',tuple(sorted((i,j))))),self.S,frame)
        return triples_moment_blocks(r,self.rows[r][0],*self.projected(r),transfer,
            lambda i:self.electronic(('t1',i)),ordered,
            lambda i:C('m,mn,nb->b',self.L.slice((slice(None),i)),self.F,frame),
            C('ma,mn,nb->ab',frame,self.F,frame))

    def mixed_row(self,r):
        return TriplesMixedRow(*(x.uv for x in self.blocks(r)))

    def edge(self,r,q):
        return C('ma,mn,nb->ab',self.frame(r),self.S,self.frame(q))

    def occupied(self,i,j):
        return C('m,mn,n->',self.L.slice((slice(None),i)),self.F,self.L.slice((slice(None),j)))
