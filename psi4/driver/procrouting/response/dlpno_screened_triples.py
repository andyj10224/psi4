"""Screened fixed-branch TNO and ordinary-triples molecular mixed inputs."""
from functools import lru_cache
from collections import OrderedDict
from dataclasses import fields
import numpy as np

from .dlpno_pno_curvature import _moving_pao_parent
from .dlpno_triples_curvature import MixedBlock, mixed_product as C, mixed_tno_frame, triples_moment_blocks
from .dlpno_triples_response import TriplesMixedRow


class TriplesFirstResponseCache:
    """Reference-owned, byte-bounded first directions and one native snapshot."""
    def __init__(self,wfn,max_bytes=None):
        from psi4 import core
        self.wfn=wfn
        self.max_bytes=max(0,int(core.get_memory()//8 if max_bytes is None else max_bytes))
        self.entries=OrderedDict();self.bytes=0;self.builds=0;self.hits=0
        self.native=None

    def snapshot(self):
        from psi4 import core
        if self.native is None:
            self.native=(core.dlpno_triples_response_state(self.wfn),core.dlpno_triples_basis_state(self.wfn))
        return self.native

    @staticmethod
    def freeze(value):
        if isinstance(value,np.ndarray):
            if not np.isfinite(value).all():raise ValueError('Nonfinite cached triples response.')
            result=value.copy();result.setflags(write=False);return result
        if isinstance(value,dict):return {k:TriplesFirstResponseCache.freeze(v) for k,v in value.items()}
        if isinstance(value,(tuple,list)):
            result=[TriplesFirstResponseCache.freeze(v) for v in value]
            return type(value)(*result) if hasattr(value,'_fields') else tuple(result)
        return value

    @staticmethod
    def storage(value):
        if isinstance(value,np.ndarray):return value.nbytes
        if isinstance(value,dict):return sum(TriplesFirstResponseCache.storage(v) for v in value.values())
        if isinstance(value,(tuple,list)):return sum(TriplesFirstResponseCache.storage(v) for v in value)
        return 0

    def get(self,provider,component,coordinate):
        if provider.fit.wfn is not self.wfn:
            raise ValueError('Triples first-response cache belongs to a different wavefunction.')
        if not 0<=coordinate<3*self.wfn.molecule().natom():
            raise ValueError('Invalid triples first-response coordinate.')
        chart=provider.first_chart(component)
        if coordinate in self.entries:
            record=self.entries[coordinate]
            if record['chart'].keys()!=chart.keys():raise ValueError('Cached triples first-response blocks changed.')
            for key,value in chart.items():
                old=record['chart'][key]
                if value.shape!=old.shape or not np.isfinite(value).all() or not np.allclose(value,old,atol=2.e-9,rtol=0.):
                    raise ValueError(f'Cached triples first-response chart changed: coordinate={coordinate}, block={key!r}.')
            direction=record['direction']
            provider.check_first_frames(direction,component)
            self.entries.move_to_end(coordinate);self.hits+=1
            return direction
        direction=provider._first_direction(component);self.builds+=1
        # Count before copying; oversized directions remain ephemeral.
        values={f.name:getattr(direction,f.name) for f in fields(direction) if f.name!='wavefunction'}
        cost=self.storage(values)+self.storage(chart)
        if cost>self.max_bytes:return direction
        while self.entries and self.bytes+cost>self.max_bytes:
            _,old=self.entries.popitem(last=False);self.bytes-=old['bytes']
        frozen=type(direction)(wavefunction=direction.wavefunction,**self.freeze(values))
        self.entries[coordinate]=dict(direction=frozen,chart=self.freeze(chart),bytes=cost)
        self.bytes+=cost
        return frozen


class ScreenedTriplesInputs:
    """Deliver TNOs, W/V/Fvv, occupied Fock and edge-overlap mixed blocks.

    Consume the SAME affine CCSD path and screened fitting provider as the
    residual. Mixed CCSD amplitudes remain eliminated by the existing adjoint.
    Native triplet PAO lists and full parent pivots are retained explicitly,
    including for full-rank TNO selections. No geometry samples enter here.
    """
    def __init__(self,wfn,fitting,first_cache=None):
        from psi4 import core
        if core.get_option('DLPNO','F_CUT_T') != 0.:
            raise ValueError('Screened mixed triples currently requires F_CUT_T=0.')
        self.fit=fitting
        self.first_cache=first_cache
        for name in ('parent','frame','projected','blocks','edge','occupied'):
            setattr(self,name,lru_cache(None)(getattr(self,name)))
        if first_cache is None:
            self.rows=core.dlpno_triples_response_state(wfn)
            self.basis=core.dlpno_triples_basis_state(wfn)
        else:
            if first_cache.wfn is not wfn:raise ValueError('Triples snapshot belongs to a different wavefunction.')
            self.rows,self.basis=first_cache.snapshot()
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
        return self._projected_in_frame(r,self.frame(r))

    def _projected_in_frame(self,r,frame):
        domain,aux=tuple(self.rows[r][0]),tuple(self.rows[r][1])
        ids=tuple(self.basis[r]['PAOS'].np[:,0].astype(int))
        X=self.fit.coefficients(ids,self.basis[r]['PARENT'].np,frame)
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

    def first_chart(self,component):
        keys=['lmo','fock',*[('pno',p) for p in self.fit.pair_paos],
              *[('t2',p) for p in self.fit.pair_paos],*[('t1',i) for i in range(self.L.value.shape[1])]]
        return {'overlap':self.S[component],**{key:self.electronic(key)[component] for key in keys}}

    def check_first_frames(self,direction,component):
        for r,dy in direction.frame_tangents.items():
            np.testing.assert_allclose(dy,self.frame(r)[component],atol=2.e-8,rtol=0.,
                                       err_msg='First/mixed TNO chart consistency')

    def first_direction(self,component,coordinate=None):
        if component not in (1,2):raise ValueError('Expected a first-derivative component, 1 or 2.')
        if self.first_cache is not None and coordinate is not None:
            return self.first_cache.get(self,component,coordinate)
        return self._first_direction(component)

    def _first_direction(self,component):
        """Prepare native triples right/left sources using the older first kernel.

        The independent first TNO selection must agree with this provider's
        mixed-frame chart before its right/left sources enter the Hessian.
        """
        from .dlpno_orbital import CCSDPairFrameResponse, PairDFFactors
        from .dlpno_triples_response import triples_gradient_direction, triples_moment_integrals
        if component not in (1,2):
            raise ValueError('Expected a first-derivative component, 1 or 2.')
        pairs=tuple(self.fit.pair_paos)
        frames={p:self.electronic(('pno',p)) for p in pairs}
        response=CCSDPairFrameResponse({p:x.value for p,x in frames.items()},
            {p:x[component] for p,x in frames.items()},self.S.value,self.S[component])
        t1=[self.electronic(('t1',i))[component] for i in range(self.L.value.shape[1])]
        t2={p:self.electronic(('t2',p))[component] for p in pairs}
        def parent(r):
            x=self.parent(r)
            return x.value,x[component]
        def integrals(r,domain,aux,y,dy):
            parts=[y,*[np.zeros_like(y) for _ in range(3)]]
            parts[component]=dy
            metric,*panels=self._projected_in_frame(r,MixedBlock(*parts))
            return triples_moment_integrals(domain,metric.value,metric[component],
                PairDFFactors(*(x.value for x in panels)),PairDFFactors(*(x[component] for x in panels)))
        direction=triples_gradient_direction(self.fit.wfn,response,t1,t2,self.L[component],
                                             self.F[component],parent,integrals,reference_rows=self.rows)
        self.check_first_frames(direction,component)
        return direction
