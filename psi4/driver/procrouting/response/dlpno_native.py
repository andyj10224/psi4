"""Reference-owned C++ response stages; Python supplies molecular inputs/solvers.

The independent Python equations remain numerical oracles. Production pair and
triplet traversal, fitting, packing, subspace response and contractions execute
in dlpno/native_*.cc. This adapter only snapshots metadata and marshals stage I/O.
"""
from collections import OrderedDict
from types import SimpleNamespace
import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres
from .dlpno_triples_curvature import MixedBlock
from .dlpno_pno_curvature import PNOSubspaceCurvature, FirstPNOCurvature
from .dlpno_screened_pno import ScreenedFirstPNOResponse
from .dlpno_screened_ccsd import CCSDFirstResponseCache


def _selection(x):
    return PNOSubspaceCurvature(MixedBlock(*x['frame']),MixedBlock(*x['projector']),
                               x['occupations'],x['boundary_gap'])


def _energies(stage):
    return {p:{k:(x if k=='rank' else MixedBlock(*x)) for k,x in one.items()} for p,one in stage.items()}


class NativeHessianWorkspace:
    def __init__(self,wfn):
        from psi4 import core
        if any(core.get_option('DLPNO',name)!=0. for name in ('F_CUT','F_CUT_T')):
            raise ValueError('Native screened Hessian requires F_CUT=F_CUT_T=0.')
        if wfn.variable('DLPNO FIRST PNO AO INTS TOL')!=0.:
            raise ValueError('Native screened Hessian requires the forward AO Schwarz screen off.')
        self.wfn=wfn
        self.panel_counts=np.zeros(6,dtype=np.int64);self.panel_peak=0
        domains=core.dlpno_response_domain_state(wfn)
        first,native=core.dlpno_first_pno_state(wfn)
        arrays=lambda name:{p:x.np for p,x in core.dlpno_lmp2_blocks(wfn,name).items()}
        frames,final,amp,edges=(arrays(k) for k in ('FRAME','FINAL_FRAME','AMPLITUDE','RESPONSE_EDGES'))
        self.pair_occ={tuple(p):tuple(x) for p,x in zip(domains['PAIRS'],domains['PAIR_OCC'])}
        self.pair_aux={tuple(p):tuple(x) for p,x in zip(domains['PAIRS'],domains['PAIR_AUX'])}
        self.pair_paos={tuple(p):tuple(x) for p,x in zip(first['PAIRS'],first['PAIR_PAO'])}
        if not (native.keys()==frames.keys()==final.keys()==amp.keys()==self.pair_occ.keys()):
            raise ValueError('Native Hessian pair snapshots disagree.')
        for p,ids in zip(domains['PAIRS'],domains['PAIR_PAO']):
            if self.pair_paos[tuple(p)]!=tuple(ids):raise ValueError('First/final PAO domains disagree.')
        first_aux={tuple(p):tuple(a) for p,a in zip(first['PAIRS'],first['PAIR_AUX'])}
        pairs={p:dict(first_auxiliary=first_aux[p],occupied=self.pair_occ[p],paos=self.pair_paos[p],auxiliary=self.pair_aux[p],
                     parent=x['PARENT'].np,first_frame=frames[p],final_frame=final[p],
                     first_amplitude=x['AMPLITUDE'].np,lmp2_amplitude=amp[p],qi=x['QI'].np,qj=x['QJ'].np,
                     occupations=x['OCCUPATIONS'].np[:,0],edges=edges[p].astype(int).tolist()) for p,x in native.items()}
        rows=core.dlpno_triples_response_state(wfn);basis=core.dlpno_triples_basis_state(wfn)
        if rows.keys()!=basis.keys():raise ValueError('Native triples parent snapshots disagree.')
        triples={}
        for r,(occ,aux,y,t,w,es,z,v) in rows.items():
            n=y.np.shape[1]
            triples[r]=dict(occupied=occ,auxiliary=aux,paos=basis[r]['PAOS'].np[:,0].astype(int).tolist(),
                           parent=basis[r]['PARENT'].np,frame=y.np,t=t.np.reshape((n,)*3),
                           z=z.np.reshape((n,)*3),w=w.np.reshape((n,)*3),v=v.np.reshape((n,)*3),edges=es)
        aux=wfn.get_basisset('DF_BASIS_CC');self.naux=aux.nbf()
        self.native=core.DLPNOResponseWorkspace(domains,first,pairs,triples,
            [aux.function_to_center(q) for q in range(self.naux)],
            {p:x.np for p,x in core.dlpno_pno_truncation_lmp2_response(wfn,True).items()},
            wfn.variable('DLPNO PNO TRUNCATION ERROR'),min(core.get_memory()//16,512*1024**2))
        screened_domains,screened_state=core.dlpno_screened_pair_state(wfn)
        self.native.screened_snapshot(screened_domains,{p:{k:x.np for k,x in v.items()} for p,v in screened_state.items()})
        self.first=NativeTriplesCache(wfn)

    def entry(self,u,v,S,F,L,D,df):
        np.testing.assert_allclose(S.value,self.wfn.reference_wavefunction().S().np,atol=2.e-10,rtol=0.)
        np.testing.assert_allclose(F.value,self.wfn.reference_wavefunction().Fa().np,atol=2.e-10,rtol=0.)
        np.testing.assert_allclose(L.value,self.wfn.variable('DLPNO RESPONSE ACTIVE LMOS').np,atol=2.e-10,rtol=0.)
        if hasattr(D,'density'):density=D.density
        else:
            eye=np.eye(len(S.value));zero=np.zeros_like(eye)
            density=D(MixedBlock(eye,zero,zero,zero))
        J,Q=df.full if hasattr(df,'full') else df(tuple(range(self.naux)))
        return self.native.entry(u,v,[S,F,L,density,J,Q])

    def pno(self,entry):
        from psi4 import core
        matrix=core.Matrix.from_array
        matrices=lambda xs:{p:matrix(x) for p,x in xs.items()}
        # Whole native LMP2 solves retain the DLPNO option preparation in core.
        def first(dy,dk,s,f,foo):
            return {p:x.np for p,x in core.dlpno_lmp2_tangent(self.wfn,matrices(dy),matrices(dk),
                                                           matrix(s),matrix(f),matrix(foo)).items()}
        def mixed(source):
            return {p:x.np for p,x in core.dlpno_lmp2_mixed_tangent(self.wfn,matrices(source)).items()}
        raw=entry.pno(first,mixed)
        pairs={p:FirstPNOCurvature(*(MixedBlock(*x[k]) for k in ('parent','exchange','fock','amplitudes')),
                                  _selection(x['selection'])) for p,x in raw['first'].items()}
        initial,final=_energies(raw['initial']),_energies(raw['final'])
        return SimpleNamespace(first_response=ScreenedFirstPNOResponse(pairs,
            {p:MixedBlock(*x) for p,x in raw['normalized'].items()},{},{},None,{}),
            initial=initial,final=final,final_selection={p:_selection(x) for p,x in raw['selection'].items()},
            mixed_source=raw['source'],residual_errors=raw['residual_errors'],loss=MixedBlock(*raw['loss']),
            initial_loss=MixedBlock(*raw['initial_loss']),final_loss=MixedBlock(*raw['final_loss']),
            stationary_loss=raw['stationary_loss'])


class NativeFitting:
    def __init__(self,workspace,entry,electronic,S,df):
        self.wfn=workspace.wfn;self.workspace=workspace;self.native=entry
        self.electronic=electronic;self.S=S;self.df=df
        self.pair_occ=workspace.pair_occ;self.pair_aux=workspace.pair_aux;self.pair_paos=workspace.pair_paos
    def projected(self,p):
        raw=self.native.projected(p)
        return (*[MixedBlock(*x) for x in raw[:5]],None)


def ccsd_response(workspace,entry,pnos,S,F,L,D,df,*,first_cache,coordinates):
    from psi4 import core
    from .dlpno_ccsd_response import coupled_ccsd_tangent
    if first_cache.base is None:
        t2={p:x.np.copy() for p,x in core.dlpno_lmp2_blocks(workspace.wfn,'CCSD_DOUBLES').items()}
        t1=[]
        for i in range(L.value.shape[1]):
            domain,one,_=core.dlpno_ccsd_df_pair_state(workspace.wfn,i,i)
            t1.append(one[domain.index(i)].np.ravel().copy())
        first_cache.base=(t1,t2)
    t1,t2=first_cache.base
    constant=lambda x:MixedBlock(x,*[np.zeros_like(x) for _ in range(3)])
    one=[constant(x) for x in t1];two={p:constant(x) for p,x in t2.items()}
    def residual(path):
        a,b=entry.ccsd(one,two,path)
        return [MixedBlock(*x) for x in a],{p:MixedBlock(*x) for p,x in b.items()}
    fixed=None;tangent=[];pending={}
    for k,coordinate in enumerate(coordinates,1):
        chart=first_cache.chart(S,F,L,pnos,k)
        record=pending.get(coordinate,first_cache.entries.get(coordinate))
        if record is None:
            if fixed is None:fixed=residual('fixed');first_cache.source_replays+=1
            saved=coupled_ccsd_tangent(workspace.wfn,[x[k] for x in fixed[0]],{p:x[k] for p,x in fixed[1].items()})
            first_cache.solves+=1;pending[coordinate]=first_cache.record(chart,saved)
            saved=pending[coordinate]['tangent']
        else:saved=first_cache.check(record,chart,coordinate)
        tangent.append(saved)
    one=[MixedBlock(x,tangent[0][0][i],tangent[1][0][i],np.zeros_like(x)) for i,x in enumerate(t1)]
    two={p:MixedBlock(x,tangent[0][1][p],tangent[1][1][p],np.zeros_like(x)) for p,x in t2.items()}
    result=residual('affine')
    errors=[max(np.max(abs(x[k]),initial=0.) for x in [*result[0],*result[1].values()]) for k in (0,1,2)]
    if not np.isfinite(errors).all() or max(errors)>2.e-9:raise ValueError('Native CCSD value/first residual response failed.')
    first_cache.entries.update(pending)
    def electronic(key):
        if key=='lmo':return L
        if key=='fock':return F
        kind,index=key
        if kind=='pno':return pnos.final_selection[index].frame
        return one[index] if kind=='t1' else two[index]
    fit=NativeFitting(workspace,entry,electronic,S,df)
    panel=entry.panel_statistics
    workspace.panel_counts+=panel[:6];workspace.panel_peak=max(workspace.panel_peak,panel[6])
    return dict(electronic=electronic,fitting=fit,residual=result,fixed_residual=fixed,tangent_u=tangent[0],
                tangent_v=tangent[1],base=(t1,t2),residual_errors=errors,native_amplitudes=(one,two),native_profile=entry.ccsd_profile)


class NativeTriplesCache:
    def __init__(self,wfn):
        from psi4 import core
        self.wfn=wfn;self.max_bytes=core.get_memory()//8
        self.entries=OrderedDict();self.bytes=0;self.builds=0;self.hits=0
    def get(self,provider,component,coordinate):
        if provider.fit.wfn is not self.wfn:raise ValueError('Triples response cache belongs to another wavefunction.')
        if not 0<=coordinate<3*self.wfn.molecule().natom():raise ValueError('Invalid triples response coordinate.')
        chart=provider.chart(component)
        if coordinate in self.entries:
            old=self.entries[coordinate]
            if old['chart'].keys()!=chart.keys():raise ValueError('Native triples first chart keys changed.')
            for k,x in chart.items():np.testing.assert_allclose(x,old['chart'][k],atol=2.e-9,rtol=0.)
            self.entries.move_to_end(coordinate);self.hits+=1
            provider.native.check_frames(old['direction'].native,component)
            return old['direction']
        from psi4 import core
        native=provider.native.direction(component,max(2.e-10,1.01*core.get_option('DLPNO','R_CONVERGENCE')),
                                         self.wfn.variable('(T) CORRECTION ENERGY'))
        direction=SimpleNamespace(native=native,wavefunction=self.wfn)
        self.builds+=1
        size=native.bytes+sum(x.nbytes for x in chart.values())
        if size<=self.max_bytes:
            while self.entries and self.bytes+size>self.max_bytes:
                _,old=self.entries.popitem(last=False);self.bytes-=old['bytes']
            saved={k:np.array(x,copy=True) for k,x in chart.items()}
            self.entries[coordinate]=dict(direction=direction,chart=saved,bytes=size);self.bytes+=size
        return direction


class NativeTriplesInputs:
    def __init__(self,workspace,response):
        self.workspace=workspace;self.fit=response['fitting'];self.electronic=response['electronic']
        if self.fit.workspace is not workspace:raise ValueError('Triples fitting belongs to another Hessian reference.')
        self.native=self.fit.native.triples(*response['native_amplitudes'])
    def chart(self,k):
        keys=['lmo','fock',*[('pno',p) for p in self.fit.pair_paos],*[('t2',p) for p in self.fit.pair_paos],
              *[('t1',i) for i in range(self.electronic('lmo').value.shape[1])]]
        return {'overlap':self.fit.S[k],**{key:self.electronic(key)[k] for key in keys}}
    def first_direction(self,component,coordinate):return self.workspace.first.get(self,component,coordinate)
    def occupied(self,i,j):return MixedBlock(*self.native.occupied(i,j))
    def mixed_row(self,r):return tuple(x[3] for x in self.native.blocks(r))
    def edge(self,r,q):return MixedBlock(*self.native.edge(r,q))


class NativeTriplesColumn:
    def __init__(self,direction,*,tolerance=1.e-10):
        self.direction=direction
        native=direction.native;diag=np.asarray(native.diagonal)
        if not np.isfinite(diag).all() or np.any(abs(diag)<1.e-12):raise ValueError('Singular triples tangent denominator.')
        def solve(source,left):
            rhs=-np.asarray(source)
            if not np.isfinite(rhs).all():raise ValueError('Invalid triples tangent source.')
            if not rhs.size:return rhs,0.
            op=LinearOperator((rhs.size,rhs.size),matvec=lambda x:native.apply(x,left),dtype=float)
            pre=LinearOperator(op.shape,matvec=lambda x:x/diag,dtype=float)
            x,info=gmres(op,rhs,M=pre,rtol=tolerance,atol=0.)
            error=np.linalg.norm(op@x-rhs)
            if info or not np.isfinite(error) or error>tolerance*max(np.linalg.norm(rhs),1.e-14):
                raise RuntimeError(f'Native triples tangent residual {error:.3e} (info={info}).')
            return x,error
        dt,right=solve(native.right,False);dz,left=solve(native.left,True)
        if not np.isclose(native.relaxed_energy(dt),native.energy_tangent,atol=max(1.e-10,tolerance),rtol=1.e-8):
            raise RuntimeError('Native triples energy tangent disagrees with adjoint contraction.')
        self.native=native.column(dt,dz)
        self.tangent=SimpleNamespace(residual=right,adjoint_residual=left)
    def ccsd_stationary_element(self,row,occupied,mixed_row,mixed_overlap,adjoint,residual):
        if row.wavefunction is not self.direction.wavefunction or adjoint.wavefunction is not row.wavefunction:
            raise ValueError('Native triples curvature references disagree.')
        provider=mixed_row.__self__
        if provider.fit.wfn is not row.wavefunction:raise ValueError('Mixed triples inputs belong to another wavefunction.')
        return self.native.element(row.native,provider.native)+adjoint.correction(*residual)
