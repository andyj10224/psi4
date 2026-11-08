"""Native complete response stages against the independent Python equations.

These algebra tests require the compiled bindings, but no molecular SCF run.
The cc-pVDZ molecular tests remain the derivative/branch correctness gate.
"""
import ast
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from psi4 import core
from psi4.driver.procrouting.response import dlpno_triples_curvature as algebra
from psi4.driver.procrouting.response.dlpno_triples_curvature import MixedBlock as B
from psi4.driver.procrouting.response.dlpno_pno_curvature import _polar_normalization
from psi4.driver.procrouting.response.dlpno_screened_ccsd import ScreenedFitting
from psi4.driver.procrouting.response.dlpno_ccsd_mixed import mixed_ccsd_residual

pytestmark=pytest.mark.skipif(not hasattr(core,'DLPNOResponseWorkspace'),reason='Native response binding required')


def block(rng,value,scale=.01,symmetric=False):
    xs=[np.array(value),*[rng.normal(scale=scale,size=np.shape(value)) for _ in range(3)]]
    if symmetric:xs=[(x+np.swapaxes(x,-1,-2))*.5 for x in xs]
    return B(*xs)


def constant(x):
    return B(np.array(x),*[np.zeros_like(x) for _ in range(3)])


def first_pno_domains(domains):
    """Match the snapshot exported by ccsd.cc's first PNO construction.

    It contains the OV-fit maps, not the later VV PAO_PAIRS mask or PAIR_OCC.
    Supplying the entire final domain state here hides production key errors.
    """
    return {name:domains[name] for name in
            ('AO_OCC','AO_VIR','OCC','PAO','PAIRS','PAIR_PAO','PAIR_AUX')}


def assert_blocks(a,b,atol=3.e-11,rtol=1.e-10):
    for x,y in zip(a,b):np.testing.assert_allclose(x,y,atol=atol,rtol=rtol)


def equations():
    tree=ast.parse(Path(__file__).with_name('test_dlpno_mixed_products.py').read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='test_response_equations_against_independent_product_rule')
    return ast.literal_eval(fn.decorator_list[0].args[1])+['ab,ba->']


@pytest.mark.parametrize('equation',equations())
def test_native_product_rules(equation):
    rng=np.random.default_rng(812)
    dims={c:2+(i%3) for i,c in enumerate(sorted(set(equation.replace('->','').replace(',',''))))}
    shapes=[tuple(dims[c] for c in s) for s in equation.split('->')[0].split(',')]
    xs=[B(*(rng.normal(scale=.1,size=s) for _ in range(4))) for s in shapes]
    expected=algebra._expanded_mixed_product(equation,*xs)
    assert_blocks(core.dlpno_native_response_test(equation,xs),expected)


@pytest.mark.parametrize('op', ['inverse','filtered','projector','polar','solve'])
@pytest.mark.parametrize('rank',[2,4])
def test_native_metric_and_subspace_response(op,rank):
    rng=np.random.default_rng(31);q=np.linalg.qr(rng.normal(size=(4,4)))[0]
    values=np.array([-.4,.2,.7,1.2]) if op=='filtered' else np.array([.15,.35,.8,1.3])
    if op=='polar':values=np.ones(4)+np.arange(4)*1.e-10
    m=block(rng,(q*values)@q.T,symmetric=True)
    xs=[m]
    if op=='solve':
        from psi4.driver.procrouting.response.dlpno_frame_curvature import mixed_metric_solve
        rhs=block(rng,rng.normal(size=(4,3)));xs.append(rhs);expected=mixed_metric_solve(m,rhs)
    else:expected={'inverse':algebra._symmetric_inverse,'filtered':lambda m:algebra._filtered_inverse(m,algebra._symmetric_inverse(m)),
                   'projector':lambda m:algebra._mixed_projector(m,rank),'polar':_polar_normalization}[op](m)
    assert_blocks(core.dlpno_native_response_test(op,xs,rank),expected)


def fitting_fixture(seed=16,cache=0):
    rng=np.random.default_rng(seed);n,no,nq=6,2,5;paos=(2,3,4,5)
    s=block(rng,np.eye(n),symmetric=True);f=block(rng,np.diag([-1.2,-.8,.2,.4,.6,.9]),symmetric=True)
    l=block(rng,np.eye(n)[:,:no]);d=block(rng,l.value@l.value.T,symmetric=True)
    a=rng.normal(size=(nq,nq));j=block(rng,a@a.T+np.eye(nq),symmetric=True)
    q=block(rng,rng.normal(scale=.1,size=(nq,n,n)),symmetric=True)
    pairs=[(0,0),(0,1),(1,1)];ranks=[2,3,2];centers=[0,1,0,1,0]
    domains={'PAIRS':pairs,'PAIR_OCC':[[1,0]]*3,'PAIR_AUX':[[4,1,0],[3,0,2,1],[1,2,4]],'PAIR_PAO':[paos]*3,
             'AO_OCC':[[0,1,2,3,4],[0,1,3,4,5]],'AO_VIR':[[1,2,3,4,5],[0,2,3,4,5]],
             'OCC':[[0],[1,0]],'PAO':[[2,3,5],[2,4,5]],
             'PAO_PAIRS':[]}
    domains['PAO_PAIRS']=[[v for pair in [(2,2),(3,3),(4,4),(5,5),(2,3),(2,5),(3,5)] for v in pair],
                          [v for pair in [(2,2),(3,3),(4,4),(5,5),(2,4),(3,5),(4,5)] for v in pair]]
    metadata={};frames={};one=[];two={}
    parent=np.eye(4)
    for p,rank,aux in zip(pairs,ranks,domains['PAIR_AUX']):
        frame=block(rng,np.eye(n)[:,2:2+rank]);frames[p]=frame
        metadata[p]=dict(occupied=[1,0],paos=paos,auxiliary=aux,parent=parent,first_frame=np.eye(n)[:,2:],
                         final_frame=frame.value,first_amplitude=np.zeros((4,4)),lmp2_amplitude=np.zeros((4,4)),
                         qi=np.zeros((len(aux),4)),qj=np.zeros((len(aux),4)),occupations=np.zeros(4),edges=[])
        t=block(rng,rng.normal(scale=.02,size=(rank,rank)));two[p]=t._replace(uv=np.zeros_like(t.value))
    for i in range(no):
        t=block(rng,rng.normal(scale=.02,size=frames[i,i].value.shape[1]));one.append(t._replace(uv=np.zeros_like(t.value)))
    w=core.DLPNOResponseWorkspace(domains,first_pno_domains(domains),metadata,{},centers,{},0.,cache)
    entry=w.entry(0,1,[s,f,l,d,j,q]);entry.set_frames(frames)
    def electronic(key):
        if key=='lmo':return l
        if key=='fock':return f
        kind,index=key
        return frames[index] if kind=='pno' else one[index] if kind=='t1' else two[index]
    fit=ScreenedFitting.__new__(ScreenedFitting)
    fit.wfn=None;fit.electronic=electronic;fit.S=s;fit.L=l
    fit.occupied_action=lambda x:algebra.mixed_product('mn,na->ma',d,x)
    fit.df=lambda aux:(j.slice(np.ix_(aux,aux)),q.slice(list(aux)))
    fit.domains=domains;fit.centers=centers;fit.amplitude_pairs=tuple(pairs)
    fit.pair_paos={p:paos for p in pairs};fit.pair_occ={p:(1,0) for p in pairs};fit.pair_aux=dict(zip(pairs,map(tuple,domains['PAIR_AUX'])))
    fit.parents={p:{'PARENT':SimpleNamespace(np=parent)} for p in pairs}
    fit.masks=[]
    for labels in domains['PAO_PAIRS']:
        mask=set(zip(labels[::2],labels[1::2]));fit.masks.append(mask|{(b,a) for a,b in mask})
    for name in ('paos','refit','pair_coefficients','raw','projected','mixed_integrals'):
        setattr(fit,name,lru_cache(None)(getattr(fit,name)))
    return SimpleNamespace(workspace=w,entry=entry,fit=fit,one=one,two=two,frames=frames,inputs=[s,f,l,d,j,q],metadata=metadata,domains=domains,centers=centers)


@pytest.mark.parametrize('budget',[0,4096,32*1024**2])
def test_complete_native_ccsd_with_local_masks(budget):
    x=fitting_fixture(cache=budget)
    expected=mixed_ccsd_residual(None,x.fit.electronic,x.fit.S,x.fit.df,fitted=x.fit)
    actual=x.entry.ccsd(x.one,x.two,'affine')
    for a,b in zip(actual[0],expected[0]):assert_blocks(a,b)
    for p,b in expected[1].items():assert_blocks(actual[1][p],b)
    for p in x.frames:
        for a,b in zip(x.entry.projected(p)[:5],x.fit.projected(p)[:5]):assert_blocks(a,b)
    # New entry, same reference/first inputs: reuse cannot change any order.
    other=x.workspace.entry(0,1,x.inputs);other.set_frames(x.frames)
    again=other.ccsd(x.one,x.two,'affine')
    for p in x.two:assert_blocks(again[1][p],actual[1][p])
    assert x.workspace.statistics[0]<=budget


def triples_fixture(monkeypatch,cache=0):
    from psi4.driver.procrouting.response.dlpno_screened_triples import ScreenedTriplesInputs
    from psi4.driver.procrouting.response.dlpno_triples_response import _energy_moment,_transform
    x=fitting_fixture(cache=cache)
    # Nonzero occupied coupling and two unequal triplet ranks, with an edge
    # whose occupied order differs from its canonical stored triplet.
    x.inputs[1].value[0,1]=x.inputs[1].value[1,0]=.025
    r1,r2=(0,0,1),(0,1,1);ranks={r1:2,r2:3}
    raw_edges={r1:[(0,1,1,0,1,0)],r2:[(1,0,0,0,1,0)]}
    basis={r:{'PAOS':SimpleNamespace(np=np.array([2,3,4,5])[:,None]),'PARENT':SimpleNamespace(np=np.eye(4))} for r in ranks}
    rows={}
    parent=np.eye(6)[:,2:]
    for r,rank in ranks.items():
        density=np.zeros((4,4))
        for p in ((r[0],r[1]),(r[1],r[2]),(r[0],r[2])):
            t=x.two[p].value;spin=2*t-t.T;transfer=x.frames[p].value.T@parent
            density+=transfer.T@(spin@t.T+spin.T@t)@transfer/(3*(1+(p[0]==p[1])))
        _,u=np.linalg.eigh(density);y=parent@u[:,-rank:]
        _,rotation=np.linalg.eigh(y.T@x.inputs[1].value@y);y=y@rotation
        zero=SimpleNamespace(np=np.zeros((rank,rank*rank)))
        rows[r]=((1,0),(4,1,0,3),SimpleNamespace(np=y),zero,zero,raw_edges[r],zero,zero)
    provider=ScreenedTriplesInputs.__new__(ScreenedTriplesInputs)
    provider.fit=x.fit;provider.rows=rows;provider.basis=basis;provider.first_cache=None
    provider.electronic=x.fit.electronic;provider.S=x.inputs[0];provider.F=x.inputs[1];provider.L=x.inputs[2]
    for name in ('parent','frame','projected','blocks','edge','occupied'):
        setattr(provider,name,lru_cache(None)(getattr(provider,name)))
    blocks={r:provider.blocks(r) for r in rows}
    gaps={}
    foo=x.inputs[2].value.T@x.inputs[1].value@x.inputs[2].value
    for r,(_,_,f) in blocks.items():
        eps=np.diag(f.value);gaps[r]=eps[:,None,None]+eps[None,:,None]+eps[None,None,:]-sum(foo[i,i] for i in r)
    keys=list(rows);cuts=np.cumsum([ranks[r]**3 for r in keys])[:-1]
    pack=lambda ts:np.concatenate([ts[r].ravel() for r in keys])
    unpack=lambda ts:{r:y.reshape((ranks[r],)*3) for r,y in zip(keys,np.split(ts,cuts))}
    def apply(flat):
        ts=unpack(flat);out={r:gaps[r]*ts[r] for r in keys}
        for r in keys:
            for l,m,i,j,k,_ in raw_edges[r]:
                order=(i,j,k);q=tuple(sorted(order));axes=np.argsort(np.argsort(order,kind='stable'))
                s=provider.edge(r,q).value
                out[r]-=foo[l,m]*_transform(ts[q].transpose(axes),s,s,s)
        return pack(out)
    size=sum(ranks[r]**3 for r in keys);dense=np.column_stack([apply(y) for y in np.eye(size)])
    ts=unpack(np.linalg.solve(dense,-pack({r:blocks[r][0].value for r in keys})))
    seeds={r:_energy_moment(r,blocks[r][1].value) for r in keys}
    zs=unpack(np.linalg.solve(dense.T,-pack(seeds)))
    energy=sum(np.vdot(seeds[r],ts[r]) for r in keys)
    metadata={}
    for r in rows:
        dom,aux,y,_,_,es,_,_=rows[r]
        rows[r]=(dom,aux,y,SimpleNamespace(np=ts[r]),SimpleNamespace(np=blocks[r][0].value),es,
                 SimpleNamespace(np=zs[r]),SimpleNamespace(np=blocks[r][1].value))
        metadata[r]=dict(occupied=dom,auxiliary=aux,paos=[2,3,4,5],parent=np.eye(4),frame=y.np,t=ts[r],z=zs[r],
                         w=blocks[r][0].value,v=blocks[r][1].value,edges=es)
    wfn=SimpleNamespace(variable=lambda name:energy if name=='(T) CORRECTION ENERGY' else SimpleNamespace(np=x.inputs[2].value),
                        reference_wavefunction=lambda:SimpleNamespace(Fa=lambda:SimpleNamespace(np=x.inputs[1].value)))
    provider.fit.wfn=wfn
    monkeypatch.setattr(core,'dlpno_lmp2_blocks',lambda *args:{p:SimpleNamespace(np=t.value) for p,t in x.two.items()},raising=False)
    monkeypatch.setattr(core,'dlpno_ccsd_df_pair_state',lambda w,i,j:([i],[SimpleNamespace(np=x.one[i].value)],{}),raising=False)
    monkeypatch.setattr(core,'get_option',lambda *args:1.e-11 if args[-1]=='R_CONVERGENCE' else 0.)
    w=core.DLPNOResponseWorkspace(x.domains,first_pno_domains(x.domains),x.metadata,metadata,x.centers,{},0.,cache)
    entry=w.entry(0,1,x.inputs);entry.set_frames(x.frames)
    native=entry.triples(x.one,x.two)
    return SimpleNamespace(x=x,workspace=w,entry=entry,native=native,provider=provider,keys=keys,energy=energy,dense=dense)


@pytest.mark.parametrize('cache',[0,64*1024**2])
def test_native_triples_complete_sources_operator_and_curvature(monkeypatch,cache):
    from psi4.driver.procrouting.response.dlpno_triples_response import TriplesHessianColumn
    x=triples_fixture(monkeypatch,cache)
    directions=[x.native.direction(k,2.e-10,x.energy) for k in (1,2)]
    reference=[x.provider._first_direction(k) for k in (1,2)]
    for actual,expected in zip(directions,reference):
        data=actual.data
        for key in ('frame_tangents','amplitudes','adjoints','gaps','energy_seeds','energy_seed_tangents','right_source','left_source'):
            for r,value in getattr(expected,key).items():np.testing.assert_allclose(data[key][r],value,atol=2.e-10,rtol=1.e-9)
    for r in x.keys:
        for a,b in zip(x.native.blocks(r),x.provider.blocks(r)):assert_blocks(a,b)
    rng=np.random.default_rng(40);probe=rng.normal(size=len(x.dense))
    np.testing.assert_allclose(directions[0].apply(probe,False),x.dense@probe,atol=2.e-12)
    np.testing.assert_allclose(directions[0].apply(probe,True),x.dense.T@probe,atol=2.e-12)
    column=TriplesHessianColumn(reference[1]);dt=np.concatenate([column.tangent.amplitudes[r].ravel() for r in x.keys]);dz=np.concatenate([column.tangent.adjoints[r].ravel() for r in x.keys])
    native_column=directions[1].column(dt,dz)
    expected=column.element(reference[0],lambda i,j:x.provider.occupied(i,j).uv,
                            x.provider.mixed_row,lambda r,q:x.provider.edge(r,q).uv)
    np.testing.assert_allclose(native_column.element(directions[0],x.native),expected,atol=2.e-11,rtol=1.e-9)
    # Second evaluation uses the column's retained overlap covectors.
    np.testing.assert_allclose(native_column.element(directions[0],x.native),expected,atol=2.e-11,rtol=1.e-9)


@pytest.mark.parametrize('budget',[0,32*1024**2])
@pytest.mark.parametrize('changed_final_domain',[None,'AO_OCC','AO_VIR','OCC','PAO'])
def test_native_pno_pipeline_and_first_solve_reuse(budget,changed_final_domain):
    from psi4.driver.procrouting.response.dlpno_pno_curvature import (
        first_pno_curvature,mixed_pair_exchange,lmp2_residual,pno_truncation_energy,mixed_pno_selection)
    x=fitting_fixture(cache=budget);C=algebra.mixed_product
    # A well-separated truncated PNO spectrum; domain masks are independently
    # exercised by the CCSD and triples fixtures above.
    x.domains.update(AO_OCC=[list(range(6))]*2,AO_VIR=[list(range(6))]*2,
                     OCC=[[0,1]]*2,PAO=[[2,3,4,5]]*2)
    # Reference state is held fixed; J/Q carry arbitrary analytic directions.
    inputs=[constant(z.value) if k<4 else z for k,z in enumerate(x.inputs)]
    S,F,L,D,J,Q=inputs;x.fit.S=S;x.fit.L=L
    x.fit.occupied_action=lambda panel:C('mn,na->ma',D,panel)
    x.fit.df=lambda aux:(J.slice(np.ix_(aux,aux)),Q.slice(list(aux)))
    foo=C('mi,mn,nj->ij',L,F,L);first={};initial={};final={};metadata={}
    edges={(0,0):[(0,1,0,1),(1,0,1,0)],(0,1):[(0,0,0,1),(1,1,0,1)],(1,1):[(1,0,0,1),(0,1,1,0)]}
    # Both orientation branches are exercised even at a diagonal occupied Fock.
    for p in x.frames:
        ids=x.fit.pair_paos[p];aux=x.fit.pair_aux[p];N=x.fit.paos(ids)
        qi=x.fit.raw('ov',aux,(p[0],),ids).slice((slice(None),0,slice(None)))
        qj=x.fit.raw('ov',aux,(p[1],),ids).slice((slice(None),0,slice(None)))
        K=mixed_pair_exchange(x.fit.df(aux)[0],qi,qj)
        # Choose the actual first retained subspace, then semicanonicalize it.
        parent=np.eye(6)[:,2:];f=parent.T@F.value@parent
        t=K.value/(foo.value[p[0],p[0]]+foo.value[p[1],p[1]]-np.diag(f)[:,None]-np.diag(f)[None,:])
        spin=2*t-t.T;density=(spin@t.T+spin.T@t)/(1+(p[0]==p[1]))
        e,u=np.linalg.eigh(density);y=parent@u[:,-3:];_,rot=np.linalg.eigh(y.T@F.value@y);y=y@rot
        out=first_pno_curvature(N,S,F,K,B(*(a[p[0],p[0]]+a[p[1],p[1]] for a in foo)),np.eye(4),y,diagonal=p[0]==p[1])
        first[p]=out;Y=out.selection.frame;coeff=C('ma,mn,nb->ab',out.parent,S,Y)
        initial[p]=dict(exchange=out.exchange,amplitudes=out.amplitudes,fock=out.fock,
                        fov=C('m,mn,na->a',L.slice((slice(None),p[0])),F,out.parent),occupied_energy=foo.slice((p[0],p[0])),rank=3)
        final[p]=dict(exchange=C('ma,mn,nb->ab',coeff,out.exchange,coeff),fock=C('ma,mn,nb->ab',Y,F,Y),
                      fov=C('m,mn,na->a',L.slice((slice(None),p[0])),F,Y),occupied_energy=foo.slice((p[0],p[0])),rank=2)
        metadata[p]=dict(x.metadata[p],parent=np.eye(4),first_frame=y,first_amplitude=out.amplitudes.value,
                         qi=qi.value,qj=qj.value,occupations=out.selection.occupations,edges=edges[p])
    keys=list(first)
    labels=[(p,a,b) for p in keys for a in range(3) for b in range(3) if p[0]!=p[1] or a<=b]
    def unpack(v,dual=False):
        out={p:np.zeros((3,3)) for p in keys}
        for value,(p,a,b) in zip(v,labels):
            out[p][a,b]=value/(2 if dual and p[0]==p[1] and a!=b else 1)
            if p[0]==p[1]:out[p][b,a]=out[p][a,b]
        return out
    def pack(x):return np.array([x[p][a,b] for p,a,b in labels])
    overlap=lambda p,q:C('ma,mn,nb->ab',first[p].selection.frame,S,first[q].selection.frame)
    zero=unpack(np.zeros(len(labels)))
    def stage(ts):return {p:dict(v,amplitudes=constant(ts[p])) for p,v in final.items()}
    base=lmp2_residual(stage(zero),foo,edges,overlap)
    rhs=pack({p:r.value for p,r in base.items()})
    matrix=np.column_stack([pack({p:r.value-base[p].value for p,r in lmp2_residual(stage(unpack(v)),foo,edges,overlap).items()}) for v in np.eye(len(labels))])
    t=unpack(np.linalg.solve(matrix,-rhs))
    for p in keys:
        metadata[p]['lmp2_amplitude']=t[p]
        u=2*t[p]-t[p].T;density=(u@t[p].T+u.T@t[p])/(1+(p[0]==p[1]))
        _,rot=np.linalg.eigh(density);y=first[p].selection.frame.value@rot[:,-2:]
        metadata[p]['final_frame']=y
    reference_final=stage(t)
    energy=pno_truncation_energy(initial,reference_final).value
    gradient=[]
    for v in np.eye(len(labels)):
        dt=unpack(v);trial={p:dict(a,amplitudes=B(t[p],dt[p],zero[p],zero[p])) for p,a in final.items()}
        # Only differentiate amplitudes when building the isolated loss adjoint.
        fixed_initial={p:{k:(constant(a.value) if isinstance(a,B) else a) for k,a in z.items()} for p,z in initial.items()}
        fixed_final={p:{k:(a if k=='amplitudes' else constant(a.value) if isinstance(a,B) else a) for k,a in z.items()} for p,z in trial.items()}
        gradient.append(pno_truncation_energy(fixed_initial,fixed_final).u)
    adjoint=unpack(np.linalg.solve(matrix.T,-np.array(gradient)),dual=True)
    final_domains=dict(x.domains)
    if changed_final_domain:
        # Each OV dependency independently prevents sharing with final CCSD.
        # Changing final maps must not alter the captured first-PNO response.
        final_domains[changed_final_domain]={
            'AO_OCC':[list(range(5)),list(range(1,6))],
            'AO_VIR':[list(range(5)),list(range(1,6))],
            'OCC':[[0],[1,0]],'PAO':[[2,3,5],[2,4,5]]}[changed_final_domain]
    workspace=core.DLPNOResponseWorkspace(final_domains,first_pno_domains(x.domains),metadata,{},x.centers,adjoint,float(energy),budget)
    calls=[]
    def first_solve(dy,dk,ds,df,dfoo):
        calls.append('first')
        s=B(S.value,ds,np.zeros_like(ds),np.zeros_like(ds));f=B(F.value,df,np.zeros_like(df),np.zeros_like(df))
        occupied=B(foo.value,dfoo,np.zeros_like(dfoo),np.zeros_like(dfoo))
        ys={p:B(first[p].selection.frame.value,dy[p],np.zeros_like(dy[p]),np.zeros_like(dy[p])) for p in keys}
        stage={p:dict(amplitudes=constant(t[p]),exchange=B(final[p]['exchange'].value,dk[p],zero[p],zero[p]),
                      fock=C('ma,mn,nb->ab',ys[p],f,ys[p])) for p in keys}
        residual=lmp2_residual(stage,occupied,edges,lambda p,q:C('ma,mn,nb->ab',ys[p],s,ys[q]))
        return unpack(np.linalg.solve(matrix,-pack({p:r.u for p,r in residual.items()})))
    def mixed_solve(source):
        calls.append('mixed');return unpack(np.linalg.solve(matrix,-pack(source)))
    result=workspace.entry(0,1,inputs).pno(first_solve,mixed_solve)
    assert calls.count('first')==2 and calls.count('mixed')==1
    for p,out in first.items():
        for name in ('parent','exchange','fock','amplitudes'):assert_blocks(result['first'][p][name],getattr(out,name),atol=1.e-9)
        assert_blocks(result['first'][p]['selection']['frame'],out.selection.frame,atol=1.e-9)
    complete={p:{k:(a if k=='rank' else B(*a)) for k,a in z.items()} for p,z in result['final'].items()}
    expected=pno_truncation_energy(initial,complete)
    assert_blocks(result['loss'],expected,atol=2.e-10)
    for p in keys:
        selected=mixed_pno_selection(first[p].selection.frame,S,complete[p]['amplitudes'],metadata[p]['final_frame'],diagonal=p[0]==p[1])
        assert_blocks(result['selection'][p]['frame'],selected.frame,atol=1.e-9)
    second=workspace.entry(0,1,inputs).pno(first_solve,mixed_solve)
    assert calls.count('first')==2 and calls.count('mixed')==2
    assert tuple(workspace.lmp2_counts)==(2,2)
    assert_blocks(second['loss'],result['loss'])
    reversed_inputs=[B(z.value,z.v,z.u,z.uv) for z in inputs]
    reversed_result=workspace.entry(1,0,reversed_inputs).pno(first_solve,mixed_solve)
    assert calls.count('first')==2
    assert_blocks(reversed_result['loss'],B(result['loss'][0],result['loss'][2],result['loss'][1],result['loss'][3]))


def test_native_retained_energy_reuses_fitting():
    from psi4.driver.procrouting.response.dlpno_ccsd_curvature import project_retained_ccsd_energy
    x=fitting_fixture(cache=16*1024**2)
    def pair_fit(p):
        _,ov,*_=x.fit.projected(p);dom=x.fit.pair_occ[p]
        return x.fit.df(x.fit.pair_aux[p])[0],ov.slice((slice(None),dom.index(p[0]),slice(None))),ov.slice((slice(None),dom.index(p[1]),slice(None)))
    expected=project_retained_ccsd_energy(x.inputs[2],x.frames,x.inputs[0],x.inputs[1],x.one,x.two,pair_fit)
    assert_blocks(x.entry.retained(x.one,x.two),expected)
    x.entry.ccsd(x.one,x.two,'affine')
    assert_blocks(x.entry.retained(x.one,x.two),expected)


@pytest.mark.parametrize('empty',[False,True])
def test_native_screened_corrections(monkeypatch,empty):
    from psi4.driver.procrouting.response.dlpno_screened_curvature import screened_pair_curvature
    from psi4.driver.procrouting.response.dlpno_pno_curvature import mixed_pair_three_center
    from psi4.driver.procrouting.response.dlpno_frame_curvature import mixed_paos
    x=fitting_fixture();S,F,L,D,J,Q=x.inputs
    # Keep the occupied projector fixed for the reference native parent.
    paos=(2,3,4,5);aux=(4,1,0,3)
    domains={'SCMP2_PAIRS':[] if empty else [(0,1)],'DIPOLE_PAIRS':[] if empty else [(0,1)],
             'AO_OCC':[list(range(6))]*2,'AO_VIR':[list(range(6))]*2,'DIPOLE_PAO':[paos,paos]}
    action=lambda p:algebra.mixed_product('mn,na->ma',D,p)
    N=mixed_paos(action,S,np.array(paos))[1]
    state={(0,0):{'PARENT':np.eye(4)},(1,1):{'PARENT':np.eye(4)},
           (0,1):{'PARENT':np.eye(4),'PAOS':np.array(paos)[:,None],'AUX':np.array(aux)[:,None]}}
    for label,key in ((0,'QI'),(1,'QJ')):
        panel=mixed_pair_three_center(S,L.slice((slice(None),label)),N,np.arange(6),np.arange(6),Q.slice(list(aux)))
        state[0,1][key]=panel.value
    x.workspace.screened_snapshot(domains,state)
    rng=np.random.default_rng(117)
    dip=[block(rng,np.diag(np.arange(6)*(k+1)),symmetric=True) for k in range(3)]
    actual=x.entry.screened(dip)
    values={'DLPNO SC-LMP2 PAIR ENERGY':actual[0][0],'DLPNO DIPOLE ENERGY':actual[1][0]}
    basis=SimpleNamespace(function_to_center=lambda q:x.centers[q])
    w=SimpleNamespace(variable=lambda k:values[k],reference_wavefunction=lambda:SimpleNamespace(get_basisset=lambda k:basis))
    monkeypatch.setattr(core,'dlpno_screened_pair_state',lambda w:(domains,{p:{k:SimpleNamespace(np=v) for k,v in b.items()} for p,b in state.items()}),raising=False)
    expected=screened_pair_curvature(w,S,F,L,action,dip,lambda aux:(J.slice(np.ix_(aux,aux)),Q.slice(list(aux))))
    assert_blocks(actual[0],expected['SCMP2']);assert_blocks(actual[1],expected['DIPOLE'])


def test_native_tensor_empty_strided_and_invalid_inputs():
    rng=np.random.default_rng(23)
    a=block(rng,np.empty((3,0)));b=block(rng,np.empty((0,2)))
    assert_blocks(core.dlpno_native_response_test('ab,bc->ac',[a,b]),constant(np.zeros((3,2))))
    a=B(*(rng.normal(size=(4,3))[:,::-1] for _ in range(4)));b=B(*(rng.normal(size=(3,2))[::-1] for _ in range(4)))
    assert_blocks(core.dlpno_native_response_test('ab,bc->ac',[a,b]),algebra._expanded_mixed_product('ab,bc->ac',a,b))
    with pytest.raises((ValueError,RuntimeError),match='dimension|shape|label mismatch'):
        core.dlpno_native_response_test('ab,bc->ac',[a,a])
    bad=a._replace(u=np.full((4,3),np.nan))
    with pytest.raises((ValueError,RuntimeError),match='Nonfinite'):
        core.dlpno_native_response_test('ab,bc->ac',[bad,b])
    with pytest.raises((ValueError,RuntimeError),match='Singular'):
        core.dlpno_native_response_test('solve',[constant(np.zeros((2,2))),constant(np.ones((2,1)))])


def test_native_adapter_snapshots_once_and_delivers_molecular_blocks(monkeypatch):
    from psi4.driver.procrouting.response.dlpno_native import NativeHessianWorkspace
    x=fitting_fixture();calls=[]
    def snapshot(name,result):
        def run(*args):calls.append(name);return result
        return run
    matrices=lambda data:{p:SimpleNamespace(np=v) for p,v in data.items()}
    blocks={name:matrices({p:record[field] for p,record in x.metadata.items()}) for name,field in
            [('FRAME','first_frame'),('FINAL_FRAME','final_frame'),('AMPLITUDE','lmp2_amplitude')]}
    blocks['RESPONSE_EDGES']=matrices({p:np.empty((0,4)) for p in x.metadata})
    first={p:{key:SimpleNamespace(np=data[field] if key!='OCCUPATIONS' else data[field][:,None]) for key,field in
              [('PARENT','parent'),('FRAME','first_frame'),('AMPLITUDE','first_amplitude'),('QI','qi'),('QJ','qj'),('OCCUPATIONS','occupations')]} for p,data in x.metadata.items()}
    monkeypatch.setattr(core,'dlpno_response_domain_state',snapshot('domains',x.domains),raising=False)
    monkeypatch.setattr(core,'dlpno_first_pno_state',snapshot('first',(first_pno_domains(x.domains),first)),raising=False)
    monkeypatch.setattr(core,'dlpno_lmp2_blocks',lambda w,key:blocks[key],raising=False)
    monkeypatch.setattr(core,'dlpno_triples_response_state',snapshot('triples',{}),raising=False)
    monkeypatch.setattr(core,'dlpno_triples_basis_state',snapshot('basis',{}),raising=False)
    monkeypatch.setattr(core,'dlpno_pno_truncation_lmp2_response',snapshot('adjoint',{}),raising=False)
    monkeypatch.setattr(core,'dlpno_screened_pair_state',snapshot('screened',({'SCMP2_PAIRS':[],'DIPOLE_PAIRS':[]},{})),raising=False)
    monkeypatch.setattr(core,'get_memory',lambda:2*1024**3,raising=False)
    basis=SimpleNamespace(nbf=lambda:len(x.centers),function_to_center=lambda i:x.centers[i])
    reference=SimpleNamespace(S=lambda:SimpleNamespace(np=x.inputs[0].value),Fa=lambda:SimpleNamespace(np=x.inputs[1].value))
    w=SimpleNamespace(variable=lambda key:SimpleNamespace(np=x.inputs[2].value) if key=='DLPNO RESPONSE ACTIVE LMOS' else 0.,
                      get_basisset=lambda name:basis,reference_wavefunction=lambda:reference,molecule=lambda:SimpleNamespace(natom=lambda:1))
    workspace=NativeHessianWorkspace(w)
    action=lambda x:None;action.density=x.inputs[3]
    df=lambda aux:None;df.full=tuple(x.inputs[4:])
    for _ in range(2):
        entry=workspace.entry(0,1,*x.inputs[:3],action,df);entry.set_frames(x.frames)
        for p in x.frames:
            for actual,expected in zip(entry.projected(p)[:5],x.fit.projected(p)[:5]):assert_blocks(actual,expected)
    assert all(calls.count(name)==1 for name in ('domains','first','triples','basis','adjoint','screened'))


def test_native_weighted_adjoint_and_complex_reference():
    from psi4.driver.procrouting.response.dlpno_ccsd_response import ccsd_weighted_dot
    rng=np.random.default_rng(73);one=[rng.normal(size=3),rng.normal(size=2)]
    two={(0,0):rng.normal(size=(3,3)),(0,1):rng.normal(size=(4,4)),(1,1):rng.normal(size=(2,2))}
    other_one=[x+.2 for x in one];other_two={p:x+.3 for p,x in two.items()}
    expected=sum(np.vdot(x,y) for x,y in zip(one,other_one))+sum((.5 if i==j else 1.)*np.vdot(x,other_two[i,j]) for (i,j),x in two.items())
    np.testing.assert_allclose(ccsd_weighted_dot(one,two,other_one,other_two),expected)
    complex_one=[x+1j for x in one]
    expected+=sum(np.vdot(1j*np.ones_like(x),y) for x,y in zip(one,other_one))
    np.testing.assert_allclose(ccsd_weighted_dot(complex_one,two,other_one,other_two),expected)


def test_native_cache_keeps_mixed_entries_ephemeral():
    x=fitting_fixture(cache=32*1024**2)
    first=x.entry.ccsd(x.one,x.two,'affine')
    changed=list(x.inputs)
    changed[0]=changed[0]._replace(uv=changed[0].uv+.03*np.eye(6))
    entry=x.workspace.entry(0,1,changed);entry.set_frames(x.frames)
    actual=entry.ccsd(x.one,x.two,'affine')
    fresh=fitting_fixture(cache=0)
    other=fresh.workspace.entry(0,1,changed);other.set_frames(x.frames)
    expected=other.ccsd(x.one,x.two,'affine')
    for p in x.two:
        assert_blocks(actual[1][p],expected[1][p])
        for k in (0,1,2):np.testing.assert_allclose(actual[1][p][k],first[1][p][k],atol=2.e-12)
    assert any(np.max(abs(actual[1][p][3]-first[1][p][3]))>1.e-8 for p in x.two)
    bad=list(changed);bad[0]=bad[0]._replace(u=bad[0].u+.1*np.eye(6))
    with pytest.raises((ValueError,RuntimeError),match='cached input'):
        x.workspace.entry(0,1,bad)


@pytest.mark.parametrize('budget',[0,64*1024**2])
def test_native_stage_panel_lifetime_and_bounds(budget):
    """Fixed/affine replays share whole fitting panels, including mixed parts."""
    x=fitting_fixture(cache=budget)
    a=x.entry.ccsd(x.one,x.two,'affine')
    counts=np.asarray(x.entry.panel_statistics)
    b=x.entry.ccsd(x.one,x.two,'affine')
    after=np.asarray(x.entry.panel_statistics)
    for one,two in zip(a[0],b[0]):assert_blocks(one,two)
    for p in a[1]:assert_blocks(a[1][p],b[1][p])
    if budget:
        # No raw/refit/overlap rebuild is needed in the second residual replay.
        np.testing.assert_equal(after[[0,2,4]],counts[[0,2,4]])
        assert after[1]>=counts[1] and after[5]>counts[5]
        assert 0<after[6]<=budget//4
    else:
        # J/K vertices are retained independently of optional panel admission;
        # with no panel allowance it is overlaps that must be rebuilt.
        assert after[4]>counts[4] and after[6]==0
    used,limit,*_=x.workspace.statistics
    pinned,pin_limit,*_=x.workspace.storage_statistics
    assert 0<=pinned<=pin_limit==budget//2
    assert pinned<=used<=limit==budget
    assert set(x.entry.ccsd_profile)=={'preparation','singles residual','occupied Fock','doubles residual'}


@pytest.mark.parametrize('transpose',range(4))
def test_native_shared_matrix_doublet(transpose):
    rng=np.random.default_rng(991)
    ta,tb=bool(transpose&1),bool(transpose&2)
    a=block(rng,rng.normal(size=(3,4) if ta else (4,3)))
    b=block(rng,rng.normal(size=(2,3) if tb else (3,2)))
    eq=('ki' if ta else 'ik')+','+('jk' if tb else 'kj')+'->ij'
    out=core.dlpno_native_response_test('doublet',[a,b],transpose)
    assert_blocks(out,algebra._expanded_mixed_product(eq,a,b))
    # The ndarray capsule must keep its SharedMatrix alive after the C++
    # tensors/context have been destroyed, and expose read-only storage.
    import gc
    gc.collect()
    assert out[0].shape==(4,2) and not out[0].flags.writeable
    assert np.isfinite(out[3]).all()


@pytest.mark.parametrize('shape,ranks',[((3,4,5),(2,3)),((3,4,5),(5,1))])
def test_native_binary_virtual_transform_both_orders(shape,ranks):
    rng=np.random.default_rng(665)
    q=block(rng,rng.normal(size=shape))
    left=block(rng,rng.normal(size=(shape[1],ranks[0])))
    right=block(rng,rng.normal(size=(shape[2],ranks[1])))
    assert_blocks(core.dlpno_native_response_test('virtual_transform',[q,left,right]),
                  algebra._expanded_mixed_product('ma,Qmn,nb->Qab',left,q,right))
    matrix=block(rng,rng.normal(size=shape[1:]))
    assert_blocks(core.dlpno_native_response_test('congruence',[left,matrix,right]),
                  algebra._expanded_mixed_product('ma,mn,nb->ab',left,matrix,right))


@pytest.mark.parametrize('budget',[0,4096,64*1024**2])
@pytest.mark.parametrize('filtered',[False,True])
@pytest.mark.parametrize('small_rank',[False,True])
def test_native_ccsd_mixed_vertices_keep_target_fit_and_source_pao_order(budget,filtered,small_rank):
    x=fitting_fixture(seed=902,cache=0)
    rng=np.random.default_rng(903)
    pairs=list(x.frames)
    lists=[(5,2,4),(4,5,3,2),(3,2)]
    for p,ids in zip(pairs,lists):
        rank=1 if small_rank else x.frames[p].value.shape[1]
        # Rank one favors virtual-first mixed J; the original ranks favor
        # auxiliary-first, exercising both complete contraction paths.
        x.two[p]=x.two[p].slice((slice(0,rank),slice(0,rank)))
        if p[0]==p[1]:x.one[p[0]]=x.one[p[0]].slice(slice(0,rank))
        frame=block(rng,np.eye(6)[:,ids[:rank]])
        x.frames[p]=frame
        x.metadata[p].update(paos=ids,parent=np.eye(len(ids)),first_frame=np.eye(6)[:,ids],
            final_frame=frame.value,first_amplitude=np.zeros((len(ids),len(ids))),
            lmp2_amplitude=np.zeros((len(ids),len(ids))),qi=np.zeros((len(x.metadata[p]['auxiliary']),len(ids))),
            qj=np.zeros((len(x.metadata[p]['auxiliary']),len(ids))),occupations=np.zeros(len(ids)))
        x.fit.pair_paos[p]=ids
        x.fit.parents[p]={'PARENT':SimpleNamespace(np=np.eye(len(ids)))}
    x.domains['PAIR_PAO']=lists
    if filtered:
        # A stable discarded eigenspace exercises the H^T H convention; the
        # full inverse on nonprojected vertices must remain distinct.
        x.inputs[4].value[:]=np.diag([-.3,.7,1.2,1.7,2.1])
    w=core.DLPNOResponseWorkspace(x.domains,first_pno_domains(x.domains),x.metadata,{},x.centers,{},0.,budget)
    e=w.entry(0,1,x.inputs);e.set_frames(x.frames)
    expected=mixed_ccsd_residual(None,x.fit.electronic,x.fit.S,x.fit.df,fitted=x.fit)
    actual=e.ccsd(x.one,x.two,'affine')
    for a,b in zip(actual[0],expected[0]):assert_blocks(a,b)
    for p,b in expected[1].items():assert_blocks(actual[1][p],b)
    for i,j in [(0,0),(0,1),(1,0),(1,1)]:
        for k in (0,1):
            for a,b in zip(e.mixed_integrals(i,j,k),x.fit.mixed_integrals(i,j,k)):
                assert_blocks(a,b)
    if budget>=64*1024**2:
        sizes=e.integral_storage
        assert all(name in sizes for name in ('qij','qia','qab','PNO factors','mixed J/K vertices'))
        assert all(len(v)==4 and min(v)>0 for v in sizes.values())
        assert sum(sum(sizes[name]) for name in ('qij','qia','qab'))<=e.panel_statistics[6]
