"""Contracted overlap curvature versus the original full rank-three transfer."""
from itertools import permutations
from types import SimpleNamespace
import numpy as np
import pytest

from psi4.driver.procrouting.response.dlpno_triples_response import (
    _TriplesOverlapCurvature,_mixed_transfer)


@pytest.mark.parametrize('nr,ns',[(1,1),(2,3),(3,2),(4,4)])
@pytest.mark.parametrize('axes',list(permutations(range(3))))
def test_contracted_overlap_curvature(nr,ns,axes):
    rng=np.random.default_rng(512)
    draw=lambda shape:rng.normal(scale=.3,size=shape)
    z,t=draw((nr,)*3),draw((ns,)*3).transpose(axes)
    s,sv=draw((nr,ns)),draw((nr,ns))
    f,fv=.4,-.2
    ev=SimpleNamespace(overlap=s,doverlap=sv,fock=f,dfock=fv)
    cache=_TriplesOverlapCurvature(z,t,s,sv)
    assert not cache.bar.flags.writeable and not cache.column_bar.flags.writeable
    for _ in range(3):
        su,suv=draw(s.shape),draw(s.shape)
        fu,fuv=.7,-.5
        eu=SimpleNamespace(overlap=s,doverlap=su,fock=f,dfock=fu)
        expected=np.vdot(z,_mixed_transfer(t,eu,ev,suv,fuv))
        actual=cache.element(eu,ev,suv,fuv)
        np.testing.assert_allclose(actual,expected,atol=2e-12,rtol=2e-12)
        # Independent scalar differentiation, including the explicit uv terms.
        def energy(u,v):
            ss=s+u*su+v*sv+u*v*suv
            ff=f+u*fu+v*fv+u*v*fuv
            return ff*np.einsum('ijk,ia,jb,kc,abc->',z,ss,ss,ss,t,optimize=False)
        def central(h):return (energy(h,h)-energy(h,-h)-energy(-h,h)+energy(-h,-h))/(4*h*h)
        reference=(4*central(.001)-central(.002))/3
        np.testing.assert_allclose(actual,reference,atol=2e-9,rtol=0.)


def test_overlap_curvature_rejects_incompatible_ranks():
    with pytest.raises(ValueError,match='dimensions'):
        _TriplesOverlapCurvature(np.zeros((2,2,3)),np.zeros((3,3,3)),np.zeros((2,3)),np.zeros((2,3)))


def test_complete_triples_columns_against_uncontracted_residual():
    """Retain the original full residual as an oracle for all ordered entries.

    Two coupled unequal-rank spaces exercise edge direction, permutations,
    occupied/virtual Fock terms and both actual tangent solves. The dense
    reference operator fixes the adjoint without using the iterative solver.
    """
    from psi4.driver.procrouting.response.dlpno_triples_response import (
        TriplesGradientDirection,TriplesHessianColumn,TriplesMixedRow,_TriplesEdge,
        _energy_moment,_local_fock_action)
    rng=np.random.default_rng(825)
    draw=lambda shape:rng.normal(scale=.1,size=shape)
    ranks={(0,0,1):2,(0,1,2):3}
    keys=list(ranks);cuts=np.cumsum([n**3 for n in ranks.values()])[:-1]
    unpack=lambda x:{r:b.reshape((ranks[r],)*3) for r,b in zip(keys,np.split(x,cuts))}
    pack=lambda x:np.concatenate([x[r].ravel() for r in keys])
    tensors={r:draw((n,)*3) for r,n in ranks.items()}
    adjoints={r:draw(t.shape) for r,t in tensors.items()}
    gaps={r:3.+abs(draw(t.shape)) for r,t in tensors.items()}
    overlaps={(r,q):draw((ranks[r],ranks[q])) for r,q in (keys,keys[::-1])}
    edges={r:[_TriplesEdge(q,(2,0,1),(0,2),.2,0.,overlaps[r,q],np.zeros_like(overlaps[r,q]))]
           for r,q in (keys,keys[::-1])}
    def apply(flat):
        x=unpack(flat);result={r:gaps[r]*x[r] for r in keys}
        for r in keys:
            for edge in edges[r]:
                s=edge.overlap
                result[r]-=edge.fock*np.einsum('ia,jb,kc,abc->ijk',s,s,s,
                    x[edge.source].transpose(edge.axes),optimize=False)
        return pack(result)
    size=sum(t.size for t in tensors.values())
    dense=np.column_stack([apply(e) for e in np.eye(size)])
    seeds=unpack(-dense.T@pack(adjoints))
    wfn=SimpleNamespace(variable=lambda name:SimpleNamespace(np=np.zeros((4,3))))
    directions=[]
    for _ in range(3):
        first_edges={r:[e._replace(dfock=float(draw(())),doverlap=draw(e.overlap.shape))
                        for e in edges[r]] for r in keys}
        data=lambda:{r:draw(t.shape) for r,t in tensors.items()}
        directions.append(TriplesGradientDirection(wfn,{},tensors,adjoints,gaps,seeds,
                                                   data(),data(),data(),first_edges))
    for column in directions:
        kernel=TriplesHessianColumn(column)
        for row in directions:
            occupied=draw((3,3))
            mixed={r:TriplesMixedRow(draw(t.shape),draw(t.shape),draw((len(t),len(t))))
                   for r,t in tensors.items()}
            suv={pair:draw(s.shape) for pair,s in overlaps.items()}
            expected=kernel.relaxation(row)
            for r in keys:
                t,z=tensors[r],adjoints[r]
                w,v,f=mixed[r]
                expected+=np.vdot(_energy_moment(r,v),t)
                residual=w+_local_fock_action(f,t,sum(occupied[a,a] for a in r))
                for eu,ev in zip(row.edges[r],column.edges[r]):
                    residual-=_mixed_transfer(tensors[eu.source].transpose(eu.axes),eu,ev,
                                              suv[r,eu.source],occupied[eu.occupied])
                expected+=np.vdot(z,residual)
            actual=kernel.element(row,occupied,mixed.__getitem__,lambda r,q:suv[r,q])
            np.testing.assert_allclose(actual,expected,atol=2.e-12,rtol=0.)
            assert len(kernel._overlap_curvatures)==2
