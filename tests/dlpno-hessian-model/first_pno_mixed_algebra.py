"""Independent displaced generalized-eigenproblem oracle for mixed first PNOs."""
import numpy as np
from scipy.linalg import eigh
from psi4.driver.procrouting.response.dlpno_triples_curvature import MixedBlock, mixed_product
from psi4.driver.procrouting.response.dlpno_pno_curvature import (
    first_pno_curvature, mixed_pair_three_center, mixed_pair_exchange, mixed_pno_selection)


def check():
    rng = np.random.default_rng(9284)
    def polynomial(shape, symmetric=False):
        parts = [rng.normal(size=shape)*scale for scale in (1., .1, .1, .03)]
        if symmetric: parts = [(x+x.T)/2 for x in parts]
        return MixedBlock(*parts)
    def value(block, u, v):
        return block.value + u*block.u + v*block.v + u*v*block.uv
    def inverse_root(a):
        e, z = eigh(a)
        return (z/e**.5)@z.T
    def numerical(fun, part):
        def estimate(h):
            if part == 1: return (fun(h,0)-fun(-h,0))/(2*h)
            if part == 2: return (fun(0,h)-fun(0,-h))/(2*h)
            return sum(a*b*fun(a*h,b*h) for a in (-1,1) for b in (-1,1))/(4*h*h)
        return (4*estimate(.002)-estimate(.004))/3

    n, m, count, nq = 8, 6, 4, 7
    S = polynomial((n,n), True)
    S = S._replace(value=S.value@S.value.T+np.eye(n))
    N = polynomial((n,m))
    F = polynomial((n,n), True)
    J = polynomial((nq,nq), True)
    J = J._replace(value=J.value@J.value.T+np.eye(nq))
    left, right = np.array([0,2,4,6]), np.array([1,2,3,5,7])
    Q = polynomial((nq,len(left),len(right)))
    li, lj = polynomial((n,)), polynomial((n,))
    qi = mixed_pair_three_center(S,li,N,left,right,Q)
    qj = mixed_pair_three_center(S,lj,N,left,right,Q)
    K = mixed_pair_exchange(J,qi,qj)
    def refit(l,u,v):
        s = value(S,u,v)
        c = np.linalg.solve(s[np.ix_(left,left)],s[left]@value(l,u,v))
        return np.einsum('m,Qmn,na->Qa',c,value(Q,u,v),value(N,u,v)[right])
    for analytic, fun in ((qi,lambda u,v:refit(li,u,v)),
                          (K,lambda u,v:refit(li,u,v).T@np.linalg.solve(value(J,u,v),refit(lj,u,v)))):
        for part in (1,2,3):
            np.testing.assert_allclose(analytic[part],numerical(fun,part),atol=2.e-7,rtol=1.e-8)

    pivots = np.array([0,2,3,5])
    X = np.zeros((m,count))
    X[pivots] = inverse_root(N.value[:,pivots].T@S.value@N.value[:,pivots])
    occupied = MixedBlock(np.asarray(-7.),np.asarray(.2),np.asarray(-.3),np.asarray(.1))
    maximum = 0.
    for diagonal in (False,True):
        exchange = MixedBlock(*((x+x.T)/2 for x in K)) if diagonal else K
        def independent(u,v,rank):
            # Recompute generalized semicanonical orbitals, amplitudes and PNOs.
            y = value(N,u,v)@X
            e,z = eigh(y.T@value(F,u,v)@y,y.T@value(S,u,v)@y)
            y = y@z
            k = z.T@X.T@value(exchange,u,v)@X@z
            t = k/(value(occupied,u,v)-e[:,None]-e[None,:])
            spin = 2*t-t.T
            _,p = eigh((spin@t.T+spin.T@t)/(1+diagonal))
            kept = y@p[:,-rank:]
            return kept@kept.T
        for rank in (2,count):
            base = independent(0,0,rank)
            # An arbitrary metric-orthonormal frame of the retained AO projector.
            e,z = eigh(S.value)
            root = (z*np.sqrt(e))@z.T
            _,z = eigh(root@base@root)
            reference = np.linalg.solve(root,z[:,-rank:])
            result = first_pno_curvature(N,S,F,exchange,occupied,X,reference,diagonal=diagonal)
            frame = result.selection.frame
            density = mixed_product('ma,na->mn',frame,frame)
            for part in (1,2,3):
                target = numerical(lambda u,v:independent(u,v,rank),part)
                maximum = max(maximum,np.max(abs(density[part]-target)))
                np.testing.assert_allclose(density[part],target,atol=2.e-7,rtol=0)
            gram = mixed_product('ma,mn,nb->ab',frame,S,frame)
            np.testing.assert_allclose(gram.value,np.eye(rank),atol=2.e-12,rtol=0)
            for x in gram[1:]: np.testing.assert_allclose(x,0,atol=2.e-12,rtol=0)
            squared = mixed_product('mi,ij,jn->mn',density,S,density)
            for x,y in zip(squared,density): np.testing.assert_allclose(x,y,atol=2.e-12,rtol=0)
            t = result.amplitudes
            lhs = mixed_product(',ab->ab',occupied,t)
            ft = mixed_product('ac,cb->ab',result.fock,t)
            tf = mixed_product('ac,cb->ab',t,result.fock)
            for a,b,c,k in zip(lhs,ft,tf,result.exchange):
                np.testing.assert_allclose(a-b-c,k,atol=2.e-11,rtol=0)
            swap = lambda block: MixedBlock(block.value,block.v,block.u,block.uv)
            reversed_result = first_pno_curvature(*(swap(x) for x in (N,S,F,exchange,occupied)),
                                                  X,reference,diagonal=diagonal)
            np.testing.assert_allclose(frame.uv,reversed_result.selection.frame.uv,atol=2.e-12,rtol=0)
            transposed = first_pno_curvature(N,S,F,MixedBlock(*(x.T for x in exchange)),occupied,
                                             X,reference,diagonal=diagonal)
            np.testing.assert_allclose(frame.uv,transposed.selection.frame.uv,atol=2.e-11,rtol=0)
            rotation,_ = np.linalg.qr(rng.normal(size=(count,count)))
            rotated = first_pno_curvature(N,S,F,exchange,occupied,X@rotation,reference,diagonal=diagonal)
            for a,b in zip(frame,rotated.selection.frame):
                np.testing.assert_allclose(a,b,atol=2.e-11,rtol=0)
    const = lambda x: MixedBlock(x,np.zeros_like(x),np.zeros_like(x),np.zeros_like(x))
    # Degenerate virtual energies and occupations within each cluster require
    # no eigenvector gauge; only the retained/discarded boundary must be resolved.
    degenerate = polynomial((4,4),True)._replace(value=np.diag([.1,.1,.4,.4]))
    fixed = const(np.eye(4))
    out = first_pno_curvature(fixed,fixed,fixed,degenerate,const(np.asarray(-2.)),
                              np.eye(4),np.eye(4)[:,2:],diagonal=True)
    def projector(u,v):
        t = value(degenerate,u,v)/-4.
        _,z = eigh(t@t)
        return z[:,2:]@z[:,2:].T
    density = mixed_product('ma,na->mn',out.selection.frame,out.selection.frame)
    np.testing.assert_allclose(density.uv,numerical(projector,3),atol=2.e-8,rtol=0)
    try:
        mixed_pno_selection(const(np.eye(3)),const(np.eye(3)),const(np.eye(3)),np.eye(3)[:,:2])
    except ValueError as error:
        assert 'boundary' in str(error)
    else:
        raise AssertionError('Unresolved retained/discarded occupation boundary was accepted')
    print('Mixed first-PNO algebra maximum projector error',maximum,flush=True)


if __name__ == '__main__':
    check()
