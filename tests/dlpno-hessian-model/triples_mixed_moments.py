"""Mixed moment product rules against displaced local inputs, including a filtered metric."""
from itertools import permutations
import numpy as np

from psi4.driver.procrouting.response.dlpno_triples_curvature import MixedBlock, mixed_triples_moments, _filtered_inverse, _inverse


def verify():
    rng = np.random.default_rng(534)
    domain = (0,1,2)
    triplets = ((0,1,2),(0,1,1))
    ranks = (3,4,2)
    n,aux = 3,4
    def block(shape):
        return MixedBlock(*(rng.normal(size=shape)*s for s in (1.,.1,.15,.04)))
    def at(x,u,v):
        return x.value+u*x.u+v*x.v+u*v*x.uv
    metric = block((aux,aux))
    rot,_ = np.linalg.qr(rng.normal(size=(aux,aux)))
    mu,mv,muv = ((a+a.T)/2 for a in metric[1:])
    oo,ov,vv = block((aux,3,3)),block((aux,3,n)),block((aux,n,n))
    fock = block((n,n))
    pair_rank = {(i,j):ranks[(i+j)%3] for i in domain for j in domain}
    ss = {(i,j):block((m,n)) for (i,j),m in pair_rank.items()}
    tt = {(i,j):block((m,m)) for (i,j),m in pair_rank.items()}
    ts = {i:block((pair_rank[i,i],)) for i in domain}
    fa = {i:block((n,)) for i in domain}

    # Positive semidefinite metric with a discarded positive eigenvalue
    # and repeated retained eigenvalues. Rotate its eigenvectors in two
    # directions; the filtered inverse rotates by the same orthogonal maps.
    def commutator(a,b):
        return a@b-b@a
    k,l = (x-x.T for x in rng.normal(size=(2,aux,aux)))
    def rotated(d):
        du,dv = commutator(k,d),commutator(l,d)
        return MixedBlock(d,du,dv,commutator(k,dv))
    m = rotated(np.diag([1.e-16,.8,.8,2.]))
    exact = rotated(np.diag([0.,1.25,1.25,.5]))
    np.testing.assert_allclose(_filtered_inverse(m,_inverse(m)),exact,rtol=0.,atol=2.e-12)

    # Negative out-of-spectrum eigenvalues provide a stable filtered branch.
    # Repeated eigenvalues exercise the continuous divided differences.
    for eigenvalues in ((.7,.7,1.2,1.9),(-.5,-.5,.7,.7)):
        metric = MixedBlock((rot*np.array(eigenvalues))@rot.T,mu,mv,muv)
        for r in triplets:
            mixed = mixed_triples_moments(r,domain,metric,oo,ov,vv,
                                          lambda a,b:ss[a,b],ts.get,lambda a,b:tt[a,b],fa.get,fock)
            def moments(u,v):
                m = at(metric,u,v)
                eig,y = np.linalg.eigh(m)
                active = eig >= 1.e-14*max(abs(eig))
                filtered = (y[:,active]/eig[active]) @ y[:,active].T
                raw_oo,raw_ov,raw_vv = (at(x,u,v) for x in (oo,ov,vv))
                def pair(i,j):
                    overlap = at(ss[i,j],u,v)
                    return overlap.T@at(tt[i,j],u,v)@overlap
                w = np.zeros((n,n,n))
                for perm in permutations(range(3)):
                    a,b,c = (r[j] for j in perm)
                    k = np.linalg.solve(m,raw_ov[:,a]).T@raw_vv.reshape(aux,-1)
                    part = np.einsum('abk,ck->abc',k.reshape((n,)*3),pair(c,b))
                    vertex = raw_oo[:,b].T@filtered@raw_ov[:,c]
                    for l in domain:
                        part -= pair(a,l)[:,:,None]*vertex[l]
                    w += part.transpose(np.argsort(perm))
                energy = w.copy()
                for slot,a in enumerate(r):
                    b,c = [r[i] for i in range(3) if i != slot]
                    single = at(ss[a,a],u,v).T@at(ts[a],u,v)
                    g = raw_ov[:,b].T@filtered@raw_ov[:,c]
                    extra = single[:,None,None]*g + at(fa[a],u,v)[:,None,None]*pair(b,c)
                    energy += np.moveaxis(extra,0,slot)
                return np.stack((w,energy))
            def central(h):
                return (moments(h,h)-moments(h,-h)-moments(-h,h)+moments(-h,-h))/(4*h*h)
            expected = (4*central(.001)-central(.002))/3
            diff = np.max(abs(expected-np.stack(mixed[:2])))
            print('mixed W,V error',r,eigenvalues,diff)
            np.testing.assert_allclose(np.stack(mixed[:2]),expected,rtol=1.e-7,atol=1.e-7)


if __name__ == '__main__':
    verify()
