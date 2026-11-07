"""Sparse unique-pair energy against an independently enumerated ordered sum."""
import numpy as np

from psi4.driver.procrouting.response.dlpno_ccsd_curvature import retained_ccsd_energy
from psi4.driver.procrouting.response.dlpno_triples_curvature import MixedBlock


def verify():
    rng = np.random.default_rng(113)
    def block(shape):
        return MixedBlock(*(rng.normal(size=shape)*s for s in (1.,.3,.4,.1)))
    def at(x,u,v):
        return x.value + u*x.u + v*x.v + u*v*x.uv
    ranks = {(0,0):3,(1,1):2,(2,2):4,(0,1):2,(0,2):3}
    t1 = [block((ranks[i,i],)) for i in range(3)]
    f = [block(x.value.shape) for x in t1]
    k = {p:block((n,n)) for p,n in ranks.items()}
    t2 = {p:block((n,n)) for p,n in ranks.items()}
    for (i,j) in ranks:
        if i==j:
            k[i,j] = MixedBlock(*((x+x.T)/2 for x in k[i,j]))
            t2[i,j] = MixedBlock(*((x+x.T)/2 for x in t2[i,j]))
    ov = {(p,i):block((n,ranks[i,i])) for p,n in ranks.items() for i in set(p)}
    def independent(u,v):
        value = sum(2*np.dot(at(x,u,v),at(y,u,v)) for x,y in zip(f,t1))
        for i in range(3):
            for j in range(3):
                p = tuple(sorted((i,j)))
                if p not in ranks:
                    continue
                t,g = at(t2[p],u,v),at(k[p],u,v)
                if j < i:
                    t,g = t.T,g.T
                ti = at(ov[p,i],u,v)@at(t1[i],u,v)
                tj = at(ov[p,j],u,v)@at(t1[j],u,v)
                value += np.vdot(2*g-g.T,t+np.outer(ti,tj))
        return value
    e = retained_ccsd_energy(t1,t2,lambda p,i:ov[p,i],k.get,f.__getitem__)
    def mixed(h):
        return sum(i*j*independent(i*h,j*h) for i,j in ((1,1),(-1,1),(1,-1),(-1,-1)))/(4*h*h)
    np.testing.assert_allclose(e.value,independent(0,0),atol=1.e-12,rtol=0.)
    np.testing.assert_allclose(e.uv,(4*mixed(.001)-mixed(.002))/3,atol=3.e-8,rtol=0.)
    print('Sparse pair curvature error',abs(e.uv-(4*mixed(.001)-mixed(.002))/3))


if __name__ == '__main__':
    verify()
