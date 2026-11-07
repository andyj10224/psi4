"""Small independent four-index oracle for factorized mixed DF Fock products."""
import numpy as np
from psi4.driver.procrouting.response.dlpno_triples_curvature import MixedBlock
from psi4.driver.procrouting.response.dlpno_rhf_curvature import mixed_df_fock


def verify():
    rng=np.random.default_rng(733)
    def block(shape):
        return MixedBlock(*(rng.normal(size=shape)*s for s in (1.,.1,.1,.03)))
    j=block((6,6)); j=MixedBlock(*((x+x.T)/2 for x in j))
    j=j._replace(value=j.value@j.value.T+np.eye(6))
    q=block((6,4,4)); q=MixedBlock(*((x+x.swapaxes(1,2))/2 for x in q))
    d=block((4,4)); d=MixedBlock(*((x+x.T)/2 for x in d))
    def at(x,u,v): return x.value+u*x.u+v*x.v+u*v*x.uv
    def scalar(u,v):
        J,Q,D=(at(x,u,v) for x in (j,q,d))
        eri=(Q.reshape(6,-1).T@np.linalg.solve(J,Q.reshape(6,-1))).reshape((4,)*4)
        return 2*np.einsum('mnrs,rs->mn',eri,D)-np.einsum('mrns,rs->mn',eri,D)
    def fd(h):
        return sum(a*b*scalar(a*h,b*h) for a in (-1,1) for b in (-1,1))/(4*h*h)
    result=mixed_df_fock(j,q,d)
    error=np.max(abs(result.uv-(4*fd(.001)-fd(.002))/3))
    assert error<2.e-8,error
    swap=lambda x:MixedBlock(x.value,x.v,x.u,x.uv)
    np.testing.assert_allclose(result.uv,mixed_df_fock(swap(j),swap(q),swap(d)).uv,atol=1.e-13,rtol=0)
    print('Factorized mixed DF Fock error',error)


if __name__=='__main__': verify()
