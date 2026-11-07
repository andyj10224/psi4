"""Known rotating quadratic localization maximum and degenerate-branch guard."""
import numpy as np
from psi4.driver.procrouting.response.dlpno_triples_curvature import MixedBlock,mixed_product
from psi4.driver.procrouting.response.dlpno_localization_curvature import quadratic_localization_mixed


def verify():
    rng=np.random.default_rng(527)
    for n in (1,4):
        const=lambda x:MixedBlock(x,np.zeros_like(x),np.zeros_like(x),np.zeros_like(x))
        def skew():
            a=rng.normal(size=(n,n))*.1; return a-a.T
        K,H,J=skew(),skew(),skew()
        rotation=MixedBlock(np.eye(n),K,H,J+.5*(K@H+H@K))
        operators=[mixed_product('ai,ij,bj->ab',rotation,const(np.diag(rng.normal(size=n))),rotation) for _ in range(3)]
        result=quadratic_localization_mixed(np.eye(n),const(np.eye(n)),const(np.eye(n)),operators)
        error=max(np.max(abs(x-y)) for x,y in zip(result,rotation))
        print('Known rotating localization',n,error)
        assert error<1.e-12
    try:
        quadratic_localization_mixed(np.eye(n),const(np.eye(n)),const(np.eye(n)),[const(np.eye(n))])
    except RuntimeError:
        pass
    else:
        raise AssertionError('Singular localization maximum was accepted')


if __name__=='__main__': verify()
