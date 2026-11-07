"""Mixed bare DF integrals on ordered, noncontiguous AO and auxiliary domains."""
import numpy as np
import psi4
from water_hvp import BASE, molecule


def reference(coords):
    mol = molecule(coords)
    w = psi4.core.Wavefunction.build(mol,'cc-pvdz')
    aux = psi4.core.BasisSet.build(mol,'DF_BASIS_CC','cc-pvdz-ri','RIFIT','cc-pvdz')
    w.set_basisset('DF_BASIS_CC',aux)
    return w


def verify():
    w = reference(BASE)
    aux = list(range(w.get_basisset('DF_BASIS_CC').nbf()-1,-1,-2))
    aos = list(range(w.basisset().nbf()-1,-1,-1))[1::2]
    assert len(set(aux)) == len(aux) and len(aos) > 2
    def analytic(u,v):
        m,q = psi4.core.dlpno_df_mixed_integrals(w,'DF_BASIS_CC',aux,aos,u,v)
        return m.np.copy(),q.np.copy()
    def first(coords,atom,xyz):
        mints = psi4.core.MintsHelper(reference(coords))
        m = mints.ao_metric_deriv1(atom,'DF_BASIS_CC')[xyz].np
        q = mints.ao_3center_deriv1(atom,'DF_BASIS_CC')[xyz].np
        return m[np.ix_(aux,aux)],q[np.ix_(aux,aos,aos)]
    largest = np.zeros(2)
    for u,v in ((1,5),(4,5),(5,5),(5,7)):
        atom,xyz = divmod(u,3)
        direction = np.eye(9)[v].reshape(3,3)
        def central(h):
            plus = first(BASE+h*direction,atom,xyz)
            minus = first(BASE-h*direction,atom,xyz)
            return [(a-b)/(2*h) for a,b in zip(plus,minus)]
        coarse,fine = central(.002),central(.001)
        for k,(a,c,f,b) in enumerate(zip(analytic(u,v),coarse,fine,analytic(v,u))):
            np.testing.assert_allclose(a,b,atol=2.e-12,rtol=0.)
            err = np.max(abs(a-(4*f-c)/3))
            largest[k] = max(largest[k],err)
            assert err < 2.e-9
    print('water mixed AO metric and three-center errors',largest)


if __name__ == '__main__':
    verify()
