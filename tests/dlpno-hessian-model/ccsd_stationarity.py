"""Coefficient extraction and fixed-rank projector checks for mixed CCSD stationarity."""
import numpy as np

from psi4.driver.procrouting.response.dlpno_ccsd_response import (
    ccsd_amplitude_curvature, semi_analytic_ccsd_mixed_source)
from psi4.driver.procrouting.response.dlpno_triples_curvature import MixedBlock, _mixed_projector


def polynomial_check():
    # Occupied-pair block structure with singly and multiply powered amplitudes.
    base = ([np.array([.13]),np.array([-.21])],{(0,1):np.array([[.07]])})
    u = ([np.array([.14]),np.array([-.33])],{(0,1):np.array([[.19]])})
    v = ([np.array([.52]),np.array([.28])],{(0,1):np.array([[-.43]])})

    def evaluate(s,t):
        def at(one,two):
            x,y = one[0][0],one[1][0]
            z = two[0,1][0,0]
            return [np.array([x*x*y*z + s*x*x + s*t*z]),np.array([y*z + t*z*z])], {(0,1):np.array([[x*y]])}
        return at
    x,y,z = .13,-.21,.07
    xu,yu,zu = .14,-.33,.19
    xv,yv,zv = .52,.28,-.43
    h = 2*xu*xv*y*z + 2*x*((xu*yv+xv*yu)*z + (xu*zv+xv*zu)*y) + x*x*(yu*zv+yv*zu)
    exact = np.array([h,yu*zv+yv*zu,xu*yv+xv*yu])
    def flat(blocks):
        one,two = blocks
        return np.concatenate([*one,two[0,1].ravel()])
    np.testing.assert_allclose(flat(ccsd_amplitude_curvature(evaluate(0,0),base,u,v)),exact,rtol=0.,atol=1.e-12)
    h = .01
    nodes = [(i*h,j*h,i*j/(4*h*h),evaluate(i*h,j*h)) for i,j in ((1,1),(1,-1),(-1,1),(-1,-1))]
    total = flat(semi_analytic_ccsd_mixed_source(evaluate(0,0),base,u,v,nodes))
    np.testing.assert_allclose(total,exact + [2*x*xv+z,2*z*zu,0.],rtol=0.,atol=1.e-12)


def projector_check():
    # Degenerate eigenvalues on either side of a separated occupation boundary.
    rng = np.random.default_rng(317)
    k,l = (x-x.T for x in rng.normal(size=(2,5,5)))
    def comm(a,b):
        return a@b-b@a
    def rotate(d):
        du,dv = comm(k,d),comm(l,d)
        return MixedBlock(d,du,dv,comm(k,dv))
    d = rotate(np.diag([.1,.1,.2,.9,.9]))
    exact = rotate(np.diag([0.,0.,0.,1.,1.]))
    np.testing.assert_allclose(_mixed_projector(d,2),exact,atol=2.e-13,rtol=0.)
    np.testing.assert_allclose(_mixed_projector(d,5),rotate(np.eye(5)),atol=2.e-13,rtol=0.)
    with np.testing.assert_raises(ValueError):
        _mixed_projector(d,1)


if __name__ == '__main__':
    polynomial_check()
    projector_check()
    print('Quartic residual and rotating spectral projector checks passed.')
