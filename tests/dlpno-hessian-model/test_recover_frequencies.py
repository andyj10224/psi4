"""NumPy-only recovery checks using an exactly soluble diatomic spring."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('recover_frequencies',Path(__file__).with_name('recover_frequencies.py'))
recovery=importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)
analyze,read_hessian,recover,WAVENUMBER_FACTOR=(recovery.analyze,recovery.read_hessian,
                                             recovery.recover,recovery.WAVENUMBER_FACTOR)


@pytest.mark.parametrize('direction',[(1.,0.,0.),(2.,-3.,5.)])
@pytest.mark.parametrize('sign',[1.,-1.])
def test_diatomic_reduced_mass_frequency(direction,sign):
    direction=np.array(direction);direction/=np.linalg.norm(direction)
    xyz=np.array([np.zeros(3),1.4*direction])+[2.,-1.,3.]
    masses=np.array([2.,7.]);force_constant=sign*.3
    hessian=force_constant*np.kron([[1.,-1.],[-1.,1.]],np.outer(direction,direction))
    result=analyze(hessian,xyz,masses)
    expected=sign*np.sqrt(abs(force_constant)*(1/masses[0]+1/masses[1]))*WAVENUMBER_FACTOR
    assert result['rigid_motion_rank']==5
    assert result['imaginary']==[sign<0]
    np.testing.assert_allclose(result['frequencies_cm_inverse'],[expected],atol=1e-9,rtol=0.)


def test_last_hessian_and_geometry_with_isotope_masses():
    hessian=.3*np.kron([[1.,-1.],[-1.,1.]],np.diag([1.,0.,0.]))
    log='Return hessian(): -1.\n[[0 0 0] [0 0 0] [0 0 0]]\n'
    log+='Compute hessian(): method=hf, nre=0.7\nReturn hessian(): -2.\n'+np.array2string(hessian)
    output='Geometry (in Angstrom)\n H 0 0 0 1\n\n'
    output+='Geometry (in Angstrom)\n Atom X Y Z Mass\n H 0 0 0 2\n H 1.4 0 0 3\n\nNuclear repulsion = 0.7\n'
    result=recover(log,output)
    assert result['energy']==-2. and result['masses_amu']==[2.,3.]
    np.testing.assert_allclose(result['hessian'],hessian,atol=0.,rtol=0.)
    with pytest.raises(ValueError,match='nuclear repulsion'):
        recover(log,output.replace('repulsion = 0.7','repulsion = 0.8'))


@pytest.mark.parametrize('record',[
    'No completed calculation',
    'Return hessian(): -1.\n[[1. ... 2.]]',
    'Return hessian(): -1.\n[[1. 0.] [0. 1.]]',
    'Return hessian(): -1.\n[[nan 0 0] [0 0 0] [0 0 0]]',
])
def test_incomplete_or_invalid_hessian_rejected(record):
    with pytest.raises(ValueError):read_hessian(record)
