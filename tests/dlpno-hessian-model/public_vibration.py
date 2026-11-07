"""Equilibrium H2 smoke test of the installed analytic frequency workflow.

The bond is located using native analytic gradients. Numerical differences
are an independent validation oracle, never an input to the Hessian.
H2 has no triples: screened triples/core coverage is in public_driver.py.
"""
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scipy.optimize import brentq
import psi4
from psi4.driver.procrouting import proc
from psi4.driver.procrouting.response.dlpno_screened_hessian import COMPONENTS
from psi4.driver.driver_findif import FiniteDifferenceComputer
from psi4.driver import constants


def molecule(bond):
    # Exercise the ordinary non-C1 input route; the derivative reference is C1.
    return psi4.geometry(f'0 1\nH 0 0 {-bond/2}\nH 0 0 {bond/2}\nunits bohr\nno_com\nno_reorient')


def run():
    psi4.set_memory('1 GiB');psi4.set_num_threads(1)
    psi4.set_options(dict(basis='cc-pvdz',reference='rhf',scf_type='df',freeze_core=False,
        df_basis_scf='cc-pvdz-jkfit',df_basis_cc='cc-pvdz-ri',
        scf_initial_accelerator='NONE',diis=False,e_convergence=1.e-14,d_convergence=1.e-14,
        r_convergence=1.e-11,solver_convergence=1.e-14,dlpno_hessian_model=True,
        t_cut_pno=0.,t_cut_pno_mp2=0.,t_cut_tno=0.,dlpno_ao_ints_tol=0.,f_cut=0.,f_cut_t=0.,
        t_cut_do_uv=0.,t_cut_pre=0.,t_cut_do=0.,t_cut_do_pre=0.,t_cut_do_ij=0.,t_cut_mkn=0.,
        t_cut_clmo=0.,t_cut_cpao=0.))
    samples=[]
    def radial_gradient(bond):
        gradient=psi4.gradient('dlpno-ccsd(t)',molecule=molecule(bond)).np
        samples.append(dict(bond_bohr=bond,gradient=gradient.tolist()))
        return gradient[1,2]
    bond=brentq(radial_gradient,1.3,1.6,xtol=1.e-10)
    mol=molecule(bond);point_group=mol.schoenflies_symbol();xyz=mol.geometry().np.copy()
    calls=[];forward=proc.run_dlpno
    def spy(name,**kwargs):
        calls.append(kwargs['molecule'].geometry().np.copy());return forward(name,**kwargs)
    def forbidden(*args,**kwargs):raise AssertionError('Numerical Hessian route entered frequency().')
    with patch.object(proc,'run_dlpno',spy),patch.object(FiniteDifferenceComputer,'compute',forbidden):
        energy,w=psi4.frequency('dlpno-ccsd(t)',molecule=mol,dertype=2,return_wfn=True)
    assert len(calls)==1
    # Geometry updates/C1 copying can round a coordinate by one ulp.
    np.testing.assert_allclose(calls[0],xyz,rtol=0.,atol=1.e-14)
    np.testing.assert_allclose(mol.geometry().np,xyz,rtol=0.,atol=1.e-14)
    assert point_group!='c1' and w.reference_wavefunction().nirrep()==1
    h=w.hessian().np.copy();g=w.gradient().np.copy()
    assert np.max(abs(g))<1.e-8
    parts=np.array([w.variable('DLPNO HESSIAN '+c.upper()).np for c in COMPONENTS])
    np.testing.assert_array_equal(h,parts.sum(axis=0))
    np.testing.assert_array_equal(h,psi4.core.variable('CURRENT HESSIAN').np)
    frequencies=w.frequency_analysis['omega'].data
    labels=w.frequency_analysis['TRV'].data
    active=np.array([v for v,label in zip(frequencies,labels) if label=='V'])
    assert len(active)==1 and abs(active[0].imag)<1.e-8 and active[0].real>0
    references=[]
    for step in (.004,.002):
        gradients=[]
        for sign in (1,-1):
            displaced=molecule(bond);displaced.reset_point_group('c1')
            coords=xyz.copy();coords[0,2]+=sign*step
            displaced.set_geometry(psi4.core.Matrix.from_array(coords));displaced.update_geometry()
            gradients.append(psi4.gradient('dlpno-ccsd(t)',molecule=displaced).np.ravel())
        references.append(dict(step_bohr=step,gradients=[a.tolist() for a in gradients],
                               column=((gradients[0]-gradients[1])/(2*step)).tolist()))
    reference=(4*np.array(references[1]['column'])-references[0]['column'])/3
    error=float(np.max(abs(h[:,2]-reference)))
    assert error<2.e-7
    mass=mol.mass(0)*mol.mass(1)/(mol.mass(0)+mol.mass(1))
    conversion=np.sqrt(constants.na*constants.hartree2J*1.e19)/(2*np.pi*constants.c*constants.bohr2angstroms)
    reference_frequency=float(np.sqrt(reference[2]/mass)*conversion)
    frequency_error=abs(active[0].real-reference_frequency)
    assert frequency_error<.002
    np.testing.assert_array_equal(h,w.hessian().np)
    result=dict(passed=True,system='H2/cc-pVDZ',input_point_group=point_group,
        bond_bohr=bond,energy=energy,gradient=g.tolist(),hessian=h.tolist(),
        max_gradient=float(np.max(abs(g))),forward_calls=len(calls),finite_difference_hessian_calls=0,
        frequency_cm_inverse=float(active[0].real),reference_frequency_cm_inverse=reference_frequency,
        frequency_error_cm_inverse=float(frequency_error),max_native_gradient_reference_error=error,
        mode_labels=list(labels),root_samples=samples,reference_samples=references,
        scope='Equilibrium driver/vibrational smoke test; H2 has no triples and is not screened molecular vibrational validation.')
    Path('public_vibration_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print({k:v for k,v in result.items() if k not in ('gradient','hessian','root_samples','reference_samples')},flush=True)


if __name__=='__main__':
    psi4.set_output_file('public_vibration.out',False);run()
