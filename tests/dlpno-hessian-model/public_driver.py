"""Public analytic driver against the previously validated molecular matrices.

Run each case in separate native scratch. A spy verifies exactly one forward
DLPNO calculation at one geometry; finite-difference execution is forbidden.
"""
import argparse,json,time
from pathlib import Path
from unittest.mock import patch
import numpy as np
import psi4
from psi4.driver.procrouting import proc
from psi4.driver.task_base import AtomicComputer
from psi4.driver.driver_findif import FiniteDifferenceComputer
from screened_molecules import fixture,coverage
from psi4.driver.procrouting.response.dlpno_screened_hessian import COMPONENTS


def run(system,localizer,output):
    psi4.set_memory('4 GiB')
    mol,base=fixture(system,localizer)
    original_xyz=mol.geometry().np.copy()
    record=json.loads(Path(__file__).with_name('screened_molecular_results.json').read_text())
    previous=next(f for f in record['fixtures'] if f['metadata']['system']==system and f['metadata']['localization']==localizer)
    psi4.set_options({'dlpno_hessian_model':True})
    before={(module,key):(psi4.core.get_option(module,key),psi4.core.has_local_option_changed(module,key)) for module,key in
      [('DLPNO',k) for k in ('DLPNO_GRADIENT_MODEL','DLPNO_KEEP_LMP2_RESPONSE','DLPNO_KEEP_TRIPLES_RESPONSE','R_CONVERGENCE','E_CONVERGENCE','DLPNO_MAXITER')]+
      [('SCF',k) for k in ('SAVE_JK','E_CONVERGENCE','D_CONVERGENCE','SOLVER_CONVERGENCE','MAXITER')]+[('FINDIF','FD_PROJECT') ]}
    calls=[];original=proc.run_dlpno
    def forward(name,**kwargs):
        calls.append(kwargs['molecule'].geometry().np.copy())
        return original(name,**kwargs)
    def forbidden(*args,**kwargs):raise AssertionError('Finite-difference computation entered the analytic public route.')
    with patch.object(proc,'run_dlpno',forward),patch.object(FiniteDifferenceComputer,'compute',forbidden):
        plan=psi4.hessian('dlpno-ccsd(t)',molecule=mol,dertype=2,return_plan=True)
        assert isinstance(plan,AtomicComputer) and not calls
        start=time.monotonic();h,w=psi4.hessian('dlpno-ccsd(t)',molecule=mol,dertype=2,return_wfn=True)
    elapsed=time.monotonic()-start
    assert len(calls)==1
    np.testing.assert_array_equal(calls[0],original_xyz)
    np.testing.assert_array_equal(mol.geometry().np,original_xyz)
    for (module,key),state in before.items():
        assert (psi4.core.get_option(module,key),psi4.core.has_local_option_changed(module,key))==state,(module,key)
    np.testing.assert_array_equal(h.np,w.hessian().np)
    for source in (w,psi4.core):
        for label in ('CURRENT HESSIAN','DLPNO-CCSD(T) TOTAL HESSIAN'):np.testing.assert_array_equal(h.np,source.variable(label).np)
        np.testing.assert_array_equal(w.gradient().np,source.variable('CURRENT GRADIENT').np)
        np.testing.assert_allclose(w.energy(),source.variable('CURRENT ENERGY'),rtol=0.,atol=1e-12)
    np.testing.assert_allclose(w.energy(),base.energy(),rtol=0.,atol=2e-11)
    np.testing.assert_allclose(w.gradient().np,previous['metadata']['gradient'],rtol=0.,atol=2e-9)
    actual=np.array([w.variable('DLPNO HESSIAN '+name.upper()).np for name in COMPONENTS])
    np.testing.assert_array_equal(h.np,actual.sum(axis=0))
    error=float(np.max(abs(h.np-previous['hessian'])))
    component_error=float(np.max(abs(actual-previous['component_matrices'])))
    np.testing.assert_allclose(actual,previous['component_matrices'],atol=2e-9,rtol=0.)
    np.testing.assert_allclose(h.np,previous['reference_hessian'],atol=2e-7,rtol=0.)
    current=coverage(w);assert current==previous['metadata']['coverage']
    result=dict(passed=True,system=system,localization=localizer,baseline_commit='833ec80c021747603b6719954dcadcbaf4e6dc90',
      seconds=elapsed,forward_calls=len(calls),finite_difference_calls=0,options_restored=True,coordinates_unchanged=True,
      max_previous_analytic_error=error,max_previous_component_error=component_error,
      max_native_gradient_reference_error=float(np.max(abs(h.np-previous['reference_hessian']))),
      energy=w.energy(),coordinates_bohr=original_xyz.tolist(),gradient=w.gradient().np.tolist(),hessian=h.np.tolist(),
      component_matrices=actual.tolist(),coverage=current,diagnostics={k:v for k,v in w.variables().items() if k.startswith('DLPNO HESSIAN') and isinstance(v,float)})
    Path(output).write_text(json.dumps(result,indent=2)+'\n');print({k:v for k,v in result.items() if k not in ('coverage','hessian','gradient','component_matrices','coordinates_bohr')},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--system',choices=('water_h2','ammonia_h2'),default='water_h2')
    p.add_argument('--localization',choices=('BOYS','PIPEK_MEZEY'),default='BOYS');p.add_argument('--output',default='public_driver_results.json')
    args=p.parse_args();psi4.set_output_file('public_driver.out',False);run(args.system,args.localization,args.output)
