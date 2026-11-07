"""Reproduce the frozen-core cc-pVTZ water failure with a chosen thread count.

Run in separate working/scratch directories. The optimized geometry comes
from the reported failing run. An optional gradient-difference column is an
independent oracle; it never supplies inputs to the analytic Hessian.
"""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import psi4

from psi4.driver.procrouting.response import dlpno_hessian_inputs
from psi4.driver.procrouting.response.dlpno_screened_hessian import COMPONENTS


def run(args):
    psi4.set_num_threads(args.threads)
    psi4.set_memory('28 GB')
    mol=psi4.geometry('''0 1
O 0.0  0.000000000000 -0.066416635362
H 0.0 -0.753890136879  0.527040100199
H 0.0  0.753890136879  0.527040100199
units angstrom
symmetry c1
no_com
no_reorient
''')
    psi4.set_options(dict(basis='cc-pvtz',freeze_core=True,scf_type='df',pno_convergence='tight',
        dlpno_gradient_model=True,dlpno_hessian_model=True,e_convergence=1.e-10,r_convergence=1.e-10,
        dlpno_ao_ints_tol=0.,f_cut=0.,f_cut_t=0.,t_cut_do_uv=0.))
    # Deliberately leave the global SCF solver settings at their defaults:
    # the public driver's stricter SCF-local options must reach the native API.
    with patch.object(dlpno_hessian_inputs,'_occupied_density_tangent',
                      wraps=dlpno_hessian_inputs._occupied_density_tangent) as densities:
        h,w=psi4.hessian('dlpno-ccsd(t)',molecule=mol,dertype=2,return_wfn=True)
    coordinates=[call.args[1] for call in densities.call_args_list]
    assert sorted(coordinates)==list(range(9)),coordinates
    parts=np.array([w.variable('DLPNO HESSIAN '+name.upper()).np for name in COMPONENTS])
    result=dict(passed=False,threads=args.threads,version=psi4.__version__,density_response_calls=coordinates,
        energy=w.energy(),coordinates_bohr=mol.geometry().np.tolist(),gradient=w.gradient().np.tolist(),
        hessian=h.np.tolist(),components=parts.tolist(),component_order=COMPONENTS)
    path=Path(args.output)
    def save():path.write_text(json.dumps(result,indent=2)+'\n')
    save()
    if args.compare:
        previous=json.loads(Path(args.compare).read_text())
        np.testing.assert_array_equal(previous['component_order'],COMPONENTS)
        np.testing.assert_allclose(previous['coordinates_bohr'],mol.geometry().np,atol=1e-12,rtol=0.)
        result['thread_hessian_difference']=float(np.max(abs(h.np-np.asarray(previous['hessian']))))
        result['thread_component_difference']=float(np.max(abs(parts-np.asarray(previous['components']))))
        save()
        np.testing.assert_allclose(h.np,previous['hessian'],atol=2e-7,rtol=0.)
        np.testing.assert_allclose(parts,previous['components'],atol=2e-7,rtol=0.)
    if args.fd_column is not None:
        from screened_first_pno import align_snapshot
        # Match the analytic driver's forward convergence in the independent
        # displaced calculations. Do not alter PNO or domain cutoffs.
        psi4.set_options(dict(e_convergence=1.e-14,d_convergence=1.e-14,r_convergence=1.e-11,
            solver_convergence=1.e-14,solver_maxiter=200,ints_tolerance=1.e-14,
            local_convergence=1.e-14,local_gradient_convergence=1.e-13,
            dlpno_keep_lmp2_response=True,dlpno_keep_triples_response=True,dlpno_maxiter=600))
        xyz=mol.geometry().np.copy();central=[]
        for step in (.004,.002):
            gradients=[]
            for sign in (1,-1):
                displaced=mol.clone();coords=xyz.copy();coords.flat[args.fd_column]+=sign*step
                displaced.set_geometry(psi4.core.Matrix.from_array(coords));displaced.update_geometry()
                g,other=psi4.gradient('dlpno-ccsd(t)',molecule=displaced,return_wfn=True)
                align_snapshot(other,w)
                gradients.append(g.np.ravel().copy())
            central.append((gradients[0]-gradients[1])/(2*step))
        oracle=(4*central[1]-central[0])/3
        result.update(fd_column=args.fd_column,gradient_difference_column=oracle.tolist(),
                      gradient_difference_error=float(np.max(abs(h.np[:,args.fd_column]-oracle))))
        save()
        np.testing.assert_allclose(h.np[:,args.fd_column],oracle,atol=2e-7,rtol=0.)
    result['passed']=True;save()
    print('Water response checks passed:',args.output,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads',type=int,default=12)
    parser.add_argument('--output',default='water_repeatability.json')
    parser.add_argument('--compare',help='JSON from a separate run at a different thread count')
    parser.add_argument('--fd-column',type=int,choices=range(9))
    args=parser.parse_args()
    if args.threads<1:parser.error('--threads must be positive')
    psi4.set_output_file(str(Path(args.output).with_suffix('.out')),False)
    run(args)
