"""Diagnose the small rotation defect from individual PAO-pair screening.

This is a separate control, not input to screened_matrix.py. Native gradients
are rigidly rotated back before differencing, with the same fixed-branch
checks. The zero-UV-cutoff control also evaluates an analytic torque-column
identity for an affected coordinate. No matrix symmetrization is performed.
"""
import json
from pathlib import Path
import numpy as np
import psi4
from screened_first_pno import fixture,align_snapshot
from psi4.driver.procrouting.response.dlpno_orbital import assemble_cartesian_gradient
from psi4.driver.procrouting.response.dlpno_screened_hessian import ScreenedHessianColumn
from screened_corrections import mixed_inputs
psi4.set_output_file('rotation_probe.out',False)
records=[]
for cutoff in (1.e-5,0.):
 mol,w=fixture('chain','BOYS',False,final_cutoff=1.e-3,tight=True,screened_triplets=True)
 psi4.set_options({'t_cut_do_uv':cutoff,'dlpno__debug':0})
 _,w=psi4.energy('dlpno-ccsd(t)',molecule=mol,return_wfn=True)
 xyz=mol.geometry().np.copy();g0=assemble_cartesian_gradient(w)
 estimates=[];samples=[]
 for h in (.004,.002):
  gs=[]
  for sign in (1,-1):
   t=sign*h;c,s=np.cos(t),np.sin(t);Q=np.array([[1.,0.,0.],[0.,c,-s],[0.,s,c]])
   m=mol.clone();m.set_geometry(psi4.core.Matrix.from_array(xyz@Q.T));m.update_geometry()
   _,other=psi4.energy('dlpno-ccsd(t)',molecule=m,return_wfn=True)
   valid=True;failure=None
   try:align_snapshot(other,w)
   except AssertionError as e:valid=False;failure=str(e)
   g=assemble_cartesian_gradient(other);gs.append(g@Q)
   samples.append(dict(angle=t,energy=other.energy(),native_branch_matches=valid,branch_error=failure,
                       gradient=g.tolist(),back_rotated_gradient=(g@Q).tolist()))
  estimates.append((gs[0]-gs[1])/(2*h))
 result=(4*estimates[1]-estimates[0])/3
 records.append(dict(t_cut_do_uv=cutoff,reference_energy=w.energy(),reference_gradient=g0.tolist(),
                     samples=samples,central=[x.tolist() for x in estimates],
                     covariance_derivative=result.tolist(),max_error=float(np.max(abs(result)))))
 Path('rotation_probe_results.json').write_text(json.dumps(records,indent=2)+'\n')
 print('UV cutoff',cutoff,'rotation defect',np.max(abs(result)),'branch',all(x['native_branch_matches'] for x in samples),flush=True)
 if not all(x['native_branch_matches'] for x in samples):raise AssertionError('Rotation left the native branch.')
 if cutoff==0.:
  # Rotation about x: rho_A=(0,-z_A,y_A); all reference y_A are zero.
  rhf=psi4.hessian('scf',molecule=mol.clone(),dertype=2).np.copy()
  column=ScreenedHessianColumn(w,1,lambda u,v:mixed_inputs(w,u,v),rhf[:,1])
  entries=[];contraction=0.
  for atom in range(len(xyz)):
   u=3*atom+1;entry=column.element(u);total=float(entry['components'].sum())
   contraction-=xyz[atom,2]*total
   entries.append(dict(row=u,column=1,components=entry['components'].tolist(),total=total))
  defect=contraction+g0[0,2]
  records[-1]['analytic_rotation_column']=dict(entries=entries,covariance_defect=defect)
  Path('rotation_probe_results.json').write_text(json.dumps(records,indent=2)+'\n')
  print('zero-cutoff analytic rotation column defect',defect,flush=True)
  np.testing.assert_allclose(defect,0.,atol=2.e-9,rtol=0.)
