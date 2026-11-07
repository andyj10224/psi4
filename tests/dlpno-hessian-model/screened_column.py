"""Complete screened analytic Hessian columns versus native analytic gradients.

No numerical inputs enter the assembled column. The independent oracle uses
reconverged native wavefunctions and the existing analytic Cartesian gradient.
All rows are reported unsymmetrized, without enforcing rigid-motion identities.
"""
import argparse
import gc
import json
from pathlib import Path
import time
import numpy as np
import psi4
from screened_first_pno import fixture,align_snapshot
from screened_molecules import fixture as molecular_fixture,coverage as molecular_coverage,SYSTEM_SIZES
from screened_corrections import mixed_inputs
from screened_ccsd_mixed import first
from lmp2_tangent import blocks
from psi4.driver.procrouting.response.dlpno_orbital import assemble_cartesian_gradient
from psi4.driver.procrouting.response.dlpno_screened_hessian import ScreenedHessianColumn,COMPONENTS,retained_inputs,triples_first_energy


def independent_components(w,u):
    """Relaxed CCSD and triples first energies from the older first kernels."""
    from psi4.driver.procrouting.response.dlpno_triples_curvature import MixedBlock
    from psi4.driver.procrouting.response.dlpno_screened_ccsd import ScreenedFitting
    from psi4.driver.procrouting.response.dlpno_screened_triples import ScreenedTriplesInputs
    from psi4.driver.procrouting.response.dlpno_ccsd_curvature import project_retained_ccsd_energy
    (dt1,dt2),frames,_=first(w,u)
    S,F,L,D,_,df=mixed_inputs(w,u,u)
    one=[];two=blocks(w,'CCSD_DOUBLES')
    for i in range(L.value.shape[1]):
        dom,ts,_=psi4.core.dlpno_ccsd_df_pair_state(w,i,i)
        one.append(ts[dom.index(i)].np.ravel().copy())
    def block(x,dx):return MixedBlock(x,dx,np.zeros_like(x),np.zeros_like(x))
    def only_first(x):return block(x.value,x.u)
    def electronic(key):
        if key=='lmo':return only_first(L)
        if key=='fock':return only_first(F)
        kind,label=key
        if kind=='pno':return block(frames.frames[label],frames.tangents[label])
        if kind=='t1':return block(one[label],dt1[label])
        if kind=='t2':return block(two[label],dt2[label])
        raise KeyError(key)
    fitting=ScreenedFitting(w,electronic,only_first(S),lambda p:only_first(D(only_first(p))),
                            lambda aux:tuple(only_first(x) for x in df(aux)))
    response=dict(fitting=fitting,electronic=electronic,base=(one,two))
    cc=project_retained_ccsd_energy(*retained_inputs(response)).u
    triples=ScreenedTriplesInputs(w,fitting)
    return np.array([cc,triples_first_energy(triples.first_direction(1))])


def run(args):
    mol,w=(molecular_fixture(args.system,args.localization) if args.system in SYSTEM_SIZES else
           fixture(args.system,args.localization,False,final_cutoff=1.e-3,tight=True,screened_triplets=True))
    psi4.set_options({'dlpno__debug':0})
    xyz=mol.geometry().np.copy();size=xyz.size
    column=(2 if args.system=='chain' else 5) if args.column is None else args.column
    rows=list(range(size)) if args.rows is None else args.rows
    if not 0<=column<size or any(not 0<=u<size for u in rows):raise ValueError('Invalid Cartesian coordinate.')
    gradient=assemble_cartesian_gradient(w)
    rhf=psi4.hessian('scf',molecule=mol.clone(),dertype=2).np.copy()
    result=dict(system=args.system,localization=args.localization,column=column,rows=rows,
                units='hartree/bohr^2',coordinates_bohr=xyz.tolist(),component_order=COMPONENTS,
                gradient=gradient.tolist(),step_bohr=args.step,entries=[],passed=False,
                coverage=molecular_coverage(w) if args.system in SYSTEM_SIZES else None)
    path=Path(args.output)
    def save():path.write_text(json.dumps(result,indent=2)+'\n')
    start=time.monotonic()
    provider=ScreenedHessianColumn(w,column,lambda u,v:mixed_inputs(w,u,v),rhf[:,column])
    for u in rows:
        tick=time.monotonic();entry=provider.element(u)
        parts=entry.pop('components');first=entry.pop('gradient_components')
        err=float(first.sum()-gradient.ravel()[u])
        record=dict(row=u,components=parts.tolist(),total=float(parts.sum()),
                    gradient_components=first.tolist(),first_replay_error=err,
                    seconds=time.monotonic()-tick,**entry)
        result['entries'].append(record);save()
        print('analytic row',u,'parts',parts,'first error',err,'seconds',record['seconds'],flush=True)
        np.testing.assert_allclose(err,0.,atol=2.e-8,rtol=0.)
        gc.collect()
    result['analytic_seconds']=time.monotonic()-start
    actual=np.array([e['total'] for e in result['entries']])
    if rows==list(range(size)):
        cartesian=actual.reshape(xyz.shape)
        axis=np.eye(3)[column%3]
        translation=cartesian.sum(axis=0)
        rotation=np.cross(xyz,cartesian).sum(axis=0)+np.cross(axis,gradient[column//3])
        result['column_identities']=dict(translation=translation.tolist(),rotation_covariance=rotation.tolist(),
            net_force=gradient.sum(axis=0).tolist(),torque=np.cross(xyz,gradient).sum(axis=0).tolist())
        save();print('column identities',result['column_identities'],flush=True)
        np.testing.assert_allclose(translation,0.,atol=2.e-7,rtol=0.)
        np.testing.assert_allclose(rotation,0.,atol=2.e-7,rtol=0.)
        np.testing.assert_allclose(gradient.sum(axis=0),0.,atol=2.e-8,rtol=0.)
        np.testing.assert_allclose(np.cross(xyz,gradient).sum(axis=0),0.,atol=2.e-8,rtol=0.)
    # Values are already saved before any displaced calculation is requested.
    del provider;gc.collect()
    if args.reciprocity:
        result['ordered_reciprocity']=[]
        for u in [x for x in (0,5,18) if x in rows and x!=column]:
            reverse=ScreenedHessianColumn(w,u,lambda a,b:mixed_inputs(w,a,b),rhf[:,u]).element(column)
            original=np.asarray(result['entries'][rows.index(u)]['components'])
            errors=original-reverse['components']
            result['ordered_reciprocity'].append(dict(row=u,reverse_components=reverse['components'].tolist(),errors=errors.tolist()))
            save();print('ordered reciprocity',u,errors,flush=True)
            np.testing.assert_allclose(errors,0.,atol=2.e-9,rtol=0.)
            gc.collect()
    component_rows=[u for u in (column,18 if args.system=='chain' else 12) if u in rows] if args.component_checks else []
    if args.component_rows is not None:
        if not args.component_checks or len(set(args.component_rows))!=len(args.component_rows) or any(u not in rows for u in args.component_rows):
            raise ValueError('Component rows must be unique assembled rows with --component-checks.')
        component_rows=args.component_rows
    estimates=[];samples=[];component_estimates=[]
    for h in (args.step,args.step/2):
        gradients=[];components=[]
        for sign in (1,-1):
            m=mol.clone();coords=xyz.copy();coords.flat[column]+=sign*h
            m.set_geometry(psi4.core.Matrix.from_array(coords));m.update_geometry()
            _,other=psi4.energy('dlpno-ccsd(t)',molecule=m,return_wfn=True)
            align_snapshot(other,w)
            g=assemble_cartesian_gradient(other)
            gradients.append(g.ravel()[rows])
            comp=np.array([independent_components(other,u) for u in component_rows])
            components.append(comp)
            samples.append(dict(displacement_bohr=sign*h,energy=other.energy(),gradient=g.tolist(),
                                component_gradient_rows=component_rows,independent_CCSD_T_gradients=comp.tolist()))
            print('native gradient oracle',sign*h,flush=True)
        estimates.append((gradients[0]-gradients[1])/(2*h))
        component_estimates.append((components[0]-components[1])/(2*h))
    target=(4*estimates[1]-estimates[0])/3;errors=actual-target
    result.update(gradient_samples=samples,central_columns=[x.tolist() for x in estimates],
                  reference_column=target.tolist(),errors=errors.tolist(),
                  max_error=float(np.max(abs(errors))),max_first_replay_error=max(abs(e['first_replay_error']) for e in result['entries']))
    save();print('total column error',result['max_error'],errors,flush=True)
    np.testing.assert_allclose(actual,target,atol=2.e-7,rtol=0.)
    if component_rows:
        ref=(4*component_estimates[1]-component_estimates[0])/3
        assembled=np.array([result['entries'][rows.index(u)]['components'][1:3] for u in component_rows])
        result['independent_components']=dict(rows=component_rows,order=['retained_ccsd','iterative_triples'],
            reference=ref.tolist(),errors=(assembled-ref).tolist())
        save();print('independent CCSD/T curvature errors',assembled-ref,flush=True)
        np.testing.assert_allclose(assembled[:,0],ref[:,0],atol=2.e-8,rtol=0.)
        np.testing.assert_allclose(assembled[:,1],ref[:,1],atol=2.e-9,rtol=0.)
    result['passed']=True;save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--system',choices=('chain','lih',*SYSTEM_SIZES),default='chain')
    p.add_argument('--localization',choices=('BOYS','PIPEK_MEZEY'),default='BOYS')
    p.add_argument('--column',type=int)
    p.add_argument('--reciprocity',action='store_true')
    p.add_argument('--component-checks',action='store_true')
    p.add_argument('--rows',type=int,nargs='+')
    p.add_argument('--component-rows',type=int,nargs='+')
    p.add_argument('--step',type=float,default=.004)
    p.add_argument('--output',default='screened_column_results.json')
    args=p.parse_args()
    if args.step<=0:p.error('--step must be positive.')
    psi4.set_output_file('screened_column.out',False);run(args)
