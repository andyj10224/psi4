"""Complete unsymmetrized screened Hessians, with restartable column records.

Every ordered entry is evaluated independently. No mirrored entries,
symmetrization or rigid-motion projection is used. Displaced native analytic
gradients are an independent reference, never an input to the analytic matrix.
Run analytic/reference stages with disjoint --columns in separate scratch dirs;
verify requires complete records for every column with matching fingerprints.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import psi4
from screened_first_pno import fixture,align_snapshot
from screened_corrections import mixed_inputs
from psi4.driver.procrouting.response.dlpno_orbital import assemble_cartesian_gradient
from psi4.driver.procrouting.response import dlpno_screened_hessian as assembly


def write(path,data):
    partial=path.with_suffix('.partial.json')
    partial.write_text(json.dumps(data,indent=2)+'\n')
    partial.replace(path)


def fingerprint():
    root=Path(assembly.__file__).parent
    paths=[Path(__file__),Path(__file__).with_name('screened_first_pno.py'),
           Path(__file__).with_name('screened_corrections.py'),
           Path(__file__).with_name('rhf_mixed.py'),Path(__file__).with_name('localization_mixed.py')]
    paths+=sorted(root.glob('dlpno_*.py'))
    h=hashlib.sha256()
    for p in paths:h.update(p.name.encode()+b'\0'+p.read_bytes())
    return h.hexdigest()


def configure(args):
    mol,w=fixture(args.system,args.localization,False,final_cutoff=1.e-3,tight=True,screened_triplets=True)
    psi4.set_options({'dlpno__debug':0})
    xyz=mol.geometry().np.copy()
    gradient=assemble_cartesian_gradient(w)
    rhf=psi4.hessian('scf',molecule=mol.clone(),dertype=2).np.copy()
    keys=('DLPNO RETAINED CCSD CORRELATION ENERGY','(T) CORRECTION ENERGY',
          'DLPNO INITIAL PNO TRUNCATION ENERGY','DLPNO FINAL PNO TRUNCATION ENERGY',
          'DLPNO SC-LMP2 PAIR ENERGY','DLPNO DIPOLE ENERGY','DLPNO LMP2 WEAK PAIR ENERGY')
    meta=dict(system=args.system,localization=args.localization,units='hartree/bohr^2',
              fingerprint=fingerprint(),size=xyz.size,coordinates_bohr=xyz.tolist(),
              component_order=assembly.COMPONENTS,gradient=gradient.tolist(),rhf_hessian=rhf.tolist(),
              energy=w.energy(),rhf_energy=w.reference_wavefunction().energy(),
              correlation_energies={key:w.variable(key) for key in keys},
              native_domains=psi4.core.dlpno_response_domain_state(w),
              retained_triplets=len(psi4.core.dlpno_triples_response_state(w)))
    return mol,w,meta


def matching(left,right):
    for key in ('system','localization','fingerprint','size','component_order','native_domains','retained_triplets'):
        # JSON roundtrip normalizes tuples in records and freshly built metadata.
        if json.dumps(left[key],sort_keys=True)!=json.dumps(right[key],sort_keys=True):
            raise ValueError(f'Incompatible column metadata: {key}.')
    for key in ('coordinates_bohr','gradient','rhf_hessian','energy','rhf_energy'):
        np.testing.assert_allclose(left[key],right[key],atol=2.e-11,rtol=0.,err_msg=key)
    np.testing.assert_allclose(list(left['correlation_energies'].values()),
                               list(right['correlation_energies'].values()),atol=2.e-11,rtol=0.)


def analytic(args,mol,w,meta):
    size=meta['size'];gradient=np.array(meta['gradient']).ravel();rhf=np.array(meta['rhf_hessian'])
    for v in (range(size) if args.columns is None else args.columns):
        path=args.output_dir/f'analytic-{v:03d}.json'
        record=dict(metadata=meta,column=v,entries=[],complete=False)
        if args.resume and path.exists():
            old=json.loads(path.read_text());matching(old['metadata'],meta)
            if old['column']!=v:raise ValueError('Wrong checkpoint column.')
            record=old
            if record['complete']:
                print('reused complete analytic column',v,flush=True);continue
        provider=assembly.ScreenedHessianColumn(w,v,lambda u,c:mixed_inputs(w,u,c),rhf[:,v])
        completed={e['row'] for e in record['entries']}
        start=time.monotonic()
        for u in range(size):
            if u in completed:continue
            tick=time.monotonic();entry=provider.element(u)
            parts=entry.pop('components');first=entry.pop('gradient_components')
            error=float(first.sum()-gradient[u])
            record['entries'].append(dict(row=u,components=parts.tolist(),gradient_components=first.tolist(),
                total=float(parts.sum()),first_replay_error=error,seconds=time.monotonic()-tick,**entry))
            write(path,record)
            np.testing.assert_allclose(error,0.,atol=2.e-8,rtol=0.)
            print('analytic entry',u,v,'total',parts.sum(),'first_error',error,flush=True)
            gc.collect()
        if sorted(e['row'] for e in record['entries'])!=list(range(size)):
            raise ValueError('Missing or duplicated analytic rows.')
        record['entries'].sort(key=lambda e:e['row'])
        record.update(complete=True,latest_elapsed_seconds=time.monotonic()-start)
        write(path,record);print('complete analytic column',v,flush=True)
        del provider;gc.collect()


def reference(args,mol,w,meta):
    xyz=np.array(meta['coordinates_bohr']);size=xyz.size
    for v in (range(size) if args.columns is None else args.columns):
        path=args.output_dir/f'reference-{v:03d}.json'
        if args.resume and path.exists():
            old=json.loads(path.read_text());matching(old['metadata'],meta)
            if old['column']==v and old['step_bohr']==args.step and old['complete']:
                print('reused complete reference column',v,flush=True);continue
        record=dict(metadata=meta,column=v,step_bohr=args.step,samples=[],complete=False)
        central=[]
        for h in (args.step,args.step/2):
            gradients=[]
            for sign in (1,-1):
                coords=xyz.copy();coords.flat[v]+=sign*h
                m=mol.clone();m.set_geometry(psi4.core.Matrix.from_array(coords));m.update_geometry()
                _,other=psi4.energy('dlpno-ccsd(t)',molecule=m,return_wfn=True)
                align_snapshot(other,w)
                g=assemble_cartesian_gradient(other)
                gradients.append(g.ravel())
                record['samples'].append(dict(displacement_bohr=sign*h,energy=other.energy(),gradient=g.tolist()))
                write(path,record);print('reference gradient',v,sign*h,flush=True)
            central.append((gradients[0]-gradients[1])/(2*h))
        record.update(complete=True,central_columns=[x.tolist() for x in central],
                      richardson_column=((4*central[1]-central[0])/3).tolist())
        write(path,record);print('complete reference column',v,flush=True)
        gc.collect()


def identities(H,g,xyz):
    translations=np.tile(np.eye(3),(len(xyz),1))
    rotations=np.stack([np.cross(np.broadcast_to(a,xyz.shape),xyz).ravel() for a in np.eye(3)],axis=1)
    moved_gradient=np.stack([np.cross(np.broadcast_to(a,g.shape),g).ravel() for a in np.eye(3)],axis=1)
    return dict(symmetry=float(np.max(abs(H-H.T))),
        right_translation=float(np.max(abs(H@translations))),
        left_translation=float(np.max(abs(translations.T@H))),
        rotation_covariance=float(np.max(abs(H@rotations-moved_gradient))),
        net_force=float(np.max(abs(g.sum(axis=0)))),torque=float(np.max(abs(np.cross(xyz,g).sum(axis=0)))))


def verify(args):
    first=json.loads((args.output_dir/'analytic-000.json').read_text());meta=first['metadata'];size=meta['size']
    if meta['system']!=args.system or meta['localization']!=args.localization or meta['fingerprint']!=fingerprint():
        raise ValueError('Requested matrix and saved implementation/fixture disagree.')
    columns=[];references=[]
    for v in range(size):
        a=json.loads((args.output_dir/f'analytic-{v:03d}.json').read_text())
        r=json.loads((args.output_dir/f'reference-{v:03d}.json').read_text())
        for record in (a,r):
            matching(record['metadata'],meta)
            if record['column']!=v or not record['complete']:raise ValueError('Incomplete column.')
        if r['step_bohr']!=args.step:raise ValueError('Inconsistent reference spacing.')
        if [e['row'] for e in a['entries']]!=list(range(size)):raise ValueError('Wrong ordered rows.')
        columns.append(a);references.append(r)
    parts=np.array([[e['components'] for e in col['entries']] for col in columns]).transpose(2,1,0)
    firsts=np.array([[e['gradient_components'] for e in col['entries']] for col in columns]).transpose(2,1,0)
    H=parts.sum(axis=0);target=np.array([r['richardson_column'] for r in references]).T
    if not np.isfinite(parts).all() or not np.isfinite(target).all():raise ValueError('Nonfinite Hessian.')
    xyz=np.array(meta['coordinates_bohr']);gradient=np.array(meta['gradient'])
    errors=identities(H,gradient,xyz)
    errors.update(gradient_difference=float(np.max(abs(H-target))),reference_symmetry=float(np.max(abs(target-target.T))),
        first_replay=float(np.max(abs(firsts.sum(axis=0)-gradient.ravel()[:,None]))),
        first_component_column_spread=float(np.max(abs(firsts-firsts[:,:,:1]))))
    component_errors={name:identities(part,firsts[k,:,0].reshape(xyz.shape),xyz)
                      for k,(name,part) in enumerate(zip(assembly.COMPONENTS,parts))}
    result=dict(metadata=meta,passed=False,ordered_entries=size*size,hessian=H.tolist(),
        component_matrices=parts.tolist(),component_gradients=firsts[:,:,0].tolist(),
        reference_hessian=target.tolist(),difference=(H-target).tolist(),errors=errors,
        component_identities=component_errors,reference_columns=[{k:v for k,v in r.items() if k!='metadata'} for r in references],
        solver_checks=dict(max_cc_residual=max(max(e['cc_residual_errors']) for c in columns for e in c['entries']),
                           max_pno_residual=max(max(e['pno_residual_errors']) for c in columns for e in c['entries']),
                           max_triples_right=max(e['triples_right_residual'] for c in columns for e in c['entries']),
                           max_triples_left=max(e['triples_left_residual'] for c in columns for e in c['entries'])))
    path=args.output_dir/'matrix_results.json';write(path,result)
    print('matrix verification',errors,flush=True)
    np.testing.assert_allclose(H,target,atol=2.e-7,rtol=0.)
    for name in ('symmetry','right_translation','left_translation','rotation_covariance'):
        if errors[name]>2.e-7:raise AssertionError((name,errors[name]))
        for component,data in component_errors.items():
            if data[name]>2.e-7:raise AssertionError((component,name,data[name]))
    for name in ('net_force','torque','first_replay','first_component_column_spread'):
        if errors[name]>2.e-8:raise AssertionError((name,errors[name]))
    result['passed']=True;write(path,result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--system',choices=('chain','lih'),default='chain')
    p.add_argument('--localization',choices=('BOYS','PIPEK_MEZEY'),default='BOYS')
    p.add_argument('--stage',choices=('analytic','reference','verify','all'),default='all')
    p.add_argument('--columns',type=int,nargs='+')
    p.add_argument('--step',type=float,default=.004)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--output-dir',type=Path,default=Path('screened_matrix'))
    args=p.parse_args()
    if args.step<=0:p.error('--step must be positive.')
    size=24 if args.system=='chain' else 18
    if args.columns is not None and (len(set(args.columns))!=len(args.columns) or any(not 0<=x<size for x in args.columns)):
        p.error('Invalid or duplicated Cartesian columns.')
    if args.stage=='all' and args.columns is not None:p.error('Use a separate stage for column shards.')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.stage=='verify':verify(args)
    else:
        psi4.set_output_file('screened_matrix.out',False)
        mol,w,meta=configure(args)
        if args.stage in ('analytic','all'):analytic(args,mol,w,meta)
        if args.stage in ('reference','all'):reference(args,mol,w,meta)
        if args.stage=='all':verify(args)
