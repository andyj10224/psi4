"""Nonzero screened-pair gradient and mixed-curvature checks against native energies.

Run --system chain (all electron) or --system lih (three frozen Li 1s orbitals).
Every displaced point is reconverged and checked for pair/domain/rank continuity.
No complete screened CCSD(T) Hessian is claimed by this correction-only test.
"""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import numpy as np
import psi4
from scipy.optimize import linear_sum_assignment
from localization_integration import configure
from localization_mixed import molecular_localization
from rhf_mixed import analytic_inputs
from first_pno import raw
from screened_domains import canonical_branch
from psi4.driver.procrouting.response.dlpno_orbital import assemble_cartesian_gradient
from psi4.driver.procrouting.response.dlpno_screened_curvature import screened_pair_curvature
from psi4.driver.procrouting.response.dlpno_triples_curvature import MixedBlock, mixed_product


def pair_energies(w):
    return np.array([w.variable('DLPNO SC-LMP2 PAIR ENERGY'), w.variable('DLPNO DIPOLE ENERGY')])


def normalized_paos(ref):
    C, S = ref.Ca_subset('AO','OCC').np, ref.S().np
    N = np.eye(len(S))-C@C.T@S
    q = np.einsum('ma,mn,na->a',N,S,N)
    d = np.zeros_like(q); d[q>1.e-20]=q[q>1.e-20]**(-.5)
    return N*d


def assert_branch(w, ref):
    r0, r = ref.reference_wavefunction(), w.reference_wavefunction()
    L0, L = (x.variable('DLPNO RESPONSE ACTIVE LMOS').np for x in (ref,w))
    cross = psi4.core.MintsHelper(r0).ao_overlap(r0.basisset(),r.basisset()).np
    match = L0.T@cross@L
    _, order = linear_sum_assignment(-abs(match))
    assert min(abs(match[np.arange(len(order)),order]))>.995
    inv = np.argsort(order)
    assert canonical_branch(w,inv)==canonical_branch(ref)
    d0,b0 = psi4.core.dlpno_screened_pair_state(ref)
    d,b = psi4.core.dlpno_screened_pair_state(w)
    for key in ('AO_OCC','AO_VIR'):
        assert d0[key]==d[key]
    N, S = normalized_paos(r), r.S().np
    for key in ('SCMP2_PAIRS','DIPOLE_PAIRS'):
        assert {tuple(sorted(inv[p])) for p in d[key]}==set(map(tuple,d0[key]))
    for i in range(len(order)):
        assert d0['DIPOLE_PAO'][i]==d['DIPOLE_PAO'][order[i]]
    for p, fields in b0.items():
        q = tuple(sorted(order[list(p)]))
        other = b[q]
        for key in ('PAOS','AUX'):
            if key in fields:
                np.testing.assert_array_equal(fields[key].np,other[key].np)
        ids = d0['DIPOLE_PAO'][p[0]] if p[0]==p[1] else fields['PAOS'].np[:,0].astype(int)
        old, new = fields['PARENT'].np,other['PARENT'].np
        assert old.shape==new.shape
        seed, actual = N[:,ids]@old,N[:,ids]@new
        # Different redundant Cholesky pivots may span the SAME physical space.
        # Require equality to transport of the reference pivot span, not just rank.
        expected = seed@np.linalg.solve(seed.T@S@seed,seed.T)
        np.testing.assert_allclose(actual@actual.T,expected,atol=2.e-8,rtol=0.)

    old_basis,new_basis=(psi4.core.dlpno_triples_basis_state(x) for x in (ref,w))
    for r,fields in old_basis.items():
        q=tuple(sorted(order[list(r)]));ids=fields['PAOS'].np[:,0].astype(int)
        seed=N[:,ids]@fields['PARENT'].np
        actual=N[:,ids]@new_basis[q]['PARENT'].np
        expected=seed@np.linalg.solve(seed.T@S@seed,seed.T)
        np.testing.assert_allclose(actual@actual.T,expected,atol=2.e-8,rtol=0.)


def mixed_inputs(w,u,v):
    ref = w.reference_wavefunction()
    data = {}
    _,_,active = analytic_inputs(ref,u,v,partition=True,block_output=data)
    S,F,D = (data[k] for k in ('overlap','fock','density'))
    kind = 'BOYS' if w.variable('DLPNO RESPONSE BOYS') else 'PIPEK_MEZEY'
    L = molecular_localization(ref,w.variable('DLPNO RESPONSE ACTIVE LMOS').np,kind,u,v,active)
    mints=psi4.core.MintsHelper(ref)
    au,xu=divmod(u,3); av,xv=divmod(v,3)
    dip0=mints.ao_dipole(); dipu=mints.ao_elec_dip_deriv1(au); dipv=mints.ao_elec_dip_deriv1(av)
    dip2=mints.ao_elec_dip_deriv2(au,av)
    dip=[MixedBlock(dip0[k].np,dipu[3*k+xu].np,dipv[3*k+xv].np,dip2[9*k+3*xu+xv].np) for k in range(3)]
    J,Q=raw(ref,ref.get_basisset('DF_BASIS_CC'))
    j2,q2=psi4.core.dlpno_df_mixed_integrals(ref,'DF_BASIS_CC',list(range(len(J))),list(range(len(S.value))),u,v)
    jm=MixedBlock(J,mints.ao_metric_deriv1(au,'DF_BASIS_CC')[xu].np.reshape(J.shape),
                    mints.ao_metric_deriv1(av,'DF_BASIS_CC')[xv].np.reshape(J.shape),j2.np.reshape(J.shape))
    qm=MixedBlock(Q,mints.ao_3center_deriv1(au,'DF_BASIS_CC')[xu].np.reshape(Q.shape),
                    mints.ao_3center_deriv1(av,'DF_BASIS_CC')[xv].np.reshape(Q.shape),q2.np.reshape(Q.shape))
    def df(aux):
        ids=list(aux)
        return jm.slice(np.ix_(ids,ids)),qm.slice(ids)
    return S,F,L,lambda panel:mixed_product('mn,na->ma',D,panel),dip,df


def run(system,localization,stage,path):
    configure()
    frozen=system=='lih'
    psi4.set_options(dict(freeze_core=frozen,save_jk=True,t_cut_pairs_mp2=1.e-6,t_cut_pairs=1.e-6,
        t_cut_pre=1.e-6,t_cut_do_ij=1.e-5,t_cut_do=.01,t_cut_do_pre=.03,t_cut_mkn=.001,
        t_cut_clmo=.01,t_cut_cpao=.01,t_cut_do_uv=1.e-5,dlpno__debug=1,dlpno_local_orbitals=localization,
        t_cut_pno_mp2=1.e-5,t_cut_pno=1.e-4,t_cut_tno=1.e-5))
    if frozen:
        psi4.set_options(dict(basis='def2-svp',df_basis_scf='def2-universal-jkfit',df_basis_cc='def2-svp-ri',
                              t_cut_pairs_mp2=1.e-4,t_cut_pairs=1.e-4,t_cut_pre=1.e-5,t_cut_do_ij=1.e-3))
    atoms = ([('H',x,0,z) for x in (0,3.5,7,10.5) for z in (-.37,.37)] if not frozen else
             [(a,x,0,z) for x in (0,6,12) for a,z in (('Li',0),('H',1.6))])
    mol=psi4.geometry('0 1\n'+'\n'.join(f'{a} {x} {y} {z}' for a,x,y,z in atoms)+
                      '\nunits angstrom\nsymmetry c1\nno_com\nno_reorient')
    energy,w=psi4.energy('dlpno-ccsd(t)',molecule=mol,return_wfn=True)
    xyz=mol.geometry().np.copy()
    domains,state=psi4.core.dlpno_screened_pair_state(w)
    print('correction pairs',domains['DIPOLE_PAIRS'],domains['SCMP2_PAIRS'],pair_energies(w),flush=True)
    assert domains['DIPOLE_PAIRS'] and domains['SCMP2_PAIRS']
    assert np.all(abs(pair_energies(w))>1.e-9)
    assert bool(w.reference_wavefunction().nfrzc())==frozen
    @lru_cache(None)
    def model(delta):
        if not any(delta): return w
        m=mol.clone();m.set_geometry(psi4.core.Matrix.from_array(xyz+np.array(delta).reshape(xyz.shape)));m.update_geometry()
        _,other=psi4.energy('dlpno-ccsd(t)',molecule=m,return_wfn=True)
        assert_branch(other,w)
        return other
    def sample(u,a,v=0,b=0.):
        d=np.zeros(xyz.size); d[u]+=a; d[v]+=b
        return model(tuple(d))
    u,v=(2,18) if not frozen else (5,12)
    result=dict(system=system,localization=localization,basis=psi4.core.get_global_option('BASIS'),
                frozen_orbitals=w.reference_wavefunction().nfrzc(),
                coordinates_bohr=xyz.tolist(),corrections=pair_energies(w).tolist(),domains=domains)
    if stage in ('all','gradient'):
        g=assemble_cartesian_gradient(w)
        records=[]
        for k in (u,v):
            values=[(sample(k,h).energy()-sample(k,-h).energy())/(2*h) for h in (.004,.002)]
            error=g.flat[k]-(4*values[1]-values[0])/3
            print('total gradient',k,g.flat[k],values,error,flush=True)
            np.testing.assert_allclose(error,0.,atol=2.e-8,rtol=0.)
            records.append(dict(coordinate=k,value=float(g.flat[k]),estimates=values,error=float(error)))
        np.testing.assert_allclose(g.sum(axis=0),0.,atol=2.e-8,rtol=0.)
        np.testing.assert_allclose(np.cross(xyz,g).sum(axis=0),0.,atol=2.e-8,rtol=0.)
        result.update(gradient=g.tolist(),gradient_checks=records)
    if stage in ('all','mixed'):
        records=[]
        for a,b in ((u,u),(u,v),(v,v)):
            values=screened_pair_curvature(w,*mixed_inputs(w,a,b))
            analytic=np.array([values[k].uv for k in ('SCMP2','DIPOLE')])
            def stencil(h):
                if a==b:
                    return (pair_energies(sample(a,h))+pair_energies(sample(a,-h))-2*pair_energies(w))/(h*h)
                return sum(sa*sb*pair_energies(sample(a,sa*h,b,sb*h))/(4*h*h)
                           for sa in (-1,1) for sb in (-1,1))
            numerical=[stencil(h) for h in (.004,.002)]
            target=(4*numerical[1]-numerical[0])/3
            errors=analytic-target
            print('mixed corrections',a,b,analytic,target,errors,flush=True)
            np.testing.assert_allclose(errors,0.,atol=2.e-9,rtol=0.)
            for k,coordinate in enumerate((a,b),1):
                estimates=[(pair_energies(sample(coordinate,h))-pair_energies(sample(coordinate,-h)))/(2*h)
                           for h in (.004,.002)]
                first=np.array([values[label][k] for label in ('SCMP2','DIPOLE')])
                np.testing.assert_allclose(first,(4*estimates[1]-estimates[0])/3,atol=2.e-10,rtol=0.)
            records.append(dict(coordinates=[a,b],analytic=analytic.tolist(),reference=target.tolist(),errors=errors.tolist()))
        result['mixed_checks']=records
    # The original right solver uses the same pair estimates when triples/strong
    # settings are matched. No correction is silently removed from the gradient energy.
    psi4.set_options(dict(dlpno_gradient_model=False,t_cut_triples_weak=0.,t_cut_iter=0.,
                          t_cut_tno_strong=1.e-5,t_cut_tno_weak=1.e-5))
    ordinary,other=psi4.energy('dlpno-ccsd(t)',molecule=mol,return_wfn=True)
    np.testing.assert_allclose(pair_energies(other),pair_energies(w),atol=2.e-12,rtol=0.)
    np.testing.assert_allclose(ordinary,energy,atol=1.e-9,rtol=0.)
    result.update(energy=energy,ordinary_energy=ordinary)
    Path(path).write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--system',choices=('chain','lih'),default='chain')
    parser.add_argument('--localization',choices=('BOYS','PIPEK_MEZEY'),default='BOYS')
    parser.add_argument('--stage',choices=('all','gradient','mixed'),default='all')
    parser.add_argument('--output',default='screened_corrections_results.json')
    args=parser.parse_args()
    psi4.set_output_file('screened_corrections.out',False)
    run(args.system,args.localization,args.stage,args.output)
