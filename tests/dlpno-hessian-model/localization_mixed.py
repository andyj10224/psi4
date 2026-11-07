"""Water mixed Boys/PM response and analytic electronic dipole derivatives."""
from functools import lru_cache
import numpy as np
import psi4
from scipy.optimize import linear_sum_assignment
from scipy.linalg import expm
from psi4.driver.procrouting.response.dlpno_orbital import _solve_localization_rotations
from water_hvp import BASE,molecule
from rhf_mixed import analytic_inputs
from psi4.driver.procrouting.response.dlpno_triples_curvature import MixedBlock
from psi4.driver.procrouting.response.dlpno_localization_curvature import quadratic_localization_mixed,mulliken_operators


def molecular_localization(ref,L,kind,u,v,active=None):
    if active is None: _,_,active=analytic_inputs(ref,u,v,partition=True)
    mints=psi4.core.MintsHelper(ref)
    au,xu=divmod(u,3); av,xv=divmod(v,3)
    metric=MixedBlock(ref.S().np,mints.ao_oei_deriv1('OVERLAP',au)[xu].np,
                     mints.ao_oei_deriv1('OVERLAP',av)[xv].np,
                     mints.ao_oei_deriv2('OVERLAP',au,av)[3*xu+xv].np)
    if kind=='BOYS':
        base=mints.ao_dipole()
        first=mints.ao_elec_dip_deriv1(au); second=mints.ao_elec_dip_deriv1(av)
        mixed=mints.ao_elec_dip_deriv2(au,av)
        operators=[MixedBlock(base[k].np,first[3*k+xu].np,second[3*k+xv].np,
                              mixed[9*k+3*xu+xv].np) for k in range(3)]
    elif kind=='PIPEK_MEZEY':
        operators=list(mulliken_operators(metric,[ref.basisset().function_to_center(a) for a in range(ref.basisset().nbf())]))
    else: raise ValueError(kind)
    return quadratic_localization_mixed(L,active,metric,operators)


def localize(ref,kind,reference=None,basis=None):
    C=ref.Ca_subset('AO','ACTIVE_OCC').np
    if reference is not None:
        cross=psi4.core.MintsHelper(ref).ao_overlap(ref.basisset(),basis).np
        a,_,b=np.linalg.svd(C.T@cross@reference)
        C=C@(a@b)
    loc=psi4.core.Localizer.build(kind,ref.basisset(),psi4.core.Matrix.from_array(C))
    loc.localize()
    L=loc.L.np.copy()
    # Polish a near-stationary native initial guess for the derivative oracle.
    # Forward stationarity noise is amplified by the displacement squared.
    if kind=='BOYS':
        operators=[x.np for x in psi4.core.MintsHelper(ref).ao_dipole()]
    else:
        S=ref.S().np
        centers=np.array([ref.basisset().function_to_center(a) for a in range(len(S))])
        operators=[]
        for atom in np.unique(centers):
            mask=(centers==atom).astype(float)
            operators.append(.5*(mask[:,None]*S+S*mask[None,:]))
    i,j=np.triu_indices(L.shape[1],1)
    for iteration in range(6):
        matrices=[L.T@O@L for O in operators]
        g=sum(4*(a.diagonal()[i]-a.diagonal()[j])*a[i,j] for a in matrices)
        if iteration==0 and not loc.converged and np.max(abs(g),initial=0.)>1.e-5:
            raise AssertionError('Native localization did not provide a near-stationary initial guess')
        if np.max(abs(g),initial=0.)<2.e-13: break
        step=_solve_localization_rotations(matrices,2,-g,1.e-13,require_unique=True)
        K=np.zeros((L.shape[1],L.shape[1])); K[j,i],K[i,j]=step,-step
        L=L@expm(K)
    else: raise AssertionError('Localization polishing failed')
    _solve_localization_rotations(matrices,2,np.zeros_like(g),1.e-13,require_unique=True)
    if reference is not None:
        match=reference.T@cross.T@L
        _,order=linear_sum_assignment(-abs(match))
        assert min(abs(match[np.arange(len(order)),order]))>.99
        L=L[:,order]*np.sign(match[np.arange(len(order)),order])
    return L


def check(frozen,kind,u,v):
    psi4.set_options({'freeze_core':frozen})
    _,ref=psi4.energy('scf',molecule=molecule(BASE),return_wfn=True)
    L=localize(ref,kind)
    response=molecular_localization(ref,L,kind,u,v)
    @lru_cache(None)
    def sample(delta):
        _,r=psi4.energy('scf',molecule=molecule(BASE+np.asarray(delta).reshape(3,3)),return_wfn=True)
        return localize(r,kind,L,ref.basisset())
    def stencil(h):
        nodes=[(1,0,1),(-1,0,1),(0,0,-2)] if u==v else [(a,b,a*b/4) for a in (-1,1) for b in (-1,1)]
        out=np.zeros_like(L)
        for a,b,w in nodes:
            d=np.zeros(9); d[u]+=a*h; d[v]+=b*h
            out+=w/(h*h)*sample(tuple(d))
        return out
    target=(4*stencil(.004)-stencil(.008))/3
    error=np.max(abs(target-response.uv))
    print('Mixed localization',frozen,kind,u,v,error,flush=True)
    np.testing.assert_allclose(response.uv,target,atol=3.e-6,rtol=0)
    reverse=molecular_localization(ref,L,kind,v,u)
    np.testing.assert_allclose(response.uv,reverse.uv,atol=2.e-8,rtol=0)


def check_dipoles(pure,basis_name='cc-pvdz'):
    basis=psi4.core.BasisSet.build(molecule(BASE),'BASIS',basis_name,puream=int(pure))
    m=psi4.core.MintsHelper(basis)
    maximum=0.
    for atom1,atom2 in ((0,0),(0,1),(1,1)):
        analytical=m.ao_elec_dip_deriv2(atom1,atom2)
        reverse=m.ao_elec_dip_deriv2(atom2,atom1)
        for x in range(3):
            def first(h):
                d=BASE.copy(); d[atom2,x]+=h
                b=psi4.core.BasisSet.build(molecule(d),'BASIS',basis_name,puream=int(pure))
                return [a.np for a in psi4.core.MintsHelper(b).ao_elec_dip_deriv1(atom1)]
            p,n=first(.001),first(-.001)
            p2,n2=first(.002),first(-.002)
            for k in range(3):
                for y in range(3):
                    target=(4*(p[3*k+y]-n[3*k+y])/.002-(p2[3*k+y]-n2[3*k+y])/.004)/3
                    actual=analytical[9*k+3*y+x].np
                    maximum=max(maximum,np.max(abs(actual-target)))
                    np.testing.assert_allclose(actual,reverse[9*k+3*x+y].np,atol=2.e-12,rtol=0)
                    np.testing.assert_allclose(actual,actual.T,atol=2.e-12,rtol=0)
    total=[np.zeros((basis.nbf(),basis.nbf())) for _ in range(27)]
    for atom in range(3):
        blocks=m.ao_elec_dip_deriv2(atom,1)
        for target,block in zip(total,blocks): target+=block.np
    ds=m.ao_oei_deriv1('OVERLAP',1)
    translation=0.
    for k in range(3):
        for x in range(3):
            for y in range(3):
                translation=max(translation,np.max(abs(total[9*k+3*x+y]+(ds[y].np if k==x else 0))))
    assert translation<2.e-10,translation
    print('Mixed electronic dipole',basis_name,pure,basis.nbf(),'maximum error',maximum,'translation',translation,flush=True)
    assert maximum<2.e-8


if __name__=='__main__':
    psi4.set_num_threads(1)
    psi4.set_output_file('localization_mixed.out',False)
    psi4.set_options(dict(basis='cc-pvdz',reference='rhf',scf_type='df',df_basis_scf='cc-pvdz-jkfit',
        e_convergence=1.e-13,d_convergence=1.e-13,scf__diis=False,scf_initial_accelerator='NONE',
        local_convergence=1.e-14,local_gradient_convergence=1.e-7))
    for pure in (False,True):
        psi4.set_options({'puream':pure})
        check_dipoles(pure)
        check_dipoles(pure,'cc-pvtz')
    for frozen in (True,False):
        for kind in ('BOYS','PIPEK_MEZEY'):
            for u,v in ((5,5),(1,5)):
                check(frozen,kind,u,v)
