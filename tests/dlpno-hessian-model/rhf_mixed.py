"""Analytic full-domain DF-RHF mixed-density provider and water diagnostics."""
from functools import lru_cache
import numpy as np
import psi4
from first_pno import raw
from water_hvp import BASE,molecule
from psi4.driver.procrouting.response.dlpno_triples_curvature import MixedBlock,mixed_product
from psi4.driver.procrouting.response.dlpno_rhf_curvature import mixed_df_fock,rhf_mixed_density,mixed_occupied_partition


def analytic_inputs(ref,u,v,Du=None,Dv=None,partition=False,block_output=None):
    mints=psi4.core.MintsHelper(ref)
    C=ref.Ca_subset('AO','OCC').np
    def tangent(k):
        direction=np.zeros((ref.molecule().natom(),3)); direction.flat[k]=1.
        dc=psi4.core.rhf_occupied_tangent(ref,psi4.core.Matrix.from_array(direction)).np
        return dc@C.T+C@dc.T
    if Du is None: Du=tangent(u)
    if Dv is None: Dv=Du if u==v else tangent(v)
    au,xu=divmod(u,3); av,xv=divmod(v,3)
    S=ref.S().np
    overlap=MixedBlock(S,mints.ao_oei_deriv1('OVERLAP',au)[xu].np,
                       mints.ao_oei_deriv1('OVERLAP',av)[xv].np,
                       mints.ao_oei_deriv2('OVERLAP',au,av)[3*xu+xv].np)
    J,Q=raw(ref,ref.get_basisset('DF_BASIS_SCF'))
    ju=mints.ao_metric_deriv1(au,'DF_BASIS_SCF')[xu].np.reshape(J.shape)
    jv=mints.ao_metric_deriv1(av,'DF_BASIS_SCF')[xv].np.reshape(J.shape)
    qu=mints.ao_3center_deriv1(au,'DF_BASIS_SCF')[xu].np.reshape(Q.shape)
    qv=mints.ao_3center_deriv1(av,'DF_BASIS_SCF')[xv].np.reshape(Q.shape)
    j2,q2=psi4.core.dlpno_df_mixed_integrals(ref,'DF_BASIS_SCF',list(range(len(J))),list(range(len(S))),u,v)
    metric=MixedBlock(J,ju,jv,j2.np.reshape(J.shape))
    q=MixedBlock(Q,qu,qv,q2.np.reshape(Q.shape))
    density=MixedBlock(C@C.T,Du,Dv,np.zeros_like(S))
    g=mixed_df_fock(metric,q,density)
    h0=mints.ao_kinetic().np+mints.ao_potential().np
    np.testing.assert_allclose(h0+g.value,ref.Fa().np,atol=2.e-8,rtol=0)
    def h1(atom,xyz):
        return sum(mints.ao_oei_deriv1(kind,atom)[xyz].np for kind in ('KINETIC','POTENTIAL'))
    h2=sum(mints.ao_oei_deriv2(kind,au,av)[3*xu+xv].np for kind in ('KINETIC','POTENTIAL'))
    fitted=np.linalg.solve(J,Q.reshape(len(J),-1)).reshape(Q.shape)
    def action(D):
        return (2*np.einsum('Qmn,Qrs,rs->mn',Q,fitted,D,optimize=True)
                -np.einsum('Qmr,Qns,rs->mn',Q,fitted,D,optimize=True))
    np.testing.assert_allclose(g.value,action(density.value),atol=1.e-11,rtol=0)
    fu,fv=h1(au,xu)+g.u,h1(av,xv)+g.v
    response=rhf_mixed_density(ref,overlap,Du,Dv,fu,fv,h2+g.uv,action)
    fock=MixedBlock(ref.Fa().np,fu,fv,response.fock)
    densities=density._replace(uv=response.density)
    if block_output is not None:
        block_output.update(overlap=overlap,fock=fock,density=densities)
    if partition:
        frozen,active=mixed_occupied_partition(C,ref.Ca_subset('AO','FROZEN_OCC').np.shape[1],densities,overlap,fock)
        for sub in (frozen,active):
            identity=mixed_product('mi,ij,jn->mn',sub,overlap,sub).uv-sub.uv
            np.testing.assert_allclose(identity,0,atol=2.e-10,rtol=0)
        cross=mixed_product('mi,ij,jn->mn',frozen,overlap,active).uv
        np.testing.assert_allclose(cross,0,atol=2.e-10,rtol=0)
        return response,frozen,active
    return response


def check(frozen,u,v,step=.004):
    psi4.set_options({'freeze_core':frozen,'save_jk':u!=v})
    _,ref=psi4.energy('scf',molecule=molecule(BASE),return_wfn=True)
    original=ref.jk()
    before_D,before_F=ref.Da().np.copy(),ref.Fa().np.copy()
    response,core_response,active_response=analytic_inputs(ref,u,v,partition=True)
    assert ref.jk() is original
    np.testing.assert_array_equal(before_D,ref.Da().np)
    np.testing.assert_array_equal(before_F,ref.Fa().np)
    @lru_cache(None)
    def sample(delta):
        _,r=psi4.energy('scf',molecule=molecule(BASE+np.asarray(delta).reshape(3,3)),return_wfn=True)
        core=r.Ca_subset('AO','FROZEN_OCC').np
        return r.Da().np.copy(),r.Fa().np.copy(),core@core.T
    def stencil(h):
        nodes=[(1,0,1),(-1,0,1),(0,0,-2)] if u==v else [(a,b,a*b/4) for a in (-1,1) for b in (-1,1)]
        data=[]
        for a,b,w in nodes:
            delta=np.zeros(9); delta[u]+=a*h; delta[v]+=b*h
            data.append((w/(h*h),sample(tuple(delta))))
        return [sum(w*x[k] for w,x in data) for k in (0,1,2)]
    coarse,fine=stencil(2*step),stencil(step)
    numerical=[(4*b-a)/3 for a,b in zip(coarse,fine)]
    errors=[np.max(abs(response.density-numerical[0])),np.max(abs(response.fock-numerical[1]))]
    print('RHF mixed',frozen,u,v,step,'density/Fock errors',errors,'constraints',response[2:],flush=True)
    np.testing.assert_allclose(errors[0],0,atol=2.e-7,rtol=0)
    np.testing.assert_allclose(errors[1],0,atol=5.e-7,rtol=0)
    core_error=np.max(abs(core_response.uv-numerical[2]))
    active_error=np.max(abs(active_response.uv-(numerical[0]-numerical[2])))
    print('Frozen/active mixed density errors',core_error,active_error,flush=True)
    np.testing.assert_allclose([core_error,active_error],0,atol=2.e-7,rtol=0)
    np.testing.assert_allclose(core_response.uv+active_response.uv,response.density,atol=1.e-13,rtol=0)
    if u!=v:
        reverse=analytic_inputs(ref,v,u)
        np.testing.assert_allclose(response.density,reverse.density,atol=2.e-9,rtol=0)
        np.testing.assert_allclose(response.fock,reverse.fock,atol=2.e-9,rtol=0)


if __name__=='__main__':
    psi4.set_num_threads(1)
    psi4.set_output_file('rhf_mixed.out',False)
    psi4.set_options(dict(basis='cc-pvdz',reference='rhf',scf_type='df',df_basis_scf='cc-pvdz-jkfit',
                          e_convergence=1.e-13,d_convergence=1.e-13,scf__diis=False,scf_initial_accelerator='NONE'))
    for frozen in (True,False):
        for u,v in ((5,5),(1,5)):
            check(frozen,u,v)
