"""Mixed PAO/virtual-parent checks; nuclear RHF mixed density remains sampled."""
from functools import lru_cache
import numpy as np
import psi4
from water_hvp import BASE,molecule
from psi4.driver.procrouting.response.dlpno_frame_curvature import (
    occupied_projector_action,mixed_virtual_parent,mixed_paos,mixed_metric_solve)
from psi4.driver.procrouting.response.dlpno_triples_curvature import MixedBlock,mixed_product


def check_rotation_invariance():
    rng=np.random.default_rng(541)
    C=MixedBlock(*(rng.normal(size=(9,4)) for _ in range(4)))
    panel=MixedBlock(*(rng.normal(size=(9,3)) for _ in range(4)))
    def skew():
        x=rng.normal(size=(4,4)); return x-x.T
    K,H,J=skew(),skew(),skew()
    U=MixedBlock(np.eye(4),K,H,J+.5*(K@H+H@K))
    rotated=mixed_product('mi,ij->mj',C,U)
    a,b=occupied_projector_action(C,panel),occupied_projector_action(rotated,panel)
    for x,y in zip(a,b): np.testing.assert_allclose(x,y,rtol=0,atol=2.e-12)
    metric=MixedBlock(np.eye(9)*12,*(rng.normal(size=(9,9)) for _ in range(3)))
    solved=mixed_metric_solve(metric,panel)
    recovered=mixed_product('mn,na->ma',metric,solved)
    for x,y in zip(recovered,panel): np.testing.assert_allclose(x,y,rtol=0,atol=1.e-12)
    print('Mixed occupied rotation cancellation',max(np.max(abs(x-y)) for x,y in zip(a,b)),flush=True)


def check(frozen,u,v):
    psi4.set_options({'freeze_core':frozen})
    _,ref=psi4.energy('scf',molecule=molecule(BASE),return_wfn=True)
    C=ref.Ca_subset('AO','OCC').np
    V=ref.Ca_subset('AO','VIR').np
    S=ref.S().np
    mints=psi4.core.MintsHelper(ref)
    def first(k):
        direction=np.zeros((3,3)); direction.flat[k]=1.
        dC=psi4.core.rhf_occupied_tangent(ref,psi4.core.Matrix.from_array(direction)).np
        return dC@C.T+C@dC.T
    Du,Dv=first(u),first(v)
    au,xu=divmod(u,3); av,xv=divmod(v,3)
    metric=MixedBlock(S,mints.ao_oei_deriv1('OVERLAP',au)[xu].np,
                     mints.ao_oei_deriv1('OVERLAP',av)[xv].np,
                     mints.ao_oei_deriv2('OVERLAP',au,av)[3*xu+xv].np)
    domain=np.array([0,3,6])
    @lru_cache(None)
    def sample(delta):
        _,r=psi4.energy('scf',molecule=molecule(BASE+np.asarray(delta).reshape(3,3)),return_wfn=True)
        c=r.Ca_subset('AO','OCC').np; x=r.Ca_subset('AO','VIR').np
        a,_,b=np.linalg.svd(x.T@S@V)
        y=x@(a@b)
        raw=np.eye(len(S))[:,domain]-c@(c.T@r.S().np[:,domain])
        norm=np.sqrt(np.sum(raw*(r.S().np@raw),axis=0))
        return c@c.T,y,raw,raw/norm
    def stencil(h):
        nodes=[(1,0,1),(-1,0,1),(0,0,-2)] if u==v else [(a,b,a*b/4) for a in (-1,1) for b in (-1,1)]
        values=[]
        for a,b,w in nodes:
            delta=np.zeros(9); delta[u]+=a*h; delta[v]+=b*h
            values.append((w/(h*h),sample(tuple(delta))))
        return [sum(w*x[k] for w,x in values) for k in range(4)]
    coarse,fine=stencil(.002),stencil(.001)
    Duv,Vuv,rawuv,normuv=[(4*b-a)/3 for a,b in zip(coarse,fine)]
    density=MixedBlock(C@C.T,Du,Dv,Duv)
    action=lambda panel:mixed_product('mn,na->ma',density,panel)
    parent=mixed_virtual_parent(action,metric,V)
    raw,normalized=mixed_paos(action,metric,domain)
    errors=[np.max(abs(parent.uv-Vuv)),np.max(abs(raw.uv-rawuv)),np.max(abs(normalized.uv-normuv))]
    print('Frame curvature',frozen,u,v,'parent/raw/normalized errors',errors,flush=True)
    np.testing.assert_allclose(errors,0,rtol=0,atol=3.e-6)
    gram=mixed_product('ma,mn,nb->ab',parent,metric,parent)
    np.testing.assert_allclose(gram.uv,0,rtol=0,atol=1.e-12)
    norm=mixed_product('ma,mn,na->a',normalized,metric,normalized)
    np.testing.assert_allclose(norm.uv,0,rtol=0,atol=1.e-11)
    reversed_density=MixedBlock(density.value,density.v,density.u,density.uv)
    reverse=mixed_virtual_parent(lambda p:mixed_product('mn,na->ma',reversed_density,p),
                                 MixedBlock(metric.value,metric.v,metric.u,metric.uv),V)
    np.testing.assert_allclose(parent.uv,reverse.uv,rtol=0,atol=1.e-12)


if __name__=='__main__':
    psi4.set_num_threads(1)
    psi4.set_output_file('frame_curvature.out',False)
    psi4.set_options(dict(basis='cc-pvdz',reference='rhf',scf_type='df',df_basis_scf='cc-pvdz-jkfit',
                          e_convergence=1.e-13,d_convergence=1.e-13,scf__diis=False,scf_initial_accelerator='NONE'))
    check_rotation_invariance()
    for frozen in (True,False):
        for u,v in ((5,5),(1,5)):
            check(frozen,u,v)
