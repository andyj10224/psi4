"""Analytic molecular inputs for the guarded, dense RHF DLPNO Hessian.

Promoted from the independently validated development providers. Every value
and derivative is evaluated at the reference geometry; no energy, gradient,
coordinate displacement, or import from the test tree occurs here. Full AO
three-center panels are temporary inputs, so this is not a local-scaling path.
"""
import numpy as np
from psi4 import core
from .dlpno_triples_curvature import MixedBlock, mixed_product
from .dlpno_rhf_curvature import mixed_df_fock, rhf_mixed_density, mixed_occupied_partition
from .dlpno_localization_curvature import quadratic_localization_mixed, mulliken_operators


def _raw(ref, aux):
    basis = ref.basisset()
    mints = core.MintsHelper(ref)
    zero = core.BasisSet.zero_ao_basis_set()
    J = mints.ao_eri(aux,zero,aux,zero).np.reshape(aux.nbf(),aux.nbf())
    Q = mints.ao_eri(aux,zero,basis,basis).np.reshape(aux.nbf(),basis.nbf(),basis.nbf())
    return J, Q


def _rhf_inputs(ref,u,v,Du=None,Dv=None,partition=False,block_output=None):
    mints=core.MintsHelper(ref)
    C=ref.Ca_subset('AO','OCC').np
    def tangent(k):
        direction=np.zeros((ref.molecule().natom(),3)); direction.flat[k]=1.
        dc=core.rhf_occupied_tangent(ref,core.Matrix.from_array(direction)).np
        return dc@C.T+C@dc.T
    if Du is None: Du=tangent(u)
    if Dv is None: Dv=Du if u==v else tangent(v)
    au,xu=divmod(u,3); av,xv=divmod(v,3)
    S=ref.S().np
    overlap=MixedBlock(S,mints.ao_oei_deriv1('OVERLAP',au)[xu].np,
                       mints.ao_oei_deriv1('OVERLAP',av)[xv].np,
                       mints.ao_oei_deriv2('OVERLAP',au,av)[3*xu+xv].np)
    J,Q=_raw(ref,ref.get_basisset('DF_BASIS_SCF'))
    ju=mints.ao_metric_deriv1(au,'DF_BASIS_SCF')[xu].np.reshape(J.shape)
    jv=mints.ao_metric_deriv1(av,'DF_BASIS_SCF')[xv].np.reshape(J.shape)
    qu=mints.ao_3center_deriv1(au,'DF_BASIS_SCF')[xu].np.reshape(Q.shape)
    qv=mints.ao_3center_deriv1(av,'DF_BASIS_SCF')[xv].np.reshape(Q.shape)
    j2,q2=core.dlpno_df_mixed_integrals(ref,'DF_BASIS_SCF',list(range(len(J))),list(range(len(S))),u,v)
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


def _localized_inputs(ref,L,kind,u,v,active=None):
    if active is None: _,_,active=_rhf_inputs(ref,u,v,partition=True)
    mints=core.MintsHelper(ref)
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


def molecular_inputs(w,u,v):
    ref = w.reference_wavefunction()
    data = {}
    _,_,active = _rhf_inputs(ref,u,v,partition=True,block_output=data)
    S,F,D = (data[k] for k in ('overlap','fock','density'))
    kind = 'BOYS' if w.variable('DLPNO RESPONSE BOYS') else 'PIPEK_MEZEY'
    L = _localized_inputs(ref,w.variable('DLPNO RESPONSE ACTIVE LMOS').np,kind,u,v,active)
    mints=core.MintsHelper(ref)
    au,xu=divmod(u,3); av,xv=divmod(v,3)
    dip0=mints.ao_dipole(); dipu=mints.ao_elec_dip_deriv1(au); dipv=mints.ao_elec_dip_deriv1(av)
    dip2=mints.ao_elec_dip_deriv2(au,av)
    dip=[MixedBlock(dip0[k].np,dipu[3*k+xu].np,dipv[3*k+xv].np,dip2[9*k+3*xu+xv].np) for k in range(3)]
    J,Q=_raw(ref,ref.get_basisset('DF_BASIS_CC'))
    j2,q2=core.dlpno_df_mixed_integrals(ref,'DF_BASIS_CC',list(range(len(J))),list(range(len(S.value))),u,v)
    jm=MixedBlock(J,mints.ao_metric_deriv1(au,'DF_BASIS_CC')[xu].np.reshape(J.shape),
                    mints.ao_metric_deriv1(av,'DF_BASIS_CC')[xv].np.reshape(J.shape),j2.np.reshape(J.shape))
    qm=MixedBlock(Q,mints.ao_3center_deriv1(au,'DF_BASIS_CC')[xu].np.reshape(Q.shape),
                    mints.ao_3center_deriv1(av,'DF_BASIS_CC')[xv].np.reshape(Q.shape),q2.np.reshape(Q.shape))
    def df(aux):
        ids=list(aux)
        return jm.slice(np.ix_(ids,ids)),qm.slice(ids)
    return S,F,L,lambda panel:mixed_product('mn,na->ma',D,panel),dip,df
