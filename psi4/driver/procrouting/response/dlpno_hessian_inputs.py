"""Analytic molecular inputs for the guarded, dense RHF DLPNO Hessian.

Promoted from the independently validated development providers. Every value
and derivative is evaluated at the reference geometry; no energy, gradient,
coordinate displacement, or import from the test tree occurs here. Full AO
three-center panels are temporary inputs, so this is not a local-scaling path.
"""
from functools import lru_cache
import numpy as np
from psi4 import core
from .dlpno_triples_curvature import MixedBlock, mixed_product
from .dlpno_rhf_curvature import mixed_df_fock, rhf_mixed_density, mixed_occupied_partition
from .dlpno_localization_curvature import quadratic_localization_mixed, mulliken_operators


class _MolecularIntegrals:
    """Reference-owned integral panels, with bounded streaming fallbacks."""
    def __init__(self, ref):
        self.ref = ref
        self.mints = core.MintsHelper(ref, 0)
        self._first_owned = {}
        self._raw_owned = {}
        self.prepared = False
        # Wrap bound methods on the instance: no global cache can retain a
        # wavefunction or serve integrals from an earlier geometry.
        for name, size in (('ao_kinetic', 1), ('ao_potential', 1), ('ao_dipole', 1),
                           ('ao_oei_deriv1', 6), ('ao_oei_deriv2', 3),
                           ('ao_metric_deriv1', 4), ('ao_3center_deriv1', 4),
                           ('ao_elec_dip_deriv1', 2), ('ao_elec_dip_deriv2', 1)):
            fallback = lru_cache(maxsize=size)(getattr(self.mints, name))
            def get(*args, _name=name, _fallback=fallback):
                key = (_name, *args)
                if key in self._first_owned:
                    return self._first_owned[key]
                return _fallback(*args)
            get.cache_info = fallback.cache_info
            setattr(self, name, get)

    def raw(self, name):
        if name not in self._raw_owned:
            self._raw_owned[name] = _raw(self.ref, self.ref.get_basisset(name), self.mints)
        return self._raw_owned[name]

    def prepare_first(self):
        """Construct admitted first AO panels once, before any Hessian entry.

        Pair/triplet loops consume the native atom-local qij/qia/qab jets.
        Mixed AO panels are still delivered once per ordered nuclear pair;
        retaining all 9N^2 dense mixed panels is deliberately not required.
        """
        if self.prepared:
            return
        self.prepared = True
        limit = min(core.get_memory() // 32, 512 * 1024**2)
        used = 0
        counts = dict(metric=0, three_center=0, one_electron=0)
        required = dict(counts)
        n = self.ref.basisset().nbf()
        atoms = self.ref.molecule().natom()
        def retain(category, name, args, bytes):
            nonlocal used
            required[category] += bytes
            if bytes > limit - used:
                return
            value = getattr(self.mints, name)(*args)
            actual = sum(x.np.nbytes for x in value)
            if actual != bytes:
                raise RuntimeError('Unexpected AO derivative panel dimensions.')
            self._first_owned[(name, *args)] = value
            used += actual
            counts[category] += actual
        for kind in ('OVERLAP', 'KINETIC', 'POTENTIAL'):
            for atom in range(atoms):
                retain('one_electron', 'ao_oei_deriv1', (kind, atom), 3 * n * n * 8)
        for name in ('DF_BASIS_SCF', 'DF_BASIS_CC'):
            self.raw(name)
            nq = self.ref.get_basisset(name).nbf()
            for atom in range(atoms):
                retain('metric', 'ao_metric_deriv1', (atom, name), 3 * nq * nq * 8)
                retain('three_center', 'ao_3center_deriv1', (atom, name), 3 * nq * n * n * 8)
        self.first_storage = dict(used=used, limit=limit, retained=counts, requested=required)
        metric = sum(x[0].nbytes for x in self._raw_owned.values())
        three = sum(x[1].nbytes for x in self._raw_owned.values())
        core.print_out(f'\n    Hessian reference AO DF panels: metric {metric/1024**2:.2f} MiB; '
                       f'three-center {three/1024**2:.2f} MiB.\n'
                       f'    Persistent first AO panels: {used/1024**2:.2f}/{limit/1024**2:.2f} MiB.\n')
        for name in counts:
            core.print_out(f'      {name}: retained {counts[name]/1024**2:.2f} / '
                           f'requested {required[name]/1024**2:.2f} MiB.\n')


def _raw(ref, aux, mints=None):
    basis = ref.basisset()
    if mints is None: mints = core.MintsHelper(basis)
    zero = core.BasisSet.zero_ao_basis_set()
    J = mints.ao_eri(aux,zero,aux,zero).np.reshape(aux.nbf(),aux.nbf())
    Q = mints.ao_eri(aux,zero,basis,basis).np.reshape(aux.nbf(),basis.nbf(),basis.nbf())
    return J, Q


def _occupied_density_tangent(ref, coordinate):
    """Total one-spin occupied-density derivative, including frozen core."""
    C=ref.Ca_subset('AO','OCC').np
    direction=np.zeros((ref.molecule().natom(),3)); direction.flat[coordinate]=1.
    dc=core.rhf_occupied_tangent(ref,core.Matrix.from_array(direction)).np
    return dc@C.T+C@dc.T


def _occupied_density_tangents(ref):
    """Export all 3N first densities from a single native CPHF batch."""
    C = ref.Ca_subset('AO', 'OCC').np
    tangents = core.rhf_occupied_tangents(ref)
    if len(tangents) != 3*ref.molecule().natom():
        raise RuntimeError('Incomplete native RHF response batch.')
    return tuple(dc.np@C.T + C@dc.np.T for dc in tangents)


def molecular_input_provider(w):
    """Own bounded integral caches and 3N densities for one fixed Hessian.

    Never reuse this provider after changing the reference geometry/orbitals.
    Admitted first panels persist for the full assembly. Mixed three-center
    panels are temporary. Every entry uses the same first-density batch.
    """
    ref=w.reference_wavefunction()
    size=3*ref.molecule().natom()
    integrals=_MolecularIntegrals(ref)
    integrals.prepare_first()
    densities=None
    def inputs(u,v):
        nonlocal densities
        if not 0 <= u < size or not 0 <= v < size:
            raise ValueError('Invalid nuclear response coordinate.')
        if densities is None:
            densities=_occupied_density_tangents(ref)
            for result in densities: result.setflags(write=False)
        return molecular_inputs(w,u,v,Du=densities[u],Dv=densities[v],integrals=integrals)
    return inputs


def _rhf_inputs(ref,u,v,Du=None,Dv=None,partition=False,block_output=None,integrals=None):
    mints=integrals if integrals is not None else _MolecularIntegrals(ref)
    C=ref.Ca_subset('AO','OCC').np
    if Du is None: Du=_occupied_density_tangent(ref,u)
    if Dv is None: Dv=Du if u==v else _occupied_density_tangent(ref,v)
    au,xu=divmod(u,3); av,xv=divmod(v,3)
    S=ref.S().np
    overlap=MixedBlock(S,mints.ao_oei_deriv1('OVERLAP',au)[xu].np,
                       mints.ao_oei_deriv1('OVERLAP',av)[xv].np,
                       mints.ao_oei_deriv2('OVERLAP',au,av)[3*xu+xv].np)
    J,Q=mints.raw('DF_BASIS_SCF')
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
    # The reference action is identical in every mixed entry. Retain its two
    # native panels once, alongside the raw reference integrals.
    if not hasattr(mints, 'fock_action'):
        raw=core.Matrix.from_array(Q.reshape(len(J),-1))
        fitted=core.Matrix.from_array(np.linalg.solve(J,Q.reshape(len(J),-1)))
        def fock_action(D):
            return core.dlpno_df_fock_response([raw],[fitted],[core.Matrix.from_array(D)])[0].np
        mints.fock_action=fock_action
    action=mints.fock_action
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


def _localized_inputs(ref,L,kind,u,v,active=None,integrals=None):
    mints=integrals if integrals is not None else _MolecularIntegrals(ref)
    if active is None: _,_,active=_rhf_inputs(ref,u,v,partition=True,integrals=mints)
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


def molecular_inputs(w,u,v,*,Du=None,Dv=None,integrals=None):
    ref = w.reference_wavefunction()
    mints=integrals if integrals is not None else _MolecularIntegrals(ref)
    data = {}
    _,_,active = _rhf_inputs(ref,u,v,Du=Du,Dv=Dv,partition=True,block_output=data,integrals=mints)
    S,F,D = (data[k] for k in ('overlap','fock','density'))
    kind = 'BOYS' if w.variable('DLPNO RESPONSE BOYS') else 'PIPEK_MEZEY'
    L = _localized_inputs(ref,w.variable('DLPNO RESPONSE ACTIVE LMOS').np,kind,u,v,active,integrals=mints)
    au,xu=divmod(u,3); av,xv=divmod(v,3)
    dip0=mints.ao_dipole(); dipu=mints.ao_elec_dip_deriv1(au); dipv=mints.ao_elec_dip_deriv1(av)
    dip2=mints.ao_elec_dip_deriv2(au,av)
    dip=[MixedBlock(dip0[k].np,dipu[3*k+xu].np,dipv[3*k+xv].np,dip2[9*k+3*xu+xv].np) for k in range(3)]
    J,Q=mints.raw('DF_BASIS_CC')
    j2,q2=core.dlpno_df_mixed_integrals(ref,'DF_BASIS_CC',list(range(len(J))),list(range(len(S.value))),u,v)
    jm=MixedBlock(J,mints.ao_metric_deriv1(au,'DF_BASIS_CC')[xu].np.reshape(J.shape),
                    mints.ao_metric_deriv1(av,'DF_BASIS_CC')[xv].np.reshape(J.shape),j2.np.reshape(J.shape))
    qm=MixedBlock(Q,mints.ao_3center_deriv1(au,'DF_BASIS_CC')[xu].np.reshape(Q.shape),
                    mints.ao_3center_deriv1(av,'DF_BASIS_CC')[xv].np.reshape(Q.shape),q2.np.reshape(Q.shape))
    def df(aux):
        ids=list(aux)
        return jm.slice(np.ix_(ids,ids)),qm.slice(ids)
    occupied_action=lambda panel:mixed_product('mn,na->ma',D,panel)
    occupied_action.density=D
    df.full=(jm,qm)
    return S,F,L,occupied_action,dip,df
