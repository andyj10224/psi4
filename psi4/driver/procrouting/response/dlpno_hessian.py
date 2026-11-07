"""Analytic DLPNO Hessian delivery with explicit model/resource guards."""
import gc
import numpy as np
from psi4 import core
from ...p4util.exceptions import ValidationError
from .dlpno_hessian_inputs import molecular_input_provider
from .dlpno_screened_hessian import COMPONENTS, ScreenedHessianColumn


def validate_settings():
    """Cheap checks also used by derivative negotiation, before any SCF work."""
    if not core.get_option('DLPNO', 'DLPNO_HESSIAN_MODEL'):
        raise ValidationError('Analytic DLPNO Hessians require DLPNO_HESSIAN_MODEL=true.')
    if core.get_option('SCF', 'REFERENCE') != 'RHF':
        raise ValidationError('Analytic DLPNO Hessians require RHF.')
    if core.has_global_option_changed('SCF_TYPE') and core.get_global_option('SCF_TYPE') not in ('DF','MEM_DF','DISK_DF'):
        raise ValidationError('Analytic DLPNO Hessians require density-fitted SCF.')
    if core.get_option('DLPNO', 'DLPNO_LOCAL_ORBITALS') not in ('BOYS','PIPEK_MEZEY'):
        raise ValidationError('Analytic DLPNO Hessians require Boys or Mulliken Pipek-Mezey localization.')
    if core.get_option('DLPNO','DLPNO_REFERENCE_ORBITALS') != 'HF' or core.get_option('DLPNO','DLPNO_BRUECKNER_ORBS'):
        raise ValidationError('Analytic DLPNO Hessians require HF reference orbitals, without Brueckner optimization.')
    screens=('DLPNO_AO_INTS_TOL','F_CUT','F_CUT_T','T_CUT_DO_UV')
    active=[key for key in screens if core.get_option('DLPNO',key) != 0.]
    if active:
        raise ValidationError('Analytic DLPNO Hessians currently require these screens to be explicitly zero: '+', '.join(active)+'.')
    if any(core.get_option('SCF',key) for key in ('PCM','PE','DDX','PERTURB_H')) or core.get_global_option('RELATIVISTIC') != 'NO':
        raise ValidationError('Analytic DLPNO Hessians currently require gas-phase nonrelativistic RHF without applied fields.')
    if core.get_global_option('NUM_FROZEN_UOCC') or any(core.get_global_option('FROZEN_UOCC')):
        raise ValidationError('Analytic DLPNO Hessians do not support frozen virtual orbitals.')


def preflight(molecule, kwargs):
    """Reject unsupported inputs and budget dense scratch before correlation."""
    validate_settings()
    if kwargs.get('ref_wfn') is not None:
        raise ValidationError('The guarded DLPNO Hessian driver currently builds its own RHF reference; ref_wfn is unsupported.')
    if kwargs.get('external_potentials') is not None or getattr(core,'EXTERN',None) is not None or hasattr(molecule,'EFP'):
        raise ValidationError('Analytic DLPNO Hessians do not support external potentials or EFP.')
    if molecule.multiplicity()!=1 or any(molecule.Z(a)<=0 for a in range(molecule.natom())):
        raise ValidationError('Analytic DLPNO Hessians require a closed-shell molecule without ghost centers.')
    basis=core.BasisSet.build(molecule,'BASIS',core.get_global_option('BASIS'))
    if basis.has_ECP():
        raise ValidationError('Analytic DLPNO Hessians do not support effective core potentials.')
    aux_scf=core.BasisSet.build(molecule,'DF_BASIS_SCF',core.get_option('SCF','DF_BASIS_SCF'),'JKFIT',basis.name())
    aux_cc=core.BasisSet.build(molecule,'DF_BASIS_CC',core.get_option('DLPNO','DF_BASIS_CC'),'RIFIT',basis.name())
    n=basis.nbf();a,b=aux_scf.nbf(),aux_cc.nbf();size=3*molecule.natom()
    no=int(round((sum(molecule.Z(i) for i in range(molecule.natom()))-molecule.molecular_charge())/2))
    nv=max(0,n-no);triplets=no*(no+1)*(no+2)//6-no
    # Conservative full-domain bounds for dense mixed fitting panels and
    # retained triples/adjoints. Reserve half the requested memory for native
    # forward work. Native allocation guards still apply independently.
    estimate=128*1024**2+8*(64*((a+b)*n*n+a*a+b*b)+64*triplets*nv**3+128*size*n*n)
    if estimate>core.get_memory()//2:
        raise ValidationError(f'Dense analytic DLPNO Hessian estimate {estimate/1024**2:.1f} MiB exceeds half '
                              f'the requested memory; request at least {2*estimate/1024**3:.2f} GiB. '
                              'This experimental driver does not provide local memory scaling.')
    return estimate


def _identities(matrix, gradient, xyz):
    translations=np.tile(np.eye(3),(len(xyz),1))
    rotations=np.stack([np.cross(axis,xyz).ravel() for axis in np.eye(3)],axis=1)
    rotated_gradient=np.stack([np.cross(axis,gradient).ravel() for axis in np.eye(3)],axis=1)
    return dict(symmetry=float(np.max(abs(matrix-matrix.T))),
                left_translation=float(np.max(abs(translations.T@matrix))),
                right_translation=float(np.max(abs(matrix@translations))),
                rotation_covariance=float(np.max(abs(matrix@rotations-rotated_gradient))))


def assemble_hessian(wfn):
    """Return all independently evaluated ordered entries, never projected."""
    ref=wfn.reference_wavefunction();size=3*wfn.molecule().natom()
    if ref.nirrep()!=1 or ref.Ca().np.shape[1]!=ref.basisset().nbf():
        raise ValidationError('Analytic DLPNO Hessians require a C1 reference with a full-rank AO metric.')
    gradient=wfn.gradient().np.copy();xyz=wfn.molecule().geometry().np.copy()
    rhf=core.scfhess(ref).np.copy()
    inputs=molecular_input_provider(wfn)
    parts=np.empty((len(COMPONENTS),size,size));firsts=np.empty_like(parts)
    residuals=np.zeros(4)
    for v in range(size):
        column=ScreenedHessianColumn(wfn,v,inputs,rhf[:,v])
        for u in range(size):
            entry=column.element(u)
            parts[:,u,v]=entry['components'];firsts[:,u,v]=entry['gradient_components']
            residuals=np.maximum(residuals,[max(entry['cc_residual_errors']),max(entry['pno_residual_errors']),
                                          entry['triples_right_residual'],entry['triples_left_residual']])
        core.print_out(f'    Analytic DLPNO Hessian column {v+1:3d}/{size:3d} complete.\n')
        del column
        gc.collect()
    hessian=parts.sum(axis=0)
    replay=float(np.max(abs(firsts.sum(axis=0)-gradient.ravel()[:,None])))
    spread=float(np.max(abs(firsts-firsts[:,:,:1])))
    if not all(np.isfinite(array).all() for array in (gradient,parts,firsts,residuals)) or max(replay,spread)>2.e-8:
        raise RuntimeError('Analytic DLPNO Hessian failed its native first-gradient replay.')
    identities=_identities(hessian,gradient,xyz)
    component_checks=[_identities(part,firsts[k,:,0].reshape(xyz.shape),xyz) for k,part in enumerate(parts)]
    if max(value for checks in [identities,*component_checks] for value in checks.values())>2.e-7:
        raise RuntimeError('Analytic DLPNO Hessian failed raw symmetry or rigid-motion checks; no projection applied.')
    if max(np.max(abs(gradient.sum(axis=0))),np.max(abs(np.cross(xyz,gradient).sum(axis=0))))>2.e-8:
        raise RuntimeError('Analytic DLPNO gradient failed the force/torque check.')
    result=core.Matrix.from_array(hessian)
    wfn.set_hessian(result)
    for name,part in zip(COMPONENTS,parts):
        wfn.set_variable('DLPNO HESSIAN '+name.upper(),core.Matrix.from_array(part))
    for name,value in identities.items():
        wfn.set_variable('DLPNO HESSIAN '+name.upper().replace('_',' ')+' ERROR',value)
    wfn.set_variable('DLPNO HESSIAN GRADIENT REPLAY ERROR',replay)
    wfn.set_variable('DLPNO HESSIAN COMPONENT GRADIENT SPREAD',spread)
    for name,value in zip(('CCSD','LMP2','TRIPLES RIGHT','TRIPLES LEFT'),residuals):
        wfn.set_variable('DLPNO HESSIAN '+name+' RESIDUAL ERROR',float(value))
    for label in ('CURRENT HESSIAN','DLPNO-CCSD(T) TOTAL HESSIAN'):
        wfn.set_variable(label,result)
    return result
