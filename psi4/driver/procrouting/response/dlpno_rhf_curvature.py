"""Mixed total occupied-density response for a C1 DF-RHF reference.

Reuses native CPHF; callers provide analytic Fock/metric derivative actions.
The total occupied solve includes frozen core; a spectral partition supplies
frozen/active subspace densities. Localized coefficient response is separate.
"""
from typing import NamedTuple
import numpy as np
from .dlpno_triples_curvature import MixedBlock, mixed_product
from .dlpno_frame_curvature import mixed_metric_solve


class RHFMixedResponse(NamedTuple):
    density: np.ndarray
    fock: np.ndarray
    idempotency_error: float
    stationarity_error: float
    cphf_error: float


def mixed_df_fock(metric, three_center, density):
    """2J-K mixed contraction without a four-index ERI tensor.

    Uses full-rank inverse-metric DF and storage O(naux*nao^2). This dense
    factor provider is not a local-scaling derivative implementation.
    """
    shape=three_center.value.shape
    panels=MixedBlock(*(q.reshape(shape[0],-1) for q in three_center))
    solved=mixed_metric_solve(metric,panels)
    from psi4 import core
    result=core.dlpno_df_fock_response(
        [core.Matrix.from_array(q) for q in panels],
        [core.Matrix.from_array(q) for q in solved],
        [core.Matrix.from_array(d) for d in density])
    return MixedBlock(*(m.np for m in result))


def rhf_mixed_density(reference, overlap, density_u, density_v, fock_u, fock_v,
                      fock_mixed_source, fock_action, tol=2.e-8, maxiter=100):
    """Solve mixed D S D=D and F D S=S D F with native occupied-virtual CPHF.

    D is the one-spin RHF density. fock_mixed_source excludes G[D_uv] but
    includes h_uv, G_uv[D], G_u[D_v], and G_v[D_u]. fock_action supplies
    the reference linear G=2J-K action. All arrays use fixed AO coefficients.
    """
    from psi4 import core
    if not np.isfinite(tol) or tol<=0 or maxiter<=0:
        raise ValueError('Expected a positive response tolerance and iteration limit.')
    if not isinstance(reference,core.RHF) or reference.nirrep()!=1:
        raise ValueError('Mixed density response requires a C1 RHF reference.')
    functional=reference.functional()
    if functional.needs_xc() or functional.x_alpha()!=1. or functional.is_x_lrc():
        raise ValueError('Mixed density response requires Hartree-Fock, without DFT or range separation.')
    C=reference.Ca().np
    S=overlap.value
    no=reference.nalpha()
    n=C.shape[0]
    if not 0<no<C.shape[1]:
        raise ValueError('Mixed CPHF response requires nonempty occupied and virtual spaces.')
    if C.shape[1]!=n:
        raise ValueError('Mixed density response currently requires a full-rank AO metric.')
    arrays=(*overlap,density_u,density_v,fock_u,fock_v,fock_mixed_source)
    if any(np.shape(x)!=(n,n) or not np.isfinite(x).all() for x in arrays):
        raise ValueError('Expected finite square AO mixed-response inputs.')
    if any(not np.allclose(x,x.T,rtol=0,atol=tol) for x in arrays):
        raise ValueError('RHF mixed-response inputs must be symmetric.')
    if not np.allclose(S,reference.S().np,rtol=0,atol=1.e-10):
        raise ValueError('Mixed response metric disagrees with the native RHF reference.')
    Co,Cv=C[:,:no],C[:,no:]
    D=Co@Co.T
    zero=np.zeros_like(D)
    density=MixedBlock(D,density_u,density_v,zero)
    # B collects every mixed idempotency term except those linear in D_uv.
    B=mixed_product('mi,ij,jn->mn',density,overlap,density).uv
    Bmo=C.T@S@B@S@C
    diagonal=np.zeros_like(Bmo)
    diagonal[:no,:no]=-Bmo[:no,:no]
    diagonal[no:,no:]=Bmo[no:,no:]
    seed=C@diagonal@C.T
    F=reference.Fa().np
    def residual(d,f):
        left=mixed_product('mi,ij,jn->mn',f,d,overlap).uv
        right=mixed_product('mi,ij,jn->mn',overlap,d,f).uv
        return left-right
    fseed=fock_mixed_source+fock_action(seed)
    source=residual(density._replace(uv=seed),MixedBlock(F,fock_u,fock_v,fseed))
    if not np.isfinite(source).all():
        raise ValueError('Nonfinite mixed RHF Fock source.')
    # Native RHF::cphf_Hx has the negative orbital-gap/G response convention.
    rhs=core.Matrix.from_array((Cv.T@source@Co).T)
    old_jk=reference.jk()
    created=old_jk is None
    try:
        if created:
            reference.initialize_jk(int(core.get_memory()*.8/8))
        rotation,=reference.cphf_solve([rhs],min(1.e-10,tol*.01),maxiter,0)
        check,=reference.cphf_Hx([rotation])
        cphf_error=float(np.max(abs(check.np-rhs.np),initial=0.))
    finally:
        if created:
            new_jk=reference.jk()
            reference.set_jk(old_jk)
            if new_jk is not None: new_jk.finalize()
    x=Co@rotation.np@Cv.T
    Duv=seed+x+x.T
    Duv=(Duv+Duv.T)/2
    Fuv=fock_mixed_source+fock_action(Duv)
    full_density=density._replace(uv=Duv)
    identity=mixed_product('mi,ij,jn->mn',full_density,overlap,full_density).uv-Duv
    stationarity=residual(full_density,MixedBlock(F,fock_u,fock_v,Fuv))
    ie=float(np.max(abs(identity),initial=0.))
    se=float(np.max(abs(stationarity),initial=0.))
    if not np.isfinite([ie,se,cphf_error]).all() or max(ie,se,cphf_error)>tol:
        raise RuntimeError(f'Mixed RHF constraints exceed tolerance: idempotency {ie:.3e}, '
                           f'stationarity {se:.3e}, CPHF {cphf_error:.3e}.')
    return RHFMixedResponse(Duv,Fuv,ie,se,cphf_error)


def mixed_occupied_partition(reference_occupied, frozen_count, density, overlap, fock):
    """Frozen and active occupied densities from the spectral occupied Fock split.

    The full occupied frame is polar transported from its reference. Frozen
    orbitals are the lowest occupied Fock eigenspace, with a resolved boundary.
    No individual canonical orbital eigenvector derivatives are required.
    """
    from .dlpno_triples_curvature import _mixed_projector
    no=reference_occupied.shape[1]
    if not isinstance(frozen_count,(int,np.integer)) or not 0<=frozen_count<=no:
        raise ValueError('Invalid frozen occupied rank.')
    zero=MixedBlock(*(np.zeros_like(density.value) for _ in range(4)))
    if frozen_count==0: return zero,density
    if frozen_count==no: return density,zero
    fixed=overlap.value@reference_occupied
    Y=MixedBlock(*(d@fixed for d in density))
    gram=mixed_product('ma,mn,nb->ab',Y,overlap,Y)
    eye=np.eye(no)
    if not np.allclose(gram.value,eye,rtol=0,atol=2.e-8):
        raise ValueError('Invalid reference occupied frame for the frozen-core partition.')
    norm=MixedBlock(eye,-.5*gram.u,-.5*gram.v,
                   -.5*gram.uv+.375*(gram.u@gram.v+gram.v@gram.u))
    occupied=mixed_product('ma,ab->mb',Y,norm)
    foo=mixed_product('ma,mn,nb->ab',occupied,fock,occupied)
    projector=_mixed_projector(MixedBlock(*(-x for x in foo)),frozen_count)
    frozen=mixed_product('ma,ab,nb->mn',occupied,projector,occupied)
    active=MixedBlock(*(d-c for d,c in zip(density,frozen)))
    return frozen,active
