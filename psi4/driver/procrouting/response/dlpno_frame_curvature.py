"""Mixed PAO and virtual-parent response from occupied-projector actions.

The action includes all occupied orbitals, including frozen core. Mixed RHF
response must be supplied; this module does not solve second-order CPHF.
"""
import numpy as np
from scipy.linalg import lu_factor, lu_solve

from .dlpno_triples_curvature import MixedBlock, mixed_product


def occupied_projector_action(occupied, panel):
    """Apply D=C_occ C_occ^T and its mixed product rule without forming D."""
    return mixed_product('mi,ia->ma',occupied,
                         mixed_product('mi,ma->ia',occupied,panel))


def mixed_metric_solve(metric, panel, solve=None):
    """Differentiate S Y=B on a panel; optionally reuse a reference metric solve."""
    S,Su,Sv,Suv=metric
    if solve is None:
        factor=lu_factor(S)
        solve=lambda rhs:lu_solve(factor,rhs)
    Y=solve(panel.value)
    Yu=solve(panel.u-Su@Y)
    Yv=solve(panel.v-Sv@Y)
    Yuv=solve(panel.uv-Suv@Y-Su@Yv-Sv@Yu)
    return MixedBlock(Y,Yu,Yv,Yuv)


def mixed_virtual_parent(occupied_action, overlap, reference_frame, metric_solver=None):
    """Virtual frame in the fixed-reference polar gauge.

    occupied_action(panel) supplies D times a MixedBlock panel with its total
    derivatives. reference_frame must be orthonormal in the reference metric
    and span the desired invariant virtual space. A general truncated parent
    needs its own selection response and cannot be supplied here.
    """
    B=overlap.value@reference_frame
    zero=np.zeros_like(B)
    panel=MixedBlock(B,zero,zero,zero)
    inverse=mixed_metric_solve(overlap,panel,metric_solver)
    projected=occupied_action(panel)
    Y=MixedBlock(*(a-b for a,b in zip(inverse,projected)))
    gram=mixed_product('ma,mn,nb->ab',Y,overlap,Y)
    eye=np.eye(reference_frame.shape[1])
    if (not np.allclose(Y.value,reference_frame,rtol=0,atol=2.e-8) or
            not np.allclose(gram.value,eye,rtol=0,atol=2.e-8)):
        raise ValueError('Reference virtual frame is not an orthonormal unoccupied frame.')
    norm=MixedBlock(eye,-.5*gram.u,-.5*gram.v,
                   -.5*gram.uv+.375*(gram.u@gram.v+gram.v@gram.u))
    return mixed_product('ma,ab->mb',Y,norm)


def mixed_paos(occupied_action, overlap, domain):
    """Raw and normalized PAOs for fixed AO indices, including both metric legs.

    Matches the native fixed zero-column mask (norm squared > 1e-20).
    No dense AO identity or occupied density is constructed.
    """
    domain=np.asarray(domain)
    n=overlap.value.shape[0]
    if domain.ndim!=1 or domain.dtype.kind not in 'iu' or np.any(domain<0) or np.any(domain>=n):
        raise ValueError('Expected valid AO indices for the fixed PAO domain.')
    basis=np.zeros((n,len(domain)))
    basis[domain,np.arange(len(domain))]=1.
    projection=occupied_action(overlap.slice((slice(None),domain)))
    raw=MixedBlock(basis-projection.value,*(-x for x in projection[1:]))
    norms=mixed_product('ma,mn,na->a',raw,overlap,raw)
    active=norms.value>1.e-20
    scale=[np.zeros_like(norms.value) for _ in range(4)]
    q,qu,qv,quv=(x[active] for x in norms)
    scale[0][active]=q**(-.5)
    scale[1][active]=-.5*q**(-1.5)*qu
    scale[2][active]=-.5*q**(-1.5)*qv
    scale[3][active]=.75*q**(-2.5)*qu*qv-.5*q**(-1.5)*quv
    normalized=mixed_product('ma,a->ma',raw,MixedBlock(*scale))
    if not all(np.isfinite(x).all() for x in (*raw,*normalized)):
        raise ValueError('Nonfinite mixed PAO response.')
    return raw,normalized
