"""Mixed quadratic Boys/Mulliken-PM localization on a fixed stationary branch."""
import numpy as np
from .dlpno_triples_curvature import MixedBlock,mixed_product
from .dlpno_orbital import _solve_localization_rotations


def quadratic_localization_mixed(reference, active_density, overlap, operators, tol=2.e-8):
    """Localize a mixed active occupied projector using the existing rotation Hessian.

    operators supplies AO MixedBlocks (three dipoles for Boys or Mulliken atom
    populations for PM). All orders of their nuclear response must be supplied.
    Singular or non-maximal reference localization branches are rejected.
    """
    operators=tuple(operators)
    if not operators: raise ValueError('Localization needs at least one operator.')
    fixed=overlap.value@reference
    Y=MixedBlock(*(d@fixed for d in active_density))
    gram=mixed_product('ma,mn,nb->ab',Y,overlap,Y)
    n=reference.shape[1]; eye=np.eye(n)
    if not np.allclose(gram.value,eye,atol=2.e-8,rtol=0):
        raise ValueError('Reference LMOs do not span the supplied orthonormal active space.')
    norm=MixedBlock(eye,-.5*gram.u,-.5*gram.v,
                   -.5*gram.uv+.375*(gram.u@gram.v+gram.v@gram.u))
    parent=mixed_product('ma,ab->mb',Y,norm)
    i,j=np.triu_indices(n,1)
    def projected(frame):
        return [mixed_product('ma,mn,nb->ab',frame,a,frame) for a in operators]
    def gradients(blocks):
        result=np.zeros((4,len(i)))
        for a in blocks:
            d=[x.diagonal()[i]-x.diagonal()[j] for x in a]
            r=[x[i,j] for x in a]
            result[0]+=4*d[0]*r[0]
            result[1]+=4*(d[1]*r[0]+d[0]*r[1])
            result[2]+=4*(d[2]*r[0]+d[0]*r[2])
            result[3]+=4*(d[3]*r[0]+d[1]*r[2]+d[2]*r[1]+d[0]*r[3])
        return result
    base=projected(parent); g=gradients(base)
    if np.max(abs(g[0]),initial=0.)>tol:
        raise ValueError('Mixed localization requires a stationary reference.')
    matrices=[a.value for a in base]
    def rotation(source):
        k=_solve_localization_rotations(matrices,2,-source,tol,require_unique=True)
        K=np.zeros((n,n)); K[j,i],K[i,j]=k,-k
        return K
    Ku,Kv=rotation(g[1]),rotation(g[2])
    trial=MixedBlock(parent.value,parent.u+parent.value@Ku,parent.v+parent.value@Kv,
                     parent.uv+parent.u@Kv+parent.v@Ku+parent.value@(.5*(Ku@Kv+Kv@Ku)))
    Kuv=rotation(gradients(projected(trial))[3])
    result=trial._replace(uv=trial.uv+parent.value@Kuv)
    residual=gradients(projected(result))
    if not np.isfinite(residual).all() or np.max(abs(residual),initial=0.)>10*tol:
        raise RuntimeError('Mixed localization stationarity check failed.')
    ortho=mixed_product('ma,mn,nb->ab',result,overlap,result)
    if max(np.max(abs(x),initial=0.) for x in ortho[1:])>10*tol:
        raise RuntimeError('Mixed localization orthonormality check failed.')
    return result


def mulliken_operators(overlap, centers):
    """Quadratic Mulliken-PM operators with fixed atom-to-AO assignment."""
    centers=np.asarray(centers)
    for atom in np.unique(centers):
        mask=(centers==atom).astype(float)
        yield MixedBlock(*(.5*(mask[:,None]*s+s*mask[None,:]) for s in overlap))
