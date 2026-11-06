"""Fixed-geometry orbital adjoints for the RHF DLPNO gradient model."""
import numpy as np
from scipy.sparse.linalg import LinearOperator, minres


def localization_adjoint(operators, power, orbitals, orbital_source, tol=1.e-9):
    """Return occupied rotation multipliers and sources on localized population operators.

    operators are symmetric L.T @ A_K @ L in the final active LMO frame.
    A positive angle has K[j,i]=+1 for i<j. The source is d(correlation L)/dL.
    """
    L = np.asarray(orbitals)
    n = L.shape[1]
    i, j = np.triu_indices(n, 1)
    operators = [np.asarray(A) for A in operators]
    if power not in (2, 4) or not operators or any(
        A.shape != (n, n) or not np.allclose(A, A.T) for A in operators
    ):
        raise ValueError("Expected symmetric localized operators and power 2 or 4.")
    def generator(v):
        K = np.zeros((n, n))
        K[j, i], K[i, j] = v, -v
        return K

    def weights(A):
        d = A.diagonal()
        return power * d**(power-1), power*(power-1)*d**(power-2)

    def action(v):
        K = generator(v)
        result = np.zeros_like(v)
        for A in operators:
            w, wp = weights(A)
            dA = A @ K - K @ A
            dw = wp * dA.diagonal()
            # Exponential-coordinate Hessian, including the moving-frame term.
            grad = A * (w[:, None] - w[None, :])
            result += 2*((dw[i]-dw[j])*A[i,j] + (w[i]-w[j])*dA[i,j])
            result += (K @ grad - grad @ K)[i,j]
        return result

    M = L.T @ np.asarray(orbital_source)
    rhs = M[i,j] - M[j,i]
    z = np.zeros_like(rhs)
    if np.linalg.norm(rhs):
        H = LinearOperator((rhs.size, rhs.size), matvec=action, dtype=float)
        z, info = minres(H, rhs, rtol=tol)
        if info or np.linalg.norm(action(z)-rhs) > tol*max(1., np.linalg.norm(rhs)):
            raise RuntimeError("Localization adjoint residual exceeds tolerance.")
    bars = []
    for A in operators:
        w, wp = weights(A)
        bar = np.zeros_like(A, dtype=float)
        bar[i,j] = bar[j,i] = z*(w[i]-w[j])
        delta = 2*z*A[i,j]
        np.add.at(bar, (i,i), delta*wp[i])
        np.add.at(bar, (j,j), -delta*wp[j])
        bars.append(bar)
    return z, bars


def constrained_rhf_adjoint(reference, source, tol=1.e-9, maxiter=100):
    """Solve H z = -source for an assembled canonical occupied-by-virtual RHS."""
    from psi4 import core
    if not isinstance(reference, core.RHF) or reference.nirrep() != 1:
        raise ValueError("Expected a C1 RHF reference.")
    source = np.asarray(source)
    if source.shape != (reference.nalpha(), reference.nmo()-reference.nalpha()):
        raise ValueError("Expected the complete canonical occupied-by-virtual source.")
    if not np.any(source):
        return np.zeros_like(source)
    rhs = core.Matrix.from_array(-source)
    z, = reference.cphf_solve([rhs], tol, maxiter, 0)
    hz, = reference.cphf_Hx([z])
    if not np.linalg.norm(hz.np+source) <= tol*max(1., np.linalg.norm(source)):
        raise RuntimeError("RHF adjoint residual exceeds tolerance.")
    return z.np.copy()


def dlpno_localization_response(wfn, localized, ao_operators, power, panel_width=16, tol=1.e-9):
    """Use the model's active source; return multipliers, relaxed L source and operator bars.

    Supply the actual forward AO operators (dipoles or population matrices).
    IAO operators must subsequently be differentiated through their construction.
    """
    from psi4 import core
    L = np.asarray(localized)
    if panel_width < 1:
        raise ValueError("Expected positive orbital panel width.")
    source = np.empty_like(L)
    for first in range(0, L.shape[1], panel_width):
        stop = min(first+panel_width, L.shape[1])
        panel = np.zeros((L.shape[1], stop-first))
        panel[np.arange(first, stop), np.arange(stop-first)] = 1.
        action = core.dlpno_orbital_response_action(wfn, 'ACTIVE', core.Matrix.from_array(panel))
        source[:, first:stop] = action.np
    operators = [np.asarray(A) for A in ao_operators]
    z, bars = localization_adjoint([L.T @ A @ L for A in operators], power, L, source, tol)
    for A, bar in zip(operators, bars):
        source += 2 * A @ L @ bar
    return z, source, bars
