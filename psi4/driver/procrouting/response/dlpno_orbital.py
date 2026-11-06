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
        if rhs.size <= 64:
            # Resolve near-null rotations in small occupied spaces. Require
            # the residual check below even when a gauge mode is discarded.
            dense = np.column_stack([action(v) for v in np.eye(rhs.size)])
            z = np.linalg.lstsq(dense, rhs, rcond=None)[0]
            info = 0
        else:
            z, info = minres(H, rhs, rtol=max(np.finfo(float).eps, tol * 1.e-3))
        if info or np.linalg.norm(action(z)-rhs) > tol*max(1., np.linalg.norm(rhs)):
            raise RuntimeError(f"Localization adjoint residual {np.linalg.norm(action(z)-rhs):.3e} exceeds tolerance.")
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
    L = np.asarray(localized)
    operators = [np.asarray(A) for A in ao_operators]
    return _localized_source(wfn, L, lambda: (A @ L for A in operators), power, panel_width, tol)


def _localized_source(wfn, L, actions, power, panel_width, tol):
    from psi4 import core
    if panel_width < 1:
        raise ValueError("Expected positive orbital panel width.")
    source = np.empty_like(L)
    for first in range(0, L.shape[1], panel_width):
        stop = min(first+panel_width, L.shape[1])
        panel = np.zeros((L.shape[1], stop-first))
        panel[np.arange(first, stop), np.arange(stop-first)] = 1.
        action = core.dlpno_orbital_response_action(wfn, 'ACTIVE', core.Matrix.from_array(panel))
        source[:, first:stop] = action.np
    z, bars = localization_adjoint([L.T @ AL for AL in actions()], power, L, source, tol)
    for AL, bar in zip(actions(), bars):
        source += 2 * AL @ bar
    return z, source, bars


def molecular_rhf_source(wfn, panel_width=16, tol=1.e-9):
    """Assemble dL/dx, with dCocc = -Cvir x.T, for Boys, Mulliken PM or full-rank IBO.

    Return the canonical source and the localized operator bars for subsequent
    explicit AO integral differentiation. The SCF JK settings must match the
    reference calculation.
    """
    from psi4 import core
    ref = wfn.reference_wavefunction()
    if not isinstance(ref, core.RHF) or ref.nirrep() != 1:
        raise ValueError("Expected a C1 RHF DLPNO gradient-model wavefunction.")
    basis = ref.basisset()
    S = ref.S().np
    L = wfn.variable('DLPNO RESPONSE ACTIVE LMOS').np
    power = 2
    iao_data = None
    if wfn.variable('DLPNO RESPONSE BOYS'):
        dipoles = core.MintsHelper(basis).ao_dipole()
        def actions():
            return (m.np @ L for m in dipoles)
    elif wfn.variable('DLPNO RESPONSE MULLIKEN PM'):
        centers = np.array([basis.function_to_center(mu) for mu in range(basis.nbf())])
        SL = S @ L
        def actions():
            for atom in range(ref.molecule().natom()):
                mask = (centers == atom)[:, None]
                yield .5*(mask*SL + S @ (mask*L))
    elif wfn.has_variable('DLPNO RESPONSE IAOS'):
        minimal = wfn.get_basisset("MINAO")
        idx = wfn.variable('DLPNO RESPONSE IAO INDICES').np[:, 0].astype(int)
        mints = core.MintsHelper(basis)
        cross = mints.ao_overlap(basis, minimal).np[:, idx]
        metric = mints.ao_overlap(minimal, minimal).np[np.ix_(idx, idx)]
        power = int(wfn.variable('DLPNO RESPONSE IAO POWER'))
        iaos = wfn.variable('DLPNO RESPONSE IAOS').np
        cutoff = wfn.variable('DLPNO RESPONSE IAO CONDITION')
        centers = np.array([minimal.function_to_center(int(mu)) for mu in idx])
        groups = [np.flatnonzero(centers == a) for a in np.unique(centers)]
        projected = S @ iaos
        def actions():
            for group in groups:
                orbitals = projected[:, group]
                yield orbitals @ (orbitals.T @ L)
        iao_data = cross, metric, iaos, cutoff, groups
    else:
        raise ValueError("Expected a Boys, Mulliken PM or IBO gradient model.")
    _, active_bar, bars = _localized_source(wfn, L, actions, power, panel_width, tol)
    C = ref.Ca_subset('AO', 'OCC').np
    V = ref.Ca_subset('AO', 'VIR').np
    frozen = ref.Ca_subset('AO', 'FROZEN_OCC').np
    nc = frozen.shape[1]
    active = C[:, nc:]
    U = active.T @ S @ L
    active_bar = active_bar @ U.T
    mixed = np.zeros_like(S)
    if nc:
        core_bar = core.dlpno_orbital_response_action(
            wfn, 'FROZEN', core.Matrix.from_array(np.eye(nc))).np
        # dCc = Ca k, dCa = -Cc k.T; enforce F_ac = 0.
        eps = ref.epsilon_a().np
        gap = eps[nc:C.shape[1], None] - eps[None, :nc]
        if np.any(np.abs(gap) < 1.e-10):
            raise ValueError("Unresolved frozen-active orbital-energy gap.")
        kbar = active.T @ core_bar - active_bar.T @ frozen
        mixed = -(active @ (kbar / gap) @ frozen.T)
        mixed = .5 * (mixed + mixed.T)
        occupied_bar = np.column_stack((core_bar, active_bar))
    else:
        occupied_bar = active_bar
    if iao_data is not None:
        cross, metric, iaos, cutoff, groups = iao_data
        iao_bar = np.zeros_like(iaos)
        for group, bar in zip(groups, bars):
            iao_bar[:, group] = 2*S @ L @ bar @ (L.T @ S @ iaos[:, group])
        occupied_bar += _iao_occupied_bar(C, S, cross, metric, cutoff, iaos, iao_bar)
    source = -occupied_bar.T @ V
    # Adjoint of dF = 2J[dD] - K[dD], D = Cocc Cocc.T.
    jk = core.JK.build(basis, aux=ref.get_basisset('DF_BASIS_SCF'), do_wK=False,
                       memory=int(core.get_memory() * .125 * .8))
    jk.initialize()
    try:
        for first in range(0, S.shape[0], panel_width):
            stop = min(first + panel_width, S.shape[0])
            E = np.zeros((S.shape[0], stop-first))
            E[np.arange(first,stop), np.arange(stop-first)] = 1.
            right = core.Matrix.from_array(E)
            Fbar = core.dlpno_orbital_response_action(wfn, 'FOCK', right)
            left = core.Matrix.from_array(Fbar.np + mixed[:, first:stop])
            jk.C_left_add(left)
            jk.C_right_add(right)
            jk.compute()
            potential = 2*jk.J()[0].np - jk.K()[0].np
            source -= C.T @ (potential + potential.T) @ V
            jk.C_clear()
    finally:
        jk.finalize()
    return source, bars

def _iao_occupied_bar(C, S, cross, metric, cutoff, iaos, iao_bar):
    """Differentiate full-rank Knizia IAOs at fixed AO and minimal-basis metrics."""
    def inverse(A):
        e, U = np.linalg.eigh(A)
        if e[0] <= cutoff * e[-1]:
            raise ValueError("Rank-reduced IAO metric requires a fixed-rank spectral response.")
        return (U/e) @ U.T

    Sinv = inverse(S)
    mapping = Sinv @ cross @ inverse(metric) @ cross.T
    X = mapping @ C
    gram_inv = inverse(X.T @ S @ X)
    D = C @ C.T
    Dt = X @ gram_inv @ X.T
    raw = (Sinv + 2*D @ S @ Dt - D - Dt) @ cross
    e, U = np.linalg.eigh(raw.T @ S @ raw)
    if e[0] <= cutoff * e[-1]:
        raise ValueError("Rank-reduced final IAO metric requires a spectral response.")
    roots = np.sqrt(e)
    N = (U/roots) @ U.T
    if not np.allclose(raw @ N, iaos, atol=1.e-9, rtol=1.e-9):
        raise ValueError("Reconstructed IAOs disagree with the forward localizer.")
    # Divided difference of x^(-1/2), valid also for repeated eigenvalues.
    divided = -1/(roots[:,None]*roots[None,:]*(roots[:,None]+roots[None,:]))
    Nbar = raw.T @ iao_bar
    metric_bar = U @ (divided*(U.T @ ((Nbar + Nbar.T)*.5) @ U)) @ U.T
    Pbar = (iao_bar @ N + 2*S @ raw @ metric_bar) @ cross.T
    Dbar = 2*Pbar @ Dt @ S - Pbar
    Dtbar = 2*S @ D @ Pbar - Pbar
    Xbar = (Dtbar + Dtbar.T) @ X @ gram_inv
    return (Dbar + Dbar.T) @ C + mapping.T @ (Xbar - S @ Dt @ Xbar)
