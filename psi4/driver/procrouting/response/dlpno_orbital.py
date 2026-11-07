"""Fixed-geometry orbital adjoints for the RHF DLPNO gradient model."""
from typing import NamedTuple

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh, minres


def _solve_localization_rotations(operators, power, rhs, tol, require_unique=False):
    n = operators[0].shape[0]
    i, j = np.triu_indices(n, 1)
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

    z = np.zeros_like(rhs)
    if rhs.size and (require_unique or np.linalg.norm(rhs)):
        H = LinearOperator((rhs.size, rhs.size), matvec=action, dtype=float)
        if rhs.size <= 64:
            # Resolve near-null rotations in small occupied spaces. Require
            # the residual check below even when a gauge mode is discarded.
            dense = np.column_stack([action(v) for v in np.eye(rhs.size)])
            z, _, rank, _ = np.linalg.lstsq(dense, rhs, rcond=None)
            if require_unique and (rank < rhs.size or np.linalg.eigvalsh(dense)[-1] >= 0.):
                raise RuntimeError("Boys rotation tangent requires a nonsingular localization maximum.")
            info = 0
        else:
            if require_unique:
                largest = eigsh(H, k=1, which="LA", return_eigenvectors=False, tol=tol)[0]
                scale = max(1., sum(np.linalg.norm(A)**power for A in operators))
                if largest >= -np.finfo(float).eps * rhs.size * scale:
                    raise RuntimeError("Boys rotation tangent requires a nonsingular localization maximum.")
            z, info = minres(H, rhs, rtol=max(np.finfo(float).eps, tol * 1.e-3))
        if info or np.linalg.norm(action(z)-rhs) > tol*max(1., np.linalg.norm(rhs)):
            label = "Boys tangent" if require_unique else "Localization adjoint"
            raise RuntimeError(f"{label} residual {np.linalg.norm(action(z)-rhs):.3e} exceeds tolerance.")
    return z


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
    M = L.T @ np.asarray(orbital_source)
    rhs = M[i,j] - M[j,i]
    z = _solve_localization_rotations(operators, power, rhs, tol)
    bars = []
    for A in operators:
        diag = A.diagonal()
        w, wp = power*diag**(power-1), power*(power-1)*diag**(power-2)
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
    source, bars, _, _ = _molecular_sources(wfn, panel_width, tol)
    return source, bars


def _molecular_sources(wfn, panel_width, tol):
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
    return source, bars, occupied_bar, mixed

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


def _df_fock_panel(mints, occupied, unit, response):
    """Contract sym(response @ unit.T) with the fixed-density DF Fock derivative."""
    from psi4 import core

    def jk_gradient(factors):
        return core.rhf_df_jk_energy_gradient(mints, core.Matrix.from_array(factors)).np.copy()

    plus = (unit + response)*.5
    minus = (unit - response)*.5
    # The common quadratic reference term cancels between the two differences.
    two = (jk_gradient(np.column_stack((occupied, plus))) - jk_gradient(plus)
           - jk_gradient(np.column_stack((occupied, minus))) + jk_gradient(minus))*.5
    density = response @ unit.T
    one = mints.core_hamiltonian_grad(core.Matrix.from_array((density + density.T)*.5)).np
    return one + two


def cartesian_orbital_terms(wfn, fock_derivatives=None, panel_width=16, tol=1.e-9):
    """Contract Cartesian sources for Boys/PM; yield one atom (three scalars) at a time.

    By default the SCF gradient backend contracts each panel directly. An
    optional fock_derivatives(atom) returns three AO Fock derivatives at fixed AO
    density, including the derivative of the *reference* Coulomb/exchange fit.
    """
    from psi4 import core
    if not (wfn.variable('DLPNO RESPONSE BOYS') or wfn.variable('DLPNO RESPONSE MULLIKEN PM')):
        raise NotImplementedError("Explicit primary/minimal overlap derivatives are needed for IBO.")
    rhs, bars, occupied_bar, mixed = _molecular_sources(wfn, panel_width, tol)
    ref = wfn.reference_wavefunction()
    z = constrained_rhf_adjoint(ref, rhs, tol)
    C = ref.Ca_subset('AO', 'OCC').np
    V = ref.Ca_subset('AO', 'VIR').np
    L = wfn.variable('DLPNO RESPONSE ACTIVE LMOS').np
    eps = ref.epsilon_a().np[:C.shape[1]]
    D = C @ C.T
    # Core-active constraints and occupied-virtual Brillouin multiplier.
    Z = C @ z @ V.T
    fextra = mixed + .5*(Z + Z.T)
    metric = -.5*C @ (occupied_bar.T @ C) @ C.T
    metric -= (C*eps) @ z @ V.T
    F = ref.Fa().np
    metric -= .5*(D @ F @ mixed + mixed @ F @ D)
    basis = ref.basisset()
    mints = core.MintsHelper(ref, 0)
    # The density-transport term needs only C.T @ G[Fbar] @ C.
    potential = np.zeros((C.shape[1], C.shape[1]))
    fock_gradient = np.zeros((ref.molecule().natom(), 3))
    jk = core.JK.build(basis, aux=ref.get_basisset('DF_BASIS_SCF'), do_wK=False,
                       memory=int(core.get_memory()*.1))
    # These covectors do not depend on the atom whose integral derivative is
    # contracted. Build each panel once and keep only two O(nao^2) arrays.
    fock_bar = np.empty_like(D)
    overlap_bar = np.empty_like(D)
    jk.initialize()
    try:
        Sdim = basis.nbf()
        for first in range(0, Sdim, panel_width):
            stop = min(first + panel_width, Sdim)
            E = np.zeros((Sdim, stop-first))
            E[np.arange(first,stop), np.arange(stop-first)] = 1.
            right = core.Matrix.from_array(E)
            raw = core.dlpno_orbital_response_action(wfn, 'FOCK', right).np
            fock_bar[:, first:stop] = raw
            overlap_bar[:, first:stop] = core.dlpno_orbital_response_action(wfn, 'OVERLAP', right).np
            response = raw + fextra[:, first:stop]
            jk.C_left_add(core.Matrix.from_array(response))
            jk.C_right_add(right)
            jk.compute()
            potential += C.T @ (2*jk.J()[0].np - jk.K()[0].np) @ C
            jk.C_clear()
    finally:
        jk.finalize()
    if fock_derivatives is None:
        for first in range(0, Sdim, panel_width):
            stop = min(first + panel_width, Sdim)
            E = np.eye(Sdim, stop-first, k=-first)
            response = fock_bar[:,first:stop] + fextra[:,first:stop]
            fock_gradient += _df_fock_panel(mints, C, E, response)
    metric -= C @ potential @ C.T
    boys = bool(wfn.variable('DLPNO RESPONSE BOYS'))
    if not boys:
        centers = np.array([basis.function_to_center(mu) for mu in range(basis.nbf())])
        for atom, bar in enumerate(bars):
            metric += (centers == atom)[:,None] * (L @ bar @ L.T)
    for atom in range(ref.molecule().natom()):
        ds = [m.np for m in mints.ao_oei_deriv1('OVERLAP', atom)]
        result = fock_gradient[atom] + np.array([np.sum(metric*s) for s in ds])
        blocks = [('OVERLAP', ds)]
        if fock_derivatives is not None:
            df = np.asarray(fock_derivatives(atom))
            if df.shape != (3, Sdim, Sdim):
                raise ValueError("Expected three Cartesian AO Fock derivative matrices per atom.")
            result += [np.sum(fextra*f) for f in df]
            blocks.append(('FOCK', df))
        if boys:
            dip = mints.ao_elec_dip_deriv1(atom)
            for k,bar in enumerate(bars):
                for xyz in range(3):
                    result[xyz] += np.sum(bar*(L.T @ dip[3*k+xyz].np @ L))
        for block, directions in blocks:
            bar = overlap_bar if block == 'OVERLAP' else fock_bar
            for xyz in range(3):
                result[xyz] += np.sum(bar*directions[xyz])
        yield result


def assemble_cartesian_gradient(wfn, reference_gradient=None, fock_derivatives=None, panel_width=16, tol=1.e-9):
    """Assemble a gradient-model Cartesian array, including reference and DF correlation.

    Defaults to SCFGrad on the retained RHF reference. An optional reference_gradient
    must include the RHF electronic and nuclear gradients in the same orientation.
    """
    from psi4 import core
    if reference_gradient is None:
        reference_gradient = core.scfgrad(wfn.reference_wavefunction()).np
    refgrad = np.asarray(reference_gradient, dtype=float)
    shape = (wfn.molecule().natom(), 3)
    if refgrad.shape != shape:
        raise ValueError("Expected an RHF nuclear gradient with one Cartesian row per atom.")
    gradient = refgrad.copy()
    for atom, contribution in enumerate(cartesian_orbital_terms(wfn, fock_derivatives, panel_width, tol)):
        gradient[atom] += contribution
    for source in ('CCSD', '(T)', 'PNO PROJECTOR', 'PNO TRUNCATION', 'LMP2 RESPONSE', 'SC-LMP2 PAIRS'):
        for term in ('AUXILIARY METRIC', 'THREE-CENTER'):
            gradient += wfn.variable(f'DLPNO {source} {term} GRADIENT').np
    gradient += wfn.variable('DLPNO DIPOLE PAIRS EXPLICIT GRADIENT').np
    return gradient

def boys_nuclear_tangent(wfn, direction, tol=1.e-9):
    """AO active-LMO coefficient derivative in a stationary Boys frame."""
    v, dC = _boys_reference_tangent(wfn, direction)
    return _boys_lmo_tangent(wfn, v, dC, tol)


def _boys_reference_tangent(wfn, direction):
    from psi4 import core
    if not wfn.variable('DLPNO RESPONSE BOYS'):
        raise ValueError("Expected a Boys-localized RHF DLPNO gradient model.")
    ref = wfn.reference_wavefunction()
    v = np.asarray(direction, dtype=float)
    if v.shape != (ref.molecule().natom(), 3) or not np.isfinite(v).all():
        raise ValueError("Expected a finite Cartesian direction with one row per atom.")
    return v, core.rhf_occupied_tangent(ref, core.Matrix.from_array(v)).np


def _boys_lmo_tangent(wfn, v, dC, tol):
    from psi4 import core
    ref = wfn.reference_wavefunction()
    L = wfn.variable('DLPNO RESPONSE ACTIVE LMOS').np
    nc = ref.Ca_subset('AO', 'FROZEN_OCC').np.shape[1]
    C = ref.Ca_subset('AO', 'OCC').np[:,nc:]
    U = C.T @ ref.S().np @ L
    transported = dC[:,nc:] @ U
    mints = core.MintsHelper(ref)
    dip = [m.np for m in mints.ao_dipole()]
    ddip = [np.zeros_like(d) for d in dip]
    for atom in range(v.shape[0]):
        if np.any(v[atom]):
            derivatives = mints.ao_elec_dip_deriv1(atom)
            for k in range(3):
                for xyz in range(3):
                    ddip[k] += v[atom,xyz]*derivatives[3*k+xyz].np
    i,j = np.triu_indices(L.shape[1], 1)
    operators = []
    source = np.zeros(i.size)
    gradient = np.zeros(i.size)
    for d, dd in zip(dip, ddip):
        A = L.T @ d @ L
        operators.append(A)
        motion = transported.T @ d @ L
        dA = motion + motion.T + L.T @ dd @ L
        delta = A.diagonal()[i]-A.diagonal()[j]
        gradient += 4*delta*A[i,j]
        source += 4*((dA.diagonal()[i]-dA.diagonal()[j])*A[i,j] + delta*dA[i,j])
    if np.max(abs(gradient), initial=0.) > max(tol, core.get_option('DLPNO','LOCAL_GRADIENT_CONVERGENCE'))*1.01:
        raise RuntimeError("Boys tangent requires a stationary forward localization.")
    rotation = _solve_localization_rotations(operators, 2, -source, tol, require_unique=True)
    K = np.zeros((L.shape[1], L.shape[1]))
    K[j,i], K[i,j] = rotation, -rotation
    return transported + L @ K

class PAOTangent(NamedTuple):
    raw: np.ndarray
    normalized: np.ndarray
    rotation_error: float


def pao_nuclear_tangent(wfn, direction, tol=1.e-9):
    """Raw and column-normalized PAO coefficient derivatives for a Boys model.

    rotation_error is the largest component difference between the raw PAO
    tangent from moving localized orbitals and from canonical occupied orbitals.
    """
    from psi4 import core
    v, dC = _boys_reference_tangent(wfn, direction)
    dL = _boys_lmo_tangent(wfn, v, dC, tol)
    ref = wfn.reference_wavefunction()
    C = ref.Ca_subset('AO', 'OCC').np
    S = ref.S().np
    L = wfn.variable('DLPNO RESPONSE ACTIVE LMOS').np
    nc = ref.Ca_subset('AO', 'FROZEN_OCC').np.shape[1]
    O = np.column_stack((C[:,:nc], L))
    dO = np.column_stack((dC[:,:nc], dL))
    dS = np.zeros_like(S)
    mints = core.MintsHelper(ref)
    for atom in range(v.shape[0]):
        if np.any(v[atom]):
            for weight, derivative in zip(v[atom], mints.ao_oei_deriv1('OVERLAP', atom)):
                dS += weight * derivative.np

    def projector_tangent(occ, docc):
        return -(docc @ (occ.T @ S) + occ @ (docc.T @ S + occ.T @ dS))

    raw = np.eye(S.shape[0]) - O @ (O.T @ S)
    draw = projector_tangent(O, dO)
    canonical = projector_tangent(C, dC)
    error = float(np.max(np.abs(draw-canonical)))
    if not error <= tol * max(1., np.max(abs(canonical))):
        raise RuntimeError(f"PAO occupied-rotation cancellation error {error:.3e} exceeds tolerance.")

    Sraw = S @ raw
    norm2 = np.sum(raw*Sraw, axis=0)
    dnorm2 = 2*np.sum(draw*Sraw, axis=0) + np.sum(raw*(dS @ raw), axis=0)
    # Match the energy's fixed zero-column branch.
    keep = norm2 > 1.e-20
    scale = np.zeros_like(norm2)
    scale[keep] = 1./np.sqrt(norm2[keep])
    normalized = draw*scale - .5*raw*(dnorm2*scale**3)
    return PAOTangent(draw, normalized, error)


def pair_three_center_tangent(overlap, doverlap, lmo, dlmo, pao, dpao, left, right, eri, deri):
    """Transform one auxiliary-center block in its actual AO domains.

    eri and deri must contain the same retained shell pairs on both legs.
    The LMO is refitted in left; the PAO coefficients are sliced on right.
    """
    local = np.ix_(left, left)
    fitted = np.linalg.solve(overlap[local], overlap[left] @ lmo)
    dfitted = np.linalg.solve(overlap[local], doverlap[left] @ lmo + overlap[left] @ dlmo
                              - doverlap[local] @ fitted)
    n = pao[right]
    contracted = fitted @ eri
    q = contracted @ n
    dq = (dfitted @ eri + fitted @ deri) @ n + contracted @ dpao[right]
    return q, dq


def pair_exchange_tangent(metric, dmetric, qi, dqi, qj, dqj):
    """Exchange and derivative from the pair's local metric and refitted three-center blocks."""
    fitted = np.linalg.solve(metric, qi)
    dfitted = np.linalg.solve(metric, dqi - dmetric @ fitted)
    return fitted.T @ qj, dfitted.T @ qj + fitted.T @ dqj


class PNOSubspaceTangent(NamedTuple):
    occupations: np.ndarray
    boundary_gap: float
    retained_density: np.ndarray
    retained_density_tangent: np.ndarray


FirstPNOTangent = PNOSubspaceTangent


def first_pno_tangent(pao, dpao, overlap, doverlap, fock, dfock, exchange, dexchange,
                      occupied_sum, doccupied_sum, parent, retained, diagonal=False):
    """AO contravariant retained-subspace density and its directional derivative for one pair.

    pao contains the normalized AO columns in the pair domain. exchange and
    dexchange are in this PAO basis. parent is the forward semicanonical
    transform, including the selected Cholesky pivots; retained is the actual
    forward rank, including minimum, energy and trace selection criteria.
    """
    X = np.asarray(parent)
    n = X.shape[1]
    if not 0 < retained <= n:
        raise ValueError("Invalid first-PNO retained rank.")
    if np.count_nonzero(np.any(X != 0., axis=1)) != n:
        raise ValueError("The first-PNO tangent requires a full-rank selected pivot span.")
    Y = pao @ X
    if not np.allclose(Y.T @ overlap @ Y, np.eye(n), atol=1.e-9, rtol=0):
        raise ValueError("First-PNO parent disagrees with the PAO metric.")
    # Parallel transport the fixed pivot span; the virtual Fock need only be
    # diagonal at the base geometry. Within-parent rotations drop out of the result.
    motion = dpao @ X
    half = motion.T @ overlap @ Y
    gauge = -.5*(half + half.T + Y.T @ doverlap @ Y)
    dY = motion + Y @ gauge
    f = Y.T @ fock @ Y
    eps = f.diagonal()
    if not np.allclose(f, np.diag(eps), atol=1.e-8, rtol=0):
        raise ValueError("First-PNO reference parent must be semicanonical.")
    half = dY.T @ fock @ Y
    df = half + half.T + Y.T @ dfock @ Y
    k = X.T @ exchange @ X
    dk = X.T @ dexchange @ X + gauge.T @ k + k @ gauge
    denom = occupied_sum - eps[:,None] - eps[None,:]
    if np.min(abs(denom)) < 1.e-12:
        raise ValueError("Unresolved semicanonical MP2 denominator.")
    t = k / denom
    dt = (dk + df @ t + t @ df - doccupied_sum*t) / denom
    return _pno_selection_tangent(Y, dY, t, dt, retained, diagonal)


def _pno_selection_tangent(Y, dY, t, dt, retained, diagonal):
    """Retained AO density derivative in a moving orthonormal parent."""
    n = Y.shape[1]
    if (not 0 < retained <= n or dY.shape != Y.shape or t.shape != (n, n)
            or dt.shape != t.shape):
        raise ValueError("Invalid PNO selection rank or response block dimensions.")
    tt, dtt = 2*t - t.T, 2*dt - dt.T
    factor = .5 if diagonal else 1.
    density = factor*(tt @ t.T + tt.T @ t)
    ddensity = factor*(dtt @ t.T + tt @ dt.T + dtt.T @ t + tt.T @ dt)
    return _density_selection_tangent(Y,dY,density,ddensity,retained)


def _density_selection_tangent(Y,dY,density,ddensity,retained):
    """Spectral selection response shared by the moving PNO and TNO spaces."""
    n = density.shape[0]
    if not 0 < retained <= n:
        raise ValueError("Invalid retained natural-orbital rank.")
    occ, orbitals = np.linalg.eigh(density)
    occ, orbitals = occ[::-1], orbitals[:,::-1]
    keep, drop = orbitals[:,:retained], orbitals[:,retained:]
    proj = keep @ keep.T
    dproj = np.zeros_like(proj)
    gap = np.inf
    if retained < n:
        gaps = occ[None,:retained] - occ[retained:,None]
        gap = np.min(gaps)
        if gap <= 64*np.finfo(float).eps*max(np.max(abs(occ)), np.finfo(float).tiny):
            raise ValueError("Unresolved retained/discarded PNO occupation boundary.")
        mixing = (drop.T @ ddensity @ keep) / gaps
        block = drop @ mixing @ keep.T
        dproj = block + block.T
    half = dY @ proj @ Y.T
    return PNOSubspaceTangent(occ, gap, Y @ proj @ Y.T, half + half.T + Y @ dproj @ Y.T)


class LMP2Tangent(NamedTuple):
    frames: dict
    amplitudes: dict


def _transport_pno_frame(Y, selection, S, dS, ij):
    if selection.retained_density.shape != S.shape or selection.retained_density_tangent.shape != S.shape:
        raise ValueError(f"Invalid PNO response for {ij}.")
    if (abs(np.sum(selection.retained_density*S)-Y.shape[1]) > 2.e-7
            or not np.allclose(selection.retained_density @ S @ Y, Y, atol=2.e-8, rtol=0)):
        raise ValueError(f"Retained PNO space disagrees with the forward frame for {ij}.")
    dY = selection.retained_density_tangent @ S @ Y + .5*Y @ (Y.T @ dS @ Y)
    motion = Y.T @ S @ dY
    if not np.allclose(motion+motion.T+Y.T @ dS @ Y, 0., atol=2.e-8, rtol=0):
        raise ValueError(f"Inconsistent moving PNO metric for {ij}.")
    return dY


def coupled_lmp2_tangent(wfn, first_pno, exchange_derivative, doverlap, dfock, doccupied_fock):
    """Differentiate the converged first-space amplitudes on the retained native graph.

    first_pno[ij] is the first-PNO projector response in the AO basis for
    each unique retained pair. exchange_derivative(ij,Y,dY) supplies dK in
    this transported pair frame, including both frame derivatives. The
    occupied Fock derivative includes Boys orbital motion. Request
    DLPNO_KEEP_LMP2_RESPONSE before computing the gradient-model energy.
    """
    from psi4 import core
    S = wfn.reference_wavefunction().S().np
    frames = core.dlpno_lmp2_blocks(wfn, 'FRAME')
    if set(frames) != set(first_pno):
        raise ValueError("Expected one first-PNO response per unique retained LMP2 pair.")
    dy, dk = {}, {}
    for ij, block in frames.items():
        Y = block.np
        dY = _transport_pno_frame(Y, first_pno[ij], S, doverlap, ij)
        dy[ij] = core.Matrix.from_array(dY)
        dk[ij] = core.Matrix.from_array(exchange_derivative(ij,Y,dY))
    tangent = core.dlpno_lmp2_tangent(wfn, dy, dk, core.Matrix.from_array(doverlap),
                                      core.Matrix.from_array(dfock), core.Matrix.from_array(doccupied_fock))
    return LMP2Tangent({ij: block.np.copy() for ij, block in dy.items()},
                       {ij: block.np.copy() for ij, block in tangent.items()})

def final_pno_tangent(wfn, lmp2):
    """AO final CCSD subspace densities and tangents for all unique retained pairs.

    The LMP2 frame and amplitude derivatives must describe the same direction
    in the native first-stage frames. Read the actual final ranks from the
    energy calculation, including its trace, energy and minimum-rank criteria.
    """
    from psi4 import core
    frames = core.dlpno_lmp2_blocks(wfn, 'FRAME')
    amplitudes = core.dlpno_lmp2_blocks(wfn, 'AMPLITUDE')
    final = core.dlpno_lmp2_blocks(wfn, 'FINAL_FRAME')
    if frames.keys() != lmp2.frames.keys() or frames.keys() != lmp2.amplitudes.keys():
        raise ValueError("Expected an LMP2 response for every unique final PNO pair.")
    result = {}
    for ij, parent in frames.items():
        y = final[ij].np
        selection = _pno_selection_tangent(parent.np, lmp2.frames[ij], amplitudes[ij].np,
                                           lmp2.amplitudes[ij], y.shape[1], ij[0] == ij[1])
        if not np.allclose(selection.retained_density, y @ y.T, rtol=1.e-8, atol=2.e-8):
            raise ValueError(f"Reconstructed final PNO space disagrees with the forward pair {ij}.")
        result[ij] = selection
    return result

class CCSDPairFrameResponse(NamedTuple):
    frames: dict
    tangents: dict
    overlap: np.ndarray
    doverlap: np.ndarray

    def overlap_tangent(self, p, q):
        """One requested rectangular overlap; ij and ji share a virtual frame."""
        p, q = tuple(sorted(p)), tuple(sorted(q))
        yp, yq = self.frames[p], self.frames[q]
        return (self.tangents[p].T @ self.overlap @ yq + yp.T @ self.doverlap @ yq
                + yp.T @ self.overlap @ self.tangents[q])


def ccsd_pair_frames(wfn, final, doverlap):
    """Parallel-transport the native semicanonical CCSD frames."""
    from psi4 import core
    frames = {ij: m.np.copy() for ij,m in core.dlpno_lmp2_blocks(wfn,'FINAL_FRAME').items()}
    if frames.keys() != final.keys():
        raise ValueError("Expected one final PNO selection response per retained CCSD pair.")
    S = wfn.reference_wavefunction().S().np
    dy = {ij:_transport_pno_frame(y,final[ij],S,doverlap,ij) for ij,y in frames.items()}
    return CCSDPairFrameResponse(frames,dy,S,doverlap)


def ccsd_projected_fock_source(wfn, pair_frames, dlmo, dfock):
    """Contribution to dR/dv through bare OO/OV/VV Fock blocks, at fixed T and overlap maps.

    dlmo and dfock are the relaxed active LMO and AO Fock nuclear derivatives.
    Returns singles by occupied index and doubles by unique occupied pair.
    """
    from psi4 import core
    L = wfn.variable('DLPNO RESPONSE ACTIVE LMOS').np
    F = wfn.reference_wavefunction().Fa().np
    half = dlmo.T @ F @ L
    oo = half + half.T + L.T @ dfock @ L
    domains = core.dlpno_lmp2_blocks(wfn, "OCC_DOMAIN")
    ov, vv = {}, {}
    for ij,y in pair_frames.frames.items():
        dy = pair_frames.tangents[ij]
        dom = domains[ij].np[:,0].astype(int)
        lm = L[:,dom]
        ov[ij] = core.Matrix.from_array(dlmo[:,dom].T @ F @ y + lm.T @ dfock @ y + lm.T @ F @ dy)
        half = dy.T @ F @ y
        vv[ij] = core.Matrix.from_array(half + half.T + y.T @ dfock @ y)
    b1,b2 = core.dlpno_ccsd_fock_source(wfn,core.Matrix.from_array(oo),ov,vv)
    return [b.np.copy() for b in b1], {ij:b.np.copy() for ij,b in b2.items()}


class PairDFFactors(NamedTuple):
    oo: np.ndarray
    ov: np.ndarray
    vv: np.ndarray


def ccsd_pair_df_factors(metric, dmetric, raw, draw):
    """Whiten the three projected pair blocks and differentiate the same CCSD metric as Lambda."""
    from psi4 import core
    root, droot = core.dlpno_pair_metric_tangent(core.Matrix.from_array(metric), core.Matrix.from_array(dmetric))
    values, tangents = [], []
    for q,dq in zip(raw,draw):
        if q.shape != dq.shape or q.shape[0] != root.np.shape[0]:
            raise ValueError("Incompatible raw CCSD pair fitting blocks.")
        flat = q.reshape(q.shape[0],-1)
        values.append((root.np @ flat).reshape(q.shape))
        tangents.append((droot.np @ flat + root.np @ dq.reshape(flat.shape)).reshape(q.shape))
    return PairDFFactors(*values), PairDFFactors(*tangents)


def ccsd_direct_df_source(wfn, ij, pair_frames, factors, tangent):
    """Doubles source from Jiang Eqs. 75 (direct product) and 76 (virtual ladder).

    Supply pair-local metric-whitened factors in the native auxiliary/domain
    ordering; amplitudes are fixed. The projected singles move with S_p,mm.
    """
    from psi4 import core
    i,j = ij
    domain,singles,state = core.dlpno_ccsd_df_pair_state(wfn,i,j)
    tn, t = state['TN'].np, state['T2'].np
    dtn = np.stack([(pair_frames.overlap_tangent(ij,(m,m)) @ s.np).ravel()
                    for m,s in zip(domain,singles)])
    def dressed(m, side):
        row = domain.index(m)
        oi = state['O'+side].np
        vi = state['V'+side].np
        oo = factors.oo[:,row,:] + factors.ov @ tn[row]
        v = factors.ov[:,row,:] - oi @ tn + factors.vv @ tn[row]
        if not (np.allclose(oo,oi,rtol=1.e-8,atol=1.e-9)
                and np.allclose(v,vi,rtol=1.e-8,atol=1.e-9)):
            raise ValueError(f"DF factors disagree with native T1 dressing for {ij}.")
        doi = tangent.oo[:,row,:] + tangent.ov @ tn[row] + factors.ov @ dtn[row]
        dv = (tangent.ov[:,row,:] - doi @ tn - oi @ dtn +
              tangent.vv @ tn[row] + factors.vv @ dtn[row])
        return vi,dv
    vi,dvi = dressed(i,'I')
    vj,dvj = dressed(j,'J')
    v = factors.vv - tn.T @ factors.ov
    dv = tangent.vv - dtn.T @ factors.ov - tn.T @ tangent.ov
    ladder = dv @ t @ v.transpose(0,2,1) + v @ t @ dv.transpose(0,2,1)
    return dvi.T @ vj + vi.T @ dvj + ladder.sum(axis=0)
