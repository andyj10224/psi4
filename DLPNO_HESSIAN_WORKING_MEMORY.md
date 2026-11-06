# DLPNO Hessian Working Memory Bank

## Scope and verification policy

Derive the analytic nuclear Hessian of the same closed-shell RHF ordinary iterative DLPNO-CCSD(T) functional selected by the public gradient driver. Start with Boys, then Mulliken PM; exercise both frozen-core and all-electron references. Work in a fixed Cartesian frame, in hartree/bohr², on a smooth local branch. Keep discrete selections fixed while differentiating the moving electronic subspaces. Require a test for each block before using it as a reference for the next block. Keep canonical comparisons, local finite-cutoff checks, and scaling measurements separate. [Equation record](DLPNO_HESSIAN_EQUATIONS.tex)

## 2026-11-06 — First canonical Hessian column

**Implemented.** A standalone reference generator differences analytic canonical gradients in Psi4. Run the following from the repository root in a built Psi4 Python environment:

    python tests/dlpno-hessian-model/water_hvp.py

It uses DF-SCF/cc-pVDZ-JKFIT and DF-CCSD(T)/cc-pVDZ-RI, with cc-pVDZ orbitals, at one water geometry. The first probe displaces H1 along +z, the O–H1 bond; O and H2 stay fixed. The second probe moves O along +y to check reciprocity. Cartesian displacements are ±0.008, ±0.004, and ±0.002 bohr, without rotation or recentering. This geometry need not be stationary to test a Cartesian Hessian. [Verification script](tests/dlpno-hessian-model/water_hvp.py)

**Verified.** The extrapolated DF-RHF stretch column agreed with Psi4's analytic DF-RHF Hessian to 3.69e-9 in maximum absolute component. Every displaced CCSD(T) point had a triples correction exceeding 1e-3 hartree in magnitude. Both correlated products passed the translational sum check. The independently computed mixed elements differed by 5.32e-11 (frozen O 1s) and 4.74e-11 (all-electron). The largest difference between successive Richardson estimates for the stretch was 7.00e-10 across the two core choices. Differences and components below are in hartree/bohr².

| Core | Atom | (Hv)x | (Hv)y | (Hv)z |
|---|---|---:|---:|---:|
| Frozen O 1s | O | 0 | 0.028971128 | −0.555946126 |
| | H1 | 0 | −0.015913673 | 0.571083886 |
| | H2 | 0 | −0.013057455 | −0.015137761 |
| All-electron | O | 0 | 0.029003036 | −0.554947671 |
| | H1 | 0 | −0.015913543 | 0.570071950 |
| | H2 | 0 | −0.013089494 | −0.015124279 |

The x entries round to zero by planarity. These are canonical numerical targets; no DLPNO nuclear differences or analytic DLPNO Hessian were computed. No vibrational or scaling claim follows from this column.

**Next block.** Implement the analytic fixed-frame RHF orbital tangent for this displacement in the gradient driver's CPHF convention. Verify differentiated metric orthonormality, the Brillouin residual, the density response against the canonical occupied projector, and frozen–active partition response before propagating the tangent through Boys localization. Then add the PAO/PNO, LMP2, final-PNO, CCSD and iterative triples tangents. A zero-cutoff canonical comparison tests the full column; finite-cutoff projector terms will need tests with actual retained/discarded gaps. Append specific equations and check outcomes as each block is derived.

## 2026-11-06 — Native RHF nuclear orbital response

RSCFDeriv already builds the explicit Fock/overlap derivatives, metric-density transport, nuclear CPHF right-hand sides and occupied coefficient derivatives. The new core.rhf_occupied_tangent(reference, direction) shares that implementation with the RHF Hessian. It contracts the native Upi records with a natom-by-3 Cartesian direction and returns AO-by-occupied coefficient derivatives in the symmetric occupied gauge within each active or frozen block. For frozen core, it uses the existing Fpi and Qpi records to restore the first-order canonical frozen–active partition. It stops before the final Hessian contractions and restores the original reference JK handle. Currently it builds all Cartesian nuclear response batches even when one direction is requested. The normal RHF Hessian retains its original gauge.

**Checks.** For the water H1 stretch, both core choices gave maximum differentiated metric-orthonormality errors below 4.8e-16 and occupied–virtual Brillouin residuals below 8.1e-10 hartree/bohr. The AO occupied-density derivative agreed with differences of independent DF-RHF calculations to 6.1e-10; the isolated frozen O 1s contribution agreed to 1.2e-11 (AO coefficient density per bohr). The finite-difference checks used centered 0.002 and 0.001 bohr steps and Richardson extrapolation. [Test](tests/dlpno-hessian-model/rhf_tangent.py) · [Equations](DLPNO_HESSIAN_EQUATIONS.tex)

**Next.** Use the active occupied tangent and the derivative of the Boys dipole operators to form the nuclear localization right-hand side. Check the resulting active LMO projectors against gauge-aligned displaced Boys solutions before differentiating the PAO projector.

## 2026-11-06 — Boys nuclear localization tangent

Implemented boys_nuclear_tangent(wfn, direction) for an actual Boys-localized DLPNO model wavefunction. The active columns of the native DF-RHF nuclear response supply the transported LMO derivative. Adding the explicit nuclear AO dipole derivative gives the moving-operator source. The tangent and existing localization adjoint use the same orbital rotation Hessian action. The tangent requires stationarity and checks for a nonsingular maximum before returning AO-by-active-LMO coefficient derivatives. [Implementation](psi4/driver/procrouting/response/dlpno_orbital.py) · [Equations](DLPNO_HESSIAN_EQUATIONS.tex)

**Verification.** For water/cc-pVDZ H1 stretch, rerun DF-RHF and freely relocalize with Boys at ±0.004 and ±0.002 bohr. Transport the starting guess onto the same branch, match orbitals by absolute cross-basis overlap, then compare sign-invariant rank-one AO density factors. Largest Richardson component errors per bohr were 3.56e-10 (four active frozen-core LMOs) and 7.13e-10 (five all-electron LMOs). Differentiated localization stationarity residuals were 1.50e-13 and 6.86e-11 respectively; metric orthonormality checks passed. In an initial all-electron trial, a requested forward gradient tolerance of 1e-10 plateaued at 9.78e-9; the tangent rejected that reference. The verified test uses the forward 1e-8 localization gradient tolerance and checks convergence of every displaced localizer. [Regression](tests/dlpno-hessian-model/boys_tangent.py)

**Next.** Differentiate the raw PAO projector using both active LMO and frozen occupied derivatives and the moving AO overlap. As a check, the Boys internal rotation must cancel from the full occupied projector. The local occupied integral refits and pair-specific downstream contractions still need the individual LMO response.

## 2026-11-06 — PAO nuclear tangent and rotation diagnostic

Added pao_nuclear_tangent(wfn, direction) for the Boys model. It returns derivatives of the unnormalized PAO projector and normalized PAO coefficients, together with the maximum absolute difference between the localized and canonical raw-projector derivatives. Both routes include the frozen occupied columns and AO overlap derivative and share one nuclear RHF response evaluation. An inconsistency exceeding the scaled tolerance raises an error. Normalization follows the energy's fixed zero-column convention. [Code](psi4/driver/procrouting/response/dlpno_orbital.py) · [Equations](DLPNO_HESSIAN_EQUATIONS.tex)

**Water/cc-pVDZ H1 stretch.** The localized/canonical diagnostic was 5.28e-16 per bohr with frozen O 1s and 6.11e-16 all-electron. At ±0.004 and ±0.002 bohr, independently reconverged DF-RHF PAOs gave maximum Richardson differences of 9.51e-10 per bohr for raw PAOs and 2.78e-9 for normalized PAOs across both runs. The differentiated idempotency and column normalization checks passed. Injecting an additional unit active–active skew rotation left the raw tangent unchanged; injecting a symmetric term triggered the diagnostic in both core choices. The preceding Boys LMO test also passed. [Regression](tests/dlpno-hessian-model/pao_tangent.py)

**Next block.** Propagate the PAO coefficients into the pair-domain overlap, Fock and exchange derivatives for the first semicanonical MP2 pair density; test the retained–discarded PNO response with a nonzero truncation and separated boundary occupations.

## 2026-11-06 — First semicanonical MP2 PNO selection

Implemented pair-local tangents for the auxiliary-center occupied refit, transformed three-center integrals, pair RIFIT solve and exchange matrix, moving PAO metric and Fock block, semicanonical MP2 amplitudes, spin-adapted pair density, and the retained-subspace AO density. The kernel uses a parallel-transported parent frame, so it does not divide by energy differences within the virtual block. Only retained–discarded occupation gaps enter the projector derivative. The forward rank is an input, including the effects of minimum, trace, and energy criteria. The current kernel assumes the selected PAO pivot span has full rank. Module DEBUG saves one genuinely truncated off-diagonal pair's forward domain, parent, and retained coefficients for comparison. [Implementation](psi4/driver/procrouting/response/dlpno_orbital.py) · [Equation record](DLPNO_HESSIAN_EQUATIONS.tex)

**Verification.** Water/cc-pVDZ, H1 stretch, active LMO pair (0,1) in each core treatment, first occupation cutoff 1e-5. The reconstructed AO subspace agreed with the native first-stage retained coefficients within 2e-8 entrywise. Independently reconverged DF-RHF and Boys calculations at ±0.002 and ±0.001 bohr rebuilt the pair MP2 density and kept its reference rank and pivot span. Maximum Richardson differences for the derivative of the AO retained-subspace density were 1.55e-8 per bohr (frozen O 1s, rank 15/19, occupation gap 5.33e-7) and 2.12e-8 per bohr (all-electron, rank 9/19, gap 3.33e-9). The differentiated metric-projector identity passed. A separate check perturbed a shortened AO fit and verified the fitted three-center tangent. [Reproduction script](tests/dlpno-hessian-model/first_pno.py)

**Scope and next block.** This is a one-pair nuclear verification. The molecular test uses full AO/fitting domains and dense reference integral derivatives, while the production pair kernels accept local blocks. A screened molecular provider and batching over retained pairs remain to be connected. The next correlation equation is the first-space coupled LMP2 right tangent, including moving overlap maps between pairs, before the final PNO selection. No complete analytic Hessian or frequency is reported.

## 2026-11-06 — Coupled first-space LMP2 right-amplitude tangent

Added an opt-in retained-state path (DLPNO_GRADIENT_MODEL and DLPNO_KEEP_LMP2_RESPONSE). The Python coupled_lmp2_tangent entry point transports each actual first-PNO frame using the verified AO subspace-density derivative. Given converged first-space data, the relaxed AO metric/Fock and LMO Fock derivatives and a pair-exchange derivative provider, the native code forms the differentiated residual on the original sparse graph. Both overlap legs, the occupied Fock coupling, the diagonal occupied sum and the full moving virtual Fock matrix contribute. The existing production right-row action and DIIS/Jacobi linear solver solve for the unique-pair T derivatives. Reversed-pair orientations and symmetric diagonal pairs follow the original forward conventions. Retaining the extra state is disabled by default. [Equations](DLPNO_HESSIAN_EQUATIONS.tex) · [Native solver](psi4/src/psi4/dlpno/lmp2_response.cc)

**Independent verification.** Water/cc-pVDZ H1 stretch at a first occupation cutoff of 1e-5. A preliminary free re-selection changed a pair rank on displacement. The final test therefore fixes all forward ranks and Fock-screen decisions, reconverges DF-RHF/Boys, reconstructs both the moving PNOs and pair projections, and solves the displaced LMP2 equations with an independent GMRES residual. The AO amplitude tensors reproduce the native forward result at the origin. Richardson extrapolation from ±0.001 and ±0.0005 bohr checks every unique pair derivative. [Reproduction script](tests/dlpno-hessian-model/lmp2_tangent.py)

| Core | Independent pairs | Directed edges (rectangular) | Max pair RMS equation residual | Max AO amplitude derivative error per bohr |
| --- | ---: | ---: | ---: | ---: |
| Frozen O 1s | 10 | 60 (18) | 2.21e-13 | 1.31e-10 |
| All-electron | 15 | 120 (74) | 7.28e-13 | 2.00e-10 |

The test uses full AO/fitting domains to generate molecular inputs and genuinely truncated first PNO spaces. The native solver follows the retained graph; screened molecular input generation and all-direction batching remain open. The gradient-model regression and ordinary DLPNO energy smoke test passed. Next propagate T^v into the LMP2 pair density and the final retained CCSD PNO projector, accounting for motion of the first-PNO parent itself.

## 2026-11-06 — Final CCSD PNO subspace response

Added final_pno_tangent(wfn, lmp2) for all retained unique pairs. The native accessor supplies the final semicanonical frames and therefore the actual ranks chosen by the energy's trace, energy, occupation and minimum-size tests. The same analytic spectral response kernel serves both selections. In the final stage its input is the converged LMP2 amplitude tangent in the moving first-stage frame. The result is the AO contravariant retained-space density and its derivative. It is independent of the final internal semicanonical rotation. [Implementation](psi4/driver/procrouting/response/dlpno_orbital.py) · [Equation record](DLPNO_HESSIAN_EQUATIONS.tex)

**Verified.** Water/cc-pVDZ H1 stretch with first/final occupation cutoffs 1e-5/1e-4 and full AO/fitting domains for the molecular verification inputs. The reconstructed zero-displacement spaces match the native final frames. Independent displaced DF-RHF/Boys and GMRES LMP2 solutions were used to reconstruct all final spaces at their reference ranks. Richardson estimates use ±0.001, ±0.0005, ±0.00025 and ±0.000125 bohr; the table reports the finest estimate. Scaled errors divide each pair's maximum error by max(1, maximum absolute analytic component), then take the maximum over pairs.

| Core | Final truncations | Smallest retained–discarded gap | Max AO derivative error per bohr | Max scaled error |
| --- | ---: | ---: | ---: | ---: |
| Frozen O 1s | 10/10 | 2.19e-7 | 1.23e-7 | 3.24e-8 |
| All-electron | 12/15 | 2.73e-7 | 8.85e-7 | 5.53e-8 |

The coarser estimates were less accurate, and the all-electron comparison leveled off near 9e-7 absolute. Record this numerical resolution when using the results as a reference. Differentiated AO metric idempotency, the spectral commutator, and invariance to compensating rotations of the parent frame and LMP2 amplitudes passed. The three all-electron pairs retaining full parent rank still include the moving-parent contribution. [Water regression](tests/dlpno-hessian-model/lmp2_tangent.py)

**Next.** Propagate these final subspaces into the projected CCSD integrals and interpair overlap maps in a consistent moving frame, then assemble and test the first-order singles/doubles residual source. These subspace derivatives are not a complete nuclear Hessian. Screened molecular input delivery is still open.
