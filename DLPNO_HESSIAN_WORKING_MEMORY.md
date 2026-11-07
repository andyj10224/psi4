# DLPNO Hessian Working Memory Bank

## Scope and verification policy

Derive the analytic nuclear Hessian of the same closed-shell RHF ordinary iterative DLPNO-CCSD(T) functional selected by the public gradient driver. Start with Boys, then Mulliken PM; exercise both frozen-core and all-electron references. Work in a fixed Cartesian frame, in hartree/bohr², on a smooth local branch. Keep discrete selections fixed while differentiating the moving electronic subspaces. Require a test for each block before using it as a reference for the next block. Keep canonical comparisons, local finite-cutoff checks, and scaling measurements separate. [Working equations (.txt)](DLPNO_HESSIAN_EQUATIONS.txt) · [Typesetting source (.tex)](DLPNO_HESSIAN_EQUATIONS.tex)

**Current checkpoint (7 November 2026).** Screened-domain validation does **not** yet qualify the Hessian for general production routing. A four-H2 chain passes sparse-graph CCSD residual/mixed-amplitude checks and two selected analytic-gradient checks with actual local AO/refit, PAO and auxiliary domains. Native selection fingerprints now verify the displaced branch. The mixed nuclear PNO/CCSD molecular providers still reject those screened domains; no screened nuclear Hessian was assembled. An existing synthetic PNO operator probe also exceeds its old tolerance. Analytic public Hessian routing remains disabled. The four complete full-AO/fitting-domain water Hessians from step 24 remain valid (324 independently evaluated ordered entries), including finite PNO spaces, both core choices, rigid-motion checks and full-matrix gradient differences. The target remains the public **gradient-model** functional; nonzero omitted estimates in the ordinary energy are a separate functional, not silently included in this Hessian. Historical “Next” statements below describe their earlier checkpoints.

## Complete Hessian milestone index across both chats

This index reconciles the Hessian phase of **“Implement DLPNO Gradients”** with the continuation in this chat. It was checked against every commit from `0af1a1eb` through `f18cf39a`, all 23 preceding chronological entries below, and the recovered older-chat context. Each previous Hessian commit has a corresponding ledger entry and working-equation section. Complete Cartesian validation is step 24 (`1b0c5517`); the screened-domain readiness audit is step 25. Earlier unsuccessful checks, numerical dependencies and their later resolutions are preserved in the chronological entries; they are not silently replaced by a passing result.

Equation-section numbers refer to both the `.txt` export and `.tex` source. The text export contains the entire equation body with mathematical notation preserved, not a shortened recap. Regenerate it with `python tests/dlpno-hessian-model/export_equations.py`; `--check` verifies synchronization. The exporter also checks that every displayed equation is preserved verbatim.

| Step | Hessian milestone | Commit | Working-equation sections |
|---:|---|---|---|
| 1 | Fixed-frame water targets from canonical analytic-gradient differences; symmetry and translation checks | `0af1a1eb` | 1–3 |
| 2 | Native RHF nuclear orbital tangent, including frozen/active occupied partition | `2c7d6d1b` | 4 |
| 3 | Nuclear tangent of Boys-localized occupied orbitals | `cb05d5b2` | 5 |
| 4 | Raw/normalized PAO tangent; occupied-rotation cancellation | `dd2f9639` | 6 |
| 5 | First semicanonical MP2 PNO selection, refits, metric solves and retained projector | `4e14bd09` | 7 |
| 6 | Coupled first-space LMP2 right-amplitude tangent on the retained graph | `bf77748e` | 8 |
| 7 | Final CCSD PNO tangent in the moving first-PNO parent | `44dfb798` | 9 |
| 8 | Projected CCSD Fock source and first T1-dressed fitted doubles terms | `876c7add` | 10.1–10.2 |
| 9 | Complete fixed-amplitude transformed CCSD nuclear residual source | `09eab08f` | 10.3 |
| 10 | Coupled transformed CCSD amplitude tangent and weighted-adjoint checks | `c4af6e7e` | 10.4 |
| 11 | Ordinary iterative triples right/left response, moments and TNO tangents | `a50ea0b8` | 11 |
| 12 | Stationary triples curvature; analytic mixed DF integrals, metric inverses and moments | `306480bf` | 12 |
| 13 | Eliminate mixed CCSD amplitudes; analytic trial TNO curvature and semi-analytic CCSD mixed source | `dbd06055` | 13 |
| 14 | Stationary retained-pair CCSD curvature using the converged gradient Lambda | `fd6e9086` | 14 |
| 15 | Both PNO truncation-energy losses, projector response and diagonal-pair singles curvature | `6aac84c9` | 15 |
| 16 | Isolated PNO-loss LMP2 adjoint; eliminate mixed LMP2 amplitudes for this correction | `46539933` | 16 |
| 17 | Mixed PAO and full-virtual parent response from occupied-projector actions | `c2cc6533` | 17 |
| 18 | Analytic mixed RHF density/Fock and frozen/active spectral partition | `b6dbf057` | 18 |
| 19 | Mixed quadratic Boys/Mulliken-PM response and native electronic dipole second derivatives | `d79f434b` | 19–19.1 |
| 20 | Remove forward localization precision floor; resolve failed all-electron assembled entry | `063eb109` | 19.2 |
| 21 | Analytic mixed first PNOs; remove sampled molecular inputs from PNO-loss curvature | `1b9f1b23` | 7.1, 16 |
| 22 | Native mixed coupled-LMP2 solve and final-PNO curvature; analytic frame integration | `05aea699` | 8.1, 9.1 |
| 23 | Analytic mixed nuclear CCSD residual and stationary curvature integration | `f18cf39a` | 20–20.4 |
| 24 | Four complete Cartesian Hessians; independent symmetry, rigid-motion and gradient checks | `1b0c5517` | 21–21.2 |
| 25 | Screened-domain audit, native selection fingerprints, sparse residual/gradient checks; production criterion not met | Current checkpoint | 22–22.2 |

Step 19 was recovered locally as `e8a240b2` and published as `d79f434b`; the localization precision fix was locally `d222a3ab` and published as `063eb109`. These pairs refer to the same tested content at each checkpoint, not additional Hessian steps. The failed all-electron total error of -3.37e-7 at step 19 and its resolution at step 20 are both retained below. Full Cartesian validation is reported explicitly at step 24; a general production Hessian and equilibrium vibrational validation remain open.

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

## 2026-11-06 — CCSD residual sources: projected Fock and first fitted doubles terms

**Implemented.** Transported the final CCSD PNO frames and added a requested-edge derivative of the interpair overlap. The projected bare OO/OV/VV Fock derivatives now feed native singles and unique-pair doubles residual source blocks, including the t1 dressing of the bare Fock contributions. Reused converged CCSD T1/T2, spin-adapted doubles, projected singles, the native overlap accessor and occupied-domain maps. For Jiang Eqs. 75–76, added the direct fitted doubles product and virtual ladder derivatives using the native dressed OI/VI intermediates, T2, and moving projected singles. The pair metric inverse-square-root tangent shares the spectral derivative code with the CCSD Lambda integral pullback. The two-electron source accepts metric-whitened local factors in the actual native pair auxiliary order and verifies the dressed factors before contraction. [Native Fock source](psi4/src/psi4/dlpno/ccsd_pair_response.cc) · [DF and frame tangents](psi4/driver/procrouting/response/dlpno_orbital.py) · [Equations](DLPNO_HESSIAN_EQUATIONS.tex)

**Verification.** The gradient-model regression compares every independent R1 and R2 projected Fock derivative with centered variations of the original complete right residual (fixed amplitudes and overlap maps). The contraction with the ordinary (T)-sourced Lambda plus the explicit singles energy derivative independently reproduces the native Fock covectors. These tests pass with an absolute threshold of 1e-8. The water O–H stretch check uses independently reconverged fixed-rank PNO spaces, aligns their final frames, and extrapolates the overlap and partial residual contributions from displaced calculations. Maximum absolute errors per unit displacement:

| Core | One rectangular overlap | Fock R1 | Fock R2 | Direct + ladder R2 (one off-diagonal pair) |
| --- | ---: | ---: | ---: | ---: |
| Frozen O 1s | 4.30e-9 | 6.99e-11 | 2.47e-10 | 8.95e-11 |
| All-electron | 9.90e-9 | 9.54e-11 | 1.39e-9 | 1.61e-10 |

**Scope.** This checkpoint supplies partial residual derivatives. Next differentiate the remaining two-electron singles, mixed-domain and beta/gamma/delta doubles contributions, and their explicit overlap occurrences. Then assemble the full source and solve the coupled singles/doubles tangent equations. The molecular verification uses full fitting domains; no end-to-end Hessian, screened-source scaling, or complete CCSD response is claimed. [Component check](tests/dlpno-grad-model/input.dat) · [Molecular verification](tests/dlpno-hessian-model/lmp2_tangent.py)

## 2026-11-07 — Complete nuclear source for the transformed CCSD singles and doubles residual

Added the fixed-amplitude source for the coupled right response. The assembly reuses native converged singles and doubles, ordered dressed DF factors, mixed-domain J/K, occupied maps, and the beta, gamma and delta intermediates retained for Lambda. Reconstructed the three intermediates against the native cached values before differentiating. The doubles residual now includes Jiang Eqs. 75–81, and the singles residual includes Eqs. 87–90 with its dressed Fock term. The molecular interface takes local fitted-factor and mixed-integral tangents in their native orders. [Source](psi4/driver/procrouting/response/dlpno_ccsd_response.py) · [Working equations](DLPNO_HESSIAN_EQUATIONS.tex)

Checked an O–H stretch in water/cc-pVDZ, with Boys localization, full fitting domains and native truncated PNO ranks. Compared the reference residual reconstruction with the converged native solution, and the analytic source against centered, extrapolated displaced residuals at fixed CCSD components in aligned frames. The lower-order states were recalculated at the displacements. Maximum absolute differences across all singles and doubles components, in Hartree/bohr:

| Core | Unique pairs | Singles | Doubles |
| --- | ---: | ---: | ---: |
| Frozen O 1s | 10 | 9.26e-11 | 4.47e-10 |
| All-electron | 15 | 1.19e-10 | 1.96e-9 |

Next implement the coupled Jacobian action, check its weighted transpose against the Lambda action and solve for the first-order amplitudes. The present water test verifies the residual source. Screened molecular integral delivery, its scaling, the triples response and the final Hessian comparison with differences of canonical DF-CCSD(T) gradients remain open. [Water check](tests/dlpno-hessian-model/lmp2_tangent.py)

Native build and install passed. The water diagnostic and the three regressions dfccsd-t-grad1, dlpno-grad-model and dlpnocc-1 passed. The equation record compiled in two LaTeX passes.

## 2026-11-07 — Coupled transformed CCSD right amplitude response

Implemented a matrix-free right action on simultaneous singles and doubles directions. Four evaluations of the original transformed residual at fixed geometry extract the linear coefficient of its quartic amplitude polynomial. Reference amplitudes, spin-adapted doubles and their dressed intermediates are restored after an action. The Lambda reverse contractions supply the weighted adjoint directly with their additive CC and triples energy sources omitted. Added a GMRES solve for the coupled right tangent with the semicanonical orbital gaps as preconditioner, unique occupied pairs, and symmetric diagonal-pair coordinates. Both the solver and the molecular check inspect the true unpreconditioned residual. [Native actions](psi4/src/psi4/dlpno/ccsd_gradient_lambda.cc) · [Solver](psi4/driver/procrouting/response/dlpno_ccsd_response.py) · [Equations](DLPNO_HESSIAN_EQUATIONS.tex)

The original symmetric water stretch diagnostic made the fixed-branch condition observable. For frozen-core water the first PNO parent of occupied pair (2,3) changed from 15 to 14 functions at both signs of 0.001 and 0.0005 bohr. Final ranks and the screened occupied Fock graph alone would have missed this change. The apparent maximum errors in the singles and doubles derivatives on that comparison were 2.67e-4 and 4.59e-3 per bohr. The original fixed-rank reconstruction test is retained.

The new independent check uses water/cc-pVDZ, Boys orbitals, full fitting domains, nonzero pair cutoffs (1e-9), and occupancy-only PNO selection (first/final occupation cutoffs 1e-5/1e-4). It recomputes the native DLPNO wavefunction at each displaced geometry, matches occupied signs and permutations, polar-aligns the final PNO frames with the reference metric, and verifies both PNO ranks and the occupied Fock graph. It compares two Richardson windows using displacements of 0.002, 0.001 and 0.0005 bohr. [Reproduction script](tests/dlpno-hessian-model/lmp2_tangent.py)

| Core | Absolute weighted transpose error | Finest-window max T1 error / bohr⁻¹ | Finest-window max T2 error / bohr⁻¹ |
| --- | ---: | ---: | ---: |
| Frozen O 1s | 7.82e-14 | 2.14e-11 | 3.41e-10 |
| All-electron | 3.42e-13 | 3.18e-8 | 5.33e-7 |

The all-electron extrapolation is sensitive to the displacement window. The coarser window gave 4.19e-9 and 3.55e-8 per bohr for T1 and T2. The solver check requires every component of J t^v + b^v to be within 1e-10 Hartree/bohr. This tests a single stretch on a fixed local branch. Four native residual evaluations per application preserve the cost order of a native application; large-system iteration counts and end-to-end scaling have not been measured. Next propagate the solved tangents into the iterative triples and moving triple spaces, followed by the remaining curvature contractions and the canonical DF-CCSD(T) gradient-difference Hessian comparison.

The native build, both original and new water diagnostics, dfccsd-t-grad1, dlpno-grad-model and dlpnocc-1 passed. The LaTeX equation record compiled in two passes.

## 2026-11-07 — Ordinary iterative triples amplitude and adjoint response

Implemented the directional response of the single-space RHF (T) model. The opt-in native state supplies converged T3 and Z3, the W/V moments, native TNO frames, local domain indices and the actual directed, screened occupied replacements (including multiplicities). The response uses the differentiated CCSD singles and doubles from the previous step. Each TNO density receives the three moving pair-density contributions of its triplet. Reused the PNO retained/discarded projector derivative, the transported-frame rule and the filtered fitting-metric derivative. The OVVV part of W uses the distinct full metric solve from the forward triples code. [Triples kernel](psi4/driver/procrouting/response/dlpno_triples_response.py) · [Native state](psi4/src/psi4/dlpno/triples_response.cc) · [Equations](DLPNO_HESSIAN_EQUATIONS.tex)

The two matrix-free solves apply the original packed triples operator and its transpose. Their sources include all three moving legs of every intertriplet transfer, the full transported virtual Fock matrix, the occupied diagonal and edge derivatives, and the W/V derivatives. The implementation reconstructs both moments, the original right and left residuals, and the (T) energy at the reference point. It also compares the directional energy from T3 response with the independent stationary contraction using the original Z3.

Water/cc-pVDZ H1 stretch, Boys, both core treatments. Pair cutoffs 1e-9, occupancy-only PNO thresholds 1e-5/1e-4, TNO cutoff 1e-5 and minimum rank 3. The molecular provider uses full fitting domains and the full virtual parent; the TNOs themselves are truncated. The H1 displacement windows use 0.002/0.001 and 0.001/0.0005 bohr. All comparisons align LMO labels and signs and polar-align the TNO frames. Both PNO ranks, TNO ranks, triplet membership and occupied-slot edges are checked across displacements. [Molecular diagnostic](tests/dlpno-hessian-model/lmp2_tangent.py)

| Core | Triplets | TNO ranks | T3 solve norm | Z3 solve norm | (T) directional derivative (Eh/bohr) |
| --- | ---: | --- | ---: | ---: | ---: |
| Frozen O 1s | 16 | 9–10 | 1.04e-12 | 3.45e-12 | −0.00091864104135 |
| All-electron | 30 | 3–10 | 2.15e-12 | 4.00e-12 | −0.00091102837191 |

| Core | Finer-window max T3 error (bohr⁻¹) | Finer-window max Z3 error (bohr⁻¹) | Finer-window (T) derivative error (Eh/bohr) |
| --- | ---: | ---: | ---: |
| Frozen O 1s | 2.44e-11 | 2.42e-11 | 9.45e-13 |
| All-electron | 1.96e-8 | 3.66e-8 | 4.02e-13 |

The coarser-window all-electron T3 and Z3 differences were 2.09e-9 and 3.51e-9 per bohr. One convention matters for reproducing the adjoint comparison. When localization exchanged occupied labels between displaced runs, directly relabeling Z3 gave a difference near 0.05 per bohr concentrated in repeated-index rows. The packed transpose acts on unrestricted cubes and inherits the forward tie order. The diagnostic now independently solves the displaced adjoint from the displaced native V moments in the reference occupied ordering, using the same identified edges. It compares the full packed derivative, including repeated-index rows. Displaced right amplitudes and (T) energies are taken directly from the native runs.

The checkpoint requires in-core triples state retention. Production delivery of screened triplet PAO-parent and fitted-integral tangents, disk-backed response and system-size benchmarks remain open. The next Hessian contribution is to differentiate the stationary triples contraction entering each nuclear gradient component and verify the resulting Hessian column against differences of canonical DF-CCSD(T) gradients.

The tests exercise 48 rectangular directed triplet transfers with frozen core and 230 with all electrons. The native build, the existing CCSD/LMP2 water checks, both new molecular response cases, and dfccsd-t-grad1, dlpno-grad-model and dlpnocc-1 passed. The LaTeX equation record compiled in two passes.

## 2026-11-07 — Stationary ordinary (T) contribution to a Hessian column

Separated preparation of each gradient direction from the iterative T3/Z3 solve. A column solves once for the two column tangents and reuses them across output rows. The relaxation contraction uses the row's right and left sources. The explicit contraction adds mixed W/V, all three mixed virtual Fock legs, the occupied diagonal, and the mixed occupied-slot transfers. The mixed edge product includes the six ordered placements of first derivatives on distinct virtual legs. It accepts a sparse occupied Fock callback and requests overlap blocks on the native screened graph. The previous triples tangent entry point remains available. [Column and source assembly](psi4/driver/procrouting/response/dlpno_triples_response.py) · [Equations](DLPNO_HESSIAN_EQUATIONS.tex)

Implemented analytic mixed derivatives of the ordered pair projections, triplet moments, full inverse Coulomb solve, and filtered inverse used by the spin-summed energy. A domain-indexed Libint2 kernel computes the bare mixed metric and three-center integrals for two Cartesian coordinates, preserving arbitrary local index order and the multiplicity of coincident shell centers. The mixed projection kernel can consume molecular coefficient panels for a full fit or preassembled raw local inputs for the forward triplet fit. The water checks use full fitting domains. [Mixed moments](psi4/driver/procrouting/response/dlpno_triples_curvature.py) · [Bare integrals](psi4/src/psi4/dlpno/triples_df_response.cc)

**Verification.** Mixed-moment tests cover distinct and repeated occupied indices, rectangular pair projections, the full inverse and the filtered inverse. A rotating positive metric with a discarded eigenvalue and degenerate retained eigenvalues checks the second divided difference analytically. The independent displaced-vertex tests agree within 4.0e-8 absolute. On water, mixed AO metric and three-center blocks agreed with differences of first integral derivatives within 4.51e-12 and 8.89e-12 respectively; interchanging the coordinate order also passed. [Moment checks](tests/dlpno-hessian-model/triples_mixed_moments.py) · [AO checks](tests/dlpno-hessian-model/triples_df_hessian.py)

For water/cc-pVDZ H1 z, evaluated all nine Cartesian output rows in frozen-core and all-electron models using the preceding fixed-branch settings. The triples terms, first electronic responses, AO overlap/metric/three-center derivatives and mixed projections are analytic. **The molecular diagnostic supplies mixed lower-order electronic inputs** (LMOs, PNO and TNO frames, T1, T2 and the relaxed AO Fock matrix) from independently reconverged, aligned states. It checks a second construction from displaced assembled local coefficients, reciprocity of the relaxation part, and an independent column obtained from differences of stationary (T) gradients. The full-system benchmark isolates the canonical triples contribution by differencing analytic DF-CCSD(T) and DF-CCSD gradients, then differentiating that difference with 0.004/0.002 bohr displacements. Local checks use 0.002/0.001 bohr and Richardson extrapolation. [Reproduction](tests/dlpno-hessian-model/triples_curvature.py)

| Core | H1 z relaxation | H1 z local (T) curvature, mixed electronic input diagnostic | Difference of local stationary gradients | Canonical DF (T) curvature |
| --- | ---: | ---: | ---: | ---: |
| Frozen O 1s | −0.00210153759 | −0.00126660444 | −0.00126660268 | −0.00133111687 |
| All-electron | −0.00205695985 | −0.00124087815 | −0.00124086348 | −0.00134788055 |

All entries are Eh/bohr². The local and canonical energies use different retained spaces at these cutoffs. The next step is to replace the diagnostic mixed electronic inputs with lower-order stationary contractions, starting with the CCSD adjoint contracted against the mixed transformed residual. The molecular Hessian, screened production path, scaling measurements and frequencies remain open.

Across all nine output rows, the maximum discrepancies from differences of local analytic (T) gradients were 1.77e-9 and 1.47e-8 Eh/bohr² (frozen and all-electron). The maximum differences from canonical at these working cutoffs were 4.44e-4 and 4.50e-4 Eh/bohr², in O y and H2 y respectively.


As a separate complete-space check, set both PNO occupation thresholds and the TNO threshold to zero, and include all Fock couplings and triplet domains; the 1e-9 pair cutoff still retains every water pair. The local H1 z triples curvature was −0.00133112006 (frozen) and −0.00134788260 (all-electron) Eh/bohr². Absolute differences from canonical were 3.20e-9 and 2.06e-9 Eh/bohr². These are isolated (T) longitudinal entries. No frequencies or full DLPNO-CCSD(T) Hessian have been delivered.

The native build, dfccsd-t-grad1, dlpno-grad-model, dlpnocc-1, the previous water response regression and the new curvature, mixed integral and mixed moment checks passed. The LaTeX record compiled in two passes.

## CCSD stationarity in the semi-analytic ordinary (T) Hessian

Eliminated the mixed CCSD singles and doubles from the stationary triples column. The native accessor returns the existing ordinary (T) amplitude source, including its retained-TNO density derivative. The coupled solver now accepts the weighted transpose and solves a triples-only CCSD adjoint once per reference. Its contraction replaces the mixed-amplitude term for every row and column. The ordinary CCSD energy source is excluded from this multiplier. [Native residual and source access](psi4/src/psi4/dlpno/ccsd_gradient_lambda.cc) · [Adjoint and mixed source](psi4/driver/procrouting/response/dlpno_ccsd_response.py) · [Column assembly](psi4/driver/procrouting/response/dlpno_triples_response.py)

Added residual replay at arbitrary real RHF amplitudes in the native pair frames. It validates the independent blocks, restores the reference amplitudes and rebuilds their dressed intermediates after each evaluation. The semi-analytic mixed source combines finite differences in nuclear geometry with exact amplitude derivatives of the native quartic residual. Its amplitude-only quadratic contraction uses odd–odd coefficient extraction; the geometry-dependent Jacobian actions use linear coefficient extraction at the reference amplitude components in the moving frames. Residuals are mapped back to the same labeled chart, including occupied signs and reversed-pair transposes.

Added the analytic mixed TNO projector and reference-polar frame derivative. The retained/discarded mixing solves the differentiated spectral commutator, and the two diagonal blocks follow from differentiated idempotency. Its inputs are the mixed PAO parent and pair frames, AO metric, and the CCSD amplitudes on the affine trial. Only the mixed CCSD amplitudes are set to zero. The stationary assembly does not request displaced mixed T1, T2 or TNO frame entries. [Mixed TNO kernel](psi4/driver/procrouting/response/dlpno_triples_curvature.py) · [Equations](DLPNO_HESSIAN_EQUATIONS.tex)

The water diagnostic retains the previous direct construction for comparison. It independently reconstructs TNO frames along the trial at displaced geometries and checks the mixed residual identity against the native Jacobian acting on diagnostic mixed amplitudes. A separate quartic and rotating-projector test checks coefficient extraction, internal degeneracies, a fully retained space and rejection of a split degenerate block. [Water curvature](tests/dlpno-hessian-model/triples_curvature.py) · [Algebra checks](tests/dlpno-hessian-model/ccsd_stationarity.py)

The geometric part of the mixed transformed residual and lower-order mixed LMO, PNO, triplet-parent and AO Fock inputs are still semi-analytic. The result is the isolated ordinary (T) contribution on the selected branch. A next step toward molecular frequencies is to assemble the retained-pair CCSD energy curvature with the correlated CCSD multiplier, then account separately for the lower-order energy corrections and the RHF contribution.

Water/cc-pVDZ, Boys H1 z stretch, working cutoffs as in the preceding entry. All values in this table are Eh/bohr². The old construction uses displaced mixed CCSD amplitudes. The stationary construction uses the CCSD adjoint and analytic mixed TNO selection.

| Core | CCSD adjoint correction | Stationary H1 z (T) curvature | Old construction | Absolute change |
| --- | ---: | ---: | ---: | ---: |
| Frozen O 1s | +0.00044791455 | −0.001266604469 | −0.001266604443 | 2.52e-11 |
| All-electron | +0.00046161022 | −0.001240878873 | −0.001240878147 | 7.27e-10 |

With complete PNO/TNO spaces and the positive pair cutoff, the stationary H1 z values were −0.001331120144 and −0.001347883324 Eh/bohr² respectively. Absolute differences from the canonical triples curvature were about 3.28e-9 and 2.78e-9 Eh/bohr². The differences from the old complete-space construction were 7.99e-11 and 7.19e-10 Eh/bohr².

Lean 4.16.0 with mathlib v4.16.0 checked the weighted mixed-amplitude elimination identity, the coefficient formula for an arbitrary bivariate quartic, and both diagonal projector-block identities under differentiated idempotency. These theorems assume the response equations; the native calculations test their implementation. [Lean proof](tests/dlpno-hessian-model/CCSDStationarity.lean)

Across the nine Cartesian output rows at the working cutoffs, replacing direct mixed-amplitude inputs changed the column by at most 1.02e-10 Eh/bohr² with frozen core and 3.98e-9 Eh/bohr² with all electrons. Maximum differences from independently differenced local stationary (T) gradients were 1.79e-9 and 1.54e-8 Eh/bohr². At H1 z, the adjoint correction agreed with contracting the native (T) source against displaced mixed CCSD amplitudes within 4.95e-12 and 6.45e-11 Eh/bohr². The residual replay test checks that the native amplitudes and dressed pair intermediates are restored.

Native build, dfccsd-t-grad1, dlpno-grad-model, dlpnocc-1, the previous coupled-response water diagnostic, the new water stationarity checks and the formal algebra checks passed. The equation record compiled in two LaTeX passes.

## Stationary retained-pair CCSD correlation curvature

Implemented the mixed curvature of the retained RHF CCSD energy used by Jiang Eq. 45 and the native singles Fock term. Each independent pair carries its ordered-energy multiplicity (one for ii, two for ij). The product rule includes moving diagonal-to-pair singles transfers, the bare Fock row, and the spin-adapted exchange from the original full local Coulomb solve. The kernel checks its reference energy against the native retained CCSD correlation energy. This quantity is separate from the published CCSD correlation energy, which includes local energy corrections. [Energy and curvature kernel](psi4/driver/procrouting/response/dlpno_ccsd_curvature.py) · [Native forward energy](psi4/src/psi4/dlpno/ccsd.cc) · [Equations](DLPNO_HESSIAN_EQUATIONS.tex)

The converged native gradient multiplier is sourced by retained CCSD plus ordinary (T). Subtract the previously solved triples-only CCSD multiplier in the same pair metric to obtain the retained-only multiplier. No extra solve is required when the triples multiplier is available. Contract this multiplier against the same mixed residual as the triples column. The CCSD energy receives only reference amplitudes and first tangents on the affine path. A direct water check differentiated the energy with respect to singles and doubles independently: maximum native weighted-adjoint residuals were 1.40e-12 (frozen O 1s) and 5.67e-13 (all-electron) at working cutoffs. Lean checked the multiplier subtraction identity in the weighted coordinates. [Native Lambda accessor](psi4/src/psi4/dlpno/ccsd_gradient_lambda.cc) · [Lean proof](tests/dlpno-hessian-model/CCSDStationarity.lean)

For the water/cc-pVDZ H1 z column, checked all nine Cartesian outputs on the occupation-selected Boys branch with positive pair cutoffs. Displacements were 0.002 and 0.001 bohr with Richardson extrapolation. The independent local reference differences the analytic first derivative of the ordered retained energy. Maximum absolute errors across the nine outputs (Eh/bohr²):

| Core | Against direct mixed-amplitude construction | Against differences of local retained gradients |
| --- | ---: | ---: |
| Frozen O 1s | 2.80e-10 | 3.93e-8 |
| All-electron | 1.35e-8 | 6.86e-7 |

The largest all-electron local-gradient difference was in the symmetry-zero O x component. The reported column was not symmetrized. A sparse-pair algebra check with rectangular transfers and nonzero singles/Fock terms agreed with an independent ordered-pair calculation to 3.71e-10 absolute. [Water diagnostic](tests/dlpno-hessian-model/triples_curvature.py) · [Sparse pair check](tests/dlpno-hessian-model/retained_ccsd_energy.py)

| Core | Working-cutoff retained H1 z | Complete-space retained H1 z | Canonical DF-CCSD correlation H1 z |
| --- | ---: | ---: | ---: |
| Frozen O 1s | −0.00025572698 | −0.00233355588 | −0.00233350537 |
| All-electron | −0.00127063922 | −0.00332875547 | −0.00332867762 |

Entries are Eh/bohr². The canonical reference differences analytic DF-CCSD minus DF-RHF gradients. In the complete-space benchmark, both PNO thresholds and the TNO threshold are zero and all water pairs survive the positive 1e-9 pair cutoff. Adding the analytic RHF Hessian and the previously verified triples contribution gave longitudinal total curvatures 0.57108383457 and 0.57007187180 (frozen and all-electron). Direct differences of canonical analytic DF-CCSD(T) gradients gave 0.57108388627 and 0.57007195032. The reference and sampled geometries in this check have zero local energy corrections to the test tolerance.

At working cutoffs, the native PNO truncation energy at the reference was −0.010064689976 Eh (frozen) and −0.010251784717 Eh (all-electron); the weak, eliminated-pair and dipole energies were zero. Its curvature is the next missing energy term for this fixture. Screened fit delivery, mixed lower-order electronic inputs, system-size scaling and the remaining Hessian columns remain open.

A displacement-window check at working cutoffs gave an all-electron O x discrepancy of 7.29e-9 and an H1 z discrepancy of 1.18e-7 Eh/bohr² with 0.003/0.0015 bohr steps. An exploratory 0.004/0.002 bohr all-electron comparison failed the existing Boys stationarity check at a displaced geometry and was excluded. The primary nine-row comparison uses the same 0.002/0.001 bohr steps for both core choices. At those steps, maximum absolute differences between the working-cutoff retained and canonical CCSD correlation columns were 2.08e-3 (frozen) and 2.06e-3 (all-electron) Eh/bohr². The retained-pair observable and the fully corrected energy must be distinguished at finite cutoffs.

Also assembled the complete-space mixed entry (O y, H1 z) with the analytic RHF Hessian. Differences from direct canonical DF-CCSD(T) gradient differences were 5.97e-10 (frozen) and 2.68e-8 (all-electron) Eh/bohr². For the longitudinal entry the corresponding differences were 5.17e-8 and 7.86e-8 Eh/bohr². The native build, three regressions (dfccsd-t-grad1, dlpno-grad-model, dlpnocc-1), sparse algebra check, water working-cutoff columns and both complete-space core cases passed. Lean checked the weighted-adjoint subtraction, and the LaTeX equation record compiled.

## Two-stage PNO truncation-energy curvature

**Implemented.** `dlpno_pno_curvature.py` differentiates both native full-minus-retained PNO energy losses. The first stage uses semicanonical MP2 amplitudes; the final stage uses converged coupled-LMP2 amplitudes. Both include density-projector response, parent/input motion, the unique-pair multiplicity (1 diagonal, 2 off diagonal), and diagonal-pair singles losses. The first-stage mixed amplitude is solved analytically using the reference pair Fock eigensystem, without a pair Kronecker matrix. The final-stage total mixed LMP2 amplitude remains a required input. It must not be zeroed: this step does not introduce an isolated truncation-loss LMP2 adjoint.

The module reuses `MixedBlock`, the mixed product rule, local inverse response, and the mixed spectral projector already used by the triples curvature. A resolved retained/discarded gap and fixed ranks are required. The singles denominator mask must be fully active; the kernel rejects near-zero denominators rather than silently approximating the native masked response. Full-rank losses vanish. This is an internal curvature contraction API, not a newly exposed public molecular Hessian/frequency driver.

**Verification.** `tests/dlpno-hessian-model/pno_curvature.py` compares analytic mixed contractions against independently reconstructed scalar pair energies with diagonal and off-diagonal pairs, both amplitude models, nonzero singles, truncated/full ranks, and mixed input derivatives. Maximum discrepancy was 4.77e-10. Direction interchange passed to 1e-13, and a degenerate selection boundary was rejected. The prior retained-CCSD algebra test still passed (3.71e-10). The existing Lean projector identities in `CCSDStationarity.lean` were rerun successfully. No claim of a new formal proof of the complete curvature implementation is made.

`pno_curvature_water.py` tests water/cc-pVDZ with full fitting domains and the same nonzero pair cutoffs and finite PNO cutoffs as the prior retained/triples fixture. It matches occupied labels/signs and polar-aligns the first-PNO and full-virtual frames, checks both retained ranks at every sample, and reconstructs the native `DLPNO PNO TRUNCATION ERROR` to 2e-10 hartree at every geometry. Lower-order derivatives, including final-stage LMP2 mixed amplitudes, are sampled from aligned native states. The independent scalar differences below are diagnostics of this correction on that checked branch, not a replacement for canonical gradient-difference validation or a public numerical DLPNO gradient route.

H1-z/H1-z entries, hartree/bohr², with five-point input stencils of spacing 0.001 bohr:

| Core | Initial loss curvature | Final loss curvature | Sum | Difference from scalar diagnostic |
|---|---:|---:|---:|---:|
| Frozen O 1s | -0.000392214229114 | -0.000083384500072 | -0.000475598729187 | -2.41e-12 |
| All-electron | -0.000374635147839 | +0.000045056397832 | -0.000329578750007 | -5.02e-10 |

With spacing 0.0005 bohr, the sums are -0.000475602097850 and -0.000329663162352, respectively. Diagnostic differences remain below 2.5e-9. The all-electron curvature changes by 8.44e-8 on halving the step, exposing the remaining electronic/mixed-input numerical sensitivity. No claim of more accuracy follows from the tighter agreement of two calculations using the same displaced input states. Roundoff antisymmetry in the projected Fock is removed before differencing; the kernel requires symmetric Fock inputs. The LaTeX equation record compiles without overfull or undefined-reference diagnostics. The normal install target succeeds. Regression tests `dfccsd-t-grad1`, `dlpno-grad-model`, and `dlpnocc-1` all pass (3/3).

**Scaling and scope.** Pair-local operations use cubic work and quadratic storage in the local parent rank, and visit only supplied retained pairs. Bounded ranks and O(N) retained pairs imply O(N) contraction work per supplied Hessian entry. Generating all mixed electronic inputs, responding to screened local fits, and assembling all Hessian entries have not been shown to have this cost. The dense full-domain water fixture is a verification provider only. Remaining weak/eliminated-pair/dipole corrections at general cutoffs have not been implemented here.

**Next.** Eliminate the final-stage mixed coupled-LMP2 amplitude with a truncation-specific adjoint and a mixed LMP2 residual, or implement its coupled mixed response directly. Connect the correction to the retained-CCSD/(T) molecular column provider, then verify the assembled zero-cutoff column against canonical DF-CCSD(T) gradient differences and finite-cutoff consistency separately. This step supplies the correction contraction and its native-functional tests; it does not complete that shared column assembly.

## Isolated PNO-loss LMP2 adjoint and shared column assembly

**Implemented.** The gradient's existing reverse PNO-loss source is copied before its selection records are freed, only when `DLPNO_KEEP_LMP2_RESPONSE` is requested. `core.dlpno_pno_truncation_lmp2_response(wfn, adjoint=True)` reuses the native screened transpose LMP2 solver for this isolated source; `adjoint=False` exposes the source for verification. Pair multiplicities are already included in the source. The contraction uses the unweighted unique-pair Frobenius sum (with symmetric diagonal tensors), not the CCSD Lambda metric. The initial loss has no dependence on the coupled-LMP2 amplitude at fixed parent inputs and supplies no source to this solve. No combined-gradient LMP2 multiplier is substituted for the isolated multiplier.

The retained `RESPONSE_EDGES` block exposes the native ordered source pairs and Fock indices, preserving the exact screened graph. `lmp2_residual` and `lmp2_mixed_residual` differentiate the graph's projected Fock, exchange, and both overlap legs. `PNOTruncationCurvature.element` contracts the affine-amplitude energy plus the isolated adjoint against the mixed residual. It never reads the final-stage mixed LMP2 amplitude. The initial semicanonical mixed solve remains active. The source and solve remain pair-local/matrix-free; iteration-count and end-to-end scaling are not established.

**Verification.** Rebuilt and installed the native code. Regression tests `dfccsd-t-grad1`, `dlpno-grad-model`, and `dlpnocc-1` pass (3/3). Water/cc-pVDZ with frozen O 1s and all electrons retains the previous nonzero pair cutoffs and finite PNO ranks. Native primal LMP2 residuals were checked against the reconstructed graph. Independent analytic directional derivatives of the PNO-loss energy agree with the native source within 2.54e-16; the random-direction transpose/adjoint identity agrees within 3.68e-15. The stationary entry is unchanged when its final mixed amplitudes are replaced by NaNs. Algebraic rectangular/reversed-edge mixed-residual tests agree with independent matrix evaluations within 7.16e-10, and explicitly check `R_uv = C_uv + A t_uv`.

| Core | Input spacing (bohr) | Stationary H1-z PNO curvature (hartree/bohr²) | Stationary minus direct mixed-amplitude route |
|---|---:|---:|---:|
| Frozen O 1s | 0.001 | -0.000475598701670 | +2.75e-11 |
| Frozen O 1s | 0.0005 | -0.000475601991251 | +1.07e-10 |
| All-electron | 0.001 | -0.000329578773083 | -2.31e-11 |
| All-electron | 0.0005 | -0.000329663158179 | +4.18e-12 |

The independent O_y/H1_z mixed entry, extrapolated from 0.002/0.001 bohr inputs, agrees with the direct route within 2.49e-12 frozen and 5.83e-11 all-electron. At zero PNO cutoffs the native isolated adjoint is exactly zero for both references. These checks are in `pno_curvature_water.py`; the algebraic residual checks are in `pno_curvature.py`.

**Assembly.** `triples_curvature.benchmark(..., include_pno=True)` now appends the stationary correction as column 5 and sums it with the retained-CCSD and iterative-triples entries. The H1-z correlation curvatures are -0.001997930149608 (frozen) and -0.002841096871356 (all-electron). These are finite-cutoff correlation entries, excluding RHF. Their PNO components are -0.000475598701732 and -0.000329578773599, and their direct-route discrepancies are 2.76e-11 and 2.36e-11. The existing CCSD/triples component checks also passed in this combined run. Other local correction energies are zero in this fixture; general weak/eliminated-pair/dipole curvatures remain open.

**Formal check and limitations.** Added and checked `eliminate_mixed_lmp2` in `CCSDStationarity.lean`, specializing the finite-dimensional elimination identity to the unweighted pairing. It assumes the mixed constraint and adjoint equations; it does not certify the numerical nuclear derivatives. The LaTeX record includes the native residual, source convention, adjoint and stationary energy formula. Mixed frames/Fock/exchange inputs remain sampled in the molecular provider, and the previous step-size sensitivity remains visible. This is a more stationary semi-analytic Hessian construction, not a fully analytic molecular Hessian or frequency implementation.

**Next.** Replace sampled mixed lower-order orbital/localization and local-frame inputs, and the CCSD nuclear mixed residual source, with analytic response providers. Before a production Hessian driver, assemble and validate complete Cartesian matrices and any nonzero remaining screened-energy corrections. Preserve the fixed-rank/fixed-domain branch checks.

## Mixed PAO projection and full-virtual parent response

**Implemented.** `dlpno_frame_curvature.py` constructs raw/normalized mixed PAOs on fixed AO domains and the mixed full-virtual frame in the diagnostic's fixed-reference polar gauge. Inputs are total occupied-projector response and AO overlap derivatives. All occupied orbitals, including frozen core, enter the projector. The coefficient action contracts occupied orbitals with column panels without forming a dense density or identity. Metric derivatives use one reference factorization and four panel solves, or a caller-supplied reference solver that can be reused across entries. No explicit metric inverse is formed. The full virtual frame formula is not a replacement for the response of a separately truncated local parent.

The shared triples molecular input provider now reconstructs the full-virtual parent through `(S^-1-D) S0 V0` and polar normalization. It no longer requests a sampled mixed aligned virtual frame. The first occupied-density derivatives come from native RHF tangents. **The mixed occupied density remains sampled.** This is analytic frame reconstruction conditional on electronic response inputs, not an analytic second-order RHF solver. Mixed first/final PNO and localization responses remain open. The normalized PAO kernel is tested independently; it has not replaced the remaining sampled PNO-frame provider.

**Verification.** `frame_curvature.py` checks occupied-rotation cancellation through mixed order (9.24e-14 maximum), the differentiated metric panel equation, exact mixed frame orthonormality, PAO normalization, and u/v reciprocity. Water/cc-pVDZ checks include frozen-core and all-electron settings and H1-z/H1-z and O-y/H1-z entries. Two centered stencils (0.002 and 0.001 bohr) provide Richardson comparisons against independently aligned SCF virtual frames and directly reconstructed PAOs.

| Entry | Virtual-parent mixed error | Raw PAO mixed error | Normalized PAO mixed error |
|---|---:|---:|---:|
| H1-z/H1-z | 3.33e-8 | 2.84e-9 | 1.81e-6 |
| O-y/H1-z | 7.88e-9 | 1.99e-9 | 1.19e-6 |

Both core settings give the same projector results to rounding, as expected for the same RHF full-occupied projector. The selected PAO panel includes a near-core column, making normalization substantially more sensitive than the raw projection. These coefficient errors are separate from energy-curvature errors; no stronger PAO accuracy claim follows from the parent result.

The combined retained-CCSD + triples + PNO H1-z correlation entries still pass their existing component checks. They are -0.001997930152018 (frozen) and -0.002841096874295 (all-electron) hartree/bohr², changes of -2.41e-12 and -2.94e-12 from the previous sampled-parent provider. These finite-cutoff entries exclude RHF and are not new canonical Hessian reference values.

Added and checked the Lean `mixed_metric_panel` identity for the rearranged mixed metric equation; the numerical tests cover its implementation and occupied rotation cancellation. Updated the LaTeX equations and compiled them successfully. The install target succeeds; `dfccsd-t-grad1`, `dlpno-grad-model`, and `dlpnocc-1` pass (3/3). The dense full-domain water provider is still diagnostic-only. A full-width virtual frame and dense metric factorization do not establish linear scaling; the general API allows fixed-width panels and a reused reference solver.

**Next.** Supply the mixed RHF occupied-density response analytically, then propagate it through localization and first/final PNO selections. The CCSD nuclear mixed residual and the remaining nonzero screened-energy corrections also remain before a fully analytic molecular Hessian can be claimed.

## Analytic mixed RHF occupied-density, Fock and frozen-core partition response

**Implemented.** `dlpno_rhf_curvature.py` solves the mixed total RHF density response using the existing native CPHF action/solver. Differentiating `D S D = D` fixes the occupied–occupied and virtual–virtual blocks in the reference MO chart. The mixed `F D S - S D F = 0` equation supplies the occupied–virtual source. The code checks native CPHF, full AO idempotency and full AO stationarity residuals. It requires a full-rank AO metric, a C1 Hartree–Fock reference, consistent symmetric AO inputs and the reference linear Fock action. The one-spin density includes all occupied orbitals, including frozen core. Temporary JK setup is restored after the solve; tests exercise both existing-JK and temporary-JK cases and verify that reference density/Fock matrices are unchanged.

`mixed_df_fock` builds the analytic DF Fock mixed derivative with fitted three-center panels, differentiating a metric solve rather than explicitly inverting the metric. All fit-leg, metric and density cross terms are included without constructing an AO four-index tensor. `rhf_mixed.py::analytic_inputs` supplies analytic first and mixed one-electron/DF integrals, native first RHF density tangents, and the new mixed solve. There are no nuclear displacements in this provider; displacements are used only by its independent verification. This provider uses dense full-domain, full-rank inverse-metric DF and checks its reference Fock against native SCF. It does not establish local scaling or support a screened auxiliary-metric response automatically.

**Frozen core.** `mixed_occupied_partition` polar-transports the full occupied parent, constructs its Fock matrix and differentiates the spectral projector onto the lowest frozen occupied eigenvalues. It returns frozen and active density blocks with their mixed derivatives, requiring a resolved core/active spectral gap. The zero-frozen-core case returns exactly zero core response. This computes subspace densities, not mixed localized-orbital coefficients.

**Verification.** Water/cc-pVDZ with cc-pVDZ-JKFIT, frozen O 1s and all electrons, tested H1-z/H1-z and O-y/H1-z. Against Richardson differences with 0.008/0.004 bohr displacements:

| Entry | Total density maximum error | Fock maximum error | Frozen density maximum error |
|---|---:|---:|---:|
| H1-z/H1-z | 4.77e-8 | 2.27e-7 | 4.50e-9 |
| O-y/H1-z | 3.61e-8 | 6.91e-8 | 4.95e-9 |

Total-density/Fock results are the same for the two core settings, as expected for the total RHF occupied projector. Frozen/active densities sum to the total and separately satisfy mixed idempotency and mutual orthogonality. Maximum total mixed idempotency residual is 6.11e-16; stationarity is 9.49e-9; the native CPHF residual is 1.24e-11. Interchanging u/v agrees within 2e-9. An independent small four-index oracle validates the factorized mixed DF Fock contraction to 7.06e-9.

An exploratory 0.002/0.001 bohr Richardson stencil showed errors of 9.07e-7 (density) and 3.80e-6 (Fock) for the longitudinal entry. The wider stencil reduces this reference sensitivity substantially; the analytic construction itself has no displacement step. The final tests use separate density/Fock tolerances of 2e-7 and 5e-7 and retain the strict analytic constraint checks.

**Integration.** The shared retained-CCSD/triples molecular provider now uses analytic mixed RHF density and Fock, and hence an analytic full-virtual parent constructed from them. It no longer requests sampled `occupied_density` or `fock` mixed blocks. Finite-cutoff correlation column checks still pass: H1-z retained CCSD + iterative triples + PNO is -0.001997924993102 (frozen) and -0.002841091479715 (all-electron) hartree/bohr². The PNO-loss contribution is unchanged because its separate diagnostic still samples its projected lower-order inputs.

**Formal checks and remaining scope.** Added and checked `rhf_mixed_density_blocks` in the Lean record, proving the occupied/virtual diagonal-block signs under the differentiated idempotency constraint. The LaTeX record now contains the density constraints, seed construction, native CPHF sign convention, DF mixed contraction and frozen/active spectral partition. This is an analytic RHF response block, not a fully analytic DLPNO Hessian. Mixed Boys/PM localization, first/final PNO frames, the CCSD nuclear mixed residual, and nonzero remaining screened-energy corrections still require work. The next bounded step is to use the active occupied mixed projector in localization response and propagate the analytic RHF inputs into the remaining local providers.


**Complete-space canonical comparison for this step.** With zero PNO/TNO cutoffs and the same nonzero retained-pair cutoff, the H1-z/H1-z total (analytic RHF + retained CCSD + iterative triples + vanishing PNO loss) is 0.571083839892713 hartree/bohr² frozen and 0.570071877313596 all-electron. Fresh Richardson differences of canonical DF-CCSD(T) analytic gradients give 0.571083886265774 and 0.570071950321993, respectively. Differences are -4.64e-8 and -7.31e-8. Both pass the established 2e-7 criterion. These are tested entries, not validation of an entire Hessian or vibrational analysis. The native build/install, all three regression tests (`dfccsd-t-grad1`, `dlpno-grad-model`, `dlpnocc-1`), Lean check and LaTeX compilation pass; the equation record has no overfull or undefined-reference diagnostics.

## Analytic mixed Boys and Mulliken-PM localization response

**Implemented.** `dlpno_localization_curvature.py` transports the active occupied density into an orthonormal parent, then differentiates the quadratic localization stationarity equations twice. It reuses the existing localization rotation Hessian/solver for both first rotations and the mixed rotation. The exponential-coordinate cross term is `(K_u K_v + K_v K_u)/2`; parent/rotation cross terms are retained. Singular or non-maximal reference branches are rejected, and first/mixed stationarity and metric orthonormality are checked. PM here means quadratic Mulliken PM, not a different population scheme.

Added native `MultipoleInt` electronic dipole second derivatives by applying the primitive Gaussian center derivative recurrence twice. `MintsHelper.ao_elec_dip_deriv2(A,B)` exposes 27 AO matrices ordered by dipole component, axis at A, then axis at B. Same-center shell assignments, Cartesian/spherical transformation, and the electronic minus sign are included. Higher multipoles still do not support second derivatives. Boys uses these analytic dipole derivatives; PM uses existing analytic overlap derivatives.

**Verification.** The exact rotating-orbital algebra oracle passes for one and four occupied orbitals (maximum error 5.56e-17); a degenerate localization branch is rejected. Water dipole checks against Richardson differences of analytic first derivatives pass for Cartesian and spherical cc-pVDZ and cc-pVTZ, including f functions. Maximum errors are 1.89e-12 (cc-pVDZ) and 1.05e-11 (cc-pVTZ). Mixed-coordinate reciprocity and AO symmetry pass; the differentiated translation identity has residual at most 1.07e-14.

Water/cc-pVDZ mixed localized coefficients were compared to independently relocalized SCFs at 0.008/0.004 bohr:

| Core | Localization | H1-z/H1-z maximum error | O-y/H1-z maximum error |
|---|---|---:|---:|
| Frozen O 1s | Boys | 5.42e-8 | 5.71e-8 |
| Frozen O 1s | PM | 6.59e-8 | 5.80e-8 |
| All-electron | Boys | 5.39e-8 | 5.69e-8 |
| All-electron | PM | 2.48e-7 | 2.51e-7 |

The forward localizer sometimes stalled near its requested tolerance (one displaced PM gradient was 1.31e-7). For the independent coefficient oracle only, a native near-stationary initial guess is Newton-polished to gradient below 2e-13 and checked to be a nonsingular maximum. A failed native solve with gradient above 1e-5 is rejected. This prevents forward-localization noise from dominating a second finite difference; it does not relax the production mixed-response stationarity guard. Interchanging the two nuclear coordinates agrees within 2e-8.

**Integration.** The shared molecular retained-CCSD/triples diagnostic now supplies the mixed LMO block analytically from the active RHF density and compares its first tangents with the pre-existing provider. The finite-cutoff H1-z correlation entries (retained CCSD + triples + PNO loss) pass their component checks: -0.001997997419991 (frozen) and -0.002841341166261 (all-electron) hartree/bohr². These are diagnostic correlation entries, not full canonical Hessian values. The whole molecular column runner still has Boys-specific first-response plumbing; standalone PM validation is not a claim of a complete PM Hessian driver.

**Formal record and limits.** Added the parent, rotation, stationarity, PM operator and dipole derivative equations to the LaTeX record. The Lean `localization_mixed_rotation` identity checks the noncommuting cross-term cancellation after skew-generator substitutions; it is not a proof of the entire response algorithm. Lean and LaTeX compilation pass. Native build/install succeeds with one compilation process (the initial parallel build exceeded memory). The existing three regression tests are checked separately below.

The next broad dependencies are moving local/PNO-space response and the mixed nuclear CCSD residual. These are two groups, not two small edits or a complete inventory: first/final PNO selections, sampled projected inputs in the separate PNO-loss diagnostic, remaining nonzero screened-energy corrections, full Hessian assembly and vibrational/invariance checks still require coverage. Dense full-domain diagnostics and dense occupied-operator contractions do not establish asymptotically linear scaling. The reused solver forms a dense rotation Hessian only for at most 64 rotations; larger solves use its matrix-free action.

**Canonical total-entry check and unresolved diagnostic.** At zero PNO/TNO cutoffs and the retained nonzero pair cutoff, the H1-z/H1-z total is 0.571083768633976 (frozen) and 0.570071613405484 (all-electron) hartree/bohr². Canonical DF-CCSD(T) gradient differences give 0.571083886265774 and 0.570071950321993. The frozen error -1.18e-7 passes the existing 2e-7 tolerance; the all-electron error -3.37e-7 **does not pass**. The latter is dominated by the retained-CCSD comparison (-3.44e-7), while the triples comparison differs by +6.69e-9. This is a remaining integration accuracy issue, despite passing isolated analytic localization checks. No tolerance has been loosened and no full-Hessian validation is claimed.

The mixed CCSD residual and PNO-frame inputs in this comparison still use 0.002/0.001 bohr displaced calculations. A wider 0.008/0.004 trial was rejected by the existing first-order Boys forward-stationarity guard at a displaced geometry; that guard was not bypassed. Numerical sensitivity is a hypothesis, not yet a demonstrated explanation of the failed all-electron total check. The regression suite (`dfccsd-t-grad1`, `dlpno-grad-model`, `dlpnocc-1`) passes 3/3 in 34.23 seconds. The equation record compiles without overfull or undefined-reference diagnostics.

A second sensitivity trial at 0.004/0.002 bohr also stops at the same displaced-forward Boys stationarity guard, before a complete total comparison. Thus neither wider-stencil trial establishes convergence. The next integration diagnostic must make forward localization consistently stationary and/or replace the sampled residual/local-space inputs, then repeat the unchanged canonical criterion. This step delivers a verified analytic localization block with an explicitly unresolved all-electron assembled-entry check.

## 2026-11-07 — Tight forward localization resolves the assembled-entry failure

**Recovered checkpoint and diagnosis.** The previous localization commit (`e8a240b2`) was present in the local checkout but had not yet reached the remote branch. Continued from that implementation and its failed all-electron total check. A near-maximum restart with a 1e-8-radian occupied rotation reproduced a native localization floor: Boys stopped at maximum AO-projected gradient 4.03e-8 and Mulliken PM at 1.42e-8, without converging to the requested 1e-12. Both augmented-Hessian-enabled and Jacobi-only runs failed. The quadratic pair gain lost precision in `hypot(Hd,Ho)-Hd`, and an objective-roundoff acceptance cutoff independently rejected the remaining small rotations.

**Implemented.** Rationalize the quadratic gain when `Hd > 0` and accept positive exact quadratic pair gains instead of discarding them at objective roundoff. The pair angle still comes from `atan2`; the requested gradient criterion controls convergence. The fourth-power proposal keeps its existing objective test. The same restart now converges with Boys gradient 8.44e-13 and PM gradient 1.78e-14, with or without AH. Extended `dlpnolocal-1` checks tight native localization, near-maximum restarts, independently AO-projected gradients and metric orthonormality. Its boolean checks use `compare_integers`, fixing an existing zero-tolerance `compare_values` error with the installed QCElemental.

The Hessian molecular diagnostics now request `LOCAL_GRADIENT_CONVERGENCE=1e-12`. Added `tests/dlpno-hessian-model/localization_integration.py` for selected canonical-limit entries, with both core choices, configurable nuclear spacing and coordinates. The analytic mixed RHF/localization construction has no displacement parameter; the remaining sampled PNO frames and mixed CCSD residual still do.

**Canonical comparison.** Water/cc-pVDZ, cc-pVDZ-JKFIT/RI, zero PNO/TNO cutoffs, the retained nonzero pair cutoff 1e-9, and zero Fock-coupling screens. Errors below are assembled total minus Richardson differences of canonical DF-CCSD(T) analytic gradients, in hartree/bohr². The component CCSD and triples checks also pass the original 2e-7 criterion.

| Core | Entry | Displacements used (bohr) | Total error |
|---|---|---|---:|
| Frozen O 1s | H1-z/H1-z | 0.008 / 0.004 | -7.66e-9 |
| All-electron | H1-z/H1-z | 0.008 / 0.004 | -7.71e-9 |
| Frozen O 1s | O-y/H1-z | 0.008 / 0.004 | -9.65e-11 |
| All-electron | O-y/H1-z | 0.008 / 0.004 | +4.01e-10 |
| All-electron | H1-z/H1-z | 0.002 / 0.001 | -1.33e-7 |

All pass the unchanged 2e-7 threshold, including the original narrower stencil. Thus the earlier -3.37e-7 failed entry is resolved for these settings. The narrow-stencil error remains larger: these measurements establish residual numerical sensitivity, not complete analyticity or arbitrary-spacing convergence. The full-rank fixture verifies zero screened correction energies at its sampled points; the PNO loss is identically zero on this fixed full-rank branch.

**Finite PNO spaces.** Also reran H1-z/H1-z with first/final PNO cutoffs 1e-5/1e-4 and TNO cutoff 1e-5, full AO/fitting domains, pair cutoff 1e-9 and Fock-coupling screens zero. At 0.002/0.001 bohr, retained CCSD + iterative triples + PNO loss is -0.002000951287366 (frozen) and -0.002843610990470 (all-electron). These are correlation-only entries and use a different Fock-screen setting from some earlier ledger entries. The PNO pieces are -0.000475598626877 and -0.000329495672374; stationary/direct discrepancies are 9.67e-12 and 4.13e-12. Existing frame, rank, residual, adjoint, triples and retained-energy checks pass. To reproduce, call `configure()` from the new integration script, set these three nonzero PNO/TNO cutoffs, then call `benchmark(frozen, (5,), include_pno=True)` for both core choices.

**Validation and next step.** Native build/install passes. `dfccsd-t-grad1`, `dlpno-grad-model`, `dlpnocc-1` and the expanded `dlpnolocal-1` pass. The rotating-localization algebra oracle, eight standalone Boys/PM mixed-response cases, Cartesian/spherical cc-pVDZ and cc-pVTZ dipole checks, Lean record and LaTeX compilation pass. Maximum isolated mixed-coefficient error is 2.58e-7 (PM); maximum dipole derivative error is 1.04e-11. The LaTeX record has no overfull or undefined-reference diagnostics.

This completes the mixed-localization integration checkpoint, not a full molecular Hessian. Next replace sampled first/final PNO-frame response and the CCSD mixed nuclear residual analytically, including the separate PNO-loss diagnostic's lower-order inputs. Nonzero remaining screened-energy corrections and full Cartesian-matrix/invariance/vibrational validation still remain. The assembled molecular runner is still Boys-specific; standalone PM response tests do not establish a complete PM Hessian driver or local scaling.

## 2026-11-07 — Analytic mixed first PNOs and nuclear PNO-loss curvature

**Implemented.** Extended `dlpno_pno_curvature.py` with mixed fixed-domain occupied refits, fitted three-center panels and exchange matrices, moving Cholesky-pivot parents, and the first semicanonical MP2 PNO selection. Metric responses use panel solves. For fixed forward transform `X`, polar-normalize `N X` in the moving AO metric and differentiate the projected MP2 Sylvester equation through mixed order. This avoids derivatives of individual semicanonical eigenvectors and divisions by within-parent virtual-energy gaps. Apply the existing spectral-projector equations to the spin-adapted pair density, then construct the retained frame in the fixed-reference polar chart. Its AO contractions use rectangular panels rather than an AO-square retained projector. The actual forward frame determines the rank and gauge; inconsistent frames, non-full-rank selected pivot spans and unresolved retained/discarded occupation boundaries are rejected. Degeneracies within either spectral cluster are supported.

**Molecular connection.** `first_pno_mixed.py` supplies analytic RHF density/Fock, active Boys LMOs, virtual parent, raw CC fitting integrals and all first-stage PNO frames at one geometry. It reuses the native first-order LMP2 solver twice and builds the mixed LMP2 residual source with the mixed amplitudes zeroed. The existing isolated PNO-loss adjoint eliminates those unknown mixed amplitudes. Thus the complete two-stage PNO-loss curvature in this full-domain diagnostic has no nuclear-displacement parameter. The initial semicanonical MP2 mixed amplitudes are included; only the final-stage coupled-LMP2 mixed amplitudes are eliminated. The shared retained-CCSD/triples column now assembles this analytic correction. Its old sampled PNO-loss provider remains solely as an oracle.

**Algebra verification.** `first_pno_mixed_algebra.py` uses a separate displaced generalized-eigenproblem reconstruction with moving nonorthogonal AO metrics, shortened occupied-refit AO domains, selected pivot subsets, nonsymmetric off-diagonal pairs and symmetric diagonal pairs. The maximum retained-projector error is 2.48e-11. Mixed MP2 equations, metric orthonormality, AO-projector idempotency, parent-chart rotations, pair reversal, u/v reciprocity, full retained rank and degenerate energies/occupations within clusters pass. A degenerate retained/discarded boundary is rejected. Existing PNO scalar and LMP2 mixed-residual algebra tests also pass, with errors 4.76e-10 and 7.16e-10 respectively.

**Nuclear verification.** Water/cc-pVDZ, cc-pVDZ-JKFIT/RI, full AO/fitting domains, first/final PNO cutoffs 1e-5/1e-4, TNO cutoff 1e-5, retained pair cutoff 1e-9 and Fock-coupling screens zero. All retained pairs are tested. First derivatives agree with the independent pre-existing PNO tangent and native LMP2 residuals are below 4.3e-13. The projector column below differences that independent first-order response at 0.004/0.002 bohr; scalar checks independently difference the native PNO correction energy. Energy values and errors are in hartree/bohr².

| Core | Entry | Analytic PNO-loss curvature | Error versus native scalar differences | Maximum mixed AO-projector error |
|---|---|---:|---:|---:|
| Frozen O 1s | O-y/H1-z | +0.000143232585085 | -1.27e-10 | 9.18e-8 |
| Frozen O 1s | H1-z/H1-z | -0.000475596442271 | +8.77e-10 | 6.63e-8 |
| All-electron | O-y/H1-z | +0.000167771404807 | -1.96e-10 | 1.20e-7 |
| All-electron | H1-z/H1-z | -0.000329493154587 | +7.03e-10 | 8.05e-8 |

The maximum mixed-frame error against independently polar-aligned displaced frames is 2.36e-6. Direct second differences of AO projector values were noisier: errors span 1.36e-6 to 7.71e-6 at these spacings. An exploratory frozen mixed entry at 0.008/0.004 gave 5.96e-6, while a longitudinal entry improved from 4.47e-6 to 7.51e-7 at the wider spacing. These value stencils have substantial numerical sensitivity. Rather than relaxing their initial 3e-6 projector criterion, the acceptance test now differences independent analytic first projectors and uses a stricter 5e-7 criterion; the frame-value criterion remains 3e-6. Energy-curvature comparisons retain a separate 2e-8 threshold and all pass well below it. No tolerance on the assembled energy was loosened.

The actual native debug pair's normalized PAOs and selected Cholesky transform reproduce the full-virtual analytic mixed frame within 9.14e-13. In the tested reference, pair (0,1) retains 12 orbitals with occupation gap 5.34e-6 for frozen core, and 7 with gap 5.00e-7 all-electron. Mixed u/v reciprocity passes. Setting both PNO cutoffs to zero gives vanishing PNO-loss curvature while the full retained frames still move; the full-rank test explicitly checks this distinction.

**Integration and checks.** At the 0.004/0.002 bohr spacing still used by the other sampled blocks, retained CCSD + iterative triples + analytic PNO loss is -0.002000883696456 (frozen) and -0.002843531053320 (all-electron) for H1-z/H1-z. The analytic PNO contributions differ from the sampled stationary checks by 7.66e-10 and 7.89e-10. The existing CCSD, triples, residual and adjoint checks pass. These remain correlation-only diagnostic entries, not new canonical total-Hessian references. The install target, four regressions (`dfccsd-t-grad1`, `dlpno-grad-model`, `dlpnocc-1`, `dlpnolocal-1`), existing Lean record and updated LaTeX equation record pass. No new formal proof of the complete mixed-PNO algorithm is claimed.

Reproduce the new algebra and nuclear tests in the built Psi4 environment:

    python tests/dlpno-hessian-model/first_pno_mixed_algebra.py
    python tests/dlpno-hessian-model/first_pno_mixed.py

**Next.** The newly analytic source is ready for the mixed coupled-LMP2 equation `A t_uv = -R_uv[t_uv=0]`. Solve it with the native right-response operator and feed the mixed amplitudes into the final PNO density and retained frame. That will remove the remaining sampled final-PNO motion from the CCSD/triples inputs. The mixed nuclear CCSD residual is a separate remaining block. Nonzero other screened-energy corrections, actual screened molecular input assembly, full Cartesian invariance/vibrational checks and scaling work remain. The full-domain Boys diagnostic and fixed-domain algebra kernels do not establish a complete production analytic Hessian, PM molecular Hessian or local scaling.

## 2026-11-07 — Mixed coupled-LMP2 amplitudes and final PNO response

**Native solve.** Added `core.dlpno_lmp2_mixed_tangent(wfn, mixed_residual_source)`. It solves `A t_uv = -C_uv` with the existing native right operator, sparse retained graph, unique-pair convention, semicanonical preconditioner and DIIS/Jacobi solver. The source is the full moving-frame residual differentiated with `t_uv=0`; it includes mixed exchange/Fock/occupied energies, first-amplitude cross terms and both overlap legs of every oriented coupling. No full Jacobian or derivative of the iterative history is formed. The interface checks the complete pair graph, dimensions, finiteness and diagonal symmetry, preserves the caller's source and the forward amplitudes, and handles exactly zero sources without iterations. It records the mixed response residual and iteration count.

`coupled_lmp2_curvature` assembles that source, completes the mixed amplitude blocks and checks the solved equation with the independent Python moving-frame residual. `final_pno_mixed.py` propagates these amplitudes through all eight terms of the final pair-density mixed derivative, the retained/discarded spectral projector and the fixed-reference polar frame. The parent is the analytically moving first PNO space. Full final rank still includes parent motion. First derivatives are checked against the original independent orbital/first-PNO/LMP2/final-PNO tangent path.

**Integration.** The retained-CCSD/triples molecular provider now consumes analytic mixed final PNO frames. One PNO response is prepared per coordinate pair and reused across the comparison stencils and stationary contractions. The stationary callback rejects requests for sampled mixed occupied density, Fock, LMOs, virtual parent, PNO frames, eliminated CCSD amplitudes or trial TNO frames. The remaining sampled object is the mixed nuclear CCSD residual, including its displaced local geometry; the independent direct verification routes still use displaced states. The PNO-loss adjoint remains available and avoids this extra right solve when only that correction is wanted.

**Isolated nuclear verification.** Water/cc-pVDZ with full AO/fitting domains, first/final PNO cutoffs 1e-5/1e-4, TNO cutoff 1e-5, retained pair cutoff 1e-9 and Fock-coupling screens zero. Independently reconverged native models supply the pre-existing first-order responses; Richardson differences at 0.001/0.0005 bohr check the mixed AO LMP2 amplitude tensor and final AO projector over all retained pairs. The following are maximum absolute coefficient/tensor errors, separate from energy-curvature errors.

| Core | Entry | AO LMP2 amplitude error | Final AO projector error | Native mixed-solve RMS residual | Reconstructed mixed residual, max component |
|---|---|---:|---:|---:|---:|
| Frozen O 1s | O-y/H1-z | 7.99e-9 | 1.01e-6 | 5.96e-14 | 3.41e-13 |
| Frozen O 1s | H1-z/H1-z | 4.55e-9 | 1.07e-6 | 1.27e-14 | 1.02e-13 |
| All-electron | O-y/H1-z | 8.02e-9 | 1.04e-6 | 5.71e-14 | 2.39e-13 |
| All-electron | H1-z/H1-z | 4.67e-9 | 1.04e-6 | 8.20e-14 | 4.36e-13 |

Mixed metric orthonormality and projector idempotency pass; interchanging u/v agrees within 2e-9 for amplitudes and 2e-8 for final frames. The direct two-stage PNO-loss curvature with solved mixed amplitudes agrees with the existing stationary adjoint contraction within 1.1e-14 hartree/bohr² in this tighter coefficient fixture. Missing pairs, wrong dimensions, NaNs and asymmetric diagonal sources are rejected. The zero-source solve and preservation of both inputs and forward amplitudes pass.

**Precision investigation and unchanged acceptance.** At the earlier reference settings, the frozen H1-z projector difference was 2.58e-6 at 0.002/0.001 bohr. At 0.001/0.0005 it was 1.95e-6, while the all-electron result was 2.23e-6, just outside the new 2e-6 coefficient criterion. The all-electron value changed to 9.34e-6 at 0.004/0.002 and 2.46e-6 at 0.0005/0.00025, so smaller steps alone did not resolve the sensitivity. Tightening the mixed CPHF solve or LMP2 residual convergence separately did not materially remove it; those exploratory changes were not retained.

The isolated coefficient fixture now requests SCF integral tolerance 1e-14, SCF energy/density convergence 1e-14, orbital solver convergence 1e-14 and localization gradient convergence 1e-13. Its LMP2/CCSD `R_CONVERGENCE` remains 1e-11. These reference/orbital settings reduce all projector errors to at most 1.07e-6. The largest analytic projector components in the limiting pairs are about 111–180 per bohr². The amplitude criterion remains 2e-7 and the projector criterion remains 2e-6; no acceptance tolerance was relaxed. The assembled energy benchmarks retain their prior settings and independent criteria.

**Finite-cutoff column checks.** With the standard integration fixture and its remaining 0.004/0.002 bohr CCSD-residual stencils, retained CCSD + iterative triples + analytic PNO loss gives the following correlation-only entries (hartree/bohr²):

| Core | O-y/H1-z | H1-z/H1-z |
|---|---:|---:|
| Frozen O 1s | +0.004191804380112 | -0.002000856398485 |
| All-electron | +0.004128133472784 | -0.002843503139993 |

All existing retained-energy, triples, TNO, residual and adjoint checks pass. These finite-cutoff values are not canonical total-Hessian references. The direct-versus-stationary PNO-loss comparisons remain consistent with the previous checkpoint.

**Canonical full-rank check.** With zero PNO/TNO cutoffs and the existing nonzero retained-pair cutoff, the assembled total (analytic RHF + retained CCSD + ordinary iterative triples) was compared again with differences of canonical DF-CCSD(T) analytic gradients. The remaining CCSD mixed-residual stencils use 0.008/0.004 bohr. All component and total errors pass the unchanged 2e-7 criterion; the largest component error is 5.28e-9. Total errors, in hartree/bohr², are:

| Core | O-y/H1-z total error | H1-z/H1-z total error |
|---|---:|---:|
| Frozen O 1s | -2.76e-10 | -2.45e-9 |
| All-electron | -8.62e-11 | -2.48e-9 |

This exercises full retained PNO rank with moving parents in both core treatments. It validates these selected entries; it is not a full Cartesian Hessian or vibrational validation.

**Records and formal checks.** The `.md` ledger now indexes every Hessian milestone from both chats, including the earlier failed all-electron localization check and its resolution. The complete `.txt` equation export preserves all displayed equations from the updated `.tex` source; it includes the mixed LMP2 equation, the fifteen affine four-factor edge terms and all eight final-density mixed terms. The new Lean `mixed_four_factor_product` identity verifies their ordered product-rule regrouping. It does not prove the native solver or the entire molecular algorithm. Lean, the typesetting source, lossless equation export, native build/install and the four existing regressions pass.

Reproduce the new response checks in the built Psi4 environment:

    python tests/dlpno-hessian-model/final_pno_mixed.py
    python tests/dlpno-hessian-model/export_equations.py --check

**Next analytic block.** Replace `semi_analytic_ccsd_mixed_source` with the analytic mixed nuclear source of the t1-transformed CCSD residual in the same moving final-PNO charts. This includes the T1-dressed fitted integrals and intermediates, occupied couplings, projected singles and all explicit interpair overlaps; the existing native CCSD adjoints then contract that source. Keep the existing stationary elimination of mixed CCSD amplitudes. Other nonzero screened-energy corrections, screened-domain input delivery, PM molecular integration, full Cartesian symmetry/translation/rotation/vibrational checks and scaling validation remain separate work. A complete production analytic molecular Hessian is not claimed.

## 2026-11-07 — Analytic mixed nuclear CCSD residual

**Implemented.** `psi4/driver/procrouting/response/dlpno_ccsd_mixed.py` replays the complete RHF t1-transformed singles and doubles residual with four-component mixed tensors. Its molecular inputs are the AO overlap, active LMOs, Fock matrix, moving final PNO frames, local fitting metrics and three-center integrals. Both first CCSD responses are included; the mixed CCSD amplitudes are explicitly zero. The result is

    c_uv = R_x x_uv + R_xx[x_u,x_v] + R_xt[x_u,t_v]
           + R_tx[t_u,x_v] + R_tt[t_u,t_v],   J t_uv = -c_uv.

The replay includes direct dressed DF products, the virtual ladder, beta and both overlap legs, ordered gamma/delta and nonprojected J/K vertices, singles A/B/C, and all contracted/free-index Fock dressings (Jiang 75–101). The entire occupied double-dressed Fock matrix is completed before using the G contribution. Singles B rows accumulate into their destination diagonal-pair spaces. Reversed pairs transpose amplitudes while preserving each target's auxiliary fitting domain and actual rectangular virtual legs.

**Metric distinction.** Raw T1-dressed factors are contracted with the analytic spectral derivative of the product of the two native filtered half-inverses. MP2-origin exchange and nonprojected J/K retain the native unfiltered metric solve. This preserves the original residual algebra without selecting an auxiliary eigenvector gauge. The existing mixed-product implementation supplies every two-direction cross assignment, including the two equal-direction assignments for a diagonal Hessian entry.

**Integration.** `triples_curvature.benchmark` computes this source once per coordinate pair at the reference geometry and feeds it into both the retained-CCSD and ordinary iterative-triples stationary contractions. The source and the stationary trial providers reject requests for sampled mixed electronic inputs. `semi_analytic_ccsd_mixed_source` remains an independent native-residual oracle; it no longer supplies the assembled value. The source's CCSD path rejects nonzero mixed amplitudes and unsupported screened orbital/AO domain settings. There is no geometry step in the analytic value.

**Isolated verification.** `tests/dlpno-hessian-model/ccsd_mixed.py` checks finite-rank water/cc-pVDZ with first/final/TNO cutoffs 1e-5/1e-4/1e-5, both frozen O 1s and all-electron, and Oy/H1z plus H1z/H1z entries. It verifies the residual value against native replay, both converged first-response equations, u/v interchange, random symmetric pure-amplitude mixed directions against exact native quartic coefficient extraction, and rejection of non-affine CCSD input. The largest native value discrepancy is 8.85e-12; the largest first-equation residual is 5.80e-11. Pure-amplitude mixed discrepancies are 1.18e-15 (frozen) and 1.32e-14 (all-electron). These checks respectively retain 2e-10, 2e-9 and 2e-11 absolute acceptance thresholds. u/v interchange uses 2e-11.

**Numerical-oracle investigation, retained as evidence.** Two-spacing fourth-order residual differences at 0.002/0.001 bohr amplified native integral/frame cancellation: the frozen cross source differed by 3.46e-7 with the original fixture and 3.76e-7 with tighter SCF/localization. The tighter fixture uses SCF integral/E/D tolerances 1e-14, orbital solver 1e-14 and localization gradient 1e-13 (CCSD/LMP2 R convergence remains 1e-11). At 0.004/0.002, the frozen cross error fell to 2.15e-8 but the diagonal component error was 2.62e-7. The amplitude-only check was already at roundoff and the scalar adjoint errors stayed below 4e-10. Tightening the reference alone did not resolve the small-spacing oracle noise.

The isolated component test therefore uses three wider spacings, h=0.016, 0.008 and 0.004 bohr, with the sixth-order estimate

    c_FD = (c(h) - 20 c(h/2) + 64 c(h/4))/45.

The target component acceptance is 2e-7 and triples-adjoint acceptance is 2e-9. The wider all-electron stencil exceeded the old 0.999 occupied-overlap matching cutoff; the isolated test now permits 0.995, while the original default remains 0.999 for every other test. All original fixed PNO/TNO-rank, pair-label and response-edge checks remain active. This accommodates smooth occupied rotations over the larger displacement; it does not change a residual tolerance. The assembled energy check reports the noisier short-stencil component comparison but relies on the independent first-tangent component check described next, its existing mixed-equation check, and independent gradient/adjoint checks.


The standard sixth-order residual oracle passed both cross entries (6.26e-9 frozen, 8.61e-9 all-electron) and the frozen diagonal (9.81e-8), but the all-electron diagonal still differed by 1.55e-6 in four core-singles components, with a triples-adjoint error of only -1.24e-12. A centered diagonal stencil excluding the reference residual gave 4.02e-6 at h=0.016 and 5.09e-7 at h=0.032. Thus removing the center alone did not cure the numerical comparison; neither of these exploratory component checks is represented as passing.

**Independent resolution.** Added `ccsd_mixed_tangent_check.py`, which differentiates the previously validated analytic first CCSD amplitudes at displaced geometries. It includes the derivative of the fixed-reference polar alignment: for A=Y_h^T S_0 Y_0=Q H, solve H Omega + Omega H = Q^T A_u - A_u^T Q, then transform both amplitudes and their derivatives. Merely rotating t_u would miss this connection term. Centered first differences at 0.004/0.002 bohr, with fourth-order extrapolation, produce t_uv; the native exact-polynomial Jacobian action then independently checks c_uv=-J t_uv. This uses the original first-response algebra and avoids a second subtraction of native residual values. Every source component must satisfy the same 2e-7 acceptance in every core/coordinate case. The native residual sixth-order check remains separately required for the triples-adjoint contraction at 2e-9, with its component discrepancy printed rather than hidden. The final combined test passed all four cases. Maximum first-tangent source component errors were:

| Core | Oy/H1z | H1z/H1z |
|---|---:|---:|
| Frozen O 1s | 1.21e-8 | 8.03e-9 |
| All-electron | 1.61e-8 | 6.87e-8 |

The standard sixth-order native residual adjoint errors were at most 4.36e-11. The all-electron diagonal residual-value discrepancy of 1.55e-6 is still reported by the final test; it is not claimed to satisfy the component criterion. All components independently satisfy that criterion through the transported-first-tangent oracle.

**Assembled finite-PNO results.** The retained-CCSD + iterative-triples + two-stage PNO-loss correlation curvatures use analytic inputs throughout; they exclude RHF and nuclear-repulsion curvature. Verification uses independent gradient differences and the existing mixed-equation/adjoint checks at 0.004/0.002 bohr. All four entries passed. Values are in hartree/bohr²:

| Core | Entry | Analytic correlation curvature | Retained CCSD vs gradient differences | Triples vs gradient differences |
|---|---|---:|---:|---:|
| Frozen O 1s | Oy/H1z | +0.004191800327785914 | -8.95e-12 | -1.62e-12 |
| Frozen O 1s | H1z/H1z | -0.002000840798949364 | -5.44e-11 | +6.60e-13 |
| All-electron | Oy/H1z | +0.004128129052544035 | -3.24e-11 | -3.44e-12 |
| All-electron | H1z/H1z | -0.002843487204113496 | -2.08e-10 | -3.44e-12 |

The largest independent PNO-loss comparison was 6.63e-10. The native second-amplitude-difference mixed-equation check was noisier for the all-electron diagonal (1.73e-5, below its existing 5e-5 criterion); the transported-first-tangent source test is the more precise componentwise check. Finite PNO ranks and rectangular transfers are retained. Relative to the preceding semi-analytic-residual checkpoint, the four analytic totals shift by about 4e-9 to 1.6e-8, consistent with removal of the residual stencil error.

**Canonical limit.** Full pair and triples virtual spaces, both core choices, Oy/H1z and H1z/H1z, and independent canonical analytic-gradient differences at 0.008/0.004 bohr passed the original 2e-7 component criteria. The largest total-Hessian discrepancy was 4.65e-11 hartree/bohr². The largest separate CCSD/triples component discrepancy was 5.43e-9. All screened/PNO energy corrections were explicitly verified zero on this full-rank comparison. No full Cartesian matrix or vibrational validation has been inferred from these four entries.

**Regression and records.** The module installs successfully; `dfccsd-t-grad1`, `dlpno-grad-model`, `dlpnocc-1`, and `dlpnolocal-1` all passed (4/4). The existing Lean stationarity and mixed-product identities still compile; they do not prove the complete residual implementation. Equation Section 20 documents the analytic source, metric conventions, contribution map, stationary use and independent transported-tangent oracle. The full `.txt` export contains 1,402 lines and remains synchronized with the `.tex` source; the latter compiles to 23 pages without overfull boxes or warnings. All prior milestone entries remain intact, and this checkpoint is step 23 across both chats.

**Scope and next step.** The reference-geometry full-domain stationary diagnostic now has analytic molecular inputs for the retained CCSD, iterative (T), and two-stage PNO-loss pieces, with no displaced-geometry input in their assembled values. This is not yet a general public analytic-Hessian driver. Next validate the full Cartesian matrix and invariances on this branch, then extend the screened molecular providers and account for nonzero other screened-energy corrections before production delivery and scaling claims. Assembled PM validation also remains open.


## 2026-11-07 — Complete Cartesian Hessians and rigid-motion checks

**Implemented.** `tests/dlpno-hessian-model/cartesian_hessian.py` assembles the full 9-by-9 water/cc-pVDZ matrix at the existing fixed-frame, nonstationary geometry. It prepares all nine analytic first-response directions once, then evaluates every ordered mixed entry independently. Analytic RHF electronic/nuclear curvature is added to retained CCSD, ordinary iterative triples, and both PNO truncation-loss contributions. Reverse entries are calculated separately; no matrix symmetrization, translation reconstruction or rigid-motion projection is used.

The analytic PNO construction's molecular panels are shared among the mixed CCSD residual and both stationary energy contractions. The existing provider wrappers accept these shared panels, while retaining their original default behavior. Every value and first component is checked against the independently prepared first response (2e-8 absolute); both linear CCSD residual equations are checked (2e-9). Any request for a sampled mixed electronic input raises an error. The initial driver probe needed `SAVE_JK = true` for the native gradient assembly's CPHF solve; enabling the same retention used by the public gradient driver resolved that setup error. No residual formula or acceptance tolerance was changed.

**Verification.** Both core choices were tested with finite first/final/TNO cutoffs 1e-5/1e-4/1e-5 and with all three cutoffs zero. All AO/fitting domains are complete. Screened weak-pair, SC-LMP2-pair and dipole corrections are explicitly checked zero at the reference and every displaced point; the full-space PNO-loss correction is also checked zero. At each displaced geometry, occupied labels, pair/triplet graphs and PNO/TNO ranks are checked on the same branch.

The finite-space numerical reference differences the analytic DLPNO gradient assembly used by the public driver, including both PNO losses. The complete-space reference instead differences canonical DF-CCSD(T) analytic gradients. All nine Cartesian columns use ±0.004 and ±0.002 bohr with fourth-order Richardson extrapolation. Neither numerical reference is symmetrized. Matrix comparisons and analytic symmetry/translation/rotation checks retain the 2e-7 absolute acceptance; net force and torque retain 2e-8. All four complete runs passed.

| Virtual spaces | Core | Max Hessian difference (hartree/bohr²) | Raw symmetry error | Translation residual | Rotation residual |
|---|---|---:|---:|---:|---:|
| Finite PNO | Frozen O 1s | 3.402e-08 | 7.448e-11 | 1.429e-10 | 7.513e-11 |
| Finite PNO | All-electron | 2.453e-08 | 7.427e-11 | 1.425e-10 | 6.566e-11 |
| Complete | Frozen O 1s | 1.739e-10 | 7.742e-11 | 1.479e-10 | 6.393e-11 |
| Complete | All-electron | 1.702e-10 | 7.740e-11 | 1.483e-10 | 7.036e-11 |

The identity residuals use the corresponding atomic units. The separate correlation component matrices also have maximum symmetry and translation errors below 1.62e-10. Thus the small total invariant residuals do not depend on cancellation between the retained-CCSD, triples and PNO-loss components.

**Rotation convention.** At a nonstationary geometry, the correct identity is H (Omega R) = Omega g, with Omega the block-diagonal generator of a rigid rotation. The gradient norms here range from 0.0155 to 0.0183 hartree/bohr, so setting the right-hand side to zero would be incorrect. The net torque sum_A R_A cross g_A is checked separately. No equilibrium frequencies or six rigid-body zero modes are claimed. The new Lean theorem `rotational_curvature` verifies that this covariance condition and antisymmetry of Omega imply (Omega R)^T H (Omega R) + g^T Omega^2 R = 0; it is an algebraic identity, not a proof of the complete electronic implementation.

**Reproducible records.** `cartesian_hessian_results.json` stores all four unmodified Hessians, their RHF and three correlation components, analytic reference gradients, independent numerical matrices, geometry, coordinate order, method/basis settings and error metrics. Its finite-space numerical-reference asymmetry is recorded as numerical error, not removed. Run each case in a built Psi4 environment:

    python tests/dlpno-hessian-model/cartesian_hessian.py --core frozen --spaces finite --output finite-frozen.npz
    python tests/dlpno-hessian-model/cartesian_hessian.py --core all --spaces finite --output finite-all.npz
    python tests/dlpno-hessian-model/cartesian_hessian.py --core frozen --spaces full --output full-frozen.npz
    python tests/dlpno-hessian-model/cartesian_hessian.py --core all --spaces full --output full-all.npz

`--stage analytic` assembles and checks invariances without displaced geometries. `--stage verify` checks an existing complete matrix against gradients. Each completed entry is checkpointed to the NPZ file; successful gradient verification also writes a JSON record. The script is a water validation driver, not new public `hessian()` routing.

**Checks and scope.** All four standalone matrix runs passed, Python syntax checks passed, the complete Lean file compiled, and equation Section 21 compiled without LaTeX warnings or overfull boxes (24 pages total). The complete `.txt` export is synchronized (1,468 lines). Native implementation and public derivative routing are unchanged in this checkpoint. All prior milestones and their unsuccessful exploratory checks remain documented; this is step 24 across both chats.

**Next.** Extend the molecular response providers to actual screened domains and validate the remaining nonzero screened-energy corrections. Equilibrium vibrational checks, assembled Pipek–Mezey validation, public Hessian routing and scaling remain. The full Cartesian fixed-domain validation planned at step 23 is now complete for both finite-PNO and canonical-limit water cases.

## 2026-11-07 — Screened-domain validation: production criterion not met

**User request and decision.** Validate screened domains and enable general production Hessians only if validated. The audit distinguishes a screened pair graph from actual screened molecular fitting domains. The sparse-graph algebra and selected screened-gradient checks pass, but the molecular mixed PNO/CCSD providers explicitly remain unsupported for local domains. No screened nuclear Hessian was assembled, so the conditional production criterion is **not met**. No driver registration, support guard, functional definition or numerical acceptance tolerance was relaxed.

**Added.** `core.dlpno_response_domain_state(wfn)` returns copies of the forward atom-indexed occupied/virtual AO transform domains, extended occupied and PAO labels, flattened PAO-pair masks, and unique pair occupied/PAO/auxiliary lists. It rejects non-gradient-model inputs. This is metadata, not a derivative provider. It does not expose AO Schwarz decisions. The new `tests/dlpno-hessian-model/screened_domains.py` disables AO integral screening and uses full triplet AO/fitting domains while testing genuinely local pair/refit domains. It checks the actual native lists, pair and triplet graphs, first/final PNO and TNO ranks, and response edges at each displaced geometry after occupied-label matching. Nonzero cutoff settings alone do not count as evidence of screening.

**Fixture and initial failed coverage check.** Four parallel H2 molecules at x = 0, 3.5, 7, 10.5 angstrom, with H atoms at z = ±0.37 angstrom, in a fixed C1 frame. RHF/cc-pVDZ, cc-pVDZ-JKFIT and cc-pVDZ-RI; Boys, no frozen core; first/final PNO/TNO cutoffs 1e-5/1e-4/1e-5, pair cutoff 1e-6, and zero Fock and AO-integral screens. Seven of ten unique pairs survive; two CCSD couplings have a missing partner kj pair. The first local-domain attempt at coefficient cutoffs 1e-4 did reduce PAO/auxiliary domains but left **all AO transform domains at 40 AOs**. The new fingerprint assertion rejected that insufficient AO-screening fixture. A short cutoff survey gave 30–40 AOs at 1e-3 and the final distinct occupied/virtual domains at 1e-2. The final test uses T_CUT_DO=1e-2, T_CUT_MKN=1e-3, T_CUT_CLMO=T_CUT_CPAO=1e-2, T_CUT_DO_IJ=T_CUT_DO_UV=1e-5 and T_CUT_DO_PRE=3e-2. These deliberately stronger coefficient screens exercise the local refit; they are not recommended production presets.

| Domain in final fixture | Retained size | Complete size |
|---|---:|---:|
| Unique occupied pairs | 7 | 10 |
| Occupied AO transform, per auxiliary center | 20–30 | 40 |
| Virtual AO transform, per auxiliary center | 30–40 | 40 |
| PAOs per pair | 10–20 | 40 |
| Auxiliaries per pair | 28–56 | 112 |
| Occupied labels per pair | 2–3 | 4 |

**Passing checks and limits.** On the same sparse pair graph with full molecular fitting domains, the mixed CCSD replay reproduces the native residual to 1.4273e-14. Its mixed derivative for arbitrary affine amplitude directions agrees with exact native quartic extraction to 6.1176e-17. This verifies sparse amplitude algebra, not nuclear derivatives of local refits. Replays restore the native amplitudes.

On the final genuinely local-domain model, the native raw DF replay error is 9.0206e-17; the LMO-refit and DF coefficient adjoint probe errors are 3.9810e-12 and 3.4073e-12. The analytic gradient assembler used by the public driver was compared with independently reconverged energy differences at ±0.004/±0.002 bohr, with fourth-order Richardson extrapolation and unchanged 2e-8 acceptance. All native selection fingerprints match across these eight displaced calculations.

| Zero-based coordinate | Analytic gradient (Eh/bohr) | Analytic minus Richardson (Eh/bohr) |
|---:|---:|---:|
| 2 (atom 0, z) | 0.015397233005259341 | 5.75886e-11 |
| 18 (atom 6, x) | -0.00006274620426226838 | -2.17902e-11 |

Only these two nuclear gradient components were checked. No full screened Hessian, Hessian invariance test, screened frozen-core/PM test or equilibrium frequency calculation follows from them. This fixture's final PNO selection retains its first-parent rank; the existing finite-water results remain the evidence for nonzero final-stage loss, not this chain.

**Unresolved diagnostic.** The existing `DLPNO PNO PROJECTOR OPERATOR DIRECTION ERROR` is 1.29868e-7, above the historical 1e-8 threshold. It is a synthetic first-PNO operator test using one central-difference spacing, h=1e-7. Its step-size/cancellation error has not been separated from possible derivative error here. The passing nuclear gradient comparisons do not excuse or replace it. The result is preserved as unresolved, without tolerance relaxation; other recorded native directional probes are below 1e-8. In particular, the PNO truncation DF probe is 3.17214e-9.

**Unsupported Hessian providers.** `mixed_ccsd_residual` rejects the actual local-domain model before consuming any electronic callback; `first_pno_mixed.molecular_inputs` also rejects it. Both messages are saved in the JSON audit. Local forward factors use occupied overlap refits, different occupied/virtual AO slices, extended-label zero masks and the retained PAO-pair mask before PNO projection. Full-AO L/Q/Y contractions and a global canonical virtual parent cannot simply be substituted. Equation Section 22 writes the mixed refit solve and screened raw-factor structure. The existing algebraic mixed panel primitive is not yet a complete molecular provider.

**Energy-definition control.** With matching retained-pair/triples choices, the gradient-model energy is -4.653500030876629 Eh and ordinary energy is -4.653501443256336 Eh. Their -1.4123797074e-6 Eh difference equals the ordinary eliminated-pair SC-LMP2 estimate to the numerical precision shown. The gradient model explicitly zeros this term; weak-pair and dipole terms are zero in this control. This is the already documented gradient functional, not a newly discovered missing term within that functional. General routing for the same gradient-model Hessian must preserve that definition. Claiming the Hessian of the ordinary screened energy would additionally require derivatives of its estimates and other differing pair/triplet choices.

**Reproduction and release result.**

    python tests/dlpno-hessian-model/screened_domains.py --require-production

The completed audit writes `screened_domains_results.json` containing native domain labels, counts, graph errors, all native directional probes, gradient references, energy components, rejection messages and explicit `screened_hessian_validated: false` / `production_ready: false`. With `--require-production` it exits **2**, deliberately reflecting the unmet production criterion. Without that flag, exit zero means only that the audit ran and its individual assertions completed. An explicit public `hessian('dlpno-ccsd(t)', dertype=2, return_plan=True)` request still reports requested derivative level 2 > available level 1; no numerical fallback is represented as analytic.

**Build and regression verification.** The native metadata accessor compiled successfully. The four existing CTest cases `dfccsd-t-grad1`, `dlpno-grad-model`, `dlpnocc-1`, and `dlpnolocal-1` all passed (34.45 seconds total). No energy, residual or derivative formulas changed. Python syntax and whitespace checks passed; equation Section 22 compiled without warnings or overfull boxes (26 pages total), and the complete 1,560-line `.txt` export passes its synchronization check. The old complete Hessian matrices were not recomputed because their implementation is unchanged. The milestone index and full working-equation `.txt`/`.tex` records include this audit and preserve all prior 24 steps, including the initial coverage assertion failure above.

**Next faithful implementation.** Deliver analytic first/mixed local fitting factors and the actual local PAO parent response from native domain/selection data; propagate them through first/final PNOs, TNOs and the mixed CCSD residual. Resolve the single-step PNO operator probe separately. Then validate complete screened nuclear Hessians against gradients of the same model, with raw symmetry and rigid-motion checks. General molecule support, the public Hessian/dipole-gradient contract, broader localization/core coverage, vibrational validation and scaling remain before a general production claim. The ordinary-energy corrections are a separate scope expansion if that different functional is desired.
