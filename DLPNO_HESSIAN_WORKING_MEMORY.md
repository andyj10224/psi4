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
