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
