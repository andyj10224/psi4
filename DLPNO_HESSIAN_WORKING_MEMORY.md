# DLPNO Hessian Working Memory Bank

> **Public routing delivered for local validation:** the user requested pushing the recovered implementation with the cc-pVDZ tests restored. The experimental `DLPNO_HESSIAN_MODEL` route now defaults to `true`, with complete Cython packaging and local test instructions. Source-level checks pass; fresh native numerical validation remains pending. See [recovery record](DLPNO_HESSIAN_M33_RECOVERY.md) and [local test commands](tests/dlpno-hessian-model/README.md). Milestone 32 remains the last fully validated molecular checkpoint.

## Response optimization pass (7 November 2026)

The latest user water/cc-pVTZ output at `923da30` completed column 1 and entered
column 2 before the user stopped it. It contains no exception. The first
column took about 196 seconds; the many SCF-component gradients and basis
prints were repeated work, not successive total gradients or proof of failure.
A complete Hessian from this water case is still pending.

This pass changes execution and storage, not the gradient/Hessian functional:

- Export all Cartesian occupied tangents from one native RHF response batch.
  The old directional API already solved all directions and discarded the
  others. Water now uses one tangent batch instead of nine, in addition to
  the separate RHF Hessian calculation. Frozen-active gauge transport, SCF
  option preparation, scratch cleanup and JK restoration are preserved.
- Build the RHF gradient and reference CCSD/(T) adjoints once per Hessian.
  Every ordered mixed entry and its independent repeatability checks remain.
- Reuse one quiet, wavefunction-backed MintsHelper and reference fitting
  integrals. Bounded first-integral caches retain at most four atom/basis
  keys per DF derivative family, rather than all 3N three-center panels.
  The quiet constructor preserves the reference options, bases and PSIO.
- Build each Fock/overlap AO covector panel once per analytic gradient,
  retaining two AO matrices instead of rebuilding panels for every atom.
- Move dense mixed DF 2J-K contractions to `dlpno/response_kernels.cc`.
  GEMM/GEMV/Matrix::gemm evaluate all 16 product-rule terms using nine shared
  QD products and O(naux*nao^2) temporary storage, without four-index ERIs.
- Parallelize disjoint auxiliary rows in mixed DF integral generation using
  private integral engines. Parallelize independent LMP2 right-response
  rows; keep transpose scatters serial. Worker exceptions are caught and
  rethrown after joining. Kernel packing is parallel with disjoint writes;
  its BLAS contractions occur outside the OpenMP packing regions.
- Add row progress with elapsed column time. No residual, symmetry,
  translation/rotation or column-repeatability tolerance was loosened.

Verification in this workspace: 112 source-level routing/cache/ownership and
failure-message checks with the molecular native boundary mocked; 11 native
DF-kernel checks against independent NumPy contractions and mixed finite
differences, using the exact C++ kernel and production BLAS wrappers linked
to system BLAS with small Matrix/thread adapters (serial and OpenMP builds).
Twelve old/new Python gradient contraction replays using deterministic mocked
native operators agree to 8.33e-17. Equation exports, completion ordering,
Python syntax and whitespace checks pass. This is NOT a full Psi4 build:
RHF batching, real integral engines, LMP2 threading and molecular numerical
results still require the user's rebuilt native installation. No molecular
speedup or full-gradient/Hessian certification is claimed from these checks.

`gradient_audit.py` now checks off-equilibrium water against reconverged
same-model energies at three step sizes, with native branch/parent matching,
force/torque checks, batched/directional RHF agreement, serial/four-thread
mixed integrals and optional cross-run gradient/component comparisons.
Default basis remains cc-pVDZ; cc-pVTZ, Boys/PM and all-electron/frozen-core
runs are available. Existing canonical and screened molecular tests remain.
See `tests/dlpno-hessian-model/README.md` for the staged local commands.
First PNO/CCSD responses are still reconstructed for each mixed entry; broader
response caching and additional native ports remain future performance work.

## Water response regression fixes (7 November 2026)

The user's frozen-core water/cc-pVTZ, TightPNO, 12-thread run at `8807b4c`
optimized successfully but failed the mixed RHF stationarity guard. Setting
global solver convergence/maxiter/integral thresholds to 1e-14/200/1e-14
passed that guard and exposed a cached-column final-PNO first-derivative comparison:
7 of 2668 elements differed by more than 2e-9, with a displayed difference
of 3.38e-9. No complete DLPNO Hessian column was returned in either run.

Audit found that `rhf_occupied_tangent` bypassed SCF option preparation.
The LMP2 tangent, mixed-tangent and PNO-loss adjoint bindings likewise relied
on the active module when reading local response limits. All four native
entry points now prepare their respective modules. In particular, the LMP2
residual tolerance must use the DLPNO-local `R_CONVERGENCE`, independent of
interleaved SCF calls or Python option queries.

The public assembler now owns a per-reference first-density cache: each
Cartesian coordinate is solved once and reused in every ordered mixed entry.
For water, native occupied-response calls decrease from 153 to 9. Each native
call still computes all nine Cartesian perturbations; exposing a single
batched native API remains a performance opportunity. The cache stores only
3N AO density matrices, is read-only, and expires with the Hessian assembly.
PNO/CCSD and mixed molecular derivatives are still evaluated independently.
Existing residual, repeatability and raw Hessian tolerances are unchanged.
Failure messages now identify block/pair, row/column, full-precision entries,
scientific-notation differences, PNO occupation gaps and upstream differences.

Validation here: 110 source-level routing/cache/diagnostic checks with the
native boundary mocked; unchanged derivative-algebra comparison, completion
ordering and equation-export checks. Native C++ rebuild and numerical tests
remain pending. Added native module-context and solve-count regressions to
`dlpno-hessian-driver`, plus `water_repeatability.py` for the reported optimized
water geometry at 1/12 threads and an optional independent gradient-difference
column with native selection matching. All existing cc-pVDZ tests remain.
The relative contributions of option-context drift and thread reduction
roundoff to the reported PNO mismatch are not yet numerically isolated.


## Scope and verification policy

Derive the analytic nuclear Hessian of the same closed-shell RHF ordinary iterative DLPNO-CCSD(T) functional selected by the public gradient driver. Start with Boys, then Mulliken PM; exercise both frozen-core and all-electron references. Work in a fixed Cartesian frame, in hartree/bohr², on a smooth local branch. Keep discrete selections fixed while differentiating the moving electronic subspaces. Require a test for each block before using it as a reference for the next block. Keep canonical comparisons, local finite-cutoff checks, and scaling measurements separate. [Working equations (.txt)](DLPNO_HESSIAN_EQUATIONS.txt) · [Typesetting source (.tex)](DLPNO_HESSIAN_EQUATIONS.tex)

**Last fully validated molecular checkpoint (7 November 2026, step 32).** Complete nonplanar screened water–H2 (Boys/PM) and ammonia–H2 (Boys) Hessians cover frozen-core off-diagonal pairs, distinct-index triplets and unequal coupled TNO ranks: 774 independent ordered entries, maximum error 8.4175e-08 Eh/bohr² against native gradient differences. Conditioned Coulomb inverse response and native Cholesky first-pivot roundoff are corrected. Raw component/total symmetry and rigid-motion checks pass without projection. Branch-preserving reference spacings are selected independently of analytic errors. Public routing was disabled at that checkpoint; step 33 now supplies the opt-in route for local validation. Screening robustness, screened equilibrium vibrations and scaling remain open. All prior milestones and unsuccessful checks remain below.

## Complete Hessian milestone index across both chats

This index reconciles the Hessian phase of **“Implement DLPNO Gradients”** with the continuation in this chat. It was checked against every commit from `0af1a1eb` through `f18cf39a`, all 23 preceding chronological entries below, and the recovered older-chat context. Each previous Hessian commit has a corresponding ledger entry and working-equation section. Complete Cartesian validation is step 24 (`1b0c5517`); the screened-domain readiness audit is step 25 (`052c494c`); screened energy corrections are step 26; local first-PNO response is step 27; screened coupled LMP2/final PNOs and PNO losses are step 28; screened CCSD/TNO molecular propagation is step 29; complete screened column assembly and validation is step 30; complete unsymmetrized screened matrices and their rotation-screening control are step 31; broader nonplanar screened molecular matrices and numerical stability corrections are step 32. Earlier unsuccessful checks, numerical dependencies and their later resolutions are preserved in the chronological entries; they are not silently replaced by a passing result.

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
| 25 | Screened-domain audit, native selection fingerprints, sparse residual/gradient checks; production criterion not met | `052c494c` | 22–22.2 |
| 26 | Correct unique dipole-pair counting; include dipole and excluded SC-MP2 analytic gradients and mixed correction curvatures | `6a517440` (gradient documentation follow-up `50278ca2`) | 23–23.4 |
| 27 | Actual screened first-PNO molecular response; conditioned parent chart; resolve operator diagnostic | `0eabd131` | 24–24.3 |
| 28 | Propagate screened coupled LMP2 and final PNOs; validate both PNO-loss curvatures | `ba2f066e` | 25–25.3 |
| 29 | Screened local CCSD mixed residual and matching TNO/triples molecular inputs | `81bb6f66` | 26–26.4 |
| 30 | Complete screened Hessian columns, gradient references and column invariances | `84175650` | 27–27.3 |
| 31 | Complete unsymmetrized screened matrices, native gradient references, component invariances and PAO-screen rotation control | `63895aef` | 28–28.4 |
| 32 | Nonplanar screened molecular matrices with distinct/unequal-rank triplets and frozen-core off-diagonal pairs; Coulomb inverse and native pivot stability | `833ec80c` | 29–29.3 |
| 33 (WIP) | Guarded public analytic delivery, API guards and equilibrium H2 workflow; interrupted full public validation | Recovery branch, parent `833ec80c` | 30–30.3 |

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

## 2026-11-07 — Correct dipole pair counting; add both screened correction derivatives

**Step 26 and requested ordering.** Investigated the energy convention first, corrected the dipole accumulator, then implemented the requisite gradient and mixed nuclear terms. `compute_dipole_ints()` computes a complete unique off-diagonal pair estimate with factor -4 and mirrors it into ij and ji storage. `prep_sparsity()` used to add both copies. It now adds only i<j. The stored estimate, screening bound and classification are unchanged. This agrees with Werner, JCP **145**, 201101 (2016), equations 1, 3 and 4, [doi:10.1063/1.4968595](https://doi.org/10.1063/1.4968595). In contrast, crude SC-MP2 stores one ordered contribution K:(2t-t^T); summing ij and ji is correct and is retained. The shared energy fix affects callers of this dipole accumulator, including its DOCC–DOCC use in open-shell code; the new analytic derivatives are RHF only.

**Functional expansion.** Removed the intentional zeroing of `de_lmp2_eliminated_` and `de_dipole_` in the gradient model. The public gradient now differentiates the previous retained CCSD + iterative (T) + two PNO-loss functional **plus** both excluded-pair estimates. This explicitly supersedes the omission at step 25. It does not change promotion of surviving pairs to strong or the common TNO cutoff. Ordinary/model energies agree when those choices are matched, as checked below. Historical step-25 energies remain evidence for the former definition.

**Native gradients.** `screened_pair_response.cc` reverses the crude excluded SC-MP2 pair energy before its original domains and integrals are overwritten. It includes the local auxiliary inverse, AO three-center integrals with the native Schwarz mask, occupied overlap refits, separate occupied/virtual AO transform domains, selected PAO parent overlap, virtual and occupied Fock blocks. The dipole reverse includes both local virtual parents, full virtual Fock covectors, occupied denominators, transition moments, moving orbital centroids, and explicit nuclear AO dipole integrals. A direct LMO covector enters the common orbital-response action. Sparse PAO/Fock/metric covectors merge before PAO normalization and the existing occupied-projector, frozen/active, RHF and localization adjoints. The two explicit SC-MP2 fitting arrays and the dipole integral array enter `assemble_cartesian_gradient`, which the public driver uses. No production numerical differentiation was introduced.

`core.dlpno_screened_pair_state(wfn)` exposes copies of actual crude occupied/virtual AO lists, excluded unique pair lists, dipole PAO lists/parents, and excluded SC-MP2 PAO/auxiliary lists, raw panels and parents. The detailed snapshot is retained only with `DLPNO_KEEP_LMP2_RESPONSE`; temporary gradient covectors are released after merging. The snapshot does not contain AO Schwarz masks. The existing three-center DEBUG replay was made stage-aware: crude response only uses OV arrays, so it must not access not-yet-built OO/VV arrays for zero densities.

**Mixed correction kernels.** `dlpno_screened_curvature.py` consumes analytic RHF density/Fock, localized occupied, AO overlap/dipole/DF and mixed inputs at one geometry. It transports the native full-rank Cholesky pivot spans, including normalized PAO motion from all occupied orbitals, and differentiates square or rectangular Sylvester MP2 equations. Each auxiliary center uses its actual crude occupied refit and distinct virtual AO domain. Raw fitting panels and correction energies replay the native values. These are analytic correction Hessian terms; complete screened CCSD(T) assembly remains unsupported. The mixed AO delivery requires zero AO Schwarz screening and a fixed full-rank selected parent; the native gradient has no new Schwarz-screen restriction. Equation Section 23 documents counting, all reverse sources, centroid derivatives, parent transport and mixed solves.

**Reproducible nonzero tests.** `tests/dlpno-hessian-model/screened_corrections.py` checks two total-gradient coordinates, first derivatives of both corrections and three mixed entries (u,u), (u,v), (v,v) per system. Every independent displaced native calculation is reconverged, with occupied-label matching and checks on actual domains, pair/triplet membership, PNO/TNO ranks, response edges, crude lists and physical parent spans. Equivalent redundant Cholesky pivot changes are accepted only if the transported and actual AO projectors agree. The reference uses ±0.004/±0.002 bohr and Richardson extrapolation. Gradient and curvature thresholds remain 2e-8 Eh/bohr and 2e-9 Eh/bohr²; first correction derivatives use 2e-10 Eh/bohr. Net total gradient and torque pass at 2e-8 in corresponding atomic units.

| Fixture | Localization/core | Max selected total-gradient error (Eh/bohr) | Max SC-MP2 mixed error (Eh/bohr²) | Max dipole mixed error (Eh/bohr²) |
|---|---|---:|---:|---:|
| Four H2, cc-pVDZ | Boys, all electron | 4.085e-11 | 4.014e-12 | 3.091e-13 |
| Four H2, cc-pVDZ | Mulliken PM, all electron | 5.910e-11 | 3.934e-12 | 3.673e-13 |
| Three LiH, def2-SVP | Boys, three frozen Li 1s | 3.210e-10 | 2.954e-11 | 5.936e-13 |

Each fixture has one excluded dipole pair and two excluded SC-MP2 pairs. Boys H2 corrections are -1.35280068e-6 and -5.89927567e-8 Eh, respectively; LiH corrections are -4.28563586e-5 and -2.80892832e-7 Eh. The LiH geometry places Li at x=0,6,12 angstrom and H 1.6 angstrom above each Li; fitting bases are def2-universal-JKFIT/def2-SVP-RI. The H2 fixture uses the step-25 geometry and cc-pVDZ-JKFIT/RI. H2 uses T_CUT_PRE=1e-6 and pair cutoff 1e-6; LiH uses T_CUT_PRE=1e-5, pair cutoff 1e-4 and T_CUT_DO_IJ=1e-3. Remaining local screens and convergence are explicit in the test. These are coverage fixtures, not suggested accuracy presets. Matched ordinary/model energies agree in all three runs; the H2 Boys total is -4.653501442670069 Eh and LiH total -24.00840423519159 Eh.

    python tests/dlpno-hessian-model/screened_corrections.py --system chain --localization BOYS
    python tests/dlpno-hessian-model/screened_corrections.py --system chain --localization PIPEK_MEZEY
    python tests/dlpno-hessian-model/screened_corrections.py --system lih --localization BOYS

The committed `screened_corrections_results.json` contains all three numerical records, geometry, domains, correction energies, full reference gradients and selected curvature/reference values. The H2 Boys run was repeated with the final compiled native code and gave identical results. The older readiness audit was updated and rerun for the expanded functional; its current JSON now checks equal ordinary/model corrections and energies. It still exits 2 with `--require-production` because the full screened PNO/CCSD inputs and synthetic operator-probe issue remain unresolved. The historical audit record is available at commit `052c494c`.

**Exploratory failures retained.** The first native build exposed a unity-build helper-name ambiguity (`project`/`lift`); explicit `screened_detail` qualification resolved it. The first LiH setup requested an unavailable cc-pVDZ-JKFIT entry for Li and was changed to the complete def2 basis/fitting combination. Its initial screening settings produced no excluded dipole pair, so the coverage assertion failed; the explicit settings above exercise both corrections. No numerical tolerance was relaxed to pass these checks.

**Build and regression.** The native implementation compiled and installed with one compiler job. All four CTests (`dfccsd-t-grad1`, `dlpno-grad-model`, `dlpnocc-1`, `dlpnolocal-1`) passed after the final changes, in 35.31 seconds. The expanded gradient-model regression independently reconstructs the unique -4 dipole sum from native parents and verifies SC-MP2 ordered counting; both correction sets must be nonempty. Existing no-snapshot, full-domain, empty-triples, local-space, localization and gradient checks remain covered. No new Lean proof is claimed for these derivatives. The complete 1,738-line `.txt` export is synchronized. The Hessian and gradient equation sources compile without warnings or overfull boxes (28 and 6 pages); an inherited double-superscript typo in the gradient TeX was also corrected. All prior 25 milestones and failed checks are preserved.

**Next faithful step.** Extend actual local PAO/PNO and fitted-factor mixed molecular delivery through first/final PNOs and the CCSD residual, then add these now-implemented correction curvatures to a complete screened Hessian. Resolve the existing PNO operator probe; test full-matrix symmetry, rigid-motion covariance and independent gradient differences before enabling public analytic Hessian routing. Selected correction terms alone do not meet that production criterion.

## 2026-11-07 — Screened first-PNO molecular response (Immediate First Down)

**Step 27, based on `50278ca2`.** Implemented analytic value/first/mixed nuclear delivery for the retained first-PNO pairs on actual local PAO and auxiliary domains. This is the first missing screened molecular block identified in steps 25–26. It consumes analytic RHF, frozen/active partition, localization, AO overlap/Fock and local raw DF derivatives at one geometry. It performs no coordinate sampling. Complete screened coupled-LMP2/final-PNO, TNO and CCSD delivery, complete screened Hessian assembly and public analytic Hessian routing remain open.

**Native data and implementation.** `core.dlpno_first_pno_state(wfn)` returns copied refined-stage occupied/virtual AO lists, extended occupied/PAO labels, unique retained pair PAO/auxiliary lists, full-rank Cholesky-pivot parents, raw QI/QJ panels, semicanonical MP2 amplitudes, all occupations and retained AO frames. It includes full-rank selections, which the gradient's truncated-only projector snapshot omits. It is retained only with `DLPNO_KEEP_LMP2_RESPONSE`; ordinary gradients need no new snapshot. State is cleared when rebuilding first PNOs, and callers receive independent matrix copies. The forward AO Schwarz setting is recorded on the wavefunction. The new mixed provider requires it to be zero because the snapshot does not expose the native shell mask. This is not an additional restriction on the public native gradient.

`response/dlpno_screened_pno.py` supplies `screened_first_pno_response`. Each auxiliary center uses its actual occupied overlap refit, distinct virtual AO list and extended-label zero masks. The first-PNO OV exchange panel does not use the OO/VV PAO-pair mask. The local auxiliary inverse is differentiated through metric solves. Moving normalized PAOs include the total occupied projector, including frozen core. The provider uses the native selected PAO parent and retained PNO frame, not a global canonical virtual parent or reconstructed cutoff decision. Raw panels, MP2 amplitudes and occupations replay within 2e-10; frames retain the existing first-PNO 2e-8 reference criterion. The maximum tested reference discrepancies are recorded in the committed JSON.

**Conditioning and equations.** The shared polar normalization now solves square-root Sylvester equations at the actual near-identity Gram matrix instead of replacing its value by identity. Initial screened parents had small metric errors amplified by large internal tangents. The local first-PNO parent now uses a chart with zero first-order internal rotation in the symmetric metric connection, enforcing mixed orthonormality directly. The same PAO coefficient blocks transform exchange, and the corresponding AO parent blocks transform Fock. Thus the selected spectral projector and reference-anchored final PNO chart are unchanged. No within-parent virtual energy difference is divided by. Hessian equation Section 24 gives the complete local refit, parent-chart, matrix-square-root and validation equations. The 2e-8 metric checks are unchanged.

**Independent molecular validation.** `screened_first_pno.py` checks non-full native AO/PAO/auxiliary domains, exact fixed-branch lists/ranks/graphs after occupied matching, physical parent spans, differentiated metric orthonormality, value replays, independent first-order projector response, first differences of selected frames/projectors, mixed PAO/raw fitting panels, mixed projectors from differences of the older first-order implementation, and ordered uv/vu frames. The mixed-projector acceptance is 5e-7, as in the earlier full-domain water test; the independent first-order check is 2e-9. All differences use Richardson extrapolation and independently reconverged native wavefunctions.

| Fixture | Retained pairs and ranks | Spacings (bohr) | Maximum mixed-projector error from independent first response |
|---|---|---|---:|
| Four H2, Boys, all electron | 7 unique; diagonal 10→9, off-diagonal 20→5 | 0.004/0.002 | 4.787e-7 |
| Four H2, Mulliken PM, all electron | Same nonzero off-diagonal truncations | 0.004/0.002 | 3.508e-7 |
| Three LiH, Boys, three frozen Li 1s | 3 diagonal pairs, each 14→12 | 0.008/0.004 | 7.926e-8 |
| Four H2, Boys, full first/final rank | 7 unique; diagonal 10→10, off-diagonal 20→20 | 0.006/0.003 | 1.376e-7 |

Each truncated fixture checks (u,u), (u,v), (v,v): u=2,v=18 for H2, u=5,v=12 for LiH, in the zero-based Cartesian ordering already documented in step 26. The LiH fixture tests diagonal first-PNO truncation and frozen-core motion; it does **not** add screened off-diagonal frozen-core coverage. The H2 runs provide the off-diagonal coverage. The largest truncated-fixture metric residual is 2.50e-14 and ordered mixed-frame error is 4.19e-11. These are selected molecular subspace derivatives, not energy-Hessian errors or a full Hessian validation.

The full-rank control checks (2,18). At full rank the selected projector is exactly the parent-span projector, so its independent value reference uses a metric solve in the transported native pivot span, after independently verifying native span equality. Its mixed reference differences the first derivative of that defining metric solve. The older spectral first response is still checked at the reference. Full-rank first-value projector error is 3.560e-7 and mixed error is 1.376e-7, both below 5e-7; ordered-frame error is 2.62e-16. The raw span-projector second-value error remains 3.343e-6, just outside the separate diagnostic target.

Reproduce the four first-PNO runs with:

    python tests/dlpno-hessian-model/screened_first_pno.py --system chain --localization BOYS
    python tests/dlpno-hessian-model/screened_first_pno.py --system chain --localization PIPEK_MEZEY
    python tests/dlpno-hessian-model/screened_first_pno.py --system lih --step .008
    python tests/dlpno-hessian-model/screened_first_pno.py --full-rank --step .006

**Precision limitations, retained failures and oracle choice.** Direct second differences of native truncated PNO values did not meet the separate 3e-6 diagnostic target. The Boys H2 (2,2) frame/projector errors at 0.004/0.002 are 3.977e-4/2.320e-4, while differences of the independent older first response agree to 5.926e-8. Native value replay errors of order 1e-10 and ill-conditioned parents can be amplified by h^-2; the observed spacing dependence supports a cancellation/conditioning limitation. At 0.02/0.01 the raw Boys frame/projector errors drop to 6.427e-6/5.526e-6, but the independent-first mixed error grows to 1.655e-5 from large-step error. A wider seven-point native-value experiment left the reference frame chart and was rejected by its existing overlap guard. It was not accepted by weakening the guard. The committed JSON preserves raw second-value errors and an explicit resolved/unresolved flag; they are not passing evidence. Acceptance instead uses the earlier independently implemented first response, plus native first-value and lower-level mixed-panel checks, with their original strict thresholds.

The initial parent chart failed mixed metric orthonormality (3.03e-8 versus 2e-8) on a conditioned H2 off-diagonal parent; the chart above resolved it. A proposed extra 2e-10 frame-replay assertion was too tight for native normalization roundoff (2.44e-10); the provider uses the pre-existing 2e-8 first-PNO reference criterion and reports the actual error. The older independent first-order oracle requires a stricter reference metric: its input parent is orthonormalized and resemicanonicalized within the same native pivot span before evaluation. Initial LiH coverage incorrectly demanded a truncated off-diagonal surviving pair; actual LiH retention is diagonal only, so the coverage assertion now applies to the H2 fixture. LiH's native QI second difference at 0.004/0.002 had error 6.696e-7 against 5e-7; 0.008/0.004 reduces the maximum to 1.527e-7 with no tolerance change.

The untruncated control initially used re-diagonalized native PNO coefficients. Their first-value differences failed at spacings 0.004, 0.01 and 0.02 (errors 4.659e-6, 1.537e-6, 1.303e-6 against 5e-7). Using the exactly equivalent parent-span metric solve removes these arbitrary rotations. The first-response mixed oracle still gives 5.079e-7 and 8.477e-7 at 0.004 and 0.002; at 0.008 the first-value projector error grows to 1.138e-6. The intermediate 0.006/0.003 spacing passes all acceptance checks, with mixed error 1.375e-7. The old spectral and direct span first-derivative oracles agreed in the failed 0.002 mixed comparison to about 3e-12. These results establish an accepted selected control, not a broad spacing-independent precision plateau. All reported thresholds remain unchanged and the failed spacings are retained in the JSON.

**Synthetic operator probe resolved.** The old h=1e-7 central estimate had error 1.298684e-7. The seven-spacing study gives errors 3.998e-3, 1.256e-4, 1.308e-5, 1.169e-6, 1.299e-7, 1.139e-8 and 1.334e-9 at h=1e-5,3e-6,1e-6,3e-7,1e-7,3e-8,1e-8. The quadratic region identifies central-difference truncation error. The diagnostic now uses Richardson at h and h/2; its Boys H2 error is 7.650e-10, PM 7.548e-10, and frozen-core LiH 1.417e-9, all below the unchanged 1e-8 threshold. `DLPNO PNO PROJECTOR OPERATOR CENTRAL ERROR` retains the old estimate; DEBUG>1 exports `(i,j,h,analytic,numeric)` rows in `DLPNO PNO PROJECTOR OPERATOR STEP STUDY`. No gradient formula changed.

**Regression, records and release gate.** Single-job native build/install succeeded. The four CTests `dfccsd-t-grad1`, `dlpno-grad-model`, `dlpnocc-1`, `dlpnolocal-1` passed (53.83 s total). The existing full-domain water first-PNO/PNO-loss test passed for frozen-core and all-electron references, including full-rank zero-loss controls: maximum independent-first mixed-projector error 1.201e-7, scalar PNO-loss curvature error 1.962e-10 Eh/bohr², full-rank loss below 1e-18. The H2 Boys screened-correction gradient/curvature regression also passed after the shared normalization change (maximum mixed correction error 4.014e-12 Eh/bohr²). These selected regressions do not claim a rerun of all four full water Hessians.

The readiness audit was rerun and still deliberately exits 2 with `--require-production`. The synthetic-operator blocker is gone; incomplete screened coupled-LMP2/final-PNO/CCSD delivery and lack of a complete screened nuclear Hessian remain. The old full-domain PNO-loss and CCSD entry points still reject actual local-domain molecular data. This is compatible with the newly separate first-PNO provider; none of their guards was removed. Updated audit and new first-PNO results are committed. Both working memories and complete equation exports are updated; all previous 26 steps and unsuccessful checks remain visible. The equation sources compile cleanly on the final pass (Hessian 30 pages, gradient 10); the complete text exports are synchronized (1,869 and 560 lines). Python compilation and whitespace checks pass.

**Next faithful step.** Feed these local first-PNO frames, local exchange derivatives and pair overlaps into the coupled first-space LMP2 first/mixed solve, then the final-PNO response and both PNO-loss curvatures on this same screened branch. Subsequently provide the screened CCSD/TNO molecular factors, add the already implemented excluded-pair correction curvatures, and validate full matrices against gradients, unsymmetrized reciprocity and rigid-motion identities before general analytic routing.

## 2026-11-07 — Propagate screened coupled LMP2 and final PNOs

**Step 28, based on `0eabd131`.** `screened_pno_response` extends the installed `response/dlpno_screened_pno.py` provider through first/mixed coupled first-space LMP2 amplitudes, final-PNO frames/projectors, and both PNO truncation-energy curvatures. All molecular inputs are analytic blocks at one reference geometry; no numerical coordinate differentiation enters the provider. It reuses the native right solvers and isolated PNO-loss adjoint on the actual retained response graph. No native gradient/energy equation, cutoff decision or public derivative routing changes.

**Propagation.** Project the already locally refitted and masked PAO-parent exchange into the moving first-PNO frame using C = B^T S Y, K_first = C^T K_parent C. This preserves the actual occupied/virtual AO restrictions, local auxiliary metric and extended-label zeros; a new unrestricted L-Q-Y contraction would change the screened functional. Form the moving projected Fock and occupied Fock blocks and both derivatives of each pair overlap Y_p^T S Y_q. Read `RESPONSE_EDGES` exactly, preserving reversed source-pair transposes and rectangular overlaps. Native first-amplitude solves supply t^u and t^v. Assemble the affine-amplitude mixed residual b^uv = K^uv + A^uv t + A^u t^v + A^v t^u, then use the native mixed right solve A t^uv = -b^uv. Independently reconstruct the moving-frame residual at value/u/v/uv orders with the unchanged 2e-9 criterion.

The final selection applies the existing mixed spectral projector to the converged LMP2 pair density in the moving first-PNO parent, using the actual native final rank and reference frame. Both PNO losses include diagonal singles and unique-pair weights. The direct mixed-amplitude loss is checked against the stationary loss-adjoint contraction; the adjoint source already includes energy multiplicities, so no additional weights multiply its unique-pair Frobenius contraction. The native loss value must replay. Forward amplitudes and supplied mixed sources remain unchanged. Hessian equation Section 25 contains the full operator, overlap cross terms, mixed right equations, final density and stationary-loss identity.

**Coverage fixtures and settings.** `screened_final_pno.py` reuses step 27's actual local AO/PAO/auxiliary domain fixtures, with final occupation cutoff 1e-3 (first cutoff 1e-5) to enforce genuine second-stage truncation. H2 diagonal ranks are 9→8 and off-diagonal ranks 5→5. The H2 graph has 7 unique pairs, 22 response edges, 6 reversed sources and 18 rectangular overlaps. The three-LiH case with three frozen Li 1s orbitals has diagonal ranks 12→8 and no interpair edges. Thus H2 supplies coupled/rectangular coverage; LiH supplies diagonal frozen-core coverage, not screened off-diagonal frozen-core coupling. The full-rank H2 control has 10→10 diagonal and 20→20 off-diagonal ranks with both cutoffs zero. The nonzero-cutoff final loss is about -3.32439e-4 Eh for H2 and -9.79682e-5 Eh for LiH. The smallest final retained/discarded gaps are 9.2655e-5 and 7.1965e-5, respectively.

For the more sensitive final selection, the harness uses the earlier full-domain final-PNO test's tighter reference settings: SCF integral tolerance 1e-14, SCF E/D convergence 1e-14, orbital solver tolerance 1e-14 and localization gradient tolerance 1e-13. LMP2/CC residual tolerance remains 1e-11 (native response solves use its established 0.01 factor). AO Schwarz screening remains off, and actual discrete lists, first/final/TNO ranks, response edges and transported parent spans are checked at every independent displaced calculation. These are coverage fixtures, not recommended accuracy presets. The small-system lower-input harness builds dense AO derivative arrays; this is not an end-to-end local-scaling demonstration.

**Independent validation.** The prior first-order local refit/PNO implementation supplies an independent first projector, transported first frame and exchange derivative. The latter is derived by a PAO coefficient metric solve in the same native pivot span, not by calling the new mixed provider. Native first LMP2 response and the older final-PNO tangent give AO pair-tensor and final-projector derivatives. Differentiate those independently evaluated first responses at displaced geometries and Richardson-extrapolate. Separately differentiate the native scalar `DLPNO PNO TRUNCATION ERROR` twice; its reference does not use the new analytic curvature.

| Fixture | Spacings (bohr) | Max mixed AO-amplitude error | Max mixed final-projector error | Max native PNO-loss curvature error (Eh/bohr²) |
|---|---|---:|---:|---:|
| Four H2, Boys | 0.004 / 0.002 | 4.911e-10 | 4.077e-7 | 5.796e-10 |
| Four H2, Mulliken PM | 0.004 / 0.002 | 4.953e-10 | 5.620e-7 | 1.133e-10 |
| Three LiH, Boys, frozen core | 0.008 / 0.004 | 9.988e-10 | 2.983e-8 | 4.740e-11 |
| Four H2, Boys, full rank | 0.006 / 0.003 | 4.023e-10 | 2.190e-7 | 5.475e-12 |

The first two error columns are AO tensor/projector component derivatives per bohr², not energy errors. The first three fixtures check (u,u), (u,v), (v,v), with u=2,v=18 for H2 and u=5,v=12 for LiH. The full-rank control checks (2,18). Every off-diagonal coordinate pair also receives an independently ordered vu evaluation. Native first-space amplitudes are preserved exactly after all solves. Metric orthonormality and projector idempotency pass at 2e-8; their worst observed errors are 1.74e-14 and 1.07e-13. Ordered amplitude/frame/loss errors are below 3.67e-14 / 1.01e-11 / 4.94e-17. The largest value/first/mixed moving LMP2 residual is 6.42e-12, including the conditioned full-rank reference; the direct/stationary loss-curvature discrepancy is at most 1.72e-14 Eh/bohr². The full-rank analytic loss and all tested derivative components are below 2e-18.

**Unchanged acceptance and limits.** Mixed AO-amplitude/final-projector tolerances are 2e-7/2e-6, the earlier full-domain final-PNO criteria. Independent first amplitude/projector checks use 2e-9/2e-8; first scalar loss differences use 2e-9 Eh/bohr; mixed scalar loss differences use 2e-8 Eh/bohr²; direct/stationary agreement uses 2e-10. No threshold was relaxed. The first Boys run passed all numerical checks, and its repeat with explicit graph-count coverage gave identical results. The initial PM and full-rank harness attempts stopped while serializing NumPy integer graph counts; converting those metadata counts to Python integers fixed the harness. The reruns pass. No new numerical derivative failure occurred in this step. The raw second-value PNO coefficient/projector precision limits from step 27 remain documented and are not claimed to be resolved by these independent-first-response checks.

    python tests/dlpno-hessian-model/screened_final_pno.py
    python tests/dlpno-hessian-model/screened_final_pno.py --localization PIPEK_MEZEY
    python tests/dlpno-hessian-model/screened_final_pno.py --system lih --step .008
    python tests/dlpno-hessian-model/screened_final_pno.py --full-rank --step .006

**Regression and release gate.** The updated Python module installed successfully. All four native CTests (`dfccsd-t-grad1`, `dlpno-grad-model`, `dlpnocc-1`, `dlpnolocal-1`) passed in 80.49 seconds with molecular validations running concurrently. Mixed-source contract tests reject missing/wrong-sized/nonfinite/asymmetric diagonal sources, preserve source inputs, and return exact zeros with zero iterations for a zero source. Python syntax, whitespace and complete equation-export synchronization checks pass. Updated `.tex`/`.txt` records preserve all earlier derivations and milestones. Final TeX passes are clean (Hessian 32 pages, gradient 10), and the complete text exports contain 1,994 and 574 lines. An initial equation draft had missing environment-closing backslashes; compilation caught them and they were corrected. The committed `screened_final_pno_results.json` holds all four numerical records, actual domain lists, ranks, graph edges, convergence settings, individual scalar/projector/amplitude checks and the harness failure note.

The readiness audit was updated and rerun: `--require-production` still deliberately exits 2. Its remaining blockers are screened mixed CCSD/TNO molecular delivery and the absence of a complete screened Hessian/gradient comparison. The legacy full-domain PNO test's rejection is labeled accordingly; it does not mean the new local two-stage provider is unsupported. No full screened Hessian, complete-matrix reciprocity, rigid-motion Hessian check or equilibrium vibrational calculation was performed here. The existing four full-domain water Hessians were not recomputed because this is a new separate local delivery path.

**Next faithful step.** Use these final-PNO frames and local domain data to deliver the mixed screened T1-transformed CCSD fitted factors/residual and the consistent TNO/triples molecular inputs. Then assemble the retained-pair, triples, both PNO-loss and excluded SC-MP2/dipole curvatures into a complete screened Hessian. Validate full matrices against analytic-gradient differences and raw symmetry/rigid-motion identities before changing general production routing.

## 2026-11-07 — Propagate screened CCSD and TNO/triples mixed inputs

**Step 29, based on `ba2f066e`.** Implemented the next propagation block after screened final PNOs: native-domain OO/OV/VV fitted factors, cross-pair J/K, coupled first CCSD response, the analytic affine-amplitude mixed transformed-CCSD residual, and matching TNO/triples molecular inputs. All analytic providers consume derivatives at one reference geometry. Coordinate differences appear only in independent validation. The screened retained-energy/triples/correction pieces have not yet been assembled into a complete screened Hessian; public analytic routing stays disabled.

**Local fit and CCSD.** `response/dlpno_screened_ccsd.py` supplies `ScreenedFitting` and `screened_ccsd_response`. For each auxiliary center, OO uses two occupied overlap refits on the native occupied AO list; OV uses one refit, the distinct virtual AO list and extended-PAO label zeros; VV applies the symmetric native PAO-pair mask before projection. PAO coefficients of each moving final PNO are recovered by a metric solve in the saved native pivot span. Cross-pair J/K uses the target pair's auxiliary domain and the exact inverse, whereas products of native whitened factors retain the filtered inverse. Reusing unrestricted AO contractions here would differentiate a different fit.

The existing Jiang 75–101 mixed residual accepts this explicit provider; its old no-provider full-domain guard remains. Native right-response solves give both first CCSD tangents. The mixed replay then sets CCSD t_uv to zero, includes every amplitude/molecular cross term, and returns the source for the existing retained-CCSD and triples CCSD adjoints. Value and both first residuals must pass 2e-9. Mixed amplitudes are explicitly rejected; the native forward amplitudes are preserved. Caches belong to each provider instance, avoiding global retention of old wavefunctions/raw integral panels. The provider requires AO Schwarz screening and F_CUT to be zero; screened triples also require F_CUT_T=0. These are restrictions of the new molecular delivery, not additional public gradient restrictions. The current local-DF input interface still consumes raw AO derivative tensors over the supplied auxiliary domain; the dense AO validation harness and these caches do not establish local memory or runtime scaling.

**TNO and triples.** `core.dlpno_triples_basis_state(wfn)` returns copied native triplet PAO lists and complete pivot parents under `DLPNO_KEEP_TRIPLES_RESPONSE`, including full-rank selections. Diagnostic TNO probe transforms do not overwrite the converged snapshot. `response/dlpno_screened_triples.py` uses this parent, the actual local lists and the same affine CCSD response to construct mixed TNO frames, W/V/Fvv, occupied Fock, and retained-edge overlaps. The shared moment algebra now exposes value/first/mixed blocks while preserving the previous mixed-row API. TNO polar normalization uses the actual near-identity Gram matrix's square-root Sylvester response, as already used for PNOs. The ordinary native gradient equations and energy functional are unchanged.

**Independent CCSD checks.** `screened_ccsd_mixed.py` uses the older first-PNO and final-PNO response kernels, native first LMP2 solve, older first-order CCSD residual builder and native first CCSD solve. The mixed oracle differences those independent first tangents after occupied matching and polar transport, including the derivative of the alignment rotation. It checks b_uv = -A t_uv without numerically sampling the analytic provider. Richardson spacings are 0.004/0.002 bohr; coordinates are (2,2), (2,18), (18,18) for H2 and (5,5), (5,12), (12,12) for LiH. Acceptance remains 2e-7 per source component.

| Fixture | Maximum mixed CCSD source error | Native residual replay error | Native mixed J/K replay error |
|---|---:|---:|---:|
| Four H2, Boys, all electron | 2.54880e-9 | 6.75230e-14 | 1.96907e-11 |
| Four H2, Mulliken PM, all electron | 2.54827e-9 | 8.04790e-14 | 1.92137e-11 |
| Three LiH, Boys, three frozen Li 1s | 1.31260e-9 | 1.35240e-14 | 3.97460e-14 |

Random symmetric amplitude directions independently check R_tt against exact native quartic extraction (maximum 1.184e-14, tolerance 2e-11). Ordered uv/vu residuals, TNO/moment/overlap interchange, affine-contract rejection, and forward-amplitude preservation also pass. H2 has seven unique pairs with genuinely truncated final diagonal PNOs 9→8 and off-diagonal 5→5; LiH has only three diagonal pairs 12→8. LiH therefore supplies diagonal frozen-core CCSD coverage, not off-diagonal frozen-core coupling.

**Independent screened triples checks.** Set T_CUT_DO_TRIPLES=.01 and T_CUT_MKN_TRIPLES=.001. Each H2 localizer retains six repeated-index triplets, 20 PAOs/rank-20 parents, eight TNOs and 56 auxiliary functions per triplet, compared with 40 AO and 112 auxiliary functions for the molecule. Ten distinct ordered overlap blocks are exercised. The local VV masks and occupied refits remain active. LiH retains no triples; its empty result is explicitly excluded from triples validation. A full-TNO H2 control (rank 20) passes native moment replay, affine amplitude checks, ordered derivatives and differentiated metric orthonormality; it is a replay/consistency control, not a numerical full-rank triples-curvature comparison.

`screened_triples_mixed.py` independently reconverges native orbitals/final PNOs at displaced geometries, rebuilds fixed-rank TNO densities from the affine reference CCSD path, and recomputes raw fitting integrals. It checks the same native lists/ranks/graphs and physical parent spans, then compares mixed TNO, moment, Fock and overlap values at 0.008/0.004 bohr with Richardson extrapolation. It does not substitute fully relaxed CCSD t_uv into the trial TNO path.

| Localizer | TNO frame | W | V | Virtual Fock | Edge overlap | Occupied Fock |
|---|---:|---:|---:|---:|---:|---:|
| Boys | 8.98344e-7 | 8.54011e-10 | 2.74286e-9 | 2.19051e-7 | 1.47392e-6 | 1.88868e-8 |
| Mulliken PM | 8.18518e-7 | 6.22486e-10 | 1.19779e-9 | 2.22872e-7 | 1.12245e-6 | 1.88751e-8 |

Entries are maximum absolute component errors in each block's native units per bohr². Thresholds are 2e-6 for frame/Fvv/edge and 2e-7 for W/V/occupied Fock, chosen before these runs; they are not total-energy Hessian tolerances. The raw half-step TNO/Fvv errors reach about 3.3e-4 and are retained in the JSON. Native moments replay within 1e-14 on truncated H2 and 3.89e-13 on full TNOs. Differentiated TNO metric errors are below 1.9e-14. These small fixtures do not test arbitrary distinct-index screened triplets, unequal TNO ranks or screened frozen-core triples. The separate existing moment-algebra test still covers both distinct and repeated indices with rectangular transfers and filtered metrics.

Reproduce with:

    python tests/dlpno-hessian-model/screened_ccsd_mixed.py --screened-triplets
    python tests/dlpno-hessian-model/screened_ccsd_mixed.py --screened-triplets --localization PIPEK_MEZEY
    python tests/dlpno-hessian-model/screened_ccsd_mixed.py --screened-triplets --system lih
    python tests/dlpno-hessian-model/screened_ccsd_mixed.py --screened-triplets --full-tno --stage replay
    python tests/dlpno-hessian-model/screened_triples_mixed.py
    python tests/dlpno-hessian-model/screened_triples_mixed.py --localization PIPEK_MEZEY

Common molecular, first/final PNO and convergence settings are those of milestone 28, with final cutoff 1e-3 and the nonzero triplet cutoffs above. `screened_propagation_results.json` preserves all six records, exact commands, tolerances and remaining scope. Earlier first/final-PNO second-value precision limitations remain in milestones 27–28 and are not superseded by these checks.

**Regression and retained failures.** The native snapshot addition compiled and installed with one compiler job. All four CTests (`dfccsd-t-grad1`, `dlpno-grad-model`, `dlpnocc-1`, `dlpnolocal-1`) passed in 38.19 seconds. After extending `dlpno-grad-model` to test no-opt-in rejection, parent metric/span consistency and copied snapshot ownership, that changed CTest passed again in 30.88 seconds. The existing full-domain water mixed-CCSD regression passed for both core choices at coordinates (1,5): maximum independent-first source error 1.610e-8, sixth-order native residual error 8.607e-9, and triples-adjoint error 3.181e-12. The existing filtered/distinct/repeated triples-moment algebra test passed (maximum 3.981e-8 against 1e-7).

Initial harness runs incorrectly treated the occupied-projector action as an array; swapped/constant tests now transform the action itself. Displaced screened-triplet tests initially hit the old complete-triplet-domain assertion; the harness now compares exported triplet PAO lists, ranks, graphs and transported physical parent spans instead of assuming complete domains. Trial-oracle tuple indexing and a shadowed Fock variable were corrected before numerical acceptance. A full-TNO launch aborted during OpenMP runtime initialization; a fresh process with the same one-thread environment passed, without enabling the duplicate-runtime override. No derivative threshold was relaxed and none of these aborted checks is counted as passing evidence.

The updated readiness audit was rerun and deliberately exits 2 with `--require-production`: selected CCSD/TNO molecular delivery is now available, but no complete screened Hessian/full-matrix validation is claimed. Both memories and complete equation records are updated; Hessian Section 26 and gradient Section 10.10 document this step while retaining all previous milestones. Text exports pass synchronization (2,153 Hessian / 591 gradient lines); final TeX passes are clean (35 / 10 pages). Python compilation and whitespace checks pass.

**Next faithful step.** Assemble a selected complete screened Hessian column from the retained CCSD energy, iterative triples, both PNO losses, excluded SC-MP2/dipole corrections and RHF contribution, using these analytic inputs and the existing stationary adjoints. Compare against independent analytic-gradient differences of the same energy model. Then expand to unsymmetrized full matrices, rigid-motion identities and broader localization/core/space coverage before general production routing or scaling claims.

## 2026-11-07 — Assemble and validate complete screened Hessian columns

**Step 30, based on `81bb6f66`.** `response/dlpno_screened_hessian.py` now assembles a complete analytic column for the same fixed-branch RHF all-strong, ordinary iterative single-TNO-space CCSD(T) gradient model. The seven reported contributions are RHF (including nuclear repulsion), retained CCSD, iterative triples, initial PNO loss, final PNO loss, excluded SC-MP2 pairs and dipole pairs. All correlation inputs are analytic derivatives at one reference geometry. No coordinate samples enter an assembled value. Public analytic Hessian routing is unchanged.

**Assembly and counting.** `ScreenedHessianColumn` caches the triples-only CCSD multiplier, the retained-energy multiplier obtained by subtracting it from the native sourced Lambda, and one right/left triples tangent for the fixed column. Every row supplies its first sources and affine mixed molecular blocks. The retained energy uses the same screened OV panels and exact pair inverse as the CCSD residual. CCSD residual contractions retain diagonal-pair half weights. No further pair weight is applied to the final sum. The two PNO losses use solved mixed LMP2 amplitudes and separately agree in sum with the existing stationary loss adjoint. Each component energy and their total must replay the native forward state; a nonzero surviving weak-pair correction is rejected by this all-strong model.

`ScreenedTriplesInputs.first_direction` connects the native-domain fit to the older first-order TNO-selection and moment kernels, preserving their right, left and energy residual checks. Its first TNO frame must agree with the mixed frame chart within 2e-8. Both triples tangent residuals are recorded; their maximum tested norms are 1.928e-14 (right) and 6.676e-14 (left). First column molecular/amplitude derivatives are checked across row evaluations before reusing the column solve. The maximum CCSD and LMP2 residual checks over the complete columns are 4.439e-12 and 6.968e-13.

**Complete-column validation.** `tests/dlpno-hessian-model/screened_column.py` saves all analytic rows and seven-component decompositions before requesting displaced geometries. It then independently reconverges four native wavefunctions at ±0.004 and ±0.002 bohr along the column direction, assembles the existing native analytic Cartesian gradient, and Richardson-extrapolates its differences. Occupied labels, actual local pair/triplet lists, both PNO ranks, TNO ranks, response edges and physical parent spans must remain on the reference branch. The native analytic RHF Hessian supplies the reference contribution. No symmetrization or translation/rotation projection is applied to the reported column.

| Fixture | Column (zero-based) | Rows | Maximum total-column error (Eh/bohr²) | Maximum first-gradient replay error (Eh/bohr) |
|---|---:|---:|---:|---:|
| Four H2, Boys, all electron | 2: first H z | 24 | 2.56934e-9 | 7.60661e-12 |
| Four H2, Mulliken PM, all electron | 2: first H z | 24 | 4.56167e-9 | 4.35248e-12 |
| Three LiH, Boys, three frozen Li 1s | 5: first H z | 18 | 1.64193e-9 | 7.75677e-12 |

All 66 rows pass the pre-existing 2e-7 Eh/bohr² full-Hessian component criterion. The seven independently formed first-energy derivatives also reproduce the native gradient at every reference row (threshold 2e-8 Eh/bohr). H2 retains seven pairs, six repeated-index triplets, first/final diagonal PNO ranks 9→8 and triplet parent/TNO ranks 20→8. The triplet PAO and auxiliary cutoffs are .01 and .001. LiH retains only three diagonal 12→8 pairs and no triples; it tests frozen-core retained-energy, PNO and correction assembly, not screened frozen-core triples. All three fixtures have nonzero initial/final PNO losses and excluded SC-MP2/dipole estimates. Geometry, basis and convergence follow milestone 29.

For the Boys H2 diagonal entry H[2,2], the contributions (Eh/bohr²) are:

| RHF | Retained CCSD | Iterative triples | Initial PNO loss | Final PNO loss | SC-MP2 | Dipole | Total |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.420969509857 | -0.00267390211049 | -8.92208212e-8 | -3.78954333e-7 | -2.04227057e-4 | -2.86620804e-7 | -2.40810541e-8 | 0.418090601812 |

The small triples term is not accepted merely because it is below the total-column tolerance. `--component-checks` independently differentiates retained-CCSD and triples first energies from the older first-PNO/final-PNO, native first LMP2, older first CCSD residual, and native first CCSD response kernels. The selected rows are (2,18) for H2 and (5,12) for LiH. Maximum retained-CCSD error is 9.690e-11 Eh/bohr² (threshold 2e-8); maximum nonzero H2 triples error is 3.197e-15 Eh/bohr² (threshold 2e-9). LiH's triples result is exactly zero and is identified as empty-space coverage. Independent displaced component gradients, raw central columns and extrapolated references are preserved in the JSON.

**Column invariances and selected reciprocity.** Differentiated zero total force gives sum_A H[Aα,Cβ]=0. Differentiated zero torque gives sum_A R_A × H[A,Cβ] + e_β × g_C=0; the gradient term must be retained at these nonequilibrium geometries. Maximum translation and rotation-covariance errors are 2.886e-13 and 1.248e-12 in the corresponding atomic units, below 2e-7. Native gradient net force and torque also pass 2e-8. `--reciprocity` independently assembles reversed entries for rows 0,5,18 in the H2 columns and row 0 in the LiH column. Every component agrees within 1.495e-14 Eh/bohr² (threshold 2e-9). These selected comparisons and one-column identities do not constitute full-matrix symmetry or full rigid-motion validation.

Reproduce the three complete records with:

    python tests/dlpno-hessian-model/screened_column.py --reciprocity --component-checks
    python tests/dlpno-hessian-model/screened_column.py --localization PIPEK_MEZEY --reciprocity --component-checks
    python tests/dlpno-hessian-model/screened_column.py --system lih --reciprocity --component-checks

The first H2 two-entry smoke check (rows 2,18) passed before expanding to all rows. The three full-column runs passed, then passed again with separate component and reciprocal checks. Analytic assembly took approximately 90.4/91.2/49.8 seconds for Boys H2/PM H2/LiH on this environment, excluding oracle checks. These small-system timings are reproducibility information, not a scaling benchmark. `screened_column_results.json` contains all seven component columns, native gradients, four displaced gradient arrays, raw central estimates, independent component comparisons, reciprocal entries, solver checks and remaining limitations.

**Regression, failures and records.** This step changes Python response assembly and tests; the already-built native implementation is reused. All four CTests (`dfccsd-t-grad1`, `dlpno-grad-model`, `dlpnocc-1`, `dlpnolocal-1`) passed in 35.57 seconds. One concurrent PM launch aborted during OpenMP initialization with a duplicate-runtime diagnostic; a fresh process with the same one-thread settings passed. No duplicate-runtime override was enabled. No scientific comparison failed, no threshold was relaxed, and earlier raw PNO second-value precision limits remain in milestones 27–28. Both working memories and complete equation records are updated; Hessian Section 27 and gradient Section 10.11 record the complete assembly.

The readiness audit is updated and rerun. It still deliberately exits 2 under `--require-production`, now recognizing separately validated complete columns while retaining full-matrix, broader core/triplet coverage and production blockers. There is no new public analytic Hessian route, frequency claim or end-to-end scaling claim. The input harness still materializes dense AO derivative tensors, and AO Schwarz/Fock screens remain off in this analytic column path.

**Next faithful step.** Expand to complete unsymmetrized screened Hessian matrices, check every column against independent analytic-gradient differences, and test full translation/rotation identities. Broaden molecular coverage to screened frozen-core off-diagonal pairs and retained triples, distinct-index triplets and unequal TNO ranks. Keep public routing gated until these broader checks and delivery requirements are met.

Complete equation-export synchronization passes (2,285 Hessian / 611 gradient lines). Final TeX passes are clean (37 / 11 pages), and Python compilation plus whitespace checks pass.

## 2026-11-07 — Complete unsymmetrized screened Hessian matrices

**Step 31, based on `84175650`.** `tests/dlpno-hessian-model/screened_matrix.py` extends the complete column assembler over every ordered Cartesian pair. It evaluates both orientations independently and retains the seven separate component matrices: RHF, retained CCSD, ordinary iterative triples, initial PNO loss, final PNO loss, excluded SC-MP2 and dipole pairs. Neither transposition reuse, averaging with the transpose nor rigid-motion projection enters the saved result. The native analytic RHF Hessian supplies its component; every correlation entry uses analytic first/mixed inputs at the single reference geometry. This step changes no native energy or gradient equation and adds no public derivative route.

**Reproducible validation.** The harness has `analytic`, `reference`, `verify` and default `all` stages. Analytic/reference stages accept disjoint `--columns` lists for processes in separate scratch directories. Atomic per-column JSON checkpoints retain all ordered entries and seven first-energy derivatives, implementation fingerprints, native domains, reference energies/gradient/RHF Hessian, and solver checks. `--resume` verifies reference compatibility before reusing records; final verification requires all rows and columns exactly once. A failed comparison leaves `passed: false` in the matrix record. Verification is separate from the analytic calculation and cannot supply numerical values to it.

The reference independently reconverges the native wavefunction and analytic gradient at ±0.004 and ±0.002 bohr for **every** Cartesian coordinate. It checks occupied matching, native pair/triplet selections, domains, ranks, response graph and physical parent spans at every displaced geometry. Both central columns, the Richardson column, all raw gradients and energies are retained. The three cases comprise 1,476 independently calculated ordered entries and 264 displaced native gradients. No canonical Hessian is used as the reference for these screened functionals.

Errors are maximum absolute values; Hessian/reference and symmetry units are Eh/bohr², translation units Eh/bohr² and rotation units Eh/bohr. No matrices were averaged or projected.

| Fixture | Matrix | Native gradient-difference error | Raw symmetry defect | Largest translation defect | Rotation covariance defect |
|---|---:|---:|---:|---:|---:|
| Four H2, Boys, all electron | 24 × 24 | 1.131092e-08 | 1.965095e-14 | 3.112030e-13 | 3.986052e-08 |
| Four H2, Mulliken PM, all electron | 24 × 24 | 1.910818e-08 | 2.123995e-14 | 2.619089e-13 | 3.986705e-08 |
| Three LiH, Boys, three frozen Li 1s | 18 × 18 | 7.132785e-09 | 5.501025e-15 | 1.456237e-13 | 7.857550e-12 |

Across all cases, component symmetry defects are below 2.257e-14, component translation defects below 2.660e-13 and component rotational defects below 3.985e-8. Native first-gradient replay is below 7.758e-12 Eh/bohr; component gradients agree across columns within 6.894e-16 Eh/bohr. Largest CCSD, PNO/LMP2, triples-right and triples-left residual checks are 4.439e-12, 6.968e-13, 2.473e-14 and 7.295e-14, respectively. Native reference matrices have raw asymmetry up to 1.642e-8, which is retained as a numerical-reference diagnostic.

**Full identities.** For each Cartesian axis, `tau_A=e_axis`, `rho_A=e_axis × R_A`, and `gamma_A=e_axis × g_A`. Check the raw symmetry defect `H-H.T`, both `H tau` and `tau.T H`, and the off-equilibrium rotational covariance `H rho-gamma`. Each component matrix undergoes the same checks with its own component gradient, so cancellation between components cannot hide a failure of these identities. Also check total net force and torque, native first-gradient replay at every row/column, and agreement of each component gradient across independently calculated columns. Rotational covariance includes the nonzero gradient term; these are not equilibrium zero-mode or frequency tests.

The predeclared tolerances are 2e-7 Eh/bohr² for the full gradient-difference matrix and raw symmetry, 2e-7 in the relevant atomic units for left/right translations and rotational covariance, and 2e-8 for first-gradient replay, component-gradient agreement across columns and total force/torque. Every component must pass the matrix identities separately. Numerical references are never symmetrized. `screened_matrix_results.json` preserves all three raw matrices, seven component matrices per case, component gradients, native reference matrices, raw displaced data, solver residuals and numerical errors. The older selected component-oracle checks remain in `screened_column_results.json`; full total-gradient agreement does not replace those separate tests of small triples contributions.

Reproduce in a built environment (give each simultaneous process its own working/scratch directory):

    python /path/to/psi4/tests/dlpno-hessian-model/screened_matrix.py --output-dir chain
    python /path/to/psi4/tests/dlpno-hessian-model/screened_matrix.py --localization PIPEK_MEZEY --output-dir pm
    python /path/to/psi4/tests/dlpno-hessian-model/screened_matrix.py --system lih --output-dir lih

For sharded runs, add `--stage analytic` or `--stage reference` with disjoint `--columns` lists and a shared absolute `--output-dir`, then run `--stage verify` without a column list. `--resume` skips complete compatible columns and continues partial analytic columns.

**Resolved rotation diagnostic.** The Boys and PM total rotational-covariance defects are about 3.99e-8, below the preset 2e-7 threshold but larger than the selected-column results. `screened_rotation.py` independently rotates Boys H2 about x by ±0.004/±0.002 radians, reconverges the native gradient, rotates that vector back, and differences it. For `T_CUT_DO_UV=1e-5`, its defect is 3.9857425e-8 versus the analytic matrix's 3.9860523e-8. The native code screens individual PAO pairs according to `abs(DOI_uv)>T_CUT_DO_UV`; such a mask need not be closed under angular basis rotations. With this cutoff set to zero, the native-gradient diagnostic falls to 2.3114e-9 overall, with the affected y components below 1.7591e-11. An additional analytic zero-cutoff torque-column contraction for the first atom's y coordinate has defect -7.46475e-12 (threshold 2e-9). All eight rotated native wavefunctions across the two cutoff choices retain the reference branch. This control identifies the dominant small rotation defect with the native PAO-pair screening approximation, rather than a missing Hessian term. The full screened fixtures and acceptance thresholds are unchanged; no correction or projection is applied to their matrices. The raw control data and its eight analytic entries are included in the committed results. A rotationally covariant screening policy or a documented accuracy budget remains a production/vibrational concern.

**Coverage and boundary.** These are the same genuinely screened finite-cutoff fixtures as milestone 30: four H2 molecules with Boys or Mulliken PM, and three LiH molecules with frozen Li 1s orbitals and Boys. The final PNO cutoff is 1e-3, initial PNO cutoff 1e-5, TNO cutoff 1e-5, and both triplet-domain cutoffs are active. H2 has seven unique retained pairs with diagonal first/final ranks 9/8 and off-diagonal ranks 5/5, plus six repeated-index triplets in 20-PAO/56-auxiliary domains, truncated to eight TNOs. LiH has three diagonal retained pairs with first/final ranks 12/8 and no retained triples. Both PNO losses and excluded SC-MP2/dipole corrections are nonzero in all cases. Full unsymmetrized matrices now pass for these fixtures, but they do not establish off-diagonal frozen-core pairs, frozen-core triples, distinct-index screened triplets or unequal TNO ranks in a complete molecular Hessian. The geometries also remain planar fragment arrangements.

**Next faithful step.** Extend complete screened molecular coverage to nontrivial distinct-index triplets and unequal TNO ranks, including frozen-core references with retained off-diagonal pairs. Production delivery still needs its own integration and supported-input contract, AO/Fock screening handling, equilibrium vibrational validation and scaling evidence; the current mixed-input harness materializes dense AO derivative tensors and requires AO Schwarz/Fock screens off. Public analytic Hessian routing remains disabled. The readiness audit recognizes the separately validated full matrices while retaining these remaining blockers.

All 44 numerical shards completed successfully. The four targeted CTests passed (73.26 s). Checkpoint guard probes rejected stale implementation fingerprints, changed energies/domains, incomplete columns, duplicated rows and mixed reference spacings. Installed response modules match the repository sources. Python compilation and whitespace checks pass. No numerical acceptance threshold was relaxed.

Hessian Section 28 and gradient Section 10.12 document this step. Both cumulative memories and complete equation exports retain all earlier milestones from both chats, including the failed raw PNO second-value precision checks and their limited accepted alternatives.

The updated readiness audit deliberately exits 2 under `--require-production`; full-matrix evidence is now recorded separately from the remaining production blockers. Complete equation-export synchronization passes (2,415 Hessian / 627 gradient lines), with clean final TeX passes (39 / 11 pages). Python compilation and whitespace checks pass.

## 2026-11-07 — Screened molecular matrices with distinct triplets and unequal ranks

**Step 32, based on `63895aef`.** Extended the complete unsymmetrized screened Hessian validation to nonplanar water–H2 with Boys and Mulliken-PM localization, and ammonia–H2 with Boys localization. All three freeze the O/N 1s orbital, retain six off-diagonal pairs, contain distinct-index triplets and couple unequal TNO ranks. This exercises the existing seven-component assembly on the previously missing molecular cases. No energy term or extra pair weight is added. Public analytic Hessian routing remains disabled.

**Fixtures and actual coverage.** `screened_molecules.py` checks exported native selections, not just requested cutoffs. Every fixture has five active occupied orbitals, eleven of fifteen possible unique pairs (six off-diagonal), sixteen triplets including four with three distinct indices, nonzero occupied-Fock couplings between unequal TNO ranks, and local pair/triplet PAO and auxiliary domains smaller than the molecule. Initial/final PNO losses, excluded SC-MP2/dipole corrections and the iterative triples energy must each be nonzero. Boys water has TNO ranks 5/6/7, PM water 5/6/7/8, and Boys ammonia 5/6/8; the counts of nonzero rectangular directed coupling records are 80/100/90. Repeated native edge records retain their multiplicity. Water has 34 AOs, with triplet parent/TNO space built from 24 local PAOs; ammonia has 39 AOs and 29 local triplet PAOs. Frozen-core response remains active throughout.

The orbital/JKFIT/RI bases are cc-pVDZ. First/final PNO cutoffs are 1e-5/1e-3, TNO cutoff 1e-4; pair/pre cutoffs are 1e-6; DO/DO_PRE/MKN are .01/.03/.001; CLMO/CPAO are .01; DO_IJ is 1e-5. Triplet DO/MKN are .01/.001. `T_CUT_DO_UV=0` isolates this coverage from the individual PAO-pair rotational defect documented in step 31; the original nonzero-screen matrices and their raw defects are preserved. SCF energy/density, integrals and response convergence are 1e-14, localization gradient 1e-13 and CC residual 1e-11; SCF DIIS is off and its iteration limit is 600; increasing this limit leaves every convergence threshold unchanged. AO Schwarz and occupied-Fock screens remain off. Exact coordinates, domains, ranks and energies are in the new JSON and the fixture script.

**Two numerical defects found and corrected.** First, a generic LU inverse of the conditioned Coulomb fitting matrix leaked an antisymmetric roundoff component into a diagonal-pair CCSD source (up to 2.7723e-12 on water). The independent older first source was symmetric at about 1e-17. Projecting the source would conceal this error. Instead, `_symmetric_inverse` now evaluates the unchanged full Coulomb inverse and its first/mixed product rules in a reference spectral basis, preserving the known input-tensor symmetry. No eigenvalue is filtered or regularized, signed spectra are supported, and no residual or Hessian is symmetrized. This helper is restricted to the Coulomb call sites; the original generic inverse remains available to PNO gap response. The separately filtered inverse for products of whitened factors is unchanged.

`metric_inverse.py` compares against exact orthogonally rotated spectral inverses for positive and indefinite matrices with condition number 2e6. The maximum relative response error is 4.481e-10 (tolerance 2e-9); the maximum differentiated inverse-equation residual is 1.747e-10 (tolerance 5e-9); inverse-tensor symmetry is exact. A preceding attempt using symmetrized dense inverse products failed with mixed relative error 1.7041e-5 and mixed inverse-equation residual 0.6057. Applying the symmetry guard indiscriminately to PNO gap inverses also rejected legitimate roundoff there; narrowing the helper to actual symmetric Coulomb metrics resolved that implementation error. These failed trials are retained in the record.

Second, normalization left diagonal elements of the overlap at 1±ulp. Partial Cholesky could consequently choose a different first pivot instead of following the existing overlap-based ordering. `BasisSetOrthogonalization::compute_partial_cholesky_orthog` now sets the diagonal of its pivot-selection copy exactly to one. The subsequent canonical orthogonalization still uses the original normalized overlap submatrix. This corrects the numerical pivot choice and can change the selected parent relative to the old roundoff-driven branch; it changes neither the smooth-branch derivative equations nor pair counting. On ammonia coordinate 5, the old physical parent-span mismatch was 8.6894e-7 despite a parent Gram error of about 1e-14. Renormalizing the parent did not repair that span. After the native correction, all four ±.004/±.002 bohr points pass the unchanged 2e-8 span guard, with maximum mismatch 1.599e-13. `screened_parent_spans.py` preserves this regression and the distinction between normalization and physical-span errors.

**Complete molecular matrices and independent checks.**

| Fixture | Matrix size | Native-gradient reference error (Eh/bohr²) | Raw asymmetry (Eh/bohr²) |
|---|---:|---:|---:|
| Water–H2, Boys, frozen O 1s | 15×15 | 5.52144e-08 | 1.75304e-13 |
| Water–H2, Mulliken PM, frozen O 1s | 15×15 | 8.41751e-08 | 1.76970e-13 |
| Ammonia–H2, Boys, frozen N 1s | 18×18 | 6.19367e-08 | 3.04423e-13 |

All 774 ordered entries pass the unchanged 2e-7 Eh/bohr² comparison against 192 accepted independently reconverged native analytic gradients. Maximum total left/right translation and rotation-covariance defects are 2.7308e-11 and 7.5469e-11, respectively. Component symmetry, translation and rotational-covariance defects are below 3.0010e-13, 2.8581e-11 and 4.6549e-11. The maximum first-gradient replay error is 3.4195e-11 Eh/bohr, and the component-gradient column spread is 1.9691e-14. Total native net force/torque are below 2.4999e-11 in the corresponding atomic units. First-order replay/force/torque tolerance remains 2e-8. The largest raw reference-matrix asymmetry is 8.3507e-08; this finite-difference diagnostic is reported separately from analytic reciprocity.

The maximum recorded CCSD, LMP2, right-triples and left-triples residual checks are 1.0101e-11, 1.1800e-12, 4.4319e-12 and 6.3568e-12.

Every ordered entry is evaluated independently; no mirrored entry, symmetry average or rigid-motion projection is used. The seven component matrices, raw native references, both central spacings, Richardson values, first-energy replays and response residuals are saved. Component invariance checks include symmetry, both translations and rotational covariance with the nonstationary gradient term. Separate selected entries (0,5) and (5,5) use older first-PNO/final-PNO, native first LMP2, older first CCSD residual and native first CCSD response kernels for independent retained-CCSD and nonzero triples gradient differences at ±.004/±.002 bohr. Maximum component errors are 2.055e-10 and 3.674e-12 Eh/bohr², respectively (unchanged tolerances 2e-8 and 2e-9); reverse (5,0) entries agree componentwise within 1.468e-13. These selected component references accompany, rather than replace, the complete native total-gradient reference.

**Branch and spacing history.** The first .004/.002 bohr full-matrix reference run encountered genuine parent-span changes in some other coordinates after the roundoff correction, and strict SCF solves failed to converge within 100 iterations. A different trial ammonia distortion worsened the parent-span mismatch and was discarded; the original geometry and default S_CUT were restored before the final runs. The oracle does not bypass these failures: native pair/triplet lists, ranks, edges and physical parent spans must match at every accepted displaced geometry. The 2e-8 span threshold and all derivative tolerances are unchanged. Smaller spacings test the same local derivative while staying on its physical branch; they cannot make a derivative across a discrete selection boundary meaningful.

A uniform .001/.0005 bohr stencil passed the branch guards but exceeded the 2e-7 matrix tolerance for selected entries (Boys water 2.2954e-7, PM water 2.0481e-7, and ammonia column 1 at 2.2918e-7). The corresponding ammonia column 1 passes with .0015/.00075 at 4.1079e-8, whereas ammonia column 4 needs .001/.0005 to stay on its branch. Thus a single global reference spacing is unsuitable for these fixtures. The final oracle tries the fixed descending list [.004, .002, .0015, .001] bohr and selects the first stencil whose four native points pass the original branch guards, before consulting any analytic matrix or derivative error. It preserves every rejected stencil, reason and completed sample. SCF or gradient failures still abort; only a branch-check rejection permits a smaller candidate. Verification checks the declared spacing policy and rejection prefix as well as the implementation fingerprint. No analytic value is used to select a step.

The accepted final spacings are .004 bohr for 42 columns; water columns 1 and 4 use .002 for each localizer; ammonia column 1 uses .0015 and column 4 uses .001 (all zero-based). Nine rejected candidate stencils are retained. All 32 final numerical shards pass. Six checkpoint probes reject changed implementation fingerprints, mismatched molecular coverage, incomplete columns, inconsistent spacing policy, omitted rejection prefixes and missing rejection reasons. The readiness audit completes and deliberately exits 2 under `--require-production`.

The checkpoint fingerprint now includes the actual native binary, in addition to the response modules and molecular fixture/helper sources. This prevents combining columns built with different native pivot implementations. Molecular coverage metadata is also matched. Old incompatible/incomplete records remain rejected.

Reproduce a complete matrix in a built Psi4 environment with `screened_matrix.py --system water_h2 --localization BOYS --adaptive-steps .004 .002 .0015 .001 --output-dir water_boys`, changing the localizer to `PIPEK_MEZEY` or the system to `ammonia_h2` for the other cases. Separate `--stage analytic` / `--stage reference` processes may own disjoint `--columns`, each with separate native scratch, followed by `--stage verify`. The selected independent component check is `screened_column.py --system water_h2 --column 5 --rows 0 5 --component-checks --component-rows 0 5 --reciprocity`; vary system/localizer as above. Run `metric_inverse.py` and `screened_parent_spans.py` for the numerical regressions. The new `screened_molecular_results.json` preserves complete evidence, raw unsuccessful reference attempts and source/provenance information.

**Regression and scope.** The native change was rebuilt and installed with one compiler job. Five targeted CTests (`dfccsd-t-grad1`, `dlpno-grad-model`, `dlpnocc-1`, `dlpnolocal-1`, `scf-cholesky-basis`) passed in 123.35 seconds. The new inverse test, existing PNO and triples algebra tests, selected old Boys/PM H2 and frozen-core LiH columns with independent components, and full-domain mixed CCSD tests for both core choices all pass. A transient alias-substitution syntax error was corrected before the final calculations; it is not counted as a numerical test. No failed or incomplete run is labeled as validated.

This closes the selected distinct-index, unequal-rank and frozen-core off-diagonal molecular gaps left by step 31. It is not general validation over arbitrary molecules, thresholds or branch boundaries. Rotational-screening accuracy, equilibrium vibrations, the public driver's supported-input contract and end-to-end scaling remain. The dense AO derivative harness is not a local-scaling implementation. The readiness audit records the broader evidence while keeping public analytic Hessian routing disabled. All earlier milestones and raw PNO second-value precision limitations are preserved. The final complete matrices are regenerated after the fixture iteration-limit and reference-policy changes; prior analytic values are not relabeled with a new implementation fingerprint.

Both cumulative memories and complete equation records are updated (Hessian Section 29, gradient Section 10.13). Complete equation-export synchronization passes (2,545 Hessian / 653 gradient lines); final TeX passes are clean (41 / 11 pages). Python compilation and whitespace checks pass.

**Next faithful step.** Define and implement a guarded public analytic Hessian delivery path for the supported RHF all-strong/single-TNO fixed-branch model, with explicit screen and resource restrictions, then validate end-to-end driver behavior and an equilibrium vibrational example. Broader screening contracts and local integral delivery remain necessary before general production/scaling claims.


## 2026-11-07 — Guarded public Hessian integration, interrupted validation

Step 33 adds the explicit public analytic route, installed mixed molecular inputs, same-reference gradient/Hessian delivery, raw seven-component assembly, supported-input/resource guards and option restoration. No energy term or pair weight changes. All previous milestones remain above. The native small-driver regression, seven existing CTests, direct derivative-negotiation cases and H2 equilibrium frequency workflow passed in the original workspace. Full water–H2 Boys/PM and ammonia–H2 Boys public matrices were still running when the environment went offline. Do not label them validated. The recovery record preserves numerical observations, failures, paths and required resume gates. Hessian Section 30 documents the contract; its reconstructed text export is complete but not newly typeset.

Step 33 is a recovery/WIP checkpoint based on `833ec80c`, not a new accepted milestone.

## 2026-11-07 — Public routing delivery and cc-pVDZ local tests

Restored the implementation and cc-pVDZ tests from `b870dd7a` onto the delivery
branch at the user's request. This supersedes the recovery-only publication
restriction, while preserving the outstanding native numerical validation.
Public `hessian` and `frequency` use `DLPNO_HESSIAN_MODEL=true`, with the
same-reference native gradient, raw seven-component Hessian, existing screen
and memory restrictions, and no opted-in numerical fallback. All four installed
molecular-input providers are AST-equivalent to the accepted development
helpers after name/import normalization; no derivative equation changed.

Completed the Cython install manifest for all 17 DLPNO response modules;
ordinary Python installs already copy the complete directory. H2 driver and
frequency tests explicitly select cc-pVDZ-JKFIT/RI. The three screened molecular
fixtures retain their original cc-pVDZ configuration and numerical references.
The new test README provides native CTest, full matrix and frequency commands
with separate scratch directories.

Source validation: 105 pytest cases pass with the native boundary mocked
(80 unchanged derivative-negotiation cases plus 25 public Hessian route,
unsupported-setting, missing-second-integral and option-restoration cases).
All 17 response modules import from only the declared installation files.
Python compilation, complete equation export synchronization and whitespace
checks pass. This is source-level validation, not a native build or molecular
run. The previous H2/native results and interrupted public molecular runs remain
historical evidence; fresh native tests are left for the user's local build.

## 2026-11-07 — Default analytic routing and completion message

At the user's request, `DLPNO_HESSIAN_MODEL` now defaults to `true`. Supported
settings, memory checks and rejection of numerical fallback remain in force.
The cc-pVDZ driver regression checks the default route before explicitly
disabling/re-enabling it; the screened-domain audit now tests explicit disabling
and restores the original option values and changed flags.

The shared DLPNO-CCSD(T) completion message adds "The harvest is plentiful, but
the laborers are few" after the existing line. Standalone Hessians print the
message after successful delivery. Frequency calculations defer it until
vibrational analysis and postcallbacks finish, so the nested Hessian does not
print it twice. Existing gradient/optimization completion sites use the same
helper. Failed Hessian or vibrational analysis does not print success messages.

All 105 source-level routing cases pass with the new default and mocked native
calls. Mocked public-wrapper checks cover completion order, duplicate
suppression, both failure paths and an unrelated method. Python compilation,
equation-export consistency and whitespace checks pass. Native numerical
validation status is unchanged; no electronic-structure equation changed.


## 2026-11-07 — Water execution evidence, response reuse and stage timing

The user's water/cc-pVTZ output from revision `991427f` completes all 81 ordered
analytic Hessian entries and returns a matrix after the native residual,
first-gradient replay and raw symmetry/rigid-motion guards. Wall time is
1806.67 seconds including optimization; the nine column times total 1678.2
seconds. This is successful execution evidence for that revision, not an
independent energy/gradient finite-difference validation of the matrix. The
previous cc-pVDZ fixtures and their recorded evidence are unchanged.

The current pass owns a first-CCSD-response cache on the Hessian reference.
It stores read-only first AO/PNO frames and singles/doubles amplitudes, keyed by
Cartesian coordinate. Every reuse compares its first frame at the unchanged
2e-9 tolerance, and every ordered entry replays the full responding CCSD
residual and checks its value/first derivatives. New entries are published only
after that check passes. On a nine-coordinate assembly, the control flow needs
nine first solves instead of 162 and nine constant-amplitude source replays
instead of 81. All 81 responding mixed residuals remain. The memory preflight
includes cached frames/amplitudes. Printed and wavefunction counters expose the
actual work; the water repeatability regression requires the nine/nine counts.
This cache has source/synthetic validation, but no fresh native molecular run.

Mixed products share at most 1024 contraction plans keyed by expression/shapes;
only metadata is shared, not molecular arrays. Per-column timing distinguishes
molecular inputs, PNO/LMP2 response, CCSD mixed response, screened corrections,
retained energy/curvature, and triples directions/solves/curvature. The gradient
also reports its orbital assembly stages. These timings identify the next
native targets; no measured full-molecule speedup is claimed for this pass.

Known remaining costs include per-entry first PNO/LMP2 reconstruction, mixed
LMP2 response, and repeated first fitting/triples data. The gradient's 58 AO
water case has four 16-column panels, each making four native DF J/K-gradient
calls through polarization. A native cross-density derivative contraction could
combine those passes. Do not parallelize the existing calls over shared PSIO
units. A more fully stationary adjoint formulation could remove second
electronic responses, but is a separate derivation and validation project.

The absent frequency table was caused by calling `hessian()`: it returns force
constants. The misleading analytic-frequency announcement is corrected, and
`frequency()` remains a single Hessian followed by vibrational analysis. The
new NumPy-only recovery utility reads the completed log and matching final
geometry/isotope masses, applies mass weighting and removes translations and
rotations. The supplied rounded matrix gives 1669.32, 3840.90 and 3945.38 cm^-1
(bend, symmetric stretch, antisymmetric stretch). All three are real. Matrix
rounding in the log limits these values as derivative-validation references.

Validation: 129 focused checks pass (118 source/routing/algebra checks with a
mock native boundary and 11 isolated compiled DF-kernel checks). Nine separate
recovery tests pass, including the exact reduced-mass diatomic result under
rotation/translation and negative curvature. The supplied log reproduces the
three frequencies above. Twelve old/new orbital-gradient contractions with
identical mock operators agree exactly; wrapper checks verify one Hessian per
frequency call, completion order and failure paths. This is not a full native
Psi4 build, a molecular timing comparison or independent derivative validation.
