# Tóth Lambda equations and the gradient adjoint: reconciliation

9 November 2026. Audited against branch base
`13d74e01c7d7f02b1deae831bca37a341080be20`.

Primary source: E. C. Toth, A. Jiang, and H. F. Schaefer III,
*Linear-Scaling Asymmetric Triples Correction through the Solution of the
DLPNO–CCSD Lambda Equations: DLPNO–CCSD(T)Λ*,
[DOI: 10.1021/acs.jctc.6c00758](https://doi.org/10.1021/acs.jctc.6c00758),
Sections IV.D and V.A. Equation numbers below refer to this paper unless
explicitly labeled Jiang. The supplied paper was read directly, including
the projector changes described in Section V.A.

## Result and scope

The gradient Lambda iterations now call the existing `form_goo`,
`compute_lambda_intermediates`, `compute_L_ia`, and `compute_L_iajb` in
[`ccsd_lambda.cc`](../../psi4/src/psi4/dlpno/ccsd_lambda.cc).
They no longer evaluate the separately reversed residual graph every iteration.
The existing right-hand functions in
[`ccsd.cc`](../../psi4/src/psi4/dlpno/ccsd.cc) still construct the dressed
integrals, beta, gamma, delta, and occupied Fock matrix.

An internal `exact_residual=true` argument retains the projectors and incoming
domains needed to differentiate that actual right-hand residual. It introduces
no additional local approximation. Existing asymmetric-energy callers keep
the default published projector/domain choices. A bare `2 f_ia` source was
added to the shared singles equation; it is zero for an ideal stationary RHF
reference. Sixty-four Fov=0 residual fixtures give identical published-mode
results before and after this change.

The earlier observation of a finite-space mismatch was real, but it was not
evidence of a universal sign or factor-of-two error in the paper. Section V.A
explicitly removes intermediate projections. Those simplifications need not
equal the exact transpose of the finite-space right residual. Sparse retained
pair graphs reveal additional domain-bound differences. The distinctions and
their correction are made explicit below.

This audit covers the **CCSD amplitude stationarity equations**, including
how the existing ordinary-(T) source enters them. It does not derive or newly
validate the complete iterative-(T) nuclear derivative, orbital/localization
response, PNO/TNO response, screened-energy derivatives, or a nuclear Hessian.
The existing full-gradient restrictions and regression tests remain in place.

## Conventions that must be fixed before comparing terms

Let `p=ij` denote an existing ordered pair and `O_p` its stored occupied domain.
The pair basis and fitting domain are those already assigned to that pair.
Use `S_pq = X_p^T S_PAO X_q`, with parent-domain restriction understood, and
`U_p = 2 T_p - T_p^T`. Stored doubles obey the existing ij/ji transpose relation.

The independently tested storage identity is

\[
 \mathcal L_{CC}=E_{CC}
   +\sum_i \lambda_i^T R_i
   +\frac12\sum_{p\;\mathrm{ordered}}\Lambda_p:R_p,
 \qquad
 \delta\mathcal L_{CC}=\sum_i l_i^T\delta t_i
   +\frac12\sum_{p\;\mathrm{ordered}}l_p:\delta T_p.
\]

This states the normalization used by the code, rather than comparing a bare
prefactor in Eq. 34 to a prefactor in isolation. The paper's nonsymmetric
`bar l_ij` and final partner addition (Eq. 92) must both be included. In code,
`Ln_iajb` holds the nonsymmetric terms; `L_iajb` also holds pair-symmetric
contributions evaluated once for each unordered pair. The final addition of
`Ln_ij + Ln_ji^T` supplies the ordered-pair residual. The source from the
doubles energy is consequently `2 L_ij` in this residual convention.

`lambda_iajb_bar_ = 0.5 Lambda + Lambda^T` is Eq. 55 in the existing storage
convention. The Lambda transformations used for the asymmetric triples energy
(Eqs. 98–99) are not a reason to rescale the gradient multipliers. No change to
pair weights, permutations, or the asymmetric triples-energy definition was made.

## Term-by-term residual map

`L1` and `L2` below mean `compute_L_ia` and `compute_L_iajb`; all setup and
left-residual functions are in `ccsd_lambda.cc`. Right residuals and their
intermediates are in `ccsd.cc`. The `reverse_*` column identifies the independent
reference implementation retained for validation and integral derivatives.

### Singles stationarity

| Paper term | Origin in the implemented energy/right residual | Shared implementation and reconciliation | Independent reference |
|---|---|---|---|
| Eq. 35 energy source, `2 S L S t1` | Singles-quadratic part of `E_CCSD` | Existing first pair loop in L1; evaluated once per solve | Energy seed in `gradient_lambda_residual` |
| Bare `2 f_ia` source | `singles_fock_energy() = 2 sum f_ia t_i^a` | Added to L1 when `include_source=true`; implicit zero under the paper's ideal RHF assumption | Same energy seed, also checked with nonzero Fov |
| Eq. 58, `delta_tilde lambda1` | Derivatives of dressed `F_ai` and the A/B/C singles contractions | `delta_imae_tilde_`, Eqs. 56–57, reused; the last bilinear term of Eq. 56 uses the **mm** domain in exact mode | `reverse_R1` then `reverse_t1_transform` / `reverse_pair_intermediates` |
| Eq. 58, occupied-Fock term | Direct occupied-Fock dependence on singles | Existing `F_im_double_tilde_` and overlap contraction | Same |
| Eq. 58, virtual-Fock term | Direct virtual-Fock dependence on singles | Existing `F_vv_double_tilde_` contraction | Same |
| Eq. 67a, double-dressed K of Eq. 59 | Differentiation of dressed exchange, beta dressing, and occupied-Fock singles dependence in R2 | `K_maef_dt_` unchanged | `reverse_R2`, `reverse_pair_intermediates`, `reverse_t1_transform` |
| Eq. 67b, double-dressed K of Eq. 60 | Dressed exchange, virtual ladder, and explicit singles dependence of virtual Fock in R2 | `K_eimn_dt_` unchanged | Same |
| Eq. 68a | Derivative of dressed virtual Fock, contracted with rhoVV | Existing factorized Eq. 113; reuse persistent dressed Qab when available; do not materialize Eq. 65 | Same |
| Eq. 68b | Derivative of dressed occupied Fock, contracted with rhoOO | Existing `F_knia_hat_`; Eq. 66b rows extended to **nn**, not restricted to kn | Same |
| Eq. 69 | Singles derivative of gamma (Eq. 77), occupied-index J channel | Retain `S(ii,km) S(km,mn)`; evaluate through shared G below | `reverse_pair_intermediates` gamma channel |
| Eq. 70 | Singles derivative of gamma, three-virtual J channel | Existing J tensor contracted with the same G; incoming sum belongs to the gamma pair | Same |
| Eq. 71 | Singles derivative of delta (Eq. 78), three-virtual M channel | Existing M tensor contracted with shared D; retain `1/2` and orientation | `reverse_pair_intermediates` delta channel |
| Eq. 72 | Singles derivative of delta, occupied-index M channel | Retain `S(ii,mk) S(mk,mn)`; use D and existing M tensor with `-1/2` | Same |

Eq. 56 is not an extra energy term. Its four pieces arise from the bare
mixed occupied/virtual integral, the projected N–t1 term, the M–t1 term,
and the bilinear singles term. Eq. 57 adds the L–U contribution. The
bilinear Eq. 56 contraction differentiates an mm-fitted quantity and must
retain every k in `O_mm`, including k for which ik is absent. Restricting
k to `O_im` silently loses those derivatives on a sparse graph.

### Doubles stationarity

| Paper term | Right-residual origin | Shared implementation and reconciliation | Independent reference |
|---|---|---|---|
| Eq. 36 energy source | Doubles correlation energy | Existing L source in `Ln`, followed by partner addition | Energy seed |
| Eq. 75, lambda1–Lbar | R1 A, derivative of U including dressed scalar-integral part | `L_ieab_bar_` (Eq. 73), unchanged | `reverse_R1` |
| Eq. 75, lambda1–Kbar | R1 B | `K_ijmb_bar_` (Eq. 74), unchanged | `reverse_R1` |
| Eq. 75, `(2-P_ab) lambda1 F` | R1 C | Existing Fkc/overlap terms and permutation, unchanged | `reverse_R1` |
| Eq. 86a → 108, virtual ladder | R2 A, direct derivative of T2 | Existing dressed-Qab GEMM/triplet contraction; persistent Qab reuse | `reverse_R2` |
| Eq. 86b → 108, alpha | R2 B through derivative of beta (Eq. 76) | Existing `compute_alpha_ijkl()` (Eq. 81) and Qia contraction | `reverse_pair_intermediates` beta channel |
| Eq. 87 → 109, beta | R2 B through the explicit T2 | Existing projected Lambda/beta term, unchanged | `reverse_R2` |
| Eq. 88a | R2 C through the explicit T2 | Existing total gamma/J contraction (Eq. 82), unchanged | `reverse_R2` |
| Eq. 88b | R2 C through gamma's T2 dependence (Eq. 77) | Restore the intermediate projectors removed in Eqs. 103/106; use G | `reverse_pair_intermediates` gamma channel |
| Eq. 89a | R2 D through explicit U | Existing total delta/M contraction (Eq. 83) and `(2-P_ab)`, unchanged | `reverse_R2` |
| Eq. 89b | R2 D through delta's U dependence (Eq. 78) | Restore the projectors removed in Eqs. 105/107; use D with the existing spin permutation | `reverse_pair_intermediates` delta channel |
| Eq. 90 → 110, direct virtual Fock | R2 E through explicit T2 | Existing left/right Fvv terms, unchanged | `reverse_R2` |
| Eq. 90 → 110, rhoVV | R2 E through Fvv's U dependence (Eq. 79) | Existing projected rhoVV/L contractions, unchanged | `reverse_pair_intermediates` virtual-Fock channel |
| Eq. 91, direct occupied Fock | R2 G through explicit T2 | Preserve the usual intersection-domain sum and add retained in with **jn absent**; F''jn then has its bare value | `reverse_R2` |
| Eq. 91, rhoOO | R2 G through Focc's U dependence (Eq. 80) | Existing rhoOO/L contraction, unchanged | `reverse_pair_intermediates` occupied-Fock channel |
| Eqs. 92, 111–112 | Partner permutations | Existing final `Ln_ij + Ln_ji^T`; no change of pair normalization | Final pair symmetrization in the reference |

These rows cover the energy, lambda1, and lambda2 sectors separately. The
tests set either multiplier rank to zero to avoid relying on cancellation
between the two ranks. They also differentiate the scalar Lagrangian directly,
so agreement between two left-residual implementations is not the only test.

## Precisely which projections change in gradient mode

In a common parent representation,

\[
 S_{pq}S_{qr}=X_p^T S X_q X_q^T S X_r
             =X_p^T S P_q X_r,
 \qquad P_q=X_qX_q^T S.
\]

Replacing this product by `S_pr` removes `P_q`. The two expressions are equal
only when that intermediate projector is the identity on the transported
vectors (or the particular discarded component does not contribute). Complete
PNO ranks alone do not imply equality if the PAO parents differ.

The original L1 code used `S(ii,mn)` in Eq. 69 and Eq. 72 where the displayed
equations retain an intermediate km or mk projector. In L2, Section V.A
explicitly changes Eq. 102 to 103 and Eq. 104 to 105, removing inner projector
chains; Eqs. 106 and 107 then remove outer chains from Eqs. 88 and 89.
Gradient mode evaluates the expressions with these projectors retained.
It does **not** replace a single projector in the right energy algorithm with
two, change any PNO cutoff, or alter the right residual being differentiated.

The original implementation also uses target-pair common-neighbor sums in
places where a transposed contraction needs incoming neighbors of an
intermediate pair. In a complete graph these bounds coincide. The exact
incoming sums below, together with the Eq. 56, Eq. 66, and Eq. 91 fixes in the
tables, restore the transpose on the tested sparse graphs. These are source
loop-bound corrections for gradient mode, not a claim that the paper explicitly
requires the former sparse bounds.

## Reuse audit and the two additional pair matrices

Existing `rho_oo_` (Eq. 53) traces away the virtual information needed here;
`rho_vv_` (Eq. 54) is a single-pair contraction. Neither can represent the
mixed-pair contractions with an intervening projector. Existing
`gamma_double_tilde_` / `delta_double_tilde_` store the deprojected Eqs. 103/105,
so substituting them would restore the discrepancy.

Extend **existing** `form_goo(true)` with two SharedMatrix arrays on the
**existing ordered-pair map**:

\[
 G_{ij}=\sum_{n\in O_{ij}}
 S_{ij,jn}\bar\Lambda_{jn}S_{jn,in}T_{in}S_{in,ij},
 \qquad \bar\Lambda=\tfrac12\Lambda+\Lambda^T,
\]
\[
 D_{ij}=\sum_{n\in O_{ij}}
 S_{ij,in}\Lambda_{in}S_{in,nj}U_{nj}S_{nj,ij}.
\]

`lambda_gamma_[ij]` and `lambda_delta_[ij]` store G and D. They are allocated
once and zeroed/rebuilt once per Lambda residual evaluation; both L1 and L2
reuse them. No new production C++ function, pair map, or tensor abstraction
was introduced. Two-factor Matrix products are composed with existing
`doublet`/`triplet` kernels.

For an output pair ij, the restored higher terms in the nonsymmetric L2 are

\[
 \Delta\bar l_{ij}^{(88b)}
 =\tfrac12\sum_{k\in O_{ij}}
 S_{ij,kj}G_{kj}S_{kj,ik}K_{ik}S_{ik,ij},
\]
\[
 X_{ijk}=S_{ij,ik}D_{ik}S_{ik,kj}L_{kj}S_{kj,ij},
 \qquad
 \Delta\bar l_{ij}^{(89b)}
 =\sum_{k\in O_{ij}}\left(\tfrac12 X_{ijk}-\tfrac14 X_{ijk}^T\right).
\]

For L1, flatten G for contraction with `J_kaec_bar_` (Eq. 70), and D-transpose
with `M_kace_bar_` of the reverse pair (Eq. 71). Eqs. 69/72 use the columns of
`G J_kmic_bar_^T - 0.5 D M_mkic_bar_^T`, projected into each diagonal PNO
space in the source pair's occupied domain. The shared matrices therefore
avoid a second evaluation of the same Lambda/T pair sum.

Every fitted tensor remains in its original pair's auxiliary/PAO domain.
Eqs. 37–44 and the original nonprojected J/K blocks are reused. No local metric
is moved between different fitting domains. The fixture tests explicitly use
different auxiliary subsets, each with its own inverse-half metric.

## Lifetime, parallelism, and memory

| Quantity | Lifetime / reuse |
|---|---|
| Dressed integrals; beta, gamma, delta; F'' | Fixed at converged T; built once by existing setup |
| Eqs. 56–64, 66, 73–74 | Fixed at converged T; built once in `compute_lambda_intermediates(true)` |
| Eq. 65 | Not stored; the existing Eq. 113 factorization remains active |
| Eqs. 103/105 | Not built in exact mode; retained in published mode |
| rhoOO, rhoVV, Eq. 55, G, D, alpha | Depend on Lambda; updated for each residual evaluation |
| CCSD energy derivative plus existing ordinary-(T) amplitude source | Built once; added to each source-free linear Lambda residual |
| Dressed Qab | Existing bounded persistent storage, shared by L1 and L2; recomputed if not admitted |
| Fvv in the fixed-reference record | SharedMatrix alias to the already-built Fvv; no duplicate pair sum/storage |
| Fallback PNO overlaps | Existing energy tables first, then persistent overlaps / existing semidirect panels within the budget |

The two G/D arrays cost exactly `16 * sum_p n_pno[p]^2` bytes (double
precision, ordered pairs), excluding allocator metadata. Expanded Eq. 66
storage has `sum_kn sum_(i in O_nn) n_pno[ii]` doubles. The printed memory
estimate includes the Tóth tensors, G/D, and thread-private residual/scratch
storage. It is an estimate, not an allocation-profiler measurement.

Pair-owned G/D and L2 work use existing OpenMP loops; L1 scatters use the
existing per-thread buffers. The fallback overlap map, its admission counters,
and panel access are protected by a mutex because the shared residuals now
call that mutable facility concurrently. Tables already owned by the energy
code remain shared. Cache-miss fallback work currently holds the lock; its
cost needs molecular timing before making a speedup claim.

Existing setup/L1/L2 timers remain. Output additionally reports setup, fixed
source, iteration, and pair-density wall times. Scalar variables record
`DLPNO LAMBDA FIXED SOURCE BUILDS` and
`DLPNO LAMBDA SHARED RESIDUAL EVALUATIONS` (production evaluations, excluding
the independent debug probes). There is no new claim of measured asymptotic
scaling or molecular speedup.

## What remains of the reverse implementation, and why

`gradient_lambda_residual` remains an independent reference and the existing
transpose-Jacobian action used by response machinery. `form_ccsd_pair_response`
also needs the reverse intermediates' **integral and domain covectors**, not
just the amplitude residual. The original unrelaxed OPDM does not contain
all of those covectors or the local-space/metric derivative information.
Deleting these routines would therefore remove existing derivative capability.
They are no longer the production gradient Lambda iteration engine.

The precomputed ordinary-(T) `triples_source1_` / `triples_source2_` are added
without changing their mathematics. The CCSD operator is affine in Lambda:

\[
 l(\Lambda)=b_{CC}+b_{(T)}+J_{CC}^T\Lambda.
\]

Changing which code evaluates the **same** CCSD transpose does not by itself
upgrade a model triples gradient to the complete production triples gradient.
Triples equations and their derivative verification are separate from this
reconciliation.

## Validation and reproducibility

The standalone probe extracts the **actual C++ function bodies verbatim**;
small checked Matrix/API adapters supply storage and arithmetic. It exercises
the actual `S_PNO` table/fallback/panel logic. The adapters are test-only and
are neither a molecular Psi4 build nor a BLAS performance benchmark.

| Check | Observed result |
|---|---:|
| 256 fixtures: shared versus independent reverse residual | maximum absolute difference 5.56e-16 |
| Separate rank-1/rank-2 central differences of the scalar Lagrangian | maximum discrepancy 1.87e-12 |
| 27 independent amplitude-coordinate finite differences | maximum discrepancy 1.06e-12 |
| One versus four OpenMP threads | maximum difference below 3e-16 |
| Persistent overlap/Q budgets 0, 256, 1048576 bytes; three replays | maximum difference below 3e-16 |
| Fixed-source plus source-free operator versus full residual | maximum difference 2.23e-16 |
| Original published-mode residual versus modified published mode, Fov=0 | 64 fixtures; maximum difference 0 |
| Actual-header C++ syntax checks | `ccsd.cc` and `ccsd_lambda.cc` pass |

Fixtures include full/truncated PNO ranks, distinct PAO parents and auxiliary
fits, complete and sparse retained-pair graphs, off-diagonal occupied Fock
elements, nonzero Fov, and source-only/Lambda1-only/Lambda2-only/combined
sectors. Synthetic residual units must not be interpreted as molecular
gradient errors. Detailed recorded maxima are in
[`lambda_reconciliation_results.json`](lambda_reconciliation_results.json).

From the repository root, with NumPy, pybind11, Python development headers,
and a C++17/OpenMP compiler available:

```sh
python tests/dlpno-grad-model/build_lambda_reconciliation_probe.py --build-dir /tmp/lambda-current
python tests/dlpno-grad-model/lambda_reconciliation.py /tmp/lambda-current --output /tmp/lambda-results.json
python tests/dlpno-grad-model/build_lambda_reconciliation_probe.py --ref 13d74e01c7d7f02b1deae831bca37a341080be20 --build-dir /tmp/lambda-original
python tests/dlpno-grad-model/lambda_reconciliation_legacy.py /tmp/lambda-original /tmp/lambda-current
```

Supply `--pybind-include PATH` to the builder if pybind11 is not importable
but its headers are available. Both modules must be built for the same Python.

The existing cc-pVDZ molecular input now also compares the shared L1/L2
residual to the independent reverse at independent trial multipliers and
checks that the fixed source was built once. **That molecular test and a full
Psi4 build were not run for this patch.** Local validation must still check
converged Lambda residuals, finite-difference nuclear gradients on a stable
domain branch, existing asymmetric energies, and timing/memory before claiming
production readiness or improved wall time.

## Follow-up: energy-path 2Fia and the cost of retaining projectors

The `2 f_ia` addition in `cffd222` was already outside the exact-residual
conditional: it applies to **both** Lambda pathways. The energy solver now
calls `compute_L_ia(..., false, true)` explicitly, preserving its published
projector choices and including the full source. No second copy of `2 f_ia`
was added. The scalar energy already includes `2 f_ia t_i^a` through
`singles_fock_energy()`. A bare `2 f_ia` is a singles derivative, not a
separate scalar energy term. The existing Brueckner mode that deliberately
omits Lambda1 still omits that equation.

The historical comparison test now also uses 64 **nonzero-Fov** fixtures.
Against base `13d74e0`, the energy Lambda residual changes by precisely the
projected `2 f_ia` in L1 and zero in L2, to 4.17e-17. Its original 64 Fov=0
fixtures remain unchanged. This exercises full/truncated PNO spaces,
different PAO/fitting domains, and sparse pair graphs.

### What has actually been quantified about gradients?

There is not yet a controlled, paired **molecular nuclear-gradient** result
for projector retention versus removal. The historical water directional
checks did not isolate projector choices from every other correction and
must not be reported as that force difference.

A controlled fixed-space comparison uses the same amplitudes, multipliers,
energy source, a complete pair graph, and a common PAO parent. Only the
projector choices then differ; sparse-domain corrections cannot contribute.
For seed 17, the measured maximum component changes are:

| Spaces / fitting | Lambda1 residual change | Lambda2 residual change |
|---|---:|---:|
| Complete PNO spaces, common fit | 5.56e-17 | 1.12e-16 |
| Truncated PNO spaces, common fit | 7.36e-5 | 2.79e-4 |
| Complete PNO spaces, pair-specific fits | 8.33e-17 | 5.56e-17 |
| Truncated PNO spaces, pair-specific fits | 7.72e-6 | 1.79e-4 |

These are **synthetic Lambda residuals at trial multipliers**, not gradients
in hartree/bohr, converged Lambda differences, or error estimates for water.
Another seed reaches 2.89e-4 in L1 and 2.44e-4 in L2. The random amplitudes
and fitting tensors have no claim to represent the magnitude in a molecule.

In independent amplitude coordinates, if `r = b + J^T lambda_a` is the
exact stationarity residual at an approximate multiplier, then
`lambda_a - lambda_exact = J^(-T) r`. For the fixed-space CC contribution,
with the other response blocks held fixed,
`Delta g_X = (lambda_a-lambda_exact)^T R_X`.
Consequently neither the residual size alone nor a PNO occupation threshold
sets a universal force error; the response operator and nuclear derivative
couplings also matter. A molecular A/B comparison must hold all other
approximations, domains, ranks, fitting definitions and convergence settings
fixed, and also compare the retained-projector result with finite differences
on a stable domain branch.

### Cost guarantee for evaluating the affected expressions

Let P be the number of stored ordered pairs, d the maximum occupied-domain
size, and r the maximum PNO rank. The bounds below concern a residual
evaluation; they do not assume that different Lambda operators require the
same number of iterations. Pair-overlap preparation and other gradient
terms are separate costs.

| Affected work | Published projector removal | Retained projectors |
|---|---|---|
| Fixed higher L2 intermediates, Eqs. 103/105 | O(P d² r³) setup; O(P d r²) storage | Not built |
| Shared G/D | Not present | O(P d r³) per evaluation; O(P r²) storage |
| L1 Eqs. 69–72 | O(P d r³ + P d² r²) | O(P r³ + P d r²), after G/D |
| Higher L2 Eqs. 88b/89b | O(P d r³) per evaluation | O(P d r³), after G/D |

Thus the combined retained-projector expressions cost
**O(P d r³ + P d r²)**, with the same PNO-rank power and no extra occupied
sum. The G/D cost is counted once for the combined L1/L2 evaluation.
With bounded local domains/ranks and P proportional to molecular size,
both versions are linear in that size. Removing the old setup can improve
the bound; equality of prefactors or wall time is not asserted.

The reuse audit remains the one above: extend `form_goo` and use the existing
G/D arrays and pair maps. Final G/D and higher-gamma products now accumulate
directly with `Matrix::gemm`, avoiding a pair-sized output temporary and
subsequent addition. A `DLPNO-CCSD Lambda : Mixed rho` timer isolates the
shared build. No new production C++ function is needed.

Every overlap in the added G/D and higher-L2 chains joins pairs sharing an
occupied index, and is already in the energy overlap tables. L1 now obtains
`S(oo,p)` as the **transposed application of the stored `S(p,oo)`**, rather
than issuing a reversed lookup that could trigger a PAO-space reconstruction
under `LOW_MEMORY_OVERLAP`. This is an exact transpose identity. The affected
retained-chain blocks therefore introduce no new AO/PAO-sized overlap
contraction, including in low-memory mode.

### Executable scaling checks

The optional probe scopes count the Matrix contractions in the actual C++
blocks. Its test-only `triplet` uses the same C1 flop-minimizing association
as `libmints/matrix.cc`. Scopes do not change equations or loop bounds;
`2*m*k*n` counts a matrix product and does not count all scalar arithmetic
or measure time. The test verifies:

* Replicating disconnected fixed-size occupied domains by factors two/four
  multiplies every affected block's work by exactly two/four.
* Doubling PNO rank multiplies G/D and higher-L2 work by eight; L1 grows
  by at most eight, including its quadratic terms.
* Doubling a complete occupied graph at fixed rank multiplies G/D and
  higher-L2 work by eight (P doubles quadratically, d linearly), while the
  old fixed higher-L2 setup grows by sixteen.
* Enabling `LOW_MEMORY_OVERLAP` adds **no matrix contractions** in the
  affected retained-chain blocks.

For the P=9, d=3, r=4 example, the combined counted matrix work is 60,192
operations with retained chains and 79,488 with the published removal,
excluding setup. The latter also builds 82,944 operations of fixed higher
L2 tensors. This is an operation-count example, not a molecular speedup.

```sh
python tests/dlpno-grad-model/build_lambda_reconciliation_probe.py --count-projectors --build-dir /tmp/lambda-counted
python tests/dlpno-grad-model/lambda_projector_cost.py /tmp/lambda-counted --output /tmp/projector-cost.json
python tests/dlpno-grad-model/lambda_reconciliation.py /tmp/lambda-counted
python tests/dlpno-grad-model/lambda_reconciliation_legacy.py /tmp/lambda-original /tmp/lambda-counted
```

The original probe must be built from `13d74e0`, as in the earlier commands.
The tracked [projector cost results](lambda_projector_cost_results.json)
contain all counted cases and numerical residual comparisons. Full molecular
gradient and wall-time comparisons remain outstanding.
