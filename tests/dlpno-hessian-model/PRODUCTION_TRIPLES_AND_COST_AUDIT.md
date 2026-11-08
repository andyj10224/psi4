# Production triples derivatives and response cost audit

> **8 November priority update:** further Hessian development is paused. Review
> the gradient theory, original-code reuse and molecular validation first, under
> the contract recorded in `DLPNO_GRADIENT_WORKING_MEMORY.md`.

## Measured result, before this patch

The successful `water_freq_test(8).out` is from `39ac9fb`; compare it with
`water_freq_test(6).out` from `50b6a5d`. These are observed timings, not estimates
of what the new patch will achieve. Both use the same submitted water/cc-pVTZ
input. Substage times below overlap their parent stages and must not be added
again to the total.

| Work | Run 6, seconds | Run 8, seconds |
| --- | ---: | ---: |
| Complete optimization plus frequency job | 636.51 | 385.29 |
| Nine Hessian columns | 511.9 | 259.0 |
| CCSD mixed response, summed over columns | 348.83 | 100.42 |
| Triples mixed response | 67.36 | 59.28 |
| Triples tangent solves | 34.99 | 42.67 |
| First triples directions | 22.91 | 20.65 |
| PNO/LMP2 response | 20.31 | 20.44 |
| Molecular inputs | 14.12 | 12.68 |

The Hessian is 49.4% faster, but triples tangent solves got slower. The printed
frequencies are unchanged: 1669.3180, 3840.8971, 3945.3762 cm^-1. Agreement between
these two implementations is a regression check, not an independent validation
of the gradient or Hessian.

Within the latest CCSD stage, doubles took 47.37 s and preparation 33.82 s;
singles took 4.94 s and occupied Fock work 1.15 s. The remainder includes the
first-response solves and stage overhead. These are the actual priorities.

## Operations that dominate the work

Let d=3N, q be a local fitting rank, p a pair's PNO rank, t a triplet's TNO
rank, a its PAO rank, m an occupied-domain size, and E the set of retained
intertriplet coupling edges. Ranks vary: sums over domains are more meaningful
than one global molecular power law.

| Operation | Leading arithmetic per local block | Why it matters; exact improvement |
| --- | --- | --- |
| CCSD virtual ladder, sum_Q D_Q T D'_Q^T | O(q p^3), O(q p^2) storage | Keep the binary GEMMs already used; forming a four-virtual ERI instead costs O(p^4) storage. Batch pair outputs only after their shared inputs have immutable ownership. |
| Local metric application to VV factors | O(q^2 p^2), plus O(q^3) factorization | Reference and first metric actions must survive across entries. Keep the actual full/filtered metric of each equation. Applying a full inverse to one small OV factor is cheaper for an OV-VV vertex; it is not a universal replacement for the VV-VV ladder. |
| Mixed J/K and pair transfers | Sums of O(q p_r p_s) contractions and rectangular cubic transfers | Preserve each vertex's original local fitting domain. Replacing its fit by a neighboring pair fit is an approximation. Reuse weighted OV factors and overlap orientations. |
| Iterative triples transfer r <- s | O(t_r t_s^3 + t_r^2 t_s^2 + t_r^3 t_s) per edge | This is fourth order for comparable ranks and repeats in each solver matvec. Use the original three-GEMM transfer and gather by output triplet in OpenMP. |
| Triples W/V construction | O(q t^3) OVVV formation, then O(t^4) contractions, plus occupied-domain terms | Reuse the original moments for gradient stages. Mixed derivatives still require differentiated moments and moving TNOs. |
| PAO -> PNO/TNO factors | O(q(a^2 p + a p^2)), analogous with t | Prepare atom-local qij/qia/qab derivative panels once per input jet; retain the captured atom and pair screens. |
| Local spectral response | O(a^3), O(q^3) where required | A first derivative can be shared; a genuinely different local metric or mixed direction still needs its own calculation. No rank/domain approximation was introduced. |

A binary mixed product contains four products in its uv component:

    (AB)_uv = A_uv B + A_u B_v + A_v B_u + A B_uv.

Computing all four jet components from scratch takes nine products rather than
one. The present driver visits all d^2 ordered coordinate entries. Thus a fast
energy evaluator alone does not give a fast Hessian: repeated reference/first
work, moving-domain derivatives, and repeated high-rank transfers multiply its
cost. The current implementation has not attained the requested energy-time
multiples. No molecular speedup for this patch is claimed before a rerun.

Run 8 recorded 4,360,131 LRU evictions and 6,408,511 numerical evaluations.
Protected storage was full (256/256 MiB), as was the optional stage allowance
(128 MiB). Raw/refit/overlap build counts were 243/243/8487; reuse counts were
4779/486/513. In later columns, overlap builds increased while overlap reuse
stopped increasing. This is evidence of admission starvation, not evidence that
Python alone accounts for the remaining time. Pair/triplet equations already
execute in C++; Python still supplies molecular inputs and Krylov orchestration.

## Changes in this patch

* Reuse the original `DLPNOCCSD_T::matmul_3d` implementation through
  `triples_contractions.h`. The old energy method delegates to the identical
  three-GEMM order, with index-division shuffles replaced by explicit strides.
  The Hessian operator uses that same body, removing its more
  general sequence of tensor permutations for uniform three-leg transfers.
* Apply right and transpose triples operators by independent output rows with
  OpenMP. Transpose contributions are gathered in deterministic order; workers
  never scatter into another worker's output.
* Form the fixed spin-adapted energy source P V once for each Z3 solve. It was
  being formed again during every adjoint iteration. Memory accounting includes
  the retained source and, for production stages, the saved input T0 owner.
* Reserve overlap space inside the existing memory limits. Large VV panels
  cannot consume it. Store one orientation of S_pq, with S_qp=S_pq^T. Reference
  and first overlap components have a reserved protected allowance; mixed
  overlaps have a bounded stage allowance. Peak overlap storage is printed.
* Remove `native_ccsd.cc`, `native_triples.cc`, and `ccsd_gradient_lambda.cc`
  from the build. Their response extensions live in `ccsd.cc`, `triples.cc`,
  and `ccsd_lambda.cc`, respectively. This consolidation by itself does not
  accelerate arithmetic or prove two equation sets equivalent.

The remaining support files implement tensor jets, local fitting and subspace
response; they cannot be removed by substituting value-only energy calls.
CCSD mixed derivatives and the exact residual transpose remain separate
extensions beside the original routines. The published Lambda solver is not
silently substituted for the exact transpose of the implemented right residual:
that would require a term-by-term proof with finite, different pair domains and
PNO ranks. No counterexample to the published Lambda equations is established
by this audit, and their existing entry points are unchanged.

## Mathematical differences that prevent a blind substitution

1. The former gradient model is a single TNO-space functional. Its iterative
   triples amplitudes and adjoints were already present. It does not include
   the production tight-minus-loose T0 correction or screened-triplet energy.
2. The single-space response graph searches replacement LMOs in a complementary
   pair domain. The production energy searches all occupied LMOs and then tests
   whether the replacement triplet exists. These are not equivalent for general
   sparse domains. For example, target (1,2,3) can couple through F_03 to stored
   (0,1,2), although 0 is absent from domain(1,2): the source can have been
   generated from domain(0,1). The new production route uses the full occupied
   search. Existing single-space results retain their original graph.
3. `T_CUT_ITER` permits individual production triplets to stop updating before
   every residual converges. Gradient requests now disable this stopping
   automatically: the public driver temporarily sets `T_CUT_ITER=0` and restores
   it afterward, and the native solver guards the skip condition. The derivative
   does not differentiate a partially converged stopping history.
4. In the legacy in-core iteration, one OpenMP worker updates T_ijk in place
   while another can read it as a coupled source. That code is not a suitable
   shared Jacobi response operator. The derivative solver keeps separate old
   and next generations. This patch leaves the legacy energy iteration intact
   when the new route is disabled; its threading deserves a separate fix and
   molecular regression study.

## New opt-in production-triples gradient

On the existing all-strong CCSD gradient reference, the new route differentiates
exactly the four contributions selected by the production triples workflow:

    E_corr = E_T0(screened, prescreen space)
           + E_T0(retained, tight space)
           - E_T0(retained, loose space)
           + E_T (retained, loose space).

Each stage uses its own original fitting domain, TNO selection and moments.
Screened triplets are included because their energy is included, without adding
another threshold on their derivative. Discrete membership/rank choices are
held fixed, as required for a smooth-branch derivative.

For each stage, R=W+B t=0, E=<P V,t>, and B^T z=-P V. Its stationary derivative is

    dE = <P^T t,dV> + <z,dW> + <z,(dB)t>.

For T0 stages B has the local semicanonical denominators and no intertriplet
edges. For iterative T it has the production occupied-coupling graph. The
existing response machinery differentiates TNO selection, pair amplitudes,
local fitting, metric, PAO/LMO coefficients, Fock terms and overlaps. Signed
stage covectors are accumulated in common CCSD/PAO/LMO spaces before the single
(T)-sourced CCSD Lambda solve. There is no projection between tight and loose
TNO spaces, and no extra approximation.

The original `triples_sparsity`, `tno_transform`, `sort_triplets`, and
`compute_lccsd_t0` routines select and build the stages. Small guarded hooks in
`compute_triples_correction` invoke the same adjoint machinery. Ordinary energy,
asymmetric Lambda, Brueckner and higher-order paths retain their existing calls
when the new option is off. Stage energies must reconstruct the reported
correction; each stage prints its timing and exports its discrete branch.

Enable for an analytic gradient:

```python
set {
    dlpno_gradient_full_triples true
}
gradient('dlpno-ccsd(t)', dertype=1)
```

This is **not yet a general production DLPNO-CCSD(T) gradient**: the underlying
CCSD gradient still promotes surviving pairs to strong. The new feature removes
the single-TNO triples restriction, not the separate CCSD pair restriction. It
currently requires in-core triples. It is disabled by default, and requires the
molecular validation below before scientific use.

The existing analytic Hessian cannot consume these four stage states and now
rejects that combination explicitly. To propagate the new analytic gradient to
a Hessian at present, use `frequency('dlpno-ccsd(t)', dertype=1)`, which requests
finite differences of gradients. That is a **semi-analytic Hessian**, and will
be slower; no full multi-space analytic Hessian is claimed. Such an analytic
extension needs stage-specific first responses, TNO charts and signed stationary
curvatures, with their separate branches verified. Reusing only the final loose
triples state would omit the other three contributions.

## Validation and next performance work

Completed checks: 138 native numerical tests passed in a serial build and
in the OpenMP build with 1 and 4 threads; 262 routing/algebra tests passed.
Actual-header syntax checks passed for the modified CCSD, Lambda, triples,
adjoint and binding translation units. The new full molecular stage audit has
not been run here.

A focused synthetic operator benchmark used eight triplets of ranks 48/52/56,
32 directed edges, single-thread BLAS and 12 repeated matvecs per timing. At four
OpenMP threads, the right action changed from 0.093–0.116 s to 0.030–0.034 s,
and the transpose from 0.081–0.123 s to 0.028–0.029 s. With one thread the right
action changed from 0.092–0.093 s to 0.072–0.074 s. These are kernel timings on
this host, not water Hessian speedups or a prediction for the M3 Max.

Available here: actual-header C++ syntax checking; exact-source native kernels
with real BLAS/LAPACK; independent Python contraction, derivative and transpose
oracles; serial/OpenMP builds; public driver routing tests. A full molecular
Psi4 build is not available here, so molecular equivalence, finite differences,
new-mode runtime and production energy threading are **not yet verified**.

`production_triples_gradient.py` defaults to cc-pVDZ and compares the new gradient
energy against the legacy production energy with all pairs strong and early
triplet stopping disabled. It checks the signed stage sum, dual energy, a nuclear
directional derivative at two step sizes, force/torque, exported triplet branches,
and optionally thread agreement. Run in separate scratch directories:

```bash
python production_triples_gradient.py --threads 1 --output production-1.json
python production_triples_gradient.py --threads 4 --output production-4.json --compare production-1.json
```

Also test `--screen 0` and `--screen 1` (all retained / all screened), then cc-pVTZ.
The original single-space cc-pVDZ gradient/Hessian tests remain in place.

The next substantial CCSD speedup needs persistent reference/first *stage*
objects with explicit lifetimes, not millions of small LRU admissions. The
33.82-second preparation and 47.37-second doubles stages should be profiled at
the binary-GEMM boundaries before deciding which panels deserve that memory.
Immutable prepared pair inputs would then permit parallel output-pair residuals
without shared-map mutation or duplicating every neighbor's intermediates.
For triples, retain gather metadata and reusable packing buffers across Krylov
matvecs if profiling shows their O(sum t^3) copies matter after the O(sum_E t^4)
transfer improvement. A larger memory budget alone cannot remove those transfers.
