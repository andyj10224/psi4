# Public DLPNO-CCSD(T) Hessian tests

Build and install the current `dlpno_gradients` branch first, including the
native `DLPNO_HESSIAN_MODEL` option. Use that installation's Python environment
and Psi4 executable. Reconfigure CMake so it discovers `dlpno-hessian-driver`.
The response modules are installed for both ordinary Python and Cython builds.
`DLPNO_HESSIAN_MODEL` defaults to `true`; the existing supported-setting and
memory checks still apply. Set it to `false` to disable the analytic route.

The existing numerical cases use **cc-pVDZ**, with cc-pVDZ-JKFIT and cc-pVDZ-RI.
The new water repeatability regression additionally uses **cc-pVTZ**, matching
the reported failure; it does not replace any cc-pVDZ tests.
The LANL2DZ input inside the driver regression is only an ECP-rejection check.
No smaller orbital basis replaces the derivative tests.

The user's water/cc-pVTZ run at `6ac570a` completed all 81 ordered Hessian
entries and printed frequencies of 1669.3180, 3840.8971 and 3945.3762 cm^-1.
Its native residual, gradient-replay and rigid-motion guards passed. This is
execution evidence for that revision, not an independent finite-difference
validation. The new native contraction changes below still need a molecular
rerun. Source-level routing checks do not establish numerical Hessian accuracy.

## Response reuse and timing

At `991427f`, the water run took 1806.67 seconds, including optimization;
1678.2 seconds were spent in the nine Hessian columns. At `6ac570a` the job
took 1177.93 seconds and the columns totaled 1051.1 seconds: 1.60 times faster
for the columns. The log confirms nine first CCSD solves and nine fixed-source
replays. CCSD mixed response (445.31 s), triples mixed curvature (284.01 s),
and first triples directions (170.71 s) dominate the remaining column time.
The gradient's Python orbital assembly takes about 0.4--0.47 s; its DF Fock
panels take 0.37--0.42 s, so combining those 16 native J/K calls is a lower
priority than the measured response costs.

The new pass makes these changes:

- Differentiate one shared binary contraction tree rather than repeatedly
  evaluating the full multi-operand contraction for every product-rule term.
  Eligible nodes use the C++ `dlpno_mixed_gemm` kernel and Psi4's BLAS wrapper.
  Strided inputs are packed once per operand; zero derivative blocks are
  skipped only after finite-value checks. BLAS owns contraction threading.
- Contract triples overlap derivatives with the adjoint once per column/edge.
  Subsequent rows use two small overlap covectors, avoiding repeated rank-three
  mixed transfers. Edge multiplicity and every ordered Hessian entry remain.
- Cache the native triples snapshot and first directions on the Hessian
  reference. Nine-coordinate water requires nine builds and 72 hits when the
  cache fits. Read-only copies, first-frame checks, a byte budget of one eighth
  of requested memory and LRU eviction preserve reference ownership and allow
  an uncached fallback. The preflight reserves that budget.
- Reuse reference CCSD amplitudes and native fitting/domain snapshots instead
  of exporting full tensor state for every mixed entry.
- Reuse native triples overlap matrices during right/left gradient iterations.
  This cache is populated serially and read by the existing OpenMP loops,
  bounded by the smaller of 64 MiB and one sixty-fourth of requested memory.
  Native gradient phase timings now complement the Python orbital timings.

All convergence thresholds, first-response consistency checks, raw Hessian
symmetry/rigid-motion checks, and existing cc-pVDZ tests are retained. First
PNO/LMP2 reconstruction and mixed LMP2 solves remain. A fully stationary
adjoint formulation is a separate derivation and validation project.

**Rebuild the native extension and install the changed Python modules.** The
new binding cannot be obtained by updating Python files alone. The previous
Python-only build advice applied to `6ac570a`, not this pass.

For an isolated kernel comparison on the installed build:

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python tests/dlpno-hessian-model/benchmark_response.py \
    --rank 53 --auxiliary 139 --directions 9 --threads 1 --output response_kernels.json
```

An isolated compile of the exact production kernel/binding and BLAS wrappers,
using small Matrix/thread adapters and single-thread OpenBLAS 0.3.30, gave:

| Kernel | Previous (s) | New (s) | Speedup |
| --- | ---: | ---: | ---: |
| Mixed AO projection | 0.06641 | 0.04429 | 1.50x |
| Mixed DF pair contraction | 0.15289 | 0.07103 | 2.15x |
| Nine-row triples overlap curvature, including preparation | 0.21717 | 0.01932 | 11.24x |

Maximum old/new absolute discrepancy was 1.16e-14 on these synthetic inputs.
These are isolated kernels, not a full Psi4 build or end-to-end Hessian timing.
An initial comparison with reference BLAS on the C++ side and optimized BLAS
on the NumPy side was unsuitable for timing; both sides above use optimized
OpenBLAS. The user's energy-relative runtime targets remain unverified.

## Frequencies and completed-run recovery

`hessian()` returns force constants. Use `frequency()` to additionally print
normal modes and thermochemistry, and save the full-precision Hessian before
the vibrational analysis:

```python
set hessian_write true
frequency('dlpno-ccsd(t)')
```

The frequency wrapper performs one Hessian calculation. Do not call both
`hessian()` and `frequency()` consecutively unless two evaluations are intended.
The public Hessian message now describes force constants rather than promising
frequencies. If a completed `.log` contains the Hessian, recover its modes
without another electronic-structure calculation:

```bash
python /path/to/psi4/tests/dlpno-hessian-model/recover_frequencies.py --log water_freq_test.log --output water_freq_test.out --json recovered_modes.json
python -m pytest --confcutdir=tests/dlpno-hessian-model tests/dlpno-hessian-model/test_recover_frequencies.py
```

Use the matching `.out`: its final Angstrom geometry and isotope masses define
the mass weighting and translation/rotation projection. The utility rejects
incomplete/abbreviated matrices and inconsistent nuclear repulsion energies.
It requires NumPy, not a Psi4 installation. The supplied water run gives
1669.32, 3840.90 and 3945.38 cm^-1 (bend, symmetric stretch, antisymmetric
stretch). The log rounds the matrix to eight decimal places; these recovered
frequencies are not full-precision Hessian validation references.

## Gradient-first verification after the optimization

Reconfigure and rebuild the native extension: this change adds a DLPNO C++
source file, a batched RHF response API and a quiet wavefunction-based
`MintsHelper` constructor. A Python-only reinstall is insufficient.

Start with the inexpensive DF-kernel NumPy comparisons, which require an
installed native build but no molecular calculation:

```bash
python -m pytest tests/pytests/test_dlpno_response_kernels.py
```

Then run the off-equilibrium water gradient audit in **separate working and
PSI_SCRATCH directories**. These commands use the installed Psi4 Python and
request 28 GB. `--coordinates 2,5,8` is a shorter initial check; the default
`all` checks every Cartesian component.

```bash
python /path/to/psi4/tests/dlpno-hessian-model/gradient_audit.py --threads 1 --output gradient_1.json
python /path/to/psi4/tests/dlpno-hessian-model/gradient_audit.py --threads 4 --output gradient_4.json --compare /absolute/path/to/gradient_1.json
```

The audit compares the analytic gradient with independently reconverged
**energies** at ±0.004, ±0.002 and ±0.001 bohr. Two Richardson extrapolations
must agree, and the final gradient error must be below 2e-7 Eh/bohr. All
stencils require unchanged native pair/domain/PNO/TNO selections and matching
parent spans. A branch-check failure invalidates the stencil; it does not
establish that the analytic derivative is wrong. JSON is saved progressively,
with `passed: true` only after every requested check succeeds.

It also checks force/torque, frozen-core batched versus directional RHF
response, serial/four-thread mixed DF shell blocks on reordered subsets, and
agreement of gradient/explicit components between runs (2e-8 Eh/bohr).
Repeat with `--localization PIPEK_MEZEY`, `--all-electron`, and then
`--basis cc-pvtz`. Compare JSON only for identical physical settings. The
existing `dlpno-grad-model` canonical/zero-cutoff and screened molecular tests
remain necessary: water alone is not a complete gradient validation.

The optimization reuses reference adjoints, bounded integral caches and
atom-independent AO covectors. The native DF kernel uses GEMM/GEMV and
`Matrix::gemm`, reusing nine QD products across the sixteen mixed terms.
OpenMP packing and mixed-integral loops write disjoint outputs; LMP2 right
rows are independent, while transpose scatter remains serial. The dense
Hessian memory guard, full ordered matrix, residual tolerances and rigid-motion
checks are unchanged. Row progress now includes elapsed column time.
First PNO/LMP2 responses are still reconstructed for individual mixed entries;
this is a bounded optimization, not an end-to-end local-scaling backend.
No full-molecule speedup is claimed before timings on the rebuilt installation.

## Water response repeatability regression

Rebuild the native extension as well as installing the Python files. The fix
prepares SCF options for occupied response and DLPNO options for LMP2 response
solves, so module queries cannot select another module's convergence limits.
The public Hessian owns one cached occupied-density derivative per Cartesian
coordinate. The new `rhf_occupied_tangents` API exports all of them from one
native batch, reducing water's tangent-response batches from nine to one
(the separate RHF Hessian response is still evaluated once). All mixed
entries, PNO responses and existing residual/identity tolerances are retained.

Your original water input can be rerun directly after rebuilding. To isolate
the Hessian at the reported optimized geometry, run the new script from two
separate working/scratch directories with the installed Psi4 Python:

```bash
python /path/to/psi4/tests/dlpno-hessian-model/water_repeatability.py --threads 1 --output water_1.json
python /path/to/psi4/tests/dlpno-hessian-model/water_repeatability.py --threads 12 --output water_12.json --compare /absolute/path/to/water_1.json --fd-column 0
```

The script requests 28 GB, leaves global SCF solver options at their defaults,
and requires exactly one first-density response batch, nine first CCSD solves,
nine fixed-source replays, nine first triples builds and 72 triples cache hits.
The second command
compares the component/total Hessians at 2e-7 Eh/bohr² and optionally checks
one column against independently reconverged gradient differences with native
selection matching. Omit `--fd-column` for a quicker repeatability-only run.
An assertion about a displaced selection rejects that numerical stencil; it
does not establish an analytic Hessian error. `.out` logs share the JSON stem.
Any remaining column-repeatability failure reports its block/pair, coordinates,
scientific-notation error, occupation gaps and upstream first-response errors.

## Driver and existing native regressions

From the source checkout, using your actual configured build directory:

```bash
ctest --test-dir build --output-on-failure -R '^dlpno-hessian-driver$'
ctest --test-dir build --output-on-failure -R '^(dfccsd-t-grad1|dlpno-grad-model|dlpnocc-1|dlpnolocal-1|scf-cholesky-basis|scf-hess1|scf-hess2)$'
python -m pytest tests/pytests/test_dertype.py
python -m pytest tests/pytests/test_dlpno_hessian_response.py tests/pytests/test_dlpno_response_kernels.py tests/pytests/test_dlpno_mixed_products.py tests/pytests/test_dlpno_triples_contractions.py
```

`dlpno-hessian-driver` uses H2 and 1 GiB. It checks default analytic routing,
explicit disabling and re-enabling, explicit/automatic derivative selection,
matrix and wavefunction returns, a gradient-difference column,
same-reference energy/gradient/Hessian publication, option restoration after
an injected failure, unsupported settings, and memory refusal. It forbids
finite-difference execution inside the analytic driver. H2 has no triples.
It also checks one first-density batch per Hessian, batched/directional
coefficient agreement, and native SCF/LMP2
response after unrelated option queries with conflicting global/local limits.

## Three complete screened molecular matrices

Run each case in a separate working and native scratch directory. The following
commands start from the source checkout and run the cases sequentially. Results
are written outside the checkout to keep generated files out of the source tree.

```bash
export DLPNO_SOURCE="$PWD"
export DLPNO_CHECKS="$(mktemp -d "${TMPDIR:-/tmp}/dlpno-public.XXXXXX")"

for case in water_boys water_pm ammonia_boys; do
    case "$case" in
        water_boys) system=water_h2; localizer=BOYS ;;
        water_pm) system=water_h2; localizer=PIPEK_MEZEY ;;
        ammonia_boys) system=ammonia_h2; localizer=BOYS ;;
    esac
    mkdir -p "$DLPNO_CHECKS/$case/scratch"
    (
        cd "$DLPNO_CHECKS/$case" || exit 1
        export PSI_SCRATCH="$PWD/scratch"
        python "$DLPNO_SOURCE/tests/dlpno-hessian-model/public_driver.py" \
            --system "$system" --localization "$localizer" \
            --output public_driver_results.json
    ) || break
done
```

Each case requests 4 GiB and one thread. Water–H2/Boys and water–H2/PM have
15×15 Hessians; ammonia–H2/Boys has an 18×18 Hessian. Their frozen cores,
off-diagonal pairs, distinct-index triplets, unequal coupled TNO ranks, local
domains, and nonzero correction energies are checked from native state.

The test calls the public Hessian route once, checks a single same-geometry
DLPNO forward calculation, and rejects finite-difference execution. All seven
component matrices must match the saved step-32 analytic results within
2e-9 Eh/bohr². Every total-matrix entry must match the previously recorded
native-gradient reference within 2e-7 Eh/bohr². `passed: true` is written only
after all comparisons succeed. `public_driver.out` records column progress.
The saved references in `screened_molecular_results.json` are unchanged.

## Equilibrium frequency smoke test

Using the variables set above:

```bash
mkdir -p "$DLPNO_CHECKS/h2_frequency/scratch"
(
    cd "$DLPNO_CHECKS/h2_frequency" || exit 1
    export PSI_SCRATCH="$PWD/scratch"
    python "$DLPNO_SOURCE/tests/dlpno-hessian-model/public_vibration.py"
)
```

This locates the H2 equilibrium bond with native analytic gradients and calls
`psi4.frequency(..., dertype=2)`. Independent gradient differences validate
the Hessian column and stretch frequency. The numerical oracle is outside the
analytic Hessian path. This is a workflow check; screened molecular vibrational
validation and local runtime/memory scaling remain separate work.

Equation export consistency can be checked without a native Psi4 build:

```bash
python tests/dlpno-hessian-model/export_equations.py --record all --check
```
