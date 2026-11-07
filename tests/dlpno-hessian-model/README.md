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

These tests are provided for a local numerical rerun. The recovery checkpoint's
earlier native results are historical evidence; the recovered implementation
has not yet passed a fresh native run. Its source-level routing checks do not
establish numerical Hessian accuracy.

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
First PNO/CCSD responses are still reconstructed for individual mixed entries;
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
and requires exactly one first-density response batch. The second command
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
python -m pytest tests/pytests/test_dlpno_hessian_response.py tests/pytests/test_dlpno_response_kernels.py
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
