# Public DLPNO-CCSD(T) Hessian tests

Build and install the current `dlpno_gradients` branch first, including the
native `DLPNO_HESSIAN_MODEL` option. Use that installation's Python environment
and Psi4 executable. Reconfigure CMake so it discovers `dlpno-hessian-driver`.
The response modules are installed for both ordinary Python and Cython builds.
`DLPNO_HESSIAN_MODEL` defaults to `true`; the existing supported-setting and
memory checks still apply. Set it to `false` to disable the analytic route.

All numerical cases below use **cc-pVDZ**, with cc-pVDZ-JKFIT and cc-pVDZ-RI.
The LANL2DZ input inside the driver regression is only an ECP-rejection check.
No smaller orbital basis replaces the derivative tests.

These tests are provided for a local numerical rerun. The recovery checkpoint's
earlier native results are historical evidence; the recovered implementation
has not yet passed a fresh native run. Its source-level routing checks do not
establish numerical Hessian accuracy.

## Driver and existing native regressions

From the source checkout, using your actual configured build directory:

```bash
ctest --test-dir build --output-on-failure -R '^dlpno-hessian-driver$'
ctest --test-dir build --output-on-failure -R '^(dfccsd-t-grad1|dlpno-grad-model|dlpnocc-1|dlpnolocal-1|scf-cholesky-basis|scf-hess1|scf-hess2)$'
python -m pytest tests/pytests/test_dertype.py
```

`dlpno-hessian-driver` uses H2 and 1 GiB. It checks default analytic routing,
explicit disabling and re-enabling, explicit/automatic derivative selection,
matrix and wavefunction returns, a gradient-difference column,
same-reference energy/gradient/Hessian publication, option restoration after
an injected failure, unsupported settings, and memory refusal. It forbids
finite-difference execution inside the analytic driver. H2 has no triples.

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
