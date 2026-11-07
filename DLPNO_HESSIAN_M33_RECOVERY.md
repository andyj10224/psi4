# Milestone 33 recovery checkpoint — 7 November 2026

**Status: reconstructed work in progress; do not treat this commit as a tested tree.**

The execution environment became unavailable while the final public-driver molecular matrices were running. Read-only recovery attempts failed with an exec-server transport disconnect/recovery timeout. GitHub remained accessible. This separate branch reconstructs the implementation and tests from recorded tool outputs and patches on top of accepted parent 833ec80c021747603b6719954dcadcbaf4e6dc90. It does not move dlpno_gradients, which retains the accepted milestone-32 head.

The reconstruction has not been compiled, syntax checked or numerically rerun. Comments/formatting and the condensed recovery documentation can differ from the inaccessible workspace. Revalidate the recovered tree before promotion. The original workspace may still be recoverable; do not reset or overwrite its changes without comparing them.

## Implemented next step

Explicit DLPNO_HESSIAN_MODEL opt-in for public analytic hessian()/frequency(), using the same RHF all-strong ordinary iterative single-TNO functional as the analytic gradient. Installed mixed molecular inputs are promoted from the existing raw DF, RHF, localization and screened-input helpers. Their function bodies were AST-equivalent after import/function renaming in the pre-outage workspace.

The route computes one native RHF/DLPNO reference and gradient, assembles all ordered entries of seven energy-component Hessians, publishes the raw matrix and same-reference gradient, and checks first-gradient replay, reciprocity, both translations and off-equilibrium rotational covariance. It does not symmetrize, project, or sample nuclear geometries to construct the matrix.

The initial contract requires gas-phase nonrelativistic DF-RHF, HF orbitals, Boys/Mulliken PM and a full-rank AO metric. Frozen cores and finite local domains/ranks are allowed on a smooth branch. AO Schwarz, occupied-Fock and individual PAO-pair screens must explicitly be zero (DLPNO_AO_INTS_TOL, F_CUT, F_CUT_T, T_CUT_DO_UV). Ghosts, ECP, external potentials, supplied ref_wfn, frozen virtuals and Brueckner references are refused. A supplied gradient must match the native gradient. Numerical fallback is forbidden when opted in.

Temporary convergence/retention settings and changed flags restore on success/failure. A conservative dense scratch estimate must fit within half the requested memory before correlation; native guards remain independent. This is not local memory scaling or general production readiness.

## Observed before the outage (not validation of this reconstructed commit)

- Native option was compiled and installed.
- Existing CTests passed: dfccsd-t-grad1, dlpno-grad-model, dlpnocc-1, dlpnolocal-1, scf-cholesky-basis, scf-hess1, scf-hess2.
- The final expanded new dlpno-hessian-driver CTest passed in 17.78 s. It checked routing, default matrix return, seven-component sum, a two-spacing native gradient column, same-geometry single forward execution, option restoration after injected failure, supplied-gradient shape/value checks, unsupported references/screens/ghost/external/ECP inputs, low-memory refusal and absence of test-helper imports.
- pytest was unavailable; 80 unchanged parameter cases from test_dertype.py passed direct assertions (45 selections, 17 missing-method errors, 18 excessive-derivative errors). This was not a pytest run.
- H2/cc-pVDZ equilibrium public frequency check passed: bond 1.4378899324646717 bohr; E=-1.1637178618578148 Eh; max gradient 2.1227117286137798e-13 Eh/bohr; one physical mode at 4382.9304171721005 cm^-1. Independent native-gradient differences gave 4382.930413789351 cm^-1, an error of 3.3827491279225796e-6 cm^-1 and raw column error 5.654411383559932e-10 Eh/bohr^2. Five modes were translation/rotation. Input D2h was handled through a C1 working copy. One correlation forward, no finite-difference Hessian execution. H2 has no triples: this is only an equilibrium workflow smoke test.
- The readiness audit still intentionally exited 2 with --require-production.
- Installed/source equality, Python compilation and diff checks passed before the outage. The full local equation exports were synchronized and cleanly typeset (43 Hessian / 12 gradient pages); the shorter recovered additions here require a fresh typeset.

The last observed molecular progress was water-H2 Boys 13/15 columns, water-H2 Mulliken PM 13/15, ammonia-H2 Boys 9/18. These are partial progress counts, not accepted matrix results. No completed public molecular results JSON was observed. Do not claim that all 774 entries passed. The already accepted milestone-32 matrices and independent gradient references remain available in screened_molecular_results.json.

## Preserved failure history

1. Initial H2 smoke: Matrix.ncol() is not a Python binding; use Ca().np.shape[1].
2. Initial new CTest: Psithon integer zero for DLPNO_AO_INTS_TOL is interpreted as a convergence exponent. Use 0.0.
3. Two equilibrium coordinate assertions: ordinary geometry/C1 updates rounded coordinates by 1.11e-16 bohr. Coordinate-only checks now allow 1e-14 bohr; no derivative tolerance was changed.
4. ECP test: this build lacks libecpint and refuses the basis before the Python Hessian guard. The test accepts that exact native refusal or the Python ECP rejection. The ECP-enabled guard branch was not exercised.
5. Optional pytest launch: missing dependency, followed by the direct 80-case run described above.
6. Final long runs: execution environment unavailable. Their completion and final comparisons are unknown.

The first-gradient replay threshold is 2e-8; total/component matrix identity and independent matrix-reference thresholds remain 2e-7 in the corresponding atomic units. No residual, derivative or physical-parent-span tolerance was relaxed.

## Resume

1. Recover/compare the original checkout at /workspace/scratch/9e1c00a57f56/psi4 if it returns. Logs were under /workspace/scratch/9e1c00a57f56/checks/public-hessian. Do not overwrite surviving work.
2. Compare this reconstructed tree with any surviving source. Compile/install the native option and Python modules, configure the new CTest, run syntax/export checks.
3. Run the eight CTests above and public_vibration.py on the exact recovered source. Test ECP-enabled behavior separately if available.
4. Run public_driver.py for water_h2/BOYS, water_h2/PIPEK_MEZEY and ammonia_h2/BOYS, each in separate native scratch. Its existing step-32 references are independent native-gradient data; no new reference samples are claimed by reuse. Require every component to match previous analytic values within 2e-9 and total native references within 2e-7, with actual domain/rank coverage unchanged.
5. Preserve full public matrices/components, settings, actual source/native-binary fingerprints and failure history in a committed aggregate results file. Rerun readiness, equation export/synchronization and TeX.
6. Only after all gates pass, promote with a non-forced expected-head update to dlpno_gradients. Read its current head first; do not overwrite intervening work.

After delivery validation, the next scientific validation is a full equilibrium screened molecular spectrum with nonzero triples/PNO losses and frozen-core response. Rotationally covariant screening, arbitrary branch robustness and measured local runtime/memory scaling remain open.
