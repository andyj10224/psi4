# Working Memory Bank™

## RHF DLPNO-CCSD(T) analytic nuclear gradients

**Current capability.** The `dlpno_gradients` branch computes two Cartesian contributions to the ordinary iterative (T) correction with `DLPNO_GRADIENT_MODEL = true`. The triples amplitudes respond through their adjoint. The CCSD amplitudes, Fock matrix, coefficient refits and orbital spaces are held fixed in these two contributions. A total nuclear gradient is still pending. The driver stops a total-gradient request before it schedules displacements.

**Next piece.** Form the ordinary (T) sources for the CCSD Λ₁ and Λ₂ equations, including the dependence of the TNO space on the doubles amplitudes. Check the transpose of the implemented t₁-transformed CCSD residual at fixed geometry. Subsequent pieces must account for the PNO correction and the local-space, localization, and RHF orbital responses.

## Implementation history

### 2026-11-06 · Codex · Ordinary triples adjoint and explicit DF derivatives

Implementation commit [e19cd479](https://github.com/andyj10224/psi4/commit/e19cd479e60af8378813cc35701164be41904dfc), based on [6af73889](https://github.com/andyj10224/psi4/commit/6af73889cbec5d2a299c3c03d644a914642a91f4).

| Area | Change |
| --- | --- |
| Energy functional | Added an opt-in closed-shell RHF model with all surviving pairs strong, both PNO truncation corrections, and one TNO space per retained triplet. Omitted the crude eliminated-pair and dipole energy estimates. Boys, Pipek–Mezey, and IBO are available for the energy. [Model documentation](doc/sphinxman/source/dlpnocc.rst) |
| Triples response | Added right T₃ and left Z₃ solves on the stored ordered triplets. The transpose reverses each surviving coupling, including its virtual-index permutation and rectangular TNO projection. Implemented in-memory and PSIO-backed iterations. [Solver](psi4/src/psi4/dlpno/triples_response.cc) |
| Cartesian contributions | Added `DLPNO (T) AUXILIARY METRIC GRADIENT` and `DLPNO (T) THREE-CENTER GRADIENT`. Each is an atoms-by-three array in hartree/bohr. The contractions accumulate local shell blocks. End-to-end linear scaling remains to be measured. [DF response](psi4/src/psi4/dlpno/triples_df_response.cc) |

**Numerical checks.** Frozen-core H₂O/cc-pVDZ at zero cutoffs was compared with an independent canonical DF-CCSD(T) calculation. At a single geometry, the reference varied the chosen fitting integrals along their analytic nuclear derivatives with the orbitals, Fock matrix and CCSD amplitudes fixed. [Reference generator](tests/dlpno-grad-model/canonical_reference.py)

| Contribution | DLPNO oxygen z (hartree/bohr) | Canonical oxygen z (hartree/bohr) | Maximum absolute difference across 9 components (hartree/bohr) |
| --- | ---: | ---: | ---: |
| Auxiliary metric | 0.000436162668 | 0.000436162623 | 4.5 × 10⁻¹¹ |
| Three-center integrals | 0.00190046911 | 0.00190046913 | 1.8 × 10⁻¹¹ |

The four targeted CTests (`dlpno-grad-model`, `dfccsd-t-grad1`, `dlpnocc-1`, and `dlpnocc-2`) all passed. The new regression also checked all three localizers at zero cutoffs, agreement with the original right-hand solver in matching finite TNO spaces, four-thread disk versus serial in-memory results, and the empty triples space in H₂. Lean 4.16.0 with mathlib v4.16.0 checked the projection and transpose identities, the stationary triples variation, and the local-fit variation. [Regression input](tests/dlpno-grad-model/input.dat) · [Lean proof](tests/dlpno-grad-model/TriplesAdjoint.lean)

**Inherited discrepancy.** The separate `dlpnoprop-1` test failed its DF-CCSD versus DLPNO-CCSD energy comparison by 2.44 × 10⁻⁵ hartree on both the parent and implementation commits. Its tolerance was 10⁻⁶ hartree. [Existing test](tests/dlpnoprop-1/input.dat)

## Updating the bank

Append a dated entry for each implementation piece. Record the contributor, implementation commit, changes, checks actually run, discrepancies, and next open item. Keep earlier measurements tied to their original commits. Use canonical DF-CCSD(T) as the nuclear-gradient comparison. Reserve any frozen-structure DLPNO nuclear finite-difference check for zero-cutoff DLPNO-CCSD.
