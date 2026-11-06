# Working Memory Bank™

## RHF DLPNO-CCSD(T) analytic nuclear gradients

**Current capability.** The `dlpno_gradients` branch solves the ordinary iterative (T) triples adjoint and the (T)-sourced CCSD Λ₁ and Λ₂ equations in the retained pair spaces. It reports the two explicit triples DF contributions in Cartesian coordinates. The total nuclear gradient is still pending and the gradient driver stops before scheduling displacements.

**Next piece.** Use the converged multipliers to form the derivatives of the CCSD Lagrangian with respect to the bare integral, Fock, and overlap intermediates. Check those contractions before propagating them through the two PNO constructions, LMP2, PAOs, localization, and RHF orbital relaxation. The TNO response to doubles amplitudes is complete for a fixed, separated rank. The remaining nuclear response also involves motion of the underlying orbitals and metric.

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

### 2026-11-06 · Codex · Ordinary (T)-sourced Λ₁ and Λ₂

Implementation commit [2d3cc920](https://github.com/andyj10224/psi4/commit/2d3cc920e8aa8fd6e60765ebe99dcc5029f6d8a0).

| Area | Change |
| --- | --- |
| Triples sources | Differentiated the ordinary (T) moments with respect to the singles and ordered doubles. The doubles source includes the dependence of the TNO projector on the CCSD pair densities, including virtual Fock and intertriplet overlap terms. A kept/discarded occupation degeneracy that makes this derivative undefined raises an error. [Triples response](psi4/src/psi4/dlpno/triples_response.cc) |
| CCSD Λ equations | The gradient model transposes the implemented t₁-dressed CCSD residual in the same pair domains and projection order used by the right solver. The stored Λ₂ contracts the ordered-pair residuals with weight ½. Local DF integrals can reside on disk through the solve. The asymmetric energy uses its existing Lambda procedure. [Gradient Lambda solver](psi4/src/psi4/dlpno/ccsd_gradient_lambda.cc) |

**Why the transpose changed.** An initial fixed-geometry trial of the existing Lambda residual on truncated-PNO water gave directional discrepancies of about 7.5 × 10⁻⁶ hartree for singles and 1.7 × 10⁻⁶ hartree for doubles. At zero cutoffs the same trial agreed to within 3 × 10⁻¹² hartree. The new regression also tests independent trial multipliers and perturbs each amplitude rank separately.

**Independent canonical check.** Frozen-core H₂O/cc-pVDZ at zero cutoffs, using the same JKFIT and correlation fitting bases as the preceding entry. The Λ contractions are dimensionless. The source contractions are derivatives of the (T) energy with respect to scaling the indicated amplitudes, in hartree. [Canonical PySCF reference](tests/dlpno-grad-model/canonical_reference.py)

| Quantity | DLPNO (Boys) | Canonical |
| --- | ---: | ---: |
| Λ₁ · T₁ | 0.000444023826905 | 0.000444023831502 |
| ½ Σᵢⱼ Λ₂ᵢⱼ · T₂ᵢⱼ | 0.057636460587360 | 0.057636460598332 |
| Triples singles scale derivative | 0.000085316579629 | 0.000085316579810 |
| Triples doubles scale derivative | −0.006166939335671 | −0.006166939336761 |

**Finite-cutoff check.** On water with Pipek–Mezey localization, the TNO rotation contributed 1.13 × 10⁻⁴ hartree along the chosen doubles direction, within 1.8 × 10⁻¹¹ hartree of the difference between rebuilt-TNO and frozen-TNO amplitude probes. On a separated water dimer with 32 of 112 possible triplets, the CCSD transpose checks agreed to within 7 × 10⁻¹² hartree. All probes held the nuclei fixed.

The five targeted CTests (`dlpno-grad-model`, `dfccsd-t-grad1`, `dlpnocc-1`, `dlpnocc-2`, and `dlpnobcc-1`) passed. Coverage includes finite PNO and TNO ranks, semi-direct overlaps, four-thread disk-backed runs, and the empty triples space. Lean checked the ordered-pair source normalization and the separated-occupation TNO multiplier. The touched response and CCSD translation units also compiled individually. The follow-up with this entry adds the direct includes for the standalone build. The separate `dlpnoprop-1` test reproduced the parent’s 2.44 × 10⁻⁵ hartree energy discrepancy. End-to-end scaling has yet to be measured. [Regression](tests/dlpno-grad-model/input.dat) · [Lean proof](tests/dlpno-grad-model/TriplesAdjoint.lean)

## Updating the bank

Append a dated entry for each implementation piece. Record the contributor, implementation commit, changes, checks actually run, discrepancies, and next open item. Keep earlier measurements tied to their original commits. Use canonical DF-CCSD(T) as the nuclear-gradient comparison. Reserve any frozen-structure DLPNO nuclear finite-difference check for zero-cutoff DLPNO-CCSD.
