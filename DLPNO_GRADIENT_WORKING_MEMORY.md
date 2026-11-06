# Working Memory Bank™

## RHF DLPNO-CCSD(T) analytic nuclear gradients

**Current capability.** The ordinary RHF iterative (T) and stationary CCSD contributions each report explicit Cartesian correlation-fitting derivatives for the auxiliary metric and three-center integrals. The CCSD calculation also forms internal Fock, PAO-overlap and local coefficient derivatives using the converged (T)-sourced Λ₁ and Λ₂. The total-gradient driver remains guarded.

**Next piece.** Combine the direct first-stage projector source with the source from solving the LMP2 adjoint and propagate the stationary derivatives through the two PNO constructions, LMP2, PAOs, localization and RHF orbital relaxation. Include the remaining triples nuclear response. The scoped scaling target assumes a nonzero pair cutoff and bounded local domains.

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

### 2026-11-06 · Codex · Stationary CCSD integral, Fock and overlap derivatives

Implementation commit [80b897b2](https://github.com/andyj10224/psi4/commit/80b897b2f85238668a90e11a5a2564bf0f95c087).

| Change | Result |
| --- | --- |
| Stationary CCSD contribution | Reversed the bare integral, Fock, and pair-overlap occurrences in the implemented right residual and correlation energy with the converged ordinary (T)-sourced Λ. The reverse includes mixed-domain Coulomb/exchange blocks, overlap factors on both ends, and bare occupied Fock elements that couple through a missing pair. The virtual Fock block retains its off-diagonal variations. [Pair response](psi4/src/psi4/dlpno/ccsd_pair_response.cc) |
| Explicit correlation fitting | Distinguished the unfiltered metric solves in the exchange and mixed-domain terms from the rank-filtered symmetric whitening of the other factors. Reused the triples shell contraction to report `DLPNO CCSD AUXILIARY METRIC GRADIENT` and `DLPNO CCSD THREE-CENTER GRADIENT` in hartree/bohr. [DF response](psi4/src/psi4/dlpno/ccsd_df_response.cc) |
| One-body and local-basis response | Backtransformed the occupied/virtual Fock and overlap derivatives into sparse LMO/PAO rows and accumulated both endpoints of each pair coefficient derivative. These internal covectors feed the next orbital-response step. |
| Storage and traversal | Traversed retained occupied neighbors for the G term and accumulated only visited pair overlaps. Extended the memory estimate for the response. Released the pair intermediates after contraction. Added counters for ordered pairs, overlap blocks and candidate kl couplings. |

**Accuracy.** For frozen-core H₂O/cc-pVDZ at zero cutoffs, all nine Cartesian entries in each new array passed the 10⁻⁸ hartree/bohr comparison with the independent canonical calculation. The oxygen z components were −0.0235532256 (metric) and +0.0627802928 (three-center). The PySCF reference holds the geometry and amplitudes fixed, perturbs one fitting integral family along analytic nuclear integral derivatives and evaluates the CCSD Lagrangian with canonical (T)-sourced Λ. Two central step sizes allow extrapolation. [Reference calculation](tests/dlpno-grad-model/canonical_reference.py)

**Finite cutoffs.** On Pipek–Mezey water with semi-direct overlaps and disk-backed pair factors, the absolute directional errors were 2.11 × 10⁻¹² (integral), 1.42 × 10⁻¹² (Fock), 1.23 × 10⁻¹² (overlap), and 6.67 × 10⁻¹² hartree (PNO coefficients). The last check varies arbitrary rectangular coefficients and rebuilds both the correlation integrals and the overlap maps. Five targeted CTests passed. The gradient regression also passed after the memory-estimate update. Lean 4.16 checked the stable kept–kept inverse-square-root metric factor in addition to the prior local-solve variation. The zero-cutoff water PM run printed its existing Hessian-stability warning; its fixed-basis comparisons passed. This checkpoint does not establish localization response at that solution. [Regression](tests/dlpno-grad-model/input.dat) · [Lean proof](tests/dlpno-grad-model/TriplesAdjoint.lean)

**Screened scaling.** Frozen-core waters were placed 17 Å apart with a nonzero crude pair cutoff (`T_CUT_PAIRS_MP2 = 10⁻⁶` hartree). All surviving pairs are promoted to strong in this model. [Reproduce the counts](tests/dlpno-grad-model/screened_counts.py)

| Waters | Ordered pairs | Overlap blocks | kl candidates |
| ---: | ---: | ---: | ---: |
| 1 | 16 | 232 | 160 |
| 2 | 32 | 464 | 320 |
| 3 | 48 | 696 | 480 |

An interacting H₂ chain (0.74 Å bonds perpendicular to the chain, 3.5 Å center spacing, same screening settings) retained nearest-neighbor intermolecular pairs. The four-molecule chain also exercised occupied Fock couplings between absent pairs; all four fixed-geometry derivative checks agreed within 1.2 × 10⁻¹⁰ hartree.

| H₂ molecules | Ordered pairs | Overlap blocks | kl candidates |
| ---: | ---: | ---: | ---: |
| 4 | 10 | 50 | 38 |
| 8 | 22 | 118 | 90 |
| 12 | 34 | 186 | 142 |

The kl count is the sum of squared occupied-domain sizes over unique pairs. With bounded pair degree, bounded PAO/PNO and RI domains and a linear number of retained pairs, the new response costs O(N). Each active fitting shell derivative is evaluated once across all displaced coordinates. The counts above are a contraction-level measurement on these two series. Pre-existing dense maps, reference Fock work, the orbital-response solve, and wall-clock scaling of the complete gradient remain outside this measurement.

**Next piece.** Use the Fock, PAO-overlap and coefficient derivatives to drive orbital and local-space response, including both PNO constructions, the LMP2 correction and the nuclear portion of the TNO response. Add these contributions to the RHF and explicit correlation-fitting terms before comparing a total gradient with canonical DF-CCSD(T).

### 2026-11-06 · Codex · Final PNO retained/discarded response

The stationary CCSD pair coefficient derivative now feeds the last PNO selection. For each truncated unique pair, the gradient model saves the parent basis, LMP2 amplitudes, occupation eigenvectors and final semicanonical rotation. It sums both ordered coefficient derivatives and pulls the retained/discarded rotations through the pair density to the LMP2 amplitudes. Full-rank pairs require no projector source. The procedure uses pair-local matrices and diagnoses an unresolved occupation gap.

At fixed geometry, finite-cutoff water with first and second PNO thresholds of 10⁻⁷ and 10⁻⁵ had ten truncated unique pairs. Perturbing the parent LMP2 amplitudes and rediagonalizing gave an absolute directional discrepancy of 8.0 × 10⁻¹⁰ hartree. This isolates motion of the final retained subspace, holding its internal frame fixed. LMP2 adjoint propagation, first-stage PNOs, the truncation energy, complementary triples and orbital response remain open. [Implementation](psi4/src/psi4/dlpno/ccsd_df_response.cc)

### 2026-11-06 · Codex · Direct first-stage PNO projector response

For each truncated first-stage pair, the RHF gradient model retains the semicanonical PAO eigensystem data needed to differentiate the MP2 pair density. The CCSD coefficient derivative is propagated through the final pair transformation to the first retained subspace, and from there to the semicanonical exchange integral, occupied Fock energy sum and virtual energies. The downstream LMP2 amplitudes and internal semicanonical rotations are held fixed for this direct contribution. The calculation uses the pair's actual kept rank and diagnoses an unresolved occupation gap. [Response](psi4/src/psi4/dlpno/ccsd_df_response.cc)

For frozen-core cc-pVDZ water at first/final thresholds 10⁻⁶/10⁻⁵, ten first-stage pairs were truncated. A simultaneous fixed-geometry perturbation of the exchange blocks and occupied/virtual energies, followed by rediagonalization at fixed rank, agreed with the analytic derivative to 2.8 × 10⁻¹⁰ hartree (four threads, disk-backed pair factors and semi-direct overlaps). At 10⁻⁷/10⁻⁵, one first-stage pair was truncated and the error was 1.1 × 10⁻¹⁰ hartree. [Regression](tests/dlpno-grad-model/input.dat)

Still required for a nuclear gradient: add the implicit LMP2 and truncation-energy sources, complete orbital and local-basis response, and combine the first-stage exchange/energy derivatives with the other integral and Fock derivatives.

## Updating the bank

Append a dated entry for each implementation piece. Record the contributor, implementation commit, changes, checks actually run, discrepancies, and next open item. Keep earlier measurements tied to their original commits. Use canonical DF-CCSD(T) as the nuclear-gradient comparison. Reserve any frozen-structure DLPNO nuclear finite-difference check for zero-cutoff DLPNO-CCSD.
