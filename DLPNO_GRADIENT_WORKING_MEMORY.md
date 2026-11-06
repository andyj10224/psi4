# Working Memory Bank™

## RHF DLPNO-CCSD(T) analytic nuclear gradients

**Current capability.** The ordinary RHF iterative (T), stationary CCSD, direct PNO projectors, PNO truncation corrections, and LMP2 relaxation report explicit correlation-fitting derivatives in Cartesian coordinates. The model combines the local Fock, overlap, and orbital coefficient derivatives, including moving TNOs, local LMO refits and PAO normalization. The total-gradient driver remains guarded.

**Next piece.** Reverse the occupied-space projector defining the unnormalized PAOs. Assemble its AO and occupied-orbital sources with the fitted-LMO and Fock derivatives, then handle localization and RHF orbital relaxation. The scaling target assumes a nonzero pair cutoff and bounded local domains.

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

### 2026-11-06 · Codex · Both RHF PNO truncation-energy responses

Implementation commit [6d64c701](https://github.com/andyj10224/psi4/commit/6d64c701910517bbdcf22a603654eb8eeabf604e).

The first loss differentiates the semicanonical MP2 energy in the orthonormal PAO pair domain minus its projection into the first retained PNO space. The final loss uses the converged LMP2 amplitudes in that first space and subtracts the energy in the final PNO space. Both include the diagonal-pair singles energy with the forward denominator screen. An off-diagonal unique pair contributes twice. A full-rank stage contributes zero. The final-stage coefficient derivative feeds back through the first selection. The response holds ranks fixed and transports the LMP2 tensor with its first retained space.

The wavefunction reports the combined explicit fitting contributions as `DLPNO PNO TRUNCATION AUXILIARY METRIC GRADIENT` and `DLPNO PNO TRUNCATION THREE-CENTER GRADIENT` in hartree/bohr. The implementation adds the amplitude source from the final loss to the pending LMP2 adjoint RHS. It accumulates the occupied and virtual Fock derivatives, the occupied–virtual singles derivative and the parent coefficient derivatives for later orbital response. Work is local to the truncated unique pairs; each active derivative shell block is contracted once per array. [Implementation](psi4/src/psi4/dlpno/pno_truncation_response.cc)

**Finite-cutoff water.** Frozen core, cc-pVDZ, Boys, first/final PNO cutoffs of 10⁻⁶/10⁻⁵. With four threads, disk-backed pair factors and semi-direct overlaps, the reconstructed energy losses were −0.000044112039044 and −0.001074584005998 hartree. Their sum agreed with the forward correction to within 1.1 × 10⁻¹³ hartree. Maximum per-pair directional errors were 1.8 × 10⁻¹⁰ and 2.0 × 10⁻⁹ hartree, respectively. A separate nonzero singles-Fock probe exercised terms suppressed by the RHF Brillouin condition. Perturbing the primitive fitting integrals and rebuilding both selections gave a maximum error of 3.2 × 10⁻¹⁰ hartree. First-only and final-only truncations gave corresponding errors of 1.5 × 10⁻¹⁰ and 5.2 × 10⁻¹¹ hartree. These probes all use one nuclear geometry.

**Regression and formal check.** The five targeted CTests passed. Coverage includes canonical-limit zero arrays, finite-rank energies, disk/in-core agreement, translation invariance, and a screened H₂ chain. Lean 4.16 verified the bilinear full-minus-projected energy variation including both moving projection factors. [Regression](tests/dlpno-grad-model/input.dat) · [Lean proof](tests/dlpno-grad-model/TriplesAdjoint.lean)

**Next piece.** Solve the LMP2 adjoint on the retained pair graph with the combined projector and truncation-energy sources, then propagate its integral, overlap and local-space contributions. The explicit fitting arrays above hold the parent orbital data fixed; a total gradient still needs LMP2, PAO, localization, RHF orbital, and remaining triples nuclear response.

### 2026-11-06 · Codex · Screened LMP2 adjoint

Implementation commit [202e728c](https://github.com/andyj10224/psi4/commit/202e728c94c20c7f04f550d66f3ac5235b14d525).

The RHF gradient model saves the converged first-space LMP2 amplitudes, pair bases, and the actual occupied Fock couplings selected by the right solve. The production LMP2 residual now gathers from the surviving pair neighbors. The adjoint scatters along the same edges in reverse, transposing the source virtual indices when the edge used ji instead of ij. The independent coordinates are unique pairs, with a symmetric block on ii. The projector and truncation-energy sources already carry the ordered-pair weights.

The matrix-free solve enforces Aᵀz = −s with Jacobi/DIIS and checks the maximum pair RMS on the returned iterate. Its residual contraction supplies the relaxed exchange, occupied and virtual Fock, interpair overlap, and first-stage coefficient derivatives. Reversing the motion of the first PNO projector also supplies a semicanonical MP2 contribution. The wavefunction reports `DLPNO LMP2 RESPONSE AUXILIARY METRIC GRADIENT` and `DLPNO LMP2 RESPONSE THREE-CENTER GRADIENT` in hartree/bohr, with parent orbital data held fixed. [LMP2 solver](psi4/src/psi4/dlpno/lmp2_response.cc)

**Numerical check.** Frozen-core H₂O/cc-pVDZ at first/final PNO cutoffs 10⁻⁶/10⁻⁵. The tangent check perturbs the raw fitting, Fock and PAO-overlap blocks at one geometry, rebuilds the first projector at fixed rank and solves the forward response equations. Errors below are absolute.

| Localization | LMP2 couplings (rectangular) | Adjoint max pair RMS | Transpose dot error | Relaxed vs. tangent derivative error (hartree) |
| --- | ---: | ---: | ---: | ---: |
| Boys | 60 (44) | 9.44 × 10⁻¹³ | 1.11 × 10⁻¹⁶ | 3.53 × 10⁻¹² |
| Pipek–Mezey | 30 (18) | 3.43 × 10⁻¹³ | 3.34 × 10⁻¹⁶ | 6.97 × 10⁻¹² |
| IBO | 30 (22) | 2.56 × 10⁻¹³ | 1.11 × 10⁻¹⁶ | 4.27 × 10⁻¹² |

Boys water converged in 11 updates. Its oxygen z LMP2 contributions were −0.000100895777 (metric) and +0.000266290752 (three-center). Four-thread disk-backed and serial in-memory Cartesian components agreed within 10⁻⁹ hartree/bohr. The zero-cutoff adjoint vanished; with all Fock couplings screened and a nonzero source, one diagonal update solved the equations. The zero-cutoff PM localization again printed a Hessian-stability warning; its fixed-space checks passed.

**Screened chains.** The four-H₂ regression contains 22 couplings, including 18 rectangular overlaps. At a perturbation step of 10⁻⁶ the tangent check initially missed by 1.01 × 10⁻⁸ hartree. Halving the step reduced the discrepancy fourfold. Extrapolating the two centered differences gave 3.11 × 10⁻¹⁴ hartree against the analytic result. A separate size series with the nonzero crude pair cutoff 10⁻⁶ retained 22, 54, and 86 couplings for 4, 8, and 12 H₂ molecules, with three adjoint updates at each size. For 1, 2, and 3 separated waters the corresponding counts were 60, 120, and 180, each with 13 updates. These measurements support linear contraction work for these screened families. End-to-end gradient timing is open. [Reproduction script](tests/dlpno-grad-model/screened_counts.py)

Seven targeted CTests covered the gradient model, the ordinary RHF and RO energy paths that share the LMP2 residual, the Brueckner energy, and the canonical DF gradient regression. Lean 4.16 checked the sign identity between the LMP2 adjoint and forward tangent. [Regression](tests/dlpno-grad-model/input.dat) · [Proof](tests/dlpno-grad-model/TriplesAdjoint.lean)

**Next piece.** Assemble the remaining direct CCSD and triples derivatives with respect to local bases and carry their combined orbital, Fock and overlap sources through PAO construction, localization and RHF relaxation. The total nuclear gradient is still guarded.

### 2026-11-06 · Codex · Triples local spaces, DF orbital refits and PAO normalization

Implementation commit [dc7db3b3](https://github.com/andyj10224/psi4/commit/dc7db3b313d532f74363ec96d8ca3d69d534b906).

| Change | Result |
| --- | --- |
| Ordinary iterative (T) | Reversed both endpoints and the metric of pair–triplet and triplet–triplet overlaps, including projected singles, doubles, and the dependence of the TNO density on the pair basis. Added occupied, occupied–virtual and full virtual Fock derivatives, with multiplicities for repeated occupied labels. The triplet parent metric responds even at full TNO rank. [Triples response](psi4/src/psi4/dlpno/triples_response.cc) |
| Pair spaces | Combined the CCSD and direct (T) coefficient sources, included the direct (T) source in the final PNO projector and LMP2 adjoint, and differentiated the first PNO projector through the semicanonical exchange and Fock blocks. The explicit fitting pieces from that last step are `DLPNO PNO PROJECTOR AUXILIARY METRIC GRADIENT` and `DLPNO PNO PROJECTOR THREE-CENTER GRADIENT`, in hartree/bohr. The truncation-energy and LMP2 fitting terms remain separate. [Local basis response](psi4/src/psi4/dlpno/local_basis_response.cc) |
| Orbital coefficients in the fit | Applied all correlation three-center densities to the screened AO integrals, retaining coefficient derivatives on the actual local supports. Reversed the LMO refit `S_bb L = (S C_lmo)_b` into separate derivatives with respect to its metric and right-hand side. Same-center integral values contribute to orbital motion. [DF response](psi4/src/psi4/dlpno/triples_df_response.cc) |
| PAOs | Combined parent overlap derivatives on the fixed, full-rank Cholesky pivot spans. Reversed the normalization of each occupied-projected AO, including the normalization derivative on the raw PAO metric diagonal. The internal results are sparse derivatives with respect to the unnormalized PAO coefficients and Fock/overlap blocks. The forward zero-column rule also applies in the reverse pass. |

**Fixed-geometry verification.** Frozen-core H₂O/cc-pVDZ, first/final PNO thresholds 10⁻⁶/10⁻⁵, TNO threshold 10⁻⁵. The (T) check perturbs each family separately, rebuilds the TNO selection on the fixed parent span and ranks, and re-solves the right triples equations. Absolute directional errors are in hartree.

| Localization | Final PNO coefficients | Fock blocks | PAO overlap |
| --- | ---: | ---: | ---: |
| Boys | 1.70 × 10⁻¹² | 5.2 × 10⁻¹⁴ | 1.56 × 10⁻¹² |
| Pipek–Mezey (four threads, disk-backed factors and amplitudes) | 9.83 × 10⁻¹³ | 1.54 × 10⁻¹⁴ | 8.20 × 10⁻¹³ |
| IBO | 2.17 × 10⁻¹² | 9.59 × 10⁻¹³ | 5.43 × 10⁻¹³ |

On those runs the largest combined first-PNO operator check was 2.55 × 10⁻¹⁰ hartree. The parent PAO metric, AO coefficient, LMO refit, and PAO normalization checks gave maxima of 2.04 × 10⁻¹², 1.62 × 10⁻¹¹, 8.52 × 10⁻¹² and 2.49 × 10⁻¹¹ hartree, respectively. Reconstructing the fitted forward integral contractions from AO shell values agreed to 2.23 × 10⁻¹⁶ hartree. The four-H₂ screened chain also passed. No nuclear geometry displacements were used for these probes.

**Canonical check.** At zero cutoffs, uniform shifts of the occupied and virtual Fock levels gave canonical PySCF (T) derivatives of −0.001456635184737 and +0.001456635184758, respectively (dimensionless). Boys and PM agreed to within 1.6 × 10⁻¹³; the maximum over all three localizers was 6.0 × 10⁻¹³. The new direct-projector Cartesian arrays were zero at full first PNO rank. Seven targeted CTests cover the gradient model, ordinary RHF and RO energy paths, the Brueckner energy, and the canonical DF gradient. Lean checked the moving-overlap identity, the linear LMO refit pullback, and the PAO normalization factor. The existing PM localization stability warning occurred in the zero-cutoff water run; fixed-basis derivative agreement does not validate a localization Hessian. [Regression](tests/dlpno-grad-model/input.dat) · [PySCF reference](tests/dlpno-grad-model/canonical_reference.py) · [Lean proof](tests/dlpno-grad-model/TriplesAdjoint.lean)

**Screened work.** With `T_CUT_PAIRS_MP2 = 10⁻⁶`, 1, 2, and 3 separated waters gave 132, 264, and 396 triples couplings and 1,824, 3,648, and 5,472 stored raw PAO response entries. At 4, 8, and 12 interacting H₂ molecules the corresponding counts were 12, 28, and 44 couplings and 4,100, 11,020, and 17,940 entries. These are structural counts; the timing of a complete gradient is still open. [Reproduction script](tests/dlpno-grad-model/screened_counts.py)

**Next piece.** Propagate the unnormalized PAO operator and coefficient derivatives through the occupied-space projection, including the actual frozen-core convention. Combine them with the fitted-LMO right-hand-side and AO metric derivatives, then assemble the localization and RHF orbital adjoints and the remaining AO integral derivatives. The total-gradient driver remains guarded.

## Updating the bank

Append a dated entry for each implementation piece. Record the contributor, implementation commit, changes, checks actually run, discrepancies, and next open item. Keep earlier measurements tied to their original commits. Use canonical DF-CCSD(T) as the nuclear-gradient comparison. Reserve any frozen-structure DLPNO nuclear finite-difference check for zero-cutoff DLPNO-CCSD.

## 2026-11-06 — Occupied PAO projection and frozen-core correlation source

Codex propagated the raw PAO coefficient, Fock and metric sources through the occupied projector using the active LMOs and the reference wavefunction's actual frozen occupied columns. The new panel interface also includes the direct occupied-Fock and local DF-refit contributions. An empty frozen subspace returns an AO-by-panel zero. [Implementation](psi4/src/psi4/dlpno/occupied_projection.cc) · [Cumulative LaTeX equation record](DLPNO_GRADIENT_EQUATIONS.tex)

On finite-cutoff frozen-core water the frozen-column probe gave −0.00498059336 hartree. At zero cutoffs a frozen O 1s–virtual rotation gave −0.00003018030227 hartree/radian with Boys, versus −0.00003018029576 from an independent canonical PySCF DF-CCSD(T) stationary correlation calculation. Boys, PM and IBO passed the canonical rotation test, and the empty-core panel passed. Six energy/canonical regression cases passed in the first run. A two-point debug probe exceeded its tolerance on the gradient-model test. Switching to a five-point fixed-input derivative resolved that discrepancy and the whole gradient-model test passed on rerun.

The implementation keeps the local sources sparse and applies dense AO or occupied factors to a supplied panel. Dense AO applications cost quadratic work at fixed panel width. A linear bound for a complete displaced-coordinate calculation has yet to be established. The frozen-core result is an orbital source. The localization and constrained RHF adjoints and remaining explicit AO derivatives are still needed for a relaxed nuclear gradient. Preserve the total-gradient guard.
