# CCSD and triples contraction audit

This follow-up audits Sections 6 and 7 (equations 26–45) of
[`CCSD_RESPONSE_PERFORMANCE_AUDIT.tex`](CCSD_RESPONSE_PERFORMANCE_AUDIT.tex)
against the RHF energy implementation in `ccsd.cc` and `triples.cc`. The
starting revision is `67020eb1ae22565e933c86767f84c23b9e0fbcbf`. “Jiang equation”
below denotes the equation numbers recorded in that source, rather than an
independent revalidation of the publication. The accompanying refactor changes
contraction order, storage ownership, and reuse; it introduces no new domain,
projection, metric, or amplitude approximation.

## Why the implementations looked different

The first native response port translated the independent Python equations into
a generic four-component contraction engine. Each quantity carries its value,
row derivative, column derivative, and mixed derivative. This made the product
rule straightforward to check, but hid the intended matrix shapes and sometimes
selected expensive batched contractions. It was DLPNO-owned code: these native
paths do not call DFOCC or FISAPT contraction routines.

The CCSD and triples stages now express dense contractions as explicit binary
`doublet` operations, with named intermediates corresponding to the energy
code. Storage is `SharedMatrix`. Logical reshapes share that storage without
changing another view's matrix dimensions. Compatible matrices use
`Matrix::gemm`; reshaped views use Psi4's `C_DGEMM` wrapper with explicit leading
dimensions. The mixed product rule accumulates directly into one result per
component:

\[
(AB)^{uv}=A^{uv}B+A^uB^v+A^vB^u+AB^{uv}.
\]

The change avoids separate allocation/addition of each product-rule result.
Permutations that actually reorder data still require packing. Scalar and
elementwise operations remain simple native operations; the general engine
remains available to other response code and independent kernel tests.

## Local fitting and metric rules

Write raw local factors as \(o_{Qmn},h_{Qma},f_{Qab}\), and let \(M_p\) be
the Coulomb metric on the **actual target pair auxiliary list**. Two operators
must remain distinct:

\[
G_p=M_p^{-1},\qquad G_{f,p}=H_p^T H_p,
\quad H_p=M_p^{-1/2}\text{ with the energy code's spectral filter}.
\]

`G_f` retains exactly the positive eigenvalues admitted by the existing
`1e-14` relative cutoff. Its response includes retained/discarded eigenspace
mixing. It is not generally interchangeable with `G`.

| Quantity | Energy construction | Native construction and domain rule |
| --- | --- | --- |
| MP2-origin pair `K_iajb` and `L_iajb` | Full-inverse exchange; `L = 2 K - K^T` | Apply `G` to the small OV factor; form the pair exchange by GEMM. Reverse ordered pairs use its transpose. |
| Dressed CCSD OO/OV/VV contractions | Dot products of factors transformed by `H` | Retain raw factors and `G_f`-weighted factors, then contract them. Metric application commutes with virtual/occupied dressing, so the weighted factors can be reused. |
| Nonprojected mixed `J_ikac` and `K_iakc` | `DGESV` on small `q_io/q_iv`; target fit, extended PAOs, final source-pair virtual transform | Same full inverse, raw masks, and target auxiliary list. No overlap substitutes for a raw cross-pair vertex. |
| Triples OVVV | Full inverse on `q_iv`, then raw `q_vv` | Same binary contraction; no inverse is applied to the larger VV factor. |
| Triples OOOV and V exchange | Products of inverse-half-fitted factors | Apply `G_f` once to a small OV factor and reuse it in the OO/OV dots. |

For a dot product, `(H A)^T (H B) = A^T G_f B`; differentiating the identity
preserves every first and mixed metric term. This permits the different
placement of the metric without inserting a projection. Repeated dressed
CCSD consumers justify retaining the weighted OO/OV/VV factors. For triples
OVVV and mixed J/K, applying the inverse to the small factor is preferable
to fitting a large VV panel. The implementation does not claim one metric
placement is universally optimal for every molecule or BLAS library.

All raw factors retain the atom-specific `AO_OCC`, `AO_VIR`, `OCC`, `PAO`, and
`PAO_PAIRS` masks. An extended PAO union is a way to batch requested columns;
it does not extend the fitting domain or admit new nonzero PAO pairs.

## The two W permutations in equation 28

For target pair \(p=(i,j)\), the energy's `K_ivvv_[ji]` represents
\(W^j_{cab}=(jc|ab)\) in **that pair's virtual and auxiliary spaces**.
The projected singles are \(n_i^p=S_{p,ii}t_i\). With
\(\widehat h_j=G_{f,p}h_j\), the two terms are evaluated as

\[
2\sum_b n_i^b W^j_{cab}
=2\sum_Q\left(\sum_b f_{Qab}n_i^b\right)\widehat h_{Qjc},
\]
\[
\sum_b n_i^b W^j_{bca}
=\sum_Q\left(\sum_b\widehat h_{Qjb}n_i^b\right)f_{Qca}.
\]

The native code keeps the explicit final transpose for the second expression.
This is the same contraction as `compute_delta()` (Jiang 84b), with singles
contracted before forming W. `compute_gamma()` (83b) is treated similarly.
There is no added projector on any W virtual leg. The other gamma/delta terms
retain their original cross-pair overlap chains and each K/L block's own fit.

Equation 38 admits a related saving:

\[
A_a=\sum_{Qcd}\widehat h_{Qkc}U_{dc}f_{Qda},\qquad
Z_{Qd}=\sum_c\widehat h_{Qkc}U_{dc},\qquad
A_a=\sum_{Qd}Z_{Qd}f_{Qda}.
\]

Two GEMMs therefore replace construction of a three-virtual-index W tensor
followed by its contraction. The final `S_PNO(p,ii)` transport is unchanged.

## Section 6: every displayed equation

In this table, `state` denotes the per-residual `CCSDIntermediates` object.
Amplitude-independent fitting panels live longer, in the entry's `Fitting`.

| Report equation | Energy source / Jiang equation | Native contraction and audit result |
| --- | --- | --- |
| 26, beta | `compute_beta()`, 82 | Dressed OO dot plus `sum_Q h_Q T hhat_Q^T`. The ladder uses two packed GEMMs; occupied pair domain is unchanged. |
| 27, gamma | `compute_gamma()`, 83a–d | `-n^T J`, direct W/singles contraction, vector outer product, then explicit two-matrix chains for the five-factor term. All three inter-pair overlaps of 83d remain. |
| 28, delta | `compute_delta()`, 84a–d | `-n^T(2K-J)`, the two W contractions above, vector outer product, and the original U/overlap/L chain. No extra projection of W or replacement of the L fit. |
| 29, initial occupied Fock | `t1_fock()`, 98 | Reuse `g_Q = sum_ma h_Qma n_ma`; contract with weighted OO/OV factors. `Fij` is built before singles assembly. |
| 30, final occupied Fock | `t1_fock()`, 94; `compute_Fkj_double_tilde()`, 86 | Add diagonal-pair OV-Fock/singles correction, then the projected U–K inner product. Each term retains its original occupied list and pair orientation. |
| 31, final virtual Fock | `t1_fock()`, 97; `compute_R_iajb()`, 85 | Reuse partial Fock blocks; subtract `n^T Fov`, then `S^T (U K^T) S`. Form the small `U K^T` before transport. |
| 32, C | `compute_R_iajb()`, 78 | Nonprojected J plus transported gamma; contract with ordered `T_kj^T`, then its overlap. Mixed J batches all k vertices for a target orientation. |
| 33, D | `compute_R_iajb()`, 79 | Same pattern for `2K-J` and delta, retaining the factor one-half and ordered `U_jk^T`. |
| 34, M and G | `compute_R_iajb()`, 19 and 81 | Retain `M = C/2 + C^T + D`; memoize each ordered C/D result. G uses the diagonal `ii` occupied domain exactly as the source. |
| 35, complete R2 | `compute_R_iajb()`, 75–81 | Dressed OV dot, two-GEMM virtual ladder, beta times projected T, virtual Fock products, and both ordered M/G contributions. Diagonal ordered multiplicity is retained. |

The partial Fock exchange now first contracts the occupied index in
`G_f h` with the singles rows; this avoids a larger intermediate carrying
both occupied indices. Raw and weighted T1-dressed factors share this same
matrix-level implementation. The bare Fock keeps the full moving operator,
as required by `gradient_model_`, rather than assuming it remains diagonal
away from the reference semicanonical frame.

## Section 7: every displayed equation

| Report equation | Energy source / Jiang equation | Native contraction and audit result |
| --- | --- | --- |
| 36, diagonal singles base | `t1_fock()`, 96, 98–101; `compute_R_ia()`, 87a | Reuse the partial OV/VV Fock and initial occupied Fock; use matrix-vector products in their original diagonal-pair frames. |
| 37, W definition | `compute_pno_integrals()`, `K_ivvv_` | Raw VV with filtered-metric OV in the target pair. No cross-pair projector is introduced. |
| 38, A addition | `compute_R_ia()`, first part of 88 | Contract U into the fitted OV factor first, then VV, then project the vector to `ii`. |
| 39, occupied-virtual vector | `t1_fock()`, 95 | Reuse projected singles and each source pair's `L_iajb`; transport the resulting vector to the target pair. |
| 40, C addition | `compute_R_ia()`, 90 | Matrix-vector `U b`, then the original diagonal-pair transport. |
| 41, exchange-feedback coefficient | `compute_R_ia()`, second part of 88 | Project ordered `U^T` into the `kl` pair, then take its inner product with `K_kl`. The orientation is explicit. |
| 42, exchange-feedback subtraction | Same, 88 | Scale the existing projected singles vector by that scalar. |
| 43, M row block | `compute_R_ia()`, 89 | Weighted OO/OV dot plus `n K` in the source pair fit. |
| 44, B row block | Same, 89 | Single `M U^T` GEMM. |
| 45, scatter | Same, 89 | Transport each occupied row to its own diagonal frame and accumulate in deterministic order. |

The CCSD scatter is deliberately ordered: a naive OpenMP loop over source
pairs would race on destination singles. Existing independent first-PNO and
triples moment tasks still use OpenMP with private maps and shared immutable
panels. This refactor does not add an unbudgeted residual buffer per thread.

## Mixed J/K and triples

`Fitting::mixed_integrals(i,j,k)` now prepares all k vertices for one ordered
target `(i,j)`, as `compute_pno_integrals()` does. It applies `G_p` to small
`q_io/q_iv` factors once, gathers the actual source PAO lists from an extended
union, and retains the final rectangular J/K matrices. For J it compares:

1. Contract the auxiliary/occupied leg first; perform the two virtual
   transformations on each resulting matrix.
2. Transform the left virtual leg of the raw panel first; contract the
   auxiliary/occupied leg; perform the final source-pair transform.

The multiplication count includes the actual rectangular source domains and
ranks and the cheaper final two-GEMM order. This is a cost model, not a
measurement of packing costs or a proof of globally optimal BLAS performance.
K contracts the extended OV panel with the fitted target IV factor before
the source virtual transformation. Every ordered source PAO list is restored
when slicing. Neither vertex is replaced by `S K S`.

Triples now follows the energy's small-factor metric placement for OVVV,
OOOV and exchange. W and V are assembled with binary GEMMs and the same six
permutations and repeated-index factors. Pair-to-TNO transfers retain the
original overlap definition. Three-leg transfers choose among the six axis
orders, and a first derivative shares the value-transform prefixes instead
of independently repeating three complete transfers. Overlap covectors
transform two virtual legs and finish with a flattened GEMM. Completed
parallel tasks retain results and release their private fitting workspaces.

The audit also checked the gradient pullbacks in `ccsd_df_response.cc` and
`triples_df_response.cc`: they distinguish the full-inverse and filtered-half
paths and accumulate back into local OO/OV/VV densities. They are not replaced
with a generic global fit in this patch. The existing one-time ordinary (T)
Lambda source and persistent Lambda intermediates remain in use.

## Shared derivative panels and printed memory

`Fitting::prepare_raw()` builds atom-local OO/OV/VV four-component panels,
analogous to `qij/qia/qab`, before pair/triplet requests. Later requests gather
from these shared panels instead of repeating AO refits and transformations.
The VV panel retains the original PAO-pair mask; its dense bounding rectangle
may be larger than the energy code's packed `qab_`. It is admitted only if it
fits the existing per-fitting allowance; otherwise direct construction remains
available. First-PNO and CCSD OV panels are shared only when the four captured
OV maps (`AO_OCC`, `AO_VIR`, `OCC`, `PAO`) agree. The first-PNO snapshot does
not contain the VV-only `PAO_PAIRS` mask. Different first-fit domains remain
separate.

The driver separately prepares first AO derivative panels before Hessian
entries, with a non-evicting allowance `min(requested_memory/32, 512 MiB)`.
This stores one set per admitted atom and auxiliary basis for the full Hessian
lifetime. Unadmitted panels use the existing bounded streaming fallback.
Reference metric/three-center panels are also owned by that Hessian reference.
The memory preflight reserves the new allowance.

Output reports reference AO DF storage, retained/requested first AO storage,
and native local storage in MiB for **value / row / column / mixed**:

- `qij`, `qia`, `qab` atom panels;
- projected PNO factors and local inverse metrics;
- metric-weighted factors and final mixed J/K vertices.

Native category/component values are cumulative peaks sampled after CCSD
evaluations. Shared allocations are counted once within a category/component.
These categories overlap existing native storage counters; they must not be
added to them as independent allocations, and category peaks need not occur
simultaneously. They are numerical-array accounting, not process RSS.

Mixed AO three-center derivatives are still delivered once per ordered
nuclear-coordinate pair and also serve the RHF response. This patch reduces
reconstruction and local transformations; it does not make integral delivery
fully local or retain all dense second-derivative AO panels.

## Validation and performance scope

The native checks compare complete CCSD, PNO/LMP2, triples sources/operators/
curvature, screened terms, and individual kernels against independent Python
equations. Added cases cover transpose flags, SharedMatrix lifetime,
rectangular transformation order, different first/final AO domains, reordered
source PAOs, unequal auxiliary domains, discarded metric eigenvalues, and
zero/small/resident storage budgets. All 137 native checks passed in serial
and OpenMP builds (one/four threads); another 261 existing algebra, kernel
and routing checks passed. Existing numerical tolerances are unchanged.

Native sources and bindings are compiled in an isolated harness using real
BLAS/LAPACK and small Matrix/thread API adapters. Separate syntax checks use
the actual Psi4 headers. Driver tests mock the molecular boundary. These are
not a complete linked Psi4 build or molecular finite-difference validation.

An isolated rank-48 CCSD residual benchmark uses four occupied orbitals,
ten unique pairs, 144 auxiliaries, a 512 MiB native allowance, and one
BLAS/OpenMP thread. The preceding `67020eb` results were 5.05 s cold and
2.61–2.85 s for two warm evaluations of the **same mixed entry**. The final
refactor measured 4.05 s cold and 1.30–1.34 s warm. This
demonstrates a reduction in implementation cost; it is not a water Hessian
prediction or evidence that the user's energy-relative targets are met.

Rebuild native C++ and install Python together. Repeat the existing cc-pVDZ
derivative checks and the same water/cc-pVTZ input, comparing raw Hessians,
frequencies, first-response counts, CCSD subsection times and printed memory.
All ordered Hessian entries, response consistency checks, rank/metric
thresholds, and rigid-motion checks remain enabled.
