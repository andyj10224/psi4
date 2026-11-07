import Mathlib.LinearAlgebra.Matrix.DotProduct
import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Abel
import Mathlib.Tactic.Ring
import Mathlib.Tactic.NoncommRing

/- Finite-dimensional identities for the fixed-branch RHF Hessian.
   Run with Lean 4.16.0 and mathlib v4.16.0. -/
namespace DLPNOHessian
open Matrix

-- W acts on the first vector of the pair-weighted product. The two
-- hypotheses are the mixed amplitude equation and the ordinary (T) adjoint.
theorem eliminate_mixed_ccsd {I : Type*} [Fintype I] (J W : Matrix I I ℝ)
    (tuv c eta lam : I → ℝ) (mixed : J *ᵥ tuv = -c)
    (adjoint : J.transpose *ᵥ (W *ᵥ lam) = -(W *ᵥ eta)) :
    (W *ᵥ eta) ⬝ᵥ tuv = (W *ᵥ lam) ⬝ᵥ c := by
  have h : (W *ᵥ lam) ⬝ᵥ (J *ᵥ tuv) = (J.transpose *ᵥ (W *ᵥ lam)) ⬝ᵥ tuv := by
    rw [dotProduct_mulVec, mulVec_transpose]
  rw [mixed, adjoint, dotProduct_neg, neg_dotProduct] at h
  exact (neg_inj.mp h).symm

-- LMP2 uses the unweighted unique-pair Frobenius product: energy
-- multiplicities are already in eta. Flattening the independent symmetric
-- diagonal blocks gives this identity in an orthonormal coordinate chart.
theorem eliminate_mixed_lmp2 {I : Type*} [Fintype I] [DecidableEq I]
    (A : Matrix I I ℝ) (tuv c eta z : I → ℝ)
    (mixed : A *ᵥ tuv = -c) (adjoint : A.transpose *ᵥ z = -eta) :
    eta ⬝ᵥ tuv = z ⬝ᵥ c := by
  simpa using (eliminate_mixed_ccsd A (1 : Matrix I I ℝ) tuv c eta z mixed
    (by simpa using adjoint))

-- The native gradient Lambda has the combined CCSD and ordinary triples
-- source. Subtract the triples-only solution in the same packed metric.
theorem retained_adjoint {I : Type*} [Fintype I] (J W : Matrix I I ℝ)
    (total triples ccSource tSource : I → ℝ)
    (combined : J.transpose *ᵥ (W *ᵥ total) = -(W *ᵥ (ccSource + tSource)))
    (triplesOnly : J.transpose *ᵥ (W *ᵥ triples) = -(W *ᵥ tSource)) :
    J.transpose *ᵥ (W *ᵥ (total - triples)) = -(W *ᵥ ccSource) := by
  rw [mulVec_sub, mulVec_sub, combined, triplesOnly, mulVec_add]
  abel

-- All fifteen coefficients of a bivariate polynomial of total degree ≤ 4.
def quartic (c : ℕ → ℕ → ℝ) (s t : ℝ) : ℝ :=
  c 0 0 + s*(c 1 0 + s*(c 2 0 + s*(c 3 0 + s*c 4 0))) +
  t*(c 0 1 + s*(c 1 1 + s*(c 2 1 + s*c 3 1)) +
     t*(c 0 2 + s*(c 1 2 + s*c 2 2) + t*(c 0 3 + s*c 1 3 + t*c 0 4)))

def oddOdd (c : ℕ → ℕ → ℝ) (a b : ℝ) : ℝ :=
  quartic c a b - quartic c a (-b) - quartic c (-a) b + quartic c (-a) (-b)

-- Cleared-denominator form of the implemented mixed coefficient extraction.
theorem quartic_mixed_coefficient (c : ℕ → ℕ → ℝ) (a b : ℝ) :
    16 * oddOdd c a b - oddOdd c (2*a) (2*b) = 48*a*b*c 1 1 := by
  unfold oddOdd quartic
  ring

-- x=P_uv and c=P_u P_v + P_v P_u. The diagonal blocks follow from
-- differentiated idempotency in any ring, including real square matrices.
theorem retained_projector_block {R : Type*} [Ring R] (p x c : R)
    (hp : p*p = p) (mixed : x = p*x + x*p + c) :
    p*x*p + p*c*p = 0 := by
  have h : p*x*p = p*x*p + (p*x*p + p*c*p) := calc
    p*x*p = p*(p*x + x*p + c)*p := congrArg (fun y => p*y*p) mixed
    _ = (p*p)*x*p + p*x*(p*p) + p*c*p := by noncomm_ring
    _ = p*x*p + (p*x*p + p*c*p) := by rw [hp, add_assoc]
  exact add_left_cancel (h.symm.trans (add_zero _).symm)

theorem discarded_projector_block {R : Type*} [Ring R] (p q x c : R)
    (hqp : q*p = 0) (hpq : p*q = 0) (mixed : x = p*x + x*p + c) :
    q*x*q = q*c*q := calc
    q*x*q = q*(p*x + x*p + c)*q := congrArg (fun y => q*y*q) mixed
    _ = (q*p)*x*q + q*x*(p*q) + q*c*q := by noncomm_ring
    _ = q*c*q := by simp [hqp,hpq]

-- Mixed metric panel solve: rearranging the differentiated S Y = B
-- equation requires all three response terms in addition to S Y_uv.
theorem mixed_metric_panel {R : Type*} [Ring R]
    (s su sv suv y yu yv yuv buv : R)
    (solve : s*yuv = buv-suv*y-su*yv-sv*yu) :
    suv*y+su*yv+sv*yu+s*yuv = buv := by
  rw [solve]
  abel

-- In the fixed orthonormal reference chart the known mixed metric terms
-- are b and the unknown mixed density is x. Occupied and virtual blocks
-- follow from idempotency; only the cross block needs the CPHF solve.
theorem rhf_mixed_density_blocks {R : Type*} [Ring R] (p q x b : R)
    (hp : p*p = p) (hqp : q*p = 0) (hpq : p*q = 0)
    (mixed : x = p*x + x*p + b) :
    (p*x*p + p*b*p = 0) ∧ (q*x*q = q*b*q) := by
  exact ⟨retained_projector_block p x b hp mixed,
         discarded_projector_block p q x b hqp hpq mixed⟩

-- Cleared-denominator second-order orthogonality of exp(u K + v H + uv J):
-- J, K and H are skew; R_uv = J + (K H + H K)/2. The transpose of
-- the symmetric cross term reverses factor order and equals itself.
theorem localization_mixed_rotation {R : Type*} [Ring R] (k h j : R) :
    (2*j+k*h+h*k) + (-2*j+h*k+k*h) - 2*k*h - 2*h*k = 0 := by
  noncomm_ring

end DLPNOHessian
