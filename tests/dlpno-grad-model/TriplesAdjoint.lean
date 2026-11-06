import Mathlib.LinearAlgebra.Matrix.DotProduct
import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Ring
import Mathlib.Tactic.FieldSimp

/- Run with Lean 4.16.0 and mathlib v4.16.0. These finite-dimensional
identities use the Euclidean inner product on the stored triplet coordinates.
The tangent equations are hypotheses; they specify the smooth fixed-domain
branch. No hypothesis of symmetry of the packed triples operator is used. -/

namespace DLPNO
open Matrix

variable {I J : Type*} [Fintype I] [Fintype J]

theorem pullback (A : Matrix I J ℝ) (y : I → ℝ) (x : J → ℝ) :
    y ⬝ᵥ (A *ᵥ x) = (A.transpose *ᵥ y) ⬝ᵥ x := by
  rw [dotProduct_mulVec, mulVec_transpose]

-- The TNO projection transforms each virtual index with the same rectangular S.
def lift3 (S : Matrix I J ℝ) : Matrix (I × I × I) (J × J × J) ℝ :=
  fun a b => S a.1 b.1 * S a.2.1 b.2.1 * S a.2.2 b.2.2

omit [Fintype I] [Fintype J] in
theorem lift3_transpose (S : Matrix I J ℝ) : (lift3 S).transpose = lift3 S.transpose := rfl

-- Reverse one packed edge, including the source virtual permutation P.
theorem projected_edge (S : Matrix I J ℝ) (P : Matrix (J × J × J) (J × J × J) ℝ)
    (x : J × J × J → ℝ) (y : I × I × I → ℝ) :
    y ⬝ᵥ (lift3 S *ᵥ (P *ᵥ x)) = (P.transpose *ᵥ (lift3 S.transpose *ᵥ y)) ⬝ᵥ x := by
  rw [pullback, lift3_transpose, pullback]

theorem triples_energy (B : Matrix I I ℝ) (w q t z : I → ℝ)
    (right : B *ᵥ t = -w) (left : B.transpose *ᵥ z = -q) :
    q ⬝ᵥ t = z ⬝ᵥ w := by
  have h := pullback B z t
  rw [right, left, dotProduct_neg, neg_dotProduct] at h
  exact (neg_inj.mp h).symm

theorem stationary_variation (B dB : Matrix I I ℝ) (q t z dq dw dt : I → ℝ)
    (left : B.transpose *ᵥ z = -q)
    (tangent : B *ᵥ dt = -(dw + dB *ᵥ t)) :
    dq ⬝ᵥ t + q ⬝ᵥ dt = dq ⬝ᵥ t + z ⬝ᵥ dw + z ⬝ᵥ (dB *ᵥ t) := by
  have h := pullback B z dt
  rw [left, tangent, dotProduct_neg, neg_dotProduct] at h
  rw [← neg_inj.mp h, dotProduct_add, add_assoc]

-- G = a^T J⁻¹ b. With u = J⁻¹ a, v = J⁻¹ b, the third term is the
-- signed two-center Coulomb-metric density used in the Cartesian contraction.
theorem local_fit_variation (M dM : Matrix I I ℝ) (a u v da db dv : I → ℝ)
    (fit : M.transpose *ᵥ u = a) (tangent : M *ᵥ dv = db - dM *ᵥ v) :
    da ⬝ᵥ v + a ⬝ᵥ dv = da ⬝ᵥ v + u ⬝ᵥ db - u ⬝ᵥ (dM *ᵥ v) := by
  have h := pullback M u dv
  rw [fit, tangent, dotProduct_sub] at h
  rw [← h]
  ring

-- In the ordered pairs, reversing an occupied pair transposes its virtual
-- indices. A physical doubles direction is invariant under that permutation.
omit [Fintype J] in
theorem ordered_source (rev : Equiv.Perm I) (s d : I → ℝ)
    (physical : ∀ i, d (rev i) = d i) :
    ((s + fun i => s (rev i)) ⬝ᵥ d) / 2 = s ⬝ᵥ d := by
  have h : (fun i => s (rev i)) ⬝ᵥ d = s ⬝ᵥ d := by
    change (∑ i, s (rev i) * d i) = ∑ i, s i * d i
    calc
      _ = ∑ i, s (rev i) * d (rev i) := Finset.sum_congr rfl (fun i _ => by rw [physical])
      _ = ∑ i, s i * d i := Equiv.sum_comp rev (fun i => s i * d i)
  rw [add_dotProduct, h]
  ring

-- Off-diagonal differentiated eigenproblem on a fixed separated TNO split.
theorem tno_gap_source (nk nd g dD dθ : ℝ) (gap : nk - nd ≠ 0)
    (tangent : (nk - nd) * dθ = dD) :
    (g / (nk - nd)) * dD = g * dθ := by
  rw [← tangent, ← mul_assoc, div_mul_cancel₀ g gap]

-- The kept-kept inverse-square-root divided difference can be evaluated
-- without dividing by the difference of nearly equal metric eigenvalues.
-- a,b are their positive square roots.
theorem half_metric_factor (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hs : a+b ≠ 0) :
    (1/a - 1/b) = (-(1/a)*(1/b)/(a+b)) * (a^2-b^2) := by
  field_simp
  ring

-- A maps a parent pair tensor into a retained pair space. Differentiate
-- k·u - (Ak)·(Au), including the variation of both projection factors.
theorem truncation_loss_variation (A dA : Matrix I J ℝ) (k u dk du : J → ℝ) :
    u ⬝ᵥ dk + k ⬝ᵥ du -
      ((A *ᵥ u) ⬝ᵥ (A *ᵥ dk + dA *ᵥ k) + (A *ᵥ k) ⬝ᵥ (A *ᵥ du + dA *ᵥ u)) =
    (u - A.transpose *ᵥ (A *ᵥ u)) ⬝ᵥ dk + (k - A.transpose *ᵥ (A *ᵥ k)) ⬝ᵥ du -
      ((A *ᵥ u) ⬝ᵥ (dA *ᵥ k) + (A *ᵥ k) ⬝ᵥ (dA *ᵥ u)) := by
  simp only [dotProduct_add, sub_dotProduct, pullback A]
  ring

-- Independent-pair coordinates use the ordinary Frobenius pairing. A already
-- contains the endpoint transposes for a source stored with reversed labels.
theorem lmp2_relaxation (A : Matrix I I ℝ) (s z r dt : I → ℝ)
    (adjoint : A.transpose *ᵥ z = -s) (tangent : A *ᵥ dt = -r) :
    s ⬝ᵥ dt = z ⬝ᵥ r := by
  have h := pullback A z dt
  rw [tangent, adjoint, dotProduct_neg, neg_dotProduct] at h
  exact (neg_inj.mp h).symm

-- Both endpoints of a rectangular pair/triplet overlap can move.
theorem orbital_overlap_variation (M dM : Matrix I J ℝ) (x dx : I → ℝ) (y dy : J → ℝ) :
    dx ⬝ᵥ (M *ᵥ y) + x ⬝ᵥ (dM *ᵥ y + M *ᵥ dy) =
    (M *ᵥ y) ⬝ᵥ dx + x ⬝ᵥ (dM *ᵥ y) + (M.transpose *ᵥ x) ⬝ᵥ dy := by
  rw [dotProduct_add, pullback M x dy, dotProduct_comm dx, add_assoc]

-- The fitted LMO columns solve S L = b in the selected AO rows.
theorem lmo_refit_variation (S dS : Matrix I I ℝ) (g z l db dl : I → ℝ)
    (adjoint : S.transpose *ᵥ z = g) (tangent : S *ᵥ dl = db - dS *ᵥ l) :
    g ⬝ᵥ dl = z ⬝ᵥ db - z ⬝ᵥ (dS *ᵥ l) := by
  have h := pullback S z dl
  rw [adjoint, tangent, dotProduct_sub] at h
  exact h.symm

-- On an active PAO column n² = <B,S B>, P = B/n. dpop is the
-- contraction of the coefficient seed with dB/n and pop with P.
theorem pao_normalization_variation (n dnorm ds pop dpop : ℝ)
    (hn : n ≠ 0) (tangent : ds = 2*n*dnorm) :
    dpop - pop*dnorm/n = dpop - pop*ds/(2*n^2) := by
  rw [tangent]
  field_simp
  ring

end DLPNO
