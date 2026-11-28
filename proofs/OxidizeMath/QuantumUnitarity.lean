/-
  OxidizeMath Tier 6 Formal Proof Certificate: Quantum Unitary Norm Preservation
  Proves complex amplitude squared-norm non-negativity, exact scaled-rational unitary
  phase-rotation norm invariance, and IEEE-754 significand rounding slack bounds.
-/
namespace OxidizeMath.QuantumUnitarity

/-- Squared Euclidean norm numerator `re^2 + im^2` for scaled-rational amplitude `(re/S, im/S)`. -/
def normSq (re im : Int) : Int :=
  re * re + im * im

private theorem sq_nonneg (x : Int) : 0 ≤ x * x := by
  cases Int.le_total 0 x with
  | inl h => exact Int.mul_nonneg h h
  | inr h => exact Int.mul_nonneg_of_nonpos_of_nonpos h h

/-- Every scaled-rational complex amplitude has non-negative squared norm. -/
theorem normSq_nonneg (re im : Int) : 0 ≤ normSq re im := by
  dsimp [normSq]
  have h1 : 0 ≤ re * re := sq_nonneg re
  have h2 : 0 ≤ im * im := sq_nonneg im
  omega

private theorem sq_sub_expand (x y : Int) :
    (x - y) * (x - y) = x * x + y * y - 2 * (x * y) := by
  rw [Int.sub_mul, Int.mul_sub, Int.mul_sub, Int.mul_comm y x]
  omega

private theorem sq_add_expand (x y : Int) :
    (x + y) * (x + y) = x * x + y * y + 2 * (x * y) := by
  rw [Int.add_mul, Int.mul_add, Int.mul_add, Int.mul_comm y x]
  omega

private theorem prod_sq_rearrange (u v : Int) :
    (u * v) * (u * v) = (u * u) * (v * v) := by
  calc
    (u * v) * (u * v) = u * (v * (u * v)) := by rw [Int.mul_assoc]
    _ = u * ((v * u) * v) := by rw [← Int.mul_assoc v u v]
    _ = u * ((u * v) * v) := by rw [Int.mul_comm v u]
    _ = u * (u * (v * v)) := by rw [Int.mul_assoc u v v]
    _ = (u * u) * (v * v) := by rw [← Int.mul_assoc u u (v * v)]

private theorem cross_term_eq (c s re im : Int) :
    (c * re) * (s * im) = (s * re) * (c * im) := by
  calc
    (c * re) * (s * im) = c * (re * (s * im)) := by rw [Int.mul_assoc]
    _ = c * ((re * s) * im) := by rw [← Int.mul_assoc re s im]
    _ = c * ((s * re) * im) := by rw [Int.mul_comm re s]
    _ = (c * (s * re)) * im := by rw [← Int.mul_assoc c (s * re) im]
    _ = ((c * s) * re) * im := by rw [← Int.mul_assoc c s re]
    _ = ((s * c) * re) * im := by rw [Int.mul_comm c s]
    _ = (s * (c * re)) * im := by rw [Int.mul_assoc s c re]
    _ = (s * (re * c)) * im := by rw [Int.mul_comm c re]
    _ = ((s * re) * c) * im := by rw [← Int.mul_assoc s re c]
    _ = (s * re) * (c * im) := by rw [Int.mul_assoc (s * re) c im]

/-- Exact Brahmagupta–Fibonacci unitary rotation identity `(c·re - s·im)^2 + (s·re + c·im)^2 = (c^2 + s^2)(re^2 + im^2)`. -/
theorem phase_rotate_norm_identity (c s re im : Int) :
    normSq (c * re - s * im) (s * re + c * im) = (c * c + s * s) * normSq re im := by
  dsimp [normSq]
  rw [sq_sub_expand, sq_add_expand]
  rw [prod_sq_rearrange c re, prod_sq_rearrange s im, prod_sq_rearrange s re, prod_sq_rearrange c im]
  rw [cross_term_eq c s re im]
  rw [Int.add_mul (c * c) (s * s), Int.mul_add (c * c), Int.mul_add (s * s)]
  omega

/-- Scaled-rational unitary phase factor `(c/S)^2 + (s/S)^2 = 1` preserves scaled squared norm `S^2 * (re^2 + im^2)`. -/
theorem unitary_scaled_norm_preserved (S : Nat) (_hS : 0 < S) (c s re im : Int)
    (h_unit : c * c + s * s = (S : Int) * (S : Int)) :
    normSq (c * re - s * im) (s * re + c * im) =
      ((S : Int) * (S : Int)) * normSq re im := by
  rw [phase_rotate_norm_identity, h_unit]

/-- Outward IEEE-754 significand rounding slack encloses perturbed unitary squared norm within `±slack`. -/
theorem unitary_rounding_slack_bound (S : Nat) (_hS : 0 < S) (c s re im rounded_norm : Int) (slack : Nat)
    (h_unit : c * c + s * s = (S : Int) * (S : Int))
    (herr_lo : -(slack : Int) ≤ rounded_norm - normSq (c * re - s * im) (s * re + c * im))
    (herr_hi : rounded_norm - normSq (c * re - s * im) (s * re + c * im) ≤ (slack : Int)) :
    ((S : Int) * (S : Int)) * normSq re im - (slack : Int) ≤ rounded_norm ∧
      rounded_norm ≤ ((S : Int) * (S : Int)) * normSq re im + (slack : Int) := by
  have hexact := unitary_scaled_norm_preserved S _hS c s re im h_unit
  omega

end OxidizeMath.QuantumUnitarity
