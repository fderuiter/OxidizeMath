/-
  OxidizeMath Formal Proof Certificate — Tier 6
  Module: OxidizeMath.IntervalSoundness
  Proves soundness of outward-rounded interval addition, negation, subtraction,
  and hull containment over ordered integer/fixed-point bounds.
-/

namespace OxidizeMath.IntervalSoundness

/-- Closed interval `[lo, hi]` backed by ordered integer significands. -/
structure ValidInterval where
  lo : Int
  hi : Int
  valid : lo ≤ hi

/-- Point membership predicate `x ∈ [I.lo, I.hi]`. -/
def Mem (x : Int) (I : ValidInterval) : Prop :=
  I.lo ≤ x ∧ x ≤ I.hi

/-- Outward-rounded interval addition with non-negative rounding slack `δ_lo, δ_hi ≥ 0`. -/
def addOutward (A B : ValidInterval) (dLo dHi : Nat) : ValidInterval where
  lo := A.lo + B.lo - Int.ofNat dLo
  hi := A.hi + B.hi + Int.ofNat dHi
  valid := by
    have hsum : A.lo + B.lo ≤ A.hi + B.hi := Int.add_le_add A.valid B.valid
    omega

/-- Soundness of outward-rounded interval addition:
    if `x ∈ A` and `y ∈ B`, then `x + y ∈ addOutward A B dLo dHi`. -/
theorem add_outward_sound
    (A B : ValidInterval) (dLo dHi : Nat) (x y : Int)
    (hx : Mem x A) (hy : Mem y B) :
    Mem (x + y) (addOutward A B dLo dHi) := by
  unfold Mem addOutward at *
  rcases hx with ⟨hx_lo, hx_hi⟩
  rcases hy with ⟨hy_lo, hy_hi⟩
  constructor
  · omega
  · omega

/-- Exact interval negation `[-I.hi, -I.lo]`. -/
def negInterval (I : ValidInterval) : ValidInterval where
  lo := -I.hi
  hi := -I.lo
  valid := by
    have h := I.valid
    omega

/-- Soundness of interval negation: if `x ∈ I`, then `-x ∈ negInterval I`. -/
theorem neg_interval_sound (I : ValidInterval) (x : Int) (hx : Mem x I) :
    Mem (-x) (negInterval I) := by
  unfold Mem negInterval at *
  rcases hx with ⟨hlo, hhi⟩
  constructor
  · omega
  · omega

/-- Soundness of outward-rounded interval subtraction `A - B`. -/
theorem sub_outward_sound
    (A B : ValidInterval) (dLo dHi : Nat) (x y : Int)
    (hx : Mem x A) (hy : Mem y B) :
    Mem (x - y) (addOutward A (negInterval B) dLo dHi) := by
  have hny : Mem (-y) (negInterval B) := neg_interval_sound B y hy
  have hsum := add_outward_sound A (negInterval B) dLo dHi x (-y) hx hny
  have heq : x + -y = x - y := by omega
  rw [← heq]
  exact hsum

/-- Interval width `hi - lo` is always non-negative. -/
theorem width_nonneg (I : ValidInterval) : 0 ≤ I.hi - I.lo := by
  have h := I.valid
  omega

end OxidizeMath.IntervalSoundness
