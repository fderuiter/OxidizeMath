/-
  OxidizeMath Tier 6 Formal Proof Certificate: Outward-Rounded Interval Soundness
  Core Lean 4 (zero external dependencies) formalization over exact scaled rationals
  and non-negative IEEE-754 significand rounding slack (`next_down` / `next_up`).
-/
namespace OxidizeMath.IntervalSoundness

/-- Exact scaled rational `sig / S` at positive IEEE-754 significand scale `S`. -/
structure ScaledRat where
  sig : Int
  scale : Nat
  scale_pos : 0 < scale
  deriving DecidableEq, Repr

/-- Cross-multiplied exact rational ordering `a.sig / a.scale ≤ b.sig / b.scale`. -/
def leScaledRat (a b : ScaledRat) : Prop :=
  a.sig * (b.scale : Int) ≤ b.sig * (a.scale : Int)

theorem leScaledRat_refl (a : ScaledRat) : leScaledRat a a := by
  dsimp [leScaledRat]
  exact Int.le_refl _

/-- Closed interval `[lo / S, hi / S]` over exact scaled rationals at positive IEEE-754 scale `S > 0`. -/
structure ScaledInterval (S : Nat) where
  lo : Int
  hi : Int
  scale_pos : 0 < S
  valid : lo ≤ hi
  deriving DecidableEq, Repr

/-- Membership predicate `x / S ∈ [iv.lo / S, iv.hi / S]` at shared significand scale `S`. -/
def containsScaled {S : Nat} (x : Int) (iv : ScaledInterval S) : Prop :=
  iv.lo ≤ x ∧ x ≤ iv.hi

/-- Outward `k`-ULP point enclosure `[x - δ_lo, x + δ_hi]` with non-negative IEEE-754 significand slack. -/
def pointEnclosure {S : Nat} (hS : 0 < S) (x : Int) (δ_lo δ_hi : Nat) : ScaledInterval S :=
  { lo := x - (δ_lo : Int)
    hi := x + (δ_hi : Int)
    scale_pos := hS
    valid := by omega }

theorem point_enclosure_contains {S : Nat} (hS : 0 < S) (x : Int) (δ_lo δ_hi : Nat) :
    containsScaled x (pointEnclosure hS x δ_lo δ_hi) := by
  dsimp [containsScaled, pointEnclosure]
  omega

/-- Outward-rounded scaled-rational interval addition with non-negative IEEE-754 significand slack. -/
def addOutward {S : Nat} (a b : ScaledInterval S) (δ_lo δ_hi : Nat) : ScaledInterval S :=
  { lo := a.lo + b.lo - (δ_lo : Int)
    hi := a.hi + b.hi + (δ_hi : Int)
    scale_pos := a.scale_pos
    valid := by
      have ha := a.valid
      have hb := b.valid
      omega }

theorem add_outward_sound {S : Nat} (a b : ScaledInterval S) (x y : Int) (δ_lo δ_hi : Nat)
    (hx : containsScaled x a) (hy : containsScaled y b) :
    containsScaled (x + y) (addOutward a b δ_lo δ_hi) := by
  dsimp [containsScaled, addOutward] at *
  omega

/-- Exact IEEE-754 sign-bit interval negation `[-a.hi / S, -a.lo / S]` (zero rounding slack). -/
def negInterval {S : Nat} (a : ScaledInterval S) : ScaledInterval S :=
  { lo := -a.hi
    hi := -a.lo
    scale_pos := a.scale_pos
    valid := by
      have ha := a.valid
      omega }

theorem neg_sound {S : Nat} (a : ScaledInterval S) (x : Int) (hx : containsScaled x a) :
    containsScaled (-x) (negInterval a) := by
  dsimp [containsScaled, negInterval] at *
  omega

/-- Outward-rounded scaled-rational interval subtraction `a - b = a + (-b)` with significand slack. -/
def subOutward {S : Nat} (a b : ScaledInterval S) (δ_lo δ_hi : Nat) : ScaledInterval S :=
  addOutward a (negInterval b) δ_lo δ_hi

theorem sub_outward_sound {S : Nat} (a b : ScaledInterval S) (x y : Int) (δ_lo δ_hi : Nat)
    (hx : containsScaled x a) (hy : containsScaled y b) :
    containsScaled (x - y) (subOutward a b δ_lo δ_hi) := by
  dsimp [containsScaled, subOutward, addOutward, negInterval] at *
  omega

/-- Closed interval hull `[min a.lo b.lo, max a.hi b.hi]` at shared significand scale `S`. -/
def hullInterval {S : Nat} (a b : ScaledInterval S) : ScaledInterval S :=
  { lo := min a.lo b.lo
    hi := max a.hi b.hi
    scale_pos := a.scale_pos
    valid := by
      have ha := a.valid
      omega }

theorem hull_sound_left {S : Nat} (a b : ScaledInterval S) (x : Int)
    (hx : containsScaled x a) :
    containsScaled x (hullInterval a b) := by
  dsimp [containsScaled, hullInterval] at *
  omega

theorem hull_sound_right {S : Nat} (a b : ScaledInterval S) (y : Int)
    (hy : containsScaled y b) :
    containsScaled y (hullInterval a b) := by
  dsimp [containsScaled, hullInterval] at *
  omega

end OxidizeMath.IntervalSoundness
