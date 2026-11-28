/-
  OxidizeMath Tier 6 Formal Proof Certificate: Symplectic Shear Composition
  Proves linearized 2×2 symplectic area preservation (`det_S(M_2 · M_1) = S^4`) for
  Störmer–Verlet drift/kick shears over exact scaled rationals and IEEE-754 rounding slack.
-/
namespace OxidizeMath.SymplecticComposition

/-- 2×2 scaled-rational transfer matrix `(1/S) * [[a, b], [c, d]]` at positive IEEE-754 scale `S`. -/
structure ScaledMat2 (S : Nat) where
  a : Int
  b : Int
  c : Int
  d : Int
  scale_pos : 0 < S
  deriving DecidableEq, Repr

/-- Scaled determinant numerator `a * d - b * c` (representing `S^2 * det(M)`). -/
def det2 {S : Nat} (m : ScaledMat2 S) : Int :=
  m.a * m.d - m.b * m.c

/-- Scaled matrix composition `(1/S^2) * (m1 · m2)` at significand scale `S * S`. -/
def mul2 {S : Nat} (m1 m2 : ScaledMat2 S) : ScaledMat2 (S * S) :=
  { a := m1.a * m2.a + m1.b * m2.c
    b := m1.a * m2.b + m1.b * m2.d
    c := m1.c * m2.a + m1.d * m2.c
    d := m1.c * m2.b + m1.d * m2.d
    scale_pos := Nat.mul_pos m1.scale_pos m2.scale_pos }

/-- Symplectic drift shear `[[1, h/S], [0, 1]]` represented at scale `S` as `[[S, h], [0, S]]`. -/
def driftShear {S : Nat} (hS : 0 < S) (h : Int) : ScaledMat2 S :=
  { a := (S : Int), b := h, c := 0, d := (S : Int), scale_pos := hS }

/-- Symplectic momentum kick shear `[[1, 0], [k/S, 1]]` represented at scale `S` as `[[S, 0], [k, S]]`. -/
def kickShear {S : Nat} (hS : 0 < S) (k : Int) : ScaledMat2 S :=
  { a := (S : Int), b := 0, c := k, d := (S : Int), scale_pos := hS }

theorem drift_det_scaled_unit {S : Nat} (hS : 0 < S) (h : Int) :
    det2 (driftShear hS h) = (S : Int) * (S : Int) := by
  dsimp [det2, driftShear]
  rw [Int.mul_zero]
  omega

theorem kick_det_scaled_unit {S : Nat} (hS : 0 < S) (k : Int) :
    det2 (kickShear hS k) = (S : Int) * (S : Int) := by
  dsimp [det2, kickShear]
  rw [Int.zero_mul]
  omega

private theorem mul4_swap23 (w x y z : Int) :
    (w * x) * (y * z) = (w * y) * (x * z) := by
  rw [Int.mul_assoc w x (y * z), ← Int.mul_assoc x y z, Int.mul_comm x y,
      Int.mul_assoc y x z, ← Int.mul_assoc w y (x * z)]

/-- Multiplicative determinant law `det(m1 · m2) = det(m1) * det(m2)` over scaled-rational matrices. -/
theorem det2_mul {S : Nat} (m1 m2 : ScaledMat2 S) :
    det2 (mul2 m1 m2) = det2 m1 * det2 m2 := by
  dsimp [det2, mul2]
  rw [Int.add_mul, Int.mul_add, Int.mul_add]
  rw [Int.add_mul, Int.mul_add, Int.mul_add]
  rw [Int.sub_mul, Int.mul_sub, Int.mul_sub]
  rw [mul4_swap23 m1.a m2.a m1.c m2.b,
      mul4_swap23 m1.a m2.a m1.d m2.d,
      mul4_swap23 m1.b m2.c m1.c m2.b,
      mul4_swap23 m1.b m2.c m1.d m2.d,
      mul4_swap23 m1.a m2.b m1.c m2.a,
      mul4_swap23 m1.a m2.b m1.d m2.c,
      mul4_swap23 m1.b m2.d m1.c m2.a,
      mul4_swap23 m1.b m2.d m1.d m2.c]
  rw [Int.mul_comm m2.a m2.b, Int.mul_comm m2.a m2.d, Int.mul_comm m2.c m2.b,
      Int.mul_comm m2.c m2.d, Int.mul_comm m1.b m1.c]
  omega

/-- Composing any two unit-determinant scaled symplectic maps (`det = S^2`) yields unit determinant (`S^2 * S^2`). -/
theorem symplectic_composition_preserves_scaled_unit_det {S : Nat} (m1 m2 : ScaledMat2 S) (s2 : Int)
    (h1 : det2 m1 = s2) (h2 : det2 m2 = s2) :
    det2 (mul2 m1 m2) = s2 * s2 := by
  rw [det2_mul, h1, h2]

/-- Outward IEEE-754 significand rounding slack encloses the rounded composed determinant within `±slack`. -/
theorem symplectic_det_rounding_slack {S : Nat} (m1 m2 : ScaledMat2 S) (s2 det_fl : Int) (slack : Nat)
    (h1 : det2 m1 = s2) (h2 : det2 m2 = s2)
    (herr_lo : -(slack : Int) ≤ det_fl - det2 (mul2 m1 m2))
    (herr_hi : det_fl - det2 (mul2 m1 m2) ≤ (slack : Int)) :
    s2 * s2 - (slack : Int) ≤ det_fl ∧ det_fl ≤ s2 * s2 + (slack : Int) := by
  have hexact := symplectic_composition_preserves_scaled_unit_det m1 m2 s2 h1 h2
  omega

end OxidizeMath.SymplecticComposition
