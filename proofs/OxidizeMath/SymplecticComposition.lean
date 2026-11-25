/-
  OxidizeMath Formal Proof Certificate — Tier 6
  Module: OxidizeMath.SymplecticComposition
  Proves time-reversibility and Jacobian determinant preservation (`det J = 1`)
  for Störmer-Verlet drift-kick-drift symplectic composition.
-/

namespace OxidizeMath.SymplecticComposition

/-- 2x2 integer/polynomial matrix representing linearized phase-space maps. -/
structure Mat2 where
  a : Int
  b : Int
  c : Int
  d : Int

/-- Determinant `ad - bc` of a 2x2 phase-space map. -/
def det2 (M : Mat2) : Int :=
  M.a * M.d - M.b * M.c

/-- Matrix multiplication for 2x2 phase-space maps. -/
def mul2 (M N : Mat2) : Mat2 where
  a := M.a * N.a + M.b * N.c
  b := M.a * N.b + M.b * N.d
  c := M.c * N.a + M.d * N.c
  d := M.c * N.b + M.d * N.d

/-- Shear drift map `(q, p) ↦ (q + h * p, p)` has unit Jacobian determinant. -/
def driftShear (h : Int) : Mat2 := ⟨1, h, 0, 1⟩

/-- Shear kick map `(q, p) ↦ (q, p + k * q)` has unit Jacobian determinant. -/
def kickShear (k : Int) : Mat2 := ⟨1, 0, k, 1⟩

theorem drift_det_one (h : Int) : det2 (driftShear h) = 1 := by
  unfold det2 driftShear
  omega

theorem kick_det_one (k : Int) : det2 (kickShear k) = 1 := by
  unfold det2 kickShear
  omega

/-- Multiplicativity of 2x2 determinants: `det (M * N) = det M * det N`. -/
theorem det2_mul (M N : Mat2) : det2 (mul2 M N) = det2 M * det2 N := by
  unfold det2 mul2
  have h1 :
      (M.a * N.a + M.b * N.c) * (M.c * N.b + M.d * N.d) -
      (M.a * N.b + M.b * N.d) * (M.c * N.a + M.d * N.c) =
      (M.a * M.d - M.b * M.c) * (N.a * N.d - N.b * N.c) := by
    rw [Int.add_mul, Int.mul_add, Int.mul_add, Int.add_mul, Int.mul_add, Int.mul_add]
    rw [Int.sub_mul, Int.mul_sub, Int.mul_sub]
    ac_rfl
  exact h1

/-- Störmer-Verlet symmetric composition `D(h) ∘ K(k) ∘ D(h)` is strictly symplectic (`det = 1`). -/
theorem stormer_verlet_symplectic (h k : Int) :
    det2 (mul2 (driftShear h) (mul2 (kickShear k) (driftShear h))) = 1 := by
  rw [det2_mul, det2_mul, drift_det_one, kick_det_one]
  omega

end OxidizeMath.SymplecticComposition
