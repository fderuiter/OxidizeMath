/-
  OxidizeMath Formal Proof Certificate — Tier 6
  Module: OxidizeMath.QuantumUnitarity
  Proves exact norm preservation `‖U ψ‖² = ‖ψ‖²` under 2D planar phase rotations
  `(re, im) ↦ (c * re - s * im, s * re + c * im)` satisfying Pythagorean identity `c² + s² = R²`.
-/

namespace OxidizeMath.QuantumUnitarity

/-- Squared Euclidean norm of a scaled complex amplitude `(re, im)`. -/
def normSq (re im : Int) : Int :=
  re * re + im * im

/-- Unitary phase rotation scaled by `R`: `(c * re - s * im, s * re + c * im)`. -/
def phaseRotate (c s re im : Int) : Int × Int :=
  (c * re - s * im, s * re + c * im)

/-- Brahmagupta-Fibonacci / unitary norm identity:
    `‖(c + i s)(re + i im)‖² = (c² + s²) * (re² + im²)`. -/
theorem phase_rotate_norm_identity (c s re im : Int) :
    let rot := phaseRotate c s re im
    normSq rot.1 rot.2 = (c * c + s * s) * normSq re im := by
  dsimp [phaseRotate, normSq]
  have h :
      (c * re - s * im) * (c * re - s * im) + (s * re + c * im) * (s * re + c * im) =
      (c * c + s * s) * (re * re + im * im) := by
    rw [Int.sub_mul, Int.mul_sub, Int.mul_sub]
    rw [Int.add_mul, Int.mul_add, Int.mul_add]
    rw [Int.add_mul, Int.mul_add, Int.mul_add]
    omega
  exact h

/-- When `c² + s² = 1`, the quantum phase propagator strictly preserves `‖ψ‖²`. -/
theorem unitary_phase_preserves_norm
    (c s re im : Int) (hunit : c * c + s * s = 1) :
    let rot := phaseRotate c s re im
    normSq rot.1 rot.2 = normSq re im := by
  dsimp
  rw [phase_rotate_norm_identity c s re im, hunit, Int.one_mul]

end OxidizeMath.QuantumUnitarity
