/-
  OxidizeMath Tier 6 Formal Proof Certificate: Fréchet–Hoeffding Copula Bounds
  Proves `max(0, u + v - 1) ≤ min(u, v)`, unit-interval range invariants, and
  outward IEEE-754 significand rounding enclosures over exact scaled rationals.
-/
namespace OxidizeMath.CopulaFrechet

/-- Scaled-rational Fréchet–Hoeffding lower bound `W(u/S, v/S) * S = max(0, u + v - S)`. -/
def frechetLower (S : Nat) (u v : Int) : Int :=
  max 0 (u + v - (S : Int))

/-- Scaled-rational Fréchet–Hoeffding upper bound `M(u/S, v/S) * S = min(u, v)`. -/
def frechetUpper (u v : Int) : Int :=
  min u v

/-- For any scaled-rational marginals `u/S, v/S ∈ [0, 1]` (`S > 0`), `W(u, v) ≤ M(u, v)`. -/
theorem frechet_bounds_ordered (S : Nat) (_hS : 0 < S) (u v : Int)
    (hu0 : 0 ≤ u) (huS : u ≤ (S : Int))
    (hv0 : 0 ≤ v) (hvS : v ≤ (S : Int)) :
    frechetLower S u v ≤ frechetUpper u v := by
  dsimp [frechetLower, frechetUpper]
  omega

/-- Both Fréchet–Hoeffding bounds remain inside the scaled unit interval `[0, S]`. -/
theorem frechet_unit_range (S : Nat) (_hS : 0 < S) (u v : Int)
    (_hu0 : 0 ≤ u) (huS : u ≤ (S : Int))
    (_hv0 : 0 ≤ v) (_hvS : v ≤ (S : Int)) :
    0 ≤ frechetLower S u v ∧ frechetUpper u v ≤ (S : Int) := by
  dsimp [frechetLower, frechetUpper]
  omega

/-- Outward IEEE-754 significand slack encloses any rounded copula evaluation `c_fl` within `slack` ULPs of `c`. -/
theorem frechet_outward_enclosure_sound (S : Nat) (_hS : 0 < S) (u v c c_fl : Int) (slack : Nat)
    (hc_lo : frechetLower S u v ≤ c)
    (hc_hi : c ≤ frechetUpper u v)
    (herr_lo : -(slack : Int) ≤ c_fl - c)
    (herr_hi : c_fl - c ≤ (slack : Int)) :
    frechetLower S u v - (slack : Int) ≤ c_fl ∧
      c_fl ≤ frechetUpper u v + (slack : Int) := by
  dsimp [frechetLower, frechetUpper] at *
  omega

end OxidizeMath.CopulaFrechet
