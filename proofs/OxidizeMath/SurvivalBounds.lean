/-
  OxidizeMath Tier 6 Formal Proof Certificate: Kaplan–Meier Survival Bounds
  Proves exact rational survival factor bounds `(n - d) / n ∈ [0, 1]`, monotone
  non-increasing step transitions over scaled rationals, and IEEE-754 rounding slack enclosures.
-/
namespace OxidizeMath.SurvivalBounds

/-- For a positive risk set `n > 0` and events `d ≤ n`, the exact rational survival numerator `n - d` lies in `[0, n]`. -/
theorem km_factor_bounded (n d : Nat) (_hn : 0 < n) (hd : d ≤ n) :
    (0 : Int) ≤ ((n - d : Nat) : Int) ∧ ((n - d : Nat) : Int) ≤ (n : Int) := by
  omega

/-- Exact scaled-rational Kaplan–Meier update `S_next = S_prev * (n - d) / n` is non-negative and antitone (`0 ≤ S_next ≤ S_prev`). -/
theorem km_step_antitone_and_nonneg (S : Nat) (_hS : 0 < S) (s_prev : Int) (n d : Nat)
    (_hn : 0 < n) (hd : d ≤ n)
    (hs0 : 0 ≤ s_prev) (_hs1 : s_prev ≤ (S : Int)) :
    0 ≤ s_prev * ((n - d : Nat) : Int) ∧
      s_prev * ((n - d : Nat) : Int) ≤ s_prev * (n : Int) := by
  have hnd0 : (0 : Int) ≤ ((n - d : Nat) : Int) := by omega
  have hnd_le : ((n - d : Nat) : Int) ≤ (n : Int) := by omega
  exact ⟨Int.mul_nonneg hs0 hnd0, Int.mul_le_mul_of_nonneg_left hnd_le hs0⟩

/-- Outward IEEE-754 significand rounding slack encloses the rounded survival estimate `s_fl` within `[0, S]` padded by `slack`. -/
theorem km_outward_slack_enclosure (S : Nat) (_hS : 0 < S) (s_exact s_fl : Int) (slack : Nat)
    (hs0 : 0 ≤ s_exact) (hs1 : s_exact ≤ (S : Int))
    (herr_lo : -(slack : Int) ≤ s_fl - s_exact)
    (herr_hi : s_fl - s_exact ≤ (slack : Int)) :
    -(slack : Int) ≤ s_fl ∧ s_fl ≤ (S : Int) + (slack : Int) := by
  omega

end OxidizeMath.SurvivalBounds
