/-
  OxidizeMath Tier 6 Formal Proof Certificate: Compartmental SEIR Mass Conservation
  Proves closed-population conservation `(S + E + I + R)/scale = N/scale`, bounded
  IEEE-754 significand rounding drift, and compartment non-negativity.
-/
namespace OxidizeMath.CompartmentalConservation

/-- Closed-population SEIR state `(s/S, e/S, i/S, r/S)` over positive IEEE-754 significand scale `S`. -/
structure ScaledSeirState (S : Nat) where
  s : Int
  e : Int
  i : Int
  r : Int
  scale_pos : 0 < S
  deriving DecidableEq, Repr

/-- Exact scaled-rational total population numerator `s + e + i + r` (representing `(S + E + I + R) * scale`). -/
def totalPop {S : Nat} (st : ScaledSeirState S) : Int :=
  st.s + st.e + st.i + st.r

/-- Exact scaled-rational forced SEIR step with transition fluxes `(d_se, d_ei, d_ir)`. -/
def stepForcedSeir {S : Nat} (st : ScaledSeirState S) (d_se d_ei d_ir : Int) : ScaledSeirState S :=
  { s := st.s - d_se
    e := st.e + d_se - d_ei
    i := st.i + d_ei - d_ir
    r := st.r + d_ir
    scale_pos := st.scale_pos }

/-- IEEE-754 significand-rounded forced SEIR step with per-compartment rounding perturbations. -/
def stepForcedSeirRounded {S : Nat} (st : ScaledSeirState S)
    (d_se d_ei d_ir err_s err_e err_i err_r : Int) : ScaledSeirState S :=
  { s := st.s - d_se + err_s
    e := st.e + d_se - d_ei + err_e
    i := st.i + d_ei - d_ir + err_i
    r := st.r + d_ir + err_r
    scale_pos := st.scale_pos }

/-- Exact mass conservation across forced SEIR scaled-rational transitions. -/
theorem forced_seir_exact_conservation {S : Nat} (st : ScaledSeirState S) (d_se d_ei d_ir : Int) :
    totalPop (stepForcedSeir st d_se d_ei d_ir) = totalPop st := by
  dsimp [totalPop, stepForcedSeir]
  omega

/-- Total population drift under IEEE-754 compartment rounding bounded by `slack` ULPs is at most `4 * slack`. -/
theorem forced_seir_rounding_drift_bound {S : Nat} (st : ScaledSeirState S)
    (d_se d_ei d_ir err_s err_e err_i err_r : Int) (slack : Nat)
    (hs : -(slack : Int) ≤ err_s ∧ err_s ≤ (slack : Int))
    (he : -(slack : Int) ≤ err_e ∧ err_e ≤ (slack : Int))
    (hi : -(slack : Int) ≤ err_i ∧ err_i ≤ (slack : Int))
    (hr : -(slack : Int) ≤ err_r ∧ err_r ≤ (slack : Int)) :
    -4 * (slack : Int) ≤
        totalPop (stepForcedSeirRounded st d_se d_ei d_ir err_s err_e err_i err_r) - totalPop st ∧
      totalPop (stepForcedSeirRounded st d_se d_ei d_ir err_s err_e err_i err_r) - totalPop st ≤
        4 * (slack : Int) := by
  dsimp [totalPop, stepForcedSeirRounded]
  omega

/-- Admissible bounded transition fluxes preserve non-negativity of all scaled-rational compartments. -/
theorem forced_seir_nonneg {S : Nat} (st : ScaledSeirState S) (d_se d_ei d_ir : Int)
    (_hs0 : 0 ≤ st.s) (_he0 : 0 ≤ st.e) (_hi0 : 0 ≤ st.i) (hr0 : 0 ≤ st.r)
    (_h_se_lo : 0 ≤ d_se) (h_se_hi : d_se ≤ st.s)
    (_h_ei_lo : 0 ≤ d_ei) (h_ei_hi : d_ei ≤ st.e + d_se)
    (h_ir_lo : 0 ≤ d_ir) (h_ir_hi : d_ir ≤ st.i + d_ei) :
    let next := stepForcedSeir st d_se d_ei d_ir
    0 ≤ next.s ∧ 0 ≤ next.e ∧ 0 ≤ next.i ∧ 0 ≤ next.r := by
  dsimp [stepForcedSeir]
  omega

end OxidizeMath.CompartmentalConservation
