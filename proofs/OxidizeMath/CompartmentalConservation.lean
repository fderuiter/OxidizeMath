/-
  OxidizeMath Formal Proof Certificate — Tier 6
  Module: OxidizeMath.CompartmentalConservation
  Proves exact total-population conservation `S + E + I + R = N` across discrete
  and forced compartmental SIR/SEIR flux transitions.
-/

namespace OxidizeMath.CompartmentalConservation

/-- Four-compartment SEIR state with flux transfers `(f_se, f_ei, f_ir)`. -/
structure SeirState where
  s : Int
  e : Int
  i : Int
  r : Int

/-- Total population `N = S + E + I + R`. -/
def totalPop (st : SeirState) : Int :=
  st.s + st.e + st.i + st.r

/-- Time-dependent forced SEIR transition step with arbitrary transmission `f_se(t)`,
    progression `f_ei(t)`, and recovery `f_ir(t)`. -/
def stepForcedSeir (st : SeirState) (f_se f_ei f_ir : Int) : SeirState where
  s := st.s - f_se
  e := st.e + f_se - f_ei
  i := st.i + f_ei - f_ir
  r := st.r + f_ir

/-- Exact conservation of total population for arbitrary time-dependent SEIR fluxes. -/
theorem seir_step_conserves_population
    (st : SeirState) (f_se f_ei f_ir : Int) :
    totalPop (stepForcedSeir st f_se f_ei f_ir) = totalPop st := by
  unfold totalPop stepForcedSeir
  omega

/-- SIR specialization (`E = 0`, `f_se = f_ei`) preserves `E = 0` and total population. -/
theorem sir_reduction_conserves_population
    (s i r f_si f_ir : Int) :
    let st : SeirState := ⟨s, 0, i, r⟩
    let next := stepForcedSeir st f_si f_si f_ir
    next.e = 0 ∧ totalPop next = totalPop st := by
  intro st next
  constructor
  · dsimp [next, st, stepForcedSeir]
    omega
  · exact seir_step_conserves_population st f_si f_si f_ir

end OxidizeMath.CompartmentalConservation
