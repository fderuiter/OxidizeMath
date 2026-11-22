//! Preallocated pharmacokinetic dosing workspace and `pk_therapeutic_window` inverse challenge.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::framework::{CalloutManager, ChallengeEvaluation, ChallengeMetadata};
use math_explorer::applied::pharmacokinetics::{
    half_life, solve_ka, t_max, BatemanModel, PKParameters, PharmacokineticModel,
    SuperpositionModel,
};
use serde::{Deserialize, Serialize};

/// Number of uniform concentration samples across `[0.0, 48.0]` hours (`dt = 0.1` h).
pub const PK_SAMPLE_COUNT: usize = 481;
/// Total dosing horizon in hours.
pub const PK_HORIZON_HOURS: f64 = 48.0;
/// Sample index corresponding to `t = 12.0` h (`12.0 / 0.1 = 120`).
pub const PK_WINDOW_START_INDEX: usize = 120;
/// Maximum preallocated dose timestamps in [`SuperpositionModel`].
const MAX_PREALLOCATED_DOSES: usize = 256;

/// Canonical challenge ID for the multi-dose therapeutic window inverse problem.
pub const CHALLENGE_PK_THERAPEUTIC_WINDOW: &str = "pk_therapeutic_window";

/// Locked oral bioavailability fraction `F` for `pk_therapeutic_window`.
pub const CHALLENGE_PK_F: f64 = 0.85;
/// Locked elimination rate constant `k_e` (1/h) for `pk_therapeutic_window`.
pub const CHALLENGE_PK_KE: f64 = 0.12;
/// Locked single-dose `T_max` target (h) solved via `solve_ka` for `pk_therapeutic_window`.
pub const CHALLENGE_PK_TMAX_TARGET: f64 = 1.5;
/// Locked apparent volume of distribution `V` (L) for `pk_therapeutic_window`.
pub const CHALLENGE_PK_VOLUME_L: f64 = 25.0;
/// Locked minimum effective concentration `C_MEC` (mg/L) for `pk_therapeutic_window`.
pub const CHALLENGE_PK_C_MEC: f64 = 4.0;
/// Locked minimum toxic concentration `C_MTC` (mg/L) for `pk_therapeutic_window`.
pub const CHALLENGE_PK_C_MTC: f64 = 12.0;

/// Serializable configuration for [`super::PharmacokineticsTool`].
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PkRegimenState {
    /// Maintenance oral dose `D` (mg).
    pub dose_mg: f64,
    /// Whether an initial loading dose `D_load` is administered at `t = 0` h instead of `D`.
    pub use_loading_dose: bool,
    /// Optional initial loading dose `D_load` (mg) at `t = 0` h.
    pub loading_dose_mg: f64,
    /// Dosing interval `delta_t` (hours).
    pub interval_hours: f64,
    /// Oral bioavailability fraction `F` in `(0.0, 1.0]`.
    pub bioavailability: f64,
    /// Elimination rate constant `k_e` (1/h).
    pub ke: f64,
    /// Target single-dose time-to-peak `T_max` (h) used to invert `k_a` via `solve_ka`.
    pub t_max_target: f64,
    /// Apparent volume of distribution `V` (L).
    pub volume_l: f64,
    /// Minimum effective plasma concentration `C_MEC` (mg/L).
    pub c_mec: f64,
    /// Minimum toxic plasma concentration `C_MTC` (mg/L).
    pub c_mtc: f64,
}

impl Default for PkRegimenState {
    fn default() -> Self {
        Self {
            dose_mg: 165.0,
            use_loading_dose: false,
            loading_dose_mg: 240.0,
            interval_hours: 6.0,
            bioavailability: CHALLENGE_PK_F,
            ke: CHALLENGE_PK_KE,
            t_max_target: CHALLENGE_PK_TMAX_TARGET,
            volume_l: CHALLENGE_PK_VOLUME_L,
            c_mec: CHALLENGE_PK_C_MEC,
            c_mtc: CHALLENGE_PK_C_MTC,
        }
    }
}

impl PkRegimenState {
    /// Validates finiteness and physiological parameter bounds.
    pub fn validate(&self) -> Result<(), String> {
        let vals = [
            self.dose_mg,
            self.loading_dose_mg,
            self.interval_hours,
            self.bioavailability,
            self.ke,
            self.t_max_target,
            self.volume_l,
            self.c_mec,
            self.c_mtc,
        ];
        if !vals.iter().all(|v| v.is_finite()) {
            return Err("Pharmacokinetic parameters must be finite".to_string());
        }
        if !(0.0..=1000.0).contains(&self.dose_mg)
            || !(0.0..=1500.0).contains(&self.loading_dose_mg)
            || !(1.0..=24.0).contains(&self.interval_hours)
            || !(0.05..=1.0).contains(&self.bioavailability)
            || !(0.01..=1.0).contains(&self.ke)
            || !(0.2..=8.0).contains(&self.t_max_target)
            || !(5.0..=200.0).contains(&self.volume_l)
            || !(0.1..=50.0).contains(&self.c_mec)
            || self.c_mtc <= self.c_mec
            || self.c_mtc > 100.0
        {
            return Err("Pharmacokinetic parameters out of physiological bounds".to_string());
        }
        let max_t_max = (1.0 / self.ke) * 0.98;
        if self.t_max_target >= max_t_max {
            return Err(format!(
                "t_max_target ({:.2} h) must be < {:.2} h for k_e = {:.3} 1/h",
                self.t_max_target, max_t_max, self.ke
            ));
        }
        Ok(())
    }

    /// Clamps locked physiological constants when `pk_therapeutic_window` is active.
    pub fn enforce_challenge_locks(&mut self, challenge_id: Option<&str>) {
        if challenge_id == Some(CHALLENGE_PK_THERAPEUTIC_WINDOW) {
            self.bioavailability = CHALLENGE_PK_F;
            self.ke = CHALLENGE_PK_KE;
            self.t_max_target = CHALLENGE_PK_TMAX_TARGET;
            self.volume_l = CHALLENGE_PK_VOLUME_L;
            self.c_mec = CHALLENGE_PK_C_MEC;
            self.c_mtc = CHALLENGE_PK_C_MTC;
        }
    }
}

/// Summary observables computed over the `481`-point `[0.0, 48.0]` h dosing trajectory.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PkTrajectorySummary {
    /// Absorption rate constant `k_a` (1/h) recovered via `solve_ka`.
    pub ka: f64,
    /// Analytical single-dose `T_max` (h) verified via `t_max(ka, ke)`.
    pub verified_t_max: f64,
    /// Elimination half-life `t_{1/2}` (h).
    pub elimination_half_life: f64,
    /// Maximum toxic overshoot `E_tox = max_k max(0, C(t_k) - C_MTC)` across `[0, 48]` h.
    pub e_tox: f64,
    /// Fraction `phi_sub` of `[12, 48]` h samples strictly below `C_MEC`.
    pub phi_sub: f64,
    /// Peak plasma concentration `C_max` (mg/L) across `[0, 48]` h and its timestamp (h).
    pub peak_conc: (f64, f64),
    /// Trough plasma concentration `C_min` (mg/L) across `[12, 48]` h and its timestamp (h).
    pub steady_trough: (f64, f64),
    /// Score `S_pk` in `0.0..=100.0`.
    pub score: f64,
    /// Whether `E_tox == 0.0` and `phi_sub == 0.0`.
    pub passed: bool,
}

/// Preallocated zero-heap-allocation evaluator for multi-dose `SuperpositionModel<BatemanModel>`.
pub struct PkWorkspace {
    /// Sampled plasma concentrations `C(t_k)` on `[0.0, 48.0]` h (`dt = 0.1` h).
    pub concentrations: [f64; PK_SAMPLE_COUNT],
    /// Preallocated multi-dose superposition model for maintenance doses.
    pub superposition: SuperpositionModel<BatemanModel>,
    /// Single-dose Bateman model used when an initial loading dose is enabled at `t = 0`.
    pub loading_model: BatemanModel,
    /// Latest trajectory summary metrics.
    pub summary: PkTrajectorySummary,
}

impl Default for PkWorkspace {
    fn default() -> Self {
        let default_pk = PKParameters::new(0.85, 100.0, 2.0, 0.12, 25.0)
            .unwrap_or_else(|_| unreachable!("valid default PK constants"));
        Self {
            concentrations: [0.0; PK_SAMPLE_COUNT],
            superposition: SuperpositionModel::new(
                BatemanModel::new(default_pk),
                Vec::with_capacity(MAX_PREALLOCATED_DOSES),
            ),
            loading_model: BatemanModel::new(default_pk),
            summary: PkTrajectorySummary::default(),
        }
    }
}

impl PkWorkspace {
    /// Evaluates `solve_ka`, `BatemanModel`, and `SuperpositionModel` across `481` points
    /// with zero dynamic heap allocations.
    pub fn evaluate(&mut self, state: &PkRegimenState) -> PkTrajectorySummary {
        let initial_guess = (2.0 / state.t_max_target).max(state.ke * 2.5);
        let ka = solve_ka(state.t_max_target, state.ke, initial_guess, 100, 1e-10)
            .unwrap_or(state.ke * 3.0);
        let maint_params = PKParameters::new(
            state.bioavailability,
            state.dose_mg.max(0.0),
            ka,
            state.ke,
            state.volume_l,
        )
        .unwrap_or(self.superposition.base_model.params);
        let first_dose = if state.use_loading_dose {
            state.loading_dose_mg.max(0.0)
        } else {
            state.dose_mg.max(0.0)
        };
        let load_params = maint_params.with_dose(first_dose).unwrap_or(maint_params);

        self.loading_model = BatemanModel::new(load_params);
        self.superposition.base_model = BatemanModel::new(maint_params);
        self.superposition.dose_times.clear();

        let step_h = state.interval_hours.max(1.0);
        let mut dose_idx = 1_usize;
        while (dose_idx as f64) * step_h <= PK_HORIZON_HOURS + 1e-9
            && self.superposition.dose_times.len() < self.superposition.dose_times.capacity()
        {
            self.superposition
                .dose_times
                .push((dose_idx as f64) * step_h);
            dose_idx += 1;
        }

        let summary = self.sample_and_score(state, ka);
        self.summary = summary;
        summary
    }

    fn sample_and_score(&mut self, state: &PkRegimenState, ka: f64) -> PkTrajectorySummary {
        let dt = PK_HORIZON_HOURS / (PK_SAMPLE_COUNT - 1) as f64;
        let mut e_tox = 0.0_f64;
        let mut sub_count = 0_usize;
        let steady_total = PK_SAMPLE_COUNT - PK_WINDOW_START_INDEX;
        let mut peak = (0.0_f64, 0.0_f64);
        let mut trough = (12.0_f64, f64::INFINITY);

        for k in 0..PK_SAMPLE_COUNT {
            let t = k as f64 * dt;
            let conc = self.loading_model.concentration(t) + self.superposition.concentration(t);
            self.concentrations[k] = conc;
            let overshoot = (conc - state.c_mtc).max(0.0);
            if overshoot > e_tox {
                e_tox = overshoot;
            }
            if conc > peak.1 {
                peak = (t, conc);
            }
            if k >= PK_WINDOW_START_INDEX {
                if conc < state.c_mec {
                    sub_count += 1;
                }
                if conc < trough.1 {
                    trough = (t, conc);
                }
            }
        }

        let phi_sub = sub_count as f64 / steady_total as f64;
        let raw_score = 100.0 * (1.0 - phi_sub - 2.0 * e_tox / state.c_mtc.max(1e-6));
        let score = raw_score.clamp(0.0, 100.0);
        let passed = e_tox == 0.0 && phi_sub == 0.0;

        PkTrajectorySummary {
            ka,
            verified_t_max: t_max(ka, state.ke),
            elimination_half_life: half_life(state.ke),
            e_tox,
            phi_sub,
            peak_conc: peak,
            steady_trough: trough,
            score,
            passed,
        }
    }

    /// Updates automated `CalloutManager` overlays for `C_MEC`, `C_MTC`, peak `C_max`, and trough `C_min`.
    pub fn sync_callouts(&self, state: &PkRegimenState, callouts: &mut CalloutManager) {
        let s = &self.summary;
        callouts.set_auto_callout(
            "pk_cmec",
            [PK_HORIZON_HOURS * 0.82, state.c_mec],
            format!("C_MEC = {:.1} mg/L", state.c_mec),
            [0.0_f32, 18.0_f32],
        );
        callouts.set_auto_callout(
            "pk_cmtc",
            [PK_HORIZON_HOURS * 0.82, state.c_mtc],
            format!("C_MTC = {:.1} mg/L", state.c_mtc),
            [0.0_f32, -18.0_f32],
        );
        callouts.set_auto_callout(
            "pk_cmax",
            [s.peak_conc.0, s.peak_conc.1],
            format!("Peak {:.2} mg/L @ {:.1}h", s.peak_conc.1, s.peak_conc.0),
            [24.0_f32, -22.0_f32],
        );
        callouts.set_auto_callout(
            "pk_cmin",
            [s.steady_trough.0, s.steady_trough.1],
            format!(
                "12–48h Trough {:.2} mg/L @ {:.1}h",
                s.steady_trough.1, s.steady_trough.0
            ),
            [26.0_f32, 22.0_f32],
        );
    }

    /// Evaluates the active `pk_therapeutic_window` challenge contract.
    pub fn evaluate_challenge(&self) -> ChallengeEvaluation {
        let s = &self.summary;
        let combined_residual = s.e_tox + s.phi_sub;
        ChallengeEvaluation::new(
            s.score,
            s.passed,
            "E_tox + φ_sub",
            combined_residual,
            format!(
                "E_tox = {:.3} mg/L (need 0.000), φ_sub(12..48h) = {:.1}% (need 0.0%)",
                s.e_tox,
                s.phi_sub * 100.0
            ),
        )
    }
}

fn generate_pk_challenge_baseline() -> String {
    serde_json::to_string(&PkRegimenState {
        dose_mg: 100.0,
        use_loading_dose: false,
        loading_dose_mg: 220.0,
        interval_hours: 12.0,
        bioavailability: CHALLENGE_PK_F,
        ke: CHALLENGE_PK_KE,
        t_max_target: CHALLENGE_PK_TMAX_TARGET,
        volume_l: CHALLENGE_PK_VOLUME_L,
        c_mec: CHALLENGE_PK_C_MEC,
        c_mtc: CHALLENGE_PK_C_MTC,
    })
    .unwrap_or_default()
}

fn clamp_pk_challenge_state(raw_json: &str) -> Result<String, String> {
    let mut state: PkRegimenState =
        serde_json::from_str(raw_json).map_err(|e| format!("Invalid PK challenge JSON: {e}"))?;
    state.enforce_challenge_locks(Some(CHALLENGE_PK_THERAPEUTIC_WINDOW));
    state.dose_mg = state.dose_mg.clamp(0.0, 1000.0);
    state.loading_dose_mg = state.loading_dose_mg.clamp(0.0, 1500.0);
    state.interval_hours = state.interval_hours.clamp(1.0, 24.0);
    state.validate()?;
    serde_json::to_string(&state).map_err(|e| format!("PK serialize error: {e}"))
}

inventory::submit! {
    ChallengeMetadata {
        id: CHALLENGE_PK_THERAPEUTIC_WINDOW,
        title: "Multi-Dose Therapeutic Window",
        description: "Choose maintenance dose D, interval Δt, and optional loading dose D_load so plasma C(t) never exceeds C_MTC = 12 mg/L on [0, 48] h and never drops below C_MEC = 4 mg/L on [12, 48] h.",
        category: "Pharmacokinetics",
        icon: "💊",
        domain: "clinical_trials",
        tool_name: "PharmacokineticsTool",
        locked_inputs: "F = 0.85, k_e = 0.12 h⁻¹, T_max = 1.5 h (solve_ka), V = 25 L, [C_MEC, C_MTC] = [4, 12] mg/L",
        tunable_budget: "D ∈ [0, 1000] mg, Δt ∈ [1, 24] h, optional D_load ∈ [0, 1500] mg",
        target_summary: "E_tox = 0.0 mg/L on [0, 48] h and φ_sub = 0.0 on [12, 48] h",
        baseline_state_generator: generate_pk_challenge_baseline,
        clamp_locked_state: clamp_pk_challenge_state,
    }
}

// [cite:clinical_trials_statistics]
