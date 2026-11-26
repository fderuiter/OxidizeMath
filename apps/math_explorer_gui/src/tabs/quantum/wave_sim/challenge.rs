// [cite:quantum_mechanics]
#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Federated inverse challenges (`quantum_barrier_50_split` and `quantum_fabry_perot_resonance`),
//! analytical transmission evaluators, and `CalloutManager` sync for [`super::WaveSimulatorTool`].

use eframe::egui;

use super::atlas::{compute_inline_steady_transmission, QuantumTunnelingCartographer};
use super::challenges::ActiveWaveChallenge;
use super::presets::{
    wave_parameter_constraints, WavePotentialPreset, DEFAULT_DT, DEFAULT_K0, DEFAULT_SIGMA,
    DEFAULT_STEPS_PER_FRAME, DEFAULT_WIDTH, DEFAULT_X0, DT_MAX, DT_RECOVERY_MIN, GRID_POINTS,
    K0_MAX, K0_MIN, MAX_STEPS_PER_FRAME, POTENTIAL_DISPLAY_SCALE, V0_MAX, V0_MIN, WIDTH_MAX,
    WIDTH_MIN,
};
use super::runner::WaveData;
use super::WaveSimulatorTool;
use crate::async_sim::atlas::{AtlasTeleportEvent, RegimeCartographer};
use crate::async_sim::SimCommand;
use crate::framework::{ChallengeEvaluation, ChallengeMetadata, InteractiveTool};

/// Canonical challenge ID for the 50/50 sub-barrier quantum tunneling split puzzle.
pub const CHALLENGE_QUANTUM_BARRIER_50_SPLIT: &str = "quantum_barrier_50_split";
/// Unsolved initial barrier height $V_0 = 15.0$ ($T_{\text{latched}} \approx 32\%$).
pub const CHALLENGE_BASELINE_V0: f64 = 15.0;
/// Target pre-wall transmitted probability $T = 0.50$.
pub const TARGET_TRANSMISSION: f64 = 0.50;
/// Allowed $\pm 2\%$ tolerance band around $T = 0.50$.
pub const TRANSMISSION_TOLERANCE: f64 = 0.02;

/// Canonical challenge identifier for sub-barrier Fabry-Pérot resonance tuning.
pub const CHALLENGE_FABRY_PEROT_RESONANCE: &str = "quantum_fabry_perot_resonance";
/// Locked barrier height $V_0$ for `quantum_fabry_perot_resonance`.
pub const CHALLENGE_FP_V0: f64 = 20.0;
/// Locked double-barrier characteristic width $w$ for `quantum_fabry_perot_resonance`.
pub const CHALLENGE_FP_WIDTH: f64 = 1.0;
/// Locked initial packet center $x_0$ for `quantum_fabry_perot_resonance`.
pub const CHALLENGE_FP_X0: f64 = -2.5;
/// Default packet spread $\sigma$ for `quantum_fabry_perot_resonance`.
pub const CHALLENGE_FP_SIGMA: f64 = 0.55;
/// Unsolved off-resonant baseline wave number $k_0$ for `quantum_fabry_perot_resonance`.
pub const CHALLENGE_FP_BASELINE_K0: f64 = 3.80;
/// Stationary transmission pass threshold $T(k_0)$ for sub-barrier resonance lock.
pub const CHALLENGE_FP_TARGET_T: f64 = 0.90;

impl WaveSimulatorTool {
    pub(super) fn eval_fabry_perot_challenge(&self) -> ChallengeEvaluation {
        let (t_stat, t_wp) = self.stationary_and_wavepacket_transmission();
        let e0 = 0.5 * self.k0 * self.k0;
        let sub_barrier = e0 < CHALLENGE_FP_V0 - 0.25;
        let passed = sub_barrier && t_stat >= CHALLENGE_FP_TARGET_T;
        let score = if sub_barrier {
            (t_stat / CHALLENGE_FP_TARGET_T * 100.0).clamp(0.0, 100.0)
        } else {
            (25.0 * (1.0 - (e0 - CHALLENGE_FP_V0) / 20.0)).clamp(0.0, 25.0)
        };
        let residual = if sub_barrier {
            (CHALLENGE_FP_TARGET_T - t_stat).max(0.0)
        } else {
            1.0 + (e0 - CHALLENGE_FP_V0).max(0.0)
        };
        ChallengeEvaluation::new(
            score,
            passed,
            "1 - T(k₀)",
            residual,
            format!(
                "T(k₀) = {:.1}% (need ≥ 90.0%), ⟨T⟩_σ = {:.1}%, E₀ = {:.2} < V₀ = {:.1}",
                t_stat * 100.0,
                t_wp * 100.0,
                e0,
                CHALLENGE_FP_V0
            ),
        )
    }

    /// Evaluates analytical stationary $T(k_0)$ and 12-point Gauss-Hermite wavepacket $\langle T \rangle_\sigma$.
    #[must_use]
    pub fn stationary_and_wavepacket_transmission(&self) -> (f64, f64) {
        let params = wave_parameter_constraints();
        let cart = QuantumTunnelingCartographer::new(
            &params,
            ("k0", "v0"),
            self.preset,
            self.k0,
            self.v0,
            self.width,
            self.sigma,
            &self.last_valid_expr,
            &self.sparse_potential,
        );
        let cell = cart.evaluate_cell(self.k0, self.v0);
        (cell.primary, cell.secondary)
    }

    /// Applies an [`AtlasTeleportEvent`] respecting locked challenge inputs and restarts the wavepacket.
    pub fn apply_atlas_teleport(&mut self, ev: AtlasTeleportEvent, ctx: Option<&egui::Context>) {
        for (key, val) in [(ev.x_key, ev.x_val), (ev.y_key, ev.y_val)] {
            match key {
                "k0" if !self.locks_k0() => self.k0 = val.clamp(K0_MIN, K0_MAX),
                "v0" if !self.locks_barrier_geometry() => self.v0 = val.clamp(V0_MIN, V0_MAX),
                "width" if !self.locks_barrier_geometry() => {
                    self.width = val.clamp(WIDTH_MIN, WIDTH_MAX);
                }
                _ => {}
            }
        }
        self.enforce_challenge_locks();
        self.pending_slider_reset = false;
        self.sync_config_and_reset(ctx);
        let _ = self.worker.send_command(SimCommand::Start);
    }

    /// Synchronizes plot callouts for barrier height $V_0$, incident energy $E_0$, and transmission $T$.
    pub(crate) fn sync_wave_callouts(&mut self, wave_data: &WaveData) {
        let v0_scaled = self.v0 * POTENTIAL_DISPLAY_SCALE;
        let e0_scaled = wave_data.kinetic_energy * POTENTIAL_DISPLAY_SCALE;
        self.callout_manager.set_auto_callout(
            "wave_barrier_v0",
            [0.0, v0_scaled],
            format!("V0 = {:.2}", self.v0),
            [0.0_f32, -20.0_f32],
        );
        self.callout_manager.set_auto_callout(
            "wave_incident_e0",
            [-3.4, e0_scaled],
            format!("E0 = {:.2}", wave_data.kinetic_energy),
            [0.0_f32, -18.0_f32],
        );
        let t_label = match wave_data.latched_transmission {
            Some(latched) => format!("Latched T = {:.1}%", latched * 100.0),
            None => format!("Live T(t) = {:.1}%", wave_data.transmission * 100.0),
        };
        self.callout_manager.set_auto_callout(
            "wave_scatter_t",
            [2.5, (e0_scaled + 0.28).min(2.2)],
            t_label,
            [0.0_f32, -18.0_f32],
        );
    }

    /// Collects finite named observables for `GoalChallengeMetadata` AST predicates and telemetry.
    #[must_use]
    pub(crate) fn collect_wave_observables(&self) -> Vec<(&'static str, f64)> {
        let e0 = 0.5 * self.k0 * self.k0;
        let mut overrides = [None; GRID_POINTS];
        for &(idx, val) in &self.sparse_potential {
            let s = usize::from(idx);
            if s < GRID_POINTS && val.is_finite() {
                overrides[s] = Some(val);
            }
        }
        let compiled = math_commons::ast::CompiledExpr::compile_with_vars(
            &self.last_valid_expr,
            &["x", "v0", "width"],
        )
        .ok();
        let steady_t = compute_inline_steady_transmission(
            self.preset,
            self.k0,
            self.v0,
            self.width,
            compiled.as_ref(),
            &overrides,
        );
        let (trans, refl, bar, latched) =
            self.active_wave_data()
                .map_or((0.0, 1.0, 0.0, steady_t), |d| {
                    (
                        d.transmission,
                        d.reflection,
                        d.barrier_prob,
                        d.latched_transmission.unwrap_or(d.transmission),
                    )
                });
        vec![
            ("kinetic_energy", e0),
            ("v0", self.v0),
            ("width", self.width),
            ("subbarrier_margin", self.v0 - e0),
            ("transmission", trans),
            ("reflection", refl),
            ("barrier_prob", bar),
            ("latched_transmission", latched),
            ("steady_transmission", steady_t),
        ]
    }
}

/// Pure scorer mapping optional `latched_transmission` to a [`ChallengeEvaluation`].
#[must_use]
pub fn score_latched_transmission(
    latched: Option<f64>,
    v0: f64,
    width: f64,
) -> ChallengeEvaluation {
    let Some(t_val) = latched.filter(|v| v.is_finite()) else {
        return ChallengeEvaluation::new(
            0.0,
            false,
            "|T_latched - 50%| (%)",
            50.0,
            format!("Awaiting barrier transit — click Play or Step (V0 = {v0:.2}, w = {width:.2})"),
        );
    };
    let err = (t_val - TARGET_TRANSMISSION).abs();
    let passed = err <= TRANSMISSION_TOLERANCE;
    let score = if passed {
        100.0 - (err / TRANSMISSION_TOLERANCE) * 8.0
    } else {
        (92.0 * (1.0 - (err - TRANSMISSION_TOLERANCE) / 0.48)).clamp(0.0, 91.9)
    };
    ChallengeEvaluation::new(
        score.clamp(0.0, 100.0),
        passed,
        "|T_latched - 50%| (%)",
        err * 100.0,
        format!(
            "Latched T = {:.1}% (need 50.0% ± 2.0%, V0 = {v0:.2}, w = {width:.2})",
            t_val * 100.0
        ),
    )
}

fn generate_quantum_barrier_baseline() -> String {
    let mut tool = WaveSimulatorTool::new();
    tool.active_challenge = Some(ActiveWaveChallenge::Barrier50Split);
    tool.preset = WavePotentialPreset::RectangularBarrier;
    tool.x0 = DEFAULT_X0;
    tool.k0 = DEFAULT_K0;
    tool.sigma = DEFAULT_SIGMA;
    tool.v0 = CHALLENGE_BASELINE_V0;
    tool.width = DEFAULT_WIDTH;
    tool.dt = DEFAULT_DT;
    tool.steps_per_frame = DEFAULT_STEPS_PER_FRAME;
    tool.sparse_potential.clear();
    tool.save_state().unwrap_or_default()
}

fn clamp_quantum_barrier_state(raw_json: &str) -> Result<String, String> {
    let mut val: serde_json::Value = serde_json::from_str(raw_json)
        .map_err(|e| format!("Invalid WaveSimulator challenge JSON: {e}"))?;
    let obj = val
        .as_object_mut()
        .ok_or_else(|| "WaveSimulator challenge state must be a JSON object".to_string())?;
    obj.remove("potential_type");
    obj.remove("sparse_potential");
    obj.insert(
        "preset".to_string(),
        serde_json::json!("RectangularBarrier"),
    );
    obj.insert("x0".to_string(), serde_json::json!(DEFAULT_X0));
    obj.insert("k0".to_string(), serde_json::json!(DEFAULT_K0));
    obj.insert("sigma".to_string(), serde_json::json!(DEFAULT_SIGMA));
    clamp_json_f64(obj, "v0", V0_MIN, V0_MAX, CHALLENGE_BASELINE_V0);
    clamp_json_f64(obj, "width", WIDTH_MIN, WIDTH_MAX, DEFAULT_WIDTH);
    clamp_json_f64(obj, "dt", DT_RECOVERY_MIN, DT_MAX, DEFAULT_DT);
    if let Some(spf) = obj
        .get("steps_per_frame")
        .and_then(serde_json::Value::as_u64)
    {
        let clamped = (spf as usize).clamp(1, MAX_STEPS_PER_FRAME);
        obj.insert("steps_per_frame".to_string(), serde_json::json!(clamped));
    }

    let sanitized = serde_json::to_string(&val).map_err(|e| e.to_string())?;
    let mut tool = WaveSimulatorTool::new();
    tool.apply_wave_challenge(Some(CHALLENGE_QUANTUM_BARRIER_50_SPLIT))?;
    tool.try_load_state(&sanitized)?;
    tool.save_state()
        .ok_or_else(|| "Failed to serialize clamped WaveSimulator state".to_string())
}

pub(super) fn clamp_json_f64(
    obj: &mut serde_json::Map<String, serde_json::Value>,
    key: &str,
    min: f64,
    max: f64,
    fallback: f64,
) {
    if let Some(v) = obj.get(key).and_then(serde_json::Value::as_f64) {
        let safe = if v.is_finite() {
            v.clamp(min, max)
        } else {
            fallback
        };
        obj.insert(key.to_string(), serde_json::json!(safe));
    }
}

fn generate_fp_challenge_baseline() -> String {
    let mut tool = WaveSimulatorTool::new();
    let _ = tool.set_wave_challenge(Some(CHALLENGE_FABRY_PEROT_RESONANCE));
    tool.k0 = CHALLENGE_FP_BASELINE_K0;
    tool.sigma = CHALLENGE_FP_SIGMA;
    tool.sync_config_and_reset(None);
    tool.save_state().unwrap_or_default()
}

fn clamp_fp_challenge_state(raw_json: &str) -> Result<String, String> {
    let mut tool = WaveSimulatorTool::new();
    tool.state_bounds().validate_json(raw_json)?;
    tool.try_load_state(raw_json)?;
    tool.set_wave_challenge(Some(CHALLENGE_FABRY_PEROT_RESONANCE))?;
    tool.save_state()
        .ok_or_else(|| "Failed to serialize clamped WaveSimulator challenge state".to_string())
}

inventory::submit! {
    ChallengeMetadata {
        id: CHALLENGE_QUANTUM_BARRIER_50_SPLIT,
        title: "50/50 Sub-Barrier Quantum Tunneling Split",
        description: "Tune barrier height V0 and width w so a locked Gaussian wavepacket (E0 = 12.5) splits into 50% ± 2% pre-wall transmitted probability.",
        category: "Wave Mechanics",
        icon: "🌊",
        domain: "quantum",
        tool_name: "WaveSimulator",
        locked_inputs: "RectangularBarrier, x0 = -2.0, k0 = 5.0 (E0 = 12.5), σ = 0.5, sculpting disabled",
        tunable_budget: "V0 ∈ [0, 40], width w ∈ [0.1, 2.0], Δt ∈ [0.001, 0.1]",
        target_summary: "Latched pre-wall transmission T_latched ∈ [48.0%, 52.0%]",
        baseline_state_generator: generate_quantum_barrier_baseline,
        clamp_locked_state: clamp_quantum_barrier_state,
    }
}

inventory::submit! {
    ChallengeMetadata {
        id: CHALLENGE_FABRY_PEROT_RESONANCE,
        title: "Sub-Barrier Fabry-Pérot Resonance",
        description: "Tune incident wavepacket momentum k₀ below the barrier crest (E₀ = k₀²/2 < V₀ = 20) to unlock >90% stationary Fabry-Pérot resonant transmission across the symmetric double barrier.",
        category: "Quantum Tunneling",
        icon: "🔬",
        domain: "quantum",
        tool_name: "WaveSimulator",
        locked_inputs: "DoubleBarrier, V₀ = 20.0, width w = 1.0, x₀ = -2.5 (no potential edits)",
        tunable_budget: "k₀ ∈ [1.0, 6.28] (E₀ < V₀ = 20.0), σ ∈ [0.25, 1.0]",
        target_summary: "Sub-barrier E₀ < V₀ and stationary transfer-matrix T(k₀) ≥ 90%",
        baseline_state_generator: generate_fp_challenge_baseline,
        clamp_locked_state: clamp_fp_challenge_state,
    }
}
