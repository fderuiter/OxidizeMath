//! Federated `quantum_barrier_50_split` inverse challenge and `CalloutManager` sync for [`WaveSimulatorTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::presets::{
    WavePotentialPreset, DEFAULT_DT, DEFAULT_K0, DEFAULT_SIGMA, DEFAULT_STEPS_PER_FRAME,
    DEFAULT_V0, DEFAULT_WIDTH, DEFAULT_X0, DT_MAX, DT_RECOVERY_MIN, MAX_STEPS_PER_FRAME,
    POTENTIAL_DISPLAY_SCALE, V0_MAX, V0_MIN, WIDTH_MAX, WIDTH_MIN,
};
use super::runner::WaveData;
use super::WaveSimulatorTool;
use crate::framework::{ChallengeEvaluation, ChallengeMetadata, InteractiveTool};
use eframe::egui;

/// Canonical challenge ID for the 50/50 sub-barrier quantum tunneling split puzzle.
pub const CHALLENGE_QUANTUM_BARRIER_50_SPLIT: &str = "quantum_barrier_50_split";

/// Unsolved initial barrier height $V_0 = 15.0$ ($T_{\text{latched}} \approx 32\%$).
pub const CHALLENGE_BASELINE_V0: f64 = 15.0;
/// Target pre-wall transmitted probability $T = 0.50$.
pub const TARGET_TRANSMISSION: f64 = 0.50;
/// Allowed $\pm 2\%$ tolerance band around $T = 0.50$.
pub const TRANSMISSION_TOLERANCE: f64 = 0.02;

impl WaveSimulatorTool {
    /// Returns `true` when `quantum_barrier_50_split` locks incident packet and potential family.
    #[must_use]
    pub(crate) fn is_challenge_locked(&self) -> bool {
        self.active_challenge == Some(CHALLENGE_QUANTUM_BARRIER_50_SPLIT)
    }

    /// Restores locked boundary parameters while preserving tunable `v0`, `width`, `dt`, and speed.
    pub(crate) fn enforce_challenge_locks(&mut self) {
        if self.is_challenge_locked() {
            self.preset = WavePotentialPreset::RectangularBarrier;
            self.x0 = DEFAULT_X0;
            self.k0 = DEFAULT_K0;
            self.sigma = DEFAULT_SIGMA;
            self.custom_potential = None;
            self.sparse_potential.clear();
        }
    }

    /// Configures the playable 50% sub-barrier tunneling split challenge (`E0 = 12.5`, `V0 = 11.35`, `w = 0.5`).
    pub fn apply_tunneling_split_challenge(&mut self, ctx: Option<&egui::Context>) {
        self.preset = WavePotentialPreset::RectangularBarrier;
        self.x0 = DEFAULT_X0;
        self.k0 = DEFAULT_K0;
        self.sigma = DEFAULT_SIGMA;
        self.v0 = DEFAULT_V0;
        self.width = DEFAULT_WIDTH;
        self.dt = DEFAULT_DT;
        self.custom_potential = None;
        self.sparse_potential.clear();
        self.sync_config_and_reset(ctx);
    }

    /// Activates or clears `quantum_barrier_50_split` on [`WaveSimulatorTool`].
    pub(crate) fn set_wave_challenge(&mut self, challenge_id: Option<&str>) -> Result<(), String> {
        match challenge_id {
            None => {
                self.active_challenge = None;
                Ok(())
            }
            Some(CHALLENGE_QUANTUM_BARRIER_50_SPLIT) => {
                self.active_challenge = Some(CHALLENGE_QUANTUM_BARRIER_50_SPLIT);
                self.enforce_challenge_locks();
                self.sync_config_and_reset(None);
                Ok(())
            }
            Some(other) => Err(format!(
                "Unsupported challenge '{other}' for WaveSimulatorTool"
            )),
        }
    }

    /// Evaluates `WaveData::latched_transmission` against $50\% \pm 2\%$ with zero heap allocations
    /// on the underlying `WaveRunner` hot path.
    #[must_use]
    pub(crate) fn eval_wave_challenge(&self) -> Option<ChallengeEvaluation> {
        self.active_challenge?;
        let latched = self.active_wave_data().and_then(|d| d.latched_transmission);
        Some(score_latched_transmission(latched, self.v0, self.width))
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
    tool.active_challenge = Some(CHALLENGE_QUANTUM_BARRIER_50_SPLIT);
    tool.preset = WavePotentialPreset::RectangularBarrier;
    tool.x0 = DEFAULT_X0;
    tool.k0 = DEFAULT_K0;
    tool.sigma = DEFAULT_SIGMA;
    tool.v0 = CHALLENGE_BASELINE_V0;
    tool.width = DEFAULT_WIDTH;
    tool.dt = DEFAULT_DT;
    tool.steps_per_frame = DEFAULT_STEPS_PER_FRAME;
    tool.custom_potential = None;
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
    obj.remove("custom_potential");
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
    tool.set_wave_challenge(Some(CHALLENGE_QUANTUM_BARRIER_50_SPLIT))?;
    tool.try_load_state(&sanitized)?;
    tool.save_state()
        .ok_or_else(|| "Failed to serialize clamped WaveSimulator state".to_string())
}

fn clamp_json_f64(
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

// [cite:quantum_mechanics]
