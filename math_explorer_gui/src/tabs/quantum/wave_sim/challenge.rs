// [cite:quantum_mechanics]
#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Playable sub-barrier Fabry-Pérot resonance inverse challenge for [`super::WaveSimulatorTool`].

use std::hash::{Hash, Hasher};
use std::sync::Arc;

use eframe::egui;

use super::atlas::QuantumTunnelingCartographer;
use super::presets::{
    wave_parameter_constraints, WavePotentialPreset, K0_MAX, K0_MIN, SIGMA_MAX, SIGMA_MIN, V0_MAX,
    V0_MIN, WIDTH_MAX, WIDTH_MIN,
};
use super::WaveSimulatorTool;
use crate::async_sim::atlas::{AtlasTeleportEvent, RegimeCartographer};
use crate::async_sim::SimCommand;
use crate::framework::{ChallengeEvaluation, ChallengeMetadata, InteractiveTool};

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
    pub(crate) fn enforce_wave_challenge_locks(&mut self) {
        if self.active_challenge == Some(CHALLENGE_FABRY_PEROT_RESONANCE) {
            self.preset = WavePotentialPreset::DoubleBarrier;
            self.v0 = CHALLENGE_FP_V0;
            self.width = CHALLENGE_FP_WIDTH;
            self.x0 = CHALLENGE_FP_X0;
            self.sparse_potential.clear();
            self.k0 = if self.k0.is_finite() {
                self.k0.clamp(K0_MIN, K0_MAX)
            } else {
                CHALLENGE_FP_BASELINE_K0
            };
            self.sigma = if self.sigma.is_finite() {
                self.sigma.clamp(SIGMA_MIN, SIGMA_MAX)
            } else {
                CHALLENGE_FP_SIGMA
            };
        }
    }

    /// Activates or clears the `quantum_fabry_perot_resonance` inverse challenge.
    pub fn set_wave_challenge(&mut self, challenge_id: Option<&str>) -> Result<(), String> {
        match challenge_id {
            None => {
                self.active_challenge = None;
                Ok(())
            }
            Some(CHALLENGE_FABRY_PEROT_RESONANCE) => {
                self.active_challenge = Some(CHALLENGE_FABRY_PEROT_RESONANCE);
                self.enforce_wave_challenge_locks();
                self.sync_config_and_reset(None);
                Ok(())
            }
            Some(other) => Err(format!(
                "Unsupported challenge '{other}' for WaveSimulatorTool"
            )),
        }
    }

    /// Evaluates the active `quantum_fabry_perot_resonance` challenge.
    #[must_use]
    pub fn eval_wave_challenge(&self) -> Option<ChallengeEvaluation> {
        self.active_challenge?;
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
        Some(ChallengeEvaluation::new(
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
        ))
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

    /// Applies an [`AtlasTeleportEvent`] and launches/resets the wavepacket immediately.
    pub fn apply_atlas_teleport(&mut self, ev: AtlasTeleportEvent, ctx: Option<&egui::Context>) {
        for (key, val) in [(ev.x_key, ev.x_val), (ev.y_key, ev.y_val)] {
            match key {
                "k0" => self.k0 = val.clamp(K0_MIN, K0_MAX),
                "v0" => self.v0 = val.clamp(V0_MIN, V0_MAX),
                "width" => self.width = val.clamp(WIDTH_MIN, WIDTH_MAX),
                _ => {}
            }
        }
        self.enforce_wave_challenge_locks();
        self.pending_slider_reset = false;
        self.sync_config_and_reset(ctx);
        let _ = self.worker.send_command(SimCommand::Start);
    }

    pub(super) fn render_wave_atlas_panel(&mut self, ui: &mut egui::Ui) {
        if !self.atlas.is_visible() {
            return;
        }
        let ctx = ui.ctx().clone();
        let (xk, yk) = self.atlas.axis_keys();
        let get_val = |k: &str| match k {
            "k0" => self.k0,
            "v0" => self.v0,
            "width" => self.width,
            _ => 0.0,
        };
        let mut h = std::collections::hash_map::DefaultHasher::new();
        (self.preset as u8).hash(&mut h);
        self.sigma.to_bits().hash(&mut h);
        if xk != "v0" && yk != "v0" {
            self.v0.to_bits().hash(&mut h);
        }
        if xk != "width" && yk != "width" {
            self.width.to_bits().hash(&mut h);
        }
        if self.preset == WavePotentialPreset::Custom {
            self.math_input.raw_text.hash(&mut h);
        }
        for &(idx, val) in &self.sparse_potential {
            idx.hash(&mut h);
            val.to_bits().hash(&mut h);
        }
        let params = wave_parameter_constraints();
        let (preset, k0, v0, width, sigma) =
            (self.preset, self.k0, self.v0, self.width, self.sigma);
        let (expr, sparse) = (self.last_valid_expr.clone(), self.sparse_potential.clone());
        if let Some(ev) = self.atlas.show_panel(
            ui,
            "wave_sim",
            get_val(xk),
            get_val(yk),
            h.finish(),
            move |keys| {
                Arc::new(QuantumTunnelingCartographer::new(
                    &params, keys, preset, k0, v0, width, sigma, &expr, &sparse,
                ))
            },
        ) {
            self.apply_atlas_teleport(ev, Some(&ctx));
        }
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
