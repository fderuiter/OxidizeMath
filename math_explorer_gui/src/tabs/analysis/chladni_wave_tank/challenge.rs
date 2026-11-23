#![cfg_attr(any(), verified(opt_out = "gui_tool"))]
// [cite:advanced_linear_algebra]

//! Playable 2D Chladni plate antinodal resonance inverse challenge.

use math_explorer::pure_math::analysis::pde::wave_2d::{BoundaryMode2D, Wave2DMedium};

use super::{ChladniWaveTankTool, ClampLayout, FREQ_MAX, FREQ_MIN, TankPreset};
use crate::framework::{ChallengeEvaluation, ChallengeMetadata, InteractiveTool};

/// Canonical challenge identifier for 2D Chladni antinodal resonance lock-in.
pub(crate) const CHALLENGE_CHLADNI_RESONANCE: &str = "chladni_antinodal_resonance";
/// Locked Kirchhoff-Love plate stiffness $D$ for `chladni_antinodal_resonance`.
pub(crate) const CHALLENGE_CHLADNI_STIFFNESS: f64 = 0.0004;
/// Locked damping rate $\gamma$ for `chladni_antinodal_resonance`.
pub(crate) const CHALLENGE_CHLADNI_DAMPING: f64 = 0.02;
/// Unsolved off-resonant baseline frequency (Hz).
pub(crate) const CHALLENGE_CHLADNI_BASELINE_FREQ: f64 = 0.440;
/// Unsolved nodal-line driver X coordinate.
pub(crate) const CHALLENGE_CHLADNI_BASELINE_X: f64 = 0.22;
/// Unsolved nodal-line driver Y coordinate (`y = 0.50` nodal dead zone).
pub(crate) const CHALLENGE_CHLADNI_BASELINE_Y: f64 = 0.50;
/// Maximum frequency detuning $|\Delta f|$ (Hz) to pass resonance lock.
pub(crate) const CHALLENGE_CHLADNI_TOL_HZ: f64 = 0.005;
/// Minimum normalized spatial antinode coupling $|\Phi_{\nu}(x_d, y_d)|$ to pass.
pub(crate) const CHALLENGE_CHLADNI_MIN_COUPLING: f64 = 0.55;

impl ChladniWaveTankTool {
    pub(crate) fn enforce_chladni_challenge_locks(&mut self) {
        if self.active_challenge == Some(CHALLENGE_CHLADNI_RESONANCE) {
            self.state.preset = TankPreset::SquareChladniPlate;
            self.state.medium = Wave2DMedium::BiharmonicPlate;
            self.state.boundary = BoundaryMode2D::Neumann;
            self.state.clamp_layout = ClampLayout::CenterPin;
            self.state.wave_speed = 0.0;
            self.state.plate_stiffness = CHALLENGE_CHLADNI_STIFFNESS;
            self.state.damping = CHALLENGE_CHLADNI_DAMPING;
            self.state.barrier_enabled = false;
            self.state.lens_enabled = false;
            self.state.line_source = false;
            self.state.driver_enabled = true;
            self.state.custom_clamps.clear();
            self.state.sculpt_rle.clear();
            self.sculpt_mask.fill(0);
            self.state.driver_frequency = if self.state.driver_frequency.is_finite() {
                self.state.driver_frequency.clamp(FREQ_MIN, FREQ_MAX)
            } else {
                CHALLENGE_CHLADNI_BASELINE_FREQ
            };
        }
    }

    pub(crate) fn set_chladni_challenge(
        &mut self,
        challenge_id: Option<&str>,
    ) -> Result<(), String> {
        match challenge_id {
            None => {
                self.active_challenge = None;
                Ok(())
            }
            Some(CHALLENGE_CHLADNI_RESONANCE) => {
                self.active_challenge = Some(CHALLENGE_CHLADNI_RESONANCE);
                self.enforce_chladni_challenge_locks();
                self.rebuild_domain_and_warmup(0);
                Ok(())
            }
            Some(other) => Err(format!(
                "Unsupported challenge '{other}' for ChladniWaveTankTool"
            )),
        }
    }

    #[must_use]
    pub(crate) fn eval_chladni_challenge(&self) -> Option<ChallengeEvaluation> {
        self.active_challenge?;
        let (nearest, coupling) = self.nearest_eigenmode()?;
        let detuning = (self.state.driver_frequency - nearest.frequency_hz).abs();
        let freq_ok = detuning <= CHALLENGE_CHLADNI_TOL_HZ;
        let antinode_ok = coupling >= CHALLENGE_CHLADNI_MIN_COUPLING;
        let passed = freq_ok && antinode_ok;

        let freq_score = (1.0 - (detuning / 0.12).min(1.0)) * 60.0;
        let coup_score = (coupling / CHALLENGE_CHLADNI_MIN_COUPLING).min(1.0) * 40.0;
        let score = if passed {
            100.0
        } else {
            (freq_score + coup_score).clamp(0.0, 99.0)
        };
        let residual = (detuning / CHALLENGE_CHLADNI_TOL_HZ).max(0.0)
            + (CHALLENGE_CHLADNI_MIN_COUPLING - coupling).max(0.0);

        Some(ChallengeEvaluation::new(
            score,
            passed,
            "|Δf|/5mHz + antinode deficit",
            residual,
            format!(
                "Nearest mode ({}, {}) = {:.3} Hz (|Δf| = {:.4} Hz, need ≤ 0.005 Hz), |Φ(x_d, y_d)| = {:.2} (need ≥ 0.55)",
                nearest.m, nearest.n, nearest.frequency_hz, detuning, coupling
            ),
        ))
    }
}

fn generate_chladni_challenge_baseline() -> String {
    let mut tool = ChladniWaveTankTool::default();
    let _ = tool.set_chladni_challenge(Some(CHALLENGE_CHLADNI_RESONANCE));
    tool.state.driver_frequency = CHALLENGE_CHLADNI_BASELINE_FREQ;
    tool.state.driver_x = CHALLENGE_CHLADNI_BASELINE_X;
    tool.state.driver_y = CHALLENGE_CHLADNI_BASELINE_Y;
    tool.rebuild_domain_and_warmup(0);
    tool.save_state().unwrap_or_default()
}

fn clamp_chladni_challenge_state(raw_json: &str) -> Result<String, String> {
    let mut tool = ChladniWaveTankTool::default();
    tool.state_bounds().validate_json(raw_json)?;
    tool.try_load_state(raw_json)?;
    tool.set_chladni_challenge(Some(CHALLENGE_CHLADNI_RESONANCE))?;
    tool.save_state()
        .ok_or_else(|| "Failed to serialize clamped ChladniWaveTankTool challenge state".into())
}

inventory::submit! {
    ChallengeMetadata {
        id: CHALLENGE_CHLADNI_RESONANCE,
        title: "Chladni Plate Antinodal Resonance Lock",
        description: "Rescue a center-pinned Kirchhoff-Love Chladni plate from an off-resonant y = 0.50 nodal dead zone by tuning the harmonic driver within 0.005 Hz of a surviving plate eigenfrequency and positioning (x_d, y_d) on a high-amplitude antinode.",
        category: "Modal Resonance & PDEs",
        icon: "🔔",
        domain: "analysis",
        tool_name: "ChladniWaveTankTool",
        locked_inputs: "SquareChladniPlate, BiharmonicPlate, Neumann perimeter, CenterPin, D = 0.0004, γ = 0.02",
        tunable_budget: "driver_frequency (0.001 Hz precision), driver (x_d, y_d) ∈ [0, 1]², driver_amplitude",
        target_summary: "|f - f_{m,n}| ≤ 0.005 Hz and antinodal coupling |Φ_{m,n}(x_d, y_d)| ≥ 0.55",
        baseline_state_generator: generate_chladni_challenge_baseline,
        clamp_locked_state: clamp_chladni_challenge_state,
    }
}
