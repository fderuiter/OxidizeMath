#![cfg_attr(any(), verified(opt_out = "gui_tool"))]
//! Unsolved inverse `ChallengeMetadata` cards (`quantum_subbarrier_50_split` and
//! `quantum_double_barrier_resonance`), `#challenge=` state clamping, scoring, and
//! named observables for [`super::WaveSimulatorTool`].
// [cite:quantum_mechanics]

use super::atlas::compute_inline_steady_transmission;
use super::challenge::{
    clamp_json_f64, score_latched_transmission, CHALLENGE_QUANTUM_BARRIER_50_SPLIT,
};
use super::presets::{
    default_compiled_expr, WavePotentialPreset, DEFAULT_CUSTOM_EXPR, DEFAULT_DT, DEFAULT_K0,
    DEFAULT_SIGMA, DEFAULT_STEPS_PER_FRAME, DEFAULT_X0, DT_MAX, DT_RECOVERY_MIN, GRID_POINTS,
    K0_MAX, K0_MIN, V0_MAX, V0_MIN, WIDTH_MAX, WIDTH_MIN,
};
use super::runner::{WaveRunner, WaveSimConfig};
use super::WaveSimulatorTool;
use crate::framework::{ChallengeEvaluation, ChallengeMetadata};
use std::sync::{Arc, Mutex, RwLock};

/// Canonical ID for the sub-barrier 50/50 tunneling wavepacket split inverse challenge.
pub const CHALLENGE_SUBBARRIER_50_SPLIT: &str = "quantum_subbarrier_50_split";
/// Canonical ID for the sub-barrier double-barrier Fabry-Perot resonance inverse challenge.
pub const CHALLENGE_DOUBLE_BARRIER_RESONANCE: &str = "quantum_double_barrier_resonance";

/// Target pre-wall `latched_transmission` for `quantum_subbarrier_50_split`.
pub const SUBBARRIER_TARGET_TRANSMISSION: f64 = 0.50;
/// Pass tolerance `±0.03` around `0.50` for `quantum_subbarrier_50_split`.
pub const SUBBARRIER_PASS_TOLERANCE: f64 = 0.03;
/// Unsolved initial barrier height `V0` (`E0 = 12.5 < 32.0`, `T << 0.50`) for `quantum_subbarrier_50_split`.
const SUBBARRIER_BASELINE_V0: f64 = 32.0;
/// Unsolved initial barrier width `w` for `quantum_subbarrier_50_split`.
const SUBBARRIER_BASELINE_WIDTH: f64 = 0.45;
/// Unsolved initial wave number `k0` (`E0 = 12.5`) for `quantum_subbarrier_50_split`.
const SUBBARRIER_BASELINE_K0: f64 = 5.0;

/// Locked barrier height `V0 = 28.0` for `quantum_double_barrier_resonance`.
pub const DOUBLE_BARRIER_LOCKED_V0: f64 = 28.0;
/// Locked barrier width `w = 0.8` for `quantum_double_barrier_resonance`.
pub const DOUBLE_BARRIER_LOCKED_WIDTH: f64 = 0.8;
/// Unsolved off-resonance wave number `k0 = 4.2` (`E0 = 8.82 < 28.0`, `T < 0.001`).
const DOUBLE_BARRIER_BASELINE_K0: f64 = 4.2;
/// Minimum lattice `steady_transmission` required to pass `quantum_double_barrier_resonance`.
pub const DOUBLE_BARRIER_PASS_MIN_T: f64 = 0.95;

/// Active inverse challenge mode attached to [`WaveSimulatorTool`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ActiveWaveChallenge {
    /// Tune `(V0, width)` with locked `k0 = 5.0` so stepped `latched_transmission` hits `0.50 ± 0.02`.
    Barrier50Split,
    /// Tune `(k0, V0, width)` in the sub-barrier regime `E0 < V0` so `latched_transmission` hits `0.50 ± 0.03`.
    Subbarrier50Split,
    /// Tune `k0` in the sub-barrier regime `E0 < V0` on a locked `DoubleBarrier` (`V0 = 28, w = 0.8`) to hit resonance.
    DoubleBarrierResonance,
}

impl ActiveWaveChallenge {
    /// Canonical registered challenge ID.
    #[must_use]
    pub const fn id(self) -> &'static str {
        match self {
            Self::Barrier50Split => CHALLENGE_QUANTUM_BARRIER_50_SPLIT,
            Self::Subbarrier50Split => CHALLENGE_SUBBARRIER_50_SPLIT,
            Self::DoubleBarrierResonance => CHALLENGE_DOUBLE_BARRIER_RESONANCE,
        }
    }

    /// Parses a registered challenge ID.
    #[must_use]
    pub fn from_id(id: &str) -> Option<Self> {
        match id {
            CHALLENGE_QUANTUM_BARRIER_50_SPLIT => Some(Self::Barrier50Split),
            CHALLENGE_SUBBARRIER_50_SPLIT => Some(Self::Subbarrier50Split),
            CHALLENGE_DOUBLE_BARRIER_RESONANCE => Some(Self::DoubleBarrierResonance),
            _ => None,
        }
    }
}

type LatchCacheKey = (u8, u64, u64, u64, u64);

/// Cached deterministic wavepacket transit evaluator so `SimulationFramework::show_challenge_hud`
/// resolves exact pre-wall `latched_transmission` both during live worker playback and stateless hydration.
#[derive(Default)]
pub struct PreWallLatchCache {
    cached: Mutex<Option<(LatchCacheKey, f64)>>,
}

impl PreWallLatchCache {
    /// Evaluates or returns cached pre-wall `latched_transmission` for `(preset, k0, v0, width, dt)`.
    pub fn eval_latched_transmission(
        &self,
        preset: WavePotentialPreset,
        k0: f64,
        v0: f64,
        width: f64,
        dt: f64,
    ) -> f64 {
        let key = (
            preset as u8,
            k0.to_bits(),
            v0.to_bits(),
            width.to_bits(),
            dt.to_bits(),
        );
        if let Ok(guard) = self.cached.lock() {
            if let Some((k, val)) = *guard {
                if k == key {
                    return val;
                }
            }
        }
        let cfg = WaveSimConfig {
            preset,
            x0: DEFAULT_X0,
            k0,
            sigma: DEFAULT_SIGMA,
            v0,
            width,
            dt: dt.clamp(0.005, 0.05),
            steps_per_frame: DEFAULT_STEPS_PER_FRAME,
            custom_expr: default_compiled_expr(),
            sparse_potential: Vec::new(),
        };
        let mut runner = WaveRunner::new(Arc::new(RwLock::new(cfg.clone())));
        for _ in 0..320 {
            runner.step_with_dt(cfg.dt);
            if runner.wall_arrived {
                break;
            }
        }
        let latched = runner
            .latched_transmission
            .unwrap_or_else(|| runner.build_wave_data().transmission)
            .clamp(0.0, 1.0);
        if let Ok(mut guard) = self.cached.lock() {
            *guard = Some((key, latched));
        }
        latched
    }
}

fn generate_subbarrier_50_baseline() -> String {
    serde_json::json!({
        "preset": "RectangularBarrier",
        "x0": DEFAULT_X0,
        "k0": SUBBARRIER_BASELINE_K0,
        "sigma": DEFAULT_SIGMA,
        "v0": SUBBARRIER_BASELINE_V0,
        "width": SUBBARRIER_BASELINE_WIDTH,
        "dt": DEFAULT_DT,
        "steps_per_frame": DEFAULT_STEPS_PER_FRAME,
        "custom_expr": DEFAULT_CUSTOM_EXPR,
    })
    .to_string()
}

fn generate_double_barrier_baseline() -> String {
    serde_json::json!({
        "preset": "DoubleBarrier",
        "x0": DEFAULT_X0,
        "k0": DOUBLE_BARRIER_BASELINE_K0,
        "sigma": DEFAULT_SIGMA,
        "v0": DOUBLE_BARRIER_LOCKED_V0,
        "width": DOUBLE_BARRIER_LOCKED_WIDTH,
        "dt": DEFAULT_DT,
        "steps_per_frame": DEFAULT_STEPS_PER_FRAME,
        "custom_expr": DEFAULT_CUSTOM_EXPR,
        "atlas_visible": true,
        "atlas_axes": ["k0", "v0"],
    })
    .to_string()
}

fn sanitize_challenge_object(
    raw_json: &str,
    preset_name: &str,
    locked_v0_width: Option<(f64, f64)>,
) -> Result<String, String> {
    let mut val: serde_json::Value = serde_json::from_str(raw_json)
        .map_err(|e| format!("Invalid WaveSimulator challenge JSON: {e}"))?;
    let obj = val
        .as_object_mut()
        .ok_or_else(|| "WaveSimulator challenge state must be a JSON object".to_string())?;
    obj.remove("potential_type");
    obj.remove("sparse_potential");
    obj.insert("preset".into(), serde_json::json!(preset_name));
    obj.insert("x0".into(), serde_json::json!(DEFAULT_X0));
    obj.insert("sigma".into(), serde_json::json!(DEFAULT_SIGMA));
    obj.insert("custom_expr".into(), serde_json::json!(DEFAULT_CUSTOM_EXPR));
    clamp_json_f64(obj, "k0", K0_MIN, K0_MAX, SUBBARRIER_BASELINE_K0);
    clamp_json_f64(obj, "dt", DT_RECOVERY_MIN, DT_MAX, DEFAULT_DT);
    if let Some((lv0, lw)) = locked_v0_width {
        obj.insert("v0".into(), serde_json::json!(lv0));
        obj.insert("width".into(), serde_json::json!(lw));
    } else {
        clamp_json_f64(obj, "v0", V0_MIN, V0_MAX, SUBBARRIER_BASELINE_V0);
        clamp_json_f64(
            obj,
            "width",
            WIDTH_MIN,
            WIDTH_MAX,
            SUBBARRIER_BASELINE_WIDTH,
        );
    }
    serde_json::to_string(&val).map_err(|e| e.to_string())
}

fn clamp_subbarrier_50_state(raw_json: &str) -> Result<String, String> {
    sanitize_challenge_object(raw_json, "RectangularBarrier", None)
}

fn clamp_double_barrier_state(raw_json: &str) -> Result<String, String> {
    sanitize_challenge_object(
        raw_json,
        "DoubleBarrier",
        Some((DOUBLE_BARRIER_LOCKED_V0, DOUBLE_BARRIER_LOCKED_WIDTH)),
    )
}

impl WaveSimulatorTool {
    /// Returns `true` when any active inverse challenge locks preset, initial packet center/width, and sculpting.
    #[must_use]
    pub(crate) fn is_challenge_locked(&self) -> bool {
        self.active_challenge.is_some()
    }

    /// Returns `true` when the active challenge locks incident wave number `k0`.
    #[must_use]
    pub(crate) fn locks_k0(&self) -> bool {
        self.active_challenge == Some(ActiveWaveChallenge::Barrier50Split)
    }

    /// Returns `true` when the active challenge locks barrier `v0` and `width`.
    #[must_use]
    pub(crate) fn locks_barrier_geometry(&self) -> bool {
        self.active_challenge == Some(ActiveWaveChallenge::DoubleBarrierResonance)
    }

    /// Applies or clears an inverse challenge, enforcing locked inputs.
    pub(crate) fn apply_wave_challenge(
        &mut self,
        challenge_id: Option<&str>,
    ) -> Result<(), String> {
        let Some(id) = challenge_id else {
            self.active_challenge = None;
            return Ok(());
        };
        let mode = ActiveWaveChallenge::from_id(id)
            .ok_or_else(|| format!("Unsupported WaveSimulator challenge '{id}'"))?;
        self.active_challenge = Some(mode);
        self.enforce_challenge_locks();
        self.sync_config_and_reset(None);
        Ok(())
    }

    /// Restores locked boundary parameters for whichever inverse challenge is currently active.
    pub(crate) fn enforce_challenge_locks(&mut self) {
        let Some(mode) = self.active_challenge else {
            return;
        };
        self.x0 = DEFAULT_X0;
        self.sigma = DEFAULT_SIGMA;
        self.sparse_potential.clear();
        match mode {
            ActiveWaveChallenge::Barrier50Split => {
                self.preset = WavePotentialPreset::RectangularBarrier;
                self.k0 = DEFAULT_K0;
            }
            ActiveWaveChallenge::Subbarrier50Split => {
                self.preset = WavePotentialPreset::RectangularBarrier;
            }
            ActiveWaveChallenge::DoubleBarrierResonance => {
                self.preset = WavePotentialPreset::DoubleBarrier;
                self.v0 = DOUBLE_BARRIER_LOCKED_V0;
                self.width = DOUBLE_BARRIER_LOCKED_WIDTH;
            }
        }
    }

    /// Evaluates the active inverse challenge for `SimulationFramework::show_challenge_hud`.
    #[must_use]
    pub(crate) fn evaluate_wave_challenge(&self) -> Option<ChallengeEvaluation> {
        match self.active_challenge? {
            ActiveWaveChallenge::Barrier50Split => {
                let latched = self.active_wave_data().and_then(|d| d.latched_transmission);
                Some(score_latched_transmission(latched, self.v0, self.width))
            }
            ActiveWaveChallenge::Subbarrier50Split => Some(self.eval_subbarrier_50_split()),
            ActiveWaveChallenge::DoubleBarrierResonance => {
                Some(self.eval_double_barrier_resonance())
            }
        }
    }

    fn eval_subbarrier_50_split(&self) -> ChallengeEvaluation {
        let e0 = 0.5 * self.k0 * self.k0;
        let subbarrier_ok = e0 < self.v0;
        let live_latched = self.active_wave_data().and_then(|d| d.latched_transmission);
        let latched_t = live_latched.unwrap_or_else(|| {
            self.latch_cache.eval_latched_transmission(
                self.preset,
                self.k0,
                self.v0,
                self.width,
                self.dt,
            )
        });
        let err = (latched_t - SUBBARRIER_TARGET_TRANSMISSION).abs();
        if !subbarrier_ok {
            let penalty_score = (25.0 * (1.0 - err / 0.50)).clamp(0.0, 25.0);
            return ChallengeEvaluation::new(
                penalty_score,
                false,
                "|T_latch - 0.50|",
                err,
                format!(
                    "Rejected over-barrier state: E0 = {e0:.2} ≥ V0 = {:.2} (must keep E0 < V0); T_latch = {:.3}",
                    self.v0, latched_t
                ),
            );
        }
        let passed = err <= SUBBARRIER_PASS_TOLERANCE;
        let score = (100.0 * (1.0 - err / 0.50)).clamp(0.0, 100.0);
        ChallengeEvaluation::new(
            score,
            passed,
            "|T_latch - 0.50|",
            err,
            format!(
                "Sub-barrier E0 = {e0:.2} < V0 = {:.2} | Latched T = {:.3} (target 0.500 ± 0.030)",
                self.v0, latched_t
            ),
        )
    }

    fn eval_double_barrier_resonance(&self) -> ChallengeEvaluation {
        let e0 = 0.5 * self.k0 * self.k0;
        let subbarrier_ok = e0 < self.v0;
        let steady_t = compute_inline_steady_transmission(
            WavePotentialPreset::DoubleBarrier,
            self.k0,
            self.v0,
            self.width,
            None,
            &[None; GRID_POINTS],
        );
        let residual = (1.0 - steady_t).max(0.0);
        if !subbarrier_ok {
            return ChallengeEvaluation::new(
                0.0,
                false,
                "1 - T_steady",
                residual,
                format!(
                    "Rejected over-barrier state: E0 = {e0:.2} ≥ V0 = {:.2} (must keep E0 < V0)",
                    self.v0
                ),
            );
        }
        let passed = steady_t >= DOUBLE_BARRIER_PASS_MIN_T;
        let score = (steady_t * 100.0).clamp(0.0, 100.0);
        ChallengeEvaluation::new(
            score,
            passed,
            "1 - T_steady",
            residual,
            format!(
                "Sub-barrier E0 = {e0:.2} < V0 = {:.2} | Steady T = {:.2}% (target ≥ 95.0%)",
                self.v0,
                steady_t * 100.0
            ),
        )
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

inventory::submit! {
    ChallengeMetadata {
        id: CHALLENGE_SUBBARRIER_50_SPLIT,
        title: "Sub-Barrier 50/50 Beam Splitter",
        description: "Calibrate a strictly sub-barrier rectangular barrier (E0 < V0) so an incident Gaussian wavepacket splits 50% transmitted / 50% reflected before wall arrival.",
        category: "Quantum Tunneling",
        icon: "🎯",
        domain: "quantum",
        tool_name: "WaveSimulator",
        locked_inputs: "RectangularBarrier profile, x0 = -2.0, σ = 0.5, no sculpted overrides",
        tunable_budget: "Incident k0 ∈ [1, 10], Barrier Height V0 ∈ [0, 40], Barrier Width w ∈ [0.1, 2.0]",
        target_summary: "E0 < V0 and pre-wall latched_transmission ∈ [0.47, 0.53]",
        baseline_state_generator: generate_subbarrier_50_baseline,
        clamp_locked_state: clamp_subbarrier_50_state,
    }
}

inventory::submit! {
    ChallengeMetadata {
        id: CHALLENGE_DOUBLE_BARRIER_RESONANCE,
        title: "Fabry-Perot Resonant Tunneling",
        description: "Use the 2D lattice transmission atlas to tune incident wave number k0 onto a sub-barrier quasi-bound state of a symmetric double barrier (V0 = 28, w = 0.8).",
        category: "Quantum Tunneling",
        icon: "🌊",
        domain: "quantum",
        tool_name: "WaveSimulator",
        locked_inputs: "DoubleBarrier profile, V0 = 28.0, width = 0.8, x0 = -2.0, σ = 0.5",
        tunable_budget: "Incident Wave Number k0 ∈ [1, 10] (click-to-teleport on 2D Parameter Atlas)",
        target_summary: "E0 < V0 and lattice steady_transmission ≥ 0.95",
        baseline_state_generator: generate_double_barrier_baseline,
        clamp_locked_state: clamp_double_barrier_state,
    }
}
