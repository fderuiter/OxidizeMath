//! Federated `radar_cough_zero_miss` inverse challenge, 24-slot physical beam delay ring,
//! and `CalloutManager` overlays for [`RadarGatingTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::pipeline::{RadarPipeline, BEAM_DELAY_RING_LEN, SIM_DT};
use super::state::{RadarGatingState, ScattererPoint};
use super::RadarGatingTool;
use crate::framework::{ChallengeEvaluation, ChallengeMetadata, InteractiveTool};
use math_explorer::physics::medical::radar_gating::TrackingFilter;

/// Stack-resident 2-state Constant-Velocity Kalman filter matching `TrackingFilter` algebra.
#[derive(Debug, Clone, Copy)]
pub struct FastCvKalman {
    /// State vector `[amplitude_m, velocity_m_s]`.
    pub state: [f64; 2],
    /// State error covariance `[[p00, p01], [p10, p11]]`.
    pub cov: [[f64; 2]; 2],
    /// Discretization interval in seconds.
    pub dt: f64,
    /// Process noise variance $Q$.
    pub process_var: f64,
    /// Measurement noise variance $R$.
    pub meas_var: f64,
}

impl FastCvKalman {
    /// Initializes stack state directly from a verified `TrackingFilter` instance.
    pub fn from_tracking_filter(
        tf: &TrackingFilter,
        dt: f64,
        process_var: f64,
        meas_var: f64,
    ) -> Self {
        let p = tf.covariance();
        Self {
            state: [tf.amplitude(), tf.velocity()],
            cov: [[p[(0, 0)], p[(0, 1)]], [p[(1, 0)], p[(1, 1)]]],
            dt,
            process_var,
            meas_var,
        }
    }

    /// Executes the exact `ConstantVelocityModel` predict step on stack scalars.
    pub fn predict(&mut self) {
        let dt = self.dt;
        self.state[0] += dt * self.state[1];
        let (p00, p01, p10, p11) = (
            self.cov[0][0],
            self.cov[0][1],
            self.cov[1][0],
            self.cov[1][1],
        );
        self.cov[0][0] = p00 + dt * (p10 + p01) + dt * dt * p11 + self.process_var;
        self.cov[0][1] = p01 + dt * p11;
        self.cov[1][0] = p10 + dt * p11;
        self.cov[1][1] = p11 + self.process_var;
    }

    /// Executes the exact scalar position measurement update (`H = [1, 0]`) on stack scalars.
    pub fn update(&mut self, z_m: f64) {
        let s = self.cov[0][0] + self.meas_var;
        if !s.is_finite() || s.abs() <= 1.0e-18 {
            return;
        }
        let y = z_m - self.state[0];
        let (k0, k1) = (self.cov[0][0] / s, self.cov[1][0] / s);
        self.state[0] += k0 * y;
        self.state[1] += k1 * y;
        let (p00, p01, p10, p11) = (
            self.cov[0][0],
            self.cov[0][1],
            self.cov[1][0],
            self.cov[1][1],
        );
        self.cov[0][0] = (1.0 - k0) * p00;
        self.cov[0][1] = (1.0 - k0) * p01;
        self.cov[1][0] = p10 - k1 * p00;
        self.cov[1][1] = p11 - k1 * p01;
    }
}

/// Canonical challenge ID for preventing LINAC geographic misses during a 60 ms delayed cough.
pub const CHALLENGE_RADAR_COUGH_ZERO_MISS: &str = "radar_cough_zero_miss";

/// Locked clinical LINAC beam-off hardware latency (`60.0 ms`).
pub const CHALLENGE_RADAR_LATENCY_MS: f64 = 60.0;
/// Locked respiratory excursion amplitude (`6.0 mm`).
pub const CHALLENGE_RADAR_BREATH_AMP_MM: f64 = 6.0;
/// Locked respiratory period (`2.5 s`, yielding two full breath cycles per `5.0 s` window).
pub const CHALLENGE_RADAR_BREATH_PERIOD_S: f64 = 2.5;
/// Locked deterministic RNG seed for reproducible thermal IQ and patch sparkle noise.
pub const CHALLENGE_RADAR_SEED: u64 = 20261123;
/// Clinical geographic-miss ceiling (`-1.20 mm`); physical beam-on above this irradiates healthy tissue.
pub const GEOGRAPHIC_MISS_CEILING_MM: f64 = -1.20;
/// Minimum required physical LINAC beam-on duty cycle (`22.0%`).
pub const MIN_CHALLENGE_DUTY_CYCLE_PCT: f64 = 22.0;

/// Steps the fixed 24-slot physical beam-delay ring buffer with zero heap allocations.
#[must_use]
pub(crate) fn step_physical_beam_delay(
    ring: &mut [bool; BEAM_DELAY_RING_LEN],
    cursor: &mut usize,
    latency_ms: f64,
    command_active: bool,
) -> bool {
    let delay_slots =
        ((latency_ms.max(0.0) * 1.0e-3 / SIM_DT).round() as usize).min(BEAM_DELAY_RING_LEN - 1);
    ring[*cursor] = command_active;
    let delayed_idx = (*cursor + BEAM_DELAY_RING_LEN - delay_slots) % BEAM_DELAY_RING_LEN;
    let physical_active = if delay_slots == 0 {
        command_active
    } else {
        ring[delayed_idx]
    };
    *cursor = (*cursor + 1) % BEAM_DELAY_RING_LEN;
    physical_active
}

/// Checks whether a scheduled `+10 mm` cough impulse triggers between `prev_t` and `curr_t`.
#[must_use]
pub(crate) fn should_trigger_scheduled_cough(
    prev_t: f64,
    curr_t: f64,
    breath_period_s: f64,
) -> bool {
    let cycle = (2.0 * breath_period_s).max(1.0);
    let trigger = 1.84 * breath_period_s.max(0.5);
    let prev_mod = prev_t.rem_euclid(cycle);
    let curr_mod = curr_t.rem_euclid(cycle);
    (prev_mod < trigger && curr_mod >= trigger) || (curr_mod < prev_mod && curr_mod >= trigger)
}

impl RadarGatingTool {
    /// Returns `true` when `radar_cough_zero_miss` locks latency, RF geometry, and cough schedule.
    #[must_use]
    pub(crate) fn is_challenge_locked(&self) -> bool {
        self.active_challenge == Some(CHALLENGE_RADAR_COUGH_ZERO_MISS)
    }

    /// Restores locked RF, target, respiratory, and `60 ms` hardware latency parameters.
    pub(crate) fn enforce_challenge_locks_on_state(s: &mut RadarGatingState) {
        s.primary_target = ScattererPoint {
            x_m: 0.0,
            y_m: 0.700,
            rcs: 1.0,
        };
        s.secondary_target = ScattererPoint {
            x_m: 0.0,
            y_m: 0.715,
            rcs: 0.90,
        };
        s.bandwidth_ghz = 4.0;
        s.steering_deg = 0.0;
        s.num_antennas = 8;
        s.samples_per_chirp = 40;
        s.snapshots = 20;
        s.sweep_bins = 200;
        s.noise_sigma = 0.015;
        s.static_clutter_gain = 0.45;
        s.breath_amplitude_mm = CHALLENGE_RADAR_BREATH_AMP_MM;
        s.breath_period_s = CHALLENGE_RADAR_BREATH_PERIOD_S;
        s.system_latency_ms = CHALLENGE_RADAR_LATENCY_MS;
        s.rng_seed = CHALLENGE_RADAR_SEED;
    }

    /// Activates or clears `radar_cough_zero_miss` on [`RadarGatingTool`].
    pub(crate) fn set_radar_challenge(&mut self, challenge_id: Option<&str>) -> Result<(), String> {
        match challenge_id {
            None => {
                self.active_challenge = None;
                self.pipeline.scheduled_cough_enabled = false;
                self.pipeline.rebuild_all(&self.state);
                Ok(())
            }
            Some(CHALLENGE_RADAR_COUGH_ZERO_MISS) => {
                self.active_challenge = Some(CHALLENGE_RADAR_COUGH_ZERO_MISS);
                Self::enforce_challenge_locks_on_state(&mut self.state);
                self.pipeline.scheduled_cough_enabled = true;
                self.pipeline.rebuild_all(&self.state);
                Ok(())
            }
            Some(other) => Err(format!(
                "Unsupported challenge '{other}' for RadarGatingTool"
            )),
        }
    }

    /// Evaluates live `geographic_miss_frames` and physical `duty_cycle_percent()`.
    #[must_use]
    pub(crate) fn eval_radar_challenge(&self) -> Option<ChallengeEvaluation> {
        self.active_challenge?;
        Some(score_radar_cough_challenge(&self.pipeline))
    }

    /// Synchronizes `CalloutManager` markers for peak `z_true` cough excursion and `-1.20 mm` ceiling.
    pub(crate) fn sync_radar_callouts(&mut self) {
        let mut peak_pt = [0.0_f64, -6.0_f64];
        for &[t, z_mm] in &self.pipeline.true_trace_mm {
            if z_mm > peak_pt[1] {
                peak_pt = [t, z_mm];
            }
        }
        let t_right = self
            .pipeline
            .true_trace_mm
            .last()
            .map_or(5.0, |pt| pt[0] - 0.6);
        self.callout_manager.set_auto_callout(
            "radar_z_true_peak",
            peak_pt,
            format!("Peak z_true = {:.2} mm", peak_pt[1]),
            [0.0_f32, -18.0_f32],
        );
        self.callout_manager.set_auto_callout(
            "radar_miss_ceiling",
            [t_right, GEOGRAPHIC_MISS_CEILING_MM],
            format!(
                "Miss Ceiling {:.2} mm (misses: {})",
                GEOGRAPHIC_MISS_CEILING_MM, self.pipeline.geographic_miss_frames
            ),
            [0.0_f32, -16.0_f32],
        );
    }
}

/// Pure scorer evaluating zero geographic-miss frames and `>= 22%` physical beam duty cycle.
#[must_use]
pub fn score_radar_cough_challenge(pipeline: &RadarPipeline) -> ChallengeEvaluation {
    let misses = pipeline.geographic_miss_frames;
    let duty = pipeline.physical_duty_cycle_percent();
    let passed = misses == 0 && duty >= MIN_CHALLENGE_DUTY_CYCLE_PCT;
    let duty_deficit = (MIN_CHALLENGE_DUTY_CYCLE_PCT - duty).max(0.0);
    let residual = (misses as f64) + duty_deficit;
    let score = if passed {
        let bonus = ((duty - MIN_CHALLENGE_DUTY_CYCLE_PCT) / 18.0).clamp(0.0, 1.0) * 10.0;
        (90.0 + bonus).clamp(90.0, 100.0)
    } else {
        let miss_pen = (misses as f64) * 14.0;
        let duty_pen = duty_deficit * 3.5;
        (88.0 - miss_pen - duty_pen).clamp(0.0, 88.0)
    };
    ChallengeEvaluation::new(
        score,
        passed,
        "Miss frames + duty deficit",
        residual,
        format!(
            "Geographic misses = {misses} (need 0 above -1.20 mm), Duty cycle = {duty:.1}% (need >= {MIN_CHALLENGE_DUTY_CYCLE_PCT:.0}%)"
        ),
    )
}

fn generate_radar_cough_baseline() -> String {
    let mut s = RadarGatingState {
        clutter_window: 24,
        kalman_process_var: 1.0e-6,
        kalman_meas_var: 5.0e-3,
        gating_threshold_mm: -1.1,
        gating_hysteresis_mm: 0.50,
        ..RadarGatingState::default()
    };
    RadarGatingTool::enforce_challenge_locks_on_state(&mut s);
    serde_json::to_string(&s).unwrap_or_default()
}

fn clamp_radar_cough_state(raw_json: &str) -> Result<String, String> {
    let mut val: serde_json::Value = serde_json::from_str(raw_json)
        .map_err(|e| format!("Invalid RadarGatingTool challenge JSON: {e}"))?;
    let obj = val
        .as_object_mut()
        .ok_or_else(|| "RadarGatingTool challenge state must be a JSON object".to_string())?;
    let mut canonical = RadarGatingState::default();
    RadarGatingTool::enforce_challenge_locks_on_state(&mut canonical);
    obj.insert(
        "primary_target".to_string(),
        serde_json::to_value(canonical.primary_target).map_err(|e| e.to_string())?,
    );
    obj.insert(
        "secondary_target".to_string(),
        serde_json::to_value(canonical.secondary_target).map_err(|e| e.to_string())?,
    );
    obj.insert("bandwidth_ghz".to_string(), serde_json::json!(4.0));
    obj.insert("steering_deg".to_string(), serde_json::json!(0.0));
    obj.insert("num_antennas".to_string(), serde_json::json!(8));
    obj.insert("samples_per_chirp".to_string(), serde_json::json!(40));
    obj.insert("snapshots".to_string(), serde_json::json!(20));
    obj.insert("sweep_bins".to_string(), serde_json::json!(200));
    obj.insert("noise_sigma".to_string(), serde_json::json!(0.015));
    obj.insert("static_clutter_gain".to_string(), serde_json::json!(0.45));
    obj.insert(
        "breath_amplitude_mm".to_string(),
        serde_json::json!(CHALLENGE_RADAR_BREATH_AMP_MM),
    );
    obj.insert(
        "breath_period_s".to_string(),
        serde_json::json!(CHALLENGE_RADAR_BREATH_PERIOD_S),
    );
    obj.insert(
        "system_latency_ms".to_string(),
        serde_json::json!(CHALLENGE_RADAR_LATENCY_MS),
    );
    obj.insert(
        "rng_seed".to_string(),
        serde_json::json!(CHALLENGE_RADAR_SEED),
    );

    let sanitized = serde_json::to_string(&val).map_err(|e| e.to_string())?;
    let mut tool = RadarGatingTool::default();
    tool.set_radar_challenge(Some(CHALLENGE_RADAR_COUGH_ZERO_MISS))?;
    tool.try_load_state(&sanitized)?;
    tool.save_state()
        .ok_or_else(|| "Failed to serialize clamped RadarGatingTool state".to_string())
}

inventory::submit! {
    ChallengeMetadata {
        id: CHALLENGE_RADAR_COUGH_ZERO_MISS,
        title: "60 ms LINAC Cough Lag: Zero Geographic Misses",
        description: "Tune Kalman variances (Q, R) and Schmitt bounds (T, H) so a 60 ms delayed radiation beam never irradiates tissue above -1.20 mm during a +10 mm cough while sustaining >= 22% duty cycle.",
        category: "Medical Physics",
        icon: "📡",
        domain: "medical",
        tool_name: "RadarGatingTool",
        locked_inputs: "τ_sys = 60 ms, A_breath = 6 mm, T_breath = 2.5 s, +10 mm scheduled cough, seed = 20261123",
        tunable_budget: "Kalman Q ∈ [1e-8, 1], R ∈ [1e-9, 1], Threshold T ∈ [-20, 20] mm, Hysteresis H ∈ [0.05, 10] mm, Clutter window ∈ [4, 64]",
        target_summary: "0 geographic-miss frames (physical beam ON while z_true > -1.20 mm) and duty cycle >= 22%",
        baseline_state_generator: generate_radar_cough_baseline,
        clamp_locked_state: clamp_radar_cough_state,
    }
}

// [cite:mmwave_radiotherapy_setup]
