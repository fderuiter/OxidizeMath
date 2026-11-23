//! Preallocated rotating-frame RF pulse sequence evaluator and MRI inverse challenges.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::framework::{CalloutManager, ChallengeEvaluation, ChallengeMetadata};
use math_explorer::physics::mri::bloch::{rotating_frame_effective_field, BlochSimulator};
use math_explorer::physics::mri::proton;
use math_explorer::pure_math::analysis::ode::{RungeKutta4, Solver};
use nalgebra::Vector3;
use serde::{Deserialize, Serialize};

/// Number of trajectory points in the preallocated pulse buffer (`0..=200`).
pub const MRI_PULSE_SAMPLES: usize = 201;
/// Number of spin isochromats used in Hahn spin-echo static-inhomogeneity refocusing.
pub const HAHN_ISOCHROMAT_COUNT: usize = 9;

/// Canonical challenge ID for rotating-frame $\pi$ spin inversion.
pub const CHALLENGE_BLOCH_PI_INVERSION: &str = "bloch_pi_inversion";
/// Canonical challenge ID for Hahn $90^\circ - \tau - 180^\circ$ spin-echo refocusing.
pub const CHALLENGE_BLOCH_HAHN_ECHO: &str = "bloch_hahn_spin_echo";

/// Locked constants for MRI inverse pulse challenges.
pub const CHALLENGE_MRI_T1: f64 = 1.0;
/// Locked transverse relaxation constant `T2` (s) for MRI inverse pulse challenges.
pub const CHALLENGE_MRI_T2: f64 = 0.08;
/// Locked static field `B0` (T) for MRI inverse pulse challenges.
pub const CHALLENGE_MRI_B0: f64 = 1.5;
/// Locked echo time `TE` (s) for the Hahn spin-echo challenge.
pub const CHALLENGE_MRI_TE: f64 = 0.040;

const ISOCHROMAT_OFFSETS_HZ: [f64; HAHN_ISOCHROMAT_COUNT] =
    [-42.0, -31.5, -21.0, -10.5, 0.0, 10.5, 21.0, 31.5, 42.0];

/// Operating mode of [`super::tool::MriTool`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MriPulseMode {
    /// Continuous rotating-frame magnetic field evolution (`BackgroundWorker` mode).
    #[default]
    ContinuousField,
    /// Single-pulse RF spin inversion from equilibrium $M(0) = (0, 0, M_0)$.
    InversionPulse,
    /// Hahn spin-echo refocusing of dephasing spin isochromats at $t = TE$.
    HahnSpinEcho,
}

impl MriPulseMode {
    /// Returns true when in default continuous forward mode.
    #[must_use]
    pub fn is_continuous(self) -> bool {
        matches!(self, Self::ContinuousField)
    }
}

/// Tunable RF pulse parameters in [`super::tool::MriTool`].
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RfPulseConfig {
    /// Operating sequence mode.
    #[serde(default)]
    pub mode: MriPulseMode,
    /// Transverse RF envelope amplitude $B_1$ in Tesla (`0.0..=1.0e-4`).
    pub b1_amp: f64,
    /// Rectangular RF pulse duration $\tau$ in seconds (`0.0005..=0.050`).
    pub duration_tau: f64,
    /// RF transmitter phase $\phi$ in degrees (`-180.0..=180.0`).
    pub phase_deg: f64,
    /// Carrier frequency detuning $\Delta f = (\omega_{\text{rf}} - \omega_0)/(2\pi)$ in Hz (`-5000.0..=5000.0`).
    pub detuning_hz: f64,
    /// Echo time $TE$ in seconds (`0.010..=0.200`) for Hahn spin echo.
    pub echo_time_te: f64,
}

impl Default for RfPulseConfig {
    fn default() -> Self {
        Self {
            mode: MriPulseMode::ContinuousField,
            b1_amp: 0.8e-6,
            duration_tau: 0.005,
            phase_deg: 0.0,
            detuning_hz: 0.0,
            echo_time_te: CHALLENGE_MRI_TE,
        }
    }
}

impl RfPulseConfig {
    /// Validates finiteness and bounds for all RF pulse fields.
    pub fn validate(&self) -> Result<(), String> {
        let checks = [
            ("b1_amp", self.b1_amp, 0.0, 2.0e-4),
            ("duration_tau", self.duration_tau, 0.0002, 0.050),
            ("phase_deg", self.phase_deg, -180.0, 180.0),
            ("detuning_hz", self.detuning_hz, -10_000.0, 10_000.0),
            ("echo_time_te", self.echo_time_te, 0.005, 0.250),
        ];
        for (name, val, min, max) in checks {
            if !val.is_finite() || val < min || val > max {
                return Err(format!(
                    "MRI pulse parameter '{name}' = {val} must be finite within [{min}, {max}]"
                ));
            }
        }
        Ok(())
    }

    /// Nominal on-resonance flip angle $\alpha = \gamma B_1 \tau$ in degrees.
    #[must_use]
    pub fn nominal_flip_deg(&self) -> f64 {
        (proton::GYROMAGNETIC_RATIO * self.b1_amp * self.duration_tau).to_degrees()
    }
}

/// Preallocated workspace for zero-heap-allocation rotating-frame Bloch integration.
pub struct MriPulseWorkspace {
    /// Sample timestamps across the pulse sequence horizon.
    pub times: [f64; MRI_PULSE_SAMPLES],
    /// Ensemble-averaged magnetization vectors across the sequence horizon.
    pub trajectory: [Vector3<f64>; MRI_PULSE_SAMPLES],
    isochromats: [Vector3<f64>; HAHN_ISOCHROMAT_COUNT],
    sim: BlochSimulator,
    rk4: RungeKutta4<Vector3<f64>>,
}

impl Default for MriPulseWorkspace {
    fn default() -> Self {
        let m_eq = Vector3::new(0.0, 0.0, 1.0);
        Self {
            times: [0.0; MRI_PULSE_SAMPLES],
            trajectory: [m_eq; MRI_PULSE_SAMPLES],
            isochromats: [m_eq; HAHN_ISOCHROMAT_COUNT],
            sim: BlochSimulator::new(m_eq, 1.0),
            rk4: RungeKutta4::new(&m_eq),
        }
    }
}

impl MriPulseWorkspace {
    /// Evaluates a single rotating-frame RF inversion pulse starting from $M(0) = (0, 0, M_0)$
    /// with zero dynamic heap allocations.
    pub fn evaluate_inversion(
        &mut self,
        b0: f64,
        t1: f64,
        t2: f64,
        m0: f64,
        cfg: &RfPulseConfig,
    ) -> Vector3<f64> {
        let steps = MRI_PULSE_SAMPLES - 1;
        let tau = cfg.duration_tau.max(0.0002);
        let dt = tau / (steps as f64);
        let omega_0 = proton::larmor_frequency(b0);
        let omega_rf = omega_0 + std::f64::consts::TAU * cfg.detuning_hz;
        let b_eff =
            rotating_frame_effective_field(b0, omega_rf, cfg.b1_amp, cfg.phase_deg.to_radians());

        self.sim.m0 = m0;
        self.sim.set_relaxation(t1, t2);
        self.sim.set_b_field(b_eff);
        let mut m = Vector3::new(0.0, 0.0, m0);
        self.times[0] = 0.0;
        self.trajectory[0] = m;

        for idx in 1..=steps {
            let t_prev = ((idx - 1) as f64) * dt;
            self.rk4.step(&self.sim, t_prev, &mut m, dt);
            self.times[idx] = (idx as f64) * dt;
            self.trajectory[idx] = m;
        }
        m
    }

    /// Evaluates a Hahn spin-echo sequence ($90^\circ_y$ excitation at $t=0$ followed by tunable
    /// refocusing pulse centered at $TE / 2$) across 9 static-inhomogeneity isochromats with
    /// zero dynamic heap allocations.
    pub fn evaluate_hahn_echo(
        &mut self,
        b0: f64,
        t1: f64,
        t2: f64,
        m0: f64,
        cfg: &RfPulseConfig,
    ) -> Vector3<f64> {
        let steps = MRI_PULSE_SAMPLES - 1;
        let te = cfg.echo_time_te.max(0.010);
        let dt = te / (steps as f64);
        let half_te = 0.5 * te;
        let half_tau = 0.5 * cfg.duration_tau.min(0.8 * te);
        let omega_0 = proton::larmor_frequency(b0);
        let omega_rf = omega_0 + std::f64::consts::TAU * cfg.detuning_hz;
        let phase_rad = cfg.phase_deg.to_radians();

        let m_init = Vector3::new(m0, 0.0, 0.0);
        self.isochromats.fill(m_init);
        self.sim.m0 = m0;
        self.sim.set_relaxation(t1, t2);
        self.times[0] = 0.0;
        self.trajectory[0] = m_init;

        for idx in 1..=steps {
            let t_mid = ((idx as f64) - 0.5) * dt;
            let pulse_on = (t_mid - half_te).abs() <= half_tau;
            let active_b1 = if pulse_on { cfg.b1_amp } else { 0.0 };
            let mut sum = Vector3::zeros();
            for (iso_i, m_iso) in self.isochromats.iter_mut().enumerate() {
                let b_iso = b0
                    + std::f64::consts::TAU * ISOCHROMAT_OFFSETS_HZ[iso_i]
                        / proton::GYROMAGNETIC_RATIO;
                let b_eff = rotating_frame_effective_field(b_iso, omega_rf, active_b1, phase_rad);
                self.sim.set_b_field(b_eff);
                self.rk4.step(&self.sim, t_mid - 0.5 * dt, m_iso, dt);
                sum += *m_iso;
            }
            let mean_m = sum / (HAHN_ISOCHROMAT_COUNT as f64);
            self.times[idx] = (idx as f64) * dt;
            self.trajectory[idx] = mean_m;
        }
        self.trajectory[steps]
    }
}

/// Scores spin inversion: pass when $r_{\text{inv}} = M_z(T_{\text{end}})/M_0 \le -0.90$,
/// $S_{\text{inv}} = \text{clamp}(100 (1 - \max(0, r_{\text{inv}} + 1)/2), 0, 100)$.
#[must_use]
pub fn score_pi_inversion(final_m: Vector3<f64>, m0: f64) -> ChallengeEvaluation {
    let denom = if m0.abs() > 1e-12 { m0 } else { 1.0 };
    let r_inv = final_m.z / denom;
    let score = (100.0 * (1.0 - (r_inv + 1.0).max(0.0) / 2.0)).clamp(0.0, 100.0);
    let passed = r_inv.is_finite() && r_inv <= -0.90;
    ChallengeEvaluation::new(
        score,
        passed,
        "Mz(tau)/M0",
        r_inv,
        format!("Target <= -0.9000 (current {r_inv:.4})"),
    )
}

/// Scores Hahn spin echo: pass when $\text{hypot}(M_x(TE), M_y(TE)) \ge 0.75 M_0 e^{-TE/T_2}$.
#[must_use]
pub fn score_hahn_spin_echo(
    echo_m: Vector3<f64>,
    m0: f64,
    te: f64,
    t2: f64,
) -> ChallengeEvaluation {
    let echo_xy = echo_m.x.hypot(echo_m.y);
    let ideal = m0 * (-te / t2.max(1e-6)).exp();
    let threshold = 0.75 * ideal;
    let ratio = if ideal > 1e-12 { echo_xy / ideal } else { 0.0 };
    let score = (100.0 * ratio).clamp(0.0, 100.0);
    let passed = echo_xy.is_finite() && echo_xy >= threshold;
    ChallengeEvaluation::new(
        score,
        passed,
        "|Mxy(TE)|",
        echo_xy,
        format!("Target >= {threshold:.4} (75% of T2 envelope {ideal:.4})"),
    )
}

/// Updates automated plot callouts on [`CalloutManager`] for all MRI operating modes.
pub fn sync_mri_callouts(
    mgr: &mut CalloutManager,
    mode: MriPulseMode,
    m0: f64,
    t2: f64,
    cfg: &RfPulseConfig,
    latest_t: f64,
    latest_m: Vector3<f64>,
) {
    mgr.clear_auto();
    match mode {
        MriPulseMode::InversionPulse => {
            let tau = cfg.duration_tau;
            mgr.set_auto_callout(
                "mri_inv_target",
                [tau, -0.90 * m0],
                format!("Target Mz <= {:.2}", -0.90 * m0),
                [-95.0, -24.0],
            );
            mgr.set_auto_callout(
                "mri_inv_actual",
                [tau, latest_m.z],
                format!("Mz(tau) = {:.3}", latest_m.z),
                [-85.0, 22.0],
            );
        }
        MriPulseMode::HahnSpinEcho => {
            let te = cfg.echo_time_te;
            let ideal = m0 * (-te / t2.max(1e-6)).exp();
            let echo_xy = latest_m.x.hypot(latest_m.y);
            mgr.set_auto_callout(
                "mri_echo_target",
                [te, 0.75 * ideal],
                format!("Echo Pass >= {:.3}", 0.75 * ideal),
                [-105.0, -26.0],
            );
            mgr.set_auto_callout(
                "mri_echo_actual",
                [te, echo_xy],
                format!("|Mxy(TE)| = {:.3}", echo_xy),
                [-95.0, 22.0],
            );
        }
        MriPulseMode::ContinuousField => {
            mgr.set_auto_callout(
                "mri_fid_start",
                [0.0, m0],
                "Initial M(0) = M0",
                [28.0, -22.0],
            );
            if latest_t > 0.0 {
                mgr.set_auto_callout(
                    "mri_fid_mz",
                    [latest_t, latest_m.z],
                    format!("Mz({latest_t:.2}s) = {:.3}", latest_m.z),
                    [-85.0, -24.0],
                );
            }
        }
    }
}

fn generate_pi_inversion_baseline() -> String {
    serde_json::json!({
        "t1": CHALLENGE_MRI_T1,
        "t2": CHALLENGE_MRI_T2,
        "bx": 0.0,
        "by": 0.0,
        "bz": CHALLENGE_MRI_B0,
        "time_scale": 1.0,
        "state": "Ready",
        "pulse": {
            "mode": "InversionPulse",
            "b1_amp": 0.8e-6,
            "duration_tau": 0.005,
            "phase_deg": 0.0,
            "detuning_hz": 0.0,
            "echo_time_te": CHALLENGE_MRI_TE
        }
    })
    .to_string()
}

fn generate_hahn_echo_baseline() -> String {
    serde_json::json!({
        "t1": CHALLENGE_MRI_T1,
        "t2": CHALLENGE_MRI_T2,
        "bx": 0.0,
        "by": 0.0,
        "bz": CHALLENGE_MRI_B0,
        "time_scale": 1.0,
        "state": "Ready",
        "pulse": {
            "mode": "HahnSpinEcho",
            "b1_amp": 1.2e-6,
            "duration_tau": 0.002,
            "phase_deg": 0.0,
            "detuning_hz": 0.0,
            "echo_time_te": CHALLENGE_MRI_TE
        }
    })
    .to_string()
}

fn clamp_mri_challenge_json(raw_json: &str, mode: MriPulseMode) -> Result<String, String> {
    let mut state: super::tool::MriToolState =
        serde_json::from_str(raw_json).map_err(|e| format!("Invalid MRI challenge JSON: {e}"))?;
    state.t1 = CHALLENGE_MRI_T1;
    state.t2 = CHALLENGE_MRI_T2;
    state.bx = 0.0;
    state.by = 0.0;
    state.bz = CHALLENGE_MRI_B0;
    state.time_scale = state.time_scale.clamp(1.0, 100.0);
    let mut pulse = state.pulse.unwrap_or_default();
    pulse.mode = mode;
    pulse.b1_amp = pulse.b1_amp.clamp(0.0, 5.0e-5);
    pulse.duration_tau = pulse.duration_tau.clamp(0.0005, 0.020);
    pulse.phase_deg = pulse.phase_deg.clamp(-180.0, 180.0);
    pulse.detuning_hz = pulse.detuning_hz.clamp(-5000.0, 5000.0);
    pulse.echo_time_te = CHALLENGE_MRI_TE;
    state.pulse = Some(pulse);
    state.validate()?;
    serde_json::to_string(&state).map_err(|e| format!("MRI serialize error: {e}"))
}

fn clamp_pi_inversion_state(raw: &str) -> Result<String, String> {
    clamp_mri_challenge_json(raw, MriPulseMode::InversionPulse)
}

fn clamp_hahn_echo_state(raw: &str) -> Result<String, String> {
    clamp_mri_challenge_json(raw, MriPulseMode::HahnSpinEcho)
}

inventory::submit! {
    ChallengeMetadata {
        id: CHALLENGE_BLOCH_PI_INVERSION,
        title: "Bloch 180° Spin Inversion",
        description: "Design an on-resonance rotating-frame RF pulse that flips longitudinal magnetization M(0) = (0, 0, M0) past -0.90 M0 without Larmor aliasing.",
        category: "MRI Pulse Design",
        icon: "🧲",
        domain: "mri",
        tool_name: "MriTool",
        locked_inputs: "T1 = 1.0 s, T2 = 0.08 s, B0 = 1.5 T, M(0) = (0, 0, M0)",
        tunable_budget: "B1 in [0, 50] uT, tau in [0.5, 20] ms, phase in [-180, 180] deg",
        target_summary: "Mz(tau) / M0 <= -0.90",
        baseline_state_generator: generate_pi_inversion_baseline,
        clamp_locked_state: clamp_pi_inversion_state,
    }
}

inventory::submit! {
    ChallengeMetadata {
        id: CHALLENGE_BLOCH_HAHN_ECHO,
        title: "Hahn Spin-Echo Refocusing",
        description: "Refocus 9 dephasing static-inhomogeneity spin isochromats at TE = 40 ms using a centered RF refocusing pulse.",
        category: "MRI Echo Sequences",
        icon: "📡",
        domain: "mri",
        tool_name: "MriTool",
        locked_inputs: "T1 = 1.0 s, T2 = 0.08 s, B0 = 1.5 T, TE = 40 ms",
        tunable_budget: "B1 in [0, 50] uT, tau in [0.5, 20] ms, phase in [-180, 180] deg",
        target_summary: "hypot(Mx(TE), My(TE)) >= 0.75 * M0 * exp(-TE / T2)",
        baseline_state_generator: generate_hahn_echo_baseline,
        clamp_locked_state: clamp_hahn_echo_state,
    }
}

// [cite:graph_parameters_rust]
