#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::presets::{
    default_compiled_expr, WavePotentialPreset, DEFAULT_DT, DEFAULT_K0, DEFAULT_SIGMA,
    DEFAULT_STEPS_PER_FRAME, DEFAULT_V0, DEFAULT_WIDTH, DEFAULT_X0, GRID_X_MAX, GRID_X_MIN,
    POTENTIAL_DISPLAY_SCALE,
};
use math_commons::ast::Expr;
use math_explorer::physics::quantum::QuantumState;
use nalgebra::DVector;

/// Shared parameter bundle synchronized between [`super::WaveSimulatorTool`] and [`super::WaveRunner`].
#[derive(Clone, Debug)]
pub struct WaveSimConfig {
    /// Selected potential profile or custom AST mode.
    pub preset: WavePotentialPreset,
    /// Initial Gaussian packet center $x_0$.
    pub x0: f64,
    /// Initial wave number $k_0$ ($E_0 = k_0^2 / 2$).
    pub k0: f64,
    /// Initial Gaussian spatial width $\sigma$.
    pub sigma: f64,
    /// Potential barrier scale $V_0$.
    pub v0: f64,
    /// Potential barrier width $w$.
    pub width: f64,
    /// Unitary propagation time step $\Delta t$.
    pub dt: f64,
    /// Worker steps executed per visual frame.
    pub steps_per_frame: usize,
    /// Last valid compiled AST expression for `V(x)`.
    pub custom_expr: Expr,
    /// Optional 200-point freehand sculpted potential overriding the baseline profile.
    pub custom_potential: Option<Vec<f64>>,
}

impl Default for WaveSimConfig {
    fn default() -> Self {
        Self {
            preset: WavePotentialPreset::InfiniteWell,
            x0: DEFAULT_X0,
            k0: DEFAULT_K0,
            sigma: DEFAULT_SIGMA,
            v0: DEFAULT_V0,
            width: DEFAULT_WIDTH,
            dt: DEFAULT_DT,
            steps_per_frame: DEFAULT_STEPS_PER_FRAME,
            custom_expr: default_compiled_expr(),
            custom_potential: None,
        }
    }
}

/// Threshold on interior barrier probability $B(t)$ indicating packet transit completion.
pub const BARRIER_TRANSIT_THRESHOLD: f64 = 0.02;
/// Right-wall proximity coordinate above which boundary reflection begins.
pub const RIGHT_WALL_GUARD_X: f64 = 4.25;
/// Threshold on $\sum_{x > 4.25} |\psi|^2$ before right-wall bounce freezes $T_{\text{scatter}}$.
pub const WALL_ARRIVAL_THRESHOLD: f64 = 0.02;

/// Integrated Born probabilities across reflection, barrier, transmission, and wall zones.
#[derive(Clone, Copy, Debug, Default)]
pub struct TunnelingObservables {
    /// Reflected Born probability $\sum_{x < x_{\text{left}}} |\psi(x)|^2$.
    pub reflection: f64,
    /// Interior barrier probability $\sum_{x_{\text{left}} \le x \le x_{\text{right}}} |\psi(x)|^2$.
    pub barrier_prob: f64,
    /// Transmitted Born probability $\sum_{x > x_{\text{right}}} |\psi(x)|^2$.
    pub transmission: f64,
    /// Right-wall arrival indicator $\sum_{x > 4.25} |\psi|^2$.
    pub right_wall_prob: f64,
    /// Left-wall arrival indicator $\sum_{x < -4.25} |\psi|^2$.
    pub left_wall_prob: f64,
}

/// Zero-allocation Born R/B/T integration directly over [`QuantumState`] complex amplitudes.
#[must_use]
pub(crate) fn compute_tunneling_from_psi(
    x_axis: &[f64],
    psi: &QuantumState,
    x_left: f64,
    x_right: f64,
) -> TunnelingObservables {
    let mut obs = TunnelingObservables::default();
    for (&x, c) in x_axis.iter().zip(psi.vector.as_slice().iter()) {
        let p = c.norm_sqr();
        if !p.is_finite() {
            continue;
        }
        if x < x_left {
            obs.reflection += p;
        } else if x <= x_right {
            obs.barrier_prob += p;
        } else {
            obs.transmission += p;
        }
        if x > RIGHT_WALL_GUARD_X {
            obs.right_wall_prob += p;
        } else if x < -RIGHT_WALL_GUARD_X {
            obs.left_wall_prob += p;
        }
    }
    obs
}

/// Tracks post-scattering, pre-wall transmission latch state across propagation steps.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ScatterLatchState {
    pub entered_barrier: bool,
    pub wall_arrived: bool,
    pub latched_transmission: Option<f64>,
}

impl ScatterLatchState {
    pub fn update(
        &mut self,
        time: f64,
        classical_past_barrier: bool,
        prev: TunnelingObservables,
        obs: TunnelingObservables,
    ) {
        if time <= 0.0 {
            return;
        }
        if obs.barrier_prob >= BARRIER_TRANSIT_THRESHOLD || classical_past_barrier {
            self.entered_barrier = true;
        }
        if self.entered_barrier
            && !self.wall_arrived
            && obs.barrier_prob < BARRIER_TRANSIT_THRESHOLD
        {
            if obs.right_wall_prob < WALL_ARRIVAL_THRESHOLD {
                self.latched_transmission = Some(obs.transmission);
            } else if self.latched_transmission.is_none()
                && prev.right_wall_prob < WALL_ARRIVAL_THRESHOLD
            {
                let alpha_b = if prev.barrier_prob < BARRIER_TRANSIT_THRESHOLD {
                    0.0
                } else {
                    let denom = (prev.barrier_prob - obs.barrier_prob).max(f64::EPSILON);
                    ((prev.barrier_prob - BARRIER_TRANSIT_THRESHOLD) / denom).clamp(0.0, 1.0)
                };
                let denom_w = (obs.right_wall_prob - prev.right_wall_prob).max(f64::EPSILON);
                let alpha_w =
                    ((WALL_ARRIVAL_THRESHOLD - prev.right_wall_prob) / denom_w).clamp(0.0, 1.0);
                if alpha_b <= alpha_w {
                    let t_pre =
                        prev.transmission + alpha_w * (obs.transmission - prev.transmission);
                    self.latched_transmission = Some(t_pre.clamp(0.0, 1.0));
                }
            }
        }
        if obs.right_wall_prob >= WALL_ARRIVAL_THRESHOLD
            || (self.latched_transmission.is_some() && obs.left_wall_prob >= WALL_ARRIVAL_THRESHOLD)
        {
            self.wall_arrived = true;
        }
    }
}

/// Integrates [`math_explorer::physics::quantum::QuantumState::probability_density`] into reflection, barrier, and transmission sums.
#[must_use]
pub fn compute_tunneling_observables(
    x_axis: &[f64],
    prob_density: &DVector<f64>,
    x_left: f64,
    x_right: f64,
) -> TunnelingObservables {
    let mut obs = TunnelingObservables::default();
    let len = x_axis.len().min(prob_density.len());
    for i in 0..len {
        let p = prob_density[i];
        if !p.is_finite() {
            continue;
        }
        let x = x_axis[i];
        if x < x_left {
            obs.reflection += p;
        } else if x <= x_right {
            obs.barrier_prob += p;
        } else {
            obs.transmission += p;
        }
        if x > RIGHT_WALL_GUARD_X {
            obs.right_wall_prob += p;
        } else if x < -RIGHT_WALL_GUARD_X {
            obs.left_wall_prob += p;
        }
    }
    obs
}

/// Structured snapshot payload emitted by [`super::WaveRunner`] to the UI thread.
#[derive(Clone, Debug)]
pub struct WaveData {
    /// Current simulation time $t$.
    pub time: f64,
    /// Spatial coordinate grid $x_i \in [-5, 5]$.
    pub x_axis: Vec<f64>,
    /// Sanitized Born probability density $|\psi(x_i, t)|^2$.
    pub prob_density: Vec<f64>,
    /// Clamped physical potential $V(x_i) \in [-500, 500]$.
    pub potential: Vec<f64>,
    /// Live reflection probability $R(t)$.
    pub reflection: f64,
    /// Live barrier interior probability $B(t)$.
    pub barrier_prob: f64,
    /// Live transmission probability $T(t)$.
    pub transmission: f64,
    /// Right-wall boundary probability $\sum_{x > 4.25} |\psi|^2$.
    pub right_wall_prob: f64,
    /// Latched post-scattering, pre-wall transmission readout $T_{\text{scatter}}$.
    pub latched_transmission: Option<f64>,
    /// Left edge of the barrier integration interval.
    pub x_left: f64,
    /// Right edge of the barrier integration interval.
    pub x_right: f64,
    /// Incident wavepacket kinetic energy $E_0 = k_0^2 / 2$.
    pub kinetic_energy: f64,
    /// Conserved two-region transmitted probability $P_T = \sum_{i > i_b} |\psi_i|^2$.
    pub p_transmitted: f64,
    /// Conserved two-region reflected probability $P_R = 1 - P_T$.
    pub p_reflected: f64,
    /// Detected barrier partition index $i_b$.
    pub barrier_idx: usize,
}

/// Builds exportable 2D plot series (`|ψ(x, t)|²`, scaled `V(x)`, and horizontal `E0 = k0²/2`).
#[must_use]
pub fn build_wave_datasets(wave_data: &WaveData) -> Vec<(String, Vec<[f64; 2]>)> {
    let len = wave_data
        .x_axis
        .len()
        .min(wave_data.prob_density.len())
        .min(wave_data.potential.len());
    let mut psi_points = Vec::with_capacity(len);
    let mut v_points = Vec::with_capacity(len);
    for i in 0..len {
        let x = wave_data.x_axis[i];
        let p = wave_data.prob_density[i];
        let v = wave_data.potential[i];
        if x.is_finite() && p.is_finite() && v.is_finite() {
            psi_points.push([x, p]);
            v_points.push([x, v * POTENTIAL_DISPLAY_SCALE]);
        }
    }

    let x_start = wave_data.x_axis.first().copied().unwrap_or(GRID_X_MIN);
    let x_end = wave_data.x_axis.last().copied().unwrap_or(GRID_X_MAX);
    let e0_y = if wave_data.kinetic_energy.is_finite() {
        wave_data.kinetic_energy * POTENTIAL_DISPLAY_SCALE
    } else {
        0.0
    };

    vec![
        (
            format!("Probability |ψ(x, t={:.3})|²", wave_data.time),
            psi_points,
        ),
        ("Potential V(x) (scaled)".to_string(), v_points),
        (
            format!(
                "Kinetic Energy E0 = {:.2} (scaled)",
                wave_data.kinetic_energy
            ),
            vec![[x_start, e0_y], [x_end, e0_y]],
        ),
    ]
}

pub(crate) const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Schr\u00f6dinger Wave Packet",
    "domain": "quantum",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "potential_type": 1.0,
    "dt": 0.02
  }
}"#;

const TUNNELING_BARRIER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Quantum Tunneling Barrier",
    "domain": "quantum",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "preset": "RectangularBarrier",
    "dt": 0.02,
    "x0": -2.4,
    "k0": 5.0,
    "sigma": 0.45,
    "v0": 14.0,
    "width": 0.6
  }
}"#;

const RESONANT_DOUBLE_WELL_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Resonant Double-Well Cavity",
    "domain": "quantum",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "preset": "DoubleBarrier",
    "dt": 0.02,
    "x0": -2.8,
    "k0": 5.2,
    "sigma": 0.4,
    "v0": 16.0,
    "width": 0.6
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "wave_sim_starter",
        title: "Schrödinger Wave Packet",
        description: "iℏ·∂ψ/∂t = -ℏ²/(2m)∇²ψ + V·ψ: 1D Gaussian wave packet evolution in infinite-well, harmonic, and sculpted potentials.",
        category: "Wave Mechanics",
        icon: "🌊",
        domain: "quantum",
        tool_name: Some("WaveSimulator"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "wave_sim_tunneling_barrier",
        title: "Quantum Tunneling Barrier",
        description: "Sub-barrier wavepacket scattering across a rectangular potential barrier with reflected fringes and evanescent tunneling.",
        category: "Wave Mechanics",
        icon: "⛰",
        domain: "quantum",
        tool_name: Some("WaveSimulator"),
        preset_generator: || TUNNELING_BARRIER_PRESET_JSON,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "wave_sim_resonant_double_well",
        title: "Resonant Double-Well Cavity",
        description: "Fabry-Pérot quantum resonance and quasi-bound wavepacket trapping across symmetric dual potential barriers.",
        category: "Wave Mechanics",
        icon: "⛩",
        domain: "quantum",
        tool_name: Some("WaveSimulator"),
        preset_generator: || RESONANT_DOUBLE_WELL_PRESET_JSON,
    }
}
