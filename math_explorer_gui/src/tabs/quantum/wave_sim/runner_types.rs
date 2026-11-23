//! Shared parameter configuration and snapshot payload types for [`super::WaveRunner`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::super::presets::{
    default_compiled_expr, WavePotentialPreset, DEFAULT_DT, DEFAULT_K0, DEFAULT_SIGMA,
    DEFAULT_STEPS_PER_FRAME, DEFAULT_V0, DEFAULT_WIDTH, DEFAULT_X0,
};
use math_commons::ast::Expr;

/// Shared parameter bundle synchronized between [`super::super::WaveSimulatorTool`] and [`super::WaveRunner`].
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
    /// Optional sparse `(index, value)` sculpted potential overrides.
    pub sparse_potential: Vec<(u16, f64)>,
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
            sparse_potential: Vec::new(),
        }
    }
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
    #[cfg(test)]
    pub right_wall_prob: f64,
    /// Latched post-scattering, pre-wall transmission readout $T_{\text{scatter}}$.
    pub latched_transmission: Option<f64>,
    /// Whether the wavepacket has reached either Dirichlet guard wall (`|x| > 4.25`).
    pub wall_arrived: bool,
    /// Whether `latched_transmission` was captured from a quasi-bound resonance at pre-wall arrival.
    pub quasi_bound_latched: bool,
    /// Left edge of the barrier integration interval.
    pub x_left: f64,
    /// Right edge of the barrier integration interval.
    pub x_right: f64,
    /// Incident wavepacket kinetic energy $E_0 = k_0^2 / 2$.
    pub kinetic_energy: f64,
    /// Unitary state norm squared $\|\psi\|^2$.
    pub norm_sq: f64,
}
