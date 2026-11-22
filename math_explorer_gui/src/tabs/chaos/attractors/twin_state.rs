#![cfg_attr(any(), verified(opt_out = "gui_tool"))]
//! Twin-trajectory model selection, symbolic AST perturbations, and Lyapunov metrics.

use crate::framework::{MathExpressionInputBar, TimelineRingBuffer};
use math_explorer::physics::chaos::double_pendulum::{DoublePendulumBuilder, DoublePendulumState};
use math_explorer::physics::chaos::fractals::correlation_dimension;
use math_explorer::physics::chaos::lorenz::{LorenzBuilder, LorenzState};
use math_explorer::physics::chaos::metrics::{
    double_pendulum_lyapunov, finite_time_lyapunov, lorenz_lyapunov,
};
use math_explorer::physics::chaos::rossler::{RosslerBuilder, RosslerState};
use math_explorer::pure_math::analysis::ode::{ArrayState, RungeKutta4, Solver};
use nalgebra::Vector3;
use std::collections::HashMap;

/// Canonical on-attractor Lorenz coordinates where a `1e-9` x-offset crosses `0.1`
/// separation at `t = 19.96` and flips onto opposite butterfly wings near `t = 24.61`.
pub const ON_ATTRACTOR_LORENZ_STATE: [f64; 3] =
    [-4.385295763939413, -2.064285615514713, 25.90419269166981];

/// Default initial separation magnitude (`1e-9`) between primary and shadow trajectories.
pub const DEFAULT_EPSILON: f64 = 1e-9;

/// Default symbolic AST expression for initial-state perturbations.
pub const DEFAULT_PERTURBATION_EXPR: &str = "epsilon";

/// Minimum clamped separation used when computing finite `log10(||delta x(t)||)` series.
pub const MIN_LOG_SEPARATION: f64 = 1e-16;

/// Preset formulas offered in [`MathExpressionInputBar`] for symbolic twin perturbations.
pub const PERTURBATION_PRESETS: &[(&str, &str)] = &[
    ("Axis offset (ε)", "epsilon"),
    ("Cosine x₀ modulation", "1e-9 * cos(x0)"),
    ("Radial state scaling", "epsilon * sqrt(x0^2 + y0^2 + 1)"),
    ("Symmetric y₀ coupling", "epsilon * (1 + 0.5 * sin(y0))"),
];

/// Continuous chaotic dynamical system selectable inside [`super::AttractorPlotter`].
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Default, serde::Serialize, serde::Deserialize,
)]
pub enum AttractorSystemKind {
    /// Atmospheric convection butterfly attractor $(\sigma, \rho, \beta)$.
    #[default]
    Lorenz,
    /// Spiral-stretch-and-fold attractor $(a, b, c)$.
    Rossler,
    /// Planar two-link Lagrangian double pendulum $(m_1, m_2, l_1, l_2, g)$.
    DoublePendulum,
}

impl AttractorSystemKind {
    /// All selectable chaotic models in UI display order.
    pub const ALL: [Self; 3] = [Self::Lorenz, Self::Rossler, Self::DoublePendulum];

    /// User-facing model title.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Lorenz => "Lorenz Attractor",
            Self::Rossler => "Rössler Attractor",
            Self::DoublePendulum => "Double Pendulum",
        }
    }

    /// Coordinate input labels `(c0, c1, c2)` for the active dynamical system.
    #[must_use]
    pub fn coord_labels(self) -> [&'static str; 3] {
        match self {
            Self::Lorenz | Self::Rossler => ["x₀", "y₀", "z₀"],
            Self::DoublePendulum => ["θ₁₀", "θ₂₀", "ω₁₀"],
        }
    }

    /// Default initial coordinates when switching to `self` interactively.
    #[must_use]
    pub fn default_initial_state(self) -> [f64; 3] {
        match self {
            Self::Lorenz => ON_ATTRACTOR_LORENZ_STATE,
            Self::Rossler => [0.1, 0.0, 0.0],
            Self::DoublePendulum => [2.4, 2.4, 0.0],
        }
    }

    /// Recommended integration step `dt` for `self`.
    #[must_use]
    pub fn default_dt(self) -> f64 {
        match self {
            Self::Lorenz => 0.01,
            Self::Rossler => 0.02,
            Self::DoublePendulum => 0.005,
        }
    }

    /// 3D camera center offset for projecting trajectories into screen space.
    #[must_use]
    pub fn center_offset(self) -> Vector3<f64> {
        match self {
            Self::Lorenz => Vector3::new(0.0, 0.0, 25.0),
            Self::Rossler => Vector3::new(0.0, 0.0, 5.0),
            Self::DoublePendulum => Vector3::zeros(),
        }
    }

    /// Half-extent passed to `Plot3DViewport::new` for `self`.
    #[must_use]
    pub fn viewport_half_extent(self) -> f64 {
        match self {
            Self::Lorenz => 32.0,
            Self::Rossler => 18.0,
            Self::DoublePendulum => 9.0,
        }
    }

    /// Default neighborhood radius $r$ for Grassberger-Procaccia correlation sum $C(r)$.
    #[must_use]
    pub fn default_correlation_radius(self) -> f64 {
        match self {
            Self::Lorenz => 5.0,
            Self::Rossler => 2.5,
            Self::DoublePendulum => 1.0,
        }
    }
}

/// Model-specific physical constants for Rössler and double-pendulum systems.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ExtraChaosParams {
    /// Rössler parameter $a$.
    pub rossler_a: f64,
    /// Rössler parameter $b$.
    pub rossler_b: f64,
    /// Rössler parameter $c$.
    pub rossler_c: f64,
    /// Double-pendulum upper mass $m_1$.
    pub pendulum_m1: f64,
    /// Double-pendulum lower mass $m_2$.
    pub pendulum_m2: f64,
    /// Double-pendulum upper length $l_1$.
    pub pendulum_l1: f64,
    /// Double-pendulum lower length $l_2$.
    pub pendulum_l2: f64,
    /// Gravitational acceleration $g$.
    pub pendulum_g: f64,
}

impl Default for ExtraChaosParams {
    fn default() -> Self {
        Self {
            rossler_a: 0.2,
            rossler_b: 0.2,
            rossler_c: 5.7,
            pendulum_m1: 1.0,
            pendulum_m2: 1.0,
            pendulum_l1: 1.0,
            pendulum_l2: 1.0,
            pendulum_g: 9.81,
        }
    }
}

impl ExtraChaosParams {
    /// Returns `true` when all auxiliary model parameters are finite and positive where required.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.rossler_a.is_finite()
            && self.rossler_b.is_finite()
            && self.rossler_c.is_finite()
            && self.pendulum_m1.is_finite()
            && self.pendulum_m1 > 0.0
            && self.pendulum_m2.is_finite()
            && self.pendulum_m2 > 0.0
            && self.pendulum_l1.is_finite()
            && self.pendulum_l1 > 0.0
            && self.pendulum_l2.is_finite()
            && self.pendulum_l2 > 0.0
            && self.pendulum_g.is_finite()
            && self.pendulum_g > 0.0
    }

    /// Clamps auxiliary parameters to safe physical ranges.
    pub fn sanitize(&mut self) {
        let def = Self::default();
        self.rossler_a = sanitize_range(self.rossler_a, -1.0, 2.0, def.rossler_a);
        self.rossler_b = sanitize_range(self.rossler_b, -1.0, 5.0, def.rossler_b);
        self.rossler_c = sanitize_range(self.rossler_c, 0.1, 25.0, def.rossler_c);
        self.pendulum_m1 = sanitize_range(self.pendulum_m1, 0.1, 10.0, def.pendulum_m1);
        self.pendulum_m2 = sanitize_range(self.pendulum_m2, 0.1, 10.0, def.pendulum_m2);
        self.pendulum_l1 = sanitize_range(self.pendulum_l1, 0.1, 5.0, def.pendulum_l1);
        self.pendulum_l2 = sanitize_range(self.pendulum_l2, 0.1, 5.0, def.pendulum_l2);
        self.pendulum_g = sanitize_range(self.pendulum_g, 0.5, 30.0, def.pendulum_g);
    }
}

fn sanitize_range(val: f64, min: f64, max: f64, fallback: f64) -> f64 {
    if val.is_finite() {
        val.clamp(min, max)
    } else {
        fallback
    }
}

/// Fixed-size zero-allocation frame stored inside [`TimelineRingBuffer`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TwinFrame {
    /// Simulation step index (`0` = initial state).
    pub step: usize,
    /// Elapsed simulation time $t$.
    pub time: f64,
    /// Primary state coordinates (`[x, y, z, 0.0]` or `[theta1, theta2, omega1, omega2]`).
    pub primary: [f64; 4],
    /// Shadow state coordinates.
    pub shadow: [f64; 4],
    /// Euclidean phase-space distance $\|\delta \mathbf{x}(t)\|_2$.
    pub separation: f64,
    /// Clamped base-10 logarithm $\log_{10}(\max(\|\delta \mathbf{x}(t)\|_2, 10^{-16}))$.
    pub log10_separation: f64,
}

impl Default for TwinFrame {
    fn default() -> Self {
        Self::from_states(0, 0.0, [1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 0.0], false)
    }
}

impl TwinFrame {
    /// Builds a `TwinFrame` and computes finite Euclidean and $\log_{10}$ separation.
    #[must_use]
    pub fn from_states(
        step: usize,
        time: f64,
        primary: [f64; 4],
        shadow: [f64; 4],
        has_shadow: bool,
    ) -> Self {
        let separation = if has_shadow {
            let mut sq = 0.0;
            for (&s, &p) in shadow.iter().zip(&primary) {
                let d = s - p;
                sq += d * d;
            }
            sq.sqrt()
        } else {
            0.0
        };
        let safe_sep = if separation.is_finite() {
            separation.max(MIN_LOG_SEPARATION)
        } else {
            MIN_LOG_SEPARATION
        };
        Self {
            step,
            time,
            primary,
            shadow,
            separation,
            log10_separation: safe_sep.log10(),
        }
    }

    /// Returns the 3D phase-space point for the primary trajectory.
    #[must_use]
    pub fn primary_vec3(&self) -> Vector3<f64> {
        Vector3::new(self.primary[0], self.primary[1], self.primary[2])
    }

    /// Returns the 3D phase-space point for the shadow trajectory.
    #[must_use]
    pub fn shadow_vec3(&self) -> Vector3<f64> {
        Vector3::new(self.shadow[0], self.shadow[1], self.shadow[2])
    }
}

/// Evaluates `bar`'s parsed AST expression once over `(x0, y0, z0, epsilon)` outside stepping loops.
pub fn evaluate_perturbation_bar(
    bar: &MathExpressionInputBar,
    initial_state: [f64; 3],
    epsilon: f64,
) -> Result<[f64; 3], String> {
    let expr = bar.parsed_expr.as_ref().map_err(|err| {
        format!(
            "Syntax Error: {}",
            err.user_message_for_input(bar.raw_text.trim())
        )
    })?;
    let mut vars = HashMap::with_capacity(5);
    vars.insert("x0".to_string(), initial_state[0]);
    vars.insert("y0".to_string(), initial_state[1]);
    vars.insert("z0".to_string(), initial_state[2]);
    vars.insert("epsilon".to_string(), epsilon);
    vars.insert("eps".to_string(), epsilon);

    let delta = expr
        .eval(&vars)
        .map_err(|e| format!("Evaluation Error: {e}"))?;
    if !delta.is_finite() {
        return Err("Evaluation Error: perturbation formula evaluated to non-finite value".into());
    }
    let axis = match bar.lhs_name.as_deref() {
        Some(lhs) if lhs.eq_ignore_ascii_case("dy0") || lhs.eq_ignore_ascii_case("y0") => 1,
        Some(lhs) if lhs.eq_ignore_ascii_case("dz0") || lhs.eq_ignore_ascii_case("z0") => 2,
        _ => 0,
    };
    let mut offset = [0.0; 3];
    offset[axis] = delta;
    Ok(offset)
}

/// Cached Wolf Lyapunov exponent, windowed FTLE, and on-demand correlation dimension metrics.
#[derive(Debug, Clone)]
pub struct ChaosMetricsCache {
    /// Largest Lyapunov exponent $\lambda_1$ from Wolf's renormalization method.
    pub wolf_lambda1: f64,
    /// Whether `wolf_lambda1` needs lazy recomputation on next display/query.
    pub wolf_dirty: bool,
    /// Windowed finite-time Lyapunov exponent across the active timeline cursor.
    pub windowed_ftle: f64,
    /// Neighborhood radius $r$ for on-demand `correlation_dimension`.
    pub correlation_radius: f64,
    /// On-demand `(radius_r, C(r))` result computed outside locked stepping loops.
    pub correlation_result: Option<(f64, f64)>,
}

impl Default for ChaosMetricsCache {
    fn default() -> Self {
        Self {
            wolf_lambda1: 0.9056,
            wolf_dirty: true,
            windowed_ftle: 0.0,
            correlation_radius: AttractorSystemKind::Lorenz.default_correlation_radius(),
            correlation_result: None,
        }
    }
}

impl ChaosMetricsCache {
    /// Returns Lyapunov e-folding time $T_\lambda = 1 / \lambda_1$ when $\lambda_1 > 0$.
    #[must_use]
    pub fn lyapunov_time(&self) -> Option<f64> {
        (self.wolf_lambda1.is_finite() && self.wolf_lambda1 > 1e-6).then(|| 1.0 / self.wolf_lambda1)
    }

    /// Recomputes Wolf $\lambda_1$ lazily when `wolf_dirty` is set.
    pub fn refresh_wolf_lyapunov(
        &mut self,
        kind: AttractorSystemKind,
        sigma_rho_beta: (f64, f64, f64),
        extra: &ExtraChaosParams,
        initial_state: [f64; 3],
        dt: f64,
    ) {
        if !self.wolf_dirty {
            return;
        }
        let safe_dt = if dt.is_finite() && dt > 0.0 {
            dt.clamp(0.001, 0.02)
        } else {
            kind.default_dt()
        };
        self.wolf_lambda1 = match kind {
            AttractorSystemKind::Lorenz => {
                compute_lorenz_wolf(sigma_rho_beta, initial_state, safe_dt)
            }
            AttractorSystemKind::Rossler => compute_rossler_wolf(extra, initial_state, safe_dt),
            AttractorSystemKind::DoublePendulum => {
                compute_pendulum_wolf(extra, initial_state, safe_dt)
            }
        };
        self.wolf_dirty = false;
    }

    /// Updates `windowed_ftle` from the active `TimelineRingBuffer` cursor without allocating.
    pub fn update_windowed_ftle(&mut self, timeline: &TimelineRingBuffer<TwinFrame>) {
        let Some(curr) = timeline.current() else {
            self.windowed_ftle = 0.0;
            return;
        };
        let cur_idx = timeline.cursor();
        if cur_idx == 0 {
            self.windowed_ftle = 0.0;
            return;
        }
        let start_idx = if curr.separation < 0.5 {
            0
        } else {
            cur_idx.saturating_sub(150)
        };
        if let Some(prev) = timeline.get(start_idx) {
            let dt = curr.time - prev.time;
            self.windowed_ftle = finite_time_lyapunov(prev.separation, curr.separation, dt);
        }
    }

    /// Computes Grassberger-Procaccia correlation sum $C(r)$ on demand over active 3D points.
    pub fn compute_correlation_on_demand(&mut self, points: &[Vector3<f64>]) -> f64 {
        let r = self.correlation_radius.clamp(0.05, 50.0);
        let c_r = correlation_dimension(points, r);
        self.correlation_result = Some((r, c_r));
        c_r
    }
}

fn compute_lorenz_wolf((sigma, rho, beta): (f64, f64, f64), initial: [f64; 3], dt: f64) -> f64 {
    if !sigma.is_finite() || !rho.is_finite() || !beta.is_finite() {
        return 0.0;
    }
    let sys = LorenzBuilder::new()
        .sigma(sigma)
        .rho(rho)
        .beta(beta)
        .dt(dt)
        .build(LorenzState::new(initial[0], initial[1], initial[2]));
    let mut rk4 = RungeKutta4::new(&Vector3::zeros());
    let mut start = Vector3::new(initial[0], initial[1], initial[2]);
    for _ in 0..250 {
        rk4.step(&sys, 0.0, &mut start, dt);
    }
    if !start.x.is_finite() || !start.y.is_finite() || !start.z.is_finite() {
        return 0.0;
    }
    lorenz_lyapunov(&sys, &mut rk4, start, dt, 180, (25.0 * dt).max(dt * 2.0)).unwrap_or(0.0)
}

fn compute_rossler_wolf(extra: &ExtraChaosParams, initial: [f64; 3], dt: f64) -> f64 {
    if !extra.is_valid() {
        return 0.0;
    }
    let sys = RosslerBuilder::new()
        .a(extra.rossler_a)
        .b(extra.rossler_b)
        .c(extra.rossler_c)
        .dt(dt)
        .build(RosslerState::new(initial[0], initial[1], initial[2]));
    let mut rk4 = RungeKutta4::new(&Vector3::zeros());
    let mut start = Vector3::new(initial[0], initial[1], initial[2]);
    for _ in 0..200 {
        rk4.step(&sys, 0.0, &mut start, dt);
    }
    if !start.x.is_finite() || !start.y.is_finite() || !start.z.is_finite() {
        return 0.0;
    }
    lorenz_lyapunov(&sys, &mut rk4, start, dt, 120, (40.0 * dt).max(dt * 2.0)).unwrap_or(0.0)
}

fn compute_pendulum_wolf(extra: &ExtraChaosParams, initial: [f64; 3], dt: f64) -> f64 {
    if !extra.is_valid() {
        return 0.0;
    }
    let state = DoublePendulumState::new(initial[0], initial[1], initial[2], 0.0);
    let sys = DoublePendulumBuilder::new()
        .m1(extra.pendulum_m1)
        .m2(extra.pendulum_m2)
        .l1(extra.pendulum_l1)
        .l2(extra.pendulum_l2)
        .g(extra.pendulum_g)
        .dt(dt)
        .build(state);
    let mut rk4 = RungeKutta4::new(&ArrayState([0.0; 4]));
    double_pendulum_lyapunov(
        &sys,
        &mut rk4,
        state.state,
        dt,
        60,
        (20.0 * dt).max(dt * 2.0),
    )
    .unwrap_or(0.0)
}
