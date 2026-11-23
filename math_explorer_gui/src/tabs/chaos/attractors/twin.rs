//! Twin trajectory perturbation state, zero-allocation lockstep RK4 stepping, and 3D scene helpers.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::flow::{AttractorMode, CompiledFlowJob};
use crate::framework::{Camera3D, Scene3DBuilder};
use eframe::egui;
use egui_plot::{LineStyle, MarkerShape};
use math_explorer::physics::chaos::lorenz::{LorenzBuilder, LorenzState};
use nalgebra::{Point3, Vector3};
use std::collections::VecDeque;

/// Default initial state coordinates `(x0, y0, z0)` for [`super::AttractorPlotter`].
pub const DEFAULT_INITIAL_STATE: [f64; 3] = [1.0, 1.0, 1.0];

/// Minimum allowed initial coordinate value along each axis.
pub const MIN_INITIAL_COORD: f64 = -100.0;

/// Maximum allowed initial coordinate value along each axis.
pub const MAX_INITIAL_COORD: f64 = 100.0;

/// Default twin perturbation magnitude `epsilon` (`1e-9`) along `x0`.
pub const DEFAULT_TWIN_EPSILON: f64 = 1e-9;

/// Minimum allowed twin perturbation magnitude `epsilon` (`1e-12`).
pub const MIN_TWIN_EPSILON: f64 = 1e-12;

/// Maximum allowed twin perturbation magnitude `epsilon` (`1e-1`).
pub const MAX_TWIN_EPSILON: f64 = 1e-1;

/// Numerical floor applied before `log10` so identical states remain finite.
pub const MIN_SEPARATION_FLOOR: f64 = 1e-16;

/// Minimum capacity of `divergence_history` so >= 28.48 time units at `dt = 0.01` stay retained.
pub const MIN_DIVERGENCE_HISTORY_CAP: usize = 4096;

/// User-configurable initial condition `(x0, y0, z0)` and twin perturbation settings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TwinConfig {
    /// Primary initial position `(x0, y0, z0)`.
    pub initial_state: Vector3<f64>,
    /// Whether the perturbed shadow trajectory `(x0 + epsilon, y0, z0)` is active.
    pub enabled: bool,
    /// Initial perturbation offset `epsilon` added to `x0` in `[1e-12, 1e-1]`.
    pub epsilon: f64,
}

impl Default for TwinConfig {
    fn default() -> Self {
        Self {
            initial_state: Vector3::new(
                DEFAULT_INITIAL_STATE[0],
                DEFAULT_INITIAL_STATE[1],
                DEFAULT_INITIAL_STATE[2],
            ),
            enabled: false,
            epsilon: DEFAULT_TWIN_EPSILON,
        }
    }
}

impl TwinConfig {
    /// Returns `true` when `initial_state` and `epsilon` are finite and within valid bounds.
    #[must_use]
    pub fn is_finite(&self) -> bool {
        self.initial_state.x.is_finite()
            && self.initial_state.y.is_finite()
            && self.initial_state.z.is_finite()
            && self.epsilon.is_finite()
            && self.epsilon > 0.0
    }

    /// Clamps or replaces non-finite initial coordinates and `epsilon` with safe bounded defaults.
    #[must_use]
    pub fn sanitized(&self) -> Self {
        let sanitize_coord = |v: f64, fallback: f64| {
            if v.is_finite() {
                v.clamp(MIN_INITIAL_COORD, MAX_INITIAL_COORD)
            } else {
                fallback
            }
        };
        let eps = if self.epsilon.is_finite() && self.epsilon > 0.0 {
            self.epsilon.clamp(MIN_TWIN_EPSILON, MAX_TWIN_EPSILON)
        } else {
            DEFAULT_TWIN_EPSILON
        };
        Self {
            initial_state: Vector3::new(
                sanitize_coord(self.initial_state.x, DEFAULT_INITIAL_STATE[0]),
                sanitize_coord(self.initial_state.y, DEFAULT_INITIAL_STATE[1]),
                sanitize_coord(self.initial_state.z, DEFAULT_INITIAL_STATE[2]),
            ),
            enabled: self.enabled,
            epsilon: eps,
        }
    }

    /// Initial shadow state `(x0 + epsilon, y0, z0)`.
    #[inline]
    #[must_use]
    pub fn shadow_initial_state(&self) -> Vector3<f64> {
        Vector3::new(
            self.initial_state.x + self.epsilon,
            self.initial_state.y,
            self.initial_state.z,
        )
    }

    /// Returns initial coordinates as a 3-element array `[x0, y0, z0]`.
    #[inline]
    #[must_use]
    pub fn initial_array(&self) -> [f64; 3] {
        [
            self.initial_state.x,
            self.initial_state.y,
            self.initial_state.z,
        ]
    }
}

/// Returns `true` if `coords` matches [`DEFAULT_INITIAL_STATE`] within `f64::EPSILON`.
#[must_use]
pub fn is_default_initial_state(coords: &[f64; 3]) -> bool {
    (0..3).all(|i| (coords[i] - DEFAULT_INITIAL_STATE[i]).abs() <= f64::EPSILON)
}

/// Returns `true` if `eps` matches [`DEFAULT_TWIN_EPSILON`] within relative tolerance.
#[must_use]
pub fn is_default_twin_epsilon(eps: &f64) -> bool {
    (*eps - DEFAULT_TWIN_EPSILON).abs() <= 1e-21
}

/// Maximum number of `[t, log10_dist]` samples retained for a given `max_points` setting.
#[inline]
#[must_use]
pub fn divergence_capacity(max_points: usize) -> usize {
    max_points.max(MIN_DIVERGENCE_HISTORY_CAP)
}

/// Computes `log10(||shadow - primary||)` clamped above [`MIN_SEPARATION_FLOOR`].
#[inline]
#[must_use]
pub fn log10_separation(primary: Vector3<f64>, shadow: Vector3<f64>) -> f64 {
    let dist = (shadow - primary).norm();
    if dist.is_finite() {
        dist.max(MIN_SEPARATION_FLOOR).log10()
    } else {
        f64::NAN
    }
}

/// Advances a single 3D state vector by one RK4 step without heap allocations.
#[inline]
#[must_use]
pub fn step_flow_vec(
    mode: AttractorMode,
    sigma: f64,
    rho: f64,
    beta: f64,
    dt: f64,
    custom_flow: Option<&CompiledFlowJob>,
    vec: Vector3<f64>,
) -> Vector3<f64> {
    match mode {
        AttractorMode::Lorenz => {
            let mut sys = LorenzBuilder::new()
                .sigma(sigma)
                .rho(rho)
                .beta(beta)
                .build(LorenzState { vec });
            sys.step(dt);
            sys.state.vec
        }
        AttractorMode::Custom => match custom_flow {
            Some(job) => job.rk4_step(vec, dt),
            None => vec,
        },
    }
}

/// Evaluates instantaneous 3D velocity `[dx, dy, dz]` for sonification telemetry.
#[inline]
#[must_use]
pub(super) fn eval_flow_rhs(
    mode: AttractorMode,
    sigma: f64,
    rho: f64,
    beta: f64,
    custom_flow: Option<&CompiledFlowJob>,
    v: Vector3<f64>,
) -> [f64; 3] {
    match (mode, custom_flow) {
        (AttractorMode::Custom, Some(job)) => {
            let rhs = job.eval_rhs(v);
            [rhs.x, rhs.y, rhs.z]
        }
        _ => [
            sigma * (v.y - v.x),
            v.x * (rho - v.z) - v.y,
            v.x * v.y - beta * v.z,
        ],
    }
}

/// Advances `(primary, shadow)` in lockstep by one RK4 step with zero heap allocations.
#[inline]
#[must_use]
pub fn step_twin_pair(
    mode: AttractorMode,
    lorenz: [f64; 3],
    dt: f64,
    custom_flow: Option<&CompiledFlowJob>,
    primary: Vector3<f64>,
    shadow: Vector3<f64>,
) -> (Vector3<f64>, Vector3<f64>) {
    let [sigma, rho, beta] = lorenz;
    let next_primary = step_flow_vec(mode, sigma, rho, beta, dt, custom_flow, primary);
    let next_shadow = step_flow_vec(mode, sigma, rho, beta, dt, custom_flow, shadow);
    (next_primary, next_shadow)
}

/// Mutable trajectory buffers for the perturbed shadow trajectory and logarithmic separation curve.
#[derive(Debug, Clone)]
pub struct TwinRunnerState {
    /// Current 3D position of the shadow trajectory.
    pub shadow_vec: Vector3<f64>,
    /// Bounded 3D polyline history of the shadow trajectory.
    pub twin_history: VecDeque<Vector3<f64>>,
    /// Bounded `[time_t, log10(||delta_x(t)||)]` history.
    pub divergence_history: VecDeque<[f64; 2]>,
    /// Whether twin integration was active for the current run.
    pub enabled: bool,
}

impl Default for TwinRunnerState {
    fn default() -> Self {
        Self::from_config(TwinConfig::default(), 2000)
    }
}

impl TwinRunnerState {
    /// Initializes shadow state and divergence seed from `cfg`.
    #[must_use]
    pub fn from_config(cfg: TwinConfig, max_points: usize) -> Self {
        let clean = cfg.sanitized();
        let shadow_vec = clean.shadow_initial_state();
        let mut divergence_history = VecDeque::with_capacity(if clean.enabled {
            divergence_capacity(max_points).min(4096)
        } else {
            0
        });
        if clean.enabled {
            divergence_history.push_back([0.0, log10_separation(clean.initial_state, shadow_vec)]);
        }
        Self {
            shadow_vec,
            twin_history: VecDeque::with_capacity(if clean.enabled { max_points } else { 0 }),
            divergence_history,
            enabled: clean.enabled,
        }
    }

    /// Resets shadow buffers to `cfg`'s initial perturbation state at `t = 0`.
    pub fn reset(&mut self, cfg: TwinConfig, _max_points: usize) {
        let clean = cfg.sanitized();
        self.shadow_vec = clean.shadow_initial_state();
        self.enabled = clean.enabled;
        self.twin_history.clear();
        self.divergence_history.clear();
        if clean.enabled {
            self.divergence_history
                .push_back([0.0, log10_separation(clean.initial_state, self.shadow_vec)]);
        }
    }

    /// Appends one stepped shadow state and `log10(||delta_x(t)||)` sample.
    pub fn record_step(
        &mut self,
        step_count: usize,
        dt: f64,
        max_points: usize,
        primary: Vector3<f64>,
        shadow: Vector3<f64>,
    ) {
        self.shadow_vec = shadow;
        if !self.enabled {
            return;
        }
        if self.twin_history.len() >= max_points {
            self.twin_history.pop_front();
        }
        self.twin_history.push_back(shadow);

        let div_cap = divergence_capacity(max_points);
        if self.divergence_history.len() >= div_cap {
            self.divergence_history.pop_front();
        }
        let log_d = log10_separation(primary, shadow);
        if log_d.is_finite() {
            self.divergence_history
                .push_back([step_count as f64 * dt, log_d]);
        }
    }
    /// Rebuilds primary `history`, `twin_history`, and `divergence_history` from retained keyframes.
    pub fn rebuild_histories(
        &mut self,
        history: &mut VecDeque<Vector3<f64>>,
        front: &[crate::async_sim::TimeTravelCheckpoint],
        back: &[crate::async_sim::TimeTravelCheckpoint],
    ) {
        let last_params = back
            .last()
            .or_else(|| front.last())
            .and_then(|cp| cp.params.as_scalars());
        let dt = last_params.and_then(|p| p.get(3).copied()).unwrap_or(0.01);
        let max_points = last_params
            .and_then(|p| p.get(4).copied())
            .map_or(2000usize, |v| (v as usize).max(1));
        history.clear();
        self.twin_history.clear();
        self.divergence_history.clear();

        let total_post_zero = front.iter().chain(back).filter(|cp| cp.step >= 1).count();
        let skip_traj = total_post_zero.saturating_sub(max_points);
        for cp in front
            .iter()
            .chain(back)
            .filter(|cp| cp.step >= 1)
            .skip(skip_traj)
        {
            if let Some(s) = cp.state.as_scalars().filter(|s| s.len() >= 3) {
                history.push_back(Vector3::new(s[0], s[1], s[2]));
                if s.len() >= 7 && s[6] > 0.5 {
                    self.twin_history.push_back(Vector3::new(s[3], s[4], s[5]));
                }
            }
        }

        if self.enabled {
            let div_cap = divergence_capacity(max_points);
            let total_all = front.len() + back.len();
            let skip_div = total_all.saturating_sub(div_cap);
            for cp in front.iter().chain(back).skip(skip_div) {
                if let Some(s) = cp.state.as_scalars().filter(|s| s.len() >= 7 && s[6] > 0.5) {
                    let v = Vector3::new(s[0], s[1], s[2]);
                    let sv = Vector3::new(s[3], s[4], s[5]);
                    let log_d = log10_separation(v, sv);
                    if log_d.is_finite() {
                        self.divergence_history
                            .push_back([cp.step as f64 * dt, log_d]);
                    }
                }
            }
        }
    }
}
pub(super) struct ScenePopulateArgs<'a> {
    pub mode: AttractorMode,
    pub camera: Camera3D,
    pub is_finite: bool,
    pub primary_vec: Vector3<f64>,
    pub history: &'a VecDeque<Vector3<f64>>,
    pub twin_enabled: bool,
    pub shadow_vec: Vector3<f64>,
    pub twin_history: &'a VecDeque<Vector3<f64>>,
    pub primary_color: egui::Color32,
    pub twin_color: egui::Color32,
}

pub(super) fn populate_attractor_scene(
    scene_builder: &mut Scene3DBuilder,
    primary_scratch: &mut Vec<Point3<f64>>,
    twin_scratch: &mut Vec<Point3<f64>>,
    args: ScenePopulateArgs<'_>,
) {
    let origin_offset = match args.mode {
        AttractorMode::Lorenz => Point3::new(0.0, 0.0, 25.0),
        AttractorMode::Custom => Point3::origin(),
    };
    scene_builder.begin(args.camera);
    scene_builder.set_origin_offset(origin_offset);

    add_trajectory_and_head(
        scene_builder,
        primary_scratch,
        args.history,
        args.primary_vec,
        args.is_finite,
        ("Trajectory", "Initial State", "Primary Head"),
        (args.primary_color, MarkerShape::Circle, 4.5_f32),
    );

    if args.twin_enabled {
        add_trajectory_and_head(
            scene_builder,
            twin_scratch,
            args.twin_history,
            args.shadow_vec,
            args.is_finite && is_vec_finite(args.shadow_vec),
            ("Twin Trajectory", "Twin Initial State", "Twin Head"),
            (args.twin_color, MarkerShape::Diamond, 5.0_f32),
        );
    }
}

#[inline]
#[must_use]
pub(super) fn is_vec_finite(v: Vector3<f64>) -> bool {
    v.x.is_finite() && v.y.is_finite() && v.z.is_finite()
}

fn add_trajectory_and_head(
    scene_builder: &mut Scene3DBuilder,
    scratch: &mut Vec<Point3<f64>>,
    history: &VecDeque<Vector3<f64>>,
    head_vec: Vector3<f64>,
    is_finite: bool,
    labels: (&'static str, &'static str, &'static str),
    style: (egui::Color32, MarkerShape, f32),
) {
    let (traj_label, init_label, head_label) = labels;
    let (color, marker_shape, head_radius) = style;
    scratch.clear();
    scratch.extend(history.iter().map(|v| Point3::new(v.x, v.y, v.z)));
    if !scratch.is_empty() {
        if scratch.len() == 1 {
            scene_builder.add_points(traj_label, scratch, head_radius, color, marker_shape, true);
        } else {
            scene_builder.add_polyline(
                traj_label,
                scratch,
                egui::Stroke::new(1.5_f32, color),
                LineStyle::Solid,
                true,
            );
            if is_vec_finite(head_vec) {
                scene_builder.add_points(
                    head_label,
                    &[Point3::new(head_vec.x, head_vec.y, head_vec.z)],
                    head_radius,
                    color,
                    marker_shape,
                    false,
                );
            }
        }
    } else if is_finite && is_vec_finite(head_vec) {
        scene_builder.add_points(
            init_label,
            &[Point3::new(head_vec.x, head_vec.y, head_vec.z)],
            4.0_f32,
            color,
            marker_shape,
            true,
        );
    }
}
