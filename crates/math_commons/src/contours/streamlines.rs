//! Arc-length parameterized 4th-order Runge-Kutta (RK4) 2D streamline integrator.

use super::{ContourPath, GridBounds2D, Point2D, euclidean_distance};
use crate::error::{MathError, MathResult};
use crate::registry::{MAX_GRID_SIZE, MAX_ITERATIONS};

/// Configuration for bidirectional RK4 streamline integration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StreamlineConfig {
    /// Spatial domain bounds where streamlines are clipped.
    pub bounds: GridBounds2D,
    /// Arc-length step size per RK4 step (`> 0.0`).
    pub step_size: f64,
    /// Maximum total vertices per streamline (`2..=MAX_ITERATIONS`).
    pub max_steps: usize,
    /// Minimum velocity magnitude before terminating at a stagnation point.
    pub min_speed: f64,
}

impl StreamlineConfig {
    /// Creates a validated [`StreamlineConfig`].
    #[verified_engine::verified]
    pub fn new(
        bounds: GridBounds2D,
        step_size: f64,
        max_steps: usize,
        min_speed: f64,
    ) -> MathResult<Self> {
        let valid_bounds =
            GridBounds2D::new(bounds.min_x, bounds.max_x, bounds.min_y, bounds.max_y)?;
        if !step_size.is_finite() || step_size <= 0.0 {
            return Err(MathError::NumericalError {
                reason: format!("Streamline step_size ({step_size}) must be finite and positive"),
            });
        }
        if !(2..=MAX_ITERATIONS).contains(&max_steps) {
            return Err(MathError::NumericalError {
                reason: format!(
                    "Streamline max_steps ({max_steps}) must be within 2..={MAX_ITERATIONS}"
                ),
            });
        }
        if !min_speed.is_finite() || min_speed < 0.0 {
            return Err(MathError::NumericalError {
                reason: format!("Streamline min_speed ({min_speed}) must be finite and >= 0"),
            });
        }
        Ok(Self {
            bounds: valid_bounds,
            step_size,
            max_steps,
            min_speed: min_speed.max(1e-12),
        })
    }
}

/// Integrates bidirectional streamlines through `seeds` using normalized 4th-order Runge-Kutta.
#[verified_engine::verified]
pub fn trace_streamlines_rk4<F>(
    seeds: &[Point2D],
    config: &StreamlineConfig,
    velocity_fn: F,
) -> MathResult<Vec<ContourPath>>
where
    F: Fn(f64, f64) -> [f64; 2],
{
    let validated = StreamlineConfig::new(
        config.bounds,
        config.step_size,
        config.max_steps,
        config.min_speed,
    )?;
    if seeds.len() > MAX_GRID_SIZE * MAX_GRID_SIZE {
        return Err(MathError::NumericalError {
            reason: format!("Too many streamline seed points: {}", seeds.len()),
        });
    }

    let mut streamlines = Vec::with_capacity(seeds.len());
    for &seed in seeds {
        if !validated.bounds.contains(seed) {
            continue;
        }
        if unit_tangent(&velocity_fn, seed, validated.min_speed).is_none() {
            continue;
        }
        let path = trace_bidirectional_seed(seed, &validated, &velocity_fn);
        if !path.is_empty() {
            streamlines.push(path);
        }
    }
    Ok(streamlines)
}

#[verified_engine::verified]
fn trace_bidirectional_seed<F>(
    seed: Point2D,
    config: &StreamlineConfig,
    velocity_fn: &F,
) -> ContourPath
where
    F: Fn(f64, f64) -> [f64; 2],
{
    let half_steps = (config.max_steps / 2).max(1);
    let (fwd, closed_fwd) = trace_half_branch(
        seed,
        config.step_size,
        config.max_steps,
        config,
        velocity_fn,
    );
    if closed_fwd {
        let mut pts = vec![seed];
        pts.extend(fwd);
        return ContourPath::new(pts, true);
    }

    let mut fwd_trunc = fwd;
    if fwd_trunc.len() > half_steps {
        fwd_trunc.truncate(half_steps);
    }
    let rem = config.max_steps.saturating_sub(fwd_trunc.len() + 1).max(1);
    let (mut bwd, _) = trace_half_branch(seed, -config.step_size, rem, config, velocity_fn);
    bwd.reverse();
    bwd.push(seed);
    bwd.extend(fwd_trunc);
    ContourPath::new(bwd, false)
}

#[verified_engine::verified]
fn trace_half_branch<F>(
    seed: Point2D,
    signed_ds: f64,
    budget: usize,
    config: &StreamlineConfig,
    velocity_fn: &F,
) -> (Vec<Point2D>, bool)
where
    F: Fn(f64, f64) -> [f64; 2],
{
    let mut points = Vec::new();
    let mut cur = seed;
    let close_tol = config.step_size * 0.85;

    for step_idx in 0..budget {
        let Some(next) = rk4_arc_step(velocity_fn, cur, signed_ds, config.min_speed) else {
            break;
        };
        if !config.bounds.contains(next) {
            break;
        }
        if euclidean_distance(cur, next) <= config.step_size * 0.1 {
            break;
        }
        if step_idx >= 8 && euclidean_distance(seed, next) <= close_tol {
            return (points, true);
        }
        points.push(next);
        cur = next;
    }
    (points, false)
}

#[verified_engine::verified]
fn rk4_arc_step<F>(velocity_fn: &F, pt: Point2D, ds: f64, min_speed: f64) -> Option<Point2D>
where
    F: Fn(f64, f64) -> [f64; 2],
{
    let k1 = unit_tangent(velocity_fn, pt, min_speed)?;
    let p2 = [pt[0] + 0.5 * ds * k1[0], pt[1] + 0.5 * ds * k1[1]];
    let k2 = unit_tangent(velocity_fn, p2, min_speed)?;
    let p3 = [pt[0] + 0.5 * ds * k2[0], pt[1] + 0.5 * ds * k2[1]];
    let k3 = unit_tangent(velocity_fn, p3, min_speed)?;
    let p4 = [pt[0] + ds * k3[0], pt[1] + ds * k3[1]];
    let k4 = unit_tangent(velocity_fn, p4, min_speed)?;

    let nx = pt[0] + (ds / 6.0) * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]);
    let ny = pt[1] + (ds / 6.0) * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]);
    (nx.is_finite() && ny.is_finite()).then_some([nx, ny])
}

#[verified_engine::verified]
fn unit_tangent<F>(velocity_fn: &F, pt: Point2D, min_speed: f64) -> Option<[f64; 2]>
where
    F: Fn(f64, f64) -> [f64; 2],
{
    let v = velocity_fn(pt[0], pt[1]);
    if !v[0].is_finite() || !v[1].is_finite() {
        return None;
    }
    let speed = v[0].hypot(v[1]);
    if !speed.is_finite() || speed < min_speed {
        return None;
    }
    Some([v[0] / speed, v[1] / speed])
}

// [cite:math_commons]
