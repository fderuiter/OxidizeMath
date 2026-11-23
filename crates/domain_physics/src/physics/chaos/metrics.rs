//! Quantifying Chaos (Lyapunov Exponents)

use crate::error::ChaosError;
use nalgebra::Vector3;
use pure_math::pure_math::analysis::ode::{OdeSystem, Solver, SolverExt};

const MIN_LYAPUNOV_DERIVATIVE: f64 = 1e-15;
const INITIAL_SHADOW_SEPARATION: f64 = 1e-8;
const MIN_TRAJECTORY_SEPARATION: f64 = 1e-30;

/// Calculates the Lyapunov Exponent for the Logistic Map.
///
/// The Lyapunov exponent $\lambda$ measures the average rate of divergence of nearby trajectories.
/// - $\lambda > 0$: Chaotic.
/// - $\lambda \le 0$: Regular (stable fixed point or limit cycle).
///
/// Formula: $\lambda \approx \frac{1}{n} \sum_{i=0}^{n-1} \ln | f'(x_i) |$
/// where $f'(x) = r(1 - 2x)$.
#[verified_engine::verified]
pub fn logistic_lyapunov(r: f64, x0: f64, n: usize) -> f64 {
    if n == 0 || !r.is_finite() || !x0.is_finite() {
        return 0.0;
    }

    let mut x = x0;
    let mut sum_logs = 0.0;
    let mut prod = 1.0_f64;
    let mut valid_steps = 0_usize;

    for _ in 0..n {
        if !x.is_finite() {
            break;
        }
        // Clamp zero-derivative singularities (e.g. x = 0.5 or r = 0.0) so outputs stay finite.
        let derivative = (r * (1.0 - 2.0 * x)).abs().max(MIN_LYAPUNOV_DERIVATIVE);
        if !derivative.is_finite() {
            break;
        }
        prod *= derivative;
        if !(1e-100..=1e100).contains(&prod) {
            sum_logs += prod.ln();
            prod = 1.0;
        }
        valid_steps += 1;
        x = r * x * (1.0 - x);
    }

    if valid_steps == 0 {
        0.0
    } else {
        if prod != 1.0 {
            sum_logs += prod.ln();
        }
        sum_logs / valid_steps as f64
    }
}

#[verified_engine::verified]
fn evolve_and_renormalize_pair<S, Sol>(
    system: &S,
    solver: &mut Sol,
    current_state: &mut Vector3<f64>,
    shadow_state: &mut Vector3<f64>,
    steps_per_iter: usize,
    time_step: f64,
) -> Option<f64>
where
    S: OdeSystem<Vector3<f64>> + ?Sized,
    Sol: Solver<Vector3<f64>>,
{
    for _ in 0..steps_per_iter {
        *current_state = solver.solve(system, 0.0, current_state, time_step);
        *shadow_state = solver.solve(system, 0.0, shadow_state, time_step);
        if !current_state.x.is_finite()
            || !current_state.y.is_finite()
            || !current_state.z.is_finite()
            || !shadow_state.x.is_finite()
            || !shadow_state.y.is_finite()
            || !shadow_state.z.is_finite()
        {
            return None;
        }
    }

    let dist_vec = *shadow_state - *current_state;
    let d_t = dist_vec.norm();
    if !d_t.is_finite() {
        return None;
    }

    let unit_dir = if d_t > MIN_TRAJECTORY_SEPARATION {
        dist_vec / d_t
    } else {
        Vector3::new(1.0, 0.0, 0.0)
    };
    *shadow_state = *current_state + unit_dir * INITIAL_SHADOW_SEPARATION;
    Some(d_t.max(MIN_TRAJECTORY_SEPARATION))
}

/// Calculates the largest Lyapunov Exponent for a 3D continuous dynamical system.
///
/// This uses Wolf's algorithm, tracking two trajectories separated by a tiny distance
/// and periodically renormalizing the shadow trajectory.
///
/// # Arguments
/// * `system` - The dynamical system implementing `OdeSystem<Vector3<f64>>`.
/// * `initial_state` - The starting state for the main trajectory.
/// * `time_step` - The simulation time step `dt`.
/// * `iterations` - Number of renormalization steps to perform.
/// * `evolution_time` - Time to evolve before measuring and renormalizing.
///
/// # Returns
/// Estimated largest Lyapunov exponent.
#[verified_engine::verified]
pub fn lorenz_lyapunov<S, Sol>(
    system: &S,
    solver: &mut Sol,
    initial_state: Vector3<f64>,
    time_step: f64,
    iterations: usize,
    evolution_time: f64,
) -> Result<f64, ChaosError>
where
    S: OdeSystem<Vector3<f64>> + ?Sized,
    Sol: Solver<Vector3<f64>>,
{
    if !time_step.is_finite()
        || time_step <= 0.0
        || !evolution_time.is_finite()
        || evolution_time <= 0.0
    {
        return Err(ChaosError::InvalidParameter(
            "time_step and evolution_time must be finite and positive".to_string(),
        ));
    }
    if iterations == 0
        || !initial_state.x.is_finite()
        || !initial_state.y.is_finite()
        || !initial_state.z.is_finite()
    {
        return Ok(0.0);
    }

    let effective_evolution = evolution_time.max(time_step);
    let steps_per_iter = (effective_evolution / time_step).round().max(1.0) as usize;
    let step_duration = (steps_per_iter as f64) * time_step;

    let mut current_state = initial_state;
    let mut shadow_state = current_state + Vector3::new(INITIAL_SHADOW_SEPARATION, 0.0, 0.0);

    let warmup_iters = if iterations >= 4 {
        ((4.0 / step_duration).ceil() as usize).clamp(1, iterations.min(12))
    } else {
        0
    };
    for _ in 0..warmup_iters {
        if evolve_and_renormalize_pair(
            system,
            solver,
            &mut current_state,
            &mut shadow_state,
            steps_per_iter,
            time_step,
        )
        .is_none()
        {
            return Ok(0.0);
        }
    }

    let mut sum_log_divergence = 0.0;
    let mut total_time = 0.0;

    for _ in 0..iterations {
        let Some(clamped_d) = evolve_and_renormalize_pair(
            system,
            solver,
            &mut current_state,
            &mut shadow_state,
            steps_per_iter,
            time_step,
        ) else {
            break;
        };
        sum_log_divergence += (clamped_d / INITIAL_SHADOW_SEPARATION).ln();
        total_time += step_duration;
    }

    Ok(if total_time > 0.0 {
        sum_log_divergence / total_time
    } else {
        0.0
    })
}
