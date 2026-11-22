//! Quantifying Chaos (Lyapunov Exponents)

use super::logistic::LogisticMap;
use crate::error::ChaosError;
use nalgebra::Vector3;
use pure_math::pure_math::analysis::ode::{ArrayState, OdeSystem, Solver, SolverExt};

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
    let mut map = LogisticMap::new(r, x0);
    let mut sum_logs = 0.0;

    for _ in 0..n {
        let x = map.state;
        let derivative = r * (1.0 - 2.0 * x);
        // Handle singularity at x = 0.5 (f'(x) = 0).
        sum_logs += derivative.abs().ln();
        map.next();
    }

    sum_logs / n as f64
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
    let d0 = 1e-8;
    let steps_per_iter = (evolution_time / time_step).round() as usize;

    if steps_per_iter == 0 {
        return Err(ChaosError::InvalidParameter(
            "evolution_time must be greater than time_step".to_string(),
        ));
    }

    let mut current_state = initial_state;
    let mut shadow_state = current_state + Vector3::new(d0, 0.0, 0.0);

    // Normalize initial separation
    let initial_diff = shadow_state - current_state;
    shadow_state = current_state + initial_diff.normalize() * d0;

    let mut sum_log_divergence = 0.0;
    let mut total_time = 0.0;

    for _ in 0..iterations {
        // Evolve both systems
        for _ in 0..steps_per_iter {
            current_state = solver.solve(system, 0.0, &current_state, time_step);
            shadow_state = solver.solve(system, 0.0, &shadow_state, time_step);
        }

        // Measure distance
        let dist_vec = shadow_state - current_state;
        let d_t = dist_vec.norm();

        if d_t == 0.0 {
            return Err(ChaosError::Math(
                math_commons::error::MathError::NumericalError {
                    reason: "Trajectories converged completely (distance 0), cannot compute log."
                        .to_string(),
                },
            ));
        }

        sum_log_divergence += (d_t / d0).ln();
        total_time += evolution_time;

        // Rescale: Reset shadow system to be distance d0 away from system
        // along the direction of the current separation.
        shadow_state = current_state + dist_vec.normalize() * d0;
    }

    Ok(sum_log_divergence / total_time)
}

#[verified_engine::verified]
fn array4_distance_and_rescale(
    primary: &ArrayState<4>,
    shadow: &ArrayState<4>,
    d0: f64,
) -> Option<(f64, ArrayState<4>)> {
    let mut norm_sq = 0.0;
    let mut diff = [0.0; 4];
    for ((dst, &s), &p) in diff.iter_mut().zip(&shadow.0).zip(&primary.0) {
        let d = s - p;
        *dst = d;
        norm_sq += d * d;
    }
    let norm = norm_sq.sqrt();
    if norm == 0.0 || !norm.is_finite() {
        return None;
    }
    let scale = d0 / norm;
    let mut rescaled = [0.0; 4];
    for ((dst, &p), &d) in rescaled.iter_mut().zip(&primary.0).zip(&diff) {
        *dst = p + d * scale;
    }
    Some((norm, ArrayState(rescaled)))
}

/// Estimates the largest Lyapunov exponent $\lambda_1$ for a 4D `ArrayState<4>` system
/// (such as [`super::double_pendulum::DoublePendulum`]) via Wolf's renormalization method.
#[verified_engine::verified]
pub fn double_pendulum_lyapunov<S, Sol>(
    system: &S,
    solver: &mut Sol,
    initial_state: ArrayState<4>,
    time_step: f64,
    iterations: usize,
    evolution_time: f64,
) -> Result<f64, ChaosError>
where
    S: OdeSystem<ArrayState<4>> + ?Sized,
    Sol: Solver<ArrayState<4>>,
{
    let d0 = 1e-8;
    let steps_per_iter = (evolution_time / time_step).round() as usize;
    if steps_per_iter == 0 || iterations == 0 {
        return Err(ChaosError::InvalidParameter(
            "evolution_time and iterations must be positive".to_string(),
        ));
    }

    let mut current_state = initial_state;
    let mut shadow_state = ArrayState([
        initial_state.0[0] + d0,
        initial_state.0[1],
        initial_state.0[2],
        initial_state.0[3],
    ]);
    let mut sum_log_divergence = 0.0;
    let mut total_time = 0.0;

    for _ in 0..iterations {
        for _ in 0..steps_per_iter {
            solver.step(system, 0.0, &mut current_state, time_step);
            solver.step(system, 0.0, &mut shadow_state, time_step);
        }
        let Some((d_t, rescaled)) = array4_distance_and_rescale(&current_state, &shadow_state, d0)
        else {
            return Err(ChaosError::Math(
                math_commons::error::MathError::NumericalError {
                    reason: "4D trajectories converged or diverged to non-finite separation."
                        .to_string(),
                },
            ));
        };
        sum_log_divergence += (d_t / d0).ln();
        total_time += evolution_time;
        shadow_state = rescaled;
    }

    Ok(sum_log_divergence / total_time)
}

/// Computes the windowed finite-time Lyapunov exponent
/// $\lambda_{\text{FTLE}} = \frac{1}{\Delta t} \ln\left(\frac{\|\delta \mathbf{x}(t_1)\|}{\|\delta \mathbf{x}(t_0)\|}\right)$.
#[verified_engine::verified]
pub fn finite_time_lyapunov(initial_separation: f64, final_separation: f64, delta_t: f64) -> f64 {
    if initial_separation <= 0.0
        || final_separation <= 0.0
        || delta_t <= 0.0
        || !initial_separation.is_finite()
        || !final_separation.is_finite()
        || !delta_t.is_finite()
    {
        return 0.0;
    }
    (final_separation / initial_separation).ln() / delta_t
}
