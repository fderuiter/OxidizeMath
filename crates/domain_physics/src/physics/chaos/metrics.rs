//! Quantifying Chaos (Lyapunov Exponents)

use super::logistic::LogisticMap;
use crate::error::ChaosError;
use nalgebra::Vector3;
use pure_math::pure_math::analysis::ode::{OdeSystem, Solver, SolverExt};

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
        sum_logs += derivative.abs().max(1e-15).ln();
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
