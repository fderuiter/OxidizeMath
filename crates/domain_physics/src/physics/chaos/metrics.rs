//! Quantifying Chaos (Lyapunov Exponents)

use super::logistic::LogisticMap;
use crate::error::ChaosError;
use nalgebra::Vector3;
use pure_math::pure_math::analysis::ode::{ArrayState, OdeSystem, Solver};

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
    let mut map = LogisticMap::new(r, x0);
    let mut sum_logs = 0.0;

    for _ in 0..n {
        let x = map.state;
        if !x.is_finite() {
            break;
        }
        // Clamp superstable zero derivatives above 1e-15 and cap escaping overflow at 1e150 so ln stays finite.
        let raw_deriv = (r * (1.0 - 2.0 * x)).abs();
        let derivative = if raw_deriv.is_finite() {
            raw_deriv.clamp(1e-15, 1e150)
        } else {
            1e150
        };
        sum_logs += derivative.ln();
        map.next();
    }

    sum_logs / (n as f64)
}

#[derive(Clone, Copy)]
struct WolfEpochParams {
    steps_per_iter: usize,
    time_step: f64,
    d0: f64,
}

#[verified_engine::verified]
fn step_and_renormalize_pair<S, Sol>(
    system: &S,
    solver: &mut Sol,
    current_state: &mut Vector3<f64>,
    shadow_state: &mut Vector3<f64>,
    t: &mut f64,
    params: WolfEpochParams,
) -> Result<f64, ChaosError>
where
    S: OdeSystem<Vector3<f64>> + ?Sized,
    Sol: Solver<Vector3<f64>>,
{
    for _ in 0..params.steps_per_iter {
        solver.step(system, *t, current_state, params.time_step);
        solver.step(system, *t, shadow_state, params.time_step);
        *t += params.time_step;
    }

    let dist_vec = *shadow_state - *current_state;
    let d_t = dist_vec.norm();
    if !d_t.is_finite()
        || !current_state.x.is_finite()
        || !current_state.y.is_finite()
        || !current_state.z.is_finite()
    {
        return Err(ChaosError::Math(
            math_commons::error::MathError::NumericalError {
                reason: "Non-finite state during Lyapunov integration",
            },
        ));
    }

    let unit_dir = if d_t > 1e-30 {
        dist_vec / d_t
    } else {
        Vector3::new(1.0, 0.0, 0.0)
    };
    *shadow_state = *current_state + unit_dir * params.d0;
    Ok(d_t)
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
    if !time_step.is_finite()
        || time_step <= 0.0
        || !evolution_time.is_finite()
        || evolution_time <= 0.0
        || iterations == 0
        || !initial_state.x.is_finite()
        || !initial_state.y.is_finite()
        || !initial_state.z.is_finite()
    {
        return Err(ChaosError::InvalidParameter(String::new()));
    }
    let steps_per_iter = (evolution_time / time_step).round() as usize;
    if steps_per_iter == 0 {
        return Err(ChaosError::InvalidParameter(String::new()));
    }
    let step_duration = (steps_per_iter as f64) * time_step;

    let mut current_state = initial_state;
    let mut shadow_state = current_state + Vector3::new(d0, 0.0, 0.0);
    let mut sum_log_divergence = 0.0;
    let mut total_time = 0.0;
    let mut t = 0.0;
    let epoch = WolfEpochParams {
        steps_per_iter,
        time_step,
        d0,
    };
    for _ in 0..iterations {
        let d_t = step_and_renormalize_pair(
            system,
            solver,
            &mut current_state,
            &mut shadow_state,
            &mut t,
            epoch,
        )?;
        let ratio = (d_t / d0).clamp(1e-15, 1e150);
        sum_log_divergence += ratio.ln();
        total_time += step_duration;
    }

    Ok(if total_time > 0.0 {
        sum_log_divergence / total_time
    } else {
        0.0
    })
}

#[verified_engine::verified]
fn array_distance_and_renormalize<const N: usize>(
    reference: &ArrayState<N>,
    shadow: &mut ArrayState<N>,
    d0: f64,
) -> Option<f64> {
    let mut dist_sq = 0.0;
    for i in 0..N {
        let diff = shadow.0[i] - reference.0[i];
        dist_sq += diff * diff;
    }
    let dist = dist_sq.sqrt();
    if !dist.is_finite() || dist <= 0.0 {
        return None;
    }
    let scale = d0 / dist;
    for i in 0..N {
        shadow.0[i] = reference.0[i] + (shadow.0[i] - reference.0[i]) * scale;
    }
    Some(dist)
}

/// Zero-allocation Wolf/Benettin largest Lyapunov exponent estimator over `ArrayState<N>`.
///
/// Steps `reference` and `shadow` trajectories in place using `Solver::step` and renormalizes
/// separation back to $d_0 = 10^{-8}$ every `evolution_time` interval without heap allocations.
#[must_use]
#[verified_engine::verified]
pub fn array_lyapunov_exponent<const N: usize, S, Sol>(
    system: &S,
    solver: &mut Sol,
    initial_state: ArrayState<N>,
    time_step: f64,
    iterations: usize,
    evolution_time: f64,
) -> Option<f64>
where
    S: OdeSystem<ArrayState<N>> + ?Sized,
    Sol: Solver<ArrayState<N>>,
{
    if N == 0 || iterations == 0 || !time_step.is_finite() || time_step <= 0.0 {
        return None;
    }
    let steps_per_iter = (evolution_time / time_step).round() as usize;
    if steps_per_iter == 0 || !initial_state.is_finite() {
        return None;
    }
    let d0 = 1e-8;
    let mut current = initial_state;
    let mut shadow = initial_state;
    shadow.0[0] += d0;

    let mut sum_log = 0.0;
    let mut t = 0.0;
    let step_span = steps_per_iter as f64 * time_step;
    for _ in 0..iterations {
        for _ in 0..steps_per_iter {
            solver.step(system, t, &mut current, time_step);
            solver.step(system, t, &mut shadow, time_step);
            t += time_step;
        }
        let d_t = array_distance_and_renormalize(&current, &mut shadow, d0)?;
        sum_log += (d_t / d0).ln();
    }
    Some(sum_log / (iterations as f64 * step_span))
}

/// Zero-allocation largest Lyapunov exponent for 4D phase-space systems (`ArrayState<4>`).
#[must_use]
#[verified_engine::verified]
pub fn lyapunov_exponent_4d<S, Sol>(
    system: &S,
    solver: &mut Sol,
    initial_state: ArrayState<4>,
    time_step: f64,
    iterations: usize,
    evolution_time: f64,
) -> Option<f64>
where
    S: OdeSystem<ArrayState<4>> + ?Sized,
    Sol: Solver<ArrayState<4>>,
{
    array_lyapunov_exponent(
        system,
        solver,
        initial_state,
        time_step,
        iterations,
        evolution_time,
    )
}

/// Zero-allocation largest Lyapunov exponent for 6D phase-space systems (`ArrayState<6>`).
#[must_use]
#[verified_engine::verified]
pub fn lyapunov_exponent_6d<S, Sol>(
    system: &S,
    solver: &mut Sol,
    initial_state: ArrayState<6>,
    time_step: f64,
    iterations: usize,
    evolution_time: f64,
) -> Option<f64>
where
    S: OdeSystem<ArrayState<6>> + ?Sized,
    Sol: Solver<ArrayState<6>>,
{
    array_lyapunov_exponent(
        system,
        solver,
        initial_state,
        time_step,
        iterations,
        evolution_time,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::chaos::double_pendulum::DoublePendulum;
    use crate::physics::chaos::henon_heiles::HenonHeiles;
    use pure_math::pure_math::analysis::ode::{GaussLegendre4, ImplicitMidpoint};

    struct Coupled6DOscillator {
        chaotic_drive: f64,
    }

    impl OdeSystem<ArrayState<6>> for Coupled6DOscillator {
        fn derivative(&self, t: f64, s: &ArrayState<6>) -> ArrayState<6> {
            let mut out = ArrayState::zeros();
            self.derivative_in_place(t, s, &mut out);
            out
        }

        fn derivative_in_place(&self, _t: f64, s: &ArrayState<6>, out: &mut ArrayState<6>) {
            // Unstable hyperbolic coupling when chaotic_drive > 0, pure harmonic when 0
            out.0[0] = s.0[3];
            out.0[1] = s.0[4];
            out.0[2] = s.0[5];
            out.0[3] = -s.0[0] + self.chaotic_drive * s.0[0];
            out.0[4] = -s.0[1];
            out.0[5] = -s.0[2];
        }
    }

    #[test]
    #[verified_engine::verified]
    fn test_lyapunov_4d_distinguishes_regular_and_chaotic_hamiltonians() {
        let hh = HenonHeiles::default();
        let regular_seed = hh.seed_on_poincare_section(1.0 / 12.0, 0.1, 0.0).unwrap();
        let mut mid = ImplicitMidpoint::new(&regular_seed);
        let lam_reg = lyapunov_exponent_4d(&hh, &mut mid, regular_seed, 0.05, 40, 0.5).unwrap();

        let dp = DoublePendulum::default();
        let chaotic_seed = DoublePendulum::default_chaotic_state();
        let mut gl4 = GaussLegendre4::new(&chaotic_seed);
        let lam_chaos = lyapunov_exponent_4d(&dp, &mut gl4, chaotic_seed, 0.02, 60, 0.4).unwrap();

        assert!(
            lam_reg.abs() < 0.15,
            "Regular KAM torus should have near-zero Lyapunov exponent, got {lam_reg}"
        );
        assert!(
            lam_chaos > 0.25,
            "Chaotic double pendulum should have positive Lyapunov exponent, got {lam_chaos}"
        );
    }

    #[test]
    #[verified_engine::verified]
    fn test_lyapunov_6d_regular_vs_exponential_divergence() {
        let seed = ArrayState([0.4, 0.2, -0.3, 0.1, -0.2, 0.3]);
        let mut mid = ImplicitMidpoint::new(&seed);
        let regular = Coupled6DOscillator { chaotic_drive: 0.0 };
        let unstable = Coupled6DOscillator { chaotic_drive: 2.0 };

        let lam_reg = lyapunov_exponent_6d(&regular, &mut mid, seed, 0.05, 30, 0.5).unwrap();
        let lam_div = lyapunov_exponent_6d(&unstable, &mut mid, seed, 0.05, 30, 0.5).unwrap();

        assert!(
            lam_reg.abs() < 0.05,
            "Expected ~0 for 6D torus, got {lam_reg}"
        );
        assert!(
            lam_div > 0.8,
            "Expected ~1.0 positive exponent, got {lam_div}"
        );
    }
}
