use crate::error::StatMechError;
use math_commons::error::MathError;
use rand::Rng;

/// Simulates a 1D Random Walk.
///
/// # Arguments
/// * `steps` - Number of steps N.
///
/// # Returns
/// * `f64` - Final position.
#[verified_engine::verified]
pub fn random_walk_1d(steps: usize) -> f64 {
    let mut rng = oxidize_core::rng::OxidizeRng::default();
    random_walk_1d_with_rng(steps, &mut rng)
}

/// Simulates a 1D Random Walk using an injected RNG.
#[verified_engine::verified]
pub fn random_walk_1d_with_rng<R: Rng + ?Sized>(steps: usize, rng: &mut R) -> f64 {
    let mut position = 0.0;
    for _ in 0..steps {
        if rng.gen_bool(0.5) {
            position += 1.0;
        } else {
            position -= 1.0;
        }
    }
    position
}

/// Estimates the Diffusion Coefficient D.
///
/// Formula: D ~ <x^2> / (2t)
///
/// # Arguments
/// * `num_walks` - Number of walks M to average over.
/// * `time_steps` - Duration t of each walk.
///
/// # Returns
/// * `f64` - Estimated Diffusion Coefficient D.
#[verified_engine::verified]
pub fn estimate_diffusion_coefficient(num_walks: usize, time_steps: usize) -> f64 {
    let mut rng = oxidize_core::rng::OxidizeRng::default();
    estimate_diffusion_coefficient_with_rng(num_walks, time_steps, &mut rng)
}

/// Estimates the Diffusion Coefficient D using an injected RNG.
#[verified_engine::verified]
pub fn estimate_diffusion_coefficient_with_rng<R: Rng + ?Sized>(
    num_walks: usize,
    time_steps: usize,
    rng: &mut R,
) -> f64 {
    let mut sum_sq_displacement = 0.0;

    for _ in 0..num_walks {
        let final_pos = random_walk_1d_with_rng(time_steps, rng);
        sum_sq_displacement += final_pos * final_pos;
    }

    let msd = sum_sq_displacement / num_walks as f64;
    msd / (2.0 * time_steps as f64)
}

struct OverdampedHarmonic1D {
    stiffness: f64,
    mobility: f64,
    sigma: f64,
}

impl engine_diffeq::SdeSystem<f64> for OverdampedHarmonic1D {
    fn state_dim(&self) -> usize {
        1
    }

    fn wiener_dim(&self) -> usize {
        1
    }

    fn drift(&self, _t: f64, y: &[f64], drift_out: &mut [f64]) {
        drift_out[0] = -self.mobility * self.stiffness * y[0];
    }

    fn diffusion(&self, _t: f64, _y: &[f64], diff_out: &mut [f64]) {
        diff_out[0] = self.sigma;
    }
}

struct HarmonicPotential1D {
    stiffness: f64,
}

impl engine_diffeq::LangevinSystem<f64> for HarmonicPotential1D {
    fn dim(&self) -> usize {
        1
    }

    fn grad_potential(&self, q: &[f64], grad_out: &mut [f64]) {
        grad_out[0] = self.stiffness * q[0];
    }
}

/// Fallibly advances a 1D overdamped Brownian particle in harmonic potential $V(x) = \frac{1}{2} k x^2$
/// using [`engine_diffeq::EulerMaruyamaSolver`] without heap allocations.
#[verified_engine::verified]
pub fn try_step_overdamped_brownian_1d<R: rand_core::RngCore + ?Sized>(
    x: f64,
    stiffness: f64,
    mobility: f64,
    temperature_kb: f64,
    dt: f64,
    rng: &mut R,
) -> Result<f64, StatMechError> {
    if !mobility.is_finite()
        || mobility < 0.0
        || !temperature_kb.is_finite()
        || temperature_kb < 0.0
        || !dt.is_finite()
        || dt < 0.0
    {
        return Err(MathError::OutOfBounds {
            value: mobility.min(temperature_kb).min(dt),
            min: 0.0,
            max: f64::INFINITY,
        }
        .into());
    }
    let sigma = (2.0 * mobility * temperature_kb).sqrt();
    let sys = OverdampedHarmonic1D {
        stiffness,
        mobility,
        sigma,
    };
    let solver = engine_diffeq::EulerMaruyamaSolver::new(engine_diffeq::StochasticCalculus::Ito);
    let mut state = [x];
    let mut scratch = [0.0; 3];
    let mut ws = math_commons::math_kernel::types::SolverWorkspace::new(&mut scratch);
    solver.try_step(&sys, 0.0, dt, &mut state, &mut ws, rng)?;
    Ok(state[0])
}

/// Advances a 1D overdamped Brownian particle in harmonic potential $V(x) = \frac{1}{2} k x^2$
/// using [`engine_diffeq::EulerMaruyamaSolver`] without heap allocations.
#[verified_engine::verified]
pub fn step_overdamped_brownian_1d<R: rand_core::RngCore + ?Sized>(
    x: f64,
    stiffness: f64,
    mobility: f64,
    temperature_kb: f64,
    dt: f64,
    rng: &mut R,
) -> f64 {
    try_step_overdamped_brownian_1d(x, stiffness, mobility, temperature_kb, dt, rng)
        .expect("Valid step_overdamped_brownian_1d parameters")
}

/// Fallibly advances a 1D underdamped Langevin particle $(q, p)$ in harmonic potential $V(q) = \frac{1}{2} k q^2$
/// via [`engine_diffeq::BaoabLangevinSolver`] without heap allocations.
#[allow(clippy::too_many_arguments)]
#[verified_engine::verified]
pub fn try_step_underdamped_langevin_1d<R: rand_core::RngCore + ?Sized>(
    q: f64,
    p: f64,
    mass: f64,
    stiffness: f64,
    friction: f64,
    temperature_kb: f64,
    dt: f64,
    rng: &mut R,
) -> Result<(f64, f64), StatMechError> {
    let solver = engine_diffeq::BaoabLangevinSolver::try_new(engine_diffeq::BaoabParams {
        mass,
        friction,
        temperature: temperature_kb,
        dt,
    })?;
    let sys = HarmonicPotential1D { stiffness };
    let mut q_arr = [q];
    let mut p_arr = [p];
    let mut scratch = [0.0; 2];
    let mut ws = math_commons::math_kernel::types::SolverWorkspace::new(&mut scratch);
    solver.try_step(&sys, &mut q_arr, &mut p_arr, &mut ws, rng)?;
    Ok((q_arr[0], p_arr[0]))
}

/// Advances a 1D underdamped Langevin particle $(q, p)$ in harmonic potential $V(q) = \frac{1}{2} k q^2$
/// via [`engine_diffeq::BaoabLangevinSolver`] without heap allocations.
#[allow(clippy::too_many_arguments)]
#[verified_engine::verified]
pub fn step_underdamped_langevin_1d<R: rand_core::RngCore + ?Sized>(
    q: f64,
    p: f64,
    mass: f64,
    stiffness: f64,
    friction: f64,
    temperature_kb: f64,
    dt: f64,
    rng: &mut R,
) -> (f64, f64) {
    try_step_underdamped_langevin_1d(q, p, mass, stiffness, friction, temperature_kb, dt, rng)
        .expect("Valid step_underdamped_langevin_1d parameters")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[verified_engine::verified]
    fn test_diffusion() {
        let mut rng = oxidize_core::rng::OxidizeRng::default();
        let d = estimate_diffusion_coefficient_with_rng(1000, 100, &mut rng);
        assert!(
            (d - 0.5).abs() < 0.1,
            "Diffusion coefficient should be approx 0.5. Got {}",
            d
        );
    }
}
