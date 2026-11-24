#![cfg_attr(any(), verified(opt_out = "verification_harness"))]

use oxidize_core::rng::OxidizeRng;
use pure_math::pure_math::statistics::copula::{
    Correlation, CorrelationMatrix, Probability, simulate_correlated_parlay,
};
use pure_math::pure_math::statistics::ou_process::{EulerMaruyama, OuParams, TimeStep};

/// Runs Tier 4 deterministic ChaCha-seeded statistical moment and confidence-interval checks,
/// returning the number of certified stochastic ensembles.
pub fn verify_stochastic_moments() -> Result<usize, String> {
    // 1. Ornstein-Uhlenbeck stationary mean and variance:
    //    dx = theta * (mu - x) dt + sigma dW  =>  Var_inf = sigma^2 / (2 * theta)
    let theta = 2.0_f64;
    let mu = 1.25_f64;
    let sigma = 0.6_f64;
    let params =
        OuParams::from_values(mu, theta, sigma).map_err(|e| format!("OuParams error: {e:?}"))?;
    let dt = TimeStep::new(0.005).map_err(|e| format!("TimeStep error: {e:?}"))?;
    let solver = EulerMaruyama::new(params, dt);
    let mut rng = OxidizeRng::from_seed(2026_1123);

    let paths = 600usize;
    let steps = 800usize;
    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    for _ in 0..paths {
        let mut x = mu;
        for _ in 0..steps {
            x = solver.step(x, &mut rng);
        }
        sum += x;
        sum_sq += (x - mu) * (x - mu);
    }
    let emp_mean = sum / paths as f64;
    let emp_var = sum_sq / paths as f64;
    let exact_var = (sigma * sigma) / (2.0 * theta);
    if (emp_mean - mu).abs() > 0.04 {
        return Err(format!("OU stationary mean drift: {emp_mean} vs {mu}"));
    }
    if (emp_var - exact_var).abs() > 0.025 {
        return Err(format!(
            "OU stationary variance drift: {emp_var} vs {exact_var}"
        ));
    }

    // 2. Correlated Gaussian copula Monte Carlo parlay confidence interval containing exact Sheppard value
    let rho = 0.5_f64;
    let exact_joint = 0.25 + rho.asin() / (2.0 * std::f64::consts::PI);
    let corr = CorrelationMatrix::bivariate(Correlation::new(rho).map_err(|e| format!("{e:?}"))?)
        .map_err(|e| format!("{e:?}"))?;
    let p_half = Probability::new(0.5).map_err(|e| format!("{e:?}"))?;
    let legs = [p_half, p_half];
    let mut parlay_rng = OxidizeRng::from_seed(42_4242);
    let quote = simulate_correlated_parlay(&legs, &corr, 20_000, &mut parlay_rng)
        .map_err(|e| format!("Parlay simulation failed: {e:?}"))?;
    if (quote.joint_probability.value() - exact_joint).abs() > 0.015 {
        return Err(format!(
            "Monte Carlo parlay probability {} outside tolerance of exact {}",
            quote.joint_probability.value(),
            exact_joint
        ));
    }

    Ok(2)
}
