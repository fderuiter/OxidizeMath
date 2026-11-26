use oxidize_core::rng::OxidizeRng;
use pure_math::pure_math::statistics::copula::{
    Correlation, CorrelationMatrix, Probability, simulate_correlated_parlay,
};
use pure_math::pure_math::statistics::ou_process::{
    EulerMaruyama, ExactOuSolver, OuParams, TimeStep,
};
use pure_math::pure_math::statistics::poisson::HomogeneousPoisson;
use pure_math::pure_math::statistics::queueing::Mm1Queue;

fn verify_ou_stationary_moments() -> Result<(), String> {
    let (theta, mu, sigma) = (2.0_f64, 1.25_f64, 0.6_f64);
    let params =
        OuParams::from_values(mu, theta, sigma).map_err(|e| format!("OuParams error: {e:?}"))?;
    let dt = TimeStep::new(0.005).map_err(|e| format!("TimeStep error: {e:?}"))?;
    let solver = EulerMaruyama::new(params, dt);
    let mut rng = OxidizeRng::from_seed(2026_1123);

    let (paths, steps) = (600usize, 800usize);
    let (mut sum, mut sum_sq) = (0.0, 0.0);
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
    Ok(())
}

fn verify_exact_ou_large_step_moments() -> Result<(), String> {
    let params = OuParams::from_values(1.2, 2.5, 0.8).map_err(|e| format!("{e:?}"))?;
    let dt = TimeStep::new(0.5).map_err(|e| format!("{e:?}"))?;
    let solver = ExactOuSolver::new(params, dt);
    let x0 = 3.6_f64;
    let expected_mean = solver.conditional_mean(x0, 0.5);
    let expected_var = solver.conditional_variance(0.5);
    let mut rng = OxidizeRng::from_seed(2026_1686);
    let n_paths = 12_000usize;
    let (mut sum, mut sum_sq) = (0.0_f64, 0.0_f64);
    let mut traj = [0.0_f64; 8];
    for _ in 0..n_paths {
        let x1 = solver.step(x0, &mut rng);
        sum += x1;
        sum_sq += x1 * x1;
    }
    solver.simulate_into(x0, &mut traj, &mut rng);
    if traj[0] != x0 {
        return Err("ExactOuSolver::simulate_into corrupted initial value".to_string());
    }
    let emp_mean = sum / n_paths as f64;
    let emp_var = (sum_sq / n_paths as f64) - emp_mean * emp_mean;
    if (emp_mean - expected_mean).abs() > 0.02 || (emp_var - expected_var).abs() > 0.015 {
        return Err(format!(
            "ExactOuSolver large-step moment drift: mean {emp_mean} vs {expected_mean}, var {emp_var} vs {expected_var}"
        ));
    }
    Ok(())
}

fn verify_poisson_and_mm1_steady_state() -> Result<(), String> {
    let poisson = HomogeneousPoisson::new(3.2).map_err(|e| format!("{e:?}"))?;
    let horizon = 2.5_f64;
    let expected = poisson.mean(horizon).map_err(|e| format!("{e:?}"))?;
    let mut rng = OxidizeRng::from_seed(2026_7777);
    let trials = 4_000usize;
    let (mut sum, mut sum_sq) = (0.0_f64, 0.0_f64);
    let mut scratch = [0.0_f64; 32];
    for _ in 0..trials {
        let count = poisson
            .simulate_arrivals_into(horizon, &mut scratch, &mut rng)
            .map_err(|e| format!("{e:?}"))? as f64;
        sum += count;
        sum_sq += count * count;
    }
    let emp_mean = sum / trials as f64;
    let emp_var = (sum_sq / trials as f64) - emp_mean * emp_mean;
    if (emp_mean - expected).abs() > 0.15 || (emp_var - expected).abs() > 0.35 {
        return Err(format!(
            "HomogeneousPoisson count moment drift: mean={emp_mean}, var={emp_var}, expected={expected}"
        ));
    }

    let queue = Mm1Queue::new(1.5, 4.0).map_err(|e| format!("{e:?}"))?;
    let ctmc = queue.to_truncated_ctmc(40).map_err(|e| format!("{e:?}"))?;
    let pi = ctmc
        .steady_state()
        .ok_or_else(|| "Truncated M/M/1 CTMC must admit unique steady state".to_string())?;
    let ctmc_mean_l: f64 = pi.iter().enumerate().map(|(k, &p)| (k as f64) * p).sum();
    let exact_l = queue.metrics().mean_in_system;
    if (ctmc_mean_l - exact_l).abs() > 1e-6 {
        return Err(format!(
            "M/M/1 truncated CTMC steady-state L={ctmc_mean_l} diverged from Little's Law L={exact_l}"
        ));
    }
    Ok(())
}

fn verify_copula_parlay_moments() -> Result<(), String> {
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
    Ok(())
}

/// Runs Tier 4 deterministic ChaCha-seeded statistical moment and confidence-interval checks,
/// returning the number of certified stochastic ensembles.
pub fn verify_stochastic_moments() -> Result<usize, String> {
    verify_ou_stationary_moments()?;
    verify_exact_ou_large_step_moments()?;
    verify_poisson_and_mm1_steady_state()?;
    verify_copula_parlay_moments()?;
    Ok(4)
}
