use super::*;
use crate::pure_math::statistics::kelly::{
    BankrollFraction, Odds, UnitInterval, kelly_fraction, variants,
};
use oxidize_core::rng::{BASE_SEED, OxidizeRng};

#[test]
#[verified_engine::verified]
fn test_gamblers_ruin_fair_biased_and_large_n_stability() {
    let fair = UnitInterval::new(0.5).unwrap();
    let fair_res = gamblers_ruin_with_target(30.0, 100.0, 1.0, &fair).unwrap();
    assert!((fair_res.ruin_probability - 0.70).abs() < math_commons::registry::TOLERANCE_FAST);
    assert!(
        (fair_res.target_hit_probability - 0.30).abs() < math_commons::registry::TOLERANCE_FAST
    );
    assert!((fair_res.infinite_horizon_ruin - 1.0).abs() < 1e-12);

    // Biased p = 0.55, q = 0.45, i = 10, N = 20 => (rho^10 - rho^20) / (1 - rho^20)
    let fav = UnitInterval::new(0.55).unwrap();
    let rho: f64 = 0.45 / 0.55;
    let expected_fav = (rho.powi(10) - rho.powi(20)) / (1.0 - rho.powi(20));
    let fav_res = gamblers_ruin_with_target(10.0, 20.0, 1.0, &fav).unwrap();
    assert!((fav_res.ruin_probability - expected_fav).abs() < 1e-12);
    assert!((fav_res.infinite_horizon_ruin - rho.powi(10)).abs() < 1e-12);

    // Unfavorable p = 0.45 with large target N = 5000 must not overflow rho^N
    let dog = UnitInterval::new(0.45).unwrap();
    let dog_res = gamblers_ruin_with_target(10.0, 5000.0, 1.0, &dog).unwrap();
    assert!(dog_res.ruin_probability.is_finite());
    assert!(dog_res.ruin_probability > 0.999);
}

#[test]
#[verified_engine::verified]
fn test_lundberg_adjustment_coefficient_exact_closed_form_benchmarks() {
    // Benchmark 1: even-money unit stake (b = 1.0, s = 1.0, p = 0.55):
    // 0.55 * exp(-r) + 0.45 * exp(r) = 1 => r* = ln(0.55 / 0.45) = ln(11/9)
    let p_even = UnitInterval::new(0.55).unwrap();
    let o_even = Odds::new(2.0).unwrap();
    let sol_even = solve_single_bet_lundberg(&p_even, &o_even, 1.0, 20.0).unwrap();
    let expected_r_even = (11.0_f64 / 9.0).ln();
    assert!(
        (sol_even.adjustment_coefficient - expected_r_even).abs()
            < math_commons::registry::TOLERANCE_FAST
    );
    let exact_ruin = gamblers_ruin_infinite(20.0, &p_even).unwrap();
    assert!(
        (sol_even.cramer_lundberg_bound - exact_ruin).abs()
            < math_commons::registry::TOLERANCE_FAST
    );

    // Benchmark 2: asymmetric 2:1 net payoff (decimal 3.0, b = 2.0, p = 0.40, q = 0.60):
    // 0.40 * exp(-2r) + 0.60 * exp(r) = 1 => (u - 1)(3 u^2 - 2 u - 2) = 0 where u = exp(r)
    // Positive root: r* = ln((1 + sqrt(7)) / 3)
    let p_asym = UnitInterval::new(0.40).unwrap();
    let o_asym = Odds::new(3.0).unwrap();
    let sol_asym = solve_single_bet_lundberg(&p_asym, &o_asym, 1.0, 15.0).unwrap();
    let expected_r_asym = ((1.0 + 7.0_f64.sqrt()) / 3.0).ln();
    assert!(
        (sol_asym.adjustment_coefficient - expected_r_asym).abs()
            < math_commons::registry::TOLERANCE_FAST
    );

    // Heterogeneous 3-bet portfolio solves E[exp(-r X)] = 1 residual within 1e-10
    let portfolio = [
        PortfolioBet::new(p_even, o_even, 1.0, 0.50).unwrap(),
        PortfolioBet::new(p_asym, o_asym, 1.0, 0.30).unwrap(),
        PortfolioBet::new(
            UnitInterval::new(0.25).unwrap(),
            Odds::new(5.0).unwrap(),
            0.5,
            0.20,
        )
        .unwrap(),
    ];
    let sol_port = solve_lundberg_ruin(&portfolio, 25.0).unwrap();
    let r = sol_port.adjustment_coefficient;
    let mgf: f64 = portfolio
        .iter()
        .map(|b| {
            let p = b.win_prob.value();
            let wx = b.stake * b.odds.net_profit_multiplier();
            b.weight * (p * (-r * wx).exp() + (1.0 - p) * (r * b.stake).exp())
        })
        .sum();
    assert!((mgf - 1.0).abs() < 1e-10);
    assert!(sol_port.cramer_lundberg_bound > 0.0 && sol_port.cramer_lundberg_bound < 1.0);
    assert!(sol_port.diffusion_approximation > 0.0 && sol_port.diffusion_approximation < 1.0);
}

#[test]
#[verified_engine::verified]
fn test_proportional_kelly_barrier_formulas_and_wilson_width() {
    let p = UnitInterval::new(0.55).unwrap();
    let odds = Odds::new(2.0).unwrap();
    let fk = kelly_fraction(&p, &odds).unwrap();
    let hk = variants::half_kelly(&p, &odds).unwrap();

    // At Full Kelly, exact log-return Lundberg exponent is identically theta* = 1.0,
    // so the 50%-drawdown asymptotic barrier bound is 0.50^1 = 0.50.
    let fk_barrier = kelly_barrier_risk(&p, &odds, &fk, 0.50).unwrap();
    assert!((fk_barrier.lundberg_exponent - 1.0).abs() < math_commons::registry::TOLERANCE_FAST);
    assert!(
        (fk_barrier.lundberg_barrier_bound - 0.50).abs() < math_commons::registry::TOLERANCE_FAST
    );
    assert!((fk_barrier.diffusion_barrier_prob - 0.498_256_6).abs() < 1e-5);

    // At Half Kelly, 2 * mu / sigma^2 approx 3.0 => 50%-drawdown risk drops to ~12.5%
    let hk_barrier = kelly_barrier_risk(&p, &odds, &hk, 0.50).unwrap();
    assert!((hk_barrier.lundberg_barrier_bound - 0.124_6).abs() < 1e-3);
    assert!(hk_barrier.diffusion_barrier_prob < 0.13);

    // Overbetting past 2 * f* (f = 0.30 > 0.20) yields g(f) < 0 and asymptotic barrier prob 1.0
    let over = BankrollFraction::new(0.30).unwrap();
    let over_barrier = kelly_barrier_risk(&p, &odds, &over, 0.50).unwrap();
    assert!((over_barrier.lundberg_barrier_bound - 1.0).abs() < 1e-12);

    // At 10,000 paths and 5% ruin (500 hits), 95% Wilson interval width < 1.0% (0.01)
    let ci = wilson_interval_95(500, 10_000).unwrap();
    assert!(
        ci.width() < 0.01,
        "Expected width < 0.01, got {}",
        ci.width()
    );

    // Ensemble confirms Full Kelly 50%-drawdown risk (~38% over 200 bets)
    let mut ws = BankrollEnsembleWorkspace::with_capacity(10_000, 0).unwrap();
    let mut rng = OxidizeRng::from_seed(BASE_SEED);
    let bet = ProportionalPortfolioBet::unit_weight(p, odds, fk).unwrap();
    let rep = ws
        .simulate_proportional(100.0, 50.0, &bet, 200, &mut rng)
        .unwrap();
    assert!(
        (rep.ruin_probability - 0.38).abs() < 0.02,
        "Expected ~38% 200-bet barrier hit rate, got {}",
        rep.ruin_probability
    );
    assert!(
        rep.drawdown_quantiles.p50 > 0.0
            && rep.drawdown_quantiles.p95 >= rep.drawdown_quantiles.p50
    );
}

// [cite:clinical_trials_statistics]
