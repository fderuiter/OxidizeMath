use super::*;
use statrs::distribution::Uniform;

#[test]
#[verified_engine::verified]
fn test_virtual_valuation_uniform() {
    let dist = Uniform::new(0.0, 1.0).unwrap();

    let v = 0.5;
    let j = dist.virtual_valuation(v);
    assert!((j - (2.0 * v - 1.0)).abs() < math_commons::registry::TOLERANCE_STANDARD);

    let v = 0.8;
    let j = dist.virtual_valuation(v);
    assert!((j - (2.0 * v - 1.0)).abs() < math_commons::registry::TOLERANCE_STANDARD);
    assert!(dist.virtual_valuation(f64::NAN).is_nan());
}

#[test]
#[verified_engine::verified]
fn test_optimal_reserve_uniform() {
    let dist = Uniform::new(0.0, 1.0).unwrap();
    let r_star = optimal_reserve_price(&dist, 0.0, 1.0);
    assert!((r_star - 0.5).abs() < 1e-4);
    assert_eq!(optimal_reserve_price(&dist, f64::NAN, 1.0), 0.0);
}

#[test]
#[verified_engine::verified]
fn test_revenue_simulation() {
    let dist = Uniform::new(0.0, 1.0).unwrap();
    let revenue = simulate_optimal_revenue(&dist, 1, 10_000);
    assert!((revenue - 0.25).abs() < 0.02);
}

#[test]
#[verified_engine::verified]
fn test_zero_simulations_and_bidders_never_nan_or_panic() {
    let dist = Uniform::new(0.0, 1.0).unwrap();
    assert_eq!(simulate_optimal_revenue(&dist, 0, 100), 0.0);
    assert_eq!(simulate_optimal_revenue(&dist, 3, 0), 0.0);
    assert!(try_simulate_optimal_revenue(&dist, 0, 100).is_err());
    assert!(try_simulate_optimal_revenue(&dist, 3, 0).is_err());
}

#[test]
#[verified_engine::verified]
fn test_variance_reduction_reduces_standard_error() {
    // Exact 2-bidder Uniform(0, 1) Myerson optimal revenue:
    // E[max(0, 2*U_(2) - 1)] = int_{0.5}^1 (2u - 1) * 2u du = [4/3 u^3 - u^2]_{0.5}^1
    // = (4/3 - 1) - (1/6 - 1/4) = 1/3 + 1/12 = 5/12.
    let dist = Uniform::new(0.0, 1.0).unwrap();
    let exact = 5.0 / 12.0;

    let mut rng_iid = OxidizeRng::new(0x1234_5678);
    let mut rng_anti = OxidizeRng::new(0x1234_5678);
    let mut rng_cv = OxidizeRng::new(0x1234_5678);

    let iid = try_simulate_optimal_revenue_variance_reduced(
        &dist,
        2,
        4_000,
        AuctionVarianceReduction::None,
        &mut rng_iid,
    )
    .unwrap();
    let anti = try_simulate_optimal_revenue_variance_reduced(
        &dist,
        2,
        4_000,
        AuctionVarianceReduction::Antithetic,
        &mut rng_anti,
    )
    .unwrap();
    let cv = try_simulate_optimal_revenue_variance_reduced(
        &dist,
        2,
        4_000,
        AuctionVarianceReduction::ControlVariate,
        &mut rng_cv,
    )
    .unwrap();

    assert!((iid.mean_revenue - exact).abs() < 0.025);
    assert!((anti.mean_revenue - exact).abs() < 0.015);
    assert!((cv.mean_revenue - exact).abs() < 0.015);
    assert!(anti.std_error < iid.std_error);
    assert!(cv.std_error < iid.std_error);
}

#[test]
#[verified_engine::verified]
fn test_optimal_reserve_unbracketed_fallback() {
    let dist = Uniform::new(0.0, 100.0).unwrap();

    let r = optimal_reserve_price(&dist, 60.0, 100.0);
    assert!((r - 60.0).abs() < 1e-4);

    let r = optimal_reserve_price(&dist, 0.0, 40.0);
    assert!((r - 40.0).abs() < 1e-4);
}

#[verified_engine::verified]
fn compose_auction_and_payoff_check<D: ValuationDistribution>(
    dist: &D,
    rows: usize,
    cols: usize,
    lower_bound: f64,
    upper_bound: f64,
) -> Result<f64, crate::error::AppliedError> {
    if rows != cols {
        Err(crate::error::GameTheoryError::NonSquarePayoffMatrix { rows, cols })?;
    }
    let solver = Bisection::default();
    let reserve = optimal_reserve_price_with_solver(dist, lower_bound, upper_bound, &solver)?;
    Ok(reserve)
}

#[test]
#[verified_engine::verified]
fn test_mechanism_design_composes_with_applied_error_and_diagnostic_bus() {
    use diagnostics::DiagnosticBus;
    use std::error::Error as StdError;
    use std::sync::{Arc, Mutex};

    let dist = Uniform::new(0.0, 100.0).unwrap();
    let ok_reserve = compose_auction_and_payoff_check(&dist, 2, 2, 0.0, 100.0).unwrap();
    assert!((ok_reserve - 50.0).abs() < 1e-4);

    let game_err = compose_auction_and_payoff_check(&dist, 2, 3, 0.0, 100.0).unwrap_err();
    assert!(game_err.source().is_some());

    let analysis_err = compose_auction_and_payoff_check(&dist, 2, 2, 60.0, 100.0).unwrap_err();
    assert!(analysis_err.source().is_some());

    let bus = DiagnosticBus::new();
    let captured = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&captured);
    bus.register_listener(move |diag| {
        sink.lock().unwrap().push((
            diag.metadata.get("source").cloned(),
            diag.metadata.get("umbrella_type").cloned(),
        ));
    });
    bus.emit_diagnostic_with_source("mechanism_design", &analysis_err);
    let events = captured.lock().unwrap();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].0.as_deref(), Some("mechanism_design"));
    assert_eq!(events[0].1.as_deref(), Some("AppliedError"));
}
