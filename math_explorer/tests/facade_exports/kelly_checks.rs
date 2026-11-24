use math_explorer::diagnostics::{Diagnostic, Severity};
use math_explorer::error as e;
use math_explorer::pure_math::statistics::{kelly, ruin, validation};

pub(super) fn assert_kelly_validation_and_ruin_exports() {
    let k_stake: e::Error = e::KellyError::InvalidStake { value: 0.0 }.into();
    let k_barrier: e::Error = e::KellyError::InvalidBarrier {
        floor: 1.2,
        initial_wealth: 1.0,
    }
    .into();
    assert!(!format!("{k_stake:?}{k_barrier:?}").is_empty());
    for err in [
        e::PureMathError::from(e::KellyError::InsufficientBets {
            required: 2,
            actual: 1,
        }),
        e::KellyError::DimensionMismatch {
            expected: 3,
            actual: 2,
        }
        .into(),
        e::KellyError::ConvergenceFailed {
            reason: "bracket failed",
        }
        .into(),
        e::KellyError::SolverConvergenceFailed {
            iterations: 16,
            residual: 1e-4,
        }
        .into(),
    ] {
        assert_eq!(err.severity(), Severity::Error);
    }

    let (placed, close) = (
        kelly::Odds::new(2.10).unwrap(),
        kelly::UnitInterval::new(0.50).unwrap(),
    );
    let clv = validation::single_bet_clv(&placed, &close).expect("facade single_bet_clv");
    assert!((clv.ev_pct - 5.0).abs() < 1e-9);
    let fk = kelly::kelly_fraction(&close, &placed).unwrap();
    let barrier = ruin::kelly_barrier_risk(&close, &placed, &fk, 0.5).expect("barrier risk");
    assert!((barrier.lundberg_barrier_bound - 0.5).abs() < 1e-6);
}

pub(super) fn assert_multi_state_and_drawdown_kelly_solvers() {
    use kelly::{
        DrawdownConstraint, MarketOutcome, MarketScratch, Odds, PushWager, QuarterLineWager,
        SlateScratch, SlateWager, UnitInterval, drawdown_constrained_kelly, push_kelly_fraction,
        quarter_line_kelly_fraction, solve_drawdown_constrained_slate,
        solve_mutually_exclusive_market, solve_simultaneous_slate,
    };

    let odds_110 = Odds::from_american(-110.0).unwrap();
    let pw = PushWager::new(0.50, 0.10, 0.40, odds_110).unwrap();
    assert!((push_kelly_fraction(&pw).unwrap().value() - 1.0 / 15.0).abs() < 1e-12);
    let ql = QuarterLineWager::new(0.44, 0.12, 0.0, 0.0, 0.44, odds_110).unwrap();
    assert!(quarter_line_kelly_fraction(&ql).unwrap().value() > 0.017);

    let mkt = [
        MarketOutcome::new(UnitInterval::new(0.50).unwrap(), Odds::new(2.20).unwrap()),
        MarketOutcome::new(UnitInterval::new(0.28).unwrap(), Odds::new(4.00).unwrap()),
        MarketOutcome::new(UnitInterval::new(0.22).unwrap(), Odds::new(3.20).unwrap()),
    ];
    let alloc = solve_mutually_exclusive_market(&mkt, &mut MarketScratch::new()).unwrap();
    assert_eq!(alloc.stakes()[2], 0.0);

    let (p, even) = (
        UnitInterval::new(0.58).unwrap(),
        Odds::from_net_odds(1.0).unwrap(),
    );
    let (slate, mut s_scratch) = ([SlateWager::binary(p, even); 10], SlateScratch::new());
    assert!(
        solve_simultaneous_slate(&slate, &mut s_scratch)
            .unwrap()
            .total_fraction()
            < 1.0
    );
    let (dc, p_55) = (
        DrawdownConstraint::new(0.70, 0.10).unwrap(),
        UnitInterval::new(0.55).unwrap(),
    );
    assert!(
        drawdown_constrained_kelly(&p_55, &even, &dc)
            .unwrap()
            .value()
            < 0.10
    );
    let rc_sol = solve_drawdown_constrained_slate(&slate, &dc, &mut s_scratch).unwrap();
    assert!((rc_sol.total_fraction() - 0.4148979).abs() < 1e-6);
}

pub(super) fn assert_portfolio_kelly_tail_risk_copula_and_bsm() {
    use math_explorer::nalgebra::DMatrix;
    use math_explorer::pure_math::statistics::{black_scholes, copula};

    let mu = [0.12, 0.12, 0.12];
    let cov = DMatrix::from_row_slice(
        3,
        3,
        &[0.04, 0.024, 0.024, 0.024, 0.04, 0.024, 0.024, 0.024, 0.04],
    );
    let sol = kelly::unconstrained_covariance_kelly(&mu, &cov, 0.04).unwrap();
    assert!((sol.metrics.gross_leverage - 30.0 / 11.0).abs() < 1e-10);
    let spec = kelly::PortfolioHorizonSpec::new(0.04, 1000.0, 1.0).unwrap();
    let tail = kelly::parametric_lognormal_tail_risk(&sol.weights, &mu, &cov, spec).unwrap();
    assert!((tail.cvar_95 - 550.08).abs() < 2.0);

    let corr =
        copula::CorrelationMatrix::equicorrelated(3, copula::Correlation::new(0.60).unwrap())
            .unwrap();
    let cop = copula::GaussianCopula::new(corr);
    let crash = cop
        .cdf(&[copula::Probability::new(0.05).unwrap(); 3])
        .unwrap();
    assert!(crash.value() > 0.007);

    let bsm_params = black_scholes::BlackScholesParams::new(100.0, 100.0, 0.05, 0.20, 1.0).unwrap();
    let pair = black_scholes::black_scholes_merton(&bsm_params);
    assert!((pair.call.price - 10.450_583_57).abs() < 1e-6);
}
