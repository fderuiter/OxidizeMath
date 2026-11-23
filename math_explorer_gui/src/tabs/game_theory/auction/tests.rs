#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::atlas::AuctionCartographer;
use super::challenge::{
    CHALLENGE_FIRST_PRICE_SHADING, CHALLENGE_MYERSON_WELFARE_RESERVE,
};
use super::math::{
    AuctionDistEval, AuctionDistributionKind, AuctionEvalInputs, AuctionEvalWorkspace,
    AuctionRuleMode,
};
use super::MyersonAuctionTool;
use crate::async_sim::atlas::RegimeCartographer;
use crate::framework::InteractiveTool;
use scientific_metadata::theory::TheoryDescribable;

#[test]
fn test_uniform_myerson_optimal_reserve_and_single_bidder_exclusion() {
    let mut tool = MyersonAuctionTool {
        distribution: AuctionDistributionKind::Uniform,
        uniform_min: 0.0,
        uniform_max: 100.0,
        n_bidders: 1,
        reserve_price: 50.0,
        focal_valuation: 80.0,
        candidate_bid: 80.0,
        rule_mode: AuctionRuleMode::SecondPrice,
        ..MyersonAuctionTool::default()
    };
    tool.recalculate();

    assert!(
        (tool.summary.optimal_reserve - 50.0).abs() < 1e-3,
        "Expected Myerson r* = 50.0, got {}",
        tool.summary.optimal_reserve
    );
    assert!(
        (tool.summary.trade_probability - 0.5).abs() < 1e-6,
        "Expected 50% trade probability (50% blocked trades), got {}",
        tool.summary.trade_probability
    );
    assert!(
        (tool.summary.expected_revenue - 25.0).abs() < 1e-4,
        "Expected analytical single-bidder revenue 25.0, got {}",
        tool.summary.expected_revenue
    );
    assert!(
        (tool.summary.mc_optimal_revenue - 25.0).abs() < 1.5,
        "Expected MC revenue near 25.0, got {}",
        tool.summary.mc_optimal_revenue
    );
}

#[test]
fn test_first_price_shading_equilibrium_and_distributions() {
    let mut tool = MyersonAuctionTool {
        distribution: AuctionDistributionKind::Uniform,
        uniform_min: 0.0,
        uniform_max: 100.0,
        n_bidders: 2,
        reserve_price: 50.0,
        focal_valuation: 100.0,
        candidate_bid: 62.5,
        rule_mode: AuctionRuleMode::FirstPrice,
        ..MyersonAuctionTool::default()
    };
    tool.recalculate();

    assert!(
        (tool.summary.equilibrium_bid - 62.5).abs() < 1e-6,
        "Expected symmetric First-Price b*(100; 50, 2) = 62.5, got {}",
        tool.summary.equilibrium_bid
    );
    assert!(tool.summary.is_equilibrium_fixed_point);
    assert!((tool.summary.candidate_utility - 37.5).abs() < 1e-5);

    tool.distribution = AuctionDistributionKind::Exp;
    tool.exp_rate = 0.2;
    tool.recalculate();
    assert!(
        (tool.summary.optimal_reserve - 5.0).abs() < 1e-3,
        "For Exp(0.2), J(v) = v - 5 => r* = 5.0, got {}",
        tool.summary.optimal_reserve
    );

    tool.distribution = AuctionDistributionKind::Beta;
    tool.beta_alpha = 2.0;
    tool.beta_beta = 2.0;
    tool.recalculate();
    assert!(tool.has_finite_parameters());
    assert!(tool.summary.optimal_reserve > 0.3 && tool.summary.optimal_reserve < 0.7);
}

#[test]
fn test_snapshot_manager_branching_and_state_roundtrip() {
    let mut tool = MyersonAuctionTool::default();
    assert!(tool.save_state().expect("default serializes").len() <= 4096);
    let series = tool.current_trace_series();
    assert!(tool.snapshot_manager.add_snapshot(
        "Branch 1",
        tool.param_summary(),
        series,
    ));
    tool.n_bidders = 5;
    tool.record_history_and_recalculate();
    assert_eq!(tool.snapshot_manager.len(), 1);
    assert_eq!(tool.snapshot_manager.history_len(), 1);
    assert!(tool.save_recovery_state().expect("recovery serializes").len() <= 4096);

    let json = tool.save_state().expect("serializes");
    assert!(tool.state_bounds().validate_json(&json).is_ok());
    let mut restored = MyersonAuctionTool::default();
    restored.try_load_state(&json).expect("restores");
    assert_eq!(restored.n_bidders, 5);
    assert_eq!(restored.snapshot_manager.len(), 1);
}

#[test]
fn test_auction_cartographer_and_workspace_zero_allocations() {
    let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());

    let tool = MyersonAuctionTool::default();
    let theory = tool.theory_parameters();
    let dist = AuctionDistEval::from_params(
        AuctionDistributionKind::Uniform,
        0.0,
        100.0,
        0.1,
        2.0,
        2.0,
    );
    let cartographer = AuctionCartographer::new(
        &theory,
        ("reserve_price", "n_bidders"),
        dist.clone(),
        2,
        AuctionRuleMode::FirstPrice,
    );
    let mut ws = AuctionEvalWorkspace::default();

    let (res, _metrics) = verified_engine::engine::VerifiedEngine::run_verified(|| {
        let mut acc = 0.0;
        for i in 0..16 {
            let r = i as f64 * 6.0;
            let cell = cartographer.evaluate_cell(r, 2.0 + (i % 5) as f64);
            acc += cell.primary + cell.secondary;
        }
        let summary = ws.evaluate_analytical(
            &dist,
            AuctionEvalInputs {
                rule: AuctionRuleMode::FirstPrice,
                reserve: 50.0,
                n_bidders: 2,
                focal_v: 100.0,
                candidate_bid: 62.5,
                fixed_point_tol: 0.5,
            },
        );
        acc + summary.equilibrium_bid
    });

    assert!(res.is_finite());
}

#[test]
fn test_auction_high_contrast_wcag_and_parameter_clamping() {
    let tool = MyersonAuctionTool::default();
    let theory = tool.theory_parameters();
    let dist = AuctionDistEval::from_params(
        AuctionDistributionKind::Uniform,
        0.0,
        100.0,
        0.1,
        2.0,
        2.0,
    );
    let cartographer = AuctionCartographer::new(
        &theory,
        ("reserve_price", "n_bidders"),
        dist,
        2,
        AuctionRuleMode::FirstPrice,
    );
    for r in [10.0, 50.0, 85.0] {
        let cell = cartographer.evaluate_cell(r, 3.0);
        let hc = cartographer.color_for_cell(cell, true);
        let srgb_lum = |u: u8| {
            let x = f64::from(u) / 255.0;
            if x <= 0.04045 { x / 12.92 } else { ((x + 0.055) / 1.055).powf(2.4) }
        };
        let lum = 0.2126 * srgb_lum(hc.r()) + 0.7152 * srgb_lum(hc.g()) + 0.0722 * srgb_lum(hc.b());
        assert!(
            (lum + 0.05) / 0.05 >= 3.0,
            "HighContrast AuctionCartographer pixel must satisfy >= 3.0:1 contrast on #000000"
        );
    }

    let mut dirty = MyersonAuctionTool {
        reserve_price: f64::NAN,
        ..MyersonAuctionTool::default()
    };
    assert!(!dirty.has_finite_parameters());
    assert!(dirty.clamp_parameters());
    assert!(dirty.has_finite_parameters());
}

#[test]
fn test_auction_challenges_pass_fail_and_tamper_clamping() {
    let meta_reserve =
        crate::starter_gallery::find_challenge_metadata(CHALLENGE_MYERSON_WELFARE_RESERVE)
            .expect("registered welfare reserve challenge");
    let baseline_json = meta_reserve.generate_baseline_state();
    let mut tool = MyersonAuctionTool::default();
    tool.apply_challenge(Some(CHALLENGE_MYERSON_WELFARE_RESERVE))
        .unwrap();
    tool.try_load_state(&baseline_json).unwrap();
    tool.apply_challenge(Some(CHALLENGE_MYERSON_WELFARE_RESERVE))
        .unwrap();
    let base_eval = tool.evaluate_challenge().expect("active eval");
    assert!(
        !base_eval.passed,
        "Myerson r*=50 baseline must fail P_trade >= 0.84"
    );

    tool.reserve_price = 38.0;
    tool.recalculate();
    let pass_eval = tool.evaluate_challenge().expect("active eval");
    assert!(
        pass_eval.passed,
        "Reserve r=38 yields R >= 40 and P_trade = 85.56% >= 84%: {:?}",
        pass_eval
    );

    // Tampered locked inputs (n_bidders=10) must be clamped back to n_bidders=2
    let tampered = baseline_json.replace("\"n_bidders\":2", "\"n_bidders\":10");
    let clamped = meta_reserve.clamp_state_json(&tampered).expect("clamped");
    assert!(clamped.contains("\"n_bidders\":2"));

    let meta_shading =
        crate::starter_gallery::find_challenge_metadata(CHALLENGE_FIRST_PRICE_SHADING)
            .expect("registered first-price shading challenge");
    let shading_base = meta_shading.generate_baseline_state();
    tool.apply_challenge(Some(CHALLENGE_FIRST_PRICE_SHADING))
        .unwrap();
    tool.try_load_state(&shading_base).unwrap();
    tool.apply_challenge(Some(CHALLENGE_FIRST_PRICE_SHADING))
        .unwrap();
    assert!(!tool.evaluate_challenge().unwrap().passed);

    tool.candidate_bid = 62.5;
    tool.recalculate();
    assert!(tool.evaluate_challenge().unwrap().passed);
}
