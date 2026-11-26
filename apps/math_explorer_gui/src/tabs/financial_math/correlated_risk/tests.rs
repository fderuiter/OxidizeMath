#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;

#[test]
fn test_copula_matches_independent_at_zero_and_monotonic_in_rho() {
    let mut tool = CorrelatedRiskScenarioTool {
        prob_a: 0.65,
        prob_b: 0.60,
        rho: 0.0,
        ..Default::default()
    };
    tool.last_copula_key = None;
    tool.recalculate_if_needed();

    let naive = 0.65 * 0.60;
    let zero_prob = tool.active_sgp_prob;
    assert!(
        (zero_prob - naive).abs() < 5e-3,
        "Expected rho=0 copula prob {zero_prob} to match naive p_a*p_b {naive}"
    );

    tool.rho = 0.60;
    tool.recalculate_if_needed();
    let pos_prob = tool.active_sgp_prob;
    assert!(
        pos_prob > zero_prob,
        "Positive correlation must increase joint probability ({pos_prob} <= {zero_prob})"
    );

    tool.rho = -0.60;
    tool.recalculate_if_needed();
    let neg_prob = tool.active_sgp_prob;
    assert!(
        neg_prob < zero_prob,
        "Negative correlation must decrease joint probability ({neg_prob} >= {zero_prob})"
    );
}

#[test]
fn test_ou_trajectories_deterministic_under_oxidize_rng_seed() {
    let mut run_a = CorrelatedRiskScenarioTool {
        ou_mu: 0.45,
        ou_theta: 1.8,
        ou_sigma: 0.22,
        ou_x0: 0.80,
        ou_dt: 0.02,
        ou_steps: 120,
        ou_paths: 12,
        seed: 777,
        ..Default::default()
    };
    run_a.last_ou_key = None;
    run_a.recalculate_if_needed();

    let mut run_b = CorrelatedRiskScenarioTool {
        ou_mu: 0.45,
        ou_theta: 1.8,
        ou_sigma: 0.22,
        ou_x0: 0.80,
        ou_dt: 0.02,
        ou_steps: 120,
        ou_paths: 12,
        seed: 777,
        ..Default::default()
    };
    run_b.last_ou_key = None;
    run_b.recalculate_if_needed();

    assert_eq!(run_a.ou_mean_path, run_b.ou_mean_path);
    assert_eq!(run_a.ou_sample_paths.len(), 12);
    assert_eq!(run_a.ou_sample_paths[0].1, run_b.ou_sample_paths[0].1);

    run_b.seed = 778;
    run_b.recalculate_if_needed();
    assert_ne!(run_a.ou_mean_path, run_b.ou_mean_path);
}

#[test]
fn test_state_validation_enforces_ou_and_copula_bounds_and_formula_safe_csv() {
    let mut tool = CorrelatedRiskScenarioTool::default();
    let base_json = tool.save_state().expect("serializes state");

    let mut over_paths: serde_json::Value = serde_json::from_str(&base_json).unwrap();
    over_paths["ou_paths"] = serde_json::json!(201);
    assert!(tool.try_load_state(&over_paths.to_string()).is_err());

    let mut over_steps: serde_json::Value = serde_json::from_str(&base_json).unwrap();
    over_steps["ou_steps"] = serde_json::json!(2001);
    assert!(tool.try_load_state(&over_steps.to_string()).is_err());

    let mut bad_rho: serde_json::Value = serde_json::from_str(&base_json).unwrap();
    bad_rho["rho"] = serde_json::json!(1.05);
    assert!(tool.try_load_state(&bad_rho.to_string()).is_err());

    tool.scenario_label = String::from("=CMD|' /C calc'!A0");
    let csv = tool.export_scenario_csv();
    assert!(
        csv.contains("\"'=CMD|' /C calc'!A0\"") || csv.contains("'=CMD|' /C calc'!A0"),
        "CSV export must neutralize formula injection triggers: {csv}"
    );
    let md = tool.export_scenario_markdown();
    assert!(md.contains("Correlated Risk Scenario"));
}

#[test]
fn test_ou_summary_stats_match_shared_solver_workspace_across_updates() {
    let mut tool = CorrelatedRiskScenarioTool {
        mode: RiskAnalysisMode::OrnsteinUhlenbeck,
        ou_mu: 0.42,
        ou_theta: 1.6,
        ou_sigma: 0.30,
        ou_x0: 0.75,
        ou_dt: 0.025,
        ou_steps: 100,
        ou_paths: 32,
        seed: 2026,
        ..Default::default()
    };
    tool.last_ou_key = None;
    tool.recalculate_if_needed();

    let stats = tool.ou_stats.expect("expected populated OU summary stats");
    let mut ref_ws = math_explorer::stochastic::SolverWorkspace::new(0, 32);
    let ref_summary = ref_ws
        .summarize_samples(&tool.ou_finals_scratch, None)
        .expect("expected non-empty terminal samples");
    assert_eq!(tool.ou_finals_scratch.len(), 32);
    assert!((stats.mean_final - ref_summary.mean).abs() < 1e-15);
    assert!((stats.std_final - ref_summary.std_dev).abs() < 1e-15);
    assert!((stats.p05_final - ref_summary.percentile_05).abs() < 1e-15);
    assert!((stats.p95_final - ref_summary.percentile_95).abs() < 1e-15);
    assert!(stats.p05_final <= stats.mean_final && stats.mean_final <= stats.p95_final);
}
