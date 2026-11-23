#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::{MarkovRegimeTool, RegimeWorkbenchMode, STARTER_PRESET_JSON};
use crate::framework::InteractiveTool;

#[test]
fn test_markov_regime_ctmc_steady_state_and_absorption() {
    let mut tool = MarkovRegimeTool::default();
    assert!(tool.export_capabilities().figure && tool.export_capabilities().dataset);
    assert!(tool.calc_error.is_none());
    assert_eq!(tool.ctmc_prob_series.len(), 3);
    let steady = tool
        .ctmc_steady_state
        .as_ref()
        .expect("ergodic steady state");
    assert!((steady.iter().sum::<f64>() - 1.0).abs() < 1e-6);

    // Make state 2 absorbing (row of zeros) and verify expected_absorption_times
    tool.ctmc_rates[2] = vec![0.0, 0.0, 0.0];
    tool.recalculate();
    assert_eq!(tool.ctmc_absorption_times.len(), 2);
    assert_eq!(tool.ctmc_absorption_points.len(), 2);
    assert!(tool.ctmc_absorption_times.iter().all(|(_, t)| *t > 0.0));
}

#[test]
fn test_markov_regime_hmm_baum_welch_bom_csv_and_url_warning() {
    let mut tool = MarkovRegimeTool::default();
    tool.import_preset_json(STARTER_PRESET_JSON)
        .expect("starter preset loads");
    assert_eq!(tool.mode, RegimeWorkbenchMode::HiddenMarkovHmm);
    assert!(tool.url_share_warning().is_none());

    let bom_obs = "\u{feff}symbol\n2\n2\n2\n0\n0\n0\n2\n2\n0\n0\n";
    let count = tool
        .import_observations_csv(bom_obs)
        .expect("BOM observation CSV imports");
    assert_eq!(count, 10);
    assert_eq!(tool.viterbi_states.len(), 10);

    // Reject out-of-range observation symbol
    assert!(tool.import_observations_csv("0,1,9,0").is_err());

    let report = tool.run_baum_welch_fit().expect("Baum-Welch succeeds");
    assert!(report.final_log_likelihood >= report.initial_log_likelihood - 1e-9);

    let csv = tool.export_regime_csv();
    assert!(csv.starts_with("Step,Observation,Viterbi_State"));
    let json = tool.export_regime_json().expect("valid JSON");
    assert!(json.contains("baum_welch_final_log_likelihood"));

    // Large observation log triggers URL share warning
    let large_obs = (0..3_600)
        .map(|i| (i % 3).to_string())
        .collect::<Vec<_>>()
        .join(",");
    tool.import_observations_csv(&large_obs)
        .expect("large observation CSV imports");
    assert!(tool.url_share_warning().is_some());
}

// [cite:clinical_trials_statistics]
