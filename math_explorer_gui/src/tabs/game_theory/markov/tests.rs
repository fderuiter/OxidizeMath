//! Unit and GUI frame tests for [`super::MarkovSequenceWorkbenchTool`].
// [cite:clinical_trials_statistics]

#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::{MarkovScenario, MarkovSequenceWorkbenchTool};
use crate::app::ThemeMode;
use crate::framework::InteractiveTool;
use crate::presets::MarkovWorkbenchMode;
use eframe::egui;

#[test]
fn test_utf8_and_bom_csv_and_json_imports_plus_atomic_rollback() {
    let mut tool = MarkovSequenceWorkbenchTool::default();
    let baseline = tool.save_state().expect("baseline state");

    // UTF-8 BOM-prefixed CSV observation sequence with header and labels
    let bom_csv =
        "\u{feff}step,observation\n0,Small Up\n1,Large Up\n2,Flat\n3,Small Down\n4,Large Down\n";
    let msg = tool.import_file_str(bom_csv).expect("BOM CSV import");
    assert!(msg.contains("5 CSV observation"));
    assert_eq!(tool.config.hmm.observations, vec![1, 0, 2, 3, 4]);
    assert!((tool.hmm_out.forward_likelihood - 0.000_326).abs() < 1e-6);
    assert!((tool.hmm_out.log_likelihood - -8.028_610).abs() < 1e-6);

    // Atomic rollback on malformed CSV observation symbol
    let snapshot_after_valid = tool.save_state().unwrap();
    assert!(
        tool.import_file_str("step,observation\n0,NOT_A_SYMBOL\n")
            .is_err()
    );
    assert_eq!(tool.save_state().unwrap(), snapshot_after_valid);

    // UTF-8 BOM-prefixed standalone HiddenMarkovModel JSON import and roundtrip
    let exported_hmm_json = format!("\u{feff}{}", tool.format_hmm_model_json().unwrap());
    assert!(tool.import_file_str(&exported_hmm_json).is_ok());

    // Atomic rollback on non-stochastic HMM JSON
    let bad_hmm = "\u{feff}{\"initial\":[0.5,0.5],\"transitions\":[[0.9,0.5],[0.4,0.6]],\"emissions\":[[0.5,0.5],[0.5,0.5]]}";
    let before_bad = tool.save_state().unwrap();
    assert!(tool.import_file_str(bad_hmm).is_err());
    assert_eq!(tool.save_state().unwrap(), before_bad);

    // Restore original baseline
    tool.try_load_state(&baseline).unwrap();
    assert_eq!(tool.save_state().unwrap(), baseline);
}

#[test]
fn test_exports_across_all_three_modes_and_baum_welch_em() {
    let mut tool = MarkovSequenceWorkbenchTool::default();
    tool.config.hmm.sim_length = 120;
    tool.simulate_hmm_sequence().unwrap();
    let report = tool.run_baum_welch().unwrap();
    assert!(report.final_log_likelihood >= report.initial_log_likelihood - 1e-12);

    let timeline_csv = tool.format_hmm_timeline_csv();
    assert!(timeline_csv.starts_with("step,observation_idx,observation_label,viterbi_state_idx"));
    assert_eq!(timeline_csv.lines().count(), 121);
    let fitted_json = tool.format_hmm_model_json().unwrap();
    assert!(fitted_json.contains("\"transitions\""));

    // CTMC mode CSV/JSON import + export
    tool.load_scenario(MarkovScenario::RedundantServerCtmc)
        .unwrap();
    assert_eq!(tool.config.mode, MarkovWorkbenchMode::Ctmc);
    let ctmc_csv = tool.format_ctmc_report_csv();
    assert!(ctmc_csv.contains("Dual Healthy"));
    let ctmc_json = tool.format_ctmc_report_json().unwrap();
    assert!(ctmc_json.contains("expected_absorption_times_mttf"));

    // BOM-prefixed CTMC generator CSV import
    let bom_gen_csv = "\u{feff}s0,s1\n-1.5,1.5\n2.5,-2.5\n";
    assert!(tool.import_file_str(bom_gen_csv).is_ok());
    assert!(tool.ctmc_out.steady_state.is_some());

    // Absorbing DTMC mode CSV/JSON export
    tool.load_scenario(MarkovScenario::BasketballEpvDtmc)
        .unwrap();
    assert_eq!(tool.config.mode, MarkovWorkbenchMode::AbsorbingDtmc);
    let dtmc_csv = tool.format_dtmc_report_csv();
    assert!(dtmc_csv.contains("expected_steps_to_absorption"));
    let dtmc_json = tool.format_dtmc_report_json().unwrap();
    assert!(dtmc_json.contains("expected_possession_value"));
}

#[test]
fn test_url_budget_warning_and_multi_theme_rendering() {
    let mut tool = MarkovSequenceWorkbenchTool::default();
    assert!(tool.url_share_warning().is_none());
    assert!(tool.estimated_url_state_bytes() < crate::url_state::MAX_DECODED_URL_STATE_BYTES);

    // Large 2,200-step observation sequence triggers proactive 4 KiB URL share warning
    tool.config.hmm.sim_length = 2_200;
    tool.simulate_hmm_sequence().unwrap();
    let warn = tool
        .url_share_warning()
        .expect("expected proactive >4 KiB URL warning");
    assert!(warn.contains("exceeds the 4096 B URL cap"));

    // Render across Dark, Light, and HighContrast themes across all 3 modes
    for theme in [ThemeMode::Dark, ThemeMode::Light, ThemeMode::HighContrast] {
        let ctx = egui::Context::default();
        ctx.set_visuals(theme.visuals());
        for mode in [
            MarkovWorkbenchMode::Hmm,
            MarkovWorkbenchMode::Ctmc,
            MarkovWorkbenchMode::AbsorbingDtmc,
        ] {
            tool.config.mode = mode;
            let _ = ctx.run(egui::RawInput::default(), |ctx| {
                tool.show(ctx);
            });
        }
    }
}

#[test]
fn test_slow_ctmc_steady_state_exact_and_multi_sink_none_in_workbench() {
    let mut tool = MarkovSequenceWorkbenchTool::default();
    tool.load_scenario(MarkovScenario::RedundantServerCtmc)
        .unwrap();

    // Slow-rate 2-state repairable CTMC (rates 1e-7 and 3e-7) must resolve exact [0.75, 0.25]
    let slow_csv = "\u{feff}healthy,degraded\n-1e-7,1e-7\n3e-7,-3e-7\n";
    tool.import_file_str(slow_csv).expect("slow CTMC import");
    let pi = tool
        .ctmc_out
        .steady_state
        .as_ref()
        .expect("unique steady state on slow 2-state CTMC");
    assert_eq!(pi.len(), 2);
    assert!((pi[0] - 0.75).abs() < 1e-12, "pi[0] = {}", pi[0]);
    assert!((pi[1] - 0.25).abs() < 1e-12, "pi[1] = {}", pi[1]);

    // Multi-sink CTMC has non-unique steady-state subspace and must return None
    let multi_sink_csv = "\u{feff}sink_a,sink_b\n0.0,0.0\n0.0,0.0\n";
    tool.import_file_str(multi_sink_csv)
        .expect("multi-sink CTMC import");
    assert!(tool.ctmc_out.steady_state.is_none());

    let ctx = egui::Context::default();
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        tool.show(ctx);
    });
}
