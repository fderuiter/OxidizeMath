//! Unit tests for [`super::Glicko2TournamentTool`], atomic CSV/JSON match-log I/O, and URL bounds.

#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::framework::SimulationFramework;
use crate::presets::Glicko2PlayerPreset;

#[test]
fn test_glicko2_starter_presets_fit_4096b_url_cap_and_roundtrip() {
    for raw in [
        crate::starter_gallery::PRESET_GLICKO2_SWISS_UPSET,
        crate::starter_gallery::PRESET_GLICKO2_ROUND_ROBIN_DRIFT,
    ] {
        assert!(raw.len() <= crate::url_state::MAX_DECODED_URL_STATE_BYTES);
        let mut tool = Glicko2TournamentTool::default();
        tool.import_preset_json(raw).expect("starter preset must load");
        assert!(tool.url_share_warning().is_none());
        let saved = tool.save_state().expect("must serialize");
        let mut restored = Glicko2TournamentTool::default();
        restored.try_load_state(&saved).expect("must restore");
        assert_eq!(restored.save_state(), Some(saved));
    }
}

#[test]
fn test_glicko2_atomic_csv_and_json_import_rejects_invalid_without_mutation() {
    let mut tool = Glicko2TournamentTool::default();
    tool.reset_periods();
    let baseline_state = tool.save_state().unwrap();

    // Valid UTF-8 BOM CSV import
    let valid_csv = "\u{feff}period,player_a,player_b,score_a\n1,Carlsen,Nakamura,1.0\n1,Gukesh,Praggnanandhaa,0.5\n";
    let count = tool.import_match_log_str(valid_csv).expect("valid BOM CSV");
    assert_eq!(count, 2);
    assert_eq!(tool.completed_periods.len(), 1);
    let after_valid_state = tool.save_state().unwrap();
    assert_ne!(after_valid_state, baseline_state);

    // Reject self-match, out-of-range score, and unknown player without mutating state
    for bad_payload in [
        "period,player_a,player_b,score_a\n2,Carlsen,Carlsen,1.0\n",
        "period,player_a,player_b,score_a\n2,Carlsen,Nakamura,1.5\n",
        "period,player_a,player_b,score_a\n2,Carlsen,UnknownGM,1.0\n",
        "[{\"period\":2,\"player_a\":\"Carlsen\",\"player_b\":\"Carlsen\",\"score_a\":0.5}]",
        "[{\"period\":2,\"player_a\":0,\"player_b\":99,\"score_a\":1.0}]",
    ] {
        assert!(
            tool.import_match_log_str(bad_payload).is_err(),
            "Payload should have been rejected: {bad_payload}"
        );
        assert_eq!(
            tool.save_state().as_deref(),
            Some(after_valid_state.as_str()),
            "Failed import must not mutate active tool state"
        );
    }

    // Valid JSON match array import
    let valid_json =
        "\u{feff}[{\"period\":2,\"player_a\":\"Gukesh\",\"player_b\":\"Carlsen\",\"score_a\":1.0}]";
    assert_eq!(tool.import_match_log_str(valid_json).unwrap(), 1);
    assert_eq!(tool.completed_periods.len(), 2);
}

#[test]
fn test_glicko2_large_roster_warns_url_cap_while_preserving_local_exports() {
    let mut cfg = super::default_preset_config();
    cfg.players = (0..48)
        .map(|i| Glicko2PlayerPreset {
            name: format!("Grandmaster Competitor #{i:02}"),
            rating: 1500.0 + (i as f64) * 8.0,
            rd: 120.0,
            volatility: 0.06,
            inactive: i % 11 == 0,
        })
        .collect();
    cfg.periods.clear();

    let mut tool = Glicko2TournamentTool::default();
    tool.apply_config(cfg).expect("48-player local config must succeed");
    tool.simulate_next_period().expect("period simulation must succeed");

    assert!(
        tool.estimated_url_state_bytes() > crate::url_state::MAX_DECODED_URL_STATE_BYTES,
        "48-player roster should exceed 4,096-byte URL state cap"
    );
    let warning = tool
        .url_share_warning()
        .expect("must emit proactive URL cap warning");
    assert!(warning.contains("4096"));

    // Local preset JSON, CSV, and JSON standings/pairings exports remain intact
    let preset_json = tool.export_preset_json().expect("local preset JSON works");
    let mut restored = Glicko2TournamentTool::default();
    restored
        .import_preset_json(&preset_json)
        .expect("local preset JSON round-trips above 4 KiB");
    assert_eq!(restored.competitors.len(), 48);

    let standings_csv = tool.format_standings_csv();
    let standings_json = tool.format_standings_json().expect("standings JSON");
    let pairings_csv = tool.format_pairings_csv();
    let pairings_json = tool.format_pairings_json().expect("pairings JSON");
    assert!(standings_csv.contains("Grandmaster Competitor #00"));
    assert!(standings_json.contains("\"ci95_low\""));
    assert!(pairings_csv.contains("expected_score_a"));
    assert!(pairings_json.contains("\"expected_score_a\""));

    // Framework encode_share_url enforces the 4,096 B cap cleanly
    let mut fw = SimulationFramework::new("game_theory");
    assert!(fw.select_tool_by_name("Glicko2TournamentTool"));
    fw.active_tool
        .as_mut()
        .unwrap()
        .import_preset_json(&preset_json)
        .unwrap();
    assert!(fw.encode_share_url().is_err());
}

#[test]
fn test_glicko2_render_all_four_plot_modes_and_manual_round_commit() {
    let mut tool = Glicko2TournamentTool::default();
    tool.import_preset_json(crate::starter_gallery::PRESET_GLICKO2_ROUND_ROBIN_DRIFT)
        .unwrap();
    // Multi-match period in drift preset produces non-zero sequential drift
    let max_drift = tool
        .competitors
        .iter()
        .map(|c| (c.sequential.rating.value() - c.current.rating.value()).abs())
        .fold(0.0_f64, f64::max);
    assert!(max_drift > 1e-3, "expected measurable sequential vs frozen drift");

    let prev_periods = tool.completed_periods.len();
    tool.commit_manual_period().expect("manual round commit");
    assert_eq!(tool.completed_periods.len(), prev_periods + 1);

    let ctx = egui::Context::default();
    for mode in [
        Glicko2PlotMode::ConfidenceBands,
        Glicko2PlotMode::Volatility,
        Glicko2PlotMode::WinMatrix,
        Glicko2PlotMode::SequentialDrift,
    ] {
        tool.plot_mode = mode;
        egui_plot::clear_last_plot_datasets();
        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            tool.show(ctx);
        });
        assert!(
            !egui_plot::get_last_plot_datasets().is_empty(),
            "Plot mode {mode:?} must emit datasets for figure/CSV export"
        );
    }
}
