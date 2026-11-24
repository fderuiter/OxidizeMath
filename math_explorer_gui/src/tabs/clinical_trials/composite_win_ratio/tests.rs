//! Unit, `VerifiedAllocator` zero-allocation, UI rendering, and `wasm32` browser tests for `WinRatioTool`.

#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::framework::{InteractiveTool, ToolMetadata};
use cohort::{ComparatorKind, PairingMode, WinRatioPlotMode};
use math_commons::registry::{MAX_ITERATIONS, TOLERANCE_FAST};
use math_explorer::applied::win_ratio::bmi::calculate_bmi;

#[test]
fn test_composite_hierarchy_unmatched_matched_and_equal_length_guard() {
    let mut tool = WinRatioTool::default();
    assert!(tool.error_message.is_none());
    assert_eq!(tool.eval.total_pairs, 36);
    assert!(tool.eval.wins > tool.eval.losses);
    let stats = tool.eval.stats.expect("WinRatioStats computed");
    assert!(stats.win_ratio > 1.0);
    assert!(stats.p_value < 0.05);
    assert!(tool.eval.ttest_result.is_some());

    tool.pairing_mode = PairingMode::Matched;
    tool.recalculate();
    assert!(tool.error_message.is_none());
    assert_eq!(tool.eval.total_pairs, 6);

    let unequal_tsv = "Arm\tSurvival\tHosp\tQoL\nTreatment\t800\t0\t75\nTreatment\t700\t1\t70\nControl\t400\t2\t55";
    assert!(tool.import_cohort_str(unequal_tsv).is_err());

    tool.pairing_mode = PairingMode::Unmatched;
    let (nt, nc) = tool
        .import_cohort_str(unequal_tsv)
        .expect("TSV cohort imports");
    assert_eq!((nt, nc), (2, 1));
    assert_eq!(tool.eval.total_pairs, 2);
}

#[test]
fn test_bmi_derivation_weibull_prc_convergence_and_max_iterations_cap() {
    let bmi_csv = "\
Arm,Survival,Hosp,WeightKg,HeightM
Treatment,900,0,70.0,1.75
Treatment,850,1,68.0,1.72
Control,500,2,92.0,1.70
Control,450,3,96.0,1.72";

    let mut tool = WinRatioTool {
        derive_bmi_tier3: true,
        tier3_kind: ComparatorKind::LowerIsBetter,
        horizon_c: 25.0,
        prc_samples: 64,
        ..WinRatioTool::default()
    };
    let (nt, nc) = tool
        .import_cohort_str(bmi_csv)
        .expect("5-col BMI CSV valid");
    assert_eq!((nt, nc), (2, 2));
    assert_eq!(tool.parsed_cohort.derived_bmis.len(), 4);
    let expected_first_bmi = calculate_bmi(70.0, 1.75);
    assert!((tool.parsed_cohort.treatment[0][2] - expected_first_bmi).abs() < TOLERANCE_FAST);
    assert!(
        tool.eval
            .ttest_result
            .as_ref()
            .expect("BMI t-test")
            .statistic
            < 0.0
    );

    // At large horizon c = 25.0, PR(c) converges closely to uncensored PR_W = 1 / theta^alpha
    assert!((tool.eval.terminal_pr_c - tool.eval.weibull_pr_w).abs() < 0.05);

    let over_cap: String = (0..=MAX_ITERATIONS)
        .map(|i| {
            let arm = if i % 2 == 0 { "Treatment" } else { "Control" };
            format!("{arm}, 500.0, 1.0, 70.0")
        })
        .collect::<Vec<_>>()
        .join("\n");
    assert!(tool.import_cohort_str(&over_cap).is_err());
}

#[test]
fn test_zero_alloc_hot_path_and_headless_ui_and_reports() {
    let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());

    let mut tool = WinRatioTool::default();
    tool.recalculate();

    verified_engine::engine::VerifiedEngine::run_verified(|| {
        tool.evaluate_hot_path();
    });
    assert!(!tool.prc_points.is_empty());

    let md = export::format_markdown_report(&tool);
    assert!(md.contains("# Composite Endpoint Win Ratio & Weibull PR(c) Report"));
    assert!(md.contains("Pocock 2-Tier Counts"));
    assert!(md.contains("Independent Two-Sample t-Test"));

    let csv = export::format_csv_report(&tool);
    assert!(csv.contains("summary,win_ratio,"));
    assert!(csv.contains("weibull_prc,PR(c),"));

    let ctx = egui::Context::default();
    let raw = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::Vec2::new(1280.0, 800.0),
        )),
        ..Default::default()
    };
    for mode in [
        WinRatioPlotMode::WeibullPrc,
        WinRatioPlotMode::EmpiricalHierarchy,
    ] {
        tool.plot_mode = mode;
        assert!(!tool.export_datasets().is_empty());
        let _ = ctx.run(raw.clone(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                tool.show_inside(ui);
            });
        });
    }

    let meta = inventory::iter::<ToolMetadata>
        .into_iter()
        .find(|m| m.id == "WinRatioTool")
        .expect("WinRatioTool registered");
    assert_eq!(meta.display_name, "Composite Win Ratio");
}

#[test]
fn test_clinical_trials_suite_offscreen_png_and_svg_exports() {
    use crate::tabs::clinical_trials::{
        randomization::{RandomizationTool, RandomizationType},
        sample_size::SampleSizeCalculatorTool,
        survival::{SurvivalAnalysisTool, TWO_ARM_EXAMPLE_CSV},
    };
    let opts = egui_plot::export::ExportOptions {
        width: 960,
        height: 540,
        ..Default::default()
    };

    let wr = WinRatioTool::default();
    let wr_ds = wr.export_datasets();
    let wr_png = egui_plot::export::export_custom_png(&wr_ds, &opts).expect("WR PNG");
    let wr_svg = egui_plot::export::export_custom_svg(&wr_ds, &opts).expect("WR SVG");
    let _ = std::fs::write("/tmp/win_ratio_weibull_prc.png", &wr_png);
    let _ = std::fs::write("/tmp/win_ratio_weibull_prc.svg", &wr_svg);
    let _ = std::fs::write(
        "/tmp/win_ratio_report.md",
        export::format_markdown_report(&wr),
    );

    let mut surv = SurvivalAnalysisTool::default();
    surv.import_cohort_str(TWO_ARM_EXAMPLE_CSV)
        .expect("2-arm KM");
    let surv_ds = surv.export_datasets();
    let surv_png = egui_plot::export::export_custom_png(&surv_ds, &opts).expect("KM PNG");
    let surv_svg = egui_plot::export::export_custom_svg(&surv_ds, &opts).expect("KM SVG");
    let _ = std::fs::write("/tmp/survival_multi_arm_km.png", &surv_png);
    let _ = std::fs::write("/tmp/survival_multi_arm_km.svg", &surv_svg);

    let ss = SampleSizeCalculatorTool::default();
    let ss_ds = ss.export_datasets();
    let ss_png = egui_plot::export::export_custom_png(&ss_ds, &opts).expect("Power PNG");
    let ss_svg = egui_plot::export::export_custom_svg(&ss_ds, &opts).expect("Power SVG");
    let _ = std::fs::write("/tmp/sample_size_power_curve.png", &ss_png);
    let _ = std::fs::write("/tmp/sample_size_power_curve.svg", &ss_svg);

    let mut rand_tool = RandomizationTool {
        randomization_type: RandomizationType::Block,
        n_subjects: 24,
        block_size: 4,
        ..RandomizationTool::default()
    };
    rand_tool.allocate_subjects();
    let rand_ds = rand_tool.export_datasets();
    let rand_png = egui_plot::export::export_custom_png(&rand_ds, &opts).expect("Rand PNG");
    let rand_svg = egui_plot::export::export_custom_svg(&rand_ds, &opts).expect("Rand SVG");
    let _ = std::fs::write("/tmp/randomization_block_balance.png", &rand_png);
    let _ = std::fs::write("/tmp/randomization_block_balance.svg", &rand_svg);
    assert!(
        wr_png.len() > 1024
            && surv_png.len() > 1024
            && ss_png.len() > 1024
            && rand_png.len() > 1024
    );
}

#[test]
fn test_win_ratio_state_roundtrip() {
    let json_payload = r#"{"weibull_theta":0.6,"weibull_lambda1":0.11,"weibull_lambda2":0.21,"weibull_alpha":0.78,"horizon_c":10.0,"prc_samples":32,"pairing_mode":"Matched","tier1_kind":"HigherIsBetter","tier2_kind":"LowerIsBetter","tier3_kind":{"Threshold":{"threshold":4.0}},"derive_bmi_tier3":false,"ttest_tier_index":2,"ttest_alpha":0.05,"treat_weight_kg":73.0,"treat_height_m":1.75,"ctrl_weight_kg":81.0,"ctrl_height_m":1.74,"plot_mode":"EmpiricalHierarchy","cohort_text":"Treatment, 880.0, 0.0, 78.0\nTreatment, 760.0, 1.0, 71.0\nControl, 410.0, 2.0, 56.0\nControl, 520.0, 2.0, 60.0"}"#;
    let mut roundtrip_tool = WinRatioTool::default();
    roundtrip_tool.try_load_state(json_payload).unwrap();
    let saved = roundtrip_tool.save_state().unwrap();
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&saved).unwrap(),
        serde_json::from_str::<serde_json::Value>(json_payload).unwrap()
    );
}

#[cfg(target_arch = "wasm32")]
mod wasm_browser_tests {
    use super::*;
    use crate::app::MathExplorerApp;
    use crate::starter_gallery::StarterPresetMetadata;
    use crate::url_state::MAX_URL_FRAGMENT_BYTES;
    use wasm_bindgen_test::wasm_bindgen_test;

    #[wasm_bindgen_test]
    fn wasm_win_ratio_presets_url_roundtrip_and_report_downloads() {
        for preset_id in [
            "win_ratio_composite_starter",
            "win_ratio_weibull_prc_starter",
        ] {
            let preset = inventory::iter::<StarterPresetMetadata>
                .into_iter()
                .find(|p| p.id == preset_id)
                .expect("WinRatio starter preset registered");

            let mut app = MathExplorerApp::default();
            let tab_idx = app
                .tabs
                .iter()
                .position(|t| t.url_slug() == "clinical_trials")
                .expect("clinical_trials tab exists");
            app.selected_tab = tab_idx;
            app.tabs[tab_idx]
                .framework_mut()
                .expect("framework present")
                .apply_starter_card(
                    preset.title,
                    preset.tool_name,
                    Some(preset.generate_preset()),
                )
                .expect("starter card applies cleanly");

            let share_url = app.encode_current_share_url().unwrap();
            let hash_idx = share_url.find('#').unwrap();
            let fragment = &share_url[hash_idx..];
            assert!(fragment.contains("tool=WinRatioTool"));
            assert!(fragment.len() <= MAX_URL_FRAGMENT_BYTES);

            let mut restored = MathExplorerApp::default();
            assert!(restored
                .apply_share_location_strings(fragment, "", None)
                .unwrap());
            assert_eq!(
                restored.tabs[tab_idx].active_tool_id(),
                Some("WinRatioTool")
            );
        }

        let mut tool = WinRatioTool::default();
        let md = export::format_markdown_report(&tool);
        tool.io_state
            .trigger_save_markdown("win_ratio_report.md", md);
        tool.io_state.poll_save("Win Ratio report");
        assert!(tool
            .io_state
            .status_info
            .as_deref()
            .unwrap_or("")
            .contains("win_ratio_report.md"));

        let csv = export::format_csv_report(&tool);
        tool.io_state.trigger_save_csv("win_ratio_report.csv", csv);
        tool.io_state.poll_save("Win Ratio report");
        assert!(tool
            .io_state
            .status_info
            .as_deref()
            .unwrap_or("")
            .contains("win_ratio_report.csv"));
    }
}

// [cite:win_ratio]
// [cite:clinical_trials_statistics]
