#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::state::{PopulationStructureState, PopulationViewMode};
use super::*;
use crate::framework::{ExportFormat, ExportStatusSeverity, FiniteStateGuard, InteractiveTool};
use crate::tabs::ExplorerTab;
use eframe::egui;

fn run_headless_tab_frame(tab: &mut PopulationStructureTab) -> egui::FullOutput {
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let raw = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::Vec2::new(1280.0, 800.0),
        )),
        ..Default::default()
    };
    ctx.run(raw, |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            tab.framework.show_inside(ui, "cannibalism");
        });
    })
}

#[test]
fn test_tab_routing_slug_domain_and_headless_render() {
    let mut tab = PopulationStructureTab::default();
    assert_eq!(tab.name(), "Population Structure");
    assert_eq!(tab.academic_domain(), "Applied Math");
    assert_eq!(tab.url_slug(), "cannibalism");
    assert_eq!(tab.active_tool_id(), Some("PopulationStructureTool"));

    let out = run_headless_tab_frame(&mut tab);
    assert!(!out.shapes.is_empty());
    let rev_before = tab.state_revision();
    let _ = run_headless_tab_frame(&mut tab);
    assert_eq!(
        tab.state_revision(),
        rev_before,
        "Idle headless frames must not mutate state_revision"
    );
}

#[test]
fn test_state_bounds_validation_and_lossless_roundtrip() {
    let mut tool = PopulationStructureTool::default();
    let baseline = tool.save_state().expect("serializes state");

    let mut custom = PopulationStructureState {
        age_bins: 128,
        attack_rate: 0.35,
        fecundity_rate: 25.0,
        view_mode: PopulationViewMode::PhasePortrait,
        ..PopulationStructureState::default()
    };
    let json = serde_json::to_string(&custom).unwrap();
    tool.try_load_state(&json).expect("valid custom state loads");
    assert_eq!(tool.save_state().as_deref(), Some(json.as_str()));
    assert_ne!(tool.save_state().as_deref(), Some(baseline.as_str()));

    custom.age_bins = 8;
    let too_small = serde_json::to_string(&custom).unwrap();
    assert!(tool.try_load_state(&too_small).is_err());

    custom.age_bins = 1024;
    let too_large = serde_json::to_string(&custom).unwrap();
    assert!(tool.try_load_state(&too_large).is_err());

    let huge_array = format!(r#"{{"items":[{}]}}"#, vec!["0"; 600].join(","));
    assert!(tool.state_bounds().validate_json(&huge_array).is_err());
}

#[test]
fn test_finite_state_guard_recovery_actions_and_clamping() {
    let mut tool = PopulationStructureTool::default();
    tool.config.dt = f64::NAN;
    tool.config.age_bins = 4096;
    tool.config.attack_rate = f64::INFINITY;
    assert!(!tool.is_finite_state());
    assert!(tool.diverged());

    assert!(<PopulationStructureTool as InteractiveTool>::clamp_parameters(&mut tool));
    assert_eq!(tool.config.age_bins, 512);
    assert!(tool.is_finite_state());
    assert!(!tool.diverged());

    tool.config.dt = 0.24;
    tool.diagnostic = Some(crate::framework::DiagnosticReport::new(
        1.0,
        10,
        "density",
        "Simulated divergence",
    ));
    assert!(tool.diverged());
    tool.auto_scale_step_size();
    assert!(!tool.diverged());
    assert!((tool.config.dt - 0.12).abs() < 1e-12);

    tool.reset_parameters();
    assert_eq!(tool.config, PopulationStructureState::default());
    assert!(tool.is_finite_state());
}

#[test]
fn test_playback_stepping_and_high_contrast_lexis_wcag_guard() {
    let mut tool = PopulationStructureTool::default();
    let rev_before = tool.state_revision();
    let t0 = tool.mvf_solver.time();
    tool.step_simulation();
    assert!(tool.mvf_solver.time() > t0);
    assert_eq!(
        tool.state_revision(),
        rev_before,
        "Stepping simulation must not dirty parameter state_revision"
    );

    let hc_img = tool.compute_lexis_image(true);
    assert!(hc_img.width() > 1 && hc_img.height() >= 16);
    let contrast_on_black = |px: egui::Color32| -> f64 {
        let ch = |c: u8| {
            let s = f64::from(c) / 255.0;
            if s <= 0.03928 {
                s / 12.92
            } else {
                ((s + 0.055) / 1.055).powf(2.4)
            }
        };
        let lum = 0.2126 * ch(px.r()) + 0.7152 * ch(px.g()) + 0.0722 * ch(px.b());
        (lum + 0.05) / 0.05
    };
    for &px in &hc_img.pixels {
        let ratio = contrast_on_black(px);
        assert!(
            ratio >= 3.0,
            "High-contrast Lexis pixel {px:?} must meet >=3:1 contrast on #000000, got {ratio}"
        );
    }

    tool.lexis_columns.clear();
    tool.lexis_columns.push_back(vec![0.0; 64]);
    let zero_img = tool.compute_lexis_image(false);
    assert_eq!(zero_img.pixels.len(), 64);
}

#[test]
fn test_csv_json_png_and_svg_exports_across_dashboard_and_lexis_modes() {
    for mode in [PopulationViewMode::Dashboard, PopulationViewMode::LexisHeatmap] {
        let mut tab = PopulationStructureTab::default();
        let cfg = PopulationStructureState {
            view_mode: mode,
            ..Default::default()
        };
        let json = serde_json::to_string(&cfg).unwrap();
        tab.framework
            .active_tool
            .as_mut()
            .unwrap()
            .try_load_state(&json)
            .unwrap();

        for fmt in [
            ExportFormat::Csv,
            ExportFormat::Json,
            ExportFormat::Png,
            ExportFormat::Svg,
        ] {
            let ctx = egui::Context::default();
            let raw = egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::Vec2::new(640.0, 480.0),
                )),
                ..Default::default()
            };
            let _ = ctx.run(raw, |ctx| {
                tab.framework.show_export_modal = true;
                tab.framework.queue_export(fmt);
                tab.framework.show(ctx, "cannibalism_export_test");
            });
            assert_eq!(
                tab.framework.export_status_severity,
                ExportStatusSeverity::Success,
                "Export {fmt:?} in mode {mode:?} failed: {:?}",
                tab.framework.export_status_message
            );
        }
    }
}

#[test]
fn test_cannibalism_visual_artifacts_screenshot_generation() {
    let cases = [
        (PopulationViewMode::LexisHeatmap, "/tmp/cannibalism_lexis_heatmap.png"),
        (PopulationViewMode::PhasePortrait, "/tmp/cannibalism_hopf_phase_portrait.png"),
        (PopulationViewMode::AgePyramid, "/tmp/cannibalism_age_pyramid.png"),
    ];
    for (mode, out_path) in cases {
        let mut tab = PopulationStructureTab::default();
        tab.framework.export_options.width = 1100;
        tab.framework.export_options.height = 700;
        let cfg = PopulationStructureState {
            age_bins: 96,
            attack_rate: 0.28,
            fecundity_rate: 22.0,
            view_mode: mode,
            ..Default::default()
        };
        let tool = tab.framework.active_tool.as_mut().unwrap();
        tool.try_load_state(&serde_json::to_string(&cfg).unwrap())
            .unwrap();
        for _ in 0..240 {
            tool.step_simulation();
        }
        let ctx = egui::Context::default();
        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            tab.framework.show_export_modal = true;
            tab.framework.queue_export(ExportFormat::Png);
            tab.framework.show(ctx, "cannibalism_screenshot_gen");
        });
        let bytes = tab
            .framework
            .export_queue
            .last_exported_bytes
            .as_deref()
            .expect("PNG bytes generated");
        std::fs::write(out_path, bytes).expect("writes screenshot PNG");
    }
}

