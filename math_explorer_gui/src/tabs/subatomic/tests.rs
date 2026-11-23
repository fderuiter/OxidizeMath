#![cfg_attr(any(), verified(opt_out = "tests"))]

//! Desktop unit tests for [`SubatomicPhysicsTab`], [`NuclearStabilityTool`], [`StandardModelTool`],
//! [`RelativisticJetTool`], `(Z, N)` chart interaction, `#tab=subatomic` URL hydration,
//! `FiniteStateGuard` NaN recovery, and CSV/JSON/SVG/PNG exports.

use std::collections::HashSet;

use eframe::egui;

use super::*;
use crate::app::{MathExplorerApp, ThemeMode};
use crate::framework::{ExportFormat, ExportStatusSeverity, InteractiveTool};
use crate::tabs::ExplorerTab;

fn test_raw_input(events: Vec<egui::Event>) -> egui::RawInput {
    egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::Vec2::new(1280.0, 820.0),
        )),
        events,
        ..Default::default()
    }
}

#[test]
fn subatomic_tab_renders_all_three_tools_across_dark_light_and_high_contrast() {
    let ctx = egui::Context::default();
    let mut frame = eframe::Frame::_new_kittest();
    let mut tab = SubatomicPhysicsTab::default();
    assert_eq!(tab.name(), "Subatomic Physics");
    assert_eq!(tab.academic_domain(), "Physics");
    assert_eq!(tab.url_slug(), "subatomic");

    for tool_id in [
        "NuclearStabilityTool",
        "StandardModelTool",
        "RelativisticJetTool",
    ] {
        assert!(tab.select_tool_by_name(tool_id));
        for theme in [ThemeMode::Dark, ThemeMode::Light, ThemeMode::HighContrast] {
            theme.apply_to_context(&ctx);
            let _ = ctx.run(test_raw_input(Vec::new()), |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| {
                    tab.show_inside(ui, &mut frame);
                });
            });
        }
    }
}

#[test]
fn nuclear_chart_canvas_click_and_keyboard_arrows_update_selected_nuclide() {
    let ctx = egui::Context::default();
    let mut tool = NuclearStabilityTool::default();
    assert_eq!((tool.state.selected_z, tool.state.selected_n), (26, 30));

    // Frame 1: lay out side panel and central (Z, N) canvas
    let _ = ctx.run(test_raw_input(Vec::new()), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| tool.show_inside(ui));
    });

    // Frame 2: click inside the upper-right region of the (Z, N) nuclide chart to focus & select heavy isotope
    let click_pos = egui::pos2(980.0, 85.0);
    let click_events = vec![
        egui::Event::PointerMoved(click_pos),
        egui::Event::PointerButton {
            pos: click_pos,
            button: egui::PointerButton::Primary,
            pressed: true,
            modifiers: egui::Modifiers::NONE,
        },
        egui::Event::PointerButton {
            pos: click_pos,
            button: egui::PointerButton::Primary,
            pressed: false,
            modifiers: egui::Modifiers::NONE,
        },
    ];
    let _ = ctx.run(test_raw_input(click_events), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| tool.show_inside(ui));
    });
    let after_click = (tool.state.selected_z, tool.state.selected_n);
    assert_ne!(after_click, (26, 30));
    assert!(tool.focused_canvas_id().is_some());

    // Frame 3: press ArrowRight + ArrowUp while focused on canvas
    let key_events = vec![
        egui::Event::Key {
            key: egui::Key::ArrowRight,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        },
        egui::Event::Key {
            key: egui::Key::ArrowUp,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        },
    ];
    let _ = ctx.run(test_raw_input(key_events), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| tool.show_inside(ui));
    });
    assert_eq!(tool.state.selected_n, (after_click.1 + 1).min(150));
    assert_eq!(tool.state.selected_z, (after_click.0 + 1).min(96));
}

#[test]
fn subatomic_presets_url_hydration_and_landau_pole() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let mut app = MathExplorerApp::default();
    assert!(app
        .apply_share_location_strings("#tab=subatomic&tool=StandardModelTool", "", None)
        .unwrap());
    assert_eq!(app.tabs[app.selected_tab].url_slug(), "subatomic");
    assert_eq!(
        app.tabs[app.selected_tab].active_tool_id(),
        Some("StandardModelTool")
    );

    // Verify all 4 starter presets deserialize cleanly and evaluate physical invariants
    let mut nuc = NuclearStabilityTool::default();
    nuc.import_preset_json(presets::PRESET_VALLEY_OF_STABILITY_FE56)
        .unwrap();
    assert!(nuc.evaluate_observables().unwrap().breakdown.per_nucleon > 8.5);
    nuc.import_preset_json(presets::PRESET_U238_BATEMAN_CHAIN)
        .unwrap();
    assert!(nuc.evaluate_observables().unwrap().alpha_q_mev > 3.5);

    let mut sm = StandardModelTool::default();
    sm.import_preset_json(presets::PRESET_DUNE_MSW_NEUTRINOS)
        .unwrap();
    let sm_obs = sm.evaluate_observables();
    assert!(((sm_obs.msw_probs.iter().sum::<f64>()) - 1.0).abs() < 1e-12);
    sm.state.probe_scale_gev = 0.08;
    assert!(matches!(
        sm.evaluate_observables().qcd_alpha_probe,
        Err(math_explorer::error::StandardModelError::LandauPole { .. })
    ));

    let mut jet = RelativisticJetTool::default();
    jet.import_preset_json(presets::PRESET_BLAZAR_SYNCHROTRON_JET)
        .unwrap();
    let jet_obs = jet.evaluate_observables();
    assert!(jet_obs.momentum_shell_valid && jet_obs.beta_apparent > 4.0);
    assert!((jet_obs.shock_ratio - 7.0).abs() < 1e-9);
    assert!((jet_obs.enthalpy_over_c2 - (1.0 + 4.0 * 0.33)).abs() < 1e-9);
    assert!(jet_obs.li_ma_sigma > 10.0);
}

#[test]
fn subatomic_bounds_validation_and_finite_state_guard_rollback() {
    let mut nuc = NuclearStabilityTool::default();
    assert!(nuc.try_load_state(r#"{"unknown_key": 1}"#).is_err());
    let oversized = format!("{{\"selected_z\":26,\"pad\":\"{}\"}}", "x".repeat(5000));
    assert!(nuc.try_load_state(&oversized).is_err());

    let mut tab = SubatomicPhysicsTab::default();
    assert!(tab.select_tool_by_name("NuclearStabilityTool"));
    tab.framework.cache_valid_snapshot();
    {
        let tool = tab
            .framework
            .active_tool
            .as_mut()
            .unwrap()
            .as_any_mut()
            .unwrap()
            .downcast_mut::<NuclearStabilityTool>()
            .unwrap();
        tool.state.alpha_beta = f64::NAN;
        assert!(crate::framework::recovery::FiniteStateGuard::diverged(tool));
    }
    assert!(tab.framework.inspect_runtime_state().is_err());
    assert!(tab.framework.is_diverged());
    assert!(tab.framework.rollback_step_back());
    assert!(!tab.framework.is_diverged());

    let mut sm = StandardModelTool::default();
    sm.state.neutrino_energy_gev = f64::NAN;
    assert!(crate::framework::recovery::FiniteStateGuard::diverged(&sm));
    assert!(sm.clamp_parameters() && sm.has_finite_parameters());

    let mut jet = RelativisticJetTool::default();
    jet.state.beta = f64::NAN;
    assert!(crate::framework::recovery::FiniteStateGuard::diverged(&jet));
    assert!(jet.clamp_parameters() && jet.has_finite_parameters());
}

#[test]
fn subatomic_tools_export_valid_png_svg_csv_and_json_artifacts() {
    for tool_id in [
        "NuclearStabilityTool",
        "StandardModelTool",
        "RelativisticJetTool",
    ] {
        let mut tab = SubatomicPhysicsTab::default();
        assert!(tab.select_tool_by_name(tool_id));
        let fw = tab.framework_mut().unwrap();
        fw.export_options.width = 360;
        fw.export_options.height = 240;

        for format in [
            ExportFormat::Png,
            ExportFormat::Svg,
            ExportFormat::Csv,
            ExportFormat::Json,
        ] {
            let ctx = egui::Context::default();
            let _ = ctx.run(test_raw_input(Vec::new()), |ctx| {
                egui_plot::clear_last_plot_datasets();
                fw.show_export_modal = true;
                fw.queue_export(format);
                fw.show(ctx, "subatomic_export_test");
            });
            assert_eq!(fw.export_status_severity, ExportStatusSeverity::Success);
            match format {
                ExportFormat::Png => {
                    let bytes = fw.export_queue.last_exported_bytes.as_deref().unwrap();
                    let px =
                        crate::framework::deferred_export_tests::decode_rgba_pixels(bytes).unwrap();
                    let colors: HashSet<[u8; 4]> = px.into_iter().collect();
                    assert!(colors.len() >= 2);
                }
                ExportFormat::Svg => {
                    let svg = fw.export_queue.last_exported_content.as_deref().unwrap();
                    assert!(svg.contains("<svg") && svg.len() > 250);
                }
                ExportFormat::Csv => {
                    let csv = fw.export_queue.last_exported_content.as_deref().unwrap();
                    assert!(csv.lines().count() > 20);
                }
                ExportFormat::Json => {
                    let json = fw.export_queue.last_exported_content.as_deref().unwrap();
                    let parsed: serde_json::Value = serde_json::from_str(json).unwrap();
                    assert!(parsed.is_object() || parsed.is_array());
                }
                _ => {}
            }
        }
    }
}
