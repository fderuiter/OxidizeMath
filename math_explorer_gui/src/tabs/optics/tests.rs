#![cfg_attr(any(), verified(opt_out = "tests"))]

//! Unit, preset, accessibility, URL roundtrip, and export tests for [`super::OpticsTab`].

// [cite:graph_parameters_rust]

use eframe::egui;

use super::*;
use crate::app::{MathExplorerApp, ThemeMode};
use crate::framework::{ExportFormat, ExportStatusSeverity, InteractiveTool};
use crate::tabs::ExplorerTab;

fn test_input(events: Vec<egui::Event>) -> egui::RawInput {
    egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::Vec2::new(1180.0, 820.0),
        )),
        events,
        ..Default::default()
    }
}

#[test]
fn test_default_instantiation_defers_heavy_fft_planning() {
    let tool = FourierOpticsWorkbenchTool::default();
    assert!(
        tool.cache.borrow().is_none(),
        "Default::default() must defer FFT planning and wavefield buffers until first evaluation"
    );
    assert!(tool.has_finite_parameters());
    assert_eq!(tool.name(), "Fourier Optics Workbench");
    assert_eq!(tool.domain(), "optics");
}

#[test]
fn test_all_four_starter_presets_load_and_satisfy_invariants() {
    let mut tool = FourierOpticsWorkbenchTool::default();

    tool.import_preset_json(presets::PRESET_ARAGO_BRIGHT_SPOT)
        .expect("Arago starter must load");
    let obs = tool.observables();
    let center_i = obs
        .iter()
        .find(|(k, _)| *k == "center_intensity")
        .map(|(_, v)| *v)
        .expect("center_intensity observable");
    let drift = obs
        .iter()
        .find(|(k, _)| *k == "parseval_rel_error")
        .map(|(_, v)| *v)
        .expect("parseval_rel_error observable");
    assert!(center_i > 1.0, "Arago preset center intensity {center_i} must exceed 1.0");
    assert!(drift < 1.0e-12, "Parseval drift {drift} must stay < 1.0e-12");

    tool.import_preset_json(presets::PRESET_TALBOT_CARPET)
        .expect("Talbot carpet starter must load");
    let carpet_img = tool.export_raster_image().expect("Talbot carpet raster");
    assert_eq!(carpet_img.size, [96, 128]);

    tool.import_preset_json(presets::PRESET_ZERNIKE_ABERRATED_LENS)
        .expect("Zernike lens starter must load");
    assert_eq!(tool.state.aperture, AperturePreset::ThinQuadraticLens);
    assert!(!tool.export_datasets().is_empty());

    tool.import_preset_json(presets::PRESET_GERCHBERG_SAXTON_HOLOGRAM)
        .expect("Gerchberg-Saxton starter must load");
    let gs_obs = tool.observables();
    let nrmse = gs_obs
        .iter()
        .find(|(k, _)| *k == "target_nrmse")
        .map(|(_, v)| *v)
        .expect("target_nrmse observable");
    assert!(nrmse < 0.32, "GS hologram preset NRMSE {nrmse} must beat 0.32");
}

#[test]
fn test_state_bounds_validation_and_8bit_slm_mask_export() {
    let mut tool = FourierOpticsWorkbenchTool::default();
    let oversized = format!("{{\"wavelength_nm\": 500.0, \"pad\": \"{}\"}}", "x".repeat(70_000));
    assert!(tool.try_load_state(&oversized).is_err());
    assert!(tool.try_load_state("{\"unknown_field\": 1}").is_err());

    tool.state.mode = OpticsMode::GerchbergSaxtonHolography;
    tool.state.display = OpticsDisplayMode::SlmPhaseMask8Bit;
    let mask = tool.export_raster_image().expect("8-bit SLM raster");
    assert_eq!(mask.size, [64, 64]);
    assert!(
        mask.pixels.iter().all(|px| px.r() == px.g() && px.g() == px.b()),
        "8-bit SLM phase mask must render pure calibrated [0..=255] grayscale levels"
    );
    assert!(
        mask.pixels.iter().any(|px| px.r() > 200) && mask.pixels.iter().any(|px| px.r() < 55),
        "8-bit SLM phase mask must span low and high phase levels"
    );
}

#[test]
fn test_optics_tab_themes_url_roundtrip_and_png_svg_csv_json_exports() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX.lock().unwrap();
    let mut app = MathExplorerApp::default();
    let hydrated = app
        .apply_share_location_strings(
            "#tab=optics&tool=FourierOpticsWorkbenchTool",
            "",
            None,
        )
        .expect("optics share URL must hydrate");
    assert!(hydrated);
    assert_eq!(app.tabs[app.selected_tab].url_slug(), "optics");
    assert_eq!(app.tabs[app.selected_tab].academic_domain(), "Physics");

    let mut tab = OpticsTab::default();
    assert!(tab.select_tool_by_name("FourierOpticsWorkbenchTool"));
    let ctx = egui::Context::default();
    let mut frame = eframe::Frame::_new_kittest();

    for theme in [ThemeMode::Dark, ThemeMode::Light, ThemeMode::HighContrast] {
        theme.apply_to_context(&ctx);
        let _ = ctx.run(test_input(Vec::new()), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| tab.show_inside(ui, &mut frame));
        });
    }

    let fw = tab.framework_mut().expect("framework");
    fw.export_options.width = 320;
    fw.export_options.height = 240;
    for fmt in [
        ExportFormat::Png,
        ExportFormat::Svg,
        ExportFormat::Csv,
        ExportFormat::Json,
    ] {
        let _ = ctx.run(test_input(Vec::new()), |ctx| {
            egui_plot::clear_last_plot_datasets();
            fw.show_export_modal = true;
            fw.queue_export(fmt);
            fw.show(ctx, "optics_export_test");
        });
        assert_eq!(
            fw.export_status_severity,
            ExportStatusSeverity::Success,
            "Export format {fmt:?} failed"
        );
    }
}
