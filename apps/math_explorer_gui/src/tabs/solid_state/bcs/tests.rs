//! Unit, export, accessibility, multi-pane, and WASM smoke tests for `BcsSuperconductivityTool`.

#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::{export, BcsPlotView, BcsSuperconductivityTool};
use crate::framework::InteractiveTool;
use eframe::egui::{self, accesskit::Role};

fn test_screen() -> egui::RawInput {
    egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1280.0, 800.0),
        )),
        ..Default::default()
    }
}

#[test]
fn test_bcs_default_weak_coupling_and_normal_state_transition() {
    let mut tool = BcsSuperconductivityTool::default();
    assert!(tool.has_finite_parameters());
    assert!(tool.metrics.delta_zero_mev > 1.0);
    assert!(tool.metrics.tc_kelvin > 5.0);
    assert!(
        (tool.metrics.universal_ratio - 3.528).abs() < 0.08,
        "Expected weak-coupling ratio ~3.53, got {}",
        tool.metrics.universal_ratio
    );
    assert!((tool.metrics.cooper_car_norm - 1.0).abs() < 1e-12);

    // Above T_c, gap closes smoothly to 0 and NIS conductance equals 1.0 everywhere.
    tool.temperature_k = tool.metrics.tc_kelvin + 2.5;
    tool.recalculate();
    assert!(tool.metrics.delta_t_mev.abs() < 1e-9);
    assert!((tool.metrics.zero_bias_conductance - 1.0).abs() < 1e-9);
    for (&[_, u2], &[_, v2]) in tool.u_sq_points.iter().zip(tool.v_sq_points.iter()) {
        assert!((u2 + v2 - 1.0).abs() < 1e-12);
    }
}

#[test]
fn test_bcs_state_roundtrip_and_bounds_rejection() {
    let tool = BcsSuperconductivityTool {
        plot_view: BcsPlotView::NisTunnelingAndCoupling,
        ..Default::default()
    };
    let saved = tool.save_state().expect("save_state must succeed");
    let mut restored = BcsSuperconductivityTool::default();
    restored
        .try_load_state(&saved)
        .expect("valid state JSON must deserialize");
    assert_eq!(restored.plot_view, BcsPlotView::NisTunnelingAndCoupling);

    assert!(restored
        .try_load_state(r#"{"temperature_k": -5.0, "debye_energy_mev": 20.0, "coupling_nv": 0.3, "dos_per_ev": 1.0, "dynes_gamma_mev": 0.04, "frohlich_alpha": 0.4, "plot_view": "GapVsTemperature"}"#)
        .is_err());
    assert!(restored
        .try_load_state(r#"{"temperature_k": 4.2, "unknown_key": 1}"#)
        .is_err());
}

#[test]
fn test_bcs_exports_and_datasheets_are_deterministic_and_finite() {
    let tool = BcsSuperconductivityTool::default();
    let ds_a = tool.export_datasets();
    let ds_b = tool.export_datasets();
    assert_eq!(ds_a, ds_b);
    assert!(ds_a.len() >= 7);
    for (label, pts) in &ds_a {
        assert!(!label.is_empty());
        assert!(!pts.is_empty());
        assert!(pts.iter().all(|p| p[0].is_finite() && p[1].is_finite()));
    }

    let md = export::format_markdown_card(&tool);
    assert!(md.contains("# BCS Superconductivity & NIS Tunneling Datasheet"));
    assert!(md.contains("Universal Ratio 2Δ(0)/(k_B T_c)"));

    let csv = export::format_csv_card(&tool);
    assert!(csv.starts_with("section,key,value,unit\n"));
    assert!(csv.contains("metric,universal_ratio_2delta_kbtc,"));

    let opts = egui_plot::export::ExportOptions::default();
    let series_csv = egui_plot::export::format_csv(&ds_a);
    let series_json = egui_plot::export::format_json(&ds_a);
    assert!(series_csv.contains("BCS Gap Delta(T) (meV)"));
    assert!(series_json.contains("NIS Conductance dI/dV"));
    let prims = egui_plot::export_primitives::datasets_to_primitives(&ds_a);
    let svg = egui_plot::export_primitives::export_primitives_svg(&prims, &opts)
        .expect("SVG plot export must succeed");
    assert!(svg.contains("<svg"));

    // Verify upgraded BandStructureTool deterministic CSV/JSON/SVG series + MD/CSV datasheets
    let mut band = crate::tabs::solid_state::band_structure::BandStructureTool::default();
    assert!(band.clamp_parameters() && band.has_finite_parameters());
    let band_ds = band.export_datasets();
    assert_eq!(band_ds, band.export_datasets());
    assert!(egui_plot::export::format_csv(&band_ds).contains("Tight Binding E(k) (eV)"));
    assert!(egui_plot::export::format_json(&band_ds).contains("Tight Binding E(k) (eV)"));
    let band_svg = egui_plot::export_primitives::export_primitives_svg(
        &egui_plot::export_primitives::datasets_to_primitives(&band_ds),
        &opts,
    )
    .expect("BandStructure SVG export must succeed");
    assert!(band_svg.contains("<svg"));
    assert!(band
        .format_markdown_card()
        .contains("# 1D Electronic Band Structure Datasheet"));
    assert!(band.format_csv_card().contains("metric,bandwidth_4t,"));
}

#[test]
fn test_bcs_ui_accessibility_and_multipane_isolation() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let mut pane_a = BcsSuperconductivityTool::default();
    let mut pane_b = BcsSuperconductivityTool {
        plot_view: BcsPlotView::QuasiparticleAndCoherence,
        ..Default::default()
    };

    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let out = ctx.run(test_screen(), |c| {
        egui::CentralPanel::default().show(c, |ui| {
            ui.columns(2, |cols| {
                cols[0].push_id("pane_left", |ui| {
                    pane_a.show_inside(ui);
                });
                cols[1].push_id("pane_right", |ui| {
                    pane_b.show_inside(ui);
                });
            });
        });
    });

    let ak = out
        .platform_output
        .accesskit_update
        .expect("AccessKit tree must be emitted");
    let docs = ak
        .nodes
        .iter()
        .filter(|(_, n)| n.role() == Role::GraphicsDocument)
        .count();
    assert!(
        docs >= 2,
        "Expected >= 2 accessible plot documents, got {docs}"
    );
}

#[cfg(target_arch = "wasm32")]
mod wasm_tests {
    use super::*;
    use wasm_bindgen_test::*;

    #[wasm_bindgen_test]
    fn wasm_bcs_superconductivity_smoke() {
        let tool = BcsSuperconductivityTool::default();
        assert!(!tool.export_datasets().is_empty());
        assert!(export::format_markdown_card(&tool).contains("T_c"));
    }
}

// [cite:mmwave_radiotherapy_setup]
