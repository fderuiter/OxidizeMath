//! Unit, rollback guard, export, accessibility, and WASM tests for `QuantumStatsPhononTool`.

#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::{export, QuantumPhononView, QuantumStatsMode, QuantumStatsPhononTool};
use crate::framework::{InteractiveTool, SimulationFramework};
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
fn test_fermi_dirac_physical_si_units_eliminates_049_clamping() {
    let mut tool = QuantumStatsPhononTool {
        temperature_k: 300.0,
        chemical_potential_ev: 0.0,
        energy_min_ev: 0.0,
        energy_span_ev: 0.20,
        ..Default::default()
    };
    tool.recalculate();

    // Midpoint E = 0.10 eV above mu = 0.0 eV at T = 300 K (k_B T ~ 0.02585 eV) => f_FD ~ 0.0205, NOT 0.49!
    let mid = tool.fd_points[60];
    assert!((mid[0] - 0.10).abs() < 1e-9);
    assert!(
        (mid[1] - 0.020_47).abs() < 1e-3,
        "Expected physical Fermi-Dirac occupancy ~0.0205 at E - mu = 0.10 eV, got {}",
        mid[1]
    );
    assert!(tool.metrics.partition_z > 1.0 && tool.metrics.partition_z.is_finite());
    assert!((tool.metrics.boson_ccr_norm - 1.0).abs() < 1e-12);
}

#[test]
fn test_bose_einstein_mu_ge_e_triggers_stat_mech_error_and_framework_rollback() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let mut fw = SimulationFramework::new("solid_state");
    assert!(fw.select_tool_by_name("Quantum Statistics & Phonons"));

    let ctx = egui::Context::default();
    let _ = ctx.run(test_screen(), |c| fw.show(c, "solid_state"));
    assert!(!fw.is_diverged);

    {
        let tool = fw
            .active_tool
            .as_mut()
            .and_then(|t| t.as_any_mut())
            .and_then(|a| a.downcast_mut::<QuantumStatsPhononTool>())
            .expect("Active tool must downcast to QuantumStatsPhononTool");
        tool.stats_mode = QuantumStatsMode::BoseEinstein;
        tool.energy_min_ev = 0.05;
        tool.chemical_potential_ev = 0.08;
        tool.recalculate();
        assert!(tool
            .stat_mech_error
            .as_deref()
            .unwrap_or_default()
            .to_ascii_lowercase()
            .contains("chemical potential"));
        assert!(!tool.has_finite_parameters());
        // Last valid curves remain completely finite with zero NaNs
        assert!(tool
            .export_datasets()
            .iter()
            .all(|(_, pts)| pts.iter().all(|p| p[0].is_finite() && p[1].is_finite())));
    }

    // Running a frame triggers SimulationFramework rollback guard while preserving last_valid_snapshot
    let _ = ctx.run(test_screen(), |c| fw.show(c, "solid_state"));
    assert!(fw.is_diverged);
    assert!(fw.last_valid_snapshot.is_some());

    // Clamp parameters resolves the singularity cleanly
    let clamped = fw
        .active_tool
        .as_mut()
        .expect("active tool")
        .clamp_parameters();
    assert!(clamped);
    assert!(fw
        .active_tool
        .as_ref()
        .expect("active tool")
        .has_finite_parameters());
}

#[test]
fn test_quantum_stats_clamp_recovers_non_finite_phonon_and_screening_parameters() {
    let mut tool = QuantumStatsPhononTool {
        sound_velocity_eva: f64::NAN,
        optical_omega_mev: f64::INFINITY,
        debye_theta_k: f64::NEG_INFINITY,
        k_tf_inv_a: f64::NAN,
        exchange_j_mev: f64::NAN,
        ..Default::default()
    };
    assert!(!tool.has_finite_parameters());
    assert!(tool.clamp_parameters());
    assert!(tool.has_finite_parameters());
    assert!(tool
        .export_datasets()
        .iter()
        .all(|(_, pts)| pts.iter().all(|p| p[0].is_finite() && p[1].is_finite())));
}

#[test]
fn test_quantum_stats_exports_and_datasheets_deterministic() {
    let mut tool = QuantumStatsPhononTool {
        plot_view: QuantumPhononView::ScreeningAndYukawa,
        ..Default::default()
    };
    let ds1 = tool.export_datasets();
    let ds2 = tool.export_datasets();
    assert_eq!(ds1, ds2);
    assert!(ds1.len() >= 8);

    let md = export::format_markdown_card(&tool);
    assert!(md.contains("# Quantum Statistics, Phonon & Screening Datasheet"));
    assert!(md.contains("Canonical Partition Function Z"));

    let csv = export::format_csv_card(&tool);
    assert!(csv.starts_with("section,key,value,unit\n"));
    assert!(csv.contains("metric,debye_cv_per_nk,"));

    let opts = egui_plot::export::ExportOptions::default();
    let series_csv = egui_plot::export::format_csv(&ds1);
    let series_json = egui_plot::export::format_json(&ds1);
    assert!(series_csv.contains("Fermi-Dirac Occupancy f_FD(E)"));
    assert!(series_json.contains("Yukawa Screened Potential V(r)"));
    let prims = egui_plot::export_primitives::datasets_to_primitives(&ds1);
    let svg = egui_plot::export_primitives::export_primitives_svg(&prims, &opts)
        .expect("SVG export must succeed");
    assert!(svg.contains("<svg"));

    // Invalid Bose-Einstein preset JSON must reject atomically
    assert!(tool
        .try_load_state(r#"{"temperature_k":300.0,"chemical_potential_ev":0.10,"energy_min_ev":0.05,"energy_span_ev":0.5,"sound_velocity_eva":3.5,"optical_omega_mev":36.0,"debye_theta_k":343.0,"k_tf_inv_a":1.2,"exchange_j_mev":8.0,"stats_mode":"BoseEinstein","plot_view":"QuantumOccupancy"}"#)
        .is_err());
}

#[test]
fn test_quantum_stats_ui_accessibility_and_multipane() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let mut left = QuantumStatsPhononTool::default();
    let mut right = QuantumStatsPhononTool {
        plot_view: QuantumPhononView::PhononAndDebye,
        ..Default::default()
    };

    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let out = ctx.run(test_screen(), |c| {
        egui::CentralPanel::default().show(c, |ui| {
            ui.columns(2, |cols| {
                cols[0].push_id("q_left", |ui| left.show_inside(ui));
                cols[1].push_id("q_right", |ui| right.show_inside(ui));
            });
        });
    });
    let ak = out
        .platform_output
        .accesskit_update
        .expect("AccessKit tree");
    let docs = ak
        .nodes
        .iter()
        .filter(|(_, n)| n.role() == Role::GraphicsDocument)
        .count();
    assert!(docs >= 2);
}

#[cfg(target_arch = "wasm32")]
mod wasm_tests {
    use super::*;
    use wasm_bindgen_test::*;

    #[wasm_bindgen_test]
    fn wasm_quantum_stats_phonon_smoke() {
        let tool = QuantumStatsPhononTool::default();
        assert!(!tool.export_datasets().is_empty());
        assert!(export::format_csv_card(&tool).contains("partition_z"));
    }
}

// [cite:mmwave_radiotherapy_setup]
