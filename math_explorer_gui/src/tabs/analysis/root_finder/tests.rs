#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::tabs::analysis::AnalysisTab;
use crate::tabs::ExplorerTab;

#[test]
fn test_analysis_tab_registers_standalone_root_finder_tool() {
    let mut tab = AnalysisTab::default();
    assert!(
        tab.select_tool_by_name("Root Finder"),
        "AnalysisTab must register and select 'Root Finder'"
    );
}

#[test]
fn test_root_finder_all_1d_methods_converge_and_record_geometry() {
    let mut step_counts = std::collections::BTreeMap::new();
    for (idx, method) in ScalarRootMethod::ALL.into_iter().enumerate() {
        let mut tool = RootFinderTool {
            dimension: RootDimension::Scalar1D,
            scalar_method: method,
            bracket_min: 0.1,
            bracket_max: 1.5,
            tolerance: 1e-8,
            ..Default::default()
        };
        tool.scalar_input.set_expression("cos(x) - x");
        tool.recalculate();

        let root = tool
            .converged_root_1d
            .unwrap_or_else(|| panic!("{method:?} failed to converge: {:?}", tool.status_error));
        assert!(
            (root - 0.739_085_133_2).abs() < 1e-6,
            "{method:?} root {root}"
        );
        assert!(tool.final_residual < 1e-6);
        assert!(!tool.scalar_steps.is_empty());
        step_counts.insert(idx, tool.scalar_steps.len());
        if method.is_bracketing() {
            assert!(tool.scalar_steps[0].bracket.is_some());
        }
        assert!(tool.scalar_steps[0].construction_segment.is_some());
    }
    // Brent (index 1) must converge strictly faster than Bisection (index 0)
    assert!(step_counts[&1] < step_counts[&0]);
}

#[test]
fn test_root_finder_invalid_bracket_reports_clean_error_without_panic() {
    let mut tool = RootFinderTool {
        dimension: RootDimension::Scalar1D,
        scalar_method: ScalarRootMethod::Bisection,
        bracket_min: -1.0,
        bracket_max: 5.0,
        ..Default::default()
    };
    tool.scalar_input.set_expression("ln(x) - 1");
    tool.recalculate();
    assert!(tool.converged_root_1d.is_none());
    assert!(tool.status_error.is_some());
    assert!(tool.has_finite_parameters());
}

#[test]
fn test_root_finder_2d_multivariate_newton_and_zero_contours() {
    let mut tool = RootFinderTool {
        dimension: RootDimension::System2D,
        init_2d: [1.6, 1.4],
        domain_span_2d: 3.0,
        tolerance: 1e-8,
        ..Default::default()
    };
    tool.recalculate();

    let [rx, ry] = tool.converged_root_2d.expect("2D Newton must converge");
    assert!((rx * rx + ry * ry - 4.0).abs() < 1e-6);
    assert!((rx * rx - ry - 0.5).abs() < 1e-6);
    assert!(!tool.contours_f1.is_empty());
    assert!(!tool.contours_f2.is_empty());
    assert!(tool.system_steps.len() >= 2);
}

#[test]
fn test_root_finder_state_roundtrip_presets_and_deferred_export() {
    let mut tool = RootFinderTool::default();
    assert!(tool
        .import_preset_json(presets::BRENT_TRANSCENDENTAL_PRESET_JSON)
        .is_ok());
    assert!(tool
        .import_preset_json(presets::NEWTON_2D_CONTOUR_PRESET_JSON)
        .is_ok());

    let saved = tool.save_state().expect("save state");
    assert!(saved.len() < 4096, "state size {} >= 4 KiB", saved.len());

    let mut restored = RootFinderTool::default();
    restored.try_load_state(&saved).expect("restore state");
    assert_eq!(restored.dimension, RootDimension::System2D);
    assert!(restored.converged_root_2d.is_some());

    assert!(restored
        .try_load_state(r#"{"unknown_field": 123}"#)
        .is_err());

    let png = render_tool_deferred_png(Box::new(restored));
    assert!(!png.is_empty());
    let _ = std::fs::write("/tmp/root_finder_2d_contours.png", &png);
}

fn render_tool_deferred_png(tool: Box<dyn crate::framework::InteractiveTool>) -> Vec<u8> {
    let mut fw = crate::framework::SimulationFramework::new("__screenshot__");
    fw.export_options.width = 960;
    fw.export_options.height = 600;
    fw.active_tool = Some(tool);
    let ctx = eframe::egui::Context::default();
    let raw = eframe::egui::RawInput {
        screen_rect: Some(eframe::egui::Rect::from_min_size(
            eframe::egui::Pos2::ZERO,
            eframe::egui::Vec2::new(1024.0, 768.0),
        )),
        ..Default::default()
    };
    let _ = ctx.run(raw, |ctx| {
        egui_plot::clear_last_plot_datasets();
        fw.show_export_modal = true;
        fw.queue_export(crate::framework::ExportFormat::Png);
        fw.show(ctx, "root_finder_screenshot_fw");
    });
    fw.export_queue
        .last_exported_bytes
        .expect("deferred PNG bytes")
}

#[test]
fn test_export_root_finder_and_gk15_screenshots() {
    use crate::tabs::analysis::riemann::{
        FunctionPreset, QuadratureMethod, RiemannIntegrationTool,
    };

    let mut rf_1d = RootFinderTool {
        dimension: RootDimension::Scalar1D,
        scalar_method: ScalarRootMethod::Brent,
        bracket_min: 0.1,
        bracket_max: 1.5,
        ..Default::default()
    };
    rf_1d.scalar_input.set_expression("cos(x) - x");
    rf_1d.recalculate();
    rf_1d.selected_step = 0;
    let png_1d = render_tool_deferred_png(Box::new(rf_1d));
    assert!(!png_1d.is_empty());
    let _ = std::fs::write("/tmp/root_finder_1d_brent.png", png_1d);

    let mut gk = RiemannIntegrationTool::default();
    gk.preset = FunctionPreset::Custom;
    gk.quadrature_method = QuadratureMethod::GaussKronrod15;
    gk.cc_tolerance = 1e-9;
    gk.min_x = -2.0;
    gk.max_x = 2.0;
    gk.steps = 24;
    gk.math_input.set_expression("clamp(sin(k * x), -0.5, 0.8)");
    gk.math_input.set_param("k", 3.0);
    gk.recalculate();
    let png_gk = render_tool_deferred_png(Box::new(gk));
    assert!(!png_gk.is_empty());
    let _ = std::fs::write("/tmp/riemann_gk15_pulse.png", png_gk);
}

#[test]
fn test_export_ode_multi_solver_comparison_screenshot() {
    use crate::presets::OdePreset;
    use crate::tabs::analysis::ode::OdeSolverTool;
    use math_explorer::pure_math::analysis::ode::IntegrationMethod;

    let mut ode = OdeSolverTool {
        preset: OdePreset::Custom,
        method: IntegrationMethod::DormandPrince45,
        comparison_mode: true,
        ic_y0: 1.0,
        dt: 0.18,
        total_time: 6.0,
        ..Default::default()
    };
    ode.math_input.set_expression("-k * y * sin(t)");
    ode.math_input.set_param("k", 1.2);
    ode.recalculate();
    let png_ode = render_tool_deferred_png(Box::new(ode));
    assert!(!png_ode.is_empty());
    let _ = std::fs::write("/tmp/ode_multi_solver_comparison.png", png_ode);
}
