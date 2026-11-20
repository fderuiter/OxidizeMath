#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;

#[test]
fn test_riemann_tool_default_initialization() {
    let tool = RiemannIntegrationTool::default();
    assert_eq!(tool.steps, 10);
    assert_eq!(tool.cached_trapezoid_polygons.len(), 10);
    assert_eq!(tool.cached_true_curve.len(), 501);
    for poly in &tool.cached_trapezoid_polygons {
        assert_eq!(poly.len(), 4);
    }
    // Sin(x) from 0 to PI area is approximately 2.0
    assert!((tool.result - 2.0).abs() < 0.05);
}

#[test]
fn test_riemann_recalculate_parameter_updates() {
    let mut tool = RiemannIntegrationTool {
        steps: 25,
        preset: FunctionPreset::XSquared,
        min_x: 0.0,
        max_x: 3.0,
        ..Default::default()
    };
    tool.recalculate();

    assert_eq!(tool.cached_trapezoid_polygons.len(), 25);
    assert_eq!(tool.cached_true_curve.len(), 501);
    // Integral of x^2 from 0 to 3 is 9.0
    assert!((tool.result - 9.0).abs() < 0.05);

    // Verify first polygon geometry
    let first_poly = &tool.cached_trapezoid_polygons[0];
    let dx = 3.0 / 25.0;
    assert_eq!(first_poly[0], [0.0, 0.0]);
    assert_eq!(first_poly[1], [dx, 0.0]);
    assert_eq!(first_poly[2], [dx, dx * dx]);
    assert_eq!(first_poly[3], [0.0, 0.0]);

    // Verify last polygon geometry
    let last_poly = &tool.cached_trapezoid_polygons[24];
    let last_x0 = 24.0 * dx;
    let last_x1 = 3.0;
    assert_eq!(last_poly[0], [last_x0, 0.0]);
    assert_eq!(last_poly[1], [last_x1, 0.0]);
    assert_eq!(last_poly[2], [last_x1, last_x1 * last_x1]);
    assert_eq!(last_poly[3], [last_x0, last_x0 * last_x0]);
}

#[test]
fn test_riemann_recalculate_invalid_bounds_clearing() {
    let mut tool = RiemannIntegrationTool::default();
    assert!(!tool.cached_trapezoid_polygons.is_empty());
    assert!(!tool.cached_true_curve.is_empty());

    // When min_x > max_x
    tool.min_x = 5.0;
    tool.max_x = 2.0;
    tool.recalculate();
    assert_eq!(tool.result, 0.0);
    assert!(tool.cached_trapezoid_polygons.is_empty());
    assert!(tool.cached_true_curve.is_empty());

    // When min_x == max_x
    tool.min_x = 2.0;
    tool.max_x = 2.0;
    tool.recalculate();
    assert_eq!(tool.result, 0.0);
    assert!(tool.cached_trapezoid_polygons.is_empty());
    assert!(tool.cached_true_curve.is_empty());
}

#[test]
fn test_riemann_recalculate_zero_steps() {
    let mut tool = RiemannIntegrationTool::default();
    assert!(!tool.cached_trapezoid_polygons.is_empty());
    assert!(!tool.cached_true_curve.is_empty());

    tool.steps = 0;
    tool.recalculate();
    assert_eq!(tool.result, 0.0);
    assert!(tool.cached_trapezoid_polygons.is_empty());
    assert!(tool.cached_true_curve.is_empty());
}

#[test]
fn test_riemann_plot_screenshot_generation() {
    let tool = RiemannIntegrationTool::default();
    assert!(!tool.cached_true_curve.is_empty());

    let datasets = vec![("sin(x)".to_string(), tool.cached_true_curve.clone())];
    let options = egui_plot::export::ExportOptions::default();
    if let Ok(png_data) = egui_plot::export::export_custom_png(&datasets, &options) {
        let _ = std::fs::write("/tmp/riemann_memoized_plot.png", png_data);
    }
}

#[test]
fn test_riemann_exp_guard_and_state_bounds() {
    let mut tool = RiemannIntegrationTool {
        preset: FunctionPreset::EToX,
        min_x: 0.0,
        max_x: 900.0,
        steps: 40,
        ..Default::default()
    };
    tool.recalculate();
    assert!(tool.result.is_finite());
    assert!(!tool.has_finite_parameters());

    assert!(tool
        .try_load_state(r#"{"preset":"EToX","min_x":-50.0,"max_x":50.0,"steps":50}"#)
        .is_ok());
    assert!(tool.result.is_finite());
    assert!(tool.has_finite_parameters());

    assert!(tool
        .try_load_state(r#"{"preset":"EToX","min_x":-55.0,"max_x":50.0,"steps":50}"#)
        .is_err());
}
