#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::export::{format_kelly_rows_csv, format_kelly_rows_markdown};
use super::{BetSizeCalculatorTool, KellyStrategyRow};
use crate::framework::InteractiveTool;
use egui_plot::PlotPoint;

#[test]
fn test_bet_size_default_populates_growth_curve_and_kelly_markers() {
    let tool = BetSizeCalculatorTool::default();
    let caps = tool.export_capabilities();
    assert!(caps.figure && caps.dataset);
    assert!(tool.calc_error.is_none());
    assert!(tool.growth_curve_points.len() > 50);
    assert_eq!(tool.quarter_marker.len(), 1);
    assert_eq!(tool.half_marker.len(), 1);
    assert_eq!(tool.full_marker.len(), 1);

    // Full Kelly maximizes g(f): g(0.25 f*) < g(0.50 f*) < g(1.00 f*)
    let q_pt = tool.quarter_marker[0];
    let h_pt = tool.half_marker[0];
    let f_pt = tool.full_marker[0];
    assert!(q_pt.x < h_pt.x && h_pt.x < f_pt.x);
    assert!(q_pt.y > 0.0 && q_pt.y < h_pt.y && h_pt.y < f_pt.y);

    // Verify zero-allocation borrowed PlotPoints slice
    let slice: &[PlotPoint] = &tool.growth_curve_points[..];
    let plot_pts: egui_plot::PlotPoints<'_> = slice.into();
    assert!(matches!(plot_pts, egui_plot::PlotPoints::Borrowed(_)));
}

#[test]
fn test_bet_size_csv_and_markdown_exports_escape_formula_injection() {
    let tool = BetSizeCalculatorTool::default();
    let csv = tool.export_sizing_csv();
    let md = tool.export_sizing_markdown();
    assert!(csv.contains("Quarter Kelly"));
    assert!(csv.contains("Half Kelly"));
    assert!(csv.contains("Full Kelly"));
    assert!(md.contains("| Quarter Kelly |"));
    assert!(md.contains("| Full Kelly |"));

    let injected = vec![KellyStrategyRow {
        label: "=CMD|' /C calc'!A0".to_string(),
        fraction: 0.1,
        bet_amount: 100.0,
        growth_rate: 0.005,
        ev_per_unit: 0.10,
    }];
    let safe_csv = format_kelly_rows_csv(&injected);
    assert!(
        safe_csv.contains("\"'=CMD|' /C calc'!A0\""),
        "Formula prefix must be neutralized with leading quote: {safe_csv}"
    );
    let safe_md = format_kelly_rows_markdown(&injected);
    assert!(safe_md.contains("\\|"));
}
