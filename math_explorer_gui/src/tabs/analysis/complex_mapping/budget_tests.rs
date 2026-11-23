#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::ui::CONTOUR_RESOLUTION;
use super::*;
use crate::app::MathExplorerApp;
use crate::framework::InteractiveTool;
use crate::web_io::PlatformInstant;

fn build_balanced_expr(var: &str, height: usize) -> String {
    let mut cur = var.to_string();
    for level in 1..=height {
        cur = match level % 3 {
            1 => format!("{cur}*{cur}"),
            2 => format!("{cur}+{cur}"),
            _ => format!("({cur})({cur})"),
        };
    }
    cur
}

#[test]
fn test_complex_mapping_rejects_invalid_and_oversized_balanced_custom_expr() {
    let expr_h8 = build_balanced_expr("z", 8);
    let expr_h10 = build_balanced_expr("z", 10);

    for bad in ["z +", "x + y", &expr_h8, &expr_h10] {
        let state = format!(
            r#"{{"function":"Custom","grid_min":-2.0,"grid_max":2.0,"grid_density":100,"inspection_point":[1.0,0.5],"custom_expr":"{bad}"}}"#
        );
        let preset = format!(
            r#"{{"metadata":{{"schema_version":"1.0","title":"BadComplex","domain":"analysis","timestamp":"2026-11-23T00:00:00Z"}},"config":{state}}}"#
        );

        let mut tool = ComplexMappingTool::default();
        let t0 = PlatformInstant::now();
        assert!(tool.import_preset_json(&preset).is_err());
        assert!(
            t0.elapsed().as_millis() < 250,
            "ComplexMapping preset rejection took {:?} (>= 250 ms)",
            t0.elapsed()
        );
        assert_eq!(tool.function, MappingFunction::ZSquared);
        assert!(tool.math_input.is_valid());
    }

    // Verify share URL hydration via MathExplorerApp::apply_share_location_strings
    let encoded_h10 = crate::url_state::encode_base64url(
        format!(
            r#"{{"function":"Custom","grid_min":-2.0,"grid_max":2.0,"grid_density":1000,"inspection_point":[1.0,0.5],"custom_expr":"{expr_h10}"}}"#
        )
        .as_bytes(),
    );
    let mut app = MathExplorerApp::default();
    let t_share = PlatformInstant::now();
    let res = app.apply_share_location_strings(
        &format!("#tab=analysis&tool=ComplexMappingTool&state={encoded_h10}"),
        "",
        None,
    );
    assert!(res.is_err());
    assert!(t_share.elapsed().as_millis() < 250);
}

#[test]
fn test_complex_mapping_caps_grid_points_times_ast_nodes_budget() {
    // Height-7 balanced expression has 255 AST nodes (<= 256 limit, valid in ComplexMapping scope).
    let expr_h7 = build_balanced_expr("z", 7);
    let state = format!(
        r#"{{"function":"Custom","grid_min":-0.6,"grid_max":0.6,"grid_density":1000,"inspection_point":[0.25,0.15],"custom_expr":"{expr_h7}"}}"#
    );
    let preset = format!(
        r#"{{"metadata":{{"schema_version":"1.0","title":"BoundedComplex255","domain":"analysis","timestamp":"2026-11-23T00:00:00Z"}},"config":{state}}}"#
    );

    let mut tool = ComplexMappingTool::default();
    let start = PlatformInstant::now();
    tool.import_preset_json(&preset)
        .expect("255-node custom ComplexMapping should load and clamp grid_density");
    let ast_nodes = tool.custom_ast_node_count();
    assert_eq!(ast_nodes, 255);
    let grid_points = 2 * tool.grid_density * CONTOUR_RESOLUTION;
    assert!(
        grid_points * ast_nodes <= MAX_COMPLEX_MAPPING_WORK_BUDGET,
        "grid_points ({grid_points}) * ast_nodes ({ast_nodes}) exceeded {MAX_COMPLEX_MAPPING_WORK_BUDGET}"
    );

    let ctx = egui::Context::default();
    let _ = ctx.run(egui::RawInput::default(), |c| tool.show(c));
    assert!(
        start.elapsed().as_millis() < 250,
        "Bounded 255-node ComplexMapping preset + UI frame took {:?} (>= 250 ms)",
        start.elapsed()
    );

    let prims = tool.export_plot_primitives();
    let png = egui_plot::export_primitives::export_primitives_png(
        &prims,
        &egui_plot::export::ExportOptions::default(),
    )
    .expect("bounded ComplexMapping PNG export");
    let _ = std::fs::write("/tmp/complex_mapping_bounded_ast_budget.png", png);
}
