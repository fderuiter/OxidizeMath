#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::budget::{MAX_CUSTOM_ODE_WORK_BUDGET, MAX_SENSITIVITY_ODE_WORK_BUDGET};
use super::*;
use crate::app::MathExplorerApp;
use crate::framework::InteractiveTool;
use crate::presets::checked_ode_step_count;
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
fn test_ode_rejects_invalid_and_oversized_balanced_custom_expr_quickly() {
    let expr_h8 = build_balanced_expr("y", 8);
    let expr_h19 = build_balanced_expr("y", 19);

    for bad in ["sin(", "x + y", &expr_h8, &expr_h19] {
        let state = format!(
            r#"{{"preset":"Custom","dt":0.01,"total_time":10.0,"param_k":1.0,"param_r":1.0,"param_cap_k":10.0,"ic_y0":1.0,"ic_v0":0.0,"custom_expr":"{bad}"}}"#
        );
        let preset = format!(
            r#"{{"metadata":{{"schema_version":"1.0","title":"Balanced","domain":"analysis","timestamp":"2026-11-23T00:00:00Z"}},"config":{state}}}"#
        );

        let mut tool = OdeSolverTool::default();
        let t0 = PlatformInstant::now();
        assert!(tool.import_preset_json(&preset).is_err());
        assert!(
            t0.elapsed().as_millis() < 250,
            "ODE preset rejection took {:?} (>= 250 ms)",
            t0.elapsed()
        );
        assert_eq!(tool.preset, OdePreset::Exponential);
        assert!(tool.math_input.is_valid());
    }

    // Also verify share link hydration via MathExplorerApp::apply_share_location_strings
    let encoded_h8 = crate::url_state::encode_base64url(
        format!(
            r#"{{"preset":"Custom","dt":0.001,"total_time":100.0,"param_k":1.0,"param_r":1.0,"param_cap_k":10.0,"ic_y0":1.0,"ic_v0":0.0,"custom_expr":"{expr_h8}"}}"#
        )
        .as_bytes(),
    );
    let mut app = MathExplorerApp::default();
    let t_share = PlatformInstant::now();
    let res = app.apply_share_location_strings(
        &format!("#tab=analysis&tool=OdeSolverTool&state={encoded_h8}"),
        "",
        None,
    );
    assert!(res.is_err());
    assert!(t_share.elapsed().as_millis() < 250);

    // Also reject invalid or oversized PlanarCustom AST payloads
    let bad_planar_state = format!(
        r#"{{"preset":"PlanarCustom","dt":0.01,"total_time":10.0,"param_k":1.0,"param_r":1.0,"param_cap_k":10.0,"ic_y0":1.0,"ic_v0":0.0,"planar_x_expr":"y","planar_y_expr":"{expr_h8}"}}"#
    );
    let mut planar_tool = OdeSolverTool::default();
    assert!(planar_tool.try_load_state(&bad_planar_state).is_err());
    assert_eq!(planar_tool.preset, OdePreset::Exponential);
}

#[test]
fn test_ode_work_budget_clamps_steps_times_ast_nodes_and_sensitivity() {
    // Height-7 balanced expression has 255 AST nodes (<= 256 limit, valid in ODE scope).
    let expr_h7 = build_balanced_expr("y", 7);
    let state = format!(
        r#"{{"preset":"Custom","dt":0.001,"total_time":100.0,"param_k":1.0,"param_r":1.0,"param_cap_k":10.0,"ic_y0":0.01,"ic_v0":0.0,"comparison_mode":true,"custom_expr":"-{expr_h7} * 0.001 + k * 0"}}"#
    );
    let preset = format!(
        r#"{{"metadata":{{"schema_version":"1.0","title":"Bounded256","domain":"analysis","timestamp":"2026-11-23T00:00:00Z"}},"config":{state}}}"#
    );

    let mut tool = OdeSolverTool::default();
    let start = PlatformInstant::now();
    // Note: -{expr_h7} * 0.001 + k * 0 is > 256 nodes (255 + 6 = 261), so use exact expr_h7 (255 nodes)!
    assert!(tool.import_preset_json(&preset).is_err());

    let valid_state = format!(
        r#"{{"preset":"Custom","dt":0.001,"total_time":10.0,"param_k":1.0,"param_r":1.0,"param_cap_k":10.0,"ic_y0":0.15,"ic_v0":0.0,"comparison_mode":true,"custom_expr":"{expr_h7}"}}"#
    );
    let valid_preset = format!(
        r#"{{"metadata":{{"schema_version":"1.0","title":"Bounded255","domain":"analysis","timestamp":"2026-11-23T00:00:00Z"}},"config":{valid_state}}}"#
    );
    tool.import_preset_json(&valid_preset)
        .expect("255-node custom ODE should load and clamp dt");
    let ast_nodes = tool.custom_ast_node_count();
    assert_eq!(ast_nodes, 255);
    let steps = checked_ode_step_count(tool.dt, tool.total_time).expect("clamped steps");
    assert!(
        steps * ast_nodes <= MAX_CUSTOM_ODE_WORK_BUDGET,
        "steps ({steps}) * ast_nodes ({ast_nodes}) exceeded {MAX_CUSTOM_ODE_WORK_BUDGET}"
    );

    let sens_steps = checked_ode_step_count(tool.sensitivity_effective_dt(), tool.total_time)
        .expect("sensitivity steps");
    assert!(sens_steps * ast_nodes <= MAX_SENSITIVITY_ODE_WORK_BUDGET);

    let ctx = egui::Context::default();
    let _ = ctx.run(egui::RawInput::default(), |c| tool.show(c));
    assert!(
        start.elapsed().as_millis() < 250,
        "Bounded 255-node ODE preset + UI frame took {:?} (>= 250 ms)",
        start.elapsed()
    );

    let datasets = tool.export_datasets();
    let png = egui_plot::export::export_custom_png(
        &datasets,
        &egui_plot::export::ExportOptions::default(),
    )
    .expect("bounded ODE PNG export");
    let _ = std::fs::write("/tmp/ode_bounded_ast_budget.png", png);
}
