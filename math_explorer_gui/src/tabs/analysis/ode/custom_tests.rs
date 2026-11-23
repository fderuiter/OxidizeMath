#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::framework::InteractiveTool;

#[test]
fn test_ode_custom_sliders_persistence_and_legacy_fallback() {
    let mut tool = OdeSolverTool {
        preset: OdePreset::Custom,
        ic_y0: 2.0,
        ..Default::default()
    };
    tool.math_input.set_expression("r * y * (1 - y / cap_k)");
    tool.recalculate();
    assert_eq!(tool.math_input.active_param_names(), &["r", "cap_k"]);
    assert_eq!(tool.math_input.param_slice(), &[1.0, 10.0]);
    assert_eq!(tool.param_summary(), r"r=1.00, cap_k=10.00, \Delta t=0.05");

    tool.math_input
        .set_expression("-alpha * |y| * sin^2(t) + beta");
    let rev_before = tool.state_revision();
    let params = std::collections::BTreeMap::from([("alpha".into(), 1.6), ("beta".into(), 0.4)]);
    tool.math_input.set_params_map(&params);
    tool.recalculate();
    assert!(!tool.diverged() && tool.state_revision() != rev_before);
    assert_eq!(
        tool.param_summary(),
        r"alpha=1.60, beta=0.40, \Delta t=0.05"
    );

    let saved = tool.save_state().expect("save custom params state");
    assert!(saved.contains(r#""custom_params""#));
    let mut restored = OdeSolverTool::default();
    restored.load_state(&saved);
    assert_eq!(restored.math_input.param_slice(), &[1.6, 0.4]);
    assert_eq!(restored.y_series, tool.y_series);

    let legacy_json = r#"{"preset":"Custom","dt":0.05,"total_time":2.0,"param_k":1.0,"param_r":0.75,"param_cap_k":20.0,"ic_y0":1.5,"ic_v0":0.0,"custom_expr":"r * y * (1 - y / cap_k)"}"#;
    let mut from_legacy = OdeSolverTool::default();
    from_legacy.load_state(legacy_json);
    assert_eq!(from_legacy.math_input.param_slice(), &[0.75, 20.0]);
    assert!(!from_legacy.diverged());
}

#[test]
fn test_ode_custom_edit_fallback_and_no_duplicate_k_slider() {
    let mut tool = OdeSolverTool {
        preset: OdePreset::Custom,
        ic_y0: -1.5,
        ..Default::default()
    };
    tool.math_input.set_expression("-k * |y| * sin^2(t)");
    tool.recalculate();
    let valid_series = tool.y_series.clone();
    assert!(!tool.diverged() && !valid_series.is_empty());

    tool.math_input.set_expression("-k * |y| +");
    tool.recalculate();
    assert!(!tool.math_input.is_valid() && !tool.diverged());
    assert_eq!(tool.y_series, valid_series);

    tool.math_input
        .set_expression("p1 + p2 + p3 + p4 + p5 + p6 + p7 + y");
    tool.recalculate();
    let err = tool.math_input.error_message().expect("max params error");
    assert!(err.contains("maximum 6") && !tool.diverged());
    assert_eq!(tool.y_series, valid_series);

    tool.math_input.set_expression("-alpha * y");
    tool.recalculate();
    let ctx = egui::Context::default();
    let out = ctx.run(egui::RawInput::default(), |ctx| tool.show(ctx));
    for cs in &out.shapes {
        if let egui::epaint::Shape::Text(t) = &cs.shape {
            assert!(!t.galley.job.text.contains("k (Rate / Parameter)"));
        }
    }
}

fn collect_all_texts(shapes: &[egui::epaint::Shape], out: &mut Vec<String>) {
    for shape in shapes {
        match shape {
            egui::epaint::Shape::Text(t) => out.push(t.galley.job.text.clone()),
            egui::epaint::Shape::Vec(inner) => collect_all_texts(inner, out),
            _ => {}
        }
    }
}

#[test]
fn test_ode_sin_x_and_incomplete_sin_hide_solver_divergence_warning() {
    let ctx = egui::Context::default();
    for bad_expr in ["sin(x)", "sin("] {
        // Test both after a genuine numerical blow-up and from a clean state
        let mut tool = OdeSolverTool {
            preset: OdePreset::Custom,
            ic_y0: 1.0,
            ..Default::default()
        };
        tool.math_input.set_expression("1 / (y - 1)");
        tool.recalculate();
        assert!(tool.diverged(), "1 / (y - 1) at y0=1 must numerically blow up");

        tool.math_input.set_expression(bad_expr);
        tool.recalculate();
        assert!(
            !tool.math_input.is_valid(),
            "'{bad_expr}' must report an inline expression error"
        );
        assert!(
            !tool.diverged() && tool.diagnostic_report().is_none(),
            "'{bad_expr}' must leave diverged false without diagnostic report"
        );

        let out = ctx.run(egui::RawInput::default(), |ctx| tool.show(ctx));
        let shapes: Vec<_> = out.shapes.into_iter().map(|cs| cs.shape).collect();
        let mut texts = Vec::new();
        collect_all_texts(&shapes, &mut texts);
        assert!(
            texts.iter().any(|t| t.contains("Syntax Error:")),
            "expected inline error badge for '{bad_expr}'"
        );
        assert!(
            !texts.iter().any(|t| t.contains("Solver Divergence Warning")),
            "must hide Solver Divergence Warning for '{bad_expr}', got {texts:?}"
        );
    }
}

#[test]
fn test_ode_custom_logistic_exposes_r_and_k_sliders_and_recovers_nonfinite_params() {
    let ctx = egui::Context::default();
    let mut tool = OdeSolverTool {
        preset: OdePreset::Custom,
        ic_y0: 1.0,
        ..Default::default()
    };
    tool.math_input.set_expression("r * y * (1 - y / cap_k)");
    tool.recalculate();

    let out = ctx.run(egui::RawInput::default(), |ctx| tool.show(ctx));
    let shapes: Vec<_> = out.shapes.into_iter().map(|cs| cs.shape).collect();
    let mut texts = Vec::new();
    collect_all_texts(&shapes, &mut texts);
    assert!(
        texts.iter().any(|t| t.contains("r (Growth Rate)")),
        "expected 'r (Growth Rate)' slider in {texts:?}"
    );
    assert!(
        texts.iter().any(|t| t.contains("K (Carrying Capacity)")),
        "expected 'K (Carrying Capacity)' slider in {texts:?}"
    );

    // Non-finite param_r and param_cap_k in Custom mode are caught and clamped cleanly
    tool.param_r = f64::NAN;
    tool.param_cap_k = f64::INFINITY;
    assert!(tool.diverged());
    tool.clamp_instability_parameters();
    assert!(!tool.diverged());
    assert!((tool.param_r - 1.0).abs() < 1e-12);
    assert!((tool.param_cap_k - 10.0).abs() < 1e-12);
}
