#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::budget::{
    MAX_ODE_COMPARISON_SYNC_INSTRUCTIONS, MAX_ODE_SYNC_INSTRUCTIONS, MIN_ODE_BUDGET_STEPS,
};
use super::*;
use crate::accessibility::{all_announcements, clear_test_announcements};
use crate::framework::{MAX_TOOL_STATE_BYTES, SimulationFramework};
use crate::presets::ReplicatorPresetConfig;
use crate::url_state::MAX_DECODED_URL_STATE_BYTES;
use crate::web_io::PlatformInstant;
use math_commons::ast::{MAX_AST_NODES, MAX_EXPR_BYTES};

/// Builds a valid scalar ODE expression `y' = -0.1*y + 0.002*sin(1*t + 0.05*y) + ...` (~201 instructions).
fn multi_node_scalar_ode_expr() -> String {
    let mut terms = vec!["-0.1 * y".to_string()];
    for idx in 1..=18 {
        terms.push(format!("0.002 * sin({idx} * t + 0.05 * y)"));
    }
    terms.join(" + ")
}

fn collect_text_shapes(shapes: &[egui::epaint::Shape], out: &mut Vec<String>) {
    for shape in shapes {
        match shape {
            egui::epaint::Shape::Text(t) => out.push(t.galley.job.text.clone()),
            egui::epaint::Shape::Vec(inner) => collect_text_shapes(inner, out),
            _ => {}
        }
    }
}

#[test]
fn test_ode_caps_sync_work_across_primary_comparison_and_sensitivity_at_max_steps() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    clear_test_announcements();

    let formula = multi_node_scalar_ode_expr();
    let mut tool = OdeSolverTool {
        preset: OdePreset::Custom,
        method: IntegrationMethod::RungeKutta4,
        dt: 0.0005,
        total_time: 100.0,
        ic_y0: 1.0,
        comparison_mode: true,
        ..Default::default()
    };
    tool.math_input.set_expression(&formula);
    assert!(tool.math_input.is_valid());
    let inst_cost = tool.rhs_instruction_cost();
    assert!(
        inst_cost >= 120,
        "expected >= 120 instructions, got {inst_cost}"
    );

    let start = PlatformInstant::now();
    tool.recalculate();
    tool.run_sensitivity_analysis(false);
    let elapsed = start.elapsed();
    assert!(
        elapsed.as_secs_f64() < 1.5,
        "expected synchronous recalculate + sensitivity to stay fast, took {:.3}s",
        elapsed.as_secs_f64()
    );

    let primary_steps = tool.y_series.len().saturating_sub(1);
    assert!(primary_steps >= MIN_ODE_BUDGET_STEPS);
    assert!(
        primary_steps.saturating_mul(4).saturating_mul(inst_cost) <= MAX_ODE_SYNC_INSTRUCTIONS,
        "primary trajectory exceeded instruction budget: steps={primary_steps}, inst={inst_cost}"
    );
    assert!(tool.budget_state.primary_notice.is_some());
    assert!(tool.budget_state.comparison_notice.is_some());
    assert!(tool.sensitivity_engine.budget_notice().is_some());

    // Comparison snapshots must share aligned timestamps and finite discrepancies.
    assert_eq!(tool.comparison_manager().len(), 3);
    let comp_steps = tool.comparison_manager().snapshots()[0].series[0]
        .points
        .len()
        .saturating_sub(1);
    assert!(
        comp_steps.saturating_mul(7).saturating_mul(inst_cost)
            <= MAX_ODE_COMPARISON_SYNC_INSTRUCTIONS
    );
    assert!(tool.discrepancy_euler_rk4().max_error.is_finite());
    assert!(tool.discrepancy_heun_rk4().max_error.is_finite());
    assert!(
        tool.discrepancy_heun_rk4().max_error <= tool.discrepancy_euler_rk4().max_error + 1e-6,
        "aligned comparison dt must preserve RK4/Heun/Euler accuracy ordering"
    );

    assert_ode_budget_ui_and_export(&mut tool);
}

fn assert_ode_budget_ui_and_export(tool: &mut OdeSolverTool) {
    let ctx = egui::Context::default();
    let out = ctx.run(egui::RawInput::default(), |c| tool.show(c));
    let mut texts = Vec::new();
    let shapes: Vec<_> = out.shapes.into_iter().map(|cs| cs.shape).collect();
    collect_text_shapes(&shapes, &mut texts);
    for expected in [
        "Solver work budget applied",
        "Multi-solver comparison budget applied",
        "Sensitivity budget applied",
    ] {
        assert!(
            texts.iter().any(|t| t.contains(expected)),
            "expected inline notice '{expected}' in {texts:?}"
        );
    }

    let first_pass = all_announcements();
    for expected in [
        "Solver work budget applied",
        "Multi-solver comparison budget applied",
        "Sensitivity budget applied",
    ] {
        assert!(
            first_pass
                .iter()
                .any(|(m, p)| p == "polite" && m.contains(expected))
        );
    }

    let count_after_first = first_pass.len();
    let _ = ctx.run(egui::RawInput::default(), |c| tool.show(c));
    assert_eq!(
        all_announcements().len(),
        count_after_first,
        "unchanged budget notices must not re-announce on idle frames"
    );

    let datasets = tool.export_datasets();
    let png = egui_plot::export::export_custom_png(
        &datasets,
        &egui_plot::export::ExportOptions::default(),
    )
    .expect("export budgeted ODE comparison PNG");
    assert!(png.starts_with(b"\x89PNG\r\n\x1a\n"));
    let _ = std::fs::write("/tmp/ode_solver_budget_comparison.png", png);
}

#[test]
fn test_ode_coupled_and_seeded_orbits_respect_sync_instruction_budget() {
    let mut tool = OdeSolverTool {
        preset: OdePreset::Custom,
        custom_coupled: true,
        phase_plane_mode: true,
        dt: 0.0005,
        total_time: 100.0,
        ..Default::default()
    };
    let dy = "v + 0.01*sin(t) + 0.01*cos(2*t) + 0.01*sin(3*t) + 0.01*cos(4*t) + 0.01*sin(5*t)";
    let dv = "-k*y - c*v + 0.01*sin(6*t) + 0.01*cos(7*t) + 0.01*sin(8*t) + 0.01*cos(9*t)";
    tool.coupled_bank.set_expressions(&[dy, dv]);
    assert!(tool.coupled_bank.is_valid());
    for idx in 0..8 {
        tool.seed_orbit([0.5 + 0.1 * idx as f64, -0.4 + 0.1 * idx as f64]);
    }
    let orbit_count = 1 + tool.seeded_trajectories.len();
    assert_eq!(orbit_count, 9);
    let steps_each = tool.y_series.len().saturating_sub(1);
    let total_inst = orbit_count
        .saturating_mul(steps_each)
        .saturating_mul(4)
        .saturating_mul(tool.rhs_instruction_cost());
    assert!(
        total_inst <= MAX_ODE_SYNC_INSTRUCTIONS,
        "combined primary + 8 seeded orbits exceeded instruction cap: {total_inst}"
    );
    assert!(tool.budget_state.primary_notice.is_some());
}

#[test]
fn test_url_4096b_and_2mib_state_payloads_reject_abusive_formulas_and_bound_hydration() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    clear_test_announcements();

    // 1. Balanced ternary tree (height 4 <= 64, 362 tokens <= 512, 283 AST nodes > 256) in < 4,096B URL is rejected atomically.
    let mut many_nodes = String::from("clamp(t y,t y,t y)");
    for _ in 0..3 {
        many_nodes = format!("clamp({many_nodes},{many_nodes},{many_nodes})");
    }
    assert!(many_nodes.len() < MAX_EXPR_BYTES);
    let url_json = format!(
        r#"{{"metadata":{{"schema_version":"1.0","title":"Abusive","domain":"analysis","timestamp":"2026-11-23T00:00:00Z"}},"config":{{"preset":"Custom","dt":0.0005,"total_time":100.0,"param_k":1.0,"param_r":1.0,"param_cap_k":10.0,"ic_y0":1.0,"ic_v0":0.0,"custom_expr":"{many_nodes}"}}}}"#
    );
    assert!(url_json.len() <= MAX_DECODED_URL_STATE_BYTES);

    let mut fw = SimulationFramework::new("analysis");
    fw.active_tool = Some(Box::new(OdeSolverTool::default()));
    let err = fw
        .import_url_preset_json(Some("OdeSolverTool"), Some(&url_json))
        .expect_err("expected URL hydration with > MAX_AST_NODES formula to fail");
    assert!(
        err.contains(&MAX_AST_NODES.to_string()) || err.contains("complexity"),
        "unexpected error: {err}"
    );
    assert!(
        all_announcements()
            .iter()
            .any(|(m, p)| p == "assertive" && m.contains("Invalid OdeSolverTool custom formula")),
        "expected assertive live-region announcement on hydration rejection"
    );

    // 2. Oversized > 2,048-byte custom expression inside a < 2 MiB tool state payload is rejected in O(1).
    let huge_expr = format!("y + {}", "1 + ".repeat(800));
    assert!(huge_expr.len() > MAX_EXPR_BYTES);
    let state_json = format!(
        r#"{{"preset":"Custom","dt":0.0005,"total_time":100.0,"param_k":1.0,"param_r":1.0,"param_cap_k":10.0,"ic_y0":1.0,"ic_v0":0.0,"custom_expr":"{huge_expr}1"}}"#
    );
    assert!(state_json.len() < MAX_TOOL_STATE_BYTES);
    let mut hydrated = OdeSolverTool::default();
    let res = hydrated.try_load_state(&state_json);
    assert!(res.is_err());

    // 3. Valid multi-node formula within 4,096-byte URL cap at MAX_ODE_STEPS hydrates rapidly under budget.
    let valid_heavy = multi_node_scalar_ode_expr();
    let valid_url_json = format!(
        r#"{{"metadata":{{"schema_version":"1.0","title":"HeavyValid","domain":"analysis","timestamp":"2026-11-23T00:00:00Z"}},"config":{{"preset":"Custom","dt":0.0005,"total_time":100.0,"param_k":1.0,"param_r":1.0,"param_cap_k":10.0,"ic_y0":1.0,"ic_v0":0.0,"comparison_mode":true,"custom_expr":"{valid_heavy}"}}}}"#
    );
    assert!(valid_url_json.len() <= MAX_DECODED_URL_STATE_BYTES);
    let t0 = PlatformInstant::now();
    fw.import_url_preset_json(Some("OdeSolverTool"), Some(&valid_url_json))
        .expect("valid heavy URL preset should hydrate cleanly under budget");
    assert!(t0.elapsed().as_secs_f64() < 1.0);
}

#[test]
fn test_builtin_ode_presets_unscaled_and_replicator_symbolic_validation() {
    for preset in [
        OdePreset::Exponential,
        OdePreset::HarmonicOscillator,
        OdePreset::LogisticGrowth,
        OdePreset::LinearSystem2D,
    ] {
        let mut tool = OdeSolverTool {
            preset,
            dt: 0.0005,
            total_time: 100.0,
            comparison_mode: true,
            ..Default::default()
        };
        tool.recalculate();
        assert!(
            tool.budget_state.primary_notice.is_none(),
            "built-in preset {preset:?} must not trigger primary budget scaling"
        );
        assert!(
            tool.budget_state.comparison_notice.is_none(),
            "built-in preset {preset:?} must not trigger comparison budget scaling"
        );
    }

    let valid_rep = ReplicatorPresetConfig {
        payoff_matrix: vec![vec![0.0, 2.0], vec![1.0, 0.0]],
        initial_population: vec![0.5, 0.5],
        time_horizon: 10.0,
        dt: 0.05,
        strategy_names: vec!["Hawk".to_string(), "Dove".to_string()],
        symbolic_payoffs: Some(vec![
            vec!["0".to_string(), "2 * r - c".to_string()],
            vec!["r".to_string(), "(r - c) / 2".to_string()],
        ]),
        symbolic_params: std::collections::BTreeMap::from([
            ("r".to_string(), 2.0),
            ("c".to_string(), 3.0),
        ]),
        topology_source: crate::app::TopologySource::SharedBus,
    };
    assert!(valid_rep.validate().is_ok());

    let too_many_params = (1..=17)
        .map(|idx| format!("p{idx}"))
        .collect::<Vec<_>>()
        .join("+");
    let bad_rep = ReplicatorPresetConfig {
        symbolic_payoffs: Some(vec![
            vec![too_many_params, "1".to_string()],
            vec!["1".to_string(), "0".to_string()],
        ]),
        ..valid_rep
    };
    let rep_err = bad_rep
        .validate()
        .expect_err("symbolic payoff formula exceeding compiled param cap must fail validation");
    assert!(rep_err.contains("Symbolic payoff") && rep_err.contains("(0, 0)"));
}
