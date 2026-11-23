#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::tabs::ai::activation_functions::{ActivationFunction, ActivationFunctionsTool};
use crate::tabs::ai::loss_landscape::LossLandscapeTool;
use crate::tabs::analysis::complex_mapping::ComplexMappingTool;
use crate::tabs::analysis::riemann::RiemannIntegrationTool;
use crate::tabs::chaos::fractals::{FractalMode, FractalViewer};
use crate::tabs::financial_math::bet_size::{BetSizeCalculatorTool, OddsFormat};
use crate::tabs::fluid_dynamics::potential_flow::{ElementType, PotentialFlowTool};
use eframe::egui;

fn screen_input(events: Vec<egui::Event>) -> egui::RawInput {
    egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1280.0, 900.0),
        )),
        events,
        ..Default::default()
    }
}

fn find_drag_value_centers(
    ctx: &egui::Context,
    render: &mut dyn FnMut(&egui::Context),
) -> Vec<egui::Pos2> {
    ctx.enable_accesskit();
    let out = ctx.run(screen_input(Vec::new()), |ctx| render(ctx));
    let mut slider_ys = Vec::new();
    let mut spin_centers = Vec::new();
    if let Some(update) = &out.platform_output.accesskit_update {
        for (_id, node) in &update.nodes {
            if let Some(b) = node.bounds() {
                let cy = ((b.y0 + b.y1) * 0.5) as f32;
                let cx = ((b.x0 + b.x1) * 0.5) as f32;
                if node.role() == egui::accesskit::Role::Slider {
                    slider_ys.push(cy);
                } else if node.role() == egui::accesskit::Role::SpinButton {
                    spin_centers.push(egui::pos2(cx, cy));
                }
            }
        }
    }
    let mut centers: Vec<egui::Pos2> = spin_centers
        .into_iter()
        .filter(|c| slider_ys.iter().all(|sy| (c.y - sy).abs() > 6.0))
        .collect();
    centers.sort_by(|a, b| a.y.total_cmp(&b.y).then_with(|| a.x.total_cmp(&b.x)));
    centers
}

fn click_at(ctx: &egui::Context, pos: egui::Pos2, render: &mut dyn FnMut(&egui::Context)) {
    let _ = ctx.run(
        screen_input(vec![
            egui::Event::PointerMoved(pos),
            egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed: true,
                modifiers: egui::Modifiers::NONE,
            },
            egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::NONE,
            },
        ]),
        |ctx| render(ctx),
    );
}

fn submit_drag_value_text(
    ctx: &egui::Context,
    pos: egui::Pos2,
    text: &str,
    render: &mut dyn FnMut(&egui::Context),
) {
    click_at(ctx, pos, render);
    let _ = ctx.run(
        screen_input(vec![
            egui::Event::Text(text.to_string()),
            egui::Event::Key {
                key: egui::Key::Enter,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: egui::Modifiers::NONE,
            },
            egui::Event::Key {
                key: egui::Key::Enter,
                physical_key: None,
                pressed: false,
                repeat: false,
                modifiers: egui::Modifiers::NONE,
            },
        ]),
        |ctx| render(ctx),
    );
    let _ = ctx.run(screen_input(Vec::new()), |ctx| render(ctx));
}

fn verify_fractal_drag_values() -> usize {
    let ctx = egui::Context::default();
    let mut tool = FractalViewer {
        mode: FractalMode::Julia,
        ..Default::default()
    };
    let centers = find_drag_value_centers(&ctx, &mut |c| tool.show(c));
    assert_eq!(centers.len(), 4);

    submit_drag_value_text(&ctx, centers[0], "1.25", &mut |c| tool.show(c));
    assert!((tool.julia_c.re - 1.25).abs() < 1e-6);
    for &pos in &centers {
        submit_drag_value_text(&ctx, pos, "1e400", &mut |c| tool.show(c));
        assert!(tool.has_finite_parameters());
        submit_drag_value_text(&ctx, pos, "999.0", &mut |c| tool.show(c));
        assert!(tool.has_finite_parameters());
    }
    assert!((tool.julia_c.re - 4.0).abs() < 1e-6);
    assert!((tool.center.im - 4.0).abs() < 1e-6);
    centers.len()
}

fn verify_activation_drag_values() -> usize {
    let ctx = egui::Context::default();
    let mut tool = ActivationFunctionsTool {
        selected_function: ActivationFunction::Gelu,
        ..Default::default()
    };
    let centers = find_drag_value_centers(&ctx, &mut |c| tool.show(c));
    assert_eq!(centers.len(), 2);

    for &pos in &centers {
        submit_drag_value_text(&ctx, pos, "1e400", &mut |c| tool.show(c));
        assert!(tool.has_finite_parameters());
    }
    submit_drag_value_text(&ctx, centers[0], "100.0", &mut |c| tool.show(c));
    assert!(tool.x_min < tool.x_max && tool.has_finite_parameters());
    submit_drag_value_text(&ctx, centers[1], "-100.0", &mut |c| tool.show(c));
    assert!(tool.x_min < tool.x_max && tool.has_finite_parameters());
    centers.len()
}

fn verify_loss_landscape_drag_values() -> usize {
    let ctx = egui::Context::default();
    let mut tool = LossLandscapeTool::default();
    let centers = find_drag_value_centers(&ctx, &mut |c| tool.show(c));
    assert_eq!(centers.len(), 4);

    for &pos in &centers {
        submit_drag_value_text(&ctx, pos, "1e400", &mut |c| tool.show(c));
        assert!(tool.has_finite_parameters());
    }
    submit_drag_value_text(&ctx, centers[0], "500.0", &mut |c| tool.show(c));
    submit_drag_value_text(&ctx, centers[3], "-500.0", &mut |c| tool.show(c));
    assert!(tool.w_min < tool.w_max && tool.b_min < tool.b_max);
    assert!(tool.has_finite_parameters());
    centers.len()
}

fn verify_complex_mapping_drag_values() -> usize {
    let ctx = egui::Context::default();
    let mut tool = ComplexMappingTool::default();
    let centers = find_drag_value_centers(&ctx, &mut |c| tool.show(c));
    assert_eq!(centers.len(), 4);

    for &pos in &centers {
        submit_drag_value_text(&ctx, pos, "1e400", &mut |c| tool.show(c));
        assert!(tool.has_finite_parameters());
    }
    submit_drag_value_text(&ctx, centers[0], "200.0", &mut |c| tool.show(c));
    submit_drag_value_text(&ctx, centers[2], "-200.0", &mut |c| tool.show(c));
    assert!(tool.grid_min < tool.grid_max);
    assert!((tool.inspection_point[0] - (-50.0)).abs() < 1e-6);
    assert!(tool.has_finite_parameters());
    centers.len()
}

fn verify_riemann_drag_values() -> usize {
    let ctx = egui::Context::default();
    let mut tool = RiemannIntegrationTool::default();
    let centers = find_drag_value_centers(&ctx, &mut |c| tool.show(c));
    assert_eq!(centers.len(), 2);

    for &pos in &centers {
        submit_drag_value_text(&ctx, pos, "1e400", &mut |c| tool.show(c));
        assert!(tool.has_finite_parameters());
    }
    submit_drag_value_text(&ctx, centers[0], "999.0", &mut |c| tool.show(c));
    assert!(tool.min_x < tool.max_x && tool.has_finite_parameters());
    centers.len()
}

fn verify_potential_flow_drag_values() -> usize {
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let mut tool = PotentialFlowTool {
        selected_element_type: ElementType::Source,
        ..Default::default()
    };
    let out = ctx.run(screen_input(Vec::new()), |c| tool.show(c));
    let header_pos = out
        .platform_output
        .accesskit_update
        .as_ref()
        .and_then(|u| {
            u.nodes
                .iter()
                .find(|(_, n)| n.label().is_some_and(|l| l.contains("Add Element")))
                .and_then(|(_, n)| n.bounds())
                .map(|b| egui::pos2(((b.x0 + b.x1) * 0.5) as f32, ((b.y0 + b.y1) * 0.5) as f32))
        })
        .expect("Add Element collapsing header must exist");
    click_at(&ctx, header_pos, &mut |c| tool.show(c));

    let centers = find_drag_value_centers(&ctx, &mut |c| tool.show(c));
    assert_eq!(centers.len(), 3);
    for &pos in &centers {
        submit_drag_value_text(&ctx, pos, "1e400", &mut |c| tool.show(c));
        assert!(tool.has_finite_parameters());
    }
    submit_drag_value_text(&ctx, centers[0], "99999.0", &mut |c| tool.show(c));
    submit_drag_value_text(&ctx, centers[1], "-999.0", &mut |c| tool.show(c));
    submit_drag_value_text(&ctx, centers[2], "999.0", &mut |c| tool.show(c));
    assert!((tool.new_element_strength - 1_000.0).abs() < 1e-6);
    assert!((tool.new_element_x - (-50.0)).abs() < 1e-6);
    assert!((tool.new_element_y - 50.0).abs() < 1e-6);
    assert!(tool.has_finite_parameters());

    let (origins, tips) = tool.compute_arrow_field();
    assert!(!origins.is_empty());
    assert!(origins
        .iter()
        .chain(tips.iter())
        .all(|p| p[0].is_finite() && p[1].is_finite()));
    centers.len()
}

#[test]
fn test_all_19_drag_value_widgets_reject_1e400_and_preserve_strict_ordering() {
    let total = verify_fractal_drag_values()
        + verify_activation_drag_values()
        + verify_loss_landscape_drag_values()
        + verify_complex_mapping_drag_values()
        + verify_riemann_drag_values()
        + verify_potential_flow_drag_values();
    assert_eq!(total, 19);
}

#[test]
fn test_bet_size_drag_values_sync_formats_and_skip_american_dead_zone() {
    let ctx = egui::Context::default();
    let mut tool = BetSizeCalculatorTool::default();
    let dec_centers = find_drag_value_centers(&ctx, &mut |c| tool.show(c));
    assert_eq!(dec_centers.len(), 3);
    for &pos in &dec_centers {
        submit_drag_value_text(&ctx, pos, "1e400", &mut |c| tool.show(c));
        assert!(tool.has_finite_parameters());
    }
    submit_drag_value_text(&ctx, dec_centers[1], "2.5", &mut |c| tool.show(c));
    assert!((tool.american_odds - 150.0).abs() < 1e-6);
    assert!((tool.fractional_numerator - 3.0).abs() < 1e-6);
    assert!((tool.fractional_denominator - 2.0).abs() < 1e-6);
    assert!((tool.implied_probability_input - 40.0).abs() < 1e-6);

    tool.odds_format = OddsFormat::American;
    let am_centers = find_drag_value_centers(&ctx, &mut |c| tool.show(c));
    assert_eq!(am_centers.len(), 3);
    submit_drag_value_text(&ctx, am_centers[1], "99.0", &mut |c| tool.show(c));
    assert!((tool.american_odds - (-100.0)).abs() < 1e-6);
    assert!((tool.decimal_odds - 2.0).abs() < 1e-6);
    submit_drag_value_text(&ctx, am_centers[1], "-99.0", &mut |c| tool.show(c));
    assert!((tool.american_odds - 100.0).abs() < 1e-6);
    assert!(tool.has_finite_parameters());

    tool.odds_format = OddsFormat::Fractional;
    let frac_centers = find_drag_value_centers(&ctx, &mut |c| tool.show(c));
    assert_eq!(frac_centers.len(), 4);
    for &pos in &frac_centers {
        submit_drag_value_text(&ctx, pos, "1e400", &mut |c| tool.show(c));
        assert!(tool.has_finite_parameters());
    }

    tool.odds_format = OddsFormat::ImpliedProbability;
    let imp_centers = find_drag_value_centers(&ctx, &mut |c| tool.show(c));
    assert_eq!(imp_centers.len(), 3);
    submit_drag_value_text(&ctx, imp_centers[1], "25.0", &mut |c| tool.show(c));
    assert!((tool.decimal_odds - 4.0).abs() < 1e-6);
    assert!((tool.american_odds - 300.0).abs() < 1e-6);
    assert!(tool.has_finite_parameters());
}

fn verify_tool_snapshot_gate_and_rollback(
    ctx: &egui::Context,
    domain: &str,
    tool_id: &str,
    invalid_snapshot_json: &str,
) {
    let mut fw = SimulationFramework::new(domain);
    assert!(fw.select_tool_by_name(tool_id), "Select {tool_id}");
    let valid_baseline = fw.baseline_tool_state.clone().expect("baseline expected");

    if tool_id == "FractalViewer" {
        fw.active_tool
            .as_mut()
            .unwrap()
            .as_any_mut()
            .unwrap()
            .downcast_mut::<FractalViewer>()
            .unwrap()
            .zoom = f64::NAN;
    } else if tool_id == "ActivationFunctionsTool" {
        fw.active_tool
            .as_mut()
            .unwrap()
            .as_any_mut()
            .unwrap()
            .downcast_mut::<ActivationFunctionsTool>()
            .unwrap()
            .x_max = 500.0;
    } else if tool_id == "BetSizeCalculatorTool" {
        fw.active_tool
            .as_mut()
            .unwrap()
            .as_any_mut()
            .unwrap()
            .downcast_mut::<BetSizeCalculatorTool>()
            .unwrap()
            .american_odds = 0.0;
    }
    if !fw.active_tool.as_ref().unwrap().has_finite_parameters() {
        fw.cache_valid_snapshot();
        assert_eq!(
            fw.last_valid_snapshot()
                .and_then(|s| s.tool_state.as_deref()),
            Some(valid_baseline.as_str())
        );
    }

    fw.last_valid_snapshot = Some(StateSnapshot {
        tool_state: Some(invalid_snapshot_json.to_string()),
        plot_datasets: vec![("BadPlot".to_string(), vec![[0.0, f64::NAN]])],
        step_index: 3,
        timestamp: crate::web_io::PlatformInstant::now(),
    });
    fw.is_diverged = true;
    fw.diagnostic_report = Some(DiagnosticReport::new(0.1, 3, "param", "diverged"));

    assert!(fw.rollback_step_back(), "rollback_step_back for {tool_id}");
    assert!(!fw.is_diverged());
    assert!(fw.active_tool.as_ref().unwrap().has_finite_parameters());

    let _ = ctx.run(screen_input(Vec::new()), |c| {
        fw.show(c, &format!("rollback_verify_{tool_id}"));
    });
    assert!(
        !fw.is_diverged(),
        "{tool_id} re-diverged on next frame after rollback_step_back"
    );
}

#[test]
fn test_cache_valid_snapshot_gates_on_has_finite_parameters_and_rollback_validates_snapshot() {
    let cases: [(&str, &str, &str); 7] = [
        (
            "chaos",
            "FractalViewer",
            r#"{"center_re":0.0,"center_im":0.0,"zoom":999999.0,"max_iter":100,"mode":"Mandelbrot","julia_re":0.0,"julia_im":0.0}"#,
        ),
        (
            "ai",
            "ActivationFunctionsTool",
            r#"{"selected_function":"Gelu","x_min":-500.0,"x_max":500.0,"points":200}"#,
        ),
        (
            "ai",
            "LossLandscapeTool",
            r#"{"w_min":10.0,"w_max":-10.0,"b_min":-1.0,"b_max":3.0,"resolution":20}"#,
        ),
        (
            "analysis",
            "ComplexMappingTool",
            r#"{"function":"Exponential","grid_min":-200.0,"grid_max":200.0,"grid_density":10,"inspection_point":[0.0,0.0]}"#,
        ),
        (
            "analysis",
            "RiemannIntegrationTool",
            r#"{"preset":"EToX","min_x":-200.0,"max_x":200.0,"steps":20}"#,
        ),
        (
            "fluid_dynamics",
            "PotentialFlowTool",
            r#"{"elements":[],"grid_size":20,"view_range":500.0,"selected_element_type":"Source","new_element_strength":50000.0,"new_element_x":0.0,"new_element_y":0.0,"new_element_angle":0.0}"#,
        ),
        (
            "financial_math",
            "BetSizeCalculatorTool",
            r#"{"probability_input":55.0,"odds_format":"American","decimal_odds":2.0,"american_odds":50.0,"fractional_numerator":1.0,"fractional_denominator":1.0,"bankroll":1000.0}"#,
        ),
    ];

    let ctx = egui::Context::default();
    for (domain, tool_id, invalid_json) in cases {
        verify_tool_snapshot_gate_and_rollback(&ctx, domain, tool_id, invalid_json);
    }
}
