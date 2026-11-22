#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::framework::{InteractiveTool, SimulationFramework};
use crate::url_state::{encode_share_fragment, parse_share_location, validate_url_preset_json};
use eframe::egui;
use math_explorer::physics::chaos::metrics::logistic_lyapunov;

fn screen_input(events: Vec<egui::Event>) -> egui::RawInput {
    egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1280.0, 800.0),
        )),
        focused: true,
        events,
        ..Default::default()
    }
}

fn key_press(key: egui::Key) -> Vec<egui::Event> {
    vec![egui::Event::Key {
        key,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers: egui::Modifiers::NONE,
    }]
}

fn click_at(ctx: &egui::Context, tool: &mut BifurcationDiagram, pos: egui::Pos2) {
    let _ = ctx.run(
        screen_input(vec![
            egui::Event::PointerMoved(pos),
            egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed: true,
                modifiers: egui::Modifiers::NONE,
            },
        ]),
        |c| tool.show(c),
    );
    let _ = ctx.run(
        screen_input(vec![egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed: false,
            modifiers: egui::Modifiers::NONE,
        }]),
        |c| tool.show(c),
    );
}

fn find_bifurcation_canvas_bounds(out: &egui::FullOutput) -> egui::Rect {
    let ak = out
        .platform_output
        .accesskit_update
        .as_ref()
        .expect("AccessKit update expected");
    for (_, node) in &ak.nodes {
        if node.role() == egui::accesskit::Role::GraphicsDocument
            && node
                .label()
                .is_some_and(|l| l.contains("Bifurcation Diagram Canvas"))
        {
            if let Some(b) = node.bounds() {
                return egui::Rect::from_min_max(
                    egui::pos2(b.x0 as f32, b.y0 as f32),
                    egui::pos2(b.x1 as f32, b.y1 as f32),
                );
            }
        }
    }
    panic!("Bifurcation Diagram Canvas AccessKit node not found");
}

#[test]
fn test_bifurcation_recompute_and_borrowed_slice() {
    let mut tool = BifurcationDiagram::default();
    assert!(!tool.points.is_empty());

    let count_initial = tool.points.len();
    tool.steps = 200;
    tool.recompute();
    assert_ne!(tool.points.len(), count_initial);

    let slice: &[PlotPoint] = &tool.points[..];
    let plot_points: egui_plot::PlotPoints<'_> = slice.into();
    assert!(matches!(plot_points, egui_plot::PlotPoints::Borrowed(_)));
    assert_eq!(plot_points.points().len(), tool.points.len());
}

#[test]
fn test_bifurcation_animated_slider_sweep() {
    use crate::reflective_ui::AnimatedSliderState;
    let mut tool = BifurcationDiagram::default();
    let ctx = egui::Context::default();
    let initial_r_min = tool.r_min;

    let id = egui::Id::new("bifurcation_r_min");
    ctx.data_mut(|d| {
        d.insert_temp(
            id,
            AnimatedSliderState {
                playing: true,
                direction: 1.0,
            },
        );
    });

    let raw_input = egui::RawInput {
        predicted_dt: 0.1,
        ..Default::default()
    };
    let _ = ctx.run(raw_input, |ctx| {
        tool.show(ctx);
    });

    assert!(tool.r_min > initial_r_min);

    let points_vec: Vec<[f64; 2]> = tool.points.iter().map(|p| [p.x, p.y]).collect();
    let datasets = vec![("Bifurcation Diagram (Logistic Map)".to_string(), points_vec)];
    if let Ok(bytes) = egui_plot::export::export_custom_png(
        &datasets,
        &egui_plot::export::ExportOptions::default(),
    ) {
        let _ = std::fs::write("/tmp/bifurcation_diagram_animated.png", bytes);
    }
}

#[test]
fn test_period3_window_r_3_83_matches_logistic_lyapunov_and_cobweb_cycle() {
    let mut tool = BifurcationDiagram::default();
    tool.import_preset_json(STARTER_PERIOD3_WINDOW_JSON)
        .expect("period-3 preset must import");
    assert!((tool.selected_r - 3.83).abs() < 1e-9);
    assert!((tool.x0 - 0.1).abs() < 1e-9);
    assert!(tool.r_pinned);

    let expected_lyap = logistic_lyapunov(3.83, 0.1, LYAPUNOV_ITERATIONS);
    assert!((expected_lyap - (-0.344_763_888)).abs() < 1e-5);
    assert!((tool.selected_lyapunov - expected_lyap).abs() < 1e-12);
    assert!(
        tool.lyapunov_points
            .iter()
            .any(|p| (p.x - 3.83).abs() < 1e-9 && (p.y - expected_lyap).abs() < 1e-12)
    );

    let settled = tool.sample_settled_orbit(12);
    assert_eq!(settled.len(), 12);
    for idx in 0..9 {
        assert!((settled[idx + 3] - settled[idx]).abs() < 1e-4);
    }
    assert!((settled[1] - settled[0]).abs() > 0.1 && (settled[2] - settled[0]).abs() > 0.1);
    assert!(tool.cobweb_cycle.len() >= 13);
}

#[test]
fn test_hover_click_arrow_keys_enter_and_tab_exit_same_frame() {
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let mut tool = BifurcationDiagram::default();

    let _ = ctx.run(screen_input(Vec::new()), |c| tool.show(c));
    let out = ctx.run(screen_input(Vec::new()), |c| tool.show(c));
    let rect = find_bifurcation_canvas_bounds(&out);

    let hover_pos = egui::pos2(rect.left() + rect.width() * 0.82, rect.center().y);
    let _ = ctx.run(
        screen_input(vec![egui::Event::PointerMoved(hover_pos)]),
        |c| tool.show(c),
    );
    assert!(!tool.r_pinned && (tool.selected_r - 3.25).abs() > 0.05);
    let expected = clamp_lyapunov_value(logistic_lyapunov(
        tool.selected_r,
        tool.x0,
        LYAPUNOV_ITERATIONS,
    ));
    assert!((tool.selected_lyapunov - expected).abs() < 1e-12);

    click_at(
        &ctx,
        &mut tool,
        egui::pos2(rect.left() + rect.width() * 0.65, rect.center().y),
    );
    assert!(tool.r_pinned);
    let pinned_r = tool.selected_r;

    let _ = ctx.run(screen_input(key_press(egui::Key::ArrowRight)), |c| {
        tool.show(c);
    });
    assert!(tool.selected_r > pinned_r);

    let _ = ctx.run(screen_input(key_press(egui::Key::Enter)), |c| tool.show(c));
    assert!(!tool.r_pinned);

    let pre_focus = ctx.memory(|m| m.focused());
    assert!(pre_focus.is_some());
    let _ = ctx.run(screen_input(key_press(egui::Key::Tab)), |c| tool.show(c));
    assert_ne!(ctx.memory(|m| m.focused()), pre_focus);
}

#[test]
fn test_custom_ast_map_valid_and_invalid_syntax_no_panic() {
    let ctx = egui::Context::default();
    let mut fw = SimulationFramework::new("chaos");
    assert!(fw.select_tool_by_name("BifurcationDiagram"));

    let tool = fw
        .active_tool
        .as_mut()
        .unwrap()
        .as_any_mut()
        .unwrap()
        .downcast_mut::<BifurcationDiagram>()
        .unwrap();
    tool.use_custom_map = true;
    tool.r_min = 0.2;
    tool.r_max = 1.0;
    tool.selected_r = 0.85;
    tool.math_input.set_expression("r * sin(pi * x)");
    tool.recompute();
    assert!(tool.has_finite_parameters() && tool.custom_eval_error.is_none());
    assert!(!tool.points.is_empty());

    tool.math_input.set_expression("r * sin(");
    tool.recompute();
    assert!(tool.math_input.error_message().is_some());

    let _ = ctx.run(screen_input(Vec::new()), |c| fw.show(c, "chaos"));
    assert!(!fw.is_diverged());
}

#[test]
fn test_custom_ast_map_divergence_triggers_state_guard_rollback() {
    let ctx = egui::Context::default();
    let mut fw = SimulationFramework::new("chaos");
    assert!(fw.select_tool_by_name("BifurcationDiagram"));
    let _ = ctx.run(screen_input(Vec::new()), |c| fw.show(c, "chaos"));

    {
        let tool = fw
            .active_tool
            .as_mut()
            .unwrap()
            .as_any_mut()
            .unwrap()
            .downcast_mut::<BifurcationDiagram>()
            .unwrap();
        tool.use_custom_map = true;
        tool.math_input.set_expression("x^2 + 100");
        tool.recompute();
        assert!(tool.custom_eval_error.is_some());
        assert!(!tool.has_finite_parameters());
    }
    let _ = ctx.run(screen_input(Vec::new()), |c| fw.show(c, "chaos"));
    assert!(fw.is_diverged());
    assert!(fw.rollback_step_back());
    assert!(!fw.is_diverged());
    assert!(fw.active_tool.as_ref().unwrap().has_finite_parameters());
}

#[test]
fn test_legacy_and_period3_preset_url_roundtrip_within_bounds() {
    for raw_preset in [STARTER_PRESET_JSON, STARTER_PERIOD3_WINDOW_JSON] {
        validate_url_preset_json(raw_preset).unwrap();
        let mut tool = BifurcationDiagram::default();
        tool.import_preset_json(raw_preset).unwrap();
        let envelope = tool.export_preset_json().unwrap();
        validate_url_preset_json(&envelope).unwrap();

        let frag = encode_share_fragment("chaos", Some("BifurcationDiagram"), Some(&envelope))
            .expect("encode_share_fragment");
        let parsed = parse_share_location(&frag, "")
            .unwrap()
            .expect("parsed share params");
        let mut restored = BifurcationDiagram::default();
        restored
            .import_preset_json(parsed.state_json.as_deref().unwrap())
            .unwrap();
        assert_eq!(restored.save_state(), tool.save_state());
    }
}
