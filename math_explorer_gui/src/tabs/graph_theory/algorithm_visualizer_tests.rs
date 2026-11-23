#![cfg_attr(any(), verified(opt_out = "tests"))]
use super::*;
use eframe::egui;

#[test]
fn test_algorithm_visualizer_defaults() {
    let tool = AlgorithmVisualizerTool::default();
    assert!(!tool.is_playing);
    assert_eq!(tool.playback_speed, 2.0);
    assert_eq!(tool.step_timer, 0.0);
    assert_eq!(tool.animation_step, 0);
}

#[test]
fn test_algorithm_visualizer_timer_progression() {
    let ctx = egui::Context::default();
    let mut tool = AlgorithmVisualizerTool {
        is_playing: true,
        playback_speed: 2.0,
        ..Default::default()
    };

    let raw_input = egui::RawInput {
        predicted_dt: 0.6,
        ..Default::default()
    };
    let _ = ctx.run(raw_input, |ctx| {
        tool.show(ctx);
    });

    assert_eq!(tool.animation_step, 1);
    assert!(tool.is_playing);
}

#[test]
fn test_algorithm_visualizer_stop_at_end() {
    let ctx = egui::Context::default();
    let mut tool = AlgorithmVisualizerTool::default();
    tool.animation_step = tool.visit_order.len();
    tool.is_playing = true;

    let raw_input = egui::RawInput {
        predicted_dt: 1.0,
        ..Default::default()
    };
    let _ = ctx.run(raw_input, |ctx| {
        tool.show(ctx);
    });

    assert!(!tool.is_playing);
    assert_eq!(tool.animation_step, tool.visit_order.len());
    assert_eq!(tool.step_timer, 0.0);
}

#[test]
fn test_run_algorithm_resets_playback_state() {
    let mut tool = AlgorithmVisualizerTool {
        is_playing: true,
        step_timer: 0.4,
        animation_step: 3,
        ..Default::default()
    };

    tool.run_algorithm();

    assert!(!tool.is_playing);
    assert_eq!(tool.step_timer, 0.0);
    assert_eq!(tool.animation_step, 0);
}

fn algo_screen_raw() -> egui::RawInput {
    egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1280.0, 800.0),
        )),
        ..Default::default()
    }
}

fn algo_key_raw(key: egui::Key) -> egui::RawInput {
    let mut raw = algo_screen_raw();
    raw.events.push(egui::Event::Key {
        key,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers: egui::Modifiers::NONE,
    });
    raw
}

fn focus_algo_canvas(ctx: &egui::Context, tool: &mut AlgorithmVisualizerTool) -> egui::Id {
    let out0 = ctx.run(algo_screen_raw(), |ctx| tool.show(ctx));
    let update = out0
        .platform_output
        .accesskit_update
        .expect("AccessKit update expected");
    let canvas_node_id = update
        .nodes
        .iter()
        .find_map(|(id, node)| {
            (node.role() == egui::accesskit::Role::GraphicsDocument
                && node
                    .label()
                    .is_some_and(|l| l.contains("Graph Theory Algorithm Visualization")))
            .then_some(*id)
        })
        .expect("Algorithm visualizer canvas must expose AccessKit node");
    let mut focus_raw = algo_screen_raw();
    focus_raw.events.push(egui::Event::AccessKitActionRequest(
        egui::accesskit::ActionRequest {
            action: egui::accesskit::Action::Focus,
            target: canvas_node_id,
            data: None,
        },
    ));
    let _ = ctx.run(focus_raw, |ctx| tool.show(ctx));
    let _ = ctx.run(algo_screen_raw(), |ctx| tool.show(ctx));
    ctx.memory(|m| m.focused()).expect("Canvas focused")
}

#[test]
fn test_algorithm_visualizer_arrow_keys_and_enter_sets_start_node() {
    crate::accessibility::clear_test_announcements();
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let mut tool = AlgorithmVisualizerTool::default();
    assert_eq!(tool.start_node, Some(0));

    let canvas_id = focus_algo_canvas(&ctx, &mut tool);

    // Navigate 0 -> 1 (Right) -> 5 (Down on 4x4 grid).
    let _ = ctx.run(algo_key_raw(egui::Key::ArrowRight), |ctx| tool.show(ctx));
    assert_eq!(tool.focused_node, 1);
    let _ = ctx.run(algo_key_raw(egui::Key::ArrowDown), |ctx| tool.show(ctx));
    assert_eq!(tool.focused_node, 5);
    assert_eq!(
        tool.start_node,
        Some(0),
        "Arrow keys move cursor without mutating start_node"
    );
    assert_eq!(ctx.memory(|m| m.focused()), Some(canvas_id));

    crate::accessibility::clear_test_announcements();
    let _ = ctx.run(algo_key_raw(egui::Key::Enter), |ctx| tool.show(ctx));
    assert_eq!(tool.start_node, Some(5));
    assert_eq!(tool.visit_order.first().copied(), Some(5));
    assert!(
        crate::accessibility::all_announcements()
            .iter()
            .any(|(msg, _)| msg.contains("Node 5")),
        "Enter activation must announce the new start node"
    );

    // Tab must move focus cleanly out of the canvas without trapping.
    let _ = ctx.run(algo_key_raw(egui::Key::Tab), |ctx| tool.show(ctx));
    assert_ne!(ctx.memory(|m| m.focused()), Some(canvas_id));
}

#[test]
fn test_algorithm_visualizer_exposes_start_node_selector_in_side_panel() {
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let mut tool = AlgorithmVisualizerTool::default();
    let out = ctx.run(
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(1280.0, 800.0),
            )),
            ..Default::default()
        },
        |ctx| tool.show(ctx),
    );
    let update = out.platform_output.accesskit_update.unwrap();
    let has_combo = update
        .nodes
        .iter()
        .any(|(_, node)| node.role() == egui::accesskit::Role::ComboBox);
    assert!(
        has_combo,
        "Side panel must render Start Node ComboBox selector"
    );
}
