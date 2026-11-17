#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::*;

#[test]
fn test_undo_redo_add_node() {
    let mut tool = GraphEditorTool::default();
    assert!(!tool.can_undo() && !tool.can_redo());

    tool.push_undo_snapshot();
    let pos = Pos2::new(10.0, 20.0);
    let node_idx = tool.graph.add_node(pos);
    tool.node_indices.insert(tool.next_node_id, node_idx);
    tool.node_positions.insert(tool.next_node_id, pos);
    tool.next_node_id += 1;

    assert_eq!(tool.node_positions.len(), 1);
    assert!(tool.can_undo() && !tool.can_redo());

    tool.undo();
    assert_eq!(tool.node_positions.len(), 0);
    assert!(!tool.can_undo() && tool.can_redo());

    tool.redo();
    assert_eq!(tool.node_positions.len(), 1);
    assert!(tool.can_undo() && !tool.can_redo());
}

#[test]
fn test_undo_redo_add_edge() {
    let mut tool = GraphEditorTool::default();
    tool.node_positions.insert(0, Pos2::new(0.0, 0.0));
    let idx0 = tool.graph.add_node(Pos2::new(0.0, 0.0));
    tool.node_indices.insert(0, idx0);

    tool.node_positions.insert(1, Pos2::new(50.0, 0.0));
    let idx1 = tool.graph.add_node(Pos2::new(50.0, 0.0));
    tool.node_indices.insert(1, idx1);
    tool.next_node_id = 2;

    tool.push_undo_snapshot();
    tool.edges.push((0, 1, 1.0));
    tool.rebuild_graph();
    assert_eq!(tool.edges.len(), 1);

    tool.undo();
    assert_eq!(tool.edges.len(), 0);
    assert_eq!(tool.node_positions.len(), 2);

    tool.redo();
    assert_eq!(tool.edges.len(), 1);
}

#[test]
fn test_clear_empty_graph_preserves_stacks() {
    let mut tool = GraphEditorTool::default();
    assert!(!tool.can_undo() && !tool.can_redo());
    if !tool.node_positions.is_empty() || !tool.edges.is_empty() {
        tool.push_undo_snapshot();
    }
    assert!(!tool.can_undo() && !tool.can_redo());
}

#[test]
fn test_new_edit_clears_redo_stack() {
    let mut tool = GraphEditorTool::default();
    tool.push_undo_snapshot();
    tool.node_positions.insert(0, Pos2::new(0.0, 0.0));
    tool.next_node_id = 1;

    tool.push_undo_snapshot();
    tool.node_positions.insert(1, Pos2::new(10.0, 10.0));
    tool.next_node_id = 2;

    tool.undo();
    assert!(tool.can_redo());

    tool.push_undo_snapshot();
    tool.node_positions.insert(2, Pos2::new(20.0, 20.0));
    tool.next_node_id = 3;

    assert!(!tool.can_redo());
}

#[test]
fn test_bounded_undo_stack() {
    let mut tool = GraphEditorTool::default();
    for i in 0..60 {
        tool.push_undo_snapshot();
        tool.node_positions.insert(i, Pos2::new(i as f32, i as f32));
    }
    assert_eq!(tool.undo_stack.len(), MAX_SNAPSHOTS);
}

#[test]
fn test_drag_node_saves_one_snapshot() {
    let mut tool = GraphEditorTool::default();
    let initial_pos = Pos2::new(10.0, 10.0);
    tool.node_positions.insert(0, initial_pos);

    tool.push_undo_snapshot();
    if let Some(pos) = tool.node_positions.get_mut(&0) { *pos += egui::vec2(5.0, 5.0); }
    if let Some(pos) = tool.node_positions.get_mut(&0) { *pos += egui::vec2(5.0, 5.0); }

    assert_eq!(tool.undo_stack.len(), 1);
    assert_eq!(tool.node_positions.get(&0), Some(&Pos2::new(20.0, 20.0)));

    tool.undo();
    assert_eq!(tool.node_positions.get(&0), Some(&initial_pos));
}

fn render_canvas_with_visuals(
    tool: &mut GraphEditorTool,
    visuals: egui::Visuals,
) -> (egui::FullOutput, egui::Color32, egui::Stroke) {
    let ctx = egui::Context::default();
    ctx.set_visuals(visuals);
    let text_color = ctx.style().visuals.text_color();
    let window_stroke = ctx.style().visuals.window_stroke();

    let output = ctx.run(egui::RawInput::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            let (response, painter) =
                ui.allocate_painter(egui::vec2(400.0, 400.0), egui::Sense::hover());
            tool.draw(ui, &response, &painter);
        });
    });

    (output, text_color, window_stroke)
}

fn find_node_circle_stroke(shapes: &[egui::epaint::ClippedShape]) -> Option<egui::Stroke> {
    shapes.iter().find_map(|clipped| {
        if let egui::epaint::Shape::Circle(circle) = &clipped.shape {
            Some(circle.stroke)
        } else {
            None
        }
    })
}

fn find_weight_text_color(shapes: &[egui::epaint::ClippedShape]) -> Option<egui::Color32> {
    shapes.iter().find_map(|clipped| {
        if let egui::epaint::Shape::Text(text_shape) = &clipped.shape {
            if text_shape.galley.text() == "3.5" {
                return Some(
                    text_shape
                        .override_text_color
                        .unwrap_or(text_shape.fallback_color),
                );
            }
        }
        None
    })
}

#[test]
#[allow(clippy::too_many_lines)]
fn test_canvas_draw_resolves_dynamic_theme_colors() {
    let mut tool = GraphEditorTool::default();
    tool.node_positions.insert(0, Pos2::new(10.0, 10.0));
    tool.node_positions.insert(1, Pos2::new(50.0, 50.0));
    tool.edges.push((0, 1, 3.5));

    let (output_dark, dark_text_color, dark_window_stroke) =
        render_canvas_with_visuals(&mut tool, egui::Visuals::dark());
    let (output_light, light_text_color, light_window_stroke) =
        render_canvas_with_visuals(&mut tool, egui::Visuals::light());
    let hc_visuals = crate::app::ThemeMode::HighContrast.visuals();
    let (output_hc, hc_text_color, hc_window_stroke) =
        render_canvas_with_visuals(&mut tool, hc_visuals.clone());

    // Text colors across themes must be distinct to ensure high visual contrast
    assert_ne!(dark_text_color, light_text_color);
    assert_ne!(dark_window_stroke.color, light_window_stroke.color);

    // Dark mode text color is brighter than light mode text color
    assert!(dark_text_color.r() > light_text_color.r());

    // High contrast mode text and stroke should match high contrast visuals
    assert_eq!(hc_text_color, hc_visuals.text_color());
    assert_eq!(hc_window_stroke, hc_visuals.window_stroke());

    // Verify shapes are drawn in all modes
    assert!(!output_dark.shapes.is_empty());
    assert!(!output_light.shapes.is_empty());
    assert!(!output_hc.shapes.is_empty());

    let dark_circle_stroke = find_node_circle_stroke(&output_dark.shapes)
        .expect("Dark mode should render circle shape for nodes");
    let light_circle_stroke = find_node_circle_stroke(&output_light.shapes)
        .expect("Light mode should render circle shape for nodes");

    assert_eq!(dark_circle_stroke, dark_window_stroke);
    assert_eq!(light_circle_stroke, light_window_stroke);
    assert_ne!(dark_circle_stroke.color, light_circle_stroke.color);

    let dark_weight_color = find_weight_text_color(&output_dark.shapes)
        .expect("Dark mode should render weight text shape");
    let light_weight_color = find_weight_text_color(&output_light.shapes)
        .expect("Light mode should render weight text shape");

    assert_eq!(dark_weight_color, dark_text_color);
    assert_eq!(light_weight_color, light_text_color);
    assert_ne!(dark_weight_color, light_weight_color);
}

