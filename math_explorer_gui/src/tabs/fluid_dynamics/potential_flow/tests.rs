#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;

fn headless_input_with_events(events: Vec<egui::Event>) -> egui::RawInput {
    egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1920.0, 1080.0),
        )),
        events,
        ..Default::default()
    }
}

#[test]
fn potential_flow_clamps_high_grid_state_and_caches_static_arrow_field() {
    let mut tool = PotentialFlowTool::default();
    let high_grid = r#"{"elements":[{"element_type":"Uniform","strength":10.0,"x":0.0,"y":0.0,"angle":0.0}],"grid_size":1000,"view_range":10.0}"#;
    tool.try_load_state(high_grid)
        .expect("grid_size 1000 should load and clamp to interactive cap");
    assert_eq!(tool.grid_size, crate::presets::MAX_INTERACTIVE_GRID_SIZE);
    assert!(tool.has_finite_parameters());

    let over_max = r#"{"elements":[],"grid_size":1001,"view_range":10.0}"#;
    assert!(tool.try_load_state(over_max).is_err());

    let mut live_tool = PotentialFlowTool::default();
    let ctx = egui::Context::default();
    egui_plot::clear_last_plot_datasets();
    let _ = ctx.run(headless_input_with_events(Vec::new()), |c| live_tool.show(c));
    assert_eq!(egui_plot::get_last_plot_item_counts(), vec![1]);

    egui_plot::reset_materialized_plot_datasets_call_count();
    let _ = ctx.run(headless_input_with_events(Vec::new()), |c| live_tool.show(c));
    assert_eq!(live_tool.cache.eval_count, 1);
    assert_eq!(egui_plot::materialized_plot_datasets_call_count(), 0);

    live_tool.grid_size = 24;
    let _ = ctx.run(headless_input_with_events(Vec::new()), |c| live_tool.show(c));
    assert_eq!(live_tool.cache.eval_count, 2);
}

#[test]
fn potential_flow_removes_individual_element_without_clearing_remaining_superposition() {
    let mut tool = PotentialFlowTool::default();
    tool.clear();
    tool.selected_element_type = ElementType::Source;
    tool.new_element_strength = 15.0;
    tool.new_element_x = -2.0;
    tool.new_element_y = 1.0;
    tool.add_element();

    tool.selected_element_type = ElementType::Vortex;
    tool.new_element_strength = 8.5;
    tool.new_element_x = 0.5;
    tool.new_element_y = -1.5;
    tool.add_element();

    tool.selected_element_type = ElementType::Doublet;
    tool.new_element_strength = 20.0;
    tool.new_element_x = 3.0;
    tool.new_element_y = 2.0;
    tool.add_element();

    assert_eq!(tool.elements.len(), 3);
    let removed = tool.remove_element(1).expect("remove middle Vortex");
    assert_eq!(removed.element_type, ElementType::Vortex);
    assert_eq!(tool.elements.len(), 2);
    assert_eq!(tool.elements[0].element_type, ElementType::Source);
    assert_eq!(tool.elements[0].strength, 15.0);
    assert_eq!(tool.elements[1].element_type, ElementType::Doublet);
    assert_eq!(tool.elements[1].strength, 20.0);
    assert!(tool.remove_element(99).is_none());
}

fn collect_button_centers(
    shape: &egui::epaint::Shape,
    needle: &str,
    out: &mut Vec<egui::Pos2>,
) {
    match shape {
        egui::epaint::Shape::Text(t) if t.galley.text() == needle => {
            out.push(t.pos + t.galley.size() * 0.5);
        }
        egui::epaint::Shape::Vec(inner) => {
            for s in inner {
                collect_button_centers(s, needle, out);
            }
        }
        _ => {}
    }
}

fn build_three_element_tool() -> PotentialFlowTool {
    let mut tool = PotentialFlowTool {
        selected_element_type: ElementType::Vortex,
        new_element_strength: 12.0,
        new_element_x: 1.25,
        new_element_y: -0.75,
        ..Default::default()
    };
    tool.add_element();
    tool
}

#[test]
fn potential_flow_ui_lists_active_elements_and_exposes_accessible_delete_buttons() {
    crate::accessibility::clear_test_announcements();
    let mut tool = build_three_element_tool();
    assert_eq!(tool.elements.len(), 3);

    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let _ = ctx.run(headless_input_with_events(Vec::new()), |c| tool.show(c));
    let out = ctx.run(headless_input_with_events(Vec::new()), |c| tool.show(c));
    let ak = out.platform_output.accesskit_update.expect("AccessKit tree");

    let remove_buttons: Vec<(egui::accesskit::NodeId, String)> = ak
        .nodes
        .iter()
        .filter_map(|(id, node)| {
            let label = node.label()?;
            (node.role() == egui::accesskit::Role::Button && label.starts_with("Remove element "))
                .then(|| (*id, label.to_string()))
        })
        .collect();
    assert_eq!(remove_buttons.len(), 3);
    assert!(remove_buttons
        .iter()
        .any(|(_, l)| l.contains("Vortex (Γ = 12.00, x = 1.25, y = -0.75)")));

    let mut centers = Vec::new();
    for cs in &out.shapes {
        collect_button_centers(&cs.shape, "✕ Remove", &mut centers);
    }
    centers.sort_by(|a, b| a.y.total_cmp(&b.y));
    assert_eq!(centers.len(), 3);
    let target_center = centers[1];
    let _ = ctx.run(
        headless_input_with_events(vec![
            egui::Event::PointerMoved(target_center),
            egui::Event::PointerButton {
                pos: target_center,
                button: egui::PointerButton::Primary,
                pressed: true,
                modifiers: egui::Modifiers::NONE,
            },
        ]),
        |c| tool.show(c),
    );
    let _ = ctx.run(
        headless_input_with_events(vec![egui::Event::PointerButton {
            pos: target_center,
            button: egui::PointerButton::Primary,
            pressed: false,
            modifiers: egui::Modifiers::NONE,
        }]),
        |c| tool.show(c),
    );
    assert_eq!(tool.elements.len(), 2);
    assert_eq!(tool.elements[0].element_type, ElementType::Uniform);
    assert_eq!(tool.elements[1].element_type, ElementType::Vortex);
}

#[test]
fn potential_flow_ui_deletes_element_via_keyboard_focus_and_enter() {
    let mut tool = build_three_element_tool();
    let _ = tool.remove_element(1);
    assert_eq!(tool.elements.len(), 2);

    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let _ = ctx.run(headless_input_with_events(Vec::new()), |c| tool.show(c));
    let out2 = ctx.run(headless_input_with_events(Vec::new()), |c| tool.show(c));
    let ak2 = out2
        .platform_output
        .accesskit_update
        .expect("AccessKit tree");
    let vortex_node_id = ak2
        .nodes
        .iter()
        .find_map(|(id, node)| {
            (node.role() == egui::accesskit::Role::Button
                && node.label().is_some_and(|l| l.starts_with("Remove element 2:")))
            .then_some(*id)
        })
        .expect("remaining Vortex remove button");
    let _ = ctx.run(
        headless_input_with_events(vec![egui::Event::AccessKitActionRequest(
            egui::accesskit::ActionRequest {
                action: egui::accesskit::Action::Focus,
                target: vortex_node_id,
                data: None,
            },
        )]),
        |c| tool.show(c),
    );
    let _ = ctx.run(
        headless_input_with_events(vec![egui::Event::Key {
            key: egui::Key::Enter,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        }]),
        |c| tool.show(c),
    );
    assert_eq!(tool.elements.len(), 1);
    assert_eq!(tool.elements[0].element_type, ElementType::Uniform);
}
