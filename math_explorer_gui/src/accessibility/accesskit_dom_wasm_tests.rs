#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use eframe::egui::{
    self,
    accesskit::{Action, Node, NodeId, Orientation, Rect, Role, Tree, TreeUpdate},
};
use wasm_bindgen_test::wasm_bindgen_test;

fn sample_tree_update() -> TreeUpdate {
    let root_id = NodeId(1);
    let region_id = NodeId(2);
    let heading_id = NodeId(3);
    let splitter_id = NodeId(4);
    let plot_id = NodeId(5);

    let mut root = Node::new(Role::Window);
    root.set_children(vec![region_id, splitter_id]);

    let mut region = Node::new(Role::Region);
    region.set_label("Left Pane");
    region.set_bounds(Rect {
        x0: 0.0,
        y0: 40.0,
        x1: 500.0,
        y1: 760.0,
    });
    region.set_children(vec![heading_id, plot_id]);

    let mut heading = Node::new(Role::Heading);
    heading.set_label("Settings");

    let mut splitter = Node::new(Role::Splitter);
    splitter.set_label("Split pane divider");
    splitter.set_value("50%");
    splitter.set_numeric_value(50.0);
    splitter.set_min_numeric_value(20.0);
    splitter.set_max_numeric_value(80.0);
    splitter.set_orientation(Orientation::Vertical);
    splitter.add_action(Action::SetValue);

    let mut plot = Node::new(Role::GraphicsDocument);
    plot.set_label("Trajectory Plot");
    plot.set_value("2 series");
    plot.add_action(Action::Focus);

    TreeUpdate {
        nodes: vec![
            (root_id, root),
            (region_id, region),
            (heading_id, heading),
            (splitter_id, splitter),
            (plot_id, plot),
        ],
        tree: Some(Tree::new(root_id)),
        focus: plot_id,
    }
}

#[wasm_bindgen_test]
fn wasm_reconcile_tree_update_populates_and_clears_accesskit_dom_root() {
    let doc = web_sys::window().and_then(|w| w.document()).unwrap();
    clear_accesskit_dom_root();

    let canvas_rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1000.0, 800.0));
    reconcile_tree_update(&doc, &sample_tree_update(), canvas_rect, 1.0);

    let dom_root = doc
        .get_element_by_id(ACCESSKIT_DOM_ROOT_ID)
        .expect("#accesskit_dom_root should be mounted");
    assert!(dom_root.child_element_count() > 0);

    let region = dom_root
        .query_selector("[role='region'][aria-label='Left Pane']")
        .unwrap()
        .expect("Role::Region should reconcile");
    assert!(region
        .query_selector("h2[role='heading']")
        .unwrap()
        .is_some());
    assert!(region
        .query_selector("[role='graphics-document'][aria-label='Trajectory Plot']")
        .unwrap()
        .is_some());

    let sep = dom_root
        .query_selector("[role='separator'][aria-label='Split pane divider']")
        .unwrap()
        .expect("Role::Splitter should reconcile as separator");
    assert_eq!(sep.get_attribute("aria-valuenow").as_deref(), Some("50"));
    assert_eq!(
        sep.get_attribute("aria-orientation").as_deref(),
        Some("vertical")
    );

    clear_accesskit_dom_root();
    assert_eq!(dom_root.child_element_count(), 0);
}
