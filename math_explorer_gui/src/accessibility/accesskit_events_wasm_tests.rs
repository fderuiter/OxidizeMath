#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::accessibility::accesskit_dom::{
    clear_accesskit_dom_root, reconcile_tree_update, ACCESSKIT_DOM_ROOT_ID,
};
use eframe::egui::{
    self,
    accesskit::{Action, ActionData, ActionRequest, Node, NodeId, Role, Tree, TreeUpdate},
};
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::JsCast;
use wasm_bindgen_test::wasm_bindgen_test;

#[derive(Clone, Default)]
struct RecordingSink {
    actions: Rc<RefCell<Vec<ActionRequest>>>,
    events: Rc<RefCell<Vec<egui::Event>>>,
}

impl AccessKitEventSink for RecordingSink {
    fn send_action(&self, request: ActionRequest) {
        self.actions.borrow_mut().push(request);
    }

    fn send_egui_event(&self, event: egui::Event) {
        self.events.borrow_mut().push(event);
    }
}

fn interactive_tree_update() -> TreeUpdate {
    let root_id = NodeId(10);
    let btn_id = NodeId(11);
    let slider_id = NodeId(12);
    let sash_id = NodeId(13);

    let mut root = Node::new(Role::Window);
    root.set_children(vec![btn_id, slider_id, sash_id]);

    let mut btn = Node::new(Role::Button);
    btn.set_label("Run Step");
    btn.add_action(Action::Focus);
    btn.add_action(Action::Click);

    let mut slider = Node::new(Role::Slider);
    slider.set_label("Beta");
    slider.set_numeric_value(0.25);
    slider.set_min_numeric_value(0.0);
    slider.set_max_numeric_value(1.0);
    slider.add_action(Action::SetValue);

    let mut sash = Node::new(Role::Splitter);
    sash.set_label("Split pane divider");
    sash.set_numeric_value(50.0);
    sash.set_min_numeric_value(20.0);
    sash.set_max_numeric_value(80.0);
    sash.add_action(Action::Increment);
    sash.add_action(Action::Decrement);
    sash.add_action(Action::SetValue);

    TreeUpdate {
        nodes: vec![
            (root_id, root),
            (btn_id, btn),
            (slider_id, slider),
            (sash_id, sash),
        ],
        tree: Some(Tree::new(root_id)),
        focus: root_id,
    }
}

#[wasm_bindgen_test]
fn wasm_dom_focus_click_input_and_keydown_route_to_accesskit_action_requests() {
    let doc = web_sys::window().and_then(|w| w.document()).unwrap();
    clear_accesskit_dom_root();

    let rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(800.0, 600.0));
    reconcile_tree_update(&doc, &interactive_tree_update(), rect, 1.0);

    let dom_root = doc.get_element_by_id(ACCESSKIT_DOM_ROOT_ID).unwrap();
    let sink = RecordingSink::default();
    attach_delegated_dom_listeners(&dom_root, sink.clone());

    let btn = dom_root.query_selector("button").unwrap().unwrap();
    let bubbled = web_sys::EventInit::new();
    bubbled.set_bubbles(true);
    let _ = btn.dispatch_event(&web_sys::Event::new("focus").unwrap());
    let _ =
        btn.dispatch_event(&web_sys::Event::new_with_event_init_dict("click", &bubbled).unwrap());

    let slider = dom_root
        .query_selector("input[role='slider']")
        .unwrap()
        .unwrap()
        .dyn_into::<web_sys::HtmlInputElement>()
        .unwrap();
    slider.set_value("0.75");
    let _ = slider
        .dispatch_event(&web_sys::Event::new_with_event_init_dict("input", &bubbled).unwrap());

    let sash = dom_root
        .query_selector("[role='separator']")
        .unwrap()
        .unwrap();
    let kb_init = web_sys::KeyboardEventInit::new();
    kb_init.set_bubbles(true);
    kb_init.set_cancelable(true);
    kb_init.set_key("ArrowRight");
    let kd =
        web_sys::KeyboardEvent::new_with_keyboard_event_init_dict("keydown", &kb_init).unwrap();
    let _ = sash.dispatch_event(&kd);
    assert!(kd.default_prevented());

    let _ = sash.set_attribute("aria-valuenow", "65");
    let _ =
        sash.dispatch_event(&web_sys::Event::new_with_event_init_dict("input", &bubbled).unwrap());

    let actions = sink.actions.borrow();
    assert!(actions
        .iter()
        .any(|a| a.target == NodeId(11) && a.action == Action::Focus));
    assert!(actions
        .iter()
        .any(|a| a.target == NodeId(11) && a.action == Action::Click));
    assert!(actions.iter().any(|a| {
        a.target == NodeId(12)
            && a.action == Action::SetValue
            && a.data == Some(ActionData::NumericValue(0.75))
    }));
    assert!(actions
        .iter()
        .any(|a| a.target == NodeId(13) && a.action == Action::Increment));

    detach_accesskit_dom_listeners();
    clear_accesskit_dom_root();
}
