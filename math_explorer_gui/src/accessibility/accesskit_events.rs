#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Delegated DOM event bridge routing `focus`, `click`, `input`, and `keydown` on
//! `#accesskit_dom_root` into `egui::Event::AccessKitActionRequest` on the active `WebRunner`.

use super::accesskit_dom::{
    clear_accesskit_dom_root, consume_suppressed_focus_echo, ensure_accesskit_dom_root,
    lookup_dom_node_spec, reconcile_accesskit_output,
};
use super::speech::{speak_if_enabled, take_frame_speech_suppression};
use eframe::egui::{
    self,
    accesskit::{Action, ActionData, ActionRequest, NodeId},
};
use std::cell::RefCell;
use wasm_bindgen::{closure::Closure, JsCast, JsValue};

/// Sink trait allowing both `eframe::WebRunner` and unit-test harnesses to receive routed events.
pub trait AccessKitEventSink: 'static {
    /// Enqueues an `AccessKit` `ActionRequest` for the next `egui` pass.
    fn send_action(&self, request: ActionRequest);
    /// Enqueues an auxiliary `egui::Event` (such as keyboard traversal or text paste) for the next pass.
    fn send_egui_event(&self, event: egui::Event);
}

impl AccessKitEventSink for eframe::WebRunner {
    fn send_action(&self, request: ActionRequest) {
        self.dispatch_accesskit_action(request);
    }

    fn send_egui_event(&self, event: egui::Event) {
        self.enqueue_egui_event(event);
    }
}

struct RegisteredDomListener {
    target: web_sys::EventTarget,
    event_name: &'static str,
    use_capture: bool,
    closure: Closure<dyn FnMut(web_sys::Event)>,
}

thread_local! {
    static ACTIVE_DOM_LISTENERS: RefCell<Vec<RegisteredDomListener>> = const { RefCell::new(Vec::new()) };
}

/// Detaches any previously registered delegated listeners on `#accesskit_dom_root`.
pub fn detach_accesskit_dom_listeners() {
    ACTIVE_DOM_LISTENERS.with(|cell| {
        for item in cell.borrow_mut().drain(..) {
            let _ = item.target.remove_event_listener_with_callback_and_bool(
                item.event_name,
                item.closure.as_ref().unchecked_ref(),
                item.use_capture,
            );
        }
    });
}

/// Attaches `TreeUpdate` DOM reconciliation, `WebRunner::destroy()` cleanup, opt-in speech
/// narration, and delegated DOM action listeners to `runner`.
pub fn attach_web_runner_accesskit(runner: &eframe::WebRunner) {
    runner.set_accesskit_output_handler(|output| {
        reconcile_accesskit_output(&output);
        let suppressed = take_frame_speech_suppression();
        if !suppressed && !output.events_description.is_empty() {
            speak_if_enabled(&output.events_description);
        }
    });
    runner.set_destroy_handler(|| {
        detach_accesskit_dom_listeners();
        clear_accesskit_dom_root();
    });
    if let Some(doc) = web_sys::window().and_then(|w| w.document()) {
        if let Some(dom_root) = ensure_accesskit_dom_root(&doc) {
            attach_delegated_dom_listeners(&dom_root, runner.clone());
        }
    }
}

/// Installs delegated `focus`, `click`, `input`, `change`, and `keydown` listeners on `dom_root`.
pub fn attach_delegated_dom_listeners<S: AccessKitEventSink + Clone>(
    dom_root: &web_sys::Element,
    sink: S,
) {
    detach_accesskit_dom_listeners();
    let target: web_sys::EventTarget = dom_root.clone().into();
    register_listener(&target, "focus", true, {
        let s = sink.clone();
        move |ev| handle_focus_event(&ev, &s)
    });
    register_listener(&target, "click", false, {
        let s = sink.clone();
        move |ev| handle_click_event(&ev, &s)
    });
    register_listener(&target, "input", false, {
        let s = sink.clone();
        move |ev| handle_value_event(&ev, &s)
    });
    register_listener(&target, "change", false, {
        let s = sink.clone();
        move |ev| handle_value_event(&ev, &s)
    });
    register_listener(&target, "keydown", false, move |ev| {
        handle_keydown_event(&ev, &sink);
    });
}

fn register_listener(
    target: &web_sys::EventTarget,
    event_name: &'static str,
    use_capture: bool,
    mut handler: impl FnMut(web_sys::Event) + 'static,
) {
    let closure = Closure::wrap(Box::new(move |ev: web_sys::Event| {
        handler(ev);
    }) as Box<dyn FnMut(web_sys::Event)>);
    let _ = target.add_event_listener_with_callback_and_bool(
        event_name,
        closure.as_ref().unchecked_ref(),
        use_capture,
    );
    ACTIVE_DOM_LISTENERS.with(|cell| {
        cell.borrow_mut().push(RegisteredDomListener {
            target: target.clone(),
            event_name,
            use_capture,
            closure,
        });
    });
}

fn find_target_ak_element(event: &web_sys::Event) -> Option<(web_sys::Element, NodeId)> {
    let mut current = event
        .target()
        .and_then(|t| t.dyn_into::<web_sys::Element>().ok())?;
    for _ in 0..16 {
        if let Some(raw) = current.get_attribute("data-ak-id") {
            if let Ok(id) = raw.parse::<u64>() {
                return Some((current, NodeId(id)));
            }
        }
        current = current.parent_element()?;
    }
    None
}

fn is_element_disabled(el: &web_sys::Element) -> bool {
    el.has_attribute("disabled") || el.get_attribute("aria-disabled").as_deref() == Some("true")
}

fn handle_focus_event<S: AccessKitEventSink>(event: &web_sys::Event, sink: &S) {
    let Some((el, target)) = find_target_ak_element(event) else {
        return;
    };
    if el.has_attribute("disabled") || consume_suppressed_focus_echo(target) {
        return;
    }
    sink.send_action(ActionRequest {
        action: Action::Focus,
        target,
        data: None,
    });
}

fn handle_click_event<S: AccessKitEventSink>(event: &web_sys::Event, sink: &S) {
    let Some((el, target)) = find_target_ak_element(event) else {
        return;
    };
    if is_element_disabled(&el) {
        return;
    }
    sink.send_action(ActionRequest {
        action: Action::Focus,
        target,
        data: None,
    });
    sink.send_action(ActionRequest {
        action: Action::Click,
        target,
        data: None,
    });
}

fn read_element_string_value(el: &web_sys::Element) -> Option<String> {
    if let Ok(input) = el.clone().dyn_into::<web_sys::HtmlInputElement>() {
        return Some(input.value());
    }
    if let Ok(area) = el.clone().dyn_into::<web_sys::HtmlTextAreaElement>() {
        return Some(area.value());
    }
    if let Ok(prop) = js_sys::Reflect::get(el.as_ref(), &JsValue::from_str("value")) {
        if let Some(s) = prop.as_string() {
            return Some(s);
        }
        if let Some(n) = prop.as_f64() {
            return Some(n.to_string());
        }
    }
    el.get_attribute("data-value")
        .or_else(|| el.get_attribute("aria-valuenow"))
}

fn handle_value_event<S: AccessKitEventSink>(event: &web_sys::Event, sink: &S) {
    let Some((el, target)) = find_target_ak_element(event) else {
        return;
    };
    if is_element_disabled(&el) {
        return;
    }
    let ak_role = el.get_attribute("data-ak-role").unwrap_or_default();
    let Some(raw_val) = read_element_string_value(&el) else {
        return;
    };
    sink.send_action(ActionRequest {
        action: Action::Focus,
        target,
        data: None,
    });
    if matches!(ak_role.as_str(), "slider" | "spinbutton" | "splitter") {
        if let Ok(num) = raw_val.trim().parse::<f64>() {
            if num.is_finite() {
                sink.send_action(ActionRequest {
                    action: Action::SetValue,
                    target,
                    data: Some(ActionData::NumericValue(num)),
                });
            }
        }
    } else if matches!(
        ak_role.as_str(),
        "textinput" | "searchinput" | "passwordinput" | "multilinetextinput"
    ) {
        dispatch_text_value_replacement(sink, target, raw_val);
    }
}

fn dispatch_text_value_replacement<S: AccessKitEventSink>(
    sink: &S,
    target: NodeId,
    new_text: String,
) {
    sink.send_action(ActionRequest {
        action: Action::SetValue,
        target,
        data: Some(ActionData::Value(new_text.clone().into_boxed_str())),
    });
    let select_all = egui::Modifiers {
        ctrl: true,
        command: true,
        ..egui::Modifiers::NONE
    };
    sink.send_egui_event(egui::Event::Key {
        key: egui::Key::A,
        physical_key: Some(egui::Key::A),
        pressed: true,
        repeat: false,
        modifiers: select_all,
    });
    if new_text.is_empty() {
        sink.send_egui_event(egui::Event::Key {
            key: egui::Key::Backspace,
            physical_key: Some(egui::Key::Backspace),
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        });
    } else {
        sink.send_egui_event(egui::Event::Paste(new_text));
    }
}

fn handle_keydown_event<S: AccessKitEventSink>(event: &web_sys::Event, sink: &S) {
    let Ok(kb) = event.clone().dyn_into::<web_sys::KeyboardEvent>() else {
        return;
    };
    let Some((el, target)) = find_target_ak_element(event) else {
        return;
    };
    if is_element_disabled(&el) {
        return;
    }
    let key_str = kb.key();
    let ak_role = el.get_attribute("data-ak-role").unwrap_or_default();
    if try_handle_range_or_splitter_key(&kb, sink, target, &ak_role, &key_str) {
        return;
    }
    if try_handle_activation_key(&kb, sink, target, &ak_role, &key_str) {
        return;
    }
    forward_general_key_event(&kb, sink, target, &ak_role, &key_str);
}

fn try_handle_range_or_splitter_key<S: AccessKitEventSink>(
    kb: &web_sys::KeyboardEvent,
    sink: &S,
    target: NodeId,
    ak_role: &str,
    key: &str,
) -> bool {
    if !matches!(ak_role, "slider" | "spinbutton" | "splitter") {
        return false;
    }
    match key {
        "ArrowRight" | "ArrowUp" => {
            kb.prevent_default();
            sink.send_action(ActionRequest {
                action: Action::Increment,
                target,
                data: None,
            });
            true
        }
        "ArrowLeft" | "ArrowDown" => {
            kb.prevent_default();
            sink.send_action(ActionRequest {
                action: Action::Decrement,
                target,
                data: None,
            });
            true
        }
        "Home" | "End" | "Enter" => handle_range_boundary_key(kb, sink, target, ak_role, key),
        _ => false,
    }
}

fn handle_range_boundary_key<S: AccessKitEventSink>(
    kb: &web_sys::KeyboardEvent,
    sink: &S,
    target: NodeId,
    ak_role: &str,
    key: &str,
) -> bool {
    let val = if ak_role == "splitter" && matches!(key, "Home" | "Enter") {
        Some(50.0)
    } else if key == "Home" {
        lookup_dom_node_spec(target).and_then(|s| s.numeric_min)
    } else if key == "End" {
        lookup_dom_node_spec(target).and_then(|s| s.numeric_max)
    } else {
        None
    };
    let Some(num) = val else {
        return false;
    };
    kb.prevent_default();
    sink.send_action(ActionRequest {
        action: Action::SetValue,
        target,
        data: Some(ActionData::NumericValue(num)),
    });
    true
}

fn try_handle_activation_key<S: AccessKitEventSink>(
    kb: &web_sys::KeyboardEvent,
    sink: &S,
    target: NodeId,
    ak_role: &str,
    key: &str,
) -> bool {
    if !matches!(key, "Enter" | " ") {
        return false;
    }
    let clickable = matches!(
        ak_role,
        "button"
            | "togglebutton"
            | "checkbox"
            | "switch"
            | "radiobutton"
            | "tab"
            | "combobox"
            | "menuitem"
            | "menuitemcheckbox"
            | "menuitemradio"
            | "link"
    );
    if !clickable {
        return false;
    }
    kb.prevent_default();
    sink.send_action(ActionRequest {
        action: Action::Focus,
        target,
        data: None,
    });
    sink.send_action(ActionRequest {
        action: Action::Click,
        target,
        data: None,
    });
    true
}

fn forward_general_key_event<S: AccessKitEventSink>(
    kb: &web_sys::KeyboardEvent,
    sink: &S,
    target: NodeId,
    ak_role: &str,
    key_str: &str,
) {
    if key_str == "Tab"
        || matches!(
            ak_role,
            "textinput" | "searchinput" | "passwordinput" | "multilinetextinput"
        )
    {
        return;
    }
    let Some(key) = map_dom_key_to_egui(key_str) else {
        return;
    };
    if matches!(
        key,
        egui::Key::ArrowLeft | egui::Key::ArrowRight | egui::Key::ArrowUp | egui::Key::ArrowDown
    ) {
        kb.prevent_default();
    }
    let modifiers = egui::Modifiers {
        alt: kb.alt_key(),
        ctrl: kb.ctrl_key(),
        shift: kb.shift_key(),
        mac_cmd: kb.meta_key(),
        command: kb.ctrl_key() || kb.meta_key(),
    };
    sink.send_action(ActionRequest {
        action: Action::Focus,
        target,
        data: None,
    });
    sink.send_egui_event(egui::Event::Key {
        key,
        physical_key: Some(key),
        pressed: true,
        repeat: kb.repeat(),
        modifiers,
    });
}

fn map_dom_key_to_egui(key: &str) -> Option<egui::Key> {
    match key {
        "ArrowDown" => Some(egui::Key::ArrowDown),
        "ArrowLeft" => Some(egui::Key::ArrowLeft),
        "ArrowRight" => Some(egui::Key::ArrowRight),
        "ArrowUp" => Some(egui::Key::ArrowUp),
        "Escape" => Some(egui::Key::Escape),
        "Enter" => Some(egui::Key::Enter),
        " " => Some(egui::Key::Space),
        "Home" => Some(egui::Key::Home),
        "End" => Some(egui::Key::End),
        "PageUp" => Some(egui::Key::PageUp),
        "PageDown" => Some(egui::Key::PageDown),
        "F6" => Some(egui::Key::F6),
        "+" | "=" => Some(egui::Key::Plus),
        "-" => Some(egui::Key::Minus),
        "0" => Some(egui::Key::Num0),
        "k" | "K" => Some(egui::Key::K),
        "/" => Some(egui::Key::Slash),
        "?" => Some(egui::Key::Questionmark),
        _ => None,
    }
}

#[cfg(all(test, target_arch = "wasm32"))]
#[path = "accesskit_events_wasm_tests.rs"]
mod wasm_tests;
