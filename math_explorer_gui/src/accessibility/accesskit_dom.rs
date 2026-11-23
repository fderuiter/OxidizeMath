#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Incremental browser DOM reconciler projecting [`AccessKitTreeStore`] into `#accesskit_dom_root`.

use super::accesskit_model::{AccessKitTreeStore, CssBounds, DomNodeSpec};
use eframe::egui::{
    self,
    accesskit::{NodeId, TreeUpdate},
};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use wasm_bindgen::JsCast;

/// DOM container identifier holding reconciled `AccessKit` elements beside `#the_canvas_id`.
pub const ACCESSKIT_DOM_ROOT_ID: &str = "accesskit_dom_root";

const ROOT_OVERLAY_STYLE: &str =
    "position: fixed; inset: 0; width: 100vw; height: 100vh; overflow: hidden; pointer-events: none; opacity: 0; z-index: 2;";

#[derive(Default)]
struct DomReconcilerState {
    store: AccessKitTreeStore,
    elements: HashMap<NodeId, (web_sys::Element, DomNodeSpec)>,
    suppress_focus_echo: Option<NodeId>,
}

thread_local! {
    static DOM_RECONCILER: RefCell<DomReconcilerState> = RefCell::new(DomReconcilerState::default());
}

/// Ensures `#accesskit_dom_root` exists in `document` (beside `#the_canvas_id`) and returns it.
pub fn ensure_accesskit_dom_root(document: &web_sys::Document) -> Option<web_sys::Element> {
    if let Some(existing) = document.get_element_by_id(ACCESSKIT_DOM_ROOT_ID) {
        return Some(existing);
    }
    let body = document.body()?;
    let root = document.create_element("div").ok()?;
    root.set_id(ACCESSKIT_DOM_ROOT_ID);
    let _ = root.set_attribute("class", "accesskit-dom-overlay");
    let _ = root.set_attribute("style", ROOT_OVERLAY_STYLE);
    let _ = body.append_child(&root);
    Some(root)
}

/// Clears `#accesskit_dom_root` and resets the cached `AccessKit` tree state.
pub fn clear_accesskit_dom_root() {
    DOM_RECONCILER.with(|cell| {
        let mut state = cell.borrow_mut();
        state.store.clear();
        state.elements.clear();
        state.suppress_focus_echo = None;
    });
    if let Some(doc) = web_sys::window().and_then(|w| w.document()) {
        if let Some(root) = doc.get_element_by_id(ACCESSKIT_DOM_ROOT_ID) {
            root.set_inner_html("");
        }
    }
}

/// Returns `true` and consumes the programmatic focus guard if `id` matches the pending sync focus.
pub(crate) fn consume_suppressed_focus_echo(id: NodeId) -> bool {
    DOM_RECONCILER.with(|cell| {
        let mut state = cell.borrow_mut();
        if state.suppress_focus_echo == Some(id) {
            state.suppress_focus_echo = None;
            true
        } else {
            false
        }
    })
}

/// Looks up the cached [`DomNodeSpec`] for `id` from the active DOM reconciler.
#[must_use]
pub fn lookup_dom_node_spec(id: NodeId) -> Option<DomNodeSpec> {
    DOM_RECONCILER.with(|cell| cell.borrow().elements.get(&id).map(|(_, s)| s.clone()))
}

/// Reconciles an `eframe::WebAccessKitOutput` frame update into `#accesskit_dom_root`.
pub fn reconcile_accesskit_output(output: &eframe::WebAccessKitOutput) {
    if let Some(ref update) = output.update {
        if let Some(doc) = web_sys::window().and_then(|w| w.document()) {
            reconcile_tree_update(&doc, update, output.canvas_rect, output.zoom_factor);
        }
    }
}

/// Reconciles an `accesskit::TreeUpdate` directly into `#accesskit_dom_root` within `document`.
pub fn reconcile_tree_update(
    document: &web_sys::Document,
    update: &TreeUpdate,
    canvas_rect: egui::Rect,
    zoom_factor: f32,
) {
    let Some(dom_root) = ensure_accesskit_dom_root(document) else {
        return;
    };
    let focus_target_el = DOM_RECONCILER.with(|cell| {
        let mut state = cell.borrow_mut();
        state.store.apply_update(update);
        let Some(root_id) = state.store.root_id() else {
            dom_root.set_inner_html("");
            state.elements.clear();
            return None;
        };
        let mut visited = HashSet::with_capacity(state.store.len());
        if let Some(root_el) = sync_subtree(
            document,
            &mut state,
            root_id,
            canvas_rect,
            zoom_factor,
            &mut visited,
        ) {
            sync_single_child(&dom_root, &root_el);
        } else {
            dom_root.set_inner_html("");
        }
        prune_stale_elements(&mut state, &visited);
        resolve_programmatic_dom_focus(document, &dom_root, &mut state)
    });
    if let Some(html_el) = focus_target_el {
        let _ = html_el.focus();
    }
}

fn sync_subtree(
    document: &web_sys::Document,
    state: &mut DomReconcilerState,
    id: NodeId,
    canvas_rect: egui::Rect,
    zoom_factor: f32,
    visited: &mut HashSet<NodeId>,
) -> Option<web_sys::Element> {
    if !visited.insert(id) {
        return None;
    }
    let spec = state.store.build_dom_spec(id, canvas_rect, zoom_factor)?;
    let mut child_els = Vec::with_capacity(spec.children.len());
    for &child_id in &spec.children {
        if let Some(cel) =
            sync_subtree(document, state, child_id, canvas_rect, zoom_factor, visited)
        {
            child_els.push(cel);
        }
    }

    let needs_create = state
        .elements
        .get(&id)
        .is_none_or(|(_, prev)| prev.tag != spec.tag);
    let el = if needs_create {
        let created = document.create_element(spec.tag).ok()?;
        let _ = created.set_attribute("data-ak-id", &id.0.to_string());
        apply_spec_attributes(&created, &spec, None);
        created
    } else {
        let (existing, prev_spec) = state.elements.get(&id).expect("checked Some");
        if prev_spec != &spec {
            apply_spec_attributes(existing, &spec, Some(prev_spec));
        }
        existing.clone()
    };

    reconcile_child_nodes(&el, &child_els);
    state.elements.insert(id, (el.clone(), spec));
    Some(el)
}

fn sync_single_child(parent: &web_sys::Element, child: &web_sys::Element) {
    let current = parent.first_element_child();
    if current.as_ref() != Some(child) || parent.child_element_count() != 1 {
        parent.set_inner_html("");
        let _ = parent.append_child(child);
    }
}

fn reconcile_child_nodes(parent: &web_sys::Element, desired: &[web_sys::Element]) {
    if desired.is_empty() {
        while let Some(first_el) = parent.first_element_child() {
            first_el.remove();
        }
        return;
    }
    let mut cursor = parent.first_element_child();
    for want in desired {
        if cursor.as_ref() == Some(want) {
            cursor = cursor.and_then(|c| c.next_element_sibling());
        } else {
            let ref_node = cursor.as_ref().map(|c| c.as_ref() as &web_sys::Node);
            let _ = parent.insert_before(want, ref_node);
        }
    }
    while let Some(extra) = cursor {
        cursor = extra.next_element_sibling();
        extra.remove();
    }
}

fn prune_stale_elements(state: &mut DomReconcilerState, visited: &HashSet<NodeId>) {
    state.elements.retain(|id, (el, _)| {
        let keep = visited.contains(id);
        if !keep {
            el.remove();
        }
        keep
    });
}

fn resolve_programmatic_dom_focus(
    document: &web_sys::Document,
    dom_root: &web_sys::Element,
    state: &mut DomReconcilerState,
) -> Option<web_sys::HtmlElement> {
    let focus_id = state.store.focus_id()?;
    let active = document.active_element()?;
    if !dom_root.contains(Some(&active)) {
        return None;
    }
    let active_ak_id = active
        .get_attribute("data-ak-id")
        .and_then(|s| s.parse::<u64>().ok())
        .map(NodeId);
    if active_ak_id == Some(focus_id) {
        return None;
    }
    let (target_el, _) = state.elements.get(&focus_id)?;
    let html_el = target_el.clone().dyn_into::<web_sys::HtmlElement>().ok()?;
    state.suppress_focus_echo = Some(focus_id);
    Some(html_el)
}

fn set_or_remove_attr(el: &web_sys::Element, name: &str, val: Option<&str>) {
    match val {
        Some(v) => {
            let _ = el.set_attribute(name, v);
        }
        None => {
            let _ = el.remove_attribute(name);
        }
    }
}

fn format_bounds_style(bounds: Option<CssBounds>) -> String {
    match bounds {
        Some(b) => format!(
            "position:fixed;left:{:.1}px;top:{:.1}px;width:{:.1}px;height:{:.1}px;margin:0;padding:0;border:0;pointer-events:none;box-sizing:border-box;",
            b.left, b.top, b.width, b.height
        ),
        None => "position:fixed;left:0;top:0;width:1px;height:1px;pointer-events:none;".to_owned(),
    }
}

fn apply_spec_attributes(el: &web_sys::Element, spec: &DomNodeSpec, prev: Option<&DomNodeSpec>) {
    let _ = el.set_attribute("data-ak-role", spec.data_ak_role);
    set_or_remove_attr(el, "role", spec.role);
    set_or_remove_attr(
        el,
        "type",
        spec.input_type
            .or_else(|| (spec.tag == "button").then_some("button")),
    );
    set_or_remove_attr(el, "aria-label", spec.aria_label.as_deref());
    set_or_remove_attr(el, "aria-description", spec.aria_description.as_deref());
    set_or_remove_attr(
        el,
        "aria-roledescription",
        spec.aria_roledescription.as_deref(),
    );
    set_or_remove_attr(el, "placeholder", spec.placeholder.as_deref());
    set_or_remove_attr(el, "aria-valuetext", spec.aria_valuetext.as_deref());
    set_or_remove_attr(el, "aria-checked", spec.aria_checked);
    set_or_remove_attr(el, "aria-pressed", spec.aria_pressed);
    set_or_remove_attr(el, "aria-selected", spec.aria_selected);
    set_or_remove_attr(el, "aria-expanded", spec.aria_expanded);
    set_or_remove_attr(el, "aria-orientation", spec.aria_orientation);
    set_or_remove_attr(el, "aria-live", spec.aria_live);
    set_or_remove_attr(el, "aria-modal", spec.aria_modal.then_some("true"));
    set_or_remove_attr(
        el,
        "aria-level",
        spec.aria_level.map(|lvl| lvl.to_string()).as_deref(),
    );
    set_or_remove_attr(
        el,
        "tabindex",
        spec.tab_index.map(|idx| idx.to_string()).as_deref(),
    );
    apply_numeric_and_value_attrs(el, spec);
    apply_state_and_bounds_attrs(el, spec, prev);
}

fn apply_numeric_and_value_attrs(el: &web_sys::Element, spec: &DomNodeSpec) {
    let now_s = spec.numeric_now.map(|v| v.to_string());
    let min_s = spec.numeric_min.map(|v| v.to_string());
    let max_s = spec.numeric_max.map(|v| v.to_string());
    let step_s = spec.numeric_step.map(|v| v.to_string());
    set_or_remove_attr(el, "aria-valuenow", now_s.as_deref());
    set_or_remove_attr(el, "aria-valuemin", min_s.as_deref());
    set_or_remove_attr(el, "aria-valuemax", max_s.as_deref());
    if spec.tag == "input" {
        set_or_remove_attr(el, "min", min_s.as_deref());
        set_or_remove_attr(el, "max", max_s.as_deref());
        let step_attr = step_s
            .as_deref()
            .or_else(|| matches!(spec.input_type, Some("range" | "number")).then_some("any"));
        set_or_remove_attr(el, "step", step_attr);
        let val_str = spec.value.as_deref().unwrap_or("");
        set_or_remove_attr(el, "value", spec.value.as_deref());
        if let Ok(input) = el.clone().dyn_into::<web_sys::HtmlInputElement>() {
            if input.value() != val_str {
                input.set_value(val_str);
            }
        }
    } else if spec.tag == "textarea" {
        let val_str = spec.value.as_deref().unwrap_or("");
        if let Ok(area) = el.clone().dyn_into::<web_sys::HtmlTextAreaElement>() {
            if area.value() != val_str {
                area.set_value(val_str);
            }
        }
    } else {
        set_or_remove_attr(el, "data-ak-value", spec.value.as_deref());
    }
}

fn apply_state_and_bounds_attrs(
    el: &web_sys::Element,
    spec: &DomNodeSpec,
    prev: Option<&DomNodeSpec>,
) {
    if matches!(spec.tag, "button" | "input" | "textarea") && !spec.focusable_when_disabled {
        set_or_remove_attr(el, "disabled", spec.disabled.then_some(""));
        let _ = el.remove_attribute("aria-disabled");
    } else {
        let _ = el.remove_attribute("disabled");
        set_or_remove_attr(el, "aria-disabled", spec.disabled.then_some("true"));
    }
    set_or_remove_attr(el, "readonly", spec.read_only.then_some(""));

    if prev.is_none_or(|p| p.text_content != spec.text_content) && spec.children.is_empty() {
        el.set_text_content(spec.text_content.as_deref());
    }
    if prev.is_none_or(|p| p.bounds != spec.bounds) {
        let _ = el.set_attribute("style", &format_bounds_style(spec.bounds));
    }
}

#[cfg(all(test, target_arch = "wasm32"))]
#[path = "accesskit_dom_wasm_tests.rs"]
mod wasm_tests;
