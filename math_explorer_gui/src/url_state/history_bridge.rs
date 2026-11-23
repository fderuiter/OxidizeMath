//! Browser `Window` `hashchange` and `popstate` navigation bridge per the WHATWG HTML Living Standard.
//!
//! Queues external same-document URL navigations (address-bar hash edits, Back/Forward traversal,
//! and document anchor clicks) while suppressing echoes from self-synced `History::replaceState`
//! updates and waking `egui::Context` immediately.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use eframe::egui;
use std::cell::RefCell;
use std::collections::VecDeque;

#[derive(Default)]
struct HistoryBridgeState {
    egui_ctx: Option<egui::Context>,
    #[cfg(target_arch = "wasm32")]
    listeners_installed: bool,
    last_self_synced_hash: Option<String>,
    pending_navigations: VecDeque<(String, String)>,
}

std::thread_local! {
    static HISTORY_BRIDGE: RefCell<HistoryBridgeState> = RefCell::new(HistoryBridgeState::default());
}

pub(super) fn normalize_hash_fragment(fragment: &str) -> String {
    let trimmed = fragment.trim();
    if trimmed.is_empty() || trimmed == "#" {
        String::new()
    } else if trimmed.starts_with('#') {
        trimmed.to_string()
    } else {
        format!("#{trimmed}")
    }
}

pub(super) fn has_share_query_params(search: &str) -> bool {
    let raw = search.strip_prefix('?').unwrap_or(search).trim();
    raw.split('&').filter(|s| !s.is_empty()).any(|pair| {
        let key = pair.split_once('=').map_or(pair, |(k, _)| k).trim();
        matches!(
            key,
            "ws" | "tab" | "tool" | "state" | "preset" | "challenge" | "step"
        )
    })
}

/// Records `fragment` as the most recently self-synced browser hash so synchronous or delayed
/// browser event echoes matching it are skipped.
pub(crate) fn record_self_synced_fragment(fragment: &str) {
    let normalized = normalize_hash_fragment(fragment);
    HISTORY_BRIDGE.with(|cell| {
        cell.borrow_mut().last_self_synced_hash = Some(normalized);
    });
}

#[cfg(any(test, target_arch = "wasm32"))]
fn enqueue_external_navigation(hash: String, search: String) {
    let normalized_hash = normalize_hash_fragment(&hash);
    let maybe_ctx = HISTORY_BRIDGE.with(|cell| {
        let mut state = cell.borrow_mut();
        let is_echo = state.last_self_synced_hash.as_deref() == Some(normalized_hash.as_str())
            && !has_share_query_params(&search);
        if is_echo {
            return None;
        }
        state.last_self_synced_hash = None;
        let candidate = (normalized_hash, search);
        if state.pending_navigations.back() != Some(&candidate) {
            state.pending_navigations.push_back(candidate);
        }
        state.egui_ctx.clone()
    });
    if let Some(ctx) = maybe_ctx {
        ctx.request_repaint();
    }
}

/// Drains all queued external `(location.hash, location.search)` navigations in FIFO order.
#[must_use]
pub fn drain_pending_browser_navigations() -> Vec<(String, String)> {
    HISTORY_BRIDGE.with(|cell| cell.borrow_mut().pending_navigations.drain(..).collect())
}

/// Enqueues a simulated browser `hashchange` / `popstate` navigation in tests.
#[cfg(test)]
pub fn enqueue_browser_navigation_for_test(hash: &str, search: &str) {
    enqueue_external_navigation(hash.to_string(), search.to_string());
}

/// Resets queued navigations and self-sync echo tracking between tests.
#[cfg(test)]
pub fn reset_history_bridge_for_test() {
    HISTORY_BRIDGE.with(|cell| {
        let mut state = cell.borrow_mut();
        state.last_self_synced_hash = None;
        state.pending_navigations.clear();
    });
}

/// Binds `popstate` and `hashchange` listeners once on `web_sys::Window` and refreshes the
/// active `egui::Context` repaint handle (safe across WebGL `WebRunner` restarts).
pub fn install_browser_history_listeners(ctx: &egui::Context) {
    HISTORY_BRIDGE.with(|cell| {
        cell.borrow_mut().egui_ctx = Some(ctx.clone());
    });
    #[cfg(target_arch = "wasm32")]
    install_wasm_window_listeners();
}

#[cfg(target_arch = "wasm32")]
// theory_verification!
fn install_wasm_window_listeners() {
    use wasm_bindgen::closure::Closure;
    use wasm_bindgen::JsCast;

    let already_installed = HISTORY_BRIDGE.with(|cell| {
        let mut state = cell.borrow_mut();
        if state.listeners_installed {
            true
        } else {
            state.listeners_installed = true;
            false
        }
    });
    if already_installed {
        return;
    }
    let Some(window) = web_sys::window() else {
        return;
    };
    for event_name in ["popstate", "hashchange"] {
        let cb = Closure::wrap(Box::new(move |_event: web_sys::Event| {
            if let Some((hash, search)) = super::read_browser_location_parts() {
                enqueue_external_navigation(hash, search);
            }
        }) as Box<dyn FnMut(web_sys::Event)>);
        let _ = window.add_event_listener_with_callback(event_name, cb.as_ref().unchecked_ref());
        cb.forget();
    }
}

#[cfg(all(test, target_arch = "wasm32"))]
#[path = "history_bridge_wasm_tests.rs"]
mod wasm_tests;
