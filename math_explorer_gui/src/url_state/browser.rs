//! Cross-platform browser `Location` / `History` bindings and clipboard share-link helpers.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::codec::CANONICAL_WEB_BASE_URL;
use super::history_bridge::{
    has_share_query_params, normalize_hash_fragment, record_self_synced_fragment,
};
use eframe::egui;

#[cfg(target_arch = "wasm32")]
const SELF_SYNC_STATE_FLAG: &str = "__oxidize_self_sync";
#[cfg(target_arch = "wasm32")]
const SELF_SYNC_FRAGMENT_KEY: &str = "fragment";

#[cfg(test)]
std::thread_local! {
    static LAST_COPIED_SHARE_URL: std::cell::RefCell<Option<String>> = const { std::cell::RefCell::new(None) };
    static MOCK_WEB_BASE_URL: std::cell::RefCell<Option<String>> = const { std::cell::RefCell::new(None) };
    static LAST_REPLACED_BROWSER_FRAGMENT: std::cell::RefCell<Option<String>> = const { std::cell::RefCell::new(None) };
    static MOCK_SELF_SYNCED_FRAGMENT: std::cell::RefCell<Option<String>> = const { std::cell::RefCell::new(None) };
    static MOCK_BROWSER_LOCATION: std::cell::RefCell<Option<(String, String)>> = const { std::cell::RefCell::new(None) };
}

/// Returns and clears the most recently copied share URL in unit tests.
#[cfg(test)]
#[must_use]
pub fn take_last_copied_share_url() -> Option<String> {
    LAST_COPIED_SHARE_URL.with(|slot| slot.borrow_mut().take())
}

/// Returns the most recent fragment written via [`replace_browser_url_fragment`] in tests.
#[cfg(test)]
#[must_use]
pub fn last_replaced_browser_fragment() -> Option<String> {
    LAST_REPLACED_BROWSER_FRAGMENT.with(|slot| slot.borrow().clone())
}

/// Temporarily overrides [`read_browser_location_parts`] with `(hash, search)` in unit tests.
#[cfg(test)]
pub fn with_mock_browser_location<R>(hash: &str, search: &str, f: impl FnOnce() -> R) -> R {
    MOCK_BROWSER_LOCATION
        .with(|slot| *slot.borrow_mut() = Some((hash.to_string(), search.to_string())));
    let out = f();
    MOCK_BROWSER_LOCATION.with(|slot| *slot.borrow_mut() = None);
    out
}

/// Temporarily simulates `History.state` self-sync metadata for `fragment` in native unit tests.
#[cfg(test)]
pub fn with_mock_self_synced_fragment<R>(fragment: &str, f: impl FnOnce() -> R) -> R {
    let norm = normalize_hash_fragment(fragment);
    MOCK_SELF_SYNCED_FRAGMENT.with(|slot| *slot.borrow_mut() = Some(norm));
    let out = f();
    MOCK_SELF_SYNCED_FRAGMENT.with(|slot| *slot.borrow_mut() = None);
    out
}

/// Temporarily overrides the base URL origin used by [`build_share_url_from_fragment`] in tests.
#[cfg(test)]
pub fn with_mock_web_base_url<R>(base_url: &str, f: impl FnOnce() -> R) -> R {
    MOCK_WEB_BASE_URL.with(|slot| *slot.borrow_mut() = Some(base_url.to_string()));
    let out = f();
    MOCK_WEB_BASE_URL.with(|slot| *slot.borrow_mut() = None);
    out
}

/// Reads `(location.hash, location.search)` from the active browser `Window` on `wasm32`,
/// returning `None` on native desktop builds.
#[must_use]
pub fn read_browser_location_parts() -> Option<(String, String)> {
    #[cfg(test)]
    if let Some(parts) = MOCK_BROWSER_LOCATION.with(|slot| slot.borrow().clone()) {
        return Some(parts);
    }
    #[cfg(target_arch = "wasm32")]
    // theory_verification!
    {
        let window = web_sys::window()?;
        let location = window.location();
        let hash = location.hash().unwrap_or_default();
        let search = location.search().unwrap_or_default();
        Some((hash, search))
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        None
    }
}

#[cfg(any(target_arch = "wasm32", test))]
fn retain_non_share_query(search: &str) -> String {
    let raw = search.strip_prefix('?').unwrap_or(search).trim();
    if raw.is_empty() {
        return String::new();
    }
    let kept: Vec<&str> = raw
        .split('&')
        .filter(|pair| {
            !pair.is_empty() && {
                let key = pair.split_once('=').map_or(*pair, |(k, _)| k);
                !matches!(key, "ws" | "tab" | "tool" | "challenge" | "state")
            }
        })
        .collect();
    if kept.is_empty() {
        String::new()
    } else {
        format!("?{}", kept.join("&"))
    }
}

#[cfg(target_arch = "wasm32")]
// theory_verification!
fn build_self_sync_history_state(hash_part: &str) -> wasm_bindgen::JsValue {
    let state_obj = js_sys::Object::new();
    let _ = js_sys::Reflect::set(
        &state_obj,
        &wasm_bindgen::JsValue::from_str(SELF_SYNC_STATE_FLAG),
        &wasm_bindgen::JsValue::from_str("1"),
    );
    let _ = js_sys::Reflect::set(
        &state_obj,
        &wasm_bindgen::JsValue::from_str(SELF_SYNC_FRAGMENT_KEY),
        &wasm_bindgen::JsValue::from_str(hash_part),
    );
    state_obj.into()
}

/// Updates the browser address bar in place via `History::replaceState` without pushing a history
/// entry, tagging `History.state` with self-sync metadata so reloads preserve `localStorage`.
pub fn replace_browser_url_fragment(fragment: &str) -> Result<(), String> {
    let hash_part = normalize_hash_fragment(fragment);
    record_self_synced_fragment(&hash_part);

    #[cfg(test)]
    LAST_REPLACED_BROWSER_FRAGMENT.with(|slot| *slot.borrow_mut() = Some(hash_part.clone()));

    #[cfg(target_arch = "wasm32")]
    // theory_verification!
    {
        let window = web_sys::window().ok_or_else(|| "Browser window unavailable".to_string())?;
        let history = window
            .history()
            .map_err(|_| "Browser History API unavailable".to_string())?;
        let location = window.location();
        let pathname = location.pathname().unwrap_or_else(|_| "/".to_string());
        let search = location.search().unwrap_or_default();
        let clean_query = retain_non_share_query(&search);
        let next_url = format!("{pathname}{clean_query}{hash_part}");
        let state_val = build_self_sync_history_state(&hash_part);
        history
            .replace_state_with_url(&state_val, "", Some(&next_url))
            .map_err(|_| "History.replaceState failed".to_string())?;
    }
    Ok(())
}

/// Returns `true` when `History.state` proves that `(hash, search)` was written by this tab's own
/// `History::replaceState` synchronization rather than an external user navigation.
#[must_use]
pub fn is_browser_location_self_synced(hash: &str, search: &str) -> bool {
    if has_share_query_params(search) {
        return false;
    }
    let normalized = normalize_hash_fragment(hash);

    #[cfg(test)]
    if let Some(mock_frag) = MOCK_SELF_SYNCED_FRAGMENT.with(|slot| slot.borrow().clone()) {
        return mock_frag == normalized;
    }

    #[cfg(target_arch = "wasm32")]
    // theory_verification!
    {
        let Some(window) = web_sys::window() else {
            return false;
        };
        let Ok(history) = window.history() else {
            return false;
        };
        let Ok(state) = history.state() else {
            return false;
        };
        if state.is_null() || state.is_undefined() || !state.is_object() {
            return false;
        }
        let is_self = js_sys::Reflect::get(
            &state,
            &wasm_bindgen::JsValue::from_str(SELF_SYNC_STATE_FLAG),
        )
        .ok()
        .is_some_and(|v| v.as_bool() == Some(true) || v.as_string().as_deref() == Some("1"));
        if !is_self {
            return false;
        }
        let recorded = js_sys::Reflect::get(
            &state,
            &wasm_bindgen::JsValue::from_str(SELF_SYNC_FRAGMENT_KEY),
        )
        .ok()
        .and_then(|v| v.as_string());
        recorded.as_deref() == Some(normalized.as_str())
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = normalized;
        false
    }
}

/// Resolves a full shareable web URL (`https://...#tab=...`) from a `#tab=...` fragment:
/// - In the browser (`wasm32`), uses `window.location.origin` + `pathname` (preserving non-share query params).
/// - On native desktop builds, uses [`CANONICAL_WEB_BASE_URL`] (`https://fderuiter.github.io/OxidizeMath/`).
#[must_use]
pub fn build_share_url_from_fragment(fragment: &str) -> String {
    let hash_part = if fragment.starts_with('#') || fragment.is_empty() {
        fragment
    } else {
        return format!("{}{}", base_share_origin(), format_args!("#{fragment}"));
    };
    format!("{}{hash_part}", base_share_origin())
}

fn base_share_origin() -> String {
    #[cfg(test)]
    if let Some(mock) = MOCK_WEB_BASE_URL.with(|slot| slot.borrow().clone()) {
        let (before_hash, _) = mock.split_once('#').unwrap_or((&mock, ""));
        let (path_part, query_part) = before_hash
            .split_once('?')
            .map_or((before_hash, ""), |(p, q)| (p, q));
        let clean_query = retain_non_share_query(query_part);
        return format!("{path_part}{clean_query}");
    }

    #[cfg(target_arch = "wasm32")]
    // theory_verification!
    if let Some(window) = web_sys::window() {
        let loc = window.location();
        let origin = loc.origin().unwrap_or_default();
        let pathname = loc.pathname().unwrap_or_else(|_| "/".to_string());
        let search = loc.search().unwrap_or_default();
        if !origin.is_empty() && origin != "null" {
            let clean_query = retain_non_share_query(&search);
            return format!("{origin}{pathname}{clean_query}");
        }
    }

    CANONICAL_WEB_BASE_URL.to_string()
}

/// Copies `share_url` to the system/browser clipboard and invokes `on_result` exactly once.
pub fn copy_share_url_to_clipboard(
    ctx: &egui::Context,
    share_url: String,
    on_result: impl FnOnce(Result<(), String>) + 'static,
) {
    #[cfg(test)]
    {
        LAST_COPIED_SHARE_URL.with(|slot| *slot.borrow_mut() = Some(share_url));
        on_result(Ok(()));
        ctx.request_repaint();
    }

    #[cfg(not(test))]
    {
        let ctx_clone = ctx.clone();
        let cb_cell = std::rc::Rc::new(std::cell::RefCell::new(Some(on_result)));
        let cb_inner = std::rc::Rc::clone(&cb_cell);
        let sync_res =
            egui_plot::export::copy_text_to_clipboard_with_callback(&share_url, move |res| {
                if let Some(cb) = cb_inner.borrow_mut().take() {
                    cb(res);
                }
                ctx_clone.request_repaint();
            });
        if let Err(err) = sync_res {
            if let Some(cb) = cb_cell.borrow_mut().take() {
                cb(Err(err));
            }
            ctx.request_repaint();
        }
    }
}
