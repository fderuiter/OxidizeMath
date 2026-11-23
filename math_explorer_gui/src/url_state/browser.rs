//! Cross-platform browser `Location` / `History` bindings and clipboard share-link helpers.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::codec::CANONICAL_WEB_BASE_URL;
use eframe::egui;

#[cfg(test)]
std::thread_local! {
    static LAST_COPIED_SHARE_URL: std::cell::RefCell<Option<String>> = const { std::cell::RefCell::new(None) };
    static MOCK_WEB_BASE_URL: std::cell::RefCell<Option<String>> = const { std::cell::RefCell::new(None) };
}

/// Returns and clears the most recently copied share URL in unit tests.
#[cfg(test)]
#[must_use]
pub fn take_last_copied_share_url() -> Option<String> {
    LAST_COPIED_SHARE_URL.with(|slot| slot.borrow_mut().take())
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
    #[cfg(target_arch = "wasm32")]
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
                !matches!(
                    key,
                    "tab" | "tool" | "state" | "preset" | "challenge" | "step"
                )
            }
        })
        .collect();
    if kept.is_empty() {
        String::new()
    } else {
        format!("?{}", kept.join("&"))
    }
}

/// Updates the browser address bar in place via `History::replaceState` without pushing a history entry.
#[cfg(target_arch = "wasm32")]
pub fn replace_browser_url_fragment(fragment: &str) -> Result<(), String> {
    let window = web_sys::window().ok_or_else(|| "Browser window unavailable".to_string())?;
    let history = window
        .history()
        .map_err(|_| "Browser History API unavailable".to_string())?;
    let location = window.location();
    let pathname = location.pathname().unwrap_or_else(|_| "/".to_string());
    let search = location.search().unwrap_or_default();
    let clean_query = retain_non_share_query(&search);
    let hash_part = if fragment.starts_with('#') || fragment.is_empty() {
        fragment.to_string()
    } else {
        format!("#{fragment}")
    };
    let next_url = format!("{pathname}{clean_query}{hash_part}");
    history
        .replace_state_with_url(&wasm_bindgen::JsValue::NULL, "", Some(&next_url))
        .map_err(|_| "History.replaceState failed".to_string())
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

#[cfg(all(test, target_arch = "wasm32"))]
#[path = "browser_wasm_tests.rs"]
mod wasm_tests;
