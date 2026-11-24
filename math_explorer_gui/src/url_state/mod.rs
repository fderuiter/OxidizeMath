//! Self-contained browser URL fragment (`#tab=<slug>&tool=<id>&state=<base64url>`)
//! encoding, startup hydration parsing, 300 ms debounced `History::replaceState` sync,
//! and cross-platform clipboard share-link helpers.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

mod base64url;
mod browser;
mod codec;
mod history_bridge;
mod json_validator;
pub mod pipeline;
mod sync;
mod workspace_codec;

#[cfg(all(test, target_arch = "wasm32"))]
mod browser_wasm_tests;
#[cfg(all(
    test,
    feature = "ai",
    feature = "applied",
    feature = "biology",
    feature = "climate",
    feature = "epidemiology",
    feature = "physics",
    feature = "pure_math"
))]
mod contract_tests;
#[cfg(test)]
mod param_only_tests;
#[cfg(test)]
mod tests;
#[cfg(all(
    test,
    feature = "ai",
    feature = "applied",
    feature = "biology",
    feature = "climate",
    feature = "epidemiology",
    feature = "physics",
    feature = "pure_math"
))]
mod workspace_contract_tests;
#[cfg(all(
    test,
    feature = "ai",
    feature = "applied",
    feature = "biology",
    feature = "climate",
    feature = "epidemiology",
    feature = "physics",
    feature = "pure_math"
))]
mod workspace_large_tools_tests;
#[cfg(test)]
mod workspace_tests;

pub use base64url::{decode as decode_base64url, encode as encode_base64url};
pub use browser::{
    build_share_url_from_fragment, copy_share_url_to_clipboard, is_browser_location_self_synced,
    read_browser_location_parts, replace_browser_url_fragment,
};
#[cfg(test)]
pub use browser::{
    last_replaced_browser_fragment, take_last_copied_share_url, with_mock_browser_location,
    with_mock_self_synced_fragment, with_mock_web_base_url,
};
#[cfg(test)]
pub(crate) use codec::encode_share_fragment_with_hud_prevalidated;
pub use codec::{
    encode_share_fragment, encode_share_fragment_with_challenge, encode_share_fragment_with_hud,
    parse_share_location, parse_workspace_or_share_location, validate_url_preset_json,
    CANONICAL_WEB_BASE_URL, MAX_DECODED_URL_STATE_BYTES, MAX_URL_FRAGMENT_BYTES, URL_JSON_BOUNDS,
};
pub use history_bridge::{drain_pending_browser_navigations, install_browser_history_listeners};
#[cfg(test)]
pub use history_bridge::{enqueue_browser_navigation_for_test, reset_history_bridge_for_test};
pub(crate) use json_validator::validate_json_streaming_with_bounds;
pub use sync::{PaneUrlSignature, UrlStateSignature, UrlSyncController, URL_SYNC_DEBOUNCE_MS};
pub use workspace_codec::{
    decode_workspace_share_param, encode_single_pane_smart_fragment,
    encode_single_pane_smart_fragment_from_params,
    encode_single_pane_smart_fragment_with_challenge, encode_workspace_share_fragment,
    validate_workspace_url_preset_json, ParsedShareLocation, WorkspaceShareParams,
    MAX_DECODED_WORKSPACE_URL_BYTES, WORKSPACE_SHARE_SCHEMA_VERSION,
};

/// Canonical sample count used when persisting compact 1D custom potential arrays in URLs.
pub const CUSTOM_POTENTIAL_POINTS: usize = 200;

/// Quantizes a finite scalar to `0.01` precision for compact URL JSON serialization.
#[inline]
#[must_use]
pub fn quantize_url_centi(val: f64, min: f64, max: f64) -> f64 {
    if !val.is_finite() {
        return 0.0_f64.clamp(min, max);
    }
    let clamped = val.clamp(min, max);
    let q = (clamped * 100.0).round() / 100.0;
    if q == 0.0 {
        0.0
    } else {
        q
    }
}

/// Quantizes and clamps a bounded 1D numeric series for compact URL JSON serialization.
pub fn quantize_bounded_url_series(
    series: &[f64],
    expected_len: usize,
    min_val: f64,
    max_val: f64,
) -> Result<Vec<f64>, String> {
    if series.len() != expected_len {
        return Err(format!(
            "Custom URL series length ({}) must equal expected length ({expected_len})",
            series.len()
        ));
    }
    let mut out = Vec::with_capacity(expected_len);
    for &v in series {
        if !v.is_finite() {
            return Err("Custom URL series contains non-finite value".to_string());
        }
        out.push(quantize_url_centi(v, min_val, max_val));
    }
    Ok(out)
}

/// Parsed simulation share-link parameters extracted from `location.hash` or `location.search`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShareUrlParams {
    /// Target domain tab slug or normalized tab identifier.
    pub tab: String,
    /// Optional target `InteractiveTool` identifier or display name.
    pub tool: Option<String>,
    /// Optional decoded and structurally validated `PresetWrapper` JSON string.
    pub state_json: Option<String>,
    /// Optional registered starter snapshot or guided story identifier (`&preset=<id>`).
    pub preset: Option<String>,
    /// Optional registered inverse or AST goal challenge identifier (`&challenge=<id>`).
    pub challenge: Option<String>,
    /// Optional 0-based guided story step index (`&step=<idx>`).
    pub step: Option<usize>,
    /// Optional validated multi-pane dataflow pipeline specification (`&pipe=<base64url>`).
    pub pipeline: Option<pipeline::ValidatedPipelineUrl>,
}

fn normalize_slug_tokens(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut prev_sep = true;
    for ch in raw.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
            prev_sep = false;
        } else if !prev_sep {
            out.push('_');
            prev_sep = true;
        }
    }
    if out.ends_with('_') {
        out.pop();
    }
    out
}

/// Returns true if `requested` matches a tab's canonical `url_slug` or display `tab_name`.
#[must_use]
pub fn matches_tab_slug(requested: &str, canonical_slug: &str, tab_name: &str) -> bool {
    let req = normalize_slug_tokens(requested);
    if req.is_empty() {
        return false;
    }
    req == normalize_slug_tokens(canonical_slug)
        || req == normalize_slug_tokens(tab_name)
        || (canonical_slug == "morphogenesis" && req == "biology")
}

// theory_verification!
