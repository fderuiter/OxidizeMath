//! Self-contained browser URL fragment (`#tab=<slug>&tool=<id>&state=<base64url>`)
//! encoding, startup hydration parsing, 300 ms debounced `History::replaceState` sync,
//! and cross-platform clipboard share-link helpers.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

mod base64url;
mod browser;
mod codec;
mod history_bridge;
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
pub use codec::{
    encode_share_fragment, encode_share_fragment_with_challenge, matches_tab_slug,
    parse_share_location, parse_workspace_or_share_location, validate_url_preset_json,
    ShareUrlParams, CANONICAL_WEB_BASE_URL, MAX_DECODED_URL_STATE_BYTES, MAX_URL_FRAGMENT_BYTES,
};
pub use history_bridge::{drain_pending_browser_navigations, install_browser_history_listeners};
#[cfg(test)]
pub use history_bridge::{enqueue_browser_navigation_for_test, reset_history_bridge_for_test};
pub use sync::{PaneUrlSignature, UrlStateSignature, UrlSyncController, URL_SYNC_DEBOUNCE_MS};
pub use workspace_codec::{
    decode_workspace_share_param, encode_single_pane_smart_fragment,
    encode_single_pane_smart_fragment_with_challenge, encode_workspace_share_fragment,
    validate_workspace_url_preset_json, ParsedShareLocation, WorkspaceShareParams,
    MAX_DECODED_WORKSPACE_URL_BYTES, WORKSPACE_SHARE_SCHEMA_VERSION,
};

// theory_verification!
