//! Self-contained browser URL fragment (`#tab=<slug>&tool=<id>&state=<base64url>`)
//! encoding, startup hydration parsing, 300 ms debounced `History::replaceState` sync,
//! and cross-platform clipboard share-link helpers.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

mod base64url;
mod browser;
mod codec;
mod sync;

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

pub use base64url::{decode as decode_base64url, encode as encode_base64url};
#[cfg(target_arch = "wasm32")]
pub use browser::replace_browser_url_fragment;
pub use browser::{
    build_share_url_from_fragment, copy_share_url_to_clipboard, read_browser_location_parts,
};
#[cfg(test)]
pub use browser::{take_last_copied_share_url, with_mock_web_base_url};
pub use codec::{
    encode_share_fragment, encode_share_fragment_with_hud, matches_tab_slug, parse_share_location,
    validate_url_preset_json, ShareUrlParams, CANONICAL_WEB_BASE_URL, MAX_DECODED_URL_STATE_BYTES,
    MAX_URL_FRAGMENT_BYTES,
};
pub use sync::{UrlStateSignature, UrlSyncController, URL_SYNC_DEBOUNCE_MS};

// theory_verification!
