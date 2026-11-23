use super::{
    build_share_url_from_fragment, read_browser_location_parts, replace_browser_url_fragment,
};
use crate::url_state::parse_share_location;
use wasm_bindgen_test::wasm_bindgen_test;

#[wasm_bindgen_test]
fn wasm_url_state_replaces_fragment_and_builds_live_origin_share_links() {
    let window = web_sys::window().expect("browser window must exist");
    let history = window.history().expect("browser history must exist");
    let initial_href = window.location().href().unwrap_or_default();

    history
        .replace_state_with_url(
            &wasm_bindgen::JsValue::NULL,
            "",
            Some("/index.html?keep=1&tab=stale&tool=old#initial"),
        )
        .expect("replaceState setup should succeed");

    replace_browser_url_fragment("tab=physics&tool=quantum")
        .expect("replace_browser_url_fragment should succeed");

    let (hash, search) =
        read_browser_location_parts().expect("wasm32 location parts should be readable");
    assert_eq!(hash, "#tab=physics&tool=quantum");
    assert_eq!(search, "?keep=1");

    let share_url = build_share_url_from_fragment("#tab=chaos&tool=lorenz");
    let origin = window.location().origin().expect("origin should resolve");
    assert_eq!(
        share_url,
        format!("{origin}/index.html?keep=1#tab=chaos&tool=lorenz")
    );

    replace_browser_url_fragment("").expect("clearing URL fragment should succeed");
    let (cleared_hash, _) =
        read_browser_location_parts().expect("location parts after clear should resolve");
    assert_eq!(cleared_hash, "");

    if !initial_href.is_empty() {
        let _ =
            history.replace_state_with_url(&wasm_bindgen::JsValue::NULL, "", Some(&initial_href));
    }
}

#[wasm_bindgen_test]
fn wasm_challenge_url_hydration_and_lock_clamping() {
    let window = web_sys::window().expect("browser window must exist");
    let history = window.history().expect("browser history must exist");
    let initial_href = window.location().href().unwrap_or_default();

    replace_browser_url_fragment("tab=mri&challenge=bloch_pi_inversion")
        .expect("setting challenge fragment should succeed");
    let (hash, search) = read_browser_location_parts().expect("location parts readable");
    let parsed = parse_share_location(&hash, &search)
        .expect("challenge fragment must be valid")
        .expect("ShareUrlParams expected");
    assert_eq!(parsed.tab, "mri");
    assert_eq!(parsed.challenge.as_deref(), Some("bloch_pi_inversion"));

    let mut app = crate::app::MathExplorerApp::default();
    assert!(app
        .apply_share_location_strings(&hash, &search, None)
        .expect("wasm app challenge hydration must succeed"));
    let encoded = app
        .encode_current_share_url()
        .expect("encoding active challenge URL must succeed");
    assert!(encoded.contains("challenge=bloch_pi_inversion"));

    if !initial_href.is_empty() {
        let _ =
            history.replace_state_with_url(&wasm_bindgen::JsValue::NULL, "", Some(&initial_href));
    }
}
