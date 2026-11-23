use super::{
    build_share_url_from_fragment, is_browser_location_self_synced, read_browser_location_parts,
    replace_browser_url_fragment,
};
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
            Some("/index.html?keep=1&ws=oldws&tab=stale&tool=old#initial"),
        )
        .expect("replaceState setup should succeed");
    assert!(
        !is_browser_location_self_synced("#initial", "?keep=1"),
        "External History.state == null must not be treated as self-synced"
    );

    replace_browser_url_fragment("ws=eyJ2IjoyfQ")
        .expect("replace_browser_url_fragment should succeed");

    let (hash, search) =
        read_browser_location_parts().expect("wasm32 location parts should be readable");
    assert_eq!(hash, "#ws=eyJ2IjoyfQ");
    assert_eq!(
        search, "?keep=1",
        "stale ws/tab/tool query params must be stripped"
    );
    assert!(
        is_browser_location_self_synced(&hash, &search),
        "History.state written by replace_browser_url_fragment must report self-synced"
    );
    assert!(
        !is_browser_location_self_synced("#tab=chaos", &search),
        "Mismatched hash must not match self-synced History.state"
    );

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
    assert!(is_browser_location_self_synced("", "?keep=1"));

    if !initial_href.is_empty() {
        let _ =
            history.replace_state_with_url(&wasm_bindgen::JsValue::NULL, "", Some(&initial_href));
    }
}
