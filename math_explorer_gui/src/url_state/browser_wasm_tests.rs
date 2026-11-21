use super::{
    build_share_url_from_fragment, read_browser_location_parts, replace_browser_url_fragment,
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
