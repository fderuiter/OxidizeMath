use super::{query_browser_media_flags, resolve_startup_theme, sync_document_theme, ThemeMode};
use wasm_bindgen::JsCast;
use wasm_bindgen_test::wasm_bindgen_test;

#[wasm_bindgen_test]
fn wasm_theme_queries_media_flags_and_synchronizes_document_metadata() {
    let (forced_colors, prefers_contrast, prefers_light) = query_browser_media_flags();
    let resolved = resolve_startup_theme(forced_colors, prefers_contrast, prefers_light, None);
    assert!(matches!(
        resolved,
        ThemeMode::Dark | ThemeMode::Light | ThemeMode::HighContrast
    ));

    let doc = web_sys::window()
        .and_then(|w| w.document())
        .expect("browser document should exist");
    let root = doc
        .document_element()
        .expect("documentElement should exist")
        .dyn_into::<web_sys::HtmlElement>()
        .expect("root should be HtmlElement");

    for (mode, expected_theme, expected_scheme, expected_hex) in [
        (ThemeMode::Light, "light", "light", "#f8f8f8"),
        (ThemeMode::HighContrast, "high-contrast", "dark", "#000000"),
        (ThemeMode::Dark, "dark", "dark", "#121214"),
    ] {
        sync_document_theme(mode);
        assert_eq!(
            root.get_attribute("data-theme").as_deref(),
            Some(expected_theme)
        );
        assert_eq!(
            root.style()
                .get_property_value("color-scheme")
                .unwrap_or_default(),
            expected_scheme
        );
        let scheme_meta = doc
            .query_selector("meta[name=\"color-scheme\"]")
            .ok()
            .flatten()
            .expect("color-scheme meta element should exist");
        assert_eq!(
            scheme_meta.get_attribute("content").as_deref(),
            Some(expected_scheme)
        );
        let theme_meta = doc
            .query_selector("meta[name=\"theme-color\"]")
            .ok()
            .flatten()
            .expect("theme-color meta element should exist");
        assert_eq!(
            theme_meta.get_attribute("content").as_deref(),
            Some(expected_hex)
        );
    }
}
