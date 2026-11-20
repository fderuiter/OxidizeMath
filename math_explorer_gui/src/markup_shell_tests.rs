#![cfg_attr(any(), verified(opt_out = "tests"))]

const INDEX_HTML: &str = include_str!("../index.html");
const RELOAD_JS: &str = include_str!("../reload.js");
const FAVICON_SVG: &str = include_str!("../icons/favicon.svg");
const SITE_WEBMANIFEST: &str = include_str!("../site.webmanifest");

#[test]
fn test_document_title_and_metadata_tags() {
    assert!(
        INDEX_HTML.contains("<title>Math Explorer</title>"),
        "index.html must set <title>Math Explorer</title>"
    );
    assert!(
        !INDEX_HTML.contains("<title>Math Explorer GUI</title>"),
        "index.html must not retain legacy <title>Math Explorer GUI</title>"
    );
    for snippet in [
        "name=\"description\"",
        "name=\"application-name\" content=\"Math Explorer\"",
        "name=\"color-scheme\" content=\"dark light\"",
        "name=\"theme-color\" media=\"(prefers-color-scheme: dark)\"",
        "name=\"theme-color\" media=\"(prefers-color-scheme: light)\"",
        "property=\"og:title\" content=\"Math Explorer\"",
        "property=\"og:description\"",
        "property=\"og:type\" content=\"website\"",
        "property=\"og:image\"",
    ] {
        assert!(
            INDEX_HTML.contains(snippet),
            "index.html head must contain metadata snippet: {snippet}"
        );
    }
}

#[test]
fn test_favicon_manifest_and_trunk_staging_links() {
    for link_snippet in [
        "<link rel=\"icon\" href=\"favicon.ico\" sizes=\"any\">",
        "<link rel=\"icon\" href=\"favicon.svg\" type=\"image/svg+xml\">",
        "<link rel=\"apple-touch-icon\" href=\"256x256.png\">",
        "<link rel=\"manifest\" href=\"site.webmanifest\">",
    ] {
        assert!(
            INDEX_HTML.contains(link_snippet),
            "index.html must declare browser link tag: {link_snippet}"
        );
    }

    for staged_asset in [
        "icons/favicon.svg",
        "icons/icon.ico",
        "icons/32x32.png",
        "icons/128x128.png",
        "icons/256x256.png",
        "icons/512x512.png",
        "site.webmanifest",
    ] {
        let directive = format!("data-trunk rel=\"copy-file\" href=\"{staged_asset}\"");
        assert!(
            INDEX_HTML.contains(&directive),
            "index.html must stage asset via Trunk copy-file: {directive}"
        );
    }
}

#[test]
fn test_loading_indicator_contains_decorative_inline_svg() {
    let loading_start = INDEX_HTML
        .find("id=\"loading_indicator\"")
        .expect("#loading_indicator must exist");
    let loading_end = INDEX_HTML[loading_start..]
        .find("</div>\n\n    <!-- WebGL error boundary")
        .map(|offset| loading_start + offset)
        .expect("#loading_indicator closing block must exist");
    let loading_block = &INDEX_HTML[loading_start..loading_end];

    let svg_start = loading_block
        .find("<svg")
        .expect("#loading_indicator must include an inline <svg> brand mark");
    let svg_open_end = loading_block[svg_start..]
        .find('>')
        .expect("<svg> opening tag must close")
        + svg_start;
    let svg_open_tag = &loading_block[svg_start..=svg_open_end];

    assert!(
        svg_open_tag.contains("aria-hidden=\"true\""),
        "Inline brand mark <svg> must set aria-hidden=\"true\", found: {svg_open_tag}"
    );
    assert!(
        loading_block.contains("</svg>"),
        "Inline brand mark <svg> must close inside #loading_indicator"
    );
}

#[test]
fn test_trunk_rust_link_configures_initializer() {
    let rust_link_start = INDEX_HTML
        .find("<link data-trunk rel=\"rust\"")
        .expect("Trunk rust <link> tag must exist in index.html");
    let rust_link_end = INDEX_HTML[rust_link_start..]
        .find('>')
        .expect("Trunk rust <link> tag must close")
        + rust_link_start;
    let rust_link_tag = &INDEX_HTML[rust_link_start..=rust_link_end];

    assert!(
        rust_link_tag.contains("data-initializer=\"initializer.js\""),
        "Trunk rust link must configure data-initializer=\"initializer.js\", found: {rust_link_tag}"
    );
}

#[test]
fn test_loading_indicator_progress_and_byte_counter() {
    let loading_start = INDEX_HTML
        .find("id=\"loading_indicator\"")
        .expect("#loading_indicator must exist in index.html");
    let error_start = INDEX_HTML
        .find("id=\"error_container\"")
        .expect("#error_container must exist after #loading_indicator");
    assert!(loading_start < error_start);

    let loading_section = &INDEX_HTML[loading_start..error_start];
    assert!(
        loading_section.contains("aria-busy=\"true\""),
        "#loading_indicator must initialize aria-busy=\"true\""
    );
    assert!(
        loading_section.contains("<progress")
            && loading_section.contains("id=\"loading_progress\"")
            && loading_section.contains("role=\"progressbar\""),
        "#loading_indicator must contain <progress id=\"loading_progress\" role=\"progressbar\">"
    );
    assert!(
        loading_section.contains("id=\"loading_bytes\""),
        "#loading_indicator must contain a byte counter element (#loading_bytes)"
    );
}

#[test]
fn test_adaptive_shell_theme_media_queries() {
    for rule in [
        "@media (prefers-color-scheme: light)",
        "@media (prefers-contrast: more)",
        "@media (forced-colors: active)",
        "border-color: GrayText;",
        "border-top-color: Highlight;",
    ] {
        assert!(
            INDEX_HTML.contains(rule),
            "index.html stylesheet must define adaptive theme rule: {rule}"
        );
    }
}

#[test]
fn test_media_queries_scoped_to_not_data_theme_and_contrast_precedence() {
    let light_pos = INDEX_HTML
        .find("@media (prefers-color-scheme: light)")
        .expect("light media query must exist");
    let contrast_pos = INDEX_HTML
        .find("@media (prefers-contrast: more)")
        .expect("prefers-contrast media query must exist");
    let forced_pos = INDEX_HTML
        .find("@media (forced-colors: active)")
        .expect("forced-colors media query must exist");

    assert!(
        light_pos < contrast_pos,
        "@media (prefers-contrast: more) must follow @media (prefers-color-scheme: light)"
    );
    let light_block = &INDEX_HTML[light_pos..contrast_pos];
    let contrast_block = &INDEX_HTML[contrast_pos..forced_pos];

    for selector in [
        "html:not([data-theme])",
        "html:not([data-theme]) body",
        "html:not([data-theme]) #loading_indicator",
        "html:not([data-theme]) #error_container",
        "html:not([data-theme]) .spinner",
    ] {
        assert!(
            light_block.contains(selector),
            "light media block must scope selector: {selector}"
        );
        assert!(
            contrast_block.contains(selector),
            "prefers-contrast media block must scope selector: {selector}"
        );
    }
}

#[test]
fn test_explicit_data_theme_selectors_and_theme_color_tokens() {
    for selector in [
        "html[data-theme=\"dark\"]",
        "html[data-theme=\"dark\"] body",
        "html[data-theme=\"dark\"] #loading_indicator",
        "html[data-theme=\"dark\"] #error_container",
        "html[data-theme=\"dark\"] .spinner",
        "html[data-theme=\"light\"]",
        "html[data-theme=\"light\"] body",
        "html[data-theme=\"light\"] #loading_indicator",
        "html[data-theme=\"light\"] #error_container",
        "html[data-theme=\"light\"] .spinner",
        "html[data-theme=\"high-contrast\"]",
        "html[data-theme=\"high-contrast\"] body",
        "html[data-theme=\"high-contrast\"] #loading_indicator",
        "html[data-theme=\"high-contrast\"] #error_container",
        "html[data-theme=\"high-contrast\"] .spinner",
    ] {
        assert!(
            INDEX_HTML.contains(selector),
            "index.html must define explicit override selector: {selector}"
        );
    }

    for token in [
        "content=\"#121214\"",
        "content=\"#f8f8f8\"",
        "content=\"#000000\"",
    ] {
        assert!(
            INDEX_HTML.contains(token),
            "index.html head must declare theme-color token: {token}"
        );
    }
}

#[test]
fn test_reload_js_syncs_saved_theme_from_local_storage_safely() {
    for required in [
        "window.localStorage",
        "getItem('app')",
        "JSON.parse",
        "try {",
        "catch (_err)",
        "schema_version",
        "app_state",
        "dataset.theme",
        "color-scheme",
        "meta[name=\"theme-color\"]",
        "#121214",
        "#f8f8f8",
        "#000000",
        "high-contrast",
    ] {
        assert!(
            RELOAD_JS.contains(required),
            "reload.js must include pre-boot theme synchronization contract: {required}"
        );
    }
}

#[test]
fn test_prefers_reduced_motion_halts_spinner() {
    let media_pos = INDEX_HTML
        .find("@media (prefers-reduced-motion: reduce)")
        .expect("index.html must include @media (prefers-reduced-motion: reduce) styles");
    let style_end = INDEX_HTML[media_pos..]
        .find("</style>")
        .expect("</style> tag must exist")
        + media_pos;
    let reduced_motion_block = &INDEX_HTML[media_pos..style_end];

    assert!(
        reduced_motion_block.contains(".spinner")
            && reduced_motion_block.contains("animation: none"),
        "prefers-reduced-motion block must disable .spinner animation"
    );
}

#[test]
fn test_prerendered_twin_slot_live_regions() {
    for id in [
        "id=\"aria-live-polite\"",
        "id=\"aria-live-polite-slot-0\"",
        "id=\"aria-live-polite-slot-1\"",
        "id=\"aria-live-assertive\"",
        "id=\"aria-live-assertive-slot-0\"",
        "id=\"aria-live-assertive-slot-1\"",
    ] {
        assert!(
            INDEX_HTML.contains(id),
            "index.html must pre-render twin-slot live region node {id}"
        );
    }
}

#[test]
fn test_favicon_svg_and_site_webmanifest_assets() {
    assert!(
        FAVICON_SVG.contains("<svg") && FAVICON_SVG.contains("</svg>"),
        "icons/favicon.svg must be a non-empty valid SVG document"
    );

    let manifest: serde_json::Value =
        serde_json::from_str(SITE_WEBMANIFEST).expect("site.webmanifest must parse as valid JSON");
    assert_eq!(manifest["name"], "Math Explorer");
    assert_eq!(manifest["short_name"], "Math Explorer");
    let icons = manifest["icons"]
        .as_array()
        .expect("site.webmanifest must define an icons array");
    assert!(
        icons.len() >= 5,
        "site.webmanifest must declare vector and raster icon entries"
    );
}

#[test]
fn test_error_container_supports_categories_and_webgl_instructions() {
    assert!(
        INDEX_HTML.contains("data-error-category=\"webgl\""),
        "#error_container must default to data-error-category=\"webgl\""
    );
    for required in [
        "id=\"error_title\"",
        "id=\"error_message\"",
        "id=\"error_recovery_steps\"",
        "id=\"error_recovery_list\"",
        "id=\"webgl_instructions\"",
        "id=\"error_details\"",
        "data-error-category=\"network\"",
        "data-error-category=\"compile\"",
        "data-error-category=\"timeout\"",
    ] {
        assert!(
            INDEX_HTML.contains(required),
            "index.html must include {required} for multi-category error recovery"
        );
    }
}
