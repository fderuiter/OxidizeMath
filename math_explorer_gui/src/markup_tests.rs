#[cfg(test)]
mod tests {
    const INDEX_HTML: &str = include_str!("../index.html");

    #[test]
    fn test_canvas_tabindex_and_attributes() {
        let canvas_start = INDEX_HTML
            .find("<canvas")
            .expect("<canvas> tag must exist in index.html");
        let canvas_open_end = INDEX_HTML[canvas_start..]
            .find('>')
            .expect("canvas opening tag must close")
            + canvas_start;
        let canvas_tag = &INDEX_HTML[canvas_start..=canvas_open_end];

        assert!(
            canvas_tag.contains("id=\"the_canvas_id\""),
            "Canvas tag must contain id=\"the_canvas_id\", found: {canvas_tag}"
        );
        assert!(
            canvas_tag.contains("tabindex=\"0\""),
            "Canvas tag must contain tabindex=\"0\", found: {canvas_tag}"
        );
    }

    #[test]
    fn test_reload_button_identifier() {
        assert!(
            INDEX_HTML.contains("id=\"webgl_reload_button\""),
            "index.html must contain an element with id=\"webgl_reload_button\""
        );

        let button_pos = INDEX_HTML
            .find("id=\"webgl_reload_button\"")
            .expect("element with id webgl_reload_button must exist");
        let tag_start = INDEX_HTML[..button_pos].rfind('<').unwrap_or(0);
        let tag_end = INDEX_HTML[button_pos..]
            .find('>')
            .map(|offset| button_pos + offset)
            .unwrap_or(INDEX_HTML.len());
        let button_tag = &INDEX_HTML[tag_start..=tag_end];

        assert!(
            button_tag.starts_with("<button"),
            "Expected <button> element for webgl_reload_button, found: {button_tag}"
        );
        assert!(
            button_tag.contains("class=\"reload-btn\""),
            "Reload button must preserve reload-btn class"
        );
    }

    #[test]
    fn test_canvas_encloses_accessible_fallback_text() {
        let canvas_start = INDEX_HTML
            .find("<canvas")
            .expect("<canvas> tag must exist in index.html");
        let canvas_open_end = INDEX_HTML[canvas_start..]
            .find('>')
            .expect("canvas opening tag must close")
            + canvas_start;
        let canvas_close_start = INDEX_HTML[canvas_open_end..]
            .find("</canvas>")
            .expect("</canvas> closing tag must exist in index.html")
            + canvas_open_end;

        let fallback_content = &INDEX_HTML[(canvas_open_end + 1)..canvas_close_start];
        let trimmed_text = fallback_content.trim();

        assert!(
            !trimmed_text.is_empty(),
            "Canvas tag must enclose non-empty accessible fallback text content"
        );
        assert!(
            trimmed_text.len() >= 10,
            "Fallback text should be descriptive (at least 10 chars), got: {trimmed_text}"
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
}
