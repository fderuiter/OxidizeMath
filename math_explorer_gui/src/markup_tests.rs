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
    fn test_critical_inline_styles_precede_external_scripts() {
        let style_start = INDEX_HTML
            .find("<style>")
            .expect("<style> block must exist in index.html");
        let style_end = INDEX_HTML
            .find("</style>")
            .expect("</style> closing tag must exist in index.html");
        let first_script = INDEX_HTML
            .find("<script")
            .expect("startup <script> tags must exist in index.html");

        assert!(
            style_start < style_end && style_end < first_script,
            "Critical inline <style> block must appear before any <script> tag"
        );
    }

    #[test]
    fn test_all_startup_scripts_are_external_and_deferred() {
        let mut search_from = 0;
        let mut script_sources = Vec::new();

        while let Some(offset) = INDEX_HTML[search_from..].find("<script") {
            let tag_start = search_from + offset;
            let tag_end = INDEX_HTML[tag_start..]
                .find('>')
                .map(|end| tag_start + end)
                .expect("<script> opening tag must close");
            let open_tag = &INDEX_HTML[tag_start..=tag_end];

            assert!(
                open_tag.contains("src=\""),
                "All <script> tags must reference external files (no inline scripts): {open_tag}"
            );
            assert!(
                open_tag.contains(" defer") || open_tag.contains("\tdefer"),
                "All startup <script> tags must be marked defer: {open_tag}"
            );

            let close_start = INDEX_HTML[(tag_end + 1)..]
                .find("</script>")
                .map(|end| tag_end + 1 + end)
                .expect("</script> closing tag must exist");
            let inline_body = INDEX_HTML[(tag_end + 1)..close_start].trim();
            assert!(
                inline_body.is_empty(),
                "External <script> tag must have empty inline body, found: {inline_body}"
            );

            script_sources.push(open_tag.to_string());
            search_from = close_start + "</script>".len();
        }

        assert!(
            script_sources
                .iter()
                .any(|t| t.contains("src=\"coi-serviceworker.min.js\"")),
            "index.html must include deferred coi-serviceworker.min.js"
        );
        assert!(
            script_sources
                .iter()
                .any(|t| t.contains("src=\"reload.js\"")),
            "index.html must include deferred reload.js"
        );
        assert!(
            script_sources
                .iter()
                .any(|t| t.contains("src=\"bootstrap.js\"")),
            "index.html must include deferred bootstrap.js"
        );
    }

    #[test]
    fn test_strict_csp_and_dom_contracts_preserved() {
        assert!(
            INDEX_HTML.contains("script-src 'self' 'wasm-unsafe-eval' blob:;"),
            "Strict script-src CSP directive without unsafe-inline must be preserved"
        );
        assert!(
            !INDEX_HTML.contains("rel=\"preload\"") && !INDEX_HTML.contains("rel=\"modulepreload\""),
            "Source HTML must not contain unconditional preload or modulepreload hints"
        );
        assert!(
            INDEX_HTML.contains("id=\"loading_indicator\""),
            "Loading indicator DOM id must be preserved"
        );
        assert!(
            INDEX_HTML.contains("id=\"error_container\"")
                && INDEX_HTML.contains("id=\"error_details\""),
            "Error boundary DOM ids must be preserved"
        );
    }
}
