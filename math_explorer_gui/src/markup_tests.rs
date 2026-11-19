#[cfg(test)]
#[path = "markup_shell_tests.rs"]
mod shell_tests;

#[cfg(test)]
mod tests {
    const INDEX_HTML: &str = include_str!("../index.html");
    const TRUNK_TOML: &str = include_str!("../Trunk.toml");
    const BOOTSTRAP_JS: &str = include_str!("../bootstrap.js");

    fn extract_meta_csp(html: &str) -> &str {
        let csp_marker = "http-equiv=\"Content-Security-Policy\"";
        let meta_pos = html
            .find(csp_marker)
            .expect("index.html must declare a Content-Security-Policy meta tag");
        let tag_start = html[..meta_pos]
            .rfind("<meta")
            .expect("CSP marker must reside inside a <meta> tag");
        let tag_end = html[meta_pos..]
            .find('>')
            .map(|offset| meta_pos + offset)
            .expect("CSP <meta> tag must close");
        let meta_tag = &html[tag_start..=tag_end];
        let content_prefix = "content=\"";
        let content_start = meta_tag
            .find(content_prefix)
            .expect("CSP <meta> tag must specify content attribute")
            + content_prefix.len();
        let content_end = meta_tag[content_start..]
            .find('"')
            .expect("CSP content attribute must have closing quote")
            + content_start;
        &meta_tag[content_start..content_end]
    }

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

    fn trunk_section(section_header: &str) -> &str {
        let start = TRUNK_TOML
            .find(section_header)
            .unwrap_or_else(|| panic!("Trunk.toml must contain {section_header}"))
            + section_header.len();
        let rest = &TRUNK_TOML[start..];
        let end = rest.find("\n[").unwrap_or(rest.len());
        &rest[..end]
    }

    #[test]
    fn test_trunk_toml_disables_inline_script_and_preload_injection() {
        let build_section = trunk_section("[build]");
        for setting in [
            "inject_scripts = false",
            "filehash = false",
            "pattern_script = \"\"",
            "pattern_preload = \"\"",
        ] {
            assert!(
                build_section.lines().any(|line| line.trim() == setting),
                "[build] in Trunk.toml must set {setting}"
            );
        }

        let serve_section = trunk_section("[serve]");
        assert!(
            serve_section
                .lines()
                .any(|line| line.trim() == "no_autoreload = true"),
            "[serve] in Trunk.toml must set no_autoreload = true"
        );
    }

    #[test]
    fn test_csp_directives_match_and_forbid_inline_scripts() {
        let html_csp = extract_meta_csp(INDEX_HTML);
        let serve_section = trunk_section("[serve]");
        let csp_key = "\"Content-Security-Policy\" = \"";
        let csp_start = serve_section
            .find(csp_key)
            .expect("[serve] headers in Trunk.toml must define Content-Security-Policy")
            + csp_key.len();
        let csp_end = serve_section[csp_start..]
            .find('"')
            .expect("Trunk.toml Content-Security-Policy value must close")
            + csp_start;
        let trunk_csp = &serve_section[csp_start..csp_end];

        assert_eq!(
            html_csp, trunk_csp,
            "index.html meta CSP and Trunk.toml [serve] header CSP must match"
        );

        let script_directive = html_csp
            .split(';')
            .map(str::trim)
            .find(|d| d.starts_with("script-src "))
            .expect("CSP must contain a script-src directive");
        assert_eq!(
            script_directive, "script-src 'self' 'wasm-unsafe-eval' blob:",
            "script-src must strictly allow only 'self', 'wasm-unsafe-eval', and blob:"
        );
        assert!(
            !script_directive.contains("unsafe-inline") && !script_directive.contains("sha256-"),
            "script-src must not permit unsafe-inline or inline script hashes"
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

        for script_file in ["coi-serviceworker.min.js", "reload.js", "bootstrap.js"] {
            let src_attr = format!("src=\"{script_file}\"");
            assert!(
                script_sources.iter().any(|t| t.contains(&src_attr)),
                "index.html must include deferred {script_file}"
            );
        }
        assert!(
            INDEX_HTML.contains("<link data-trunk rel=\"copy-file\" href=\"bootstrap.js\"/>"),
            "index.html must copy bootstrap.js via data-trunk copy-file"
        );
        assert!(
            INDEX_HTML.contains("<link data-trunk rel=\"copy-file\" href=\"initializer.js\"/>"),
            "index.html must copy initializer.js via data-trunk copy-file"
        );
    }

    #[test]
    fn test_strict_csp_and_dom_contracts_preserved() {
        assert!(
            INDEX_HTML.contains("script-src 'self' 'wasm-unsafe-eval' blob:;"),
            "Strict script-src CSP directive without unsafe-inline must be preserved"
        );
        assert!(
            !INDEX_HTML.contains("rel=\"preload\"")
                && !INDEX_HTML.contains("rel=\"modulepreload\""),
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

    #[test]
    fn test_bootstrap_js_initializes_wasm_and_error_boundary() {
        for required in [
            "./initializer.js",
            "./math_explorer_gui.js",
            "./math_explorer_gui_bg.wasm",
            "window.wasmBindings",
            "TrunkApplicationStarted",
            "__mathExplorerIsolation",
            "loading_indicator",
            "error_container",
            "error_details",
        ] {
            assert!(
                BOOTSTRAP_JS.contains(required),
                "bootstrap.js must reference '{required}'"
            );
        }
    }
}
