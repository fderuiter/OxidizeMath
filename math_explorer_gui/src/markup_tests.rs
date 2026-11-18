#[cfg(test)]
mod tests {
    const INDEX_HTML: &str = include_str!("../index.html");
    const FAVICON_SVG: &str = include_str!("../icons/favicon.svg");
    const SITE_WEBMANIFEST: &str = include_str!("../site.webmanifest");

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
    fn test_favicon_svg_and_site_webmanifest_assets() {
        assert!(
            FAVICON_SVG.contains("<svg") && FAVICON_SVG.contains("</svg>"),
            "icons/favicon.svg must be a non-empty valid SVG document"
        );

        let manifest: serde_json::Value = serde_json::from_str(SITE_WEBMANIFEST)
            .expect("site.webmanifest must parse as valid JSON");
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
}
