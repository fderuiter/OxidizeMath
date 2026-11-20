#![cfg_attr(any(), verified(opt_out = "tests"))]

const INDEX_HTML: &str = include_str!("../index.html");
const FAVICON_SVG: &str = include_str!("../icons/favicon.svg");
const SITE_WEBMANIFEST: &str = include_str!("../site.webmanifest");
const ICON_32_PNG: &[u8] = include_bytes!("../icons/32x32.png");
const ICON_128_PNG: &[u8] = include_bytes!("../icons/128x128.png");
const ICON_180_PNG: &[u8] = include_bytes!("../icons/180x180.png");
const ICON_192_PNG: &[u8] = include_bytes!("../icons/192x192.png");
const ICON_256_PNG: &[u8] = include_bytes!("../icons/256x256.png");
const ICON_512_PNG: &[u8] = include_bytes!("../icons/512x512.png");
const ICON_512_MASKABLE_PNG: &[u8] = include_bytes!("../icons/512x512-maskable.png");
const OG_IMAGE_PNG: &[u8] = include_bytes!("../icons/og-image.png");
const ICON_ICO: &[u8] = include_bytes!("../icons/icon.ico");
const ICON_ICNS: &[u8] = include_bytes!("../icons/icon.icns");
const MAX_ICON_ASSET_BYTES: usize = 20_480;

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
    assert!(
        !INDEX_HTML.contains("property=\"og:image\" content=\"512x512.png\""),
        "index.html must not use relative 512x512.png for og:image"
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
        "property=\"og:url\" content=\"https://fderuiter.github.io/OxidizeMath/\"",
        "property=\"og:image\" content=\"https://fderuiter.github.io/OxidizeMath/og-image.png\"",
        "property=\"og:image:width\" content=\"1200\"",
        "property=\"og:image:height\" content=\"630\"",
        "property=\"og:image:alt\"",
        "name=\"twitter:card\" content=\"summary_large_image\"",
    ] {
        assert!(
            INDEX_HTML.contains(snippet),
            "index.html head must contain metadata snippet: {snippet}"
        );
    }
}

#[test]
fn test_favicon_manifest_and_trunk_staging_links() {
    assert!(
        !INDEX_HTML.contains("<link rel=\"apple-touch-icon\" href=\"256x256.png\">"),
        "apple-touch-icon must not point to transparent 256x256.png"
    );
    for link_snippet in [
        "<link rel=\"icon\" href=\"favicon.ico\" sizes=\"any\">",
        "<link rel=\"icon\" href=\"favicon.svg\" type=\"image/svg+xml\">",
        "<link rel=\"icon\" type=\"image/png\" sizes=\"32x32\" href=\"32x32.png\">",
        "<link rel=\"apple-touch-icon\" href=\"180x180.png\">",
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
        "icons/180x180.png",
        "icons/192x192.png",
        "icons/256x256.png",
        "icons/512x512.png",
        "icons/512x512-maskable.png",
        "icons/og-image.png",
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
    for rule in [
        "<style>",
        "@media (prefers-color-scheme: light)",
        "#172554",
        "#eff6ff",
        "#2563eb",
        "#1d4ed8",
    ] {
        assert!(
            FAVICON_SVG.contains(rule),
            "icons/favicon.svg must include dark/light brand rule: {rule}"
        );
    }

    let manifest: serde_json::Value =
        serde_json::from_str(SITE_WEBMANIFEST).expect("site.webmanifest must parse as valid JSON");
    assert_eq!(manifest["name"], "Math Explorer");
    assert_eq!(manifest["short_name"], "Math Explorer");
    let icons = manifest["icons"]
        .as_array()
        .expect("site.webmanifest must define an icons array");
    assert!(
        icons.len() >= 7,
        "site.webmanifest must declare vector, standard, 192x192, and maskable icons"
    );
    assert!(
        icons.iter().any(|entry| {
            entry["src"] == "192x192.png"
                && entry["sizes"] == "192x192"
                && entry["purpose"] == "any"
        }),
        "site.webmanifest must register 192x192.png with purpose 'any'"
    );
    assert!(
        icons.iter().any(|entry| {
            entry["src"] == "512x512-maskable.png"
                && entry["sizes"] == "512x512"
                && entry["purpose"] == "maskable"
        }),
        "site.webmanifest must register 512x512-maskable.png with purpose 'maskable'"
    );
}

fn decode_rgba_png(label: &str, data: &[u8]) -> (u32, u32, Vec<[u8; 4]>) {
    assert!(
        data.starts_with(b"\x89PNG\r\n\x1a\n"),
        "{label} must begin with PNG signature"
    );
    let decoder = png::Decoder::new(std::io::Cursor::new(data));
    let mut reader = decoder
        .read_info()
        .unwrap_or_else(|err| panic!("{label} must parse PNG header: {err}"));
    let mut buf = vec![0_u8; reader.output_buffer_size().expect("PNG buffer size")];
    let info = reader
        .next_frame(&mut buf)
        .unwrap_or_else(|err| panic!("{label} must decode PNG frame: {err}"));
    assert_eq!(
        info.color_type,
        png::ColorType::Rgba,
        "{label} must be RGBA"
    );
    assert_eq!(
        info.bit_depth,
        png::BitDepth::Eight,
        "{label} must be 8-bit"
    );
    let (pixels, rem) = buf[..info.buffer_size()].as_chunks::<4>();
    assert!(rem.is_empty(), "{label} RGBA buffer must align to 4 bytes");
    (info.width, info.height, pixels.to_vec())
}

fn assert_brand_palette(label: &str, pixels: &[[u8; 4]], expect_opaque: bool) {
    assert!(
        !pixels
            .iter()
            .any(|&[r, g, b, a]| r == 0 && g == 0 && b == 255 && a > 0),
        "{label} must not contain legacy #0000ff placeholder pixels"
    );
    for (color_name, expected) in [
        ("badge #172554", [23, 37, 84, 255]),
        ("stroke #3b82f6", [59, 130, 246, 255]),
        ("wave #ffffff", [255, 255, 255, 255]),
        ("node #38bdf8", [56, 189, 248, 255]),
    ] {
        assert!(
            pixels.contains(&expected),
            "{label} must render .brand-mark {color_name}"
        );
    }
    if expect_opaque {
        assert!(
            pixels.iter().all(|px| px[3] == 255),
            "{label} must have opaque alpha (255) on every pixel"
        );
    } else {
        assert_eq!(
            pixels[0][3], 0,
            "{label} corner pixel must remain transparent"
        );
    }
}

#[test]
fn test_branded_png_icons_and_alpha_invariants() {
    let assets: &[(&str, &[u8], u32, u32, bool)] = &[
        ("32x32.png", ICON_32_PNG, 32, 32, false),
        ("128x128.png", ICON_128_PNG, 128, 128, false),
        ("180x180.png", ICON_180_PNG, 180, 180, true),
        ("192x192.png", ICON_192_PNG, 192, 192, false),
        ("256x256.png", ICON_256_PNG, 256, 256, false),
        ("512x512.png", ICON_512_PNG, 512, 512, false),
        (
            "512x512-maskable.png",
            ICON_512_MASKABLE_PNG,
            512,
            512,
            true,
        ),
        ("og-image.png", OG_IMAGE_PNG, 1200, 630, true),
    ];
    for &(label, bytes, exp_w, exp_h, expect_opaque) in assets {
        assert!(
            bytes.len() < MAX_ICON_ASSET_BYTES,
            "{label} ({} bytes) must stay under {MAX_ICON_ASSET_BYTES} bytes",
            bytes.len()
        );
        let (w, h, pixels) = decode_rgba_png(label, bytes);
        assert_eq!((w, h), (exp_w, exp_h), "{label} dimensions mismatch");
        assert_brand_palette(label, &pixels, expect_opaque);
    }

    let (_, _, maskable_px) = decode_rgba_png("512x512-maskable.png", ICON_512_MASKABLE_PNG);
    let safe_radius = 512.0_f64 * 0.40;
    for (py, row) in maskable_px.as_chunks::<512>().0.iter().enumerate() {
        for (px, &rgba) in row.iter().enumerate() {
            if rgba != [23, 37, 84, 255] {
                let dist = (px as f64 + 0.5 - 256.0).hypot(py as f64 + 0.5 - 256.0);
                assert!(
                    dist <= safe_radius,
                    "maskable emblem pixel ({px}, {py}) at r={dist:.1} exceeds 80% safe zone"
                );
            }
        }
    }
}

#[test]
fn test_ico_and_icns_png_frames() {
    assert!(
        ICON_ICO.len() < MAX_ICON_ASSET_BYTES,
        "icon.ico ({} bytes) must stay under {MAX_ICON_ASSET_BYTES} bytes",
        ICON_ICO.len()
    );
    assert_eq!(&ICON_ICO[0..4], &[0, 0, 1, 0], "icon.ico ICONDIR header");
    let frame_count = u16::from_le_bytes([ICON_ICO[4], ICON_ICO[5]]) as usize;
    assert_eq!(frame_count, 4, "icon.ico must pack 4 PNG frames");

    let expected_dims = [16_u32, 32, 48, 256];
    for (idx, &exp_dim) in expected_dims.iter().enumerate() {
        let base = 6 + idx * 16;
        let dir_w = if ICON_ICO[base] == 0 {
            256
        } else {
            u32::from(ICON_ICO[base])
        };
        let dir_h = if ICON_ICO[base + 1] == 0 {
            256
        } else {
            u32::from(ICON_ICO[base + 1])
        };
        let size = u32::from_le_bytes(ICON_ICO[base + 8..base + 12].try_into().unwrap()) as usize;
        let offset =
            u32::from_le_bytes(ICON_ICO[base + 12..base + 16].try_into().unwrap()) as usize;
        assert_eq!((dir_w, dir_h), (exp_dim, exp_dim));
        assert!(size < MAX_ICON_ASSET_BYTES);
        let (w, h, pixels) = decode_rgba_png("icon.ico frame", &ICON_ICO[offset..offset + size]);
        assert_eq!((w, h), (exp_dim, exp_dim));
        assert_brand_palette("icon.ico frame", &pixels, false);
    }

    assert_eq!(&ICON_ICNS[0..4], b"icns", "icon.icns magic header");
    let total_len = u32::from_be_bytes(ICON_ICNS[4..8].try_into().unwrap()) as usize;
    assert_eq!(total_len, ICON_ICNS.len(), "icon.icns total length");
    let mut cursor = 8;
    let mut chunk_count = 0;
    while cursor < total_len {
        let chunk_len =
            u32::from_be_bytes(ICON_ICNS[cursor + 4..cursor + 8].try_into().unwrap()) as usize;
        let (_, _, pixels) = decode_rgba_png(
            "icon.icns chunk",
            &ICON_ICNS[cursor + 8..cursor + chunk_len],
        );
        assert_brand_palette("icon.icns chunk", &pixels, false);
        cursor += chunk_len;
        chunk_count += 1;
    }
    assert_eq!(chunk_count, 5, "icon.icns must pack 5 PNG icon chunks");
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
