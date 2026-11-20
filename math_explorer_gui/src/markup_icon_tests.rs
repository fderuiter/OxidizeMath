#![cfg_attr(any(), verified(opt_out = "tests"))]

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
