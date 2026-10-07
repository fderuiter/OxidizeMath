//! Cross-platform dataset and canvas snapshot export utilities.

/// Maximum resolution dimension (width or height) allowed for offscreen rendering.
pub const MAX_EXPORT_DIMENSION: u32 = 4096;

/// Default color palette used when drawing multiple datasets offscreen.
pub const EXPORT_PALETTE: &[egui::Color32] = &[
    egui::Color32::from_rgb(31, 119, 180),  // Blue
    egui::Color32::from_rgb(255, 127, 14),  // Orange
    egui::Color32::from_rgb(44, 160, 44),   // Green
    egui::Color32::from_rgb(214, 39, 40),   // Red
    egui::Color32::from_rgb(148, 103, 189), // Purple
    egui::Color32::from_rgb(140, 86, 75),   // Brown
    egui::Color32::from_rgb(227, 119, 194), // Pink
    egui::Color32::from_rgb(127, 127, 127), // Gray
];

/// Preset choices for export canvas aspect ratio.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub enum AspectRatioOption {
    /// Use user-specified width and height directly.
    Free,
    /// Standard 16:9 widescreen ratio.
    SixteenNine,
    /// Standard 4:3 ratio.
    FourThree,
    /// Square 1:1 ratio.
    OneOne,
}

impl AspectRatioOption {
    /// Return the numeric width-to-height aspect ratio, if fixed.
    pub fn ratio(&self) -> Option<f32> {
        match self {
            Self::Free => None,
            Self::SixteenNine => Some(16.0 / 9.0),
            Self::FourThree => Some(4.0 / 3.0),
            Self::OneOne => Some(1.0),
        }
    }
}

/// Options controlling high-resolution plot PNG export rendering.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct ExportOptions {
    /// Base output image width in pixels.
    pub width: u32,
    /// Base output image height in pixels.
    pub height: u32,
    /// Resolution multiplier scale factor (1.0, 2.0, 4.0, etc.).
    pub scale: f32,
    /// Thickness of rendered dataset plot lines in pixels.
    pub line_thickness: f32,
    /// Background opacity from 0.0 (transparent) to 1.0 (opaque).
    pub background_opacity: f32,
    /// Background color of the rendered plot image.
    pub background_color: egui::Color32,
    /// Target aspect ratio option.
    pub aspect_ratio: AspectRatioOption,
}

impl Default for ExportOptions {
    fn default() -> Self {
        Self {
            width: 1920,
            height: 1080,
            scale: 1.0,
            line_thickness: 2.0,
            background_opacity: 1.0,
            background_color: egui::Color32::from_rgb(18, 18, 18),
            aspect_ratio: AspectRatioOption::SixteenNine,
        }
    }
}

/// Format dataset points to CSV string.
pub fn format_csv(datasets: &[(String, Vec<[f64; 2]>)]) -> String {
    let mut csv = String::from("Dataset,X,Y\n");
    for (name, pts) in datasets {
        for p in pts {
            csv.push_str(&format!("{name},{},{}\n", p[0], p[1]));
        }
    }
    csv
}

/// Format dataset points to JSON string.
pub fn format_json(datasets: &[(String, Vec<[f64; 2]>)]) -> String {
    let mut json = String::from("[\n");
    let mut first = true;
    for (name, pts) in datasets {
        for p in pts {
            if !first {
                json.push_str(",\n");
            }
            first = false;
            json.push_str(&format!(
                "  {{\"dataset\": \"{}\", \"x\": {}, \"y\": {}}}",
                name, p[0], p[1]
            ));
        }
    }
    json.push_str("\n]");
    json
}

/// Export dataset points to CSV string and initiate download or file save dialog.
pub fn export_csv(datasets: &[(String, Vec<[f64; 2]>)]) {
    let csv = format_csv(datasets);
    save_file_dialog("export.csv", csv.as_bytes(), "CSV File", &["csv"]);
}

/// Export dataset points to JSON string and initiate download or file save dialog.
pub fn export_json(datasets: &[(String, Vec<[f64; 2]>)]) {
    let json = format_json(datasets);
    save_file_dialog("export.json", json.as_bytes(), "JSON File", &["json"]);
}

/// Render plot datasets into an `egui::ColorImage` buffer using offscreen rasterization.
pub fn export_custom_image(
    datasets: &[(String, Vec<[f64; 2]>)],
    options: &ExportOptions,
) -> Result<egui::ColorImage, String> {
    let target_w = (options.width as f32 * options.scale).round() as u32;
    let mut target_h = (options.height as f32 * options.scale).round() as u32;

    if let Some(ratio) = options.aspect_ratio.ratio()
        && ratio > 0.0
    {
        target_h = (target_w as f32 / ratio).round() as u32;
    }

    if target_w > MAX_EXPORT_DIMENSION || target_h > MAX_EXPORT_DIMENSION {
        return Err(format!(
            "Requested dimensions ({}x{}) exceed maximum limit of {}x{}",
            target_w, target_h, MAX_EXPORT_DIMENSION, MAX_EXPORT_DIMENSION
        ));
    }

    let width = target_w.max(1) as usize;
    let height = target_h.max(1) as usize;

    let bg_r = options.background_color.r();
    let bg_g = options.background_color.g();
    let bg_b = options.background_color.b();
    let bg_a = (options.background_color.a() as f32 * options.background_opacity.clamp(0.0, 1.0))
        .round() as u8;
    let bg_pixel = egui::Color32::from_rgba_unmultiplied(bg_r, bg_g, bg_b, bg_a);

    let mut pixels = vec![bg_pixel; width * height];

    let mut min_x = f64::INFINITY;
    let mut max_x = -f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = -f64::INFINITY;

    for (_, pts) in datasets {
        for p in pts {
            if p[0].is_finite() && p[1].is_finite() {
                min_x = min_x.min(p[0]);
                max_x = max_x.max(p[0]);
                min_y = min_y.min(p[1]);
                max_y = max_y.max(p[1]);
            }
        }
    }

    if min_x >= max_x {
        min_x -= 1.0;
        max_x += 1.0;
    }
    if min_y >= max_y {
        min_y -= 1.0;
        max_y += 1.0;
    }

    let margin_x = (max_x - min_x) * 0.05;
    let margin_y = (max_y - min_y) * 0.05;
    min_x -= margin_x;
    max_x += margin_x;
    min_y -= margin_y;
    max_y += margin_y;

    let padding = (20.0 * options.scale as f64).clamp(10.0, 100.0);
    let draw_w = (width as f64 - 2.0 * padding).max(1.0);
    let draw_h = (height as f64 - 2.0 * padding).max(1.0);

    let to_pixel = |p: [f64; 2]| -> (f64, f64) {
        let x_norm = (p[0] - min_x) / (max_x - min_x);
        let y_norm = (p[1] - min_y) / (max_y - min_y);
        let px = padding + x_norm * draw_w;
        let py = (height as f64 - padding) - y_norm * draw_h;
        (px, py)
    };

    let line_radius = (options.line_thickness * options.scale / 2.0).max(0.5) as f64;

    for (d_idx, (_, pts)) in datasets.iter().enumerate() {
        if pts.is_empty() {
            continue;
        }
        let color = EXPORT_PALETTE[d_idx % EXPORT_PALETTE.len()];

        let mut prev_pixel: Option<(f64, f64)> = None;
        for &p in pts {
            if !p[0].is_finite() || !p[1].is_finite() {
                prev_pixel = None;
                continue;
            }
            let curr = to_pixel(p);
            if let Some(prev) = prev_pixel {
                draw_segment(&mut pixels, width, height, prev, curr, line_radius, color);
            } else {
                draw_dot(&mut pixels, width, height, curr, line_radius, color);
            }
            prev_pixel = Some(curr);
        }
    }

    Ok(egui::ColorImage::new([width, height], pixels))
}

/// Render plot datasets into a high-resolution PNG image byte buffer using offscreen rasterization.
pub fn export_custom_png(
    datasets: &[(String, Vec<[f64; 2]>)],
    options: &ExportOptions,
) -> Result<Vec<u8>, String> {
    let image = export_custom_image(datasets, options)?;
    color32_image_to_png(&image)
}

fn blend_over(dst: egui::Color32, src: egui::Color32, alpha_factor: f64) -> egui::Color32 {
    let src_a = (src.a() as f64 / 255.0) * alpha_factor.clamp(0.0, 1.0);
    if src_a <= 0.0 {
        return dst;
    }
    let dst_a = dst.a() as f64 / 255.0;
    let out_a = src_a + dst_a * (1.0 - src_a);
    if out_a <= 0.0 {
        return egui::Color32::TRANSPARENT;
    }

    let blend_c = |s_c: u8, d_c: u8| -> u8 {
        let s = s_c as f64 / 255.0;
        let d = d_c as f64 / 255.0;
        let out = (s * src_a + d * dst_a * (1.0 - src_a)) / out_a;
        (out * 255.0).round().clamp(0.0, 255.0) as u8
    };

    egui::Color32::from_rgba_unmultiplied(
        blend_c(src.r(), dst.r()),
        blend_c(src.g(), dst.g()),
        blend_c(src.b(), dst.b()),
        (out_a * 255.0).round().clamp(0.0, 255.0) as u8,
    )
}

fn draw_segment(
    pixels: &mut [egui::Color32],
    w: usize,
    h: usize,
    p0: (f64, f64),
    p1: (f64, f64),
    radius: f64,
    color: egui::Color32,
) {
    let min_x =
        ((p0.0.min(p1.0) - radius - 1.0).floor() as isize).clamp(0, w as isize - 1) as usize;
    let max_x = ((p0.0.max(p1.0) + radius + 1.0).ceil() as isize).clamp(0, w as isize - 1) as usize;
    let min_y =
        ((p0.1.min(p1.1) - radius - 1.0).floor() as isize).clamp(0, h as isize - 1) as usize;
    let max_y = ((p0.1.max(p1.1) + radius + 1.0).ceil() as isize).clamp(0, h as isize - 1) as usize;

    let dx = p1.0 - p0.0;
    let dy = p1.1 - p0.1;
    let len_sq = dx * dx + dy * dy;

    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let px = x as f64 + 0.5;
            let py = y as f64 + 0.5;

            let dist = if len_sq == 0.0 {
                let d2x = px - p0.0;
                let d2y = py - p0.1;
                (d2x * d2x + d2y * d2y).sqrt()
            } else {
                let t = (((px - p0.0) * dx + (py - p0.1) * dy) / len_sq).clamp(0.0, 1.0);
                let proj_x = p0.0 + t * dx;
                let proj_y = p0.1 + t * dy;
                let d2x = px - proj_x;
                let d2y = py - proj_y;
                (d2x * d2x + d2y * d2y).sqrt()
            };

            let coverage = (radius + 0.5 - dist).clamp(0.0, 1.0);
            if coverage > 0.0 {
                let idx = y * w + x;
                pixels[idx] = blend_over(pixels[idx], color, coverage);
            }
        }
    }
}

fn draw_dot(
    pixels: &mut [egui::Color32],
    w: usize,
    h: usize,
    p: (f64, f64),
    radius: f64,
    color: egui::Color32,
) {
    let min_x = ((p.0 - radius - 1.0).floor() as isize).clamp(0, w as isize - 1) as usize;
    let max_x = ((p.0 + radius + 1.0).ceil() as isize).clamp(0, w as isize - 1) as usize;
    let min_y = ((p.1 - radius - 1.0).floor() as isize).clamp(0, h as isize - 1) as usize;
    let max_y = ((p.1 + radius + 1.0).ceil() as isize).clamp(0, h as isize - 1) as usize;

    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let dx = x as f64 + 0.5 - p.0;
            let dy = y as f64 + 0.5 - p.1;
            let dist = (dx * dx + dy * dy).sqrt();
            let coverage = (radius + 0.5 - dist).clamp(0.0, 1.0);
            if coverage > 0.0 {
                let idx = y * w + x;
                pixels[idx] = blend_over(pixels[idx], color, coverage);
            }
        }
    }
}

/// Copy an `egui::ColorImage` to the system clipboard.
pub fn copy_image_to_clipboard(_image: &egui::ColorImage) -> Result<(), String> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let mut rgba = Vec::with_capacity(_image.pixels.len() * 4);
        for p in &_image.pixels {
            rgba.push(p.r());
            rgba.push(p.g());
            rgba.push(p.b());
            rgba.push(p.a());
        }
        let img_data = arboard::ImageData {
            width: _image.width(),
            height: _image.height(),
            bytes: std::borrow::Cow::Owned(rgba),
        };
        let mut clipboard = arboard::Clipboard::new().map_err(|e| e.to_string())?;
        clipboard.set_image(img_data).map_err(|e| e.to_string())?;
        Ok(())
    }

    #[cfg(target_arch = "wasm32")]
    {
        Err("Clipboard image copy is unsupported on WASM".to_string())
    }
}

/// Convert an `egui::ColorImage` into PNG encoded bytes.
pub fn color32_image_to_png(image: &egui::ColorImage) -> Result<Vec<u8>, String> {
    let width = image.width() as u32;
    let height = image.height() as u32;
    let mut bytes = Vec::new();
    let mut encoder = png::Encoder::new(&mut bytes, width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().map_err(|e| e.to_string())?;

    let mut raw_rgba = Vec::with_capacity(image.pixels.len() * 4);
    for p in &image.pixels {
        raw_rgba.push(p.r());
        raw_rgba.push(p.g());
        raw_rgba.push(p.b());
        raw_rgba.push(p.a());
    }
    writer
        .write_image_data(&raw_rgba)
        .map_err(|e| e.to_string())?;
    drop(writer);
    Ok(bytes)
}

/// Trigger cross-platform file saving dialog or browser download action.
pub fn save_file_dialog(filename: &str, content: &[u8], _filter_name: &str, _filter_ext: &[&str]) {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let builder = rfd::FileDialog::new()
            .set_file_name(filename)
            .add_filter(_filter_name, _filter_ext);
        if let Some(path) = builder.save_file() {
            let _ = std::fs::write(path, content);
        }
    }

    #[cfg(target_arch = "wasm32")]
    {
        use wasm_bindgen::JsCast as _;
        use web_sys::{Blob, BlobPropertyBag, HtmlAnchorElement, Url};

        let mime_type = if filename.ends_with(".csv") {
            "text/csv"
        } else if filename.ends_with(".json") {
            "application/json"
        } else if filename.ends_with(".png") {
            "image/png"
        } else {
            "application/octet-stream"
        };

        if let Some(window) = web_sys::window() {
            if let Some(document) = window.document() {
                let uint8_array = js_sys::Uint8Array::from(content);
                let array = js_sys::Array::new();
                array.push(&uint8_array);
                let mut blob_props = BlobPropertyBag::new();
                blob_props.type_(mime_type);
                if let Ok(blob) = Blob::new_with_u8_array_sequence_and_options(&array, &blob_props)
                {
                    if let Ok(url) = Url::create_object_url_with_blob(&blob) {
                        if let Ok(element) = document.create_element("a") {
                            if let Ok(anchor) = element.dyn_into::<HtmlAnchorElement>() {
                                anchor.set_href(&url);
                                anchor.set_download(filename);
                                anchor.click();
                                let _ = Url::revoke_object_url(&url);
                            }
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_csv() {
        let datasets = vec![
            ("Series 1".to_string(), vec![[0.0, 1.0], [2.0, 3.0]]),
            ("Series 2".to_string(), vec![[4.0, 5.0]]),
        ];
        let csv = format_csv(&datasets);
        assert!(csv.contains("Dataset,X,Y"));
        assert!(csv.contains("Series 1,0,1"));
        assert!(csv.contains("Series 1,2,3"));
        assert!(csv.contains("Series 2,4,5"));
    }

    #[test]
    fn test_format_json() {
        let datasets = vec![("Series 1".to_string(), vec![[0.0, 1.0]])];
        let json = format_json(&datasets);
        assert!(json.contains("\"dataset\": \"Series 1\""));
        assert!(json.contains("\"x\": 0"));
        assert!(json.contains("\"y\": 1"));
    }

    #[test]
    fn test_color32_image_to_png() {
        let image = egui::ColorImage::new([2, 2], vec![egui::Color32::RED; 4]);
        let png_bytes = color32_image_to_png(&image).expect("PNG encoding failed");
        assert!(!png_bytes.is_empty());
        assert_eq!(&png_bytes[1..4], b"PNG");
    }

    #[test]
    fn test_export_custom_png_default() {
        let datasets = vec![
            (
                "Sine Wave".to_string(),
                (0..100)
                    .map(|i| {
                        let x = i as f64 * 0.1;
                        [x, x.sin()]
                    })
                    .collect(),
            ),
            (
                "Cosine Wave".to_string(),
                (0..100)
                    .map(|i| {
                        let x = i as f64 * 0.1;
                        [x, x.cos()]
                    })
                    .collect(),
            ),
        ];
        let options = ExportOptions {
            width: 800,
            height: 450,
            scale: 2.0,
            line_thickness: 3.0,
            ..Default::default()
        };
        let png_bytes = export_custom_png(&datasets, &options).expect("Offscreen export failed");
        assert!(!png_bytes.is_empty());
        assert_eq!(&png_bytes[1..4], b"PNG");
        let _ = std::fs::write("/tmp/high_res_plot.png", &png_bytes);
    }

    #[test]
    fn test_export_custom_png_scaling() {
        let datasets = vec![("Data".to_string(), vec![[0.0, 0.0], [10.0, 10.0]])];
        let options = ExportOptions {
            width: 100,
            height: 100,
            scale: 2.0,
            aspect_ratio: AspectRatioOption::OneOne,
            ..Default::default()
        };
        let png_bytes = export_custom_png(&datasets, &options).expect("Export scaled failed");
        assert!(!png_bytes.is_empty());
        assert_eq!(&png_bytes[1..4], b"PNG");
    }

    #[test]
    fn test_export_custom_png_transparent_bg() {
        let datasets = vec![("Points".to_string(), vec![[1.0, 2.0]])];
        let options = ExportOptions {
            width: 100,
            height: 100,
            background_opacity: 0.0,
            ..Default::default()
        };
        let png_bytes =
            export_custom_png(&datasets, &options).expect("Transparent bg export failed");
        assert!(!png_bytes.is_empty());
        assert_eq!(&png_bytes[1..4], b"PNG");
    }

    #[test]
    fn test_export_custom_png_max_dimension_cap() {
        let datasets = vec![("Large".to_string(), vec![[0.0, 0.0]])];
        let options = ExportOptions {
            width: 5000,
            height: 5000,
            scale: 1.0,
            ..Default::default()
        };
        let result = export_custom_png(&datasets, &options);
        assert!(result.is_err());
        let err_msg = result.unwrap_err();
        assert!(err_msg.contains("exceed maximum limit"));
    }

    #[test]
    fn test_export_custom_png_line_thickness() {
        let datasets = vec![("Thick Line".to_string(), vec![[0.0, 0.0], [1.0, 1.0]])];
        let options = ExportOptions {
            width: 200,
            height: 200,
            line_thickness: 5.0,
            ..Default::default()
        };
        let png_bytes = export_custom_png(&datasets, &options).expect("Thick line export failed");
        assert!(!png_bytes.is_empty());
        assert_eq!(&png_bytes[1..4], b"PNG");
    }
}
