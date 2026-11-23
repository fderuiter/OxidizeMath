//! Cross-platform dataset and canvas snapshot export utilities.

use crate::export_primitives::{
    PrimitiveCanvasLayout, is_valid_coord_pair,
    pixel_ops::{draw_dot, draw_segment},
    project_coord,
    raster::background_pixel,
    resolve_canvas_dimensions,
    svg::{escape_xml, svg_document_header},
};

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

/// Named dataset slice of 2D plot points.
pub type ExportDataset = (String, Vec<[f64; 2]>);

/// Frame snapshot containing multiple named plot datasets.
pub type DatasetFrame = Vec<ExportDataset>;

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
    /// Project X coordinates in base-10 logarithmic space.
    #[cfg_attr(feature = "serde", serde(default))]
    pub log_x: bool,
    /// Project Y coordinates in base-10 logarithmic space.
    #[cfg_attr(feature = "serde", serde(default))]
    pub log_y: bool,
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
            log_x: false,
            log_y: false,
        }
    }
}

#[cfg(feature = "serde")]
const fn default_show_points() -> bool {
    true
}

/// An empirical or external dataset imported into egui_plot.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct ImportedDataset {
    /// Name/label of the dataset (e.g. column header or file name).
    pub name: String,
    /// Numerical [x, y] coordinates.
    pub points: Vec<[f64; 2]>,
    /// Whether the dataset is currently visible in the plot overlay.
    pub visible: bool,
    /// Whether scatter marker points are rendered for this overlay series.
    #[cfg_attr(feature = "serde", serde(default = "default_show_points"))]
    pub show_points: bool,
    /// Whether connected polyline segments are rendered for this overlay series.
    #[cfg_attr(feature = "serde", serde(default))]
    pub show_lines: bool,
}

impl ImportedDataset {
    /// Create a new imported dataset with default visibility (`true`), points (`true`), and lines (`false`).
    pub fn new(name: impl Into<String>, points: Vec<[f64; 2]>) -> Self {
        Self {
            name: name.into(),
            points,
            visible: true,
            show_points: true,
            show_lines: false,
        }
    }
}

pub use crate::csv_import::{
    CsvImportError, CsvImportOutcome, MAX_CSV_IMPORT_BYTES, MAX_CSV_IMPORT_ROWS,
    MAX_CSV_IMPORT_SERIES, parse_csv, parse_csv_bounded, split_line, unescape_formula_label,
};

pub use crate::file_io::{
    drain_import_status, drain_imported_datasets, infer_mime_type, sanitize_download_filename,
    save_file_dialog, strip_utf8_bom, trigger_import_csv_for_plot,
};

#[cfg(target_arch = "wasm32")]
pub use crate::file_io::{
    PickerOutcome, download_browser_file, open_browser_file_picker, request_browser_repaint,
};

/// Helper function to sanitize a CSV field value under RFC 4180 and mitigate formula injection (CWE-1236).
///
/// Prefixes formula trigger characters (`=`, `+`, `-`, `@`, `\t`, `\r`) with a single quote,
/// and wraps fields containing commas, double quotes, formula characters, or line breaks in RFC 4180 double quotes.
pub fn escape_csv_field(field: &str) -> String {
    let starts_with_formula = field.starts_with(['=', '+', '-', '@', '\t', '\r'])
        || (field.starts_with(char::is_whitespace)
            && field.trim_start().starts_with(['=', '+', '-', '@']));
    let has_formula_char = field.contains(['=', '+', '-', '@']);

    let content = if starts_with_formula {
        format!("'{field}")
    } else {
        field.to_string()
    };

    let needs_quotes = starts_with_formula
        || has_formula_char
        || field.contains(',')
        || field.contains('"')
        || field.contains('\n')
        || field.contains('\r');

    if needs_quotes {
        let escaped = content.replace('"', "\"\"");
        format!("\"{escaped}\"")
    } else {
        content
    }
}

/// Helper function to escape special JSON characters under RFC 8259.
pub fn escape_json_string(input: &str) -> String {
    let mut escaped = String::with_capacity(input.len());
    for c in input.chars() {
        match c {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\x08' => escaped.push_str("\\b"),
            '\x0C' => escaped.push_str("\\f"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                use std::fmt::Write;
                let _ = write!(escaped, "\\u{:04x}", c as u32);
            }
            c => escaped.push(c),
        }
    }
    escaped
}

/// Format dataset points to CSV string.
pub fn format_csv(datasets: &[(String, Vec<[f64; 2]>)]) -> String {
    let mut csv = String::from("Dataset,X,Y\n");
    for (name, pts) in datasets {
        let escaped_name = escape_csv_field(name);
        for p in pts {
            csv.push_str(&format!("{escaped_name},{},{}\n", p[0], p[1]));
        }
    }
    csv
}

/// Format dataset points to JSON string.
pub fn format_json(datasets: &[(String, Vec<[f64; 2]>)]) -> String {
    let mut json = String::from("[\n");
    let mut first = true;
    for (name, pts) in datasets {
        let escaped_name = escape_json_string(name);
        for p in pts {
            if !first {
                json.push_str(",\n");
            }
            first = false;
            json.push_str(&format!(
                "  {{\"dataset\": \"{}\", \"x\": {}, \"y\": {}}}",
                escaped_name, p[0], p[1]
            ));
        }
    }
    json.push_str("\n]");
    json
}

/// Format 3D dataset world points `(X, Y, Z)` to CSV string.
pub fn format_csv_3d(datasets: &[(String, Vec<[f64; 3]>)]) -> String {
    let mut csv = String::from("Dataset,X,Y,Z\n");
    for (name, pts) in datasets {
        let escaped_name = escape_csv_field(name);
        for p in pts {
            csv.push_str(&format!("{escaped_name},{},{},{}\n", p[0], p[1], p[2]));
        }
    }
    csv
}

/// Format 3D dataset world points `(x, y, z)` to JSON string.
pub fn format_json_3d(datasets: &[(String, Vec<[f64; 3]>)]) -> String {
    let mut json = String::from("[\n");
    let mut first = true;
    for (name, pts) in datasets {
        let escaped_name = escape_json_string(name);
        for p in pts {
            if !first {
                json.push_str(",\n");
            }
            first = false;
            json.push_str(&format!(
                "  {{\"dataset\": \"{}\", \"x\": {}, \"y\": {}, \"z\": {}}}",
                escaped_name, p[0], p[1], p[2]
            ));
        }
    }
    json.push_str("\n]");
    json
}

fn export_csv_3d_filename(datasets: &[(String, Vec<[f64; 3]>)], filename: &str) {
    let csv = format_csv_3d(datasets);
    save_file_dialog(filename, csv.as_bytes(), "CSV File", &["csv"]);
}

/// Export 3D dataset points to CSV and initiate download or file save dialog.
pub fn export_csv_3d(datasets: &[(String, Vec<[f64; 3]>)]) {
    export_csv_3d_filename(datasets, "export.csv");
}

fn export_json_3d_filename(datasets: &[(String, Vec<[f64; 3]>)], filename: &str) {
    let json = format_json_3d(datasets);
    save_file_dialog(filename, json.as_bytes(), "JSON File", &["json"]);
}

/// Export 3D dataset points to JSON and initiate download or file save dialog.
pub fn export_json_3d(datasets: &[(String, Vec<[f64; 3]>)]) {
    export_json_3d_filename(datasets, "export.json");
}

fn export_csv_filename(datasets: &[(String, Vec<[f64; 2]>)], filename: &str) {
    let csv = format_csv(datasets);
    save_file_dialog(filename, csv.as_bytes(), "CSV File", &["csv"]);
}

/// Export dataset points to CSV string and initiate download or file save dialog.
pub fn export_csv(datasets: &[(String, Vec<[f64; 2]>)]) {
    export_csv_filename(datasets, "export.csv");
}

fn export_json_filename(datasets: &[(String, Vec<[f64; 2]>)], filename: &str) {
    let json = format_json(datasets);
    save_file_dialog(filename, json.as_bytes(), "JSON File", &["json"]);
}

/// Export dataset points to JSON string and initiate download or file save dialog.
pub fn export_json(datasets: &[(String, Vec<[f64; 2]>)]) {
    export_json_filename(datasets, "export.json");
}

fn resolve_custom_log_axis(
    bound_min: f64,
    bound_max: f64,
    data_min: f64,
    data_max: f64,
    is_log: bool,
) -> (f64, f64) {
    let b0 = if is_log && (!bound_min.is_finite() || bound_min <= 0.0) {
        if data_min.is_finite() { data_min } else { 1e-6 }
    } else {
        bound_min
    };
    let b1 = if is_log && (!bound_max.is_finite() || bound_max <= b0) {
        if data_max > b0 { data_max } else { b0 * 10.0 }
    } else {
        bound_max
    };
    (b0, b1)
}

fn project_axis_bounds(raw_min: f64, raw_max: f64, is_log: bool) -> (f64, f64) {
    if raw_min.is_finite() && raw_max.is_finite() {
        (
            project_coord(raw_min, is_log),
            project_coord(raw_max, is_log),
        )
    } else if is_log {
        (-6.0, 0.0)
    } else {
        (-1.0, 1.0)
    }
}

fn compute_export_bounds(
    datasets: &[(String, Vec<[f64; 2]>)],
    options: &ExportOptions,
    custom_bounds: Option<[f64; 4]>,
) -> [f64; 4] {
    let mut data_min_x = f64::INFINITY;
    let mut data_max_x = -f64::INFINITY;
    let mut data_min_y = f64::INFINITY;
    let mut data_max_y = -f64::INFINITY;

    for (_, pts) in datasets {
        for &p in pts {
            if is_valid_coord_pair(p, options.log_x, options.log_y) {
                data_min_x = data_min_x.min(p[0]);
                data_max_x = data_max_x.max(p[0]);
                data_min_y = data_min_y.min(p[1]);
                data_max_y = data_max_y.max(p[1]);
            }
        }
    }

    let (raw_min_x, raw_max_x, raw_min_y, raw_max_y) = if let Some(b) = custom_bounds {
        let (bx0, bx1) = resolve_custom_log_axis(b[0], b[1], data_min_x, data_max_x, options.log_x);
        let (by0, by1) = resolve_custom_log_axis(b[2], b[3], data_min_y, data_max_y, options.log_y);
        (bx0, bx1, by0, by1)
    } else {
        (data_min_x, data_max_x, data_min_y, data_max_y)
    };

    let (min_x, max_x) = project_axis_bounds(raw_min_x, raw_max_x, options.log_x);
    let (min_y, max_y) = project_axis_bounds(raw_min_y, raw_max_y, options.log_y);
    [min_x, max_x, min_y, max_y]
}

/// Render plot datasets into vector SVG XML string.
pub fn export_custom_svg(
    datasets: &[(String, Vec<[f64; 2]>)],
    options: &ExportOptions,
) -> Result<String, String> {
    let (width, height) = resolve_canvas_dimensions(options)?;
    let projected = compute_export_bounds(datasets, options, None);
    let layout = PrimitiveCanvasLayout::from_projected_bounds(width, height, projected, options);

    let stroke_width = (options.line_thickness * options.scale).max(1.0);
    let marker_radius = (stroke_width / 2.0).max(1.5);

    let mut svg = svg_document_header(width, height, options);

    for (d_idx, (name, pts)) in datasets.iter().enumerate() {
        if pts.is_empty() {
            continue;
        }
        let color = EXPORT_PALETTE[d_idx % EXPORT_PALETTE.len()];
        let hex_color = format!("#{:02x}{:02x}{:02x}", color.r(), color.g(), color.b());
        let escaped_name = escape_xml(name);

        svg.push_str(&format!(
            "  <g id=\"dataset-{}\" data-name=\"{}\">\n",
            d_idx, escaped_name
        ));

        let mut current_segment: Vec<(f64, f64)> = Vec::new();
        let flush_segment = |segment: &mut Vec<(f64, f64)>, svg: &mut String| {
            if segment.is_empty() {
                return;
            }
            if segment.len() == 1 {
                let (px, py) = segment[0];
                svg.push_str(&format!(
                    "    <circle cx=\"{:.2}\" cy=\"{:.2}\" r=\"{:.2}\" fill=\"{}\" />\n",
                    px, py, marker_radius, hex_color
                ));
            } else {
                let points_str: Vec<String> = segment
                    .iter()
                    .map(|(px, py)| format!("{:.2},{:.2}", px, py))
                    .collect();
                svg.push_str(&format!(
                    "    <polyline points=\"{}\" fill=\"none\" stroke=\"{}\" stroke-width=\"{:.2}\" stroke-linecap=\"round\" stroke-linejoin=\"round\" />\n",
                    points_str.join(" "),
                    hex_color,
                    stroke_width
                ));
                for (px, py) in segment.iter() {
                    svg.push_str(&format!(
                        "    <circle cx=\"{:.2}\" cy=\"{:.2}\" r=\"{:.2}\" fill=\"{}\" />\n",
                        px, py, marker_radius, hex_color
                    ));
                }
            }
            segment.clear();
        };

        for &p in pts {
            if layout.is_valid_point(p) {
                current_segment.push(layout.to_pixel(p));
            } else {
                flush_segment(&mut current_segment, &mut svg);
            }
        }
        flush_segment(&mut current_segment, &mut svg);

        svg.push_str("  </g>\n");
    }

    svg.push_str("</svg>\n");

    Ok(svg)
}

/// Render plot datasets into an `egui::ColorImage` buffer using offscreen rasterization.
pub fn export_custom_image(
    datasets: &[(String, Vec<[f64; 2]>)],
    options: &ExportOptions,
) -> Result<egui::ColorImage, String> {
    export_custom_image_with_bounds(datasets, options, None)
}

/// Render plot datasets into an `egui::ColorImage` buffer with custom bounding box override.
pub fn export_custom_image_with_bounds(
    datasets: &[(String, Vec<[f64; 2]>)],
    options: &ExportOptions,
    custom_bounds: Option<[f64; 4]>,
) -> Result<egui::ColorImage, String> {
    let (width, height) = resolve_canvas_dimensions(options)?;
    let mut pixels = vec![background_pixel(options); width * height];
    let projected = compute_export_bounds(datasets, options, custom_bounds);
    let layout = PrimitiveCanvasLayout::from_projected_bounds(width, height, projected, options);
    let line_radius = (options.line_thickness * options.scale / 2.0).max(0.5) as f64;

    for (d_idx, (_, pts)) in datasets.iter().enumerate() {
        if pts.is_empty() {
            continue;
        }
        let color = EXPORT_PALETTE[d_idx % EXPORT_PALETTE.len()];

        let mut prev_pixel: Option<(f64, f64)> = None;
        for &p in pts {
            if !layout.is_valid_point(p) {
                prev_pixel = None;
                continue;
            }
            let curr = layout.to_pixel(p);
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

/// Convert a sequence of `egui::ColorImage` frames into an animated GIF byte stream.
pub fn color32_images_to_gif(images: &[egui::ColorImage], fps: u32) -> Result<Vec<u8>, String> {
    if images.is_empty() {
        return Err("No frames provided for GIF encoding".to_string());
    }

    let clamped_fps = fps.clamp(1, 60);
    let delay_ms = (1000 / clamped_fps).max(1);
    let delay =
        image::Delay::from_saturating_duration(std::time::Duration::from_millis(delay_ms as u64));

    let mut gif_bytes = Vec::new();
    {
        let mut encoder = image::codecs::gif::GifEncoder::new(&mut gif_bytes);
        encoder
            .set_repeat(image::codecs::gif::Repeat::Infinite)
            .map_err(|e| format!("Failed to set GIF repeat mode: {e}"))?;

        for img in images {
            let width = img.width() as u32;
            let height = img.height() as u32;
            if width == 0 || height == 0 {
                continue;
            }

            let rgba_buf = color_image_to_rgba_bytes(img);

            let rgba_img = image::RgbaImage::from_raw(width, height, rgba_buf)
                .ok_or_else(|| "Failed to construct RGBA buffer for GIF frame".to_string())?;

            let frame = image::Frame::from_parts(rgba_img, 0, 0, delay);
            encoder
                .encode_frame(frame)
                .map_err(|e| format!("Failed to encode GIF frame: {e}"))?;
        }
    }

    if gif_bytes.is_empty() {
        return Err("GIF encoding produced empty byte stream".to_string());
    }

    Ok(gif_bytes)
}

/// Sample progressive frame steps across a dataset for time-evolution animations.
pub fn sample_progressive_frames(
    datasets: &[ExportDataset],
    num_frames: usize,
) -> Vec<DatasetFrame> {
    let count = num_frames.max(1);
    let mut frames = Vec::with_capacity(count);

    for frame_idx in 1..=count {
        let mut frame_datasets = Vec::with_capacity(datasets.len());
        for (name, pts) in datasets {
            if pts.is_empty() {
                frame_datasets.push((name.clone(), Vec::new()));
            } else {
                let keep_count = ((pts.len() * frame_idx) / count).max(1).min(pts.len());
                frame_datasets.push((name.clone(), pts[..keep_count].to_vec()));
            }
        }
        frames.push(frame_datasets);
    }

    frames
}

/// Render a sequence of plot frame datasets into an animated GIF byte stream.
pub fn export_custom_gif(
    frames: &[DatasetFrame],
    options: &ExportOptions,
    fps: u32,
) -> Result<Vec<u8>, String> {
    if frames.is_empty() {
        return Err("No dataset frames provided for GIF export".to_string());
    }

    let mut min_x = f64::INFINITY;
    let mut max_x = -f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = -f64::INFINITY;

    for frame in frames {
        for (_, pts) in frame {
            for &p in pts {
                if is_valid_coord_pair(p, options.log_x, options.log_y) {
                    min_x = min_x.min(p[0]);
                    max_x = max_x.max(p[0]);
                    min_y = min_y.min(p[1]);
                    max_y = max_y.max(p[1]);
                }
            }
        }
    }

    let custom_bounds = if min_x < max_x && min_y < max_y {
        Some([min_x, max_x, min_y, max_y])
    } else {
        None
    };

    let mut images = Vec::with_capacity(frames.len());
    for frame in frames {
        let image = export_custom_image_with_bounds(frame, options, custom_bounds)?;
        images.push(image);
    }

    color32_images_to_gif(&images, fps)
}

/// Render plot datasets into a high-resolution PNG image byte buffer using offscreen rasterization.
pub fn export_custom_png(
    datasets: &[(String, Vec<[f64; 2]>)],
    options: &ExportOptions,
) -> Result<Vec<u8>, String> {
    let image = export_custom_image(datasets, options)?;
    color32_image_to_png(&image)
}

fn color_image_to_rgba_bytes(image: &egui::ColorImage) -> Vec<u8> {
    let mut rgba = Vec::with_capacity(image.pixels.len() * 4);
    for p in &image.pixels {
        rgba.extend_from_slice(&[p.r(), p.g(), p.b(), p.a()]);
    }
    rgba
}

#[cfg(not(target_arch = "wasm32"))]
static NATIVE_CLIPBOARD: std::sync::Mutex<Option<arboard::Clipboard>> = std::sync::Mutex::new(None);

#[cfg(not(target_arch = "wasm32"))]
fn should_retry_native_clipboard() -> bool {
    cfg!(any(target_os = "macos", target_os = "windows"))
        || std::env::var_os("DISPLAY").is_some()
        || std::env::var_os("WAYLAND_DISPLAY").is_some()
}

#[cfg(not(target_arch = "wasm32"))]
fn with_native_clipboard<F>(mut op: F) -> Result<(), String>
where
    F: FnMut(&mut arboard::Clipboard) -> Result<(), arboard::Error>,
{
    let max_attempts = if should_retry_native_clipboard() {
        3
    } else {
        1
    };
    let mut slot = NATIVE_CLIPBOARD
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let mut last_err = None;
    for attempt in 0..max_attempts {
        if slot.is_none() {
            match arboard::Clipboard::new() {
                Ok(cb) => *slot = Some(cb),
                Err(err) => {
                    last_err = Some(err.to_string());
                    if attempt + 1 < max_attempts {
                        std::thread::sleep(std::time::Duration::from_millis(5));
                    }
                    continue;
                }
            }
        }
        if let Some(clipboard) = slot.as_mut() {
            match op(clipboard) {
                Ok(()) => return Ok(()),
                Err(err) => {
                    last_err = Some(err.to_string());
                    *slot = None;
                    if attempt + 1 < max_attempts {
                        std::thread::sleep(std::time::Duration::from_millis(5));
                    }
                }
            }
        }
    }
    Err(last_err.unwrap_or_else(|| "Clipboard write failed".to_string()))
}

/// Copy an `egui::ColorImage` to the system clipboard.
pub fn copy_image_to_clipboard(image: &egui::ColorImage) -> Result<(), String> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let rgba = color_image_to_rgba_bytes(image);
        let width = image.width();
        let height = image.height();
        with_native_clipboard(|clipboard| {
            let img_data = arboard::ImageData {
                width,
                height,
                bytes: std::borrow::Cow::Borrowed(&rgba),
            };
            clipboard.set_image(img_data)
        })
    }

    #[cfg(target_arch = "wasm32")]
    {
        copy_image_to_clipboard_with_callback(image, |_| {})
    }
}

/// Copy an `egui::ColorImage` to the system clipboard, invoking `on_complete` upon resolution.
pub fn copy_image_to_clipboard_with_callback<F>(
    image: &egui::ColorImage,
    on_complete: F,
) -> Result<(), String>
where
    F: FnOnce(Result<(), String>) + 'static,
{
    #[cfg(not(target_arch = "wasm32"))]
    {
        let res = copy_image_to_clipboard(image);
        on_complete(res.clone());
        res
    }

    #[cfg(target_arch = "wasm32")]
    {
        copy_image_to_clipboard_wasm(image, on_complete)
    }
}

#[cfg(target_arch = "wasm32")]
fn copy_image_to_clipboard_wasm<F>(image: &egui::ColorImage, on_complete: F) -> Result<(), String>
where
    F: FnOnce(Result<(), String>) + 'static,
{
    use web_sys::{Blob, BlobPropertyBag, ClipboardItem};

    let window = web_sys::window().ok_or_else(|| {
        "Clipboard API unavailable (no window object). Please use direct file download (Save PNG)."
            .to_string()
    })?;

    if !window.is_secure_context() {
        return Err(
            "Clipboard API unavailable in insecure context. Please use HTTPS or direct file download (Save PNG)."
                .to_string(),
        );
    }

    let navigator = window.navigator();
    let clipboard = navigator.clipboard();
    if clipboard.is_undefined() || clipboard.is_null() {
        return Err(
            "Clipboard API is not supported in this browser. Please use direct file download (Save PNG)."
                .to_string(),
        );
    }

    let png_bytes = color32_image_to_png(image).map_err(|e| format!("PNG encoding failed: {e}"))?;

    let uint8_array = js_sys::Uint8Array::from(png_bytes.as_slice());
    let array = js_sys::Array::new();
    array.push(&uint8_array);
    let blob_props = BlobPropertyBag::new();
    blob_props.set_type("image/png");
    let blob = Blob::new_with_u8_array_sequence_and_options(&array, &blob_props)
        .map_err(|e| format!("Failed to create PNG blob: {e:?}"))?;

    let item_record = js_sys::Object::new();
    js_sys::Reflect::set(
        &item_record,
        &wasm_bindgen::JsValue::from_str("image/png"),
        &blob,
    )
    .map_err(|e| format!("Failed to set ClipboardItem property: {e:?}"))?;

    let clipboard_item = ClipboardItem::new_with_record_from_str_to_blob_promise(&item_record)
        .map_err(|e| format!("Failed to create ClipboardItem: {e:?}"))?;

    let items = js_sys::Array::new();
    items.push(&clipboard_item);

    let promise = clipboard.write(&items);

    wasm_bindgen_futures::spawn_local(async move {
        let res = match wasm_bindgen_futures::JsFuture::from(promise).await {
            Ok(_) => Ok(()),
            Err(e) => {
                let err_msg = e
                    .as_string()
                    .or_else(|| {
                        js_sys::Reflect::get(&e, &wasm_bindgen::JsValue::from_str("message"))
                            .ok()
                            .and_then(|m| m.as_string())
                    })
                    .unwrap_or_else(|| "Unknown error writing to clipboard".to_string());
                Err(format!("Clipboard write failed: {err_msg}"))
            }
        };
        on_complete(res);
    });

    Ok(())
}

/// Copy UTF-8 text (such as serialized CSV or JSON datasets) to the system clipboard.
pub fn copy_text_to_clipboard(text: &str) -> Result<(), String> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        with_native_clipboard(|clipboard| clipboard.set_text(text.to_owned()))
    }

    #[cfg(target_arch = "wasm32")]
    {
        copy_text_to_clipboard_with_callback(text, |_| {})
    }
}

/// Copy UTF-8 text to the system clipboard, invoking `on_complete` upon resolution.
pub fn copy_text_to_clipboard_with_callback<F>(text: &str, on_complete: F) -> Result<(), String>
where
    F: FnOnce(Result<(), String>) + 'static,
{
    #[cfg(not(target_arch = "wasm32"))]
    {
        let res = copy_text_to_clipboard(text);
        on_complete(res.clone());
        res
    }

    #[cfg(target_arch = "wasm32")]
    {
        copy_text_to_clipboard_wasm(text, on_complete)
    }
}

#[cfg(target_arch = "wasm32")]
fn copy_text_to_clipboard_wasm<F>(text: &str, on_complete: F) -> Result<(), String>
where
    F: FnOnce(Result<(), String>) + 'static,
{
    let window = web_sys::window().ok_or_else(|| {
        "Clipboard API unavailable (no window object). Please use direct file download (Export CSV)."
            .to_string()
    })?;

    if !window.is_secure_context() {
        return Err(
            "Clipboard API unavailable in insecure context. Please use HTTPS or direct file download (Export CSV)."
                .to_string(),
        );
    }

    let navigator = window.navigator();
    let clipboard = navigator.clipboard();
    if clipboard.is_undefined() || clipboard.is_null() {
        return Err(
            "Clipboard API is not supported in this browser. Please use direct file download (Export CSV)."
                .to_string(),
        );
    }

    let promise = clipboard.write_text(text);

    wasm_bindgen_futures::spawn_local(async move {
        let res = match wasm_bindgen_futures::JsFuture::from(promise).await {
            Ok(_) => Ok(()),
            Err(e) => {
                let err_msg = e
                    .as_string()
                    .or_else(|| {
                        js_sys::Reflect::get(&e, &wasm_bindgen::JsValue::from_str("message"))
                            .ok()
                            .and_then(|m| m.as_string())
                    })
                    .unwrap_or_else(|| "Unknown error writing text to clipboard".to_string());
                Err(format!(
                    "Clipboard write failed: {err_msg}. Please use direct file download (Export CSV)."
                ))
            }
        };
        on_complete(res);
    });

    Ok(())
}

#[cfg(test)]
static TEST_CLIPBOARD_READ_OVERRIDE: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);

#[cfg(test)]
pub(crate) fn set_test_clipboard_read_override(text: Option<String>) {
    *TEST_CLIPBOARD_READ_OVERRIDE
        .lock()
        .unwrap_or_else(|p| p.into_inner()) = text;
}

/// Read UTF-8 text from the system clipboard, enforcing [`MAX_CSV_IMPORT_BYTES`] and invoking `on_complete`.
pub fn read_text_from_clipboard_with_callback<F>(on_complete: F) -> Result<(), String>
where
    F: FnOnce(Result<String, String>) + 'static,
{
    #[cfg(test)]
    if let Some(mock_text) = TEST_CLIPBOARD_READ_OVERRIDE
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .clone()
    {
        if mock_text.len() > MAX_CSV_IMPORT_BYTES {
            let err = format!(
                "Clipboard payload ({:.2} MiB) exceeds maximum allowed size of 2.0 MiB.",
                mock_text.len() as f64 / (1024.0 * 1024.0)
            );
            on_complete(Err(err.clone()));
            return Err(err);
        }
        on_complete(Ok(mock_text));
        return Ok(());
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        let mut out_text = String::new();
        let res = with_native_clipboard(|clipboard| {
            out_text = clipboard.get_text()?;
            Ok(())
        });
        match res {
            Ok(()) if out_text.len() > MAX_CSV_IMPORT_BYTES => {
                let err = format!(
                    "Clipboard payload ({:.2} MiB) exceeds maximum allowed size of 2.0 MiB.",
                    out_text.len() as f64 / (1024.0 * 1024.0)
                );
                on_complete(Err(err.clone()));
                Err(err)
            }
            Ok(()) => {
                on_complete(Ok(out_text));
                Ok(())
            }
            Err(err) => {
                let msg = format!("Clipboard read failed: {err}");
                on_complete(Err(msg.clone()));
                Err(msg)
            }
        }
    }

    #[cfg(target_arch = "wasm32")]
    {
        read_text_from_clipboard_wasm(on_complete)
    }
}

#[cfg(target_arch = "wasm32")]
fn read_text_from_clipboard_wasm<F>(on_complete: F) -> Result<(), String>
where
    F: FnOnce(Result<String, String>) + 'static,
{
    let window = web_sys::window().ok_or_else(|| {
        "Clipboard API unavailable (no window object). Please press Ctrl+V or use Import CSV."
            .to_string()
    })?;
    if !window.is_secure_context() {
        return Err(
            "Clipboard API unavailable in insecure context. Please press Ctrl+V or use Import CSV."
                .to_string(),
        );
    }
    let clipboard = window.navigator().clipboard();
    if clipboard.is_undefined() || clipboard.is_null() {
        return Err(
            "Clipboard API is not supported in this browser. Please press Ctrl+V or use Import CSV."
                .to_string(),
        );
    }
    let promise = clipboard.read_text();
    wasm_bindgen_futures::spawn_local(async move {
        let res = match wasm_bindgen_futures::JsFuture::from(promise).await {
            Ok(js_val) => match js_val.as_string() {
                Some(s) if s.len() > MAX_CSV_IMPORT_BYTES => Err(format!(
                    "Clipboard payload ({:.2} MiB) exceeds maximum allowed size of 2.0 MiB.",
                    s.len() as f64 / (1024.0 * 1024.0)
                )),
                Some(s) => Ok(s),
                None => Err("Clipboard content did not decode as UTF-8 text.".to_string()),
            },
            Err(e) => {
                let reason = e
                    .as_string()
                    .unwrap_or_else(|| "Permission denied or clipboard unavailable".to_string());
                Err(format!(
                    "Clipboard read failed: {reason}. Press Ctrl+V or use Import CSV."
                ))
            }
        };
        on_complete(res);
    });
    Ok(())
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

    let raw_rgba = color_image_to_rgba_bytes(image);
    writer
        .write_image_data(&raw_rgba)
        .map_err(|e| e.to_string())?;
    drop(writer);
    Ok(bytes)
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
    fn test_escape_csv_field_neutralization_and_quoting() {
        // Formula characters at start are prefixed with single quote and wrapped in double quotes
        assert_eq!(escape_csv_field("=SUM(A1)"), "\"'=SUM(A1)\"");
        assert_eq!(escape_csv_field("+123"), "\"'+123\"");
        assert_eq!(escape_csv_field("-456"), "\"'-456\"");
        assert_eq!(escape_csv_field("@cmd"), "\"'@cmd\"");
        assert_eq!(escape_csv_field("\t=SUM"), "\"'\t=SUM\"");
        assert_eq!(escape_csv_field("  =SUM"), "\"'  =SUM\"");

        // Names containing only formula prefixes
        assert_eq!(escape_csv_field("="), "\"'=\"");
        assert_eq!(escape_csv_field("+"), "\"'+\"");
        assert_eq!(escape_csv_field("-"), "\"\'-\"");
        assert_eq!(escape_csv_field("@"), "\"'@\"");
        assert_eq!(escape_csv_field("==="), "\"'===\"");
        assert_eq!(escape_csv_field("+-@="), "\"'+-@=\"");

        // Commas, quotes, and repeated quotes
        assert_eq!(escape_csv_field("Series, 1"), "\"Series, 1\"");
        assert_eq!(escape_csv_field("Series \"A\""), "\"Series \"\"A\"\"\"");
        assert_eq!(
            escape_csv_field("\"\"\"quoted\"\"\""),
            "\"\"\"\"\"\"\"quoted\"\"\"\"\"\"\""
        );

        // Strings with embedded line breaks
        assert_eq!(escape_csv_field("Line 1\nLine 2"), "\"Line 1\nLine 2\"");
        assert_eq!(escape_csv_field("Line 1\r\nLine 2"), "\"Line 1\r\nLine 2\"");

        // Empty string and unmodified alphanumeric names
        assert_eq!(escape_csv_field(""), "");
        assert_eq!(escape_csv_field("Series 1"), "Series 1");
        assert_eq!(escape_csv_field("Test Series"), "Test Series");
        assert_eq!(escape_csv_field("Signal A"), "Signal A");
    }

    #[test]
    fn test_escape_json_string_escaping() {
        assert_eq!(escape_json_string("Series \"1\""), "Series \\\"1\\\"");
        assert_eq!(escape_json_string("C:\\path\\file"), "C:\\\\path\\\\file");
        assert_eq!(
            escape_json_string("Line 1\nLine 2\tTab"),
            "Line 1\\nLine 2\\tTab"
        );
        assert_eq!(
            escape_json_string("\x00\x08\x0C\x1F"),
            "\\u0000\\b\\f\\u001f"
        );
        assert_eq!(escape_json_string(""), "");
        assert_eq!(escape_json_string("Series 1"), "Series 1");
    }

    #[test]
    fn test_format_csv_with_sanitization() {
        let datasets = vec![
            ("=cmd|' /C calc'!A0".to_string(), vec![[0.0, 1.0]]),
            ("Series, 1".to_string(), vec![[2.0, 3.0]]),
            ("Normal Series".to_string(), vec![[4.0, 5.0]]),
        ];
        let csv = format_csv(&datasets);
        assert!(csv.starts_with("Dataset,X,Y\n"));
        assert!(csv.contains("\"'=cmd|' /C calc'!A0\",0,1\n"));
        assert!(csv.contains("\"Series, 1\",2,3\n"));
        assert!(csv.contains("Normal Series,4,5\n"));
    }

    #[test]
    fn test_format_json_with_escaping() {
        let datasets = vec![(
            "Series \"Alpha\"\nSpecial\tData".to_string(),
            vec![[1.0, 2.0]],
        )];
        let json = format_json(&datasets);
        assert!(json.starts_with("[\n"));
        assert!(json.contains("\"dataset\": \"Series \\\"Alpha\\\"\\nSpecial\\tData\""));
        assert!(json.contains("\"x\": 1, \"y\": 2"));
    }

    #[test]
    fn test_parse_csv_quoted_tokens() {
        let content = "Dataset,X,Y\n\"Series, A\",0.0,1.0\n\"Series \"\"Special\"\" B\",1.0,2.0\n\"'=SUM(A1)\",2.0,3.0\n";
        let datasets = parse_csv(content);
        assert_eq!(datasets.len(), 3);

        let series_a = datasets.iter().find(|d| d.name == "Series, A").unwrap();
        assert_eq!(series_a.points, vec![[0.0, 1.0]]);

        let series_b = datasets
            .iter()
            .find(|d| d.name == "Series \"Special\" B")
            .unwrap();
        assert_eq!(series_b.points, vec![[1.0, 2.0]]);

        let series_formula = datasets.iter().find(|d| d.name == "=SUM(A1)").unwrap();
        assert_eq!(series_formula.points, vec![[2.0, 3.0]]);
    }

    #[test]
    fn test_split_line_comprehensive() {
        assert_eq!(split_line("", ','), vec![""]);
        assert_eq!(split_line("A,B,C", ','), vec!["A", "B", "C"]);
        assert_eq!(split_line("\"A,B\",C", ','), vec!["A,B", "C"]);
        assert_eq!(split_line("\"A\"\"B\",C", ','), vec!["A\"B", "C"]);
        assert_eq!(split_line("  \"A, B\"  ,  42  ", ','), vec!["A, B", "42"]);
        assert_eq!(split_line("\"A\tB\"\tC", '\t'), vec!["A\tB", "C"]);
        assert_eq!(split_line("A,B,", ','), vec!["A", "B", ""]);
        assert_eq!(split_line(",,", ','), vec!["", "", ""]);
        assert_eq!(split_line("\"unclosed,1,2", ','), vec!["unclosed,1,2"]);
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

    #[test]
    fn test_parse_csv_standard_export() {
        let content = "Dataset,X,Y\nSeries A,0.0,1.0\nSeries A,1.0,2.0\nSeries B,2.0,3.0\n";
        let datasets = parse_csv(content);
        assert_eq!(datasets.len(), 2);

        let series_a = datasets.iter().find(|d| d.name == "Series A").unwrap();
        assert_eq!(series_a.points, vec![[0.0, 1.0], [1.0, 2.0]]);

        let series_b = datasets.iter().find(|d| d.name == "Series B").unwrap();
        assert_eq!(series_b.points, vec![[2.0, 3.0]]);
    }

    #[test]
    fn test_parse_csv_two_column_headerless() {
        let content = "0.0, 10.0\n1.0, 20.0\n2.0, 30.0\n";
        let datasets = parse_csv(content);
        assert_eq!(datasets.len(), 1);
        assert_eq!(datasets[0].name, "Imported Dataset");
        assert_eq!(
            datasets[0].points,
            vec![[0.0, 10.0], [1.0, 20.0], [2.0, 30.0]]
        );
    }

    #[test]
    fn test_parse_csv_tab_separated() {
        let content = "Time\tTemperature\n0.0\t15.5\n1.0\t18.2\n2.0\t21.0\n";
        let datasets = parse_csv(content);
        assert_eq!(datasets.len(), 1);
        assert_eq!(datasets[0].name, "Temperature");
        assert_eq!(
            datasets[0].points,
            vec![[0.0, 15.5], [1.0, 18.2], [2.0, 21.0]]
        );
    }

    #[test]
    fn test_parse_csv_missing_values() {
        let content = "X,Y\n0.0,1.0\n1.0,\ninvalid,data\n2.0,3.0\n,4.0\n3.0,5.0\n";
        let datasets = parse_csv(content);
        assert_eq!(datasets.len(), 1);
        assert_eq!(datasets[0].points, vec![[0.0, 1.0], [2.0, 3.0], [3.0, 5.0]]);
    }

    #[test]
    fn test_parse_csv_corrupted_text() {
        let content = "Not a CSV file at all\nJust some random text\nWith no numbers anywhere\n";
        let datasets = parse_csv(content);
        assert!(datasets.is_empty());
    }

    #[test]
    fn test_export_custom_svg_structure_and_escaping() {
        let datasets = vec![(
            "Dataset & \"Special <Name>\"".to_string(),
            vec![[0.0, 1.0], [1.0, 2.0], [2.0, 4.0]],
        )];
        let options = ExportOptions::default();
        let svg = export_custom_svg(&datasets, &options).expect("SVG export failed");

        assert!(svg.starts_with("<?xml version=\"1.0\" encoding=\"UTF-8\"?>"));
        assert!(svg.contains("<svg "));
        assert!(svg.contains("xmlns=\"http://www.w3.org/2000/svg\""));
        assert!(svg.contains("<rect "));
        assert!(svg.contains("<polyline points="));
        assert!(svg.contains("<circle cx="));

        // Verify XML character escaping in dataset name attribute
        assert!(svg.contains("data-name=\"Dataset &amp; &quot;Special &lt;Name&gt;&quot;\""));
    }

    #[test]
    fn test_export_custom_svg_stroke_width_options() {
        let datasets = vec![("Test".to_string(), vec![[0.0, 0.0], [1.0, 1.0]])];
        let options = ExportOptions {
            line_thickness: 4.5,
            scale: 2.0,
            ..Default::default()
        };
        let svg = export_custom_svg(&datasets, &options).expect("SVG export failed");
        assert!(svg.contains("stroke-width=\"9.00\""));
    }

    #[test]
    fn test_export_custom_svg_non_finite_splitting() {
        let datasets = vec![(
            "NonFinite".to_string(),
            vec![
                [0.0, 0.0],
                [1.0, 1.0],
                [f64::NAN, 2.0],
                [3.0, f64::INFINITY],
                [4.0, 4.0],
                [5.0, 5.0],
            ],
        )];
        let options = ExportOptions::default();
        let svg = export_custom_svg(&datasets, &options).expect("SVG export failed");
        assert!(svg.contains("<polyline points="));
        // Verify no NaN or inf leaked into SVG string
        assert!(!svg.contains("NaN"));
        assert!(!svg.contains("inf"));
    }

    #[test]
    fn test_export_custom_svg_empty_and_zero_points() {
        let datasets: Vec<(String, Vec<[f64; 2]>)> = vec![
            ("Empty".to_string(), vec![]),
            ("ZeroPoints".to_string(), vec![]),
        ];
        let options = ExportOptions::default();
        let svg = export_custom_svg(&datasets, &options).expect("SVG empty plot failed");
        assert!(svg.contains("<svg "));
        assert!(svg.contains("<rect "));
        assert!(!svg.contains("<polyline"));
    }

    #[test]
    fn test_export_custom_svg_max_dimension_cap() {
        let datasets = vec![("Cap".to_string(), vec![[0.0, 0.0]])];
        let options = ExportOptions {
            width: MAX_EXPORT_DIMENSION + 100,
            ..Default::default()
        };
        let result = export_custom_svg(&datasets, &options);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("exceed maximum limit"));
    }

    #[test]
    fn test_color32_images_to_gif_header_and_delay() {
        let frame1 = egui::ColorImage::new([10, 10], vec![egui::Color32::RED; 100]);
        let frame2 = egui::ColorImage::new([10, 10], vec![egui::Color32::BLUE; 100]);
        let frame3 = egui::ColorImage::new([10, 10], vec![egui::Color32::GREEN; 100]);

        let gif_bytes =
            color32_images_to_gif(&[frame1, frame2, frame3], 15).expect("GIF encoding failed");
        assert!(!gif_bytes.is_empty());
        assert_eq!(&gif_bytes[0..6], b"GIF89a");
    }

    #[test]
    fn test_export_custom_gif_multiframe() {
        let frame1 = vec![("Line".to_string(), vec![[0.0, 0.0], [1.0, 1.0]])];
        let frame2 = vec![("Line".to_string(), vec![[0.0, 0.0], [1.0, 1.0], [2.0, 4.0]])];
        let frame3 = vec![(
            "Line".to_string(),
            vec![[0.0, 0.0], [1.0, 1.0], [2.0, 4.0], [3.0, 9.0]],
        )];

        let options = ExportOptions {
            width: 400,
            height: 300,
            ..Default::default()
        };

        let gif_bytes = export_custom_gif(&[frame1.clone(), frame2, frame3], &options, 10)
            .expect("Multi-frame GIF export failed");
        assert!(!gif_bytes.is_empty());
        assert_eq!(&gif_bytes[0..6], b"GIF89a");

        let _ = std::fs::write("/tmp/simulation_export.gif", &gif_bytes);
        let _ = std::fs::write("/tmp/sample_export.gif", &gif_bytes);

        if let Ok(png_bytes) = export_custom_png(&frame1, &options) {
            let _ = std::fs::write("/tmp/gif_frame_preview.png", &png_bytes);
        }
    }

    #[test]
    fn test_export_custom_gif_empty_handling() {
        let options = ExportOptions::default();
        let result = export_custom_gif(&[], &options, 10);
        assert!(result.is_err());
        let err_msg = result.unwrap_err();
        assert!(
            err_msg.contains("No dataset frames provided")
                || err_msg.contains("No animation frames provided")
        );
    }

    #[test]
    fn test_export_custom_gif_single_frame_fallback() {
        let frame1 = vec![("Single".to_string(), vec![[0.0, 1.0], [1.0, 2.0]])];
        let options = ExportOptions {
            width: 100,
            height: 100,
            ..Default::default()
        };

        let gif_bytes = export_custom_gif(&[frame1], &options, 10)
            .expect("Single-frame fallback GIF export failed");
        assert!(!gif_bytes.is_empty());
        assert_eq!(&gif_bytes[0..6], b"GIF89a");
    }

    #[test]
    fn test_export_custom_gif_max_dimension_cap() {
        let frame1 = vec![("Series".to_string(), vec![[0.0, 0.0]])];
        let options = ExportOptions {
            width: 5000,
            height: 5000,
            scale: 1.0,
            ..Default::default()
        };
        let result = export_custom_gif(&[frame1], &options, 10);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("exceed maximum limit"));
    }

    #[test]
    fn test_sample_progressive_frames() {
        let datasets = vec![(
            "Trace".to_string(),
            vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
        )];

        let frames = sample_progressive_frames(&datasets, 4);
        assert_eq!(frames.len(), 4);
        assert_eq!(frames[0][0].1.len(), 1);
        assert_eq!(frames[1][0].1.len(), 2);
        assert_eq!(frames[2][0].1.len(), 3);
        assert_eq!(frames[3][0].1.len(), 4);
    }

    #[test]
    fn test_copy_image_to_clipboard_callback_invoked() {
        let img = egui::ColorImage::new([10, 10], vec![egui::Color32::RED; 100]);
        let callback_called = std::sync::Arc::new(std::sync::Mutex::new(false));
        let flag = callback_called.clone();
        let _ = copy_image_to_clipboard_with_callback(&img, move |_res| {
            *flag.lock().unwrap() = true;
        });
        #[cfg(not(target_arch = "wasm32"))]
        assert!(*callback_called.lock().unwrap());
    }

    #[test]
    fn test_copy_image_to_clipboard_pipeline_consistency() {
        let img = egui::ColorImage::new([2, 2], vec![egui::Color32::BLUE; 4]);
        let res_direct = copy_image_to_clipboard(&img);
        let callback_res = std::sync::Arc::new(std::sync::Mutex::new(None));
        let res_slot = callback_res.clone();
        let res_cb = copy_image_to_clipboard_with_callback(&img, move |r| {
            *res_slot.lock().unwrap() = Some(r);
        });
        assert_eq!(res_direct.is_ok(), res_cb.is_ok());
        #[cfg(not(target_arch = "wasm32"))]
        assert!(callback_res.lock().unwrap().is_some());
    }

    #[test]
    fn test_copy_text_to_clipboard_callback_invoked() {
        let datasets = vec![("Series 1".to_string(), vec![[0.0, 1.0], [2.0, 3.0]])];
        let csv = format_csv(&datasets);
        let callback_called = std::sync::Arc::new(std::sync::Mutex::new(false));
        let flag = callback_called.clone();
        let _ = copy_text_to_clipboard_with_callback(&csv, move |_res| {
            *flag.lock().unwrap() = true;
        });
        #[cfg(not(target_arch = "wasm32"))]
        assert!(*callback_called.lock().unwrap());
    }

    #[test]
    fn test_copy_text_to_clipboard_pipeline_consistency() {
        let text = "Dataset,X,Y\nTrajectory,0.5,1.25\n";
        let res_direct = copy_text_to_clipboard(text);
        let callback_res = std::sync::Arc::new(std::sync::Mutex::new(None));
        let res_slot = callback_res.clone();
        let res_cb = copy_text_to_clipboard_with_callback(text, move |r| {
            *res_slot.lock().unwrap() = Some(r);
        });
        assert_eq!(res_direct.is_ok(), res_cb.is_ok());
        #[cfg(not(target_arch = "wasm32"))]
        assert!(callback_res.lock().unwrap().is_some());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_concurrent_clipboard_image_and_text_copies_thread_safe() {
        let baseline_ok = copy_text_to_clipboard("probe").is_ok();
        let img = egui::ColorImage::new([4, 4], vec![egui::Color32::GREEN; 16]);
        std::thread::scope(|scope| {
            for worker in 0..8 {
                let img_ref = &img;
                scope.spawn(move || {
                    for iter in 0..6 {
                        let payload = format!("Dataset,X,Y\nWorker{worker},{iter},1.0\n");
                        let text_res = copy_text_to_clipboard(&payload);
                        let text_cb = copy_text_to_clipboard_with_callback(&payload, |_| {});
                        let img_res = copy_image_to_clipboard(img_ref);
                        let img_cb = copy_image_to_clipboard_with_callback(img_ref, |_| {});
                        assert_eq!(text_res.is_ok(), baseline_ok);
                        assert_eq!(text_cb.is_ok(), baseline_ok);
                        assert_eq!(img_res.is_ok(), baseline_ok);
                        assert_eq!(img_cb.is_ok(), baseline_ok);
                    }
                });
            }
        });
    }

    #[test]
    fn test_export_custom_svg_logarithmic_spacing_and_splitting() {
        let datasets = vec![(
            "Decades".to_string(),
            vec![
                [1.0, 1.0],
                [10.0, 100.0],
                [0.0, -5.0],
                [100.0, 10_000.0],
                [1000.0, 1_000_000.0],
            ],
        )];
        let options = ExportOptions {
            width: 800,
            height: 600,
            log_x: true,
            log_y: true,
            ..Default::default()
        };
        let svg = export_custom_svg(&datasets, &options).expect("Log SVG export failed");
        let polylines: Vec<&str> = svg.matches("<polyline ").collect();
        assert_eq!(polylines.len(), 2, "Non-positive point must split polyline");

        let parse_segment = |tag_start: &str| -> Vec<[f64; 2]> {
            let p_attr = tag_start.split("points=\"").nth(1).unwrap();
            let pts_str = p_attr.split('"').next().unwrap();
            pts_str
                .split_whitespace()
                .map(|pair| {
                    let mut it = pair.split(',');
                    [
                        it.next().unwrap().parse::<f64>().unwrap(),
                        it.next().unwrap().parse::<f64>().unwrap(),
                    ]
                })
                .collect()
        };
        let mut chunks = svg.split("<polyline ").skip(1);
        let seg1 = parse_segment(chunks.next().unwrap());
        let seg2 = parse_segment(chunks.next().unwrap());
        let dx1 = seg1[1][0] - seg1[0][0];
        let dx2 = seg2[1][0] - seg2[0][0];
        let dy1 = (seg1[1][1] - seg1[0][1]).abs();
        let dy2 = (seg2[1][1] - seg2[0][1]).abs();
        assert!(
            (dx1 - dx2).abs() < 0.05,
            "Equal X decades in SVG: {dx1} vs {dx2}"
        );
        assert!(
            (dy1 - dy2).abs() < 0.05,
            "Equal Y decades in SVG: {dy1} vs {dy2}"
        );
    }

    #[test]
    fn test_export_custom_png_and_gif_logarithmic() {
        let sample_pts: Vec<[f64; 2]> = (0..=30)
            .map(|i| {
                let exp = -1.0 + (i as f64) * (4.0 / 30.0);
                let x = 10.0_f64.powf(exp);
                let wobble = 1.0 + 0.18 * ((i as f64) * 0.9).sin();
                [x, 2.5 * x.powf(1.6) * wobble]
            })
            .collect();
        let fit = crate::fit::fit_model_with_axes(
            &sample_pts,
            crate::FitType::Linear,
            "PowerLaw",
            true,
            true,
            false,
        )
        .unwrap();
        let fit_pts: Vec<[f64; 2]> = (0..=100)
            .map(|i| {
                let exp = -1.0 + (i as f64) * (4.0 / 100.0);
                let x = 10.0_f64.powf(exp);
                [x, fit.eval(x)]
            })
            .collect();
        let datasets = vec![
            ("Power-Law Measurements".to_string(), sample_pts.clone()),
            (format!("Log-Log Fit ({})", fit.formula), fit_pts),
        ];
        let options = ExportOptions {
            width: 960,
            height: 540,
            line_thickness: 3.0,
            background_color: egui::Color32::from_rgb(18, 22, 30),
            log_x: true,
            log_y: true,
            ..Default::default()
        };
        let png = export_custom_png(&datasets, &options).expect("Log PNG export failed");
        assert_eq!(&png[0..8], b"\x89PNG\r\n\x1a\n");
        let _ = std::fs::write("/tmp/log_space_projection_verification.png", &png);
        if let Ok(svg_log) = export_custom_svg(&datasets, &options) {
            let _ = std::fs::write("/tmp/log_space_projection_log.svg", svg_log);
        }
        let lin_opts = ExportOptions {
            log_x: false,
            log_y: false,
            ..options
        };
        if let Ok(svg_lin) = export_custom_svg(&datasets, &lin_opts) {
            let _ = std::fs::write("/tmp/log_space_projection_linear.svg", svg_lin);
        }

        let frames = sample_progressive_frames(&datasets, 3);
        let gif_opts = ExportOptions {
            width: 240,
            height: 135,
            ..options
        };
        let gif = export_custom_gif(&frames, &gif_opts, 10).expect("Log GIF export failed");
        assert_eq!(&gif[0..6], b"GIF89a");
    }

    #[test]
    fn test_color_image_to_rgba_bytes_packing_and_empty() {
        let empty = egui::ColorImage::new([0, 0], vec![]);
        assert!(color_image_to_rgba_bytes(&empty).is_empty());

        let p0 = egui::Color32::from_rgba_unmultiplied(12, 34, 56, 255);
        let p1 = egui::Color32::from_rgba_unmultiplied(200, 100, 50, 128);
        let p2 = egui::Color32::TRANSPARENT;
        let p3 = egui::Color32::from_rgb(7, 8, 9);
        let img = egui::ColorImage::new([2, 2], vec![p0, p1, p2, p3]);

        let packed = color_image_to_rgba_bytes(&img);
        assert_eq!(
            packed,
            vec![
                p0.r(),
                p0.g(),
                p0.b(),
                p0.a(),
                p1.r(),
                p1.g(),
                p1.b(),
                p1.a(),
                0,
                0,
                0,
                0,
                7,
                8,
                9,
                255,
            ]
        );
    }

    #[test]
    fn test_compute_export_bounds_log_fallbacks_and_empty_defaults() {
        let empty: Vec<ExportDataset> = Vec::new();
        let lin_opts = ExportOptions::default();
        assert_eq!(
            compute_export_bounds(&empty, &lin_opts, None),
            [-1.0, 1.0, -1.0, 1.0]
        );

        let log_opts = ExportOptions {
            log_x: true,
            log_y: true,
            ..ExportOptions::default()
        };
        assert_eq!(
            compute_export_bounds(&empty, &log_opts, None),
            [-6.0, 0.0, -6.0, 0.0]
        );
        assert_eq!(
            compute_export_bounds(&empty, &log_opts, Some([0.0, -5.0, -2.0, 0.0])),
            [-6.0, -5.0, -6.0, -5.0]
        );

        let datasets = vec![(
            "Decades".to_string(),
            vec![[-1.0, 5.0], [10.0, 100.0], [1000.0, 10_000.0]],
        )];
        assert_eq!(
            compute_export_bounds(&datasets, &log_opts, Some([0.0, -1.0, -3.0, 50.0])),
            [1.0, 3.0, 2.0, 4.0]
        );
    }

    #[test]
    fn test_dataset_and_primitive_layout_pixel_parity() {
        use crate::export_primitives::{
            compute_primitive_bounds_for_options, datasets_to_primitives, export_primitives_svg,
        };

        let datasets = vec![(
            "Signal".to_string(),
            vec![[1.0, 10.0], [10.0, 100.0], [100.0, 1000.0]],
        )];
        let prims = datasets_to_primitives(&datasets);
        for log_axes in [false, true] {
            let opts = ExportOptions {
                width: 640,
                height: 360,
                log_x: log_axes,
                log_y: log_axes,
                ..ExportOptions::default()
            };
            let (w, h) = resolve_canvas_dimensions(&opts).unwrap();
            let ds_layout = PrimitiveCanvasLayout::from_projected_bounds(
                w,
                h,
                compute_export_bounds(&datasets, &opts, None),
                &opts,
            );
            let prim_bounds = compute_primitive_bounds_for_options(&prims, &opts).unwrap();
            let prim_layout = PrimitiveCanvasLayout::new(w, h, prim_bounds, &opts);

            for pt in &datasets[0].1 {
                assert_eq!(ds_layout.to_pixel(*pt), prim_layout.to_pixel(*pt));
            }
            let ds_svg = export_custom_svg(&datasets, &opts).unwrap();
            let prim_svg = export_primitives_svg(&prims, &opts).unwrap();
            let ds_pts = ds_svg
                .split("<polyline points=\"")
                .nth(1)
                .unwrap()
                .split('"')
                .next()
                .unwrap();
            let prim_pts = prim_svg
                .split("<polyline points=\"")
                .nth(1)
                .unwrap()
                .split('"')
                .next()
                .unwrap();
            assert_eq!(ds_pts, prim_pts);
        }
    }
}

#[cfg(all(test, target_arch = "wasm32"))]
#[path = "export_wasm_tests.rs"]
mod wasm_tests;
