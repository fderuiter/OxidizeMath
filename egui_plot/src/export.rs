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
}

impl ImportedDataset {
    /// Create a new imported dataset with default visibility (true).
    pub fn new(name: impl Into<String>, points: Vec<[f64; 2]>) -> Self {
        Self {
            name: name.into(),
            points,
            visible: true,
        }
    }
}

/// Parse tabular dataset text (CSV, TSV, semicolon-separated) into imported datasets.
/// Auto-detects delimiters and headers, converts rows into numerical `[f64; 2]` pairs,
/// and skips invalid rows or non-numeric values.
pub fn parse_csv(content: &str) -> Vec<ImportedDataset> {
    let lines: Vec<&str> = content
        .lines()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty() && !s.starts_with('#'))
        .collect();

    if lines.is_empty() {
        return Vec::new();
    }

    let first_line = lines[0];

    let delimiter = if first_line.contains('\t') {
        '\t'
    } else if first_line.contains(';') {
        ';'
    } else {
        ','
    };

    fn split_line(line: &str, delimiter: char) -> Vec<&str> {
        line.split(delimiter).map(|s| s.trim()).collect()
    }

    let first_tokens = split_line(first_line, delimiter);
    let is_num = |s: &str| s.parse::<f64>().is_ok();

    let (headers, data_start_index) = if first_tokens.iter().any(|&t| !is_num(t)) {
        (
            first_tokens
                .iter()
                .map(|&s| s.to_string())
                .collect::<Vec<_>>(),
            1,
        )
    } else {
        (Vec::new(), 0)
    };

    let is_standard_export = if headers.len() == 3 {
        let h0 = headers[0].to_lowercase();
        let h1 = headers[1].to_lowercase();
        let h2 = headers[2].to_lowercase();
        h0.contains("dataset") || h0.contains("name") || (h1 == "x" && h2 == "y")
    } else {
        false
    };

    let mut datasets_map: std::collections::BTreeMap<String, Vec<[f64; 2]>> =
        std::collections::BTreeMap::new();

    if is_standard_export {
        for line in &lines[data_start_index..] {
            let tokens = split_line(line, delimiter);
            if tokens.len() >= 3
                && let (Ok(x), Ok(y)) = (tokens[1].parse::<f64>(), tokens[2].parse::<f64>())
                && x.is_finite()
                && y.is_finite()
            {
                datasets_map
                    .entry(tokens[0].to_string())
                    .or_default()
                    .push([x, y]);
            }
        }
    } else if headers.len() > 2 {
        let mut series_names = Vec::new();
        for h in &headers[1..] {
            series_names.push(if h.is_empty() {
                "Imported Dataset".to_string()
            } else {
                h.clone()
            });
        }

        for line in &lines[data_start_index..] {
            let tokens = split_line(line, delimiter);
            if tokens.len() >= 2
                && let Ok(x) = tokens[0].parse::<f64>()
            {
                for (i, y_str) in tokens[1..].iter().enumerate() {
                    if i < series_names.len()
                        && let Ok(y) = y_str.parse::<f64>()
                        && x.is_finite()
                        && y.is_finite()
                    {
                        datasets_map
                            .entry(series_names[i].clone())
                            .or_default()
                            .push([x, y]);
                    }
                }
            }
        }
    } else {
        let default_name = if headers.len() >= 2 && !headers[1].is_empty() {
            let h1 = headers[1].trim();
            if h1.eq_ignore_ascii_case("y") {
                "Imported Dataset".to_string()
            } else {
                h1.to_string()
            }
        } else {
            "Imported Dataset".to_string()
        };

        for line in &lines[data_start_index..] {
            let tokens = split_line(line, delimiter);
            if tokens.len() >= 2 {
                if tokens.len() >= 3 && !is_num(tokens[0]) {
                    if let (Ok(x), Ok(y)) = (tokens[1].parse::<f64>(), tokens[2].parse::<f64>())
                        && x.is_finite()
                        && y.is_finite()
                    {
                        datasets_map
                            .entry(tokens[0].to_string())
                            .or_default()
                            .push([x, y]);
                    }
                } else if let (Ok(x), Ok(y)) = (tokens[0].parse::<f64>(), tokens[1].parse::<f64>())
                    && x.is_finite()
                    && y.is_finite()
                {
                    datasets_map
                        .entry(default_name.clone())
                        .or_default()
                        .push([x, y]);
                }
            }
        }
    }

    datasets_map
        .into_iter()
        .filter(|(_, pts)| !pts.is_empty())
        .map(|(name, points)| ImportedDataset::new(name, points))
        .collect()
}

type PendingImport = (egui::Id, Vec<ImportedDataset>);

static PENDING_IMPORTS: std::sync::Mutex<
    Option<(
        std::sync::mpsc::Sender<PendingImport>,
        std::sync::mpsc::Receiver<PendingImport>,
    )>,
> = std::sync::Mutex::new(None);

/// Drain any asynchronously imported datasets for the given plot ID.
pub fn drain_imported_datasets(plot_id: egui::Id) -> Vec<ImportedDataset> {
    let mut datasets = Vec::new();
    if let Ok(guard) = PENDING_IMPORTS.lock()
        && let Some((_, rx)) = guard.as_ref()
    {
        while let Ok((id, ds_list)) = rx.try_recv() {
            if id == plot_id || id == egui::Id::NULL {
                datasets.extend(ds_list);
            }
        }
    }
    datasets
}

/// Trigger cross-platform CSV dataset file import dialog asynchronously.
pub fn trigger_import_csv(sender: std::sync::mpsc::Sender<Vec<ImportedDataset>>) {
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::thread::spawn(move || {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("CSV / TSV File", &["csv", "tsv", "txt"])
                .pick_file()
                && let Ok(content) = std::fs::read_to_string(path)
            {
                let datasets = parse_csv(&content);
                if !datasets.is_empty() {
                    let _ = sender.send(datasets);
                }
            }
        });
    }
}

/// Trigger CSV file import specifically for a target plot ID.
pub fn trigger_import_csv_for_plot(ctx: &egui::Context, plot_id: egui::Id) {
    let ctx_clone = ctx.clone();
    let tx = {
        let mut guard = PENDING_IMPORTS.lock().unwrap();
        if guard.is_none() {
            let (tx, rx) = std::sync::mpsc::channel();
            *guard = Some((tx, rx));
        }
        guard.as_ref().unwrap().0.clone()
    };

    #[cfg(not(target_arch = "wasm32"))]
    {
        std::thread::spawn(move || {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("CSV / TSV File", &["csv", "tsv", "txt"])
                .pick_file()
                && let Ok(content) = std::fs::read_to_string(path)
            {
                let datasets = parse_csv(&content);
                if !datasets.is_empty() {
                    let _ = tx.send((plot_id, datasets));
                    ctx_clone.request_repaint();
                }
            }
        });
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

/// Export dataset points to CSV string with a custom filename and initiate download or file save dialog.
pub fn export_csv_filename(datasets: &[(String, Vec<[f64; 2]>)], filename: &str) {
    let csv = format_csv(datasets);
    save_file_dialog(filename, csv.as_bytes(), "CSV File", &["csv"]);
}

/// Export dataset points to CSV string and initiate download or file save dialog.
pub fn export_csv(datasets: &[(String, Vec<[f64; 2]>)]) {
    export_csv_filename(datasets, "export.csv");
}

/// Export dataset points to JSON string with a custom filename and initiate download or file save dialog.
pub fn export_json_filename(datasets: &[(String, Vec<[f64; 2]>)], filename: &str) {
    let json = format_json(datasets);
    save_file_dialog(filename, json.as_bytes(), "JSON File", &["json"]);
}

/// Export dataset points to JSON string and initiate download or file save dialog.
pub fn export_json(datasets: &[(String, Vec<[f64; 2]>)]) {
    export_json_filename(datasets, "export.json");
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
    let target_w = (options.width as f32 * options.scale).round() as u32;
    let mut target_h = (options.height as f32 * options.scale).round() as u32;

    if let Some(ratio) = options.aspect_ratio.ratio().filter(|&r| r > 0.0) {
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

    let (mut min_x, mut max_x, mut min_y, mut max_y) = if let Some(b) = custom_bounds {
        (b[0], b[1], b[2], b[3])
    } else {
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
        (min_x, max_x, min_y, max_y)
    };

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

            let mut rgba_buf = Vec::with_capacity((width * height * 4) as usize);
            for p in &img.pixels {
                rgba_buf.push(p.r());
                rgba_buf.push(p.g());
                rgba_buf.push(p.b());
                rgba_buf.push(p.a());
            }

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
            for p in pts {
                if p[0].is_finite() && p[1].is_finite() {
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

/// Type alias representing a single exported frame's dataset series.
pub type ExportFrame = DatasetFrame;

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
        } else if filename.ends_with(".gif") {
            "image/gif"
        } else {
            "application/octet-stream"
        };

        if let Some(window) = web_sys::window() {
            if let Some(document) = window.document() {
                let uint8_array = js_sys::Uint8Array::from(content);
                let array = js_sys::Array::new();
                array.push(&uint8_array);
                let blob_props = BlobPropertyBag::new();
                blob_props.set_type(mime_type);
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
}
