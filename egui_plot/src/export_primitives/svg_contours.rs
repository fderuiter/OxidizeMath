//! Standalone vector `<svg>` serializer for 2D Marching Squares contours and discrete lattices.

use std::fmt::Write as _;

use crate::export::{ExportOptions, copy_text_to_clipboard, copy_text_to_clipboard_with_callback};

use super::{
    resolve_canvas_dimensions,
    svg::{escape_xml, svg_document_header},
};

/// Minimum number of contour thresholds selectable in the export modal.
pub const MIN_SVG_CONTOUR_LEVELS: usize = 1;

/// Maximum number of contour thresholds selectable in the export modal.
pub const MAX_SVG_CONTOUR_LEVELS: usize = 16;

/// Single polyline or closed loop in 2D field/lattice coordinates.
#[derive(Clone, Debug, PartialEq)]
pub struct SvgContourPath {
    /// Ordered `[x, y]` vertices in domain coordinates `[0..domain_width, 0..domain_height]`.
    pub points: Vec<[f64; 2]>,
    /// Whether the path closes back onto its starting vertex (`Z`).
    pub closed: bool,
}

/// Styled group of vector contour or lattice paths at a single threshold or layer.
#[derive(Clone, Debug, PartialEq)]
pub struct SvgContourLayer {
    /// Descriptive layer label (stored in `data-layer`).
    pub label: String,
    /// Stroke color for paths in this layer.
    pub color: egui::Color32,
    /// Stroke width multiplier in pixels before `ExportOptions::scale`.
    pub stroke_width: f32,
    /// Optional interior fill color for closed polygons.
    pub fill: Option<egui::Color32>,
    /// Vector paths belonging to this layer.
    pub paths: Vec<SvgContourPath>,
}

/// Colored discrete matrix cell for vector `<rect>` heatmap SVG exports.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SvgRectCell {
    /// Zero-based column index.
    pub col: usize,
    /// Zero-based row index (top-down).
    pub row: usize,
    /// Cell interior fill color.
    pub fill: egui::Color32,
    /// Optional `(stroke_color, stroke_width)` border.
    pub stroke: Option<(egui::Color32, f32)>,
}

struct DomainViewport {
    offset_x: f64,
    offset_y: f64,
    scale_x: f64,
    scale_y: f64,
}

impl DomainViewport {
    fn fit_aspect(
        canvas_w: usize,
        canvas_h: usize,
        domain_w: f64,
        domain_h: f64,
        scale: f32,
    ) -> Self {
        let pad = (16.0 * f64::from(scale.max(0.5))).clamp(8.0, 80.0);
        let avail_w = (canvas_w as f64 - 2.0 * pad).max(1.0);
        let avail_h = (canvas_h as f64 - 2.0 * pad).max(1.0);
        let uniform = (avail_w / domain_w).min(avail_h / domain_h);
        let used_w = domain_w * uniform;
        let used_h = domain_h * uniform;
        Self {
            offset_x: (canvas_w as f64 - used_w) * 0.5,
            offset_y: (canvas_h as f64 - used_h) * 0.5,
            scale_x: uniform,
            scale_y: uniform,
        }
    }

    #[inline]
    fn project(&self, pt: [f64; 2]) -> Option<(f64, f64)> {
        if !pt[0].is_finite() || !pt[1].is_finite() {
            return None;
        }
        let px = self.offset_x + pt[0] * self.scale_x;
        let py = self.offset_y + pt[1] * self.scale_y;
        (px.is_finite() && py.is_finite()).then_some((px, py))
    }
}

fn build_svg_path_data(path: &SvgContourPath, vp: &DomainViewport) -> Option<String> {
    let mut projected = Vec::with_capacity(path.points.len());
    for &pt in &path.points {
        if let Some(px_py) = vp.project(pt) {
            projected.push(px_py);
        }
    }
    let min_pts = if path.closed { 3 } else { 2 };
    if projected.len() < min_pts {
        return None;
    }
    let mut d = String::with_capacity(projected.len() * 18);
    let (x0, y0) = projected[0];
    let _ = write!(d, "M {x0:.2} {y0:.2}");
    for &(x, y) in &projected[1..] {
        let _ = write!(d, " L {x:.2} {y:.2}");
    }
    if path.closed {
        d.push_str(" Z");
    }
    Some(d)
}

fn write_layer_svg(
    svg: &mut String,
    layer: &SvgContourLayer,
    vp: &DomainViewport,
    options: &ExportOptions,
) -> usize {
    let stroke_op = f64::from(layer.color.a()) / 255.0;
    let sw = (f64::from(layer.stroke_width.max(0.5)) * f64::from(options.scale.max(0.5))).max(0.75);
    let (fill_attr, fill_op_attr) = match layer.fill {
        Some(fc) if fc.a() > 0 => (
            format!("rgb({},{},{})", fc.r(), fc.g(), fc.b()),
            format!(" fill-opacity=\"{:.3}\"", f64::from(fc.a()) / 255.0),
        ),
        _ => ("none".to_string(), String::new()),
    };

    let mut emitted = 0_usize;
    let escaped_label = escape_xml(&layer.label);
    let _ = writeln!(svg, "  <g data-layer=\"{escaped_label}\">");
    for path in &layer.paths {
        if let Some(d) = build_svg_path_data(path, vp) {
            let _ = writeln!(
                svg,
                "    <path d=\"{d}\" fill=\"{fill_attr}\"{fill_op_attr} stroke=\"rgb({},{},{})\" stroke-opacity=\"{stroke_op:.3}\" stroke-width=\"{sw:.2}\" stroke-linejoin=\"round\" stroke-linecap=\"round\" />",
                layer.color.r(),
                layer.color.g(),
                layer.color.b(),
            );
            emitted += 1;
        }
    }
    svg.push_str("  </g>\n");
    emitted
}

/// Serializes 2D contour or lattice [`SvgContourLayer`] slices into a standalone `<svg>` document.
pub fn export_contour_layers_svg(
    domain_width: f64,
    domain_height: f64,
    layers: &[SvgContourLayer],
    options: &ExportOptions,
) -> Result<String, String> {
    if !domain_width.is_finite()
        || !domain_height.is_finite()
        || domain_width <= 0.0
        || domain_height <= 0.0
    {
        return Err("Invalid 2D scalar-field domain dimensions for SVG export".to_string());
    }
    let (canvas_w, canvas_h) = resolve_canvas_dimensions(options)?;
    let vp = DomainViewport::fit_aspect(
        canvas_w,
        canvas_h,
        domain_width,
        domain_height,
        options.scale,
    );
    let mut svg = svg_document_header(canvas_w, canvas_h, options);
    let mut total_paths = 0_usize;
    for layer in layers {
        total_paths += write_layer_svg(&mut svg, layer, &vp, options);
    }
    if total_paths == 0 {
        return Err("No finite vector contour paths available for SVG export".to_string());
    }
    svg.push_str("</svg>\n");
    Ok(svg)
}

/// Serializes a discrete `cols x rows` grid of [`SvgRectCell`] items into crisp `<rect>` SVG geometry.
pub fn export_rect_grid_svg(
    cols: usize,
    rows: usize,
    cells: &[SvgRectCell],
    options: &ExportOptions,
) -> Result<String, String> {
    if cols == 0 || rows == 0 || cells.is_empty() {
        return Err("Empty matrix grid for SVG export".to_string());
    }
    let (canvas_w, canvas_h) = resolve_canvas_dimensions(options)?;
    let vp =
        DomainViewport::fit_aspect(canvas_w, canvas_h, cols as f64, rows as f64, options.scale);
    let mut svg = svg_document_header(canvas_w, canvas_h, options);
    svg.push_str("  <g data-layer=\"matrix-cells\" shape-rendering=\"crispEdges\">\n");

    let mut emitted = 0_usize;
    for cell in cells {
        if cell.col >= cols || cell.row >= rows {
            continue;
        }
        let Some((x0, y0)) = vp.project([cell.col as f64, cell.row as f64]) else {
            continue;
        };
        let Some((x1, y1)) = vp.project([(cell.col + 1) as f64, (cell.row + 1) as f64]) else {
            continue;
        };
        let (w, h) = ((x1 - x0).max(0.5), (y1 - y0).max(0.5));
        let fill_op = f64::from(cell.fill.a()) / 255.0;
        let stroke_attr = match cell.stroke {
            Some((sc, sw)) if sw > 0.0 => format!(
                " stroke=\"rgb({},{},{})\" stroke-width=\"{:.2}\"",
                sc.r(),
                sc.g(),
                sc.b(),
                sw
            ),
            _ => String::new(),
        };
        let _ = writeln!(
            svg,
            "    <rect x=\"{x0:.2}\" y=\"{y0:.2}\" width=\"{w:.2}\" height=\"{h:.2}\" fill=\"rgb({},{},{})\" fill-opacity=\"{fill_op:.3}\"{stroke_attr} />",
            cell.fill.r(),
            cell.fill.g(),
            cell.fill.b(),
        );
        emitted += 1;
    }
    if emitted == 0 {
        return Err("No in-bounds cells available for SVG export".to_string());
    }
    svg.push_str("  </g>\n</svg>\n");
    Ok(svg)
}

/// Copies standalone SVG vector text directly to the system clipboard.
pub fn copy_svg_to_clipboard(svg_text: &str) -> Result<(), String> {
    if svg_text.trim().is_empty() {
        return Err("Cannot copy empty SVG markup to clipboard".to_string());
    }
    copy_text_to_clipboard(svg_text)
}

/// Copies standalone SVG vector text to the clipboard and invokes `on_complete` when finished.
pub fn copy_svg_to_clipboard_with_callback(
    svg_text: &str,
    on_complete: impl FnOnce(Result<(), String>) + 'static,
) -> Result<(), String> {
    if svg_text.trim().is_empty() {
        return Err("Cannot copy empty SVG markup to clipboard".to_string());
    }
    copy_text_to_clipboard_with_callback(svg_text, on_complete)
}
