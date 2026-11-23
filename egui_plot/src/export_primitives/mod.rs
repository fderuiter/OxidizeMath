//! Offscreen plot primitive representation, capture, and rendering for PNG, SVG, and GIF exports.

/// Reusable grid-cell and node-link graph [`ExportPrimitive`] builders.
pub mod adapters;
pub(crate) mod pixel_ops;
pub(crate) mod raster;
pub(crate) mod svg;
#[cfg(test)]
mod tests;

use std::cell::RefCell;
use std::sync::Arc;

use crate::export::{
    EXPORT_PALETTE, ExportOptions, MAX_EXPORT_DIMENSION, color32_image_to_png,
    color32_images_to_gif,
};

pub use adapters::{
    GraphEdgeSpec, GraphNodeSpec, GridCellSpec, build_graph_primitives, build_grid_primitives,
};
pub use raster::{render_primitives_image, render_primitives_image_with_bounds, scale_color_image};
pub use svg::{encode_base64, export_primitives_svg, export_raster_svg};

/// Offscreen-drawable plot primitive extracted from [`crate::PlotItem`] instances.
#[derive(Clone, Debug)]
pub enum ExportPrimitive {
    /// Connected polyline or closed polygon series.
    Polyline {
        /// Series name.
        name: String,
        /// Ordered 2D coordinates in plot space.
        points: Vec<[f64; 2]>,
        /// Stroke color (`Color32::TRANSPARENT` falls back to [`EXPORT_PALETTE`]).
        color: egui::Color32,
        /// Stroke width in UI points.
        stroke_width: f32,
        /// Whether the last point connects back to the first point.
        closed: bool,
        /// Optional interior fill color for closed polygons.
        fill: Option<egui::Color32>,
    },
    /// Discrete scatter marker points.
    Points {
        /// Series name.
        name: String,
        /// 2D marker coordinates in plot space.
        points: Vec<[f64; 2]>,
        /// Marker color (`Color32::TRANSPARENT` falls back to [`EXPORT_PALETTE`]).
        color: egui::Color32,
        /// Marker radius in UI points.
        radius: f32,
    },
    /// Axis-aligned bar rectangle in plot coordinates.
    Bar {
        /// Series or bar name.
        name: String,
        /// Minimum `[x, y]` corner in plot space.
        min: [f64; 2],
        /// Maximum `[x, y]` corner in plot space.
        max: [f64; 2],
        /// Interior fill color.
        fill: egui::Color32,
        /// Outline stroke color.
        stroke_color: egui::Color32,
        /// Outline stroke width in UI points.
        stroke_width: f32,
    },
    /// Directed vector segment with an arrowhead at `tip`.
    Arrow {
        /// Series name.
        name: String,
        /// Arrow tail `[x, y]` in plot space.
        origin: [f64; 2],
        /// Arrow head `[x, y]` in plot space.
        tip: [f64; 2],
        /// Optional arrowhead wing length in UI points.
        tip_length: Option<f32>,
        /// Arrow stroke color.
        color: egui::Color32,
    },
    /// Five-number summary box-and-whisker element.
    BoxElem {
        /// Series or box name.
        name: String,
        /// `true` when oriented vertically (argument on X, spread on Y).
        vertical: bool,
        /// Coordinate along the independent axis.
        argument: f64,
        /// Lower whisker value.
        lower_whisker: f64,
        /// First quartile (25th percentile) value.
        quartile1: f64,
        /// Median (50th percentile) value.
        median: f64,
        /// Third quartile (75th percentile) value.
        quartile3: f64,
        /// Upper whisker value.
        upper_whisker: f64,
        /// IQR box width along the argument axis.
        box_width: f64,
        /// Whisker end-cap width along the argument axis.
        whisker_width: f64,
        /// IQR box fill color.
        fill: egui::Color32,
        /// Box, median, and whisker stroke color.
        stroke_color: egui::Color32,
        /// Stroke width in UI points.
        stroke_width: f32,
    },
    /// Embedded raster image in plot coordinates or full plot area.
    RasterImage {
        /// Image label.
        name: String,
        /// Raster pixel buffer.
        image: Arc<egui::ColorImage>,
        /// Optional `[min_x, max_x, min_y, max_y]` plot-space bounds.
        bounds: Option<[f64; 4]>,
    },
}

/// Deferred offscreen primitive handle stored during interactive frame rendering and materialized on export.
#[derive(Clone, Debug)]
pub enum DeferredPlotPrimitive {
    /// Zero-copy shared polyline or polygon series.
    Polyline {
        /// Series name.
        name: String,
        /// Shared plot-space points.
        points: Arc<Vec<crate::PlotPoint>>,
        /// Stroke color.
        color: egui::Color32,
        /// Stroke width in UI points.
        stroke_width: f32,
        /// Whether the polyline is closed.
        closed: bool,
        /// Optional polygon fill color.
        fill: Option<egui::Color32>,
    },
    /// Zero-copy batch of disjoint polyline segments sharing stroke style.
    PolylineBatch {
        /// Series name.
        name: String,
        /// Disjoint polyline segments in plot space.
        segments: Arc<Vec<Vec<crate::PlotPoint>>>,
        /// Stroke color.
        color: egui::Color32,
        /// Stroke width in UI points.
        stroke_width: f32,
    },
    /// Zero-copy shared scatter marker series.
    Points {
        /// Series name.
        name: String,
        /// Shared plot-space points.
        points: Arc<Vec<crate::PlotPoint>>,
        /// Marker color.
        color: egui::Color32,
        /// Marker radius in UI points.
        radius: f32,
    },
    /// Zero-copy shared vector field arrows.
    Arrows {
        /// Series name.
        name: String,
        /// Shared arrow tail coordinates.
        origins: Arc<Vec<crate::PlotPoint>>,
        /// Shared arrow head coordinates.
        tips: Arc<Vec<crate::PlotPoint>>,
        /// Optional arrowhead wing length in UI points.
        tip_length: Option<f32>,
        /// Arrow stroke color.
        color: egui::Color32,
    },
    /// Pre-built export primitives (e.g., bars, box plots, or explicit caller primitives).
    Eager(Vec<ExportPrimitive>),
}

impl DeferredPlotPrimitive {
    fn materialize_into(&self, out: &mut Vec<ExportPrimitive>) {
        match self {
            Self::Polyline {
                name,
                points,
                color,
                stroke_width,
                closed,
                fill,
            } => {
                if !points.is_empty() {
                    out.push(ExportPrimitive::Polyline {
                        name: name.clone(),
                        points: points.iter().map(|p| [p.x, p.y]).collect(),
                        color: *color,
                        stroke_width: *stroke_width,
                        closed: *closed,
                        fill: *fill,
                    });
                }
            }
            Self::PolylineBatch {
                name,
                segments,
                color,
                stroke_width,
            } => {
                for seg in segments.iter().filter(|seg| !seg.is_empty()) {
                    out.push(ExportPrimitive::Polyline {
                        name: name.clone(),
                        points: seg.iter().map(|p| [p.x, p.y]).collect(),
                        color: *color,
                        stroke_width: *stroke_width,
                        closed: false,
                        fill: None,
                    });
                }
            }
            Self::Points {
                name,
                points,
                color,
                radius,
            } => {
                if !points.is_empty() {
                    out.push(ExportPrimitive::Points {
                        name: name.clone(),
                        points: points.iter().map(|p| [p.x, p.y]).collect(),
                        color: *color,
                        radius: *radius,
                    });
                }
            }
            Self::Arrows {
                name,
                origins,
                tips,
                tip_length,
                color,
            } => {
                for (origin, tip) in origins.iter().zip(tips.iter()) {
                    out.push(ExportPrimitive::Arrow {
                        name: name.clone(),
                        origin: [origin.x, origin.y],
                        tip: [tip.x, tip.y],
                        tip_length: *tip_length,
                        color: *color,
                    });
                }
            }
            Self::Eager(items) => out.extend(items.iter().cloned()),
        }
    }
}

thread_local! {
    static MATERIALIZED_EXPORT_PRIMITIVES_COUNT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[doc(hidden)]
pub fn materialized_export_primitives_call_count() -> usize {
    MATERIALIZED_EXPORT_PRIMITIVES_COUNT.with(std::cell::Cell::get)
}

#[doc(hidden)]
pub fn reset_materialized_export_primitives_call_count() {
    MATERIALIZED_EXPORT_PRIMITIVES_COUNT.with(|c| c.set(0));
}

thread_local! {
    static LAST_PLOT_PRIMITIVES: RefCell<Vec<DeferredPlotPrimitive>> = const { RefCell::new(Vec::new()) };
}

/// Clears thread-local offscreen plot primitives captured during the current frame.
pub fn clear_last_plot_primitives() {
    LAST_PLOT_PRIMITIVES.with(|p| p.borrow_mut().clear());
}

/// Appends offscreen plot primitives captured from a rendered [`crate::Plot`] widget.
pub fn append_last_plot_primitives(primitives: &[ExportPrimitive]) {
    if primitives.is_empty() {
        return;
    }
    LAST_PLOT_PRIMITIVES.with(|p| {
        p.borrow_mut()
            .push(DeferredPlotPrimitive::Eager(primitives.to_vec()));
    });
}

pub(crate) fn append_deferred_plot_primitives(primitives: Vec<DeferredPlotPrimitive>) {
    if primitives.is_empty() {
        return;
    }
    LAST_PLOT_PRIMITIVES.with(|p| {
        p.borrow_mut().extend(primitives);
    });
}

/// Returns all offscreen plot primitives captured across [`crate::Plot`] widgets in the current frame.
pub fn get_last_plot_primitives() -> Vec<ExportPrimitive> {
    MATERIALIZED_EXPORT_PRIMITIVES_COUNT.with(|c| c.set(c.get() + 1));
    LAST_PLOT_PRIMITIVES.with(|p| {
        let borrowed = p.borrow();
        let mut out = Vec::with_capacity(borrowed.len());
        for entry in borrowed.iter() {
            entry.materialize_into(&mut out);
        }
        out
    })
}

/// Returns `true` if `datasets` contains at least one finite `[x, y]` coordinate pair.
pub fn has_finite_dataset_points(datasets: &[(String, Vec<[f64; 2]>)]) -> bool {
    datasets
        .iter()
        .flat_map(|(_, pts)| pts.iter())
        .any(|p| p[0].is_finite() && p[1].is_finite())
}

/// Returns `true` if `primitives` contains at least one finite drawable shape or non-empty raster image.
pub fn has_drawable_primitives(primitives: &[ExportPrimitive]) -> bool {
    compute_primitive_bounds(primitives).is_some()
        || primitives.iter().any(|p| {
            matches!(
                p,
                ExportPrimitive::RasterImage { image, .. }
                    if image.width() > 0 && image.height() > 0 && !image.pixels.is_empty()
            )
        })
}

pub(crate) fn is_valid_coord_pair(p: [f64; 2], log_x: bool, log_y: bool) -> bool {
    p[0].is_finite() && p[1].is_finite() && (!log_x || p[0] > 0.0) && (!log_y || p[1] > 0.0)
}

#[inline]
pub(crate) fn project_coord(val: f64, is_log: bool) -> f64 {
    if is_log {
        val.log10().clamp(-300.0, 300.0)
    } else {
        val
    }
}

pub(crate) fn sanitize_bar_corners(
    min: [f64; 2],
    max: [f64; 2],
    log_x: bool,
    log_y: bool,
) -> Option<([f64; 2], [f64; 2])> {
    if !min[0].is_finite() || !min[1].is_finite() || !max[0].is_finite() || !max[1].is_finite() {
        return None;
    }
    let (mut x0, x1) = (min[0].min(max[0]), min[0].max(max[0]));
    let (mut y0, y1) = (min[1].min(max[1]), min[1].max(max[1]));
    if log_x {
        if x1 <= 0.0 {
            return None;
        }
        if x0 <= 0.0 {
            x0 = (x1 * 1e-3).min(1e-3);
        }
    }
    if log_y {
        if y1 <= 0.0 {
            return None;
        }
        if y0 <= 0.0 {
            y0 = (y1 * 1e-3).min(1e-3);
        }
    }
    Some(([x0, y0], [x1, y1]))
}

/// Converts legacy `(name, points)` datasets into [`ExportPrimitive::Polyline`] entries.
pub fn datasets_to_primitives(datasets: &[(String, Vec<[f64; 2]>)]) -> Vec<ExportPrimitive> {
    datasets
        .iter()
        .enumerate()
        .filter(|(_, (_, pts))| pts.iter().any(|p| p[0].is_finite() && p[1].is_finite()))
        .map(|(idx, (name, pts))| ExportPrimitive::Polyline {
            name: name.clone(),
            points: pts.clone(),
            color: EXPORT_PALETTE[idx % EXPORT_PALETTE.len()],
            stroke_width: 2.0,
            closed: false,
            fill: None,
        })
        .collect()
}

/// Computes `[min_x, max_x, min_y, max_y]` across all finite coordinates in `primitives`.
pub fn compute_primitive_bounds(primitives: &[ExportPrimitive]) -> Option<[f64; 4]> {
    compute_primitive_bounds_for_options(primitives, &ExportOptions::default())
}

/// Computes `[min_x, max_x, min_y, max_y]` across valid coordinates in `primitives` under `options` axis scaling.
pub fn compute_primitive_bounds_for_options(
    primitives: &[ExportPrimitive],
    options: &ExportOptions,
) -> Option<[f64; 4]> {
    let mut bounds = BoundsAccumulator::new(options.log_x, options.log_y);
    for primitive in primitives {
        bounds.include_primitive(primitive);
    }
    bounds.finish()
}

struct BoundsAccumulator {
    log_x: bool,
    log_y: bool,
    min_x: f64,
    max_x: f64,
    min_y: f64,
    max_y: f64,
}

impl BoundsAccumulator {
    fn new(log_x: bool, log_y: bool) -> Self {
        Self {
            log_x,
            log_y,
            min_x: f64::INFINITY,
            max_x: f64::NEG_INFINITY,
            min_y: f64::INFINITY,
            max_y: f64::NEG_INFINITY,
        }
    }

    fn include_point(&mut self, p: [f64; 2]) {
        if is_valid_coord_pair(p, self.log_x, self.log_y) {
            self.min_x = self.min_x.min(p[0]);
            self.max_x = self.max_x.max(p[0]);
            self.min_y = self.min_y.min(p[1]);
            self.max_y = self.max_y.max(p[1]);
        }
    }

    fn include_primitive(&mut self, primitive: &ExportPrimitive) {
        match primitive {
            ExportPrimitive::Polyline { points, .. } | ExportPrimitive::Points { points, .. } => {
                for &pt in points {
                    self.include_point(pt);
                }
            }
            ExportPrimitive::Bar { min, max, .. } => {
                if let Some((c0, c1)) = sanitize_bar_corners(*min, *max, self.log_x, self.log_y) {
                    self.include_point(c0);
                    self.include_point(c1);
                }
            }
            ExportPrimitive::Arrow { origin, tip, .. } => {
                if is_valid_coord_pair(*origin, self.log_x, self.log_y)
                    && is_valid_coord_pair(*tip, self.log_x, self.log_y)
                {
                    self.include_point(*origin);
                    self.include_point(*tip);
                }
            }
            ExportPrimitive::BoxElem {
                vertical,
                argument,
                lower_whisker,
                upper_whisker,
                box_width,
                whisker_width,
                ..
            } => {
                let half_w = box_width.max(*whisker_width) * 0.5;
                let (a0, a1) = (argument - half_w, argument + half_w);
                if *vertical {
                    self.include_point([a0, *lower_whisker]);
                    self.include_point([a1, *upper_whisker]);
                } else {
                    self.include_point([*lower_whisker, a0]);
                    self.include_point([*upper_whisker, a1]);
                }
            }
            ExportPrimitive::RasterImage { bounds, image, .. } => {
                if let Some([x0, x1, y0, y1]) = bounds {
                    self.include_point([*x0, *y0]);
                    self.include_point([*x1, *y1]);
                } else if image.width() > 0 && image.height() > 0 {
                    let min_c = if self.log_x || self.log_y { 1.0 } else { 0.0 };
                    self.include_point([min_c, min_c]);
                    self.include_point([image.width() as f64, image.height() as f64]);
                }
            }
        }
    }

    fn finish(self) -> Option<[f64; 4]> {
        if self.min_x.is_finite()
            && self.max_x.is_finite()
            && self.min_y.is_finite()
            && self.max_y.is_finite()
        {
            Some([self.min_x, self.max_x, self.min_y, self.max_y])
        } else {
            None
        }
    }
}

pub(crate) struct PrimitiveCanvasLayout {
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) padding: f64,
    pub(crate) draw_w: f64,
    pub(crate) draw_h: f64,
    min_x: f64,
    max_x: f64,
    min_y: f64,
    max_y: f64,
    pub(crate) log_x: bool,
    pub(crate) log_y: bool,
    pub(crate) scale: f64,
    pub(crate) base_stroke: f64,
}

impl PrimitiveCanvasLayout {
    pub(crate) fn new(
        width: usize,
        height: usize,
        raw_bounds: [f64; 4],
        options: &ExportOptions,
    ) -> Self {
        let bx0 = if options.log_x && raw_bounds[0] <= 0.0 {
            1e-6
        } else {
            raw_bounds[0]
        };
        let bx1 = if options.log_x && raw_bounds[1] <= bx0 {
            bx0 * 10.0
        } else {
            raw_bounds[1]
        };
        let by0 = if options.log_y && raw_bounds[2] <= 0.0 {
            1e-6
        } else {
            raw_bounds[2]
        };
        let by1 = if options.log_y && raw_bounds[3] <= by0 {
            by0 * 10.0
        } else {
            raw_bounds[3]
        };
        let projected = [
            project_coord(bx0, options.log_x),
            project_coord(bx1, options.log_x),
            project_coord(by0, options.log_y),
            project_coord(by1, options.log_y),
        ];
        Self::from_projected_bounds(width, height, projected, options)
    }

    pub(crate) fn from_projected_bounds(
        width: usize,
        height: usize,
        projected: [f64; 4],
        options: &ExportOptions,
    ) -> Self {
        let [min_x, max_x, min_y, max_y] = padded_bounds(projected);
        let padding = (20.0 * options.scale as f64).clamp(10.0, 100.0);
        let draw_w = (width as f64 - 2.0 * padding).max(1.0);
        let draw_h = (height as f64 - 2.0 * padding).max(1.0);
        Self {
            width,
            height,
            padding,
            draw_w,
            draw_h,
            min_x,
            max_x,
            min_y,
            max_y,
            log_x: options.log_x,
            log_y: options.log_y,
            scale: (options.scale as f64).max(0.5),
            base_stroke: (options.line_thickness * options.scale).max(1.0) as f64,
        }
    }

    #[inline]
    pub(crate) fn is_valid_point(&self, p: [f64; 2]) -> bool {
        is_valid_coord_pair(p, self.log_x, self.log_y)
    }

    pub(crate) fn to_pixel(&self, p: [f64; 2]) -> (f64, f64) {
        let px_val = project_coord(p[0], self.log_x);
        let py_val = project_coord(p[1], self.log_y);
        let x_norm = (px_val - self.min_x) / (self.max_x - self.min_x);
        let y_norm = (py_val - self.min_y) / (self.max_y - self.min_y);
        let px = self.padding + x_norm * self.draw_w;
        let py = (self.height as f64 - self.padding) - y_norm * self.draw_h;
        (px, py)
    }
}

pub(crate) fn resolve_canvas_dimensions(options: &ExportOptions) -> Result<(usize, usize), String> {
    let target_w = (options.width as f32 * options.scale).round() as u32;
    let mut target_h = (options.height as f32 * options.scale).round() as u32;
    if let Some(ratio) = options.aspect_ratio.ratio().filter(|&r| r > 0.0) {
        target_h = (target_w as f32 / ratio).round() as u32;
    }
    if target_w > MAX_EXPORT_DIMENSION || target_h > MAX_EXPORT_DIMENSION {
        return Err(format!(
            "Requested dimensions ({target_w}x{target_h}) exceed maximum limit of {MAX_EXPORT_DIMENSION}x{MAX_EXPORT_DIMENSION}"
        ));
    }
    Ok((target_w.max(1) as usize, target_h.max(1) as usize))
}

pub(crate) fn padded_bounds(raw: [f64; 4]) -> [f64; 4] {
    let [mut min_x, mut max_x, mut min_y, mut max_y] = raw;
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
    [
        min_x - margin_x,
        max_x + margin_x,
        min_y - margin_y,
        max_y + margin_y,
    ]
}

pub(crate) fn resolve_color(explicit: egui::Color32, idx: usize) -> egui::Color32 {
    if explicit == egui::Color32::TRANSPARENT {
        EXPORT_PALETTE[idx % EXPORT_PALETTE.len()]
    } else {
        explicit
    }
}

/// Encodes a sequence of raster `ColorImage` frames scaled to `options` as an animated GIF.
pub fn export_raster_gif(
    frames: &[egui::ColorImage],
    options: &ExportOptions,
    fps: u32,
) -> Result<Vec<u8>, String> {
    if frames.is_empty() {
        return Err("No raster frames provided for GIF export".to_string());
    }
    let (width, height) = resolve_canvas_dimensions(options)?;
    let scaled: Vec<egui::ColorImage> = frames
        .iter()
        .filter(|img| img.width() > 0 && img.height() > 0 && !img.pixels.is_empty())
        .map(|img| scale_color_image(img, width, height, options))
        .collect();
    if scaled.is_empty() {
        return Err("All raster frames were empty".to_string());
    }
    color32_images_to_gif(&scaled, fps)
}

/// Renders `primitives` offscreen and encodes the resulting canvas as PNG bytes.
pub fn export_primitives_png(
    primitives: &[ExportPrimitive],
    options: &ExportOptions,
) -> Result<Vec<u8>, String> {
    let image = render_primitives_image(primitives, options)?;
    color32_image_to_png(&image)
}

/// Decodes PNG bytes into `(width, height, rgba_pixels)` for offscreen verification.
pub fn decode_png_rgba(png_bytes: &[u8]) -> Result<(usize, usize, Vec<[u8; 4]>), String> {
    let img = image::load_from_memory_with_format(png_bytes, image::ImageFormat::Png)
        .map_err(|e| format!("Failed to decode PNG: {e}"))?
        .to_rgba8();
    let (w, h) = img.dimensions();
    let pixels = img.pixels().map(|p| p.0).collect();
    Ok((w as usize, h as usize, pixels))
}
