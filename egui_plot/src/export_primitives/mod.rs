//! Offscreen plot primitive representation, capture, and rendering for PNG, SVG, and GIF exports.

mod pixel_ops;
mod raster;
mod svg;
#[cfg(test)]
mod tests;

use std::cell::RefCell;
use std::sync::Arc;

use crate::export::{
    EXPORT_PALETTE, ExportOptions, MAX_EXPORT_DIMENSION, color32_image_to_png,
    color32_images_to_gif,
};

pub use raster::{
    export_subplot_panels_png, render_primitives_image, render_primitives_image_with_bounds,
    render_subplot_panels_image, render_subplot_panels_image_with_bounds, scale_color_image,
};
pub use svg::{encode_base64, export_primitives_svg, export_raster_svg, export_subplot_panels_svg};

/// Captured per-subplot panel metadata, tabular datasets, and offscreen primitives from [`crate::Plot::show`].
#[derive(Clone, Debug)]
pub struct ExportSubplotPanel {
    /// Deterministic subplot identifier (e.g., `"z_plane"`, `"loss_history"`).
    pub id: String,
    /// Screen-space bounding rectangle occupied by this [`crate::Plot`] widget.
    pub screen_rect: egui::Rect,
    /// Whether the subplot X axis uses base-10 logarithmic scaling.
    pub log_x: bool,
    /// Whether the subplot Y axis uses base-10 logarithmic scaling.
    pub log_y: bool,
    /// Tabular `(series_name, points)` datasets extracted from this subplot.
    pub datasets: Vec<(String, Vec<[f64; 2]>)>,
    /// Drawable [`ExportPrimitive`] items extracted from this subplot.
    pub primitives: Vec<ExportPrimitive>,
}

/// Proportional pixel viewport rectangle assigned to a subplot on an offscreen canvas.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SubplotViewport {
    /// Left pixel coordinate on the target canvas.
    pub x: usize,
    /// Top pixel coordinate on the target canvas.
    pub y: usize,
    /// Viewport width in pixels (`>= 1`).
    pub width: usize,
    /// Viewport height in pixels (`>= 1`).
    pub height: usize,
}

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

thread_local! {
    static LAST_PLOT_PRIMITIVES: RefCell<Vec<ExportPrimitive>> = const { RefCell::new(Vec::new()) };
    static LAST_PLOT_SUBPLOTS: RefCell<Vec<ExportSubplotPanel>> = const { RefCell::new(Vec::new()) };
}

/// Clears thread-local offscreen plot primitives and subplot panels captured during the current frame.
pub fn clear_last_plot_primitives() {
    LAST_PLOT_PRIMITIVES.with(|p| p.borrow_mut().clear());
    clear_last_plot_subplots();
}

/// Appends offscreen plot primitives captured from a rendered [`crate::Plot`] widget.
pub fn append_last_plot_primitives(primitives: &[ExportPrimitive]) {
    LAST_PLOT_PRIMITIVES.with(|p| {
        p.borrow_mut().extend(primitives.iter().cloned());
    });
}

/// Returns all offscreen plot primitives captured across [`crate::Plot`] widgets in the current frame.
pub fn get_last_plot_primitives() -> Vec<ExportPrimitive> {
    LAST_PLOT_PRIMITIVES.with(|p| p.borrow().clone())
}

/// Clears thread-local [`ExportSubplotPanel`] records captured during the current frame.
pub fn clear_last_plot_subplots() {
    LAST_PLOT_SUBPLOTS.with(|s| s.borrow_mut().clear());
}

/// Appends an [`ExportSubplotPanel`] captured from a rendered [`crate::Plot`] widget.
pub fn append_last_plot_subplot(panel: ExportSubplotPanel) {
    LAST_PLOT_SUBPLOTS.with(|s| {
        s.borrow_mut().push(panel);
    });
}

/// Returns all [`ExportSubplotPanel`] records captured across [`crate::Plot`] widgets in the current frame.
pub fn get_last_plot_subplots() -> Vec<ExportSubplotPanel> {
    LAST_PLOT_SUBPLOTS.with(|s| s.borrow().clone())
}

/// Prefixes `series_name` with `subplot_id` if not already prefixed.
#[must_use]
pub fn prefix_subplot_series_name(subplot_id: &str, series_name: &str, idx: usize) -> String {
    let trimmed_id = subplot_id.trim();
    let trimmed_series = series_name.trim();
    let base = if trimmed_series.is_empty() {
        format!("series_{idx}")
    } else {
        trimmed_series.to_string()
    };
    if trimmed_id.is_empty() {
        return base;
    }
    let colon_prefix = format!("{trimmed_id}:");
    let under_prefix = format!("{trimmed_id}_");
    if base.starts_with(&colon_prefix) || base.starts_with(&under_prefix) {
        base
    } else {
        format!("{trimmed_id}: {base}")
    }
}

pub(crate) fn panel_export_options(
    panel: &ExportSubplotPanel,
    base: &ExportOptions,
) -> ExportOptions {
    ExportOptions {
        log_x: base.log_x || panel.log_x,
        log_y: base.log_y || panel.log_y,
        ..base.clone()
    }
}

/// Partitions a `canvas_w x canvas_h` target canvas into disjoint proportional [`SubplotViewport`]s.
#[must_use]
pub fn compute_subplot_viewports(
    panels: &[ExportSubplotPanel],
    canvas_w: usize,
    canvas_h: usize,
    scale: f32,
) -> Vec<SubplotViewport> {
    if panels.is_empty() {
        return Vec::new();
    }
    if panels.len() == 1 {
        return vec![SubplotViewport {
            x: 0,
            y: 0,
            width: canvas_w.max(1),
            height: canvas_h.max(1),
        }];
    }
    let gap = ((6.0 * scale).round() as usize).clamp(2, 16);
    if has_distinct_screen_rects(panels) {
        let mut union = egui::Rect::NOTHING;
        for p in panels {
            union = union.union(p.screen_rect);
        }
        let uw = (union.width() as f64).max(1.0);
        let uh = (union.height() as f64).max(1.0);
        let mut out = Vec::with_capacity(panels.len());
        for p in panels {
            let rx0 = ((p.screen_rect.min.x - union.min.x) as f64 / uw * canvas_w as f64).round();
            let rx1 = ((p.screen_rect.max.x - union.min.x) as f64 / uw * canvas_w as f64).round();
            let ry0 = ((p.screen_rect.min.y - union.min.y) as f64 / uh * canvas_h as f64).round();
            let ry1 = ((p.screen_rect.max.y - union.min.y) as f64 / uh * canvas_h as f64).round();
            let x0 = (rx0 as usize).min(canvas_w.saturating_sub(2));
            let y0 = (ry0 as usize).min(canvas_h.saturating_sub(2));
            let x1 = (rx1 as usize).clamp(x0 + 1, canvas_w);
            let y1 = (ry1 as usize).clamp(y0 + 1, canvas_h);
            out.push(SubplotViewport {
                x: x0,
                y: y0,
                width: (x1 - x0).max(1),
                height: (y1 - y0).max(1),
            });
        }
        if viewports_are_disjoint(&out) {
            return out;
        }
    }
    fallback_grid_viewports(panels.len(), canvas_w.max(1), canvas_h.max(1), gap)
}

fn has_distinct_screen_rects(panels: &[ExportSubplotPanel]) -> bool {
    for (i, a) in panels.iter().enumerate() {
        let r = a.screen_rect;
        if !r.min.x.is_finite()
            || !r.min.y.is_finite()
            || !r.max.x.is_finite()
            || !r.max.y.is_finite()
            || r.width() <= 1.0
            || r.height() <= 1.0
        {
            return false;
        }
        for b in &panels[i + 1..] {
            let inter = r.intersect(b.screen_rect);
            if inter.width() > 1.0 && inter.height() > 1.0 {
                let inter_area = inter.width() * inter.height();
                let min_area =
                    (r.width() * r.height()).min(b.screen_rect.width() * b.screen_rect.height());
                if inter_area > 0.1 * min_area {
                    return false;
                }
            }
        }
    }
    true
}

fn viewports_are_disjoint(viewports: &[SubplotViewport]) -> bool {
    for (i, a) in viewports.iter().enumerate() {
        for b in &viewports[i + 1..] {
            let x_overlap = a.x < b.x + b.width && b.x < a.x + a.width;
            let y_overlap = a.y < b.y + b.height && b.y < a.y + a.height;
            if x_overlap && y_overlap {
                return false;
            }
        }
    }
    true
}

fn fallback_grid_viewports(
    count: usize,
    canvas_w: usize,
    canvas_h: usize,
    gap: usize,
) -> Vec<SubplotViewport> {
    let cols = if count == 2 {
        2
    } else {
        (count as f64).sqrt().ceil() as usize
    };
    let rows = count.div_ceil(cols);
    let total_gap_x = gap * cols.saturating_sub(1);
    let total_gap_y = gap * rows.saturating_sub(1);
    let cell_w = canvas_w.saturating_sub(total_gap_x).div_ceil(cols).max(1);
    let cell_h = canvas_h.saturating_sub(total_gap_y).div_ceil(rows).max(1);
    let mut out = Vec::with_capacity(count);
    for idx in 0..count {
        let row = idx / cols;
        let col = idx % cols;
        let is_last_lone = idx == count - 1 && col == 0 && count > 2 && cols > 1;
        let x = (col * (cell_w + gap)).min(canvas_w.saturating_sub(1));
        let y = (row * (cell_h + gap)).min(canvas_h.saturating_sub(1));
        let width = if is_last_lone {
            canvas_w.saturating_sub(x).max(1)
        } else {
            cell_w.min(canvas_w.saturating_sub(x)).max(1)
        };
        let height = cell_h.min(canvas_h.saturating_sub(y)).max(1);
        out.push(SubplotViewport {
            x,
            y,
            width,
            height,
        });
    }
    out
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
    pub(crate) draw_x: f64,
    pub(crate) draw_y: f64,
    pub(crate) draw_w: f64,
    pub(crate) draw_h: f64,
    vp_bottom: f64,
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
        let padding = (20.0 * options.scale as f64).clamp(10.0, 100.0);
        Self::build(
            width,
            height,
            SubplotViewport {
                x: 0,
                y: 0,
                width,
                height,
            },
            padding,
            raw_bounds,
            options,
        )
    }

    pub(crate) fn new_subplot(
        vp_width: usize,
        vp_height: usize,
        raw_bounds: [f64; 4],
        options: &ExportOptions,
    ) -> Self {
        let min_dim = vp_width.min(vp_height) as f64;
        let padding = (14.0 * options.scale as f64)
            .clamp(6.0, 48.0)
            .min((min_dim * 0.15).max(2.0));
        Self::build(
            vp_width,
            vp_height,
            SubplotViewport {
                x: 0,
                y: 0,
                width: vp_width,
                height: vp_height,
            },
            padding,
            raw_bounds,
            options,
        )
    }

    pub(crate) fn new_viewport(
        canvas_width: usize,
        canvas_height: usize,
        vp: SubplotViewport,
        raw_bounds: [f64; 4],
        options: &ExportOptions,
    ) -> Self {
        let min_dim = vp.width.min(vp.height) as f64;
        let padding = (14.0 * options.scale as f64)
            .clamp(6.0, 48.0)
            .min((min_dim * 0.15).max(2.0));
        Self::build(
            canvas_width,
            canvas_height,
            vp,
            padding,
            raw_bounds,
            options,
        )
    }

    fn build(
        width: usize,
        height: usize,
        vp: SubplotViewport,
        padding: f64,
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
        let [min_x, max_x, min_y, max_y] = padded_bounds(projected);
        let draw_x = vp.x as f64 + padding;
        let draw_y = vp.y as f64 + padding;
        let draw_w = (vp.width as f64 - 2.0 * padding).max(1.0);
        let draw_h = (vp.height as f64 - 2.0 * padding).max(1.0);
        let vp_bottom = (vp.y + vp.height) as f64 - padding;
        Self {
            width,
            height,
            draw_x,
            draw_y,
            draw_w,
            draw_h,
            vp_bottom,
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
        let px = self.draw_x + x_norm * self.draw_w;
        let py = self.vp_bottom - y_norm * self.draw_h;
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

/// Renders `primitives` offscreen and encodes the resulting canvas as PNG bytes.
pub fn export_primitives_png(
    primitives: &[ExportPrimitive],
    options: &ExportOptions,
) -> Result<Vec<u8>, String> {
    let image = render_primitives_image(primitives, options)?;
    color32_image_to_png(&image)
}

/// Scales a raster `ColorImage` to `options` canvas dimensions and encodes it as PNG bytes.
pub fn export_raster_png(
    raster: &egui::ColorImage,
    options: &ExportOptions,
) -> Result<Vec<u8>, String> {
    if raster.width() == 0 || raster.height() == 0 || raster.pixels.is_empty() {
        return Err("Raster image is empty".to_string());
    }
    let (width, height) = resolve_canvas_dimensions(options)?;
    let scaled = scale_color_image(raster, width, height, options);
    color32_image_to_png(&scaled)
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

/// Decodes PNG bytes into `(width, height, rgba_pixels)` for offscreen verification.
pub fn decode_png_rgba(png_bytes: &[u8]) -> Result<(usize, usize, Vec<[u8; 4]>), String> {
    let img = image::load_from_memory_with_format(png_bytes, image::ImageFormat::Png)
        .map_err(|e| format!("Failed to decode PNG: {e}"))?
        .to_rgba8();
    let (w, h) = img.dimensions();
    let pixels = img.pixels().map(|p| p.0).collect();
    Ok((w as usize, h as usize, pixels))
}
