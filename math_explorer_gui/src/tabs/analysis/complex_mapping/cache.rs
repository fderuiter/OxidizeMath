#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::ui::CONTOUR_RESOLUTION;
use super::{COMPLEX_COORD_MAX, COMPLEX_COORD_MIN, ComplexMappingTool, MappingFunction};
use eframe::egui;
use egui::emath::Float as _;
use egui_plot::{PlotBounds, PlotGeometry, PlotItem, PlotItemBase, PlotPoint};
use num_complex::Complex64;
use std::hash::{Hash as _, Hasher as _};
use std::ops::{Range, RangeInclusive};
use std::sync::Arc;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct ConformalCacheKey {
    function: MappingFunction,
    grid_min_bits: u64,
    grid_max_bits: u64,
    grid_density: usize,
    math_fingerprint: u64,
}

pub(super) struct ContourFamily {
    pub(super) points: Arc<Vec<PlotPoint>>,
    pub(super) ranges: Vec<Range<usize>>,
    pub(super) bounds: PlotBounds,
}

impl Default for ContourFamily {
    fn default() -> Self {
        Self {
            points: Arc::new(Vec::new()),
            ranges: Vec::new(),
            bounds: PlotBounds::NOTHING,
        }
    }
}

impl ContourFamily {
    fn clear(&mut self) {
        Arc::make_mut(&mut self.points).clear();
        self.ranges.clear();
        self.bounds = PlotBounds::NOTHING;
    }

    fn reserve(&mut self, contours: usize, point_cap: usize) {
        self.ranges.reserve(contours);
        Arc::make_mut(&mut self.points).reserve(point_cap);
    }

    pub(super) fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }

    fn push_line_segment(&mut self, a: [f64; 2], b: [f64; 2]) {
        let pts = Arc::make_mut(&mut self.points);
        let start = pts.len();
        let pa = PlotPoint::new(a[0], a[1]);
        let pb = PlotPoint::new(b[0], b[1]);
        self.bounds.extend_with(&pa);
        self.bounds.extend_with(&pb);
        pts.push(pa);
        pts.push(pb);
        self.ranges.push(start..start + 2);
    }

    fn append_sampled_contour<F: Fn(usize) -> Complex64>(
        &mut self,
        resolution: usize,
        grid_span: f64,
        sample: F,
    ) {
        let max_radius = (grid_span.abs() * 12.0).max(25.0);
        let max_jump = (max_radius * 0.6).max(12.0);
        let pts = Arc::make_mut(&mut self.points);
        let mut seg_start = pts.len();
        let mut prev: Option<PlotPoint> = None;

        for j in 0..resolution {
            let w = sample(j);
            if !w.re.is_finite() || !w.im.is_finite() || w.norm() > max_radius {
                Self::finish_run(pts, &mut self.ranges, &mut self.bounds, seg_start);
                seg_start = pts.len();
                prev = None;
                continue;
            }
            if let Some(p) = prev {
                if (w.re - p.x).hypot(w.im - p.y) > max_jump {
                    Self::finish_run(pts, &mut self.ranges, &mut self.bounds, seg_start);
                    seg_start = pts.len();
                }
            }
            let pt = PlotPoint::new(w.re, w.im);
            pts.push(pt);
            prev = Some(pt);
        }
        Self::finish_run(pts, &mut self.ranges, &mut self.bounds, seg_start);
    }

    fn finish_run(
        pts: &mut Vec<PlotPoint>,
        ranges: &mut Vec<Range<usize>>,
        bounds: &mut PlotBounds,
        seg_start: usize,
    ) {
        let end = pts.len();
        if end - seg_start >= 2 {
            for pt in &pts[seg_start..end] {
                bounds.extend_with(pt);
            }
            ranges.push(seg_start..end);
        } else {
            pts.truncate(seg_start);
        }
    }
}

#[derive(Default)]
pub(crate) struct ConformalGridCache {
    key: Option<ConformalCacheKey>,
    pub(super) generation: u64,
    pub(super) z_vertical: ContourFamily,
    pub(super) z_horizontal: ContourFamily,
    pub(super) w_vertical: ContourFamily,
    pub(super) w_horizontal: ContourFamily,
    pub(crate) eval_count: usize,
}

impl ConformalGridCache {
    pub(super) fn clear(&mut self) {
        if self.key.is_some()
            || !self.z_vertical.is_empty()
            || !self.z_horizontal.is_empty()
            || !self.w_vertical.is_empty()
            || !self.w_horizontal.is_empty()
        {
            self.generation = self.generation.wrapping_add(1);
        }
        self.key = None;
        self.z_vertical.clear();
        self.z_horizontal.clear();
        self.w_vertical.clear();
        self.w_horizontal.clear();
    }
}

pub(super) struct ContourBatch<'a> {
    base: PlotItemBase,
    shared_points: &'a Arc<Vec<PlotPoint>>,
    ranges: &'a [Range<usize>],
    bounds: PlotBounds,
    stroke: egui::Stroke,
    generation: u64,
}

impl<'a> ContourBatch<'a> {
    pub(super) fn new(
        name: impl Into<String>,
        family: &'a ContourFamily,
        color: egui::Color32,
        generation: u64,
    ) -> Self {
        Self {
            base: PlotItemBase::new(name.into()),
            shared_points: &family.points,
            ranges: family.ranges.as_slice(),
            bounds: family.bounds,
            stroke: egui::Stroke::new(1.5_f32, color),
            generation,
        }
    }
}

fn flush_valid_run(
    p0: &mut Option<egui::Pos2>,
    p1: &mut Option<egui::Pos2>,
    poly: &mut Vec<egui::Pos2>,
    stroke: egui::Stroke,
    shapes: &mut Vec<egui::Shape>,
) {
    if !poly.is_empty() {
        shapes.push(egui::Shape::line(std::mem::take(poly), stroke));
    } else if let (Some(a), Some(b)) = (*p0, *p1) {
        shapes.push(egui::Shape::line_segment([a, b], stroke));
    }
    *p0 = None;
    *p1 = None;
}

fn emit_contour_shapes(
    contour: &[PlotPoint],
    transform: &egui_plot::PlotTransform,
    stroke: egui::Stroke,
    shapes: &mut Vec<egui::Shape>,
) {
    let mut p0: Option<egui::Pos2> = None;
    let mut p1: Option<egui::Pos2> = None;
    let mut poly: Vec<egui::Pos2> = Vec::new();

    for pt in contour {
        if transform.is_point_valid_for_scale(pt) {
            let pos = transform.position_from_point(pt);
            match (p0, p1) {
                (None, _) => p0 = Some(pos),
                (Some(_), None) => p1 = Some(pos),
                (Some(a), Some(b)) => {
                    if poly.is_empty() {
                        poly.reserve(contour.len());
                        poly.push(a);
                        poly.push(b);
                    }
                    poly.push(pos);
                }
            }
        } else {
            flush_valid_run(&mut p0, &mut p1, &mut poly, stroke, shapes);
        }
    }
    flush_valid_run(&mut p0, &mut p1, &mut poly, stroke, shapes);
}

fn closest_vertex_magnet(
    contour: &[PlotPoint],
    pointer: egui::Pos2,
    transform: &egui_plot::PlotTransform,
) -> Option<(PlotPoint, f32)> {
    const VERTEX_MAGNET_SQ: f32 = 36.0;
    let mut best: Option<(PlotPoint, f32)> = None;
    for &pt in contour {
        if !transform.is_point_valid_for_scale(&pt) {
            continue;
        }
        let s = transform.position_from_point(&pt);
        let d_sq = pointer.distance_sq(s);
        if s.is_finite() && d_sq <= VERTEX_MAGNET_SQ && best.is_none_or(|(_, b)| d_sq < b) {
            best = Some((pt, d_sq));
        }
    }
    best
}

fn closest_point_in_contour(
    contour: &[PlotPoint],
    pointer: egui::Pos2,
    transform: &egui_plot::PlotTransform,
) -> Option<(PlotPoint, f32)> {
    if let Some(hit) = closest_vertex_magnet(contour, pointer, transform) {
        return Some(hit);
    }
    let mut best: Option<(PlotPoint, f32)> = None;
    for pair in contour.windows(2) {
        if !transform.is_point_valid_for_scale(&pair[0])
            || !transform.is_point_valid_for_scale(&pair[1])
        {
            continue;
        }
        let s0 = transform.position_from_point(&pair[0]);
        let s1 = transform.position_from_point(&pair[1]);
        if !s0.is_finite() || !s1.is_finite() {
            continue;
        }
        let seg = s1 - s0;
        let len_sq = seg.length_sq();
        let t = if len_sq <= 1e-12 {
            0.0
        } else {
            ((pointer - s0).dot(seg) / len_sq).clamp(0.0, 1.0)
        };
        let proj = s0 + t * seg;
        let dist_sq = pointer.distance_sq(proj);
        let value = transform.value_from_position(proj);
        if value.x.is_finite() && value.y.is_finite() && best.is_none_or(|(_, b)| dist_sq < b) {
            best = Some((value, dist_sq));
        }
    }
    best
}

impl PlotItem for ContourBatch<'_> {
    fn shapes(
        &self,
        _ui: &egui::Ui,
        transform: &egui_plot::PlotTransform,
        shapes: &mut Vec<egui::Shape>,
    ) {
        let mut stroke = self.stroke;
        if self.highlighted() {
            stroke.width *= 2.0;
        }
        let pts = self.shared_points.as_slice();
        for range in self.ranges {
            if let Some(contour) = pts.get(range.clone()) {
                emit_contour_shapes(contour, transform, stroke, shapes);
            }
        }
    }

    fn initialize(&mut self, _x_range: RangeInclusive<f64>) {}

    fn color(&self) -> egui::Color32 {
        self.stroke.color
    }

    fn extract_points(&self) -> Vec<[f64; 2]> {
        self.shared_points.iter().map(|p| [p.x, p.y]).collect()
    }

    fn append_deferred_export(
        &mut self,
        datasets: &mut Vec<egui_plot::DeferredDatasetEntry>,
        _primitives: &mut Vec<egui_plot::export_primitives::DeferredPlotPrimitive>,
    ) {
        if !self.is_companion_visual_only() && !self.shared_points.is_empty() {
            datasets.push(egui_plot::DeferredDatasetEntry::SharedPoints {
                name: self.name().to_owned(),
                points: Arc::clone(self.shared_points),
            });
        }
    }

    fn item_content_hash(&self) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.id().hash(&mut hasher);
        self.name().hash(&mut hasher);
        self.stroke.color.to_array().hash(&mut hasher);
        self.stroke.width.to_bits().hash(&mut hasher);
        self.allow_hover().hash(&mut hasher);
        self.generation.hash(&mut hasher);
        self.shared_points.len().hash(&mut hasher);
        self.ranges.len().hash(&mut hasher);
        hasher.finish()
    }

    fn geometry(&self) -> PlotGeometry<'_> {
        PlotGeometry::Points(self.shared_points.as_slice())
    }

    fn bounds(&self) -> PlotBounds {
        self.bounds
    }

    fn base(&self) -> &PlotItemBase {
        &self.base
    }

    fn base_mut(&mut self) -> &mut PlotItemBase {
        &mut self.base
    }

    fn closest_continuous_point(
        &self,
        pointer: egui::Pos2,
        transform: &egui_plot::PlotTransform,
    ) -> Option<(PlotPoint, f32)> {
        let pts = self.shared_points.as_slice();
        self.ranges
            .iter()
            .filter_map(|range| {
                closest_point_in_contour(pts.get(range.clone())?, pointer, transform)
            })
            .min_by_key(|(_, d)| d.ord())
    }
}

impl ComplexMappingTool {
    pub(crate) fn contour_cache_key(&self) -> ConformalCacheKey {
        let mut mh = std::collections::hash_map::DefaultHasher::new();
        self.math_input.raw_text.hash(&mut mh);
        self.math_input.state_fingerprint().hash(&mut mh);
        ConformalCacheKey {
            function: self.function,
            grid_min_bits: self.grid_min.to_bits(),
            grid_max_bits: self.grid_max.to_bits(),
            grid_density: self.grid_density,
            math_fingerprint: mh.finish(),
        }
    }

    pub(crate) fn compute_state_revision(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        (self.function as u8).hash(&mut h);
        self.grid_min.to_bits().hash(&mut h);
        self.grid_max.to_bits().hash(&mut h);
        self.grid_density.hash(&mut h);
        self.inspection_point[0].to_bits().hash(&mut h);
        self.inspection_point[1].to_bits().hash(&mut h);
        self.math_input.raw_text.hash(&mut h);
        self.math_input.state_fingerprint().hash(&mut h);
        h.finish().max(1)
    }

    pub(super) fn has_valid_grid_bounds(&self) -> bool {
        self.grid_min.is_finite()
            && self.grid_max.is_finite()
            && (COMPLEX_COORD_MIN..=COMPLEX_COORD_MAX).contains(&self.grid_min)
            && (COMPLEX_COORD_MIN..=COMPLEX_COORD_MAX).contains(&self.grid_max)
            && self.grid_min < self.grid_max
            && (2..=crate::presets::MAX_INTERACTIVE_GRID_SIZE).contains(&self.grid_density)
    }

    pub(super) fn ensure_cached_contours(&mut self) -> bool {
        if !self.has_valid_grid_bounds() {
            self.cache.clear();
            return false;
        }
        let key = self.contour_cache_key();
        if self.cache.key != Some(key) {
            let mut cache = std::mem::take(&mut self.cache);
            self.populate_cache(&mut cache, key);
            cache.eval_count += 1;
            self.cache = cache;
        }
        true
    }

    fn populate_cache(&self, cache: &mut ConformalGridCache, key: ConformalCacheKey) {
        cache.z_vertical.clear();
        cache.z_horizontal.clear();
        cache.w_vertical.clear();
        cache.w_horizontal.clear();
        cache.z_vertical.reserve(self.grid_density, self.grid_density * 2);
        cache.z_horizontal.reserve(self.grid_density, self.grid_density * 2);
        cache.w_vertical.reserve(self.grid_density, self.grid_density * CONTOUR_RESOLUTION);
        cache.w_horizontal.reserve(self.grid_density, self.grid_density * CONTOUR_RESOLUTION);

        let span = self.grid_max - self.grid_min;
        let step = span / (self.grid_density as f64 - 1.0);
        let denom = (CONTOUR_RESOLUTION - 1) as f64;

        for i in 0..self.grid_density {
            let x = self.grid_min + (i as f64) * step;
            cache.z_vertical.push_line_segment([x, self.grid_min], [x, self.grid_max]);
            cache.w_vertical.append_sampled_contour(CONTOUR_RESOLUTION, span, |j| {
                self.eval_z(Complex64::new(x, self.grid_min + span * (j as f64) / denom))
            });
        }
        for i in 0..self.grid_density {
            let y = self.grid_min + (i as f64) * step;
            cache.z_horizontal.push_line_segment([self.grid_min, y], [self.grid_max, y]);
            cache.w_horizontal.append_sampled_contour(CONTOUR_RESOLUTION, span, |j| {
                self.eval_z(Complex64::new(self.grid_min + span * (j as f64) / denom, y))
            });
        }
        cache.key = Some(key);
        cache.generation = cache.generation.wrapping_add(1).max(1);
    }

    pub(super) fn resolve_export_cache<'a>(
        &'a self,
        cold: &'a mut ConformalGridCache,
    ) -> Option<&'a ConformalGridCache> {
        if !self.has_valid_grid_bounds() {
            return None;
        }
        let key = self.contour_cache_key();
        if self.cache.key == Some(key) {
            Some(&self.cache)
        } else {
            self.populate_cache(cold, key);
            Some(cold)
        }
    }
}
