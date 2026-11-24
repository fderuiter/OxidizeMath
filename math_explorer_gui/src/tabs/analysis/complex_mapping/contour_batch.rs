#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Zero-copy `egui_plot::PlotItem` batch renderer for conformal contour families.

use super::ContourFamily;
use eframe::egui;
use egui::emath::Float as _;
use egui_plot::{PlotBounds, PlotGeometry, PlotItem, PlotItemBase, PlotPoint};
use std::hash::{Hash as _, Hasher as _};
use std::ops::{Range, RangeInclusive};
use std::sync::Arc;

pub(in crate::tabs::analysis::complex_mapping) struct ContourBatch<'a> {
    base: PlotItemBase,
    shared_points: &'a Arc<Vec<PlotPoint>>,
    ranges: &'a [Range<usize>],
    bounds: PlotBounds,
    stroke: egui::Stroke,
    generation: u64,
}

impl<'a> ContourBatch<'a> {
    pub(in crate::tabs::analysis::complex_mapping) fn new(
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
