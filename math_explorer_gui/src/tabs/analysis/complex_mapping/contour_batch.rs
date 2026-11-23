#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::cache::ContourFamily;
use eframe::egui;
use egui::emath::Float as _;
use egui_plot::{PlotBounds, PlotGeometry, PlotItem, PlotItemBase, PlotPoint};
use std::hash::{Hash as _, Hasher as _};
use std::ops::RangeInclusive;
use std::sync::Arc;

#[derive(Clone, Copy)]
struct FamilySlice<'a> {
    label: Option<&'a str>,
    family: &'a ContourFamily,
    color: egui::Color32,
}

pub(super) struct ContourBatch<'a> {
    base: PlotItemBase,
    families: [Option<FamilySlice<'a>>; 2],
    bounds: PlotBounds,
    generation: u64,
}

impl<'a> ContourBatch<'a> {
    #[cfg(test)]
    pub(super) fn new(
        name: impl Into<String>,
        family: &'a ContourFamily,
        color: egui::Color32,
        generation: u64,
    ) -> Self {
        Self {
            base: PlotItemBase::new(name.into()),
            families: [
                Some(FamilySlice {
                    label: None,
                    family,
                    color,
                }),
                None,
            ],
            bounds: family.bounds,
            generation,
        }
    }

    pub(super) fn from_pair(
        name: impl Into<String>,
        first: (&'a str, &'a ContourFamily, egui::Color32),
        second: (&'a str, &'a ContourFamily, egui::Color32),
        generation: u64,
    ) -> Option<Self> {
        let f0 = (!first.1.is_empty()).then_some(FamilySlice {
            label: Some(first.0),
            family: first.1,
            color: first.2,
        });
        let f1 = (!second.1.is_empty()).then_some(FamilySlice {
            label: Some(second.0),
            family: second.1,
            color: second.2,
        });
        if f0.is_none() && f1.is_none() {
            return None;
        }
        let mut bounds = PlotBounds::NOTHING;
        for entry in [f0, f1].into_iter().flatten() {
            bounds.merge(&entry.family.bounds);
        }
        Some(Self {
            base: PlotItemBase::new(name.into()),
            families: [f0, f1],
            bounds,
            generation,
        })
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
        let width = if self.highlighted() { 3.0_f32 } else { 1.5_f32 };
        for entry in self.families.iter().flatten() {
            let stroke = egui::Stroke::new(width, entry.color);
            let pts = entry.family.points.as_slice();
            for range in entry.family.ranges.iter() {
                if let Some(contour) = pts.get(range.clone()) {
                    emit_contour_shapes(contour, transform, stroke, shapes);
                }
            }
        }
    }

    fn initialize(&mut self, _x_range: RangeInclusive<f64>) {}

    fn color(&self) -> egui::Color32 {
        self.families
            .iter()
            .flatten()
            .next()
            .map_or(egui::Color32::TRANSPARENT, |e| e.color)
    }

    fn point_count(&self) -> usize {
        self.families
            .iter()
            .flatten()
            .map(|e| e.family.points.len())
            .sum()
    }

    fn extract_points(&self) -> Vec<[f64; 2]> {
        let mut out = Vec::with_capacity(self.point_count());
        for entry in self.families.iter().flatten() {
            out.extend(entry.family.points.iter().map(|p| [p.x, p.y]));
        }
        out
    }

    fn append_deferred_export(
        &mut self,
        datasets: &mut Vec<egui_plot::DeferredDatasetEntry>,
        primitives: &mut Vec<egui_plot::export_primitives::DeferredPlotPrimitive>,
    ) {
        if self.is_companion_visual_only() {
            return;
        }
        let fallback_name = self.name().to_owned();
        for entry in self.families.iter().flatten() {
            if entry.family.points.is_empty() {
                continue;
            }
            let series_name = entry.label.unwrap_or(&fallback_name).to_owned();
            datasets.push(egui_plot::DeferredDatasetEntry::SharedPoints {
                name: series_name.clone(),
                points: Arc::clone(&entry.family.points),
            });
            primitives.push(
                egui_plot::export_primitives::DeferredPlotPrimitive::SlicedPolylineBatch {
                    name: series_name,
                    points: Arc::clone(&entry.family.points),
                    ranges: Arc::clone(&entry.family.ranges),
                    color: entry.color,
                    stroke_width: 1.5,
                },
            );
        }
    }

    fn item_content_hash(&self) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.id().hash(&mut hasher);
        self.name().hash(&mut hasher);
        self.allow_hover().hash(&mut hasher);
        self.generation.hash(&mut hasher);
        for entry in self.families.iter().flatten() {
            entry.label.hash(&mut hasher);
            entry.color.to_array().hash(&mut hasher);
            entry.family.points.len().hash(&mut hasher);
            entry.family.ranges.len().hash(&mut hasher);
        }
        hasher.finish()
    }

    fn geometry(&self) -> PlotGeometry<'_> {
        self.families
            .iter()
            .flatten()
            .next()
            .map_or(PlotGeometry::None, |e| {
                PlotGeometry::Points(e.family.points.as_slice())
            })
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
        self.families
            .iter()
            .flatten()
            .flat_map(|entry| {
                let pts = entry.family.points.as_slice();
                entry.family.ranges.iter().filter_map(move |range| {
                    closest_point_in_contour(pts.get(range.clone())?, pointer, transform)
                })
            })
            .min_by_key(|(_, d)| d.ord())
    }
}
