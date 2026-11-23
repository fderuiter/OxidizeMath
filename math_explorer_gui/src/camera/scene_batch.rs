#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Consolidated segment and marker batch builders for [`super::Scene3DBuilder`].

use std::sync::Arc;

use eframe::egui;
use egui_plot::{
    ColoredPoint, ColoredSegment, Line, LineStyle, MarkerShape, Plot3DSeriesSidecar, PlotBounds,
    PlotPoint, PlotUi, Points, Text,
};

use super::{Drawable3D, LabelPrimitive3D, PointPrimitive3D, SegmentPrimitive3D};

#[derive(Clone, Debug)]
pub(super) struct CachedSegmentBatch {
    pub(super) segments: Arc<Vec<ColoredSegment>>,
    pub(super) world_vertices: Arc<Vec<[f64; 3]>>,
    pub(super) bounds: PlotBounds,
    pub(super) name: String,
    pub(super) style: LineStyle,
    pub(super) allow_hover: bool,
}

#[derive(Clone, Debug)]
pub(super) struct CachedPointBatch {
    pub(super) points: Arc<Vec<ColoredPoint>>,
    pub(super) world_vertices: Arc<Vec<[f64; 3]>>,
    pub(super) bounds: PlotBounds,
    pub(super) name: String,
    pub(super) shape: MarkerShape,
    pub(super) allow_hover: bool,
}

struct SegmentAcc {
    segments: Vec<ColoredSegment>,
    world: Vec<[f64; 3]>,
    bounds: PlotBounds,
    name: String,
    style: LineStyle,
    allow_hover: bool,
}

impl SegmentAcc {
    fn new() -> Self {
        Self {
            segments: Vec::new(),
            world: Vec::new(),
            bounds: PlotBounds::NOTHING,
            name: String::new(),
            style: LineStyle::Solid,
            allow_hover: false,
        }
    }

    fn push(&mut self, seg: &SegmentPrimitive3D) {
        if self.name.is_empty() && !seg.name.is_empty() {
            self.name.clone_from(&seg.name);
            self.style = seg.style;
        }
        self.allow_hover |= seg.allow_hover;
        let a = PlotPoint::new(seg.p0_proj[0], seg.p0_proj[1]);
        let b = PlotPoint::new(seg.p1_proj[0], seg.p1_proj[1]);
        self.bounds.extend_with(&a);
        self.bounds.extend_with(&b);
        self.segments.push(ColoredSegment::new(
            vec![a, b],
            seg.stroke.color,
            seg.stroke.width,
        ));
        self.world.push(seg.p0_world);
        self.world.push(seg.p1_world);
    }

    fn finish(self) -> Option<CachedSegmentBatch> {
        (!self.segments.is_empty()).then(|| CachedSegmentBatch {
            segments: Arc::new(self.segments),
            world_vertices: Arc::new(self.world),
            bounds: self.bounds,
            name: self.name,
            style: self.style,
            allow_hover: self.allow_hover,
        })
    }
}

struct PointAcc {
    points: Vec<ColoredPoint>,
    world: Vec<[f64; 3]>,
    bounds: PlotBounds,
    name: String,
    shape: MarkerShape,
    allow_hover: bool,
}

impl PointAcc {
    fn new() -> Self {
        Self {
            points: Vec::new(),
            world: Vec::new(),
            bounds: PlotBounds::NOTHING,
            name: String::new(),
            shape: MarkerShape::Circle,
            allow_hover: false,
        }
    }

    fn push(&mut self, pt: &PointPrimitive3D) {
        if self.name.is_empty() && !pt.name.is_empty() {
            self.name.clone_from(&pt.name);
            self.shape = pt.shape;
        }
        self.allow_hover |= pt.allow_hover;
        let p = PlotPoint::new(pt.proj[0], pt.proj[1]);
        self.bounds.extend_with(&p);
        self.points.push(ColoredPoint::new(p, pt.color, pt.radius));
        self.world.push(pt.world);
    }

    fn finish(self) -> Option<CachedPointBatch> {
        (!self.points.is_empty()).then(|| CachedPointBatch {
            points: Arc::new(self.points),
            world_vertices: Arc::new(self.world),
            bounds: self.bounds,
            name: self.name,
            shape: self.shape,
            allow_hover: self.allow_hover,
        })
    }
}

pub(super) fn build_consolidated_batches(
    primitives: &[Drawable3D],
    order: &[usize],
) -> (Option<CachedSegmentBatch>, Option<CachedPointBatch>) {
    let mut seg_acc = SegmentAcc::new();
    let mut pt_acc = PointAcc::new();
    for &idx in order {
        match &primitives[idx] {
            Drawable3D::Segment(seg) => seg_acc.push(seg),
            Drawable3D::Point(pt) => pt_acc.push(pt),
        }
    }
    (seg_acc.finish(), pt_acc.finish())
}

pub(super) fn emit_consolidated_batches(
    plot_ui: &mut PlotUi<'_>,
    seg_batch: Option<&CachedSegmentBatch>,
    pt_batch: Option<&CachedPointBatch>,
    labels: &[LabelPrimitive3D],
    sidecar: &Plot3DSeriesSidecar,
    emit_sidecar: bool,
) {
    let plot_id = plot_ui.plot_id();
    if let Some(batch) = seg_batch {
        let item_id = plot_id.with("scene3d_segments");
        plot_ui.line(
            Line::from_colored_segments(batch.name.clone(), Arc::clone(&batch.segments))
                .id(item_id)
                .with_bounds(batch.bounds)
                .style(batch.style)
                .allow_hover(batch.allow_hover),
        );
    }
    if let Some(batch) = pt_batch {
        let item_id = plot_id.with("scene3d_points");
        plot_ui.points(
            Points::from_colored_points(batch.name.clone(), Arc::clone(&batch.points))
                .id(item_id)
                .with_bounds(batch.bounds)
                .shape(batch.shape)
                .allow_hover(batch.allow_hover),
        );
    }
    for lbl in labels {
        plot_ui.text(
            Text::new(
                "",
                PlotPoint::new(lbl.proj[0], lbl.proj[1]),
                egui::RichText::new(&lbl.text).strong().color(lbl.color),
            )
            .allow_hover(false),
        );
    }
    if emit_sidecar && sidecar.has_any_data() {
        plot_ui.set_3d_sidecar(sidecar.clone());
    }
}
