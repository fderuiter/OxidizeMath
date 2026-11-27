//! Single-`PlotItem` batched renderer for depth-sorted 3D scenes produced by [`super::Scene3DBuilder`].

#[path = "scene_paint.rs"]
mod scene_paint;

use std::collections::HashMap;
use std::hash::{Hash as _, Hasher as _};
use std::ops::RangeInclusive;
use std::sync::Arc;

use eframe::egui::{self, Color32, Pos2, Shape, Stroke, Ui};
use egui::emath::Float as _;
use egui_plot::{
    render_snapped_hover_point_named, ClosestElem, Cursor, DeferredPlotPrimitive, ExportPrimitive,
    LabelFormatter, LazyPlotPrimitives, LineStyle, MarkerShape, PlotBounds, PlotConfig,
    PlotGeometry, PlotItem, PlotItemBase, PlotPoint, PlotTransform, SharedPoints,
};

use scene_paint::{materialize_scene_exports, paint_draw_op, paint_label_op};

#[derive(Clone, Debug)]
pub(super) struct SegmentPrimitive3D {
    pub name: String,
    pub p0_proj: [f64; 2],
    pub p1_proj: [f64; 2],
    pub p0_world: [f64; 3],
    pub p1_world: [f64; 3],
    pub viewer_z: f64,
    pub stroke: Stroke,
    pub style: LineStyle,
    pub allow_hover: bool,
}

#[derive(Clone, Debug)]
pub(super) struct PointPrimitive3D {
    pub name: String,
    pub proj: [f64; 2],
    pub world: [f64; 3],
    pub viewer_z: f64,
    pub radius: f32,
    pub color: Color32,
    pub shape: MarkerShape,
    pub allow_hover: bool,
}

#[derive(Clone, Debug)]
pub(super) enum Drawable3D {
    Segment(SegmentPrimitive3D),
    Point(PointPrimitive3D),
}

impl Drawable3D {
    pub(super) fn viewer_z(&self) -> f64 {
        match self {
            Self::Segment(s) => s.viewer_z,
            Self::Point(p) => p.viewer_z,
        }
    }
}

#[derive(Clone, Debug)]
pub(super) struct LabelPrimitive3D {
    pub proj: [f64; 2],
    pub text: String,
    pub color: Color32,
}

#[derive(Clone, Debug)]
pub(super) enum CompiledDrawOp {
    Segment {
        p0: PlotPoint,
        p1: PlotPoint,
        stroke: Stroke,
        style: LineStyle,
        name_idx: u32,
    },
    Point {
        pos: PlotPoint,
        radius: f32,
        color: Color32,
        shape: MarkerShape,
        name_idx: u32,
    },
}

#[derive(Clone, Debug)]
struct HoverVertex3D {
    projected: PlotPoint,
    world: [f64; 3],
    color: Color32,
    name_idx: u32,
}

#[derive(Clone, Debug)]
pub(super) struct CompiledScenePayload {
    pub names: Vec<String>,
    pub draw_ops: Vec<CompiledDrawOp>,
    pub labels: Vec<LabelPrimitive3D>,
    hover_vertices: Vec<HoverVertex3D>,
    bounds: PlotBounds,
    primary_color: Color32,
    content_hash: u64,
}

#[derive(Clone, Debug)]
pub(super) struct CompiledScene3D {
    payload: Arc<CompiledScenePayload>,
    lazy_export: LazyPlotPrimitives,
}

pub(super) const SCENE_BATCH_ID_KEY: &str = "__scene3d_batched_item__";

pub(super) fn batch_item_id(plot_id: egui::Id) -> egui::Id {
    plot_id.with(SCENE_BATCH_ID_KEY)
}

fn intern_name(names: &mut Vec<String>, lookup: &mut HashMap<String, u32>, raw: &str) -> u32 {
    if let Some(&idx) = lookup.get(raw) {
        return idx;
    }
    let idx = names.len() as u32;
    names.push(raw.to_owned());
    lookup.insert(raw.to_owned(), idx);
    idx
}

pub(super) fn compile_sorted_scene(
    primitives: &[Drawable3D],
    order: &[usize],
    labels: &[LabelPrimitive3D],
    generation: u64,
) -> Option<(CompiledScene3D, Vec<[f64; 3]>)> {
    if order.is_empty() && labels.is_empty() {
        return None;
    }
    let mut names = Vec::with_capacity(8);
    let mut name_map = HashMap::with_capacity(8);
    intern_name(&mut names, &mut name_map, "");

    let (draw_ops, bounds, primary_color) =
        build_draw_ops_and_bounds(primitives, order, labels, &mut names, &mut name_map);
    let (hover_vertices, world_vertices) =
        collect_front_to_back_hover_vertices(primitives, order, &mut names, &mut name_map);

    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    generation.hash(&mut hasher);
    draw_ops.len().hash(&mut hasher);
    labels.len().hash(&mut hasher);
    let content_hash = hasher.finish().max(1);

    let payload = Arc::new(CompiledScenePayload {
        names,
        draw_ops,
        labels: labels.to_vec(),
        hover_vertices,
        bounds,
        primary_color,
        content_hash,
    });
    let export_payload = Arc::clone(&payload);
    let lazy_export = LazyPlotPrimitives::new(move |out| {
        materialize_scene_exports(&export_payload, out);
    });
    Some((
        CompiledScene3D {
            payload,
            lazy_export,
        },
        world_vertices,
    ))
}

fn build_draw_ops_and_bounds(
    primitives: &[Drawable3D],
    order: &[usize],
    labels: &[LabelPrimitive3D],
    names: &mut Vec<String>,
    name_map: &mut HashMap<String, u32>,
) -> (Vec<CompiledDrawOp>, PlotBounds, Color32) {
    let mut draw_ops = Vec::with_capacity(order.len());
    let mut bounds = PlotBounds::NOTHING;
    let mut primary_color = Color32::TRANSPARENT;

    for &idx in order {
        match &primitives[idx] {
            Drawable3D::Segment(seg) => {
                let p0 = PlotPoint::new(seg.p0_proj[0], seg.p0_proj[1]);
                let p1 = PlotPoint::new(seg.p1_proj[0], seg.p1_proj[1]);
                bounds.extend_with(&p0);
                bounds.extend_with(&p1);
                if primary_color == Color32::TRANSPARENT && seg.stroke.color != Color32::TRANSPARENT
                {
                    primary_color = seg.stroke.color;
                }
                let name_idx = intern_name(names, name_map, &seg.name);
                draw_ops.push(CompiledDrawOp::Segment {
                    p0,
                    p1,
                    stroke: seg.stroke,
                    style: seg.style,
                    name_idx,
                });
            }
            Drawable3D::Point(pt) => {
                let pos = PlotPoint::new(pt.proj[0], pt.proj[1]);
                bounds.extend_with(&pos);
                if primary_color == Color32::TRANSPARENT && pt.color != Color32::TRANSPARENT {
                    primary_color = pt.color;
                }
                let name_idx = intern_name(names, name_map, &pt.name);
                draw_ops.push(CompiledDrawOp::Point {
                    pos,
                    radius: pt.radius,
                    color: pt.color,
                    shape: pt.shape,
                    name_idx,
                });
            }
        }
    }
    for lbl in labels {
        bounds.extend_with(&PlotPoint::new(lbl.proj[0], lbl.proj[1]));
    }
    (draw_ops, bounds, primary_color)
}

fn collect_front_to_back_hover_vertices(
    primitives: &[Drawable3D],
    order: &[usize],
    names: &mut Vec<String>,
    name_map: &mut HashMap<String, u32>,
) -> (Vec<HoverVertex3D>, Vec<[f64; 3]>) {
    let mut hover_vertices = Vec::with_capacity(order.len() * 2);
    let mut world_vertices = Vec::with_capacity(order.len() * 2);
    for &idx in order.iter().rev() {
        match &primitives[idx] {
            Drawable3D::Segment(seg) if seg.allow_hover => {
                let name_idx = intern_name(names, name_map, &seg.name);
                for (proj, world) in [(seg.p0_proj, seg.p0_world), (seg.p1_proj, seg.p1_world)] {
                    hover_vertices.push(HoverVertex3D {
                        projected: PlotPoint::new(proj[0], proj[1]),
                        world,
                        color: seg.stroke.color,
                        name_idx,
                    });
                    world_vertices.push(world);
                }
            }
            Drawable3D::Point(pt) if pt.allow_hover => {
                let name_idx = intern_name(names, name_map, &pt.name);
                hover_vertices.push(HoverVertex3D {
                    projected: PlotPoint::new(pt.proj[0], pt.proj[1]),
                    world: pt.world,
                    color: pt.color,
                    name_idx,
                });
                world_vertices.push(pt.world);
            }
            _ => {}
        }
    }
    (hover_vertices, world_vertices)
}

pub(super) struct Scene3DBatchItem {
    base: PlotItemBase,
    compiled: CompiledScene3D,
}

impl Scene3DBatchItem {
    pub(super) fn new(plot_id: egui::Id, compiled: CompiledScene3D) -> Self {
        let base = PlotItemBase::new(String::new())
            .with_id(batch_item_id(plot_id))
            .with_allow_hover(!compiled.payload.hover_vertices.is_empty());
        Self { base, compiled }
    }
}

impl PlotItem for Scene3DBatchItem {
    fn shapes(&self, ui: &Ui, transform: &PlotTransform, shapes: &mut Vec<Shape>) {
        let payload = &self.compiled.payload;
        shapes.reserve(payload.draw_ops.len() + payload.labels.len());
        let scale_mult = if self.highlighted() {
            2.0_f32.sqrt()
        } else {
            1.0_f32
        };
        let projector = transform.linear_projector();
        for op in &payload.draw_ops {
            paint_draw_op(op, transform, projector.as_ref(), scale_mult, shapes);
        }
        for lbl in &payload.labels {
            paint_label_op(lbl, ui, transform, shapes);
        }
    }

    fn initialize(&mut self, _x_range: RangeInclusive<f64>) {}

    fn color(&self) -> Color32 {
        self.compiled.payload.primary_color
    }

    fn show_in_legend(&self) -> bool {
        false
    }

    fn geometry(&self) -> PlotGeometry<'_> {
        PlotGeometry::None
    }

    fn bounds(&self) -> PlotBounds {
        self.compiled.payload.bounds
    }

    fn has_any_points(&self) -> bool {
        !self.compiled.payload.draw_ops.is_empty()
    }

    fn item_content_hash(&self) -> u64 {
        self.compiled.payload.content_hash
    }

    fn find_closest(&self, point: Pos2, transform: &PlotTransform) -> Option<ClosestElem> {
        let verts = &self.compiled.payload.hover_vertices;
        if let Some(proj) = transform.linear_projector() {
            verts
                .iter()
                .enumerate()
                .filter_map(|(index, hv)| {
                    if !hv.projected.x.is_finite() || !hv.projected.y.is_finite() {
                        return None;
                    }
                    let screen = proj.project(hv.projected.x, hv.projected.y);
                    screen.is_finite().then(|| ClosestElem {
                        index,
                        dist_sq: point.distance_sq(screen),
                    })
                })
                .min_by_key(|e| e.dist_sq.ord())
        } else {
            verts
                .iter()
                .enumerate()
                .filter(|(_, hv)| transform.is_point_valid_for_scale(&hv.projected))
                .map(|(index, hv)| {
                    let screen = transform.position_from_point(&hv.projected);
                    ClosestElem {
                        index,
                        dist_sq: point.distance_sq(screen),
                    }
                })
                .min_by_key(|e| e.dist_sq.ord())
        }
    }

    fn on_hover(
        &self,
        plot_area_response: &egui::Response,
        elem: ClosestElem,
        shapes: &mut Vec<Shape>,
        cursors: &mut Vec<Cursor>,
        plot: &PlotConfig<'_>,
        label_formatter: &LabelFormatter<'_>,
    ) -> Option<PlotPoint> {
        let payload = &self.compiled.payload;
        let hv = payload.hover_vertices.get(elem.index)?;
        let name = payload
            .names
            .get(hv.name_idx as usize)
            .map_or("", String::as_str);
        let item_plot = PlotConfig {
            world_point: plot.world_point.or(Some(hv.world)),
            ..*plot
        };
        Some(render_snapped_hover_point_named(
            name,
            hv.color,
            plot_area_response,
            hv.projected,
            shapes,
            cursors,
            &item_plot,
            label_formatter,
        ))
    }

    fn extract_export_primitives(&self) -> Vec<ExportPrimitive> {
        let mut out = Vec::new();
        materialize_scene_exports(&self.compiled.payload, &mut out);
        out
    }

    fn append_deferred_export(
        &mut self,
        _datasets: &mut Vec<egui_plot::DeferredDatasetEntry>,
        primitives: &mut Vec<DeferredPlotPrimitive>,
    ) {
        if !self.compiled.payload.draw_ops.is_empty() {
            primitives.push(DeferredPlotPrimitive::Lazy(
                self.compiled.lazy_export.clone(),
            ));
        }
    }

    fn append_deferred_export_with_cached(
        &mut self,
        _cached_points: Option<&SharedPoints>,
        datasets: &mut Vec<egui_plot::DeferredDatasetEntry>,
        primitives: &mut Vec<DeferredPlotPrimitive>,
    ) {
        self.append_deferred_export(datasets, primitives);
    }

    fn base(&self) -> &PlotItemBase {
        &self.base
    }

    fn base_mut(&mut self) -> &mut PlotItemBase {
        &mut self.base
    }
}
