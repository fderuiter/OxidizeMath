#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Depth-sorted 3D primitive collector and sidecar emitter for [`super::Plot3DViewport`].

#[path = "scene_batch.rs"]
mod scene_batch;
#[path = "scene_decor.rs"]
mod scene_decor;

use eframe::egui;
use egui_plot::{LineStyle, MarkerShape, Plot3DSeries, Plot3DSeriesSidecar, PlotUi};
use nalgebra::Point3;

use super::Camera3D;
use scene_batch::{
    batch_item_id, compile_sorted_scene, CompiledScene3D, Drawable3D, LabelPrimitive3D,
    PointPrimitive3D, Scene3DBatchItem, SegmentPrimitive3D,
};

/// Reusable 3D scene builder that projects world `Point3<f64>` primitives through [`Camera3D`],
/// splits polylines into 2-point segments, sorts primitives back-to-front by `viewer_z`,
/// compiles a single cached [`PlotItem`](egui_plot::PlotItem), and attaches a shared
/// [`Plot3DSeriesSidecar`] with immutable world `(x, y, z)` coordinates.
#[derive(Clone, Debug, Default)]
pub struct Scene3DBuilder {
    camera: Camera3D,
    origin_offset: [f64; 3],
    primitives: Vec<Drawable3D>,
    order: Vec<usize>,
    sorted: bool,
    labels: Vec<LabelPrimitive3D>,
    sidecar: Plot3DSeriesSidecar,
    compiled: Option<CompiledScene3D>,
    registered_plot_id: Option<egui::Id>,
    hover_world_vertices: Vec<[f64; 3]>,
    generation: u64,
    proj_scratch: Vec<[f64; 2]>,
    vz_scratch: Vec<f64>,
    world_scratch: Vec<[f64; 3]>,
}

impl Scene3DBuilder {
    /// Creates a new `Scene3DBuilder` initialized for `camera`.
    #[must_use]
    pub fn new(camera: Camera3D) -> Self {
        Self {
            camera,
            ..Self::default()
        }
    }

    /// Resets per-frame primitive and sidecar buffers while retaining allocated `Vec` capacities.
    pub fn begin(&mut self, camera: Camera3D) {
        self.camera = camera;
        self.origin_offset = [0.0, 0.0, 0.0];
        self.primitives.clear();
        self.order.clear();
        self.sorted = false;
        self.labels.clear();
        self.sidecar.clear();
        self.compiled = None;
        self.registered_plot_id = None;
        self.hover_world_vertices.clear();
        self.generation = self.generation.wrapping_add(1);
    }

    /// Returns a reference to the shared [`Plot3DSeriesSidecar`] attached by this scene builder.
    #[must_use]
    pub fn sidecar(&self) -> &Plot3DSeriesSidecar {
        &self.sidecar
    }

    /// Number of 3D primitives collected in this scene builder.
    #[must_use]
    pub fn primitive_count(&self) -> usize {
        self.primitives.len()
    }

    /// Sets a visual centering offset subtracted before camera rotation while keeping true world
    /// coordinates intact inside [`Plot3DSeriesSidecar`].
    #[must_use]
    pub fn with_origin_offset(mut self, offset: Point3<f64>) -> Self {
        self.origin_offset = [offset.x, offset.y, offset.z];
        self
    }

    /// Updates the visual centering offset in-place for reusable builders.
    pub fn set_origin_offset(&mut self, offset: Point3<f64>) {
        self.origin_offset = [offset.x, offset.y, offset.z];
    }

    fn project_world(&self, w: Point3<f64>) -> ([f64; 2], f64, [f64; 3]) {
        let centered = [
            w.x - self.origin_offset[0],
            w.y - self.origin_offset[1],
            w.z - self.origin_offset[2],
        ];
        let (proj, vz) = self.camera.project_with_depth(&centered);
        (proj, vz, [w.x, w.y, w.z])
    }

    fn mark_dirty(&mut self) {
        self.sorted = false;
        self.compiled = None;
        self.registered_plot_id = None;
    }

    /// Adds a 2-point 3D segment, optionally recording its endpoints in a logical 3D series.
    pub fn add_segment(
        &mut self,
        name: impl Into<String>,
        p0: Point3<f64>,
        p1: Point3<f64>,
        stroke: egui::Stroke,
        record_series: bool,
    ) {
        self.add_styled_segment(
            name,
            [p0, p1],
            stroke,
            LineStyle::Solid,
            true,
            record_series,
        );
    }

    /// Adds a 2-point 3D segment with custom [`LineStyle`], hover toggle, and series recording.
    pub fn add_styled_segment(
        &mut self,
        name: impl Into<String>,
        endpoints: [Point3<f64>; 2],
        stroke: egui::Stroke,
        style: LineStyle,
        allow_hover: bool,
        record_series: bool,
    ) {
        let w0 = [endpoints[0].x, endpoints[0].y, endpoints[0].z];
        let w1 = [endpoints[1].x, endpoints[1].y, endpoints[1].z];
        self.add_styled_segment_with_world(
            name,
            (endpoints, [w0, w1]),
            stroke,
            style,
            allow_hover,
            record_series,
        );
    }

    /// Adds a 2-point 3D segment projecting `coords.0` while recording `coords.1` world endpoints
    /// in hover readouts and [`Plot3DSeriesSidecar`].
    pub fn add_styled_segment_with_world(
        &mut self,
        name: impl Into<String>,
        coords: ([Point3<f64>; 2], [[f64; 3]; 2]),
        stroke: egui::Stroke,
        style: LineStyle,
        allow_hover: bool,
        record_series: bool,
    ) {
        let (render_endpoints, world_endpoints) = coords;
        let name = name.into();
        let (p0_proj, vz0, _) = self.project_world(render_endpoints[0]);
        let (p1_proj, vz1, _) = self.project_world(render_endpoints[1]);
        if record_series {
            self.append_segment_to_series(&name, [p0_proj, p1_proj], world_endpoints, stroke.color);
        }
        self.mark_dirty();
        self.primitives
            .push(Drawable3D::Segment(SegmentPrimitive3D {
                name,
                p0_proj,
                p1_proj,
                p0_world: world_endpoints[0],
                p1_world: world_endpoints[1],
                viewer_z: 0.5 * (vz0 + vz1),
                stroke,
                style,
                allow_hover,
            }));
    }

    fn append_segment_to_series(
        &mut self,
        name: &str,
        proj: [[f64; 2]; 2],
        world: [[f64; 3]; 2],
        color: egui::Color32,
    ) {
        self.sidecar.push_series(Plot3DSeries::new(
            name.to_owned(),
            vec![proj[0], proj[1]],
            vec![world[0], world[1]],
            color,
        ));
    }

    /// Splits a 3D polyline into 2-point segments for back-to-front `viewer_z` sorting and
    /// registers the unsplit polyline as a single logical [`Plot3DSeries`] when `record_series` is `true`.
    pub fn add_polyline(
        &mut self,
        name: impl Into<String>,
        points: &[Point3<f64>],
        stroke: egui::Stroke,
        style: LineStyle,
        record_series: bool,
    ) {
        if points.is_empty() {
            return;
        }
        let name = name.into();
        self.proj_scratch.clear();
        self.vz_scratch.clear();
        self.world_scratch.clear();
        for &pt in points {
            let (proj, vz, world) = self.project_world(pt);
            self.proj_scratch.push(proj);
            self.vz_scratch.push(vz);
            self.world_scratch.push(world);
        }
        self.emit_polyline_from_scratch(name, stroke, style, record_series);
    }

    /// Splits a 3D polyline whose render coordinates are normalized separately from true `world_points`.
    pub fn add_polyline_with_world(
        &mut self,
        name: impl Into<String>,
        render_points: &[Point3<f64>],
        world_points: &[[f64; 3]],
        stroke: egui::Stroke,
        style: LineStyle,
        record_series: bool,
    ) {
        let count = render_points.len().min(world_points.len());
        if count == 0 {
            return;
        }
        let name = name.into();
        self.proj_scratch.clear();
        self.vz_scratch.clear();
        self.world_scratch.clear();
        for i in 0..count {
            let (proj, vz, _) = self.project_world(render_points[i]);
            self.proj_scratch.push(proj);
            self.vz_scratch.push(vz);
            self.world_scratch.push(world_points[i]);
        }
        self.emit_polyline_from_scratch(name, stroke, style, record_series);
    }

    fn emit_polyline_from_scratch(
        &mut self,
        name: String,
        stroke: egui::Stroke,
        style: LineStyle,
        record_series: bool,
    ) {
        self.mark_dirty();
        let n = self.proj_scratch.len();
        for i in 0..n.saturating_sub(1) {
            self.primitives
                .push(Drawable3D::Segment(SegmentPrimitive3D {
                    name: name.clone(),
                    p0_proj: self.proj_scratch[i],
                    p1_proj: self.proj_scratch[i + 1],
                    p0_world: self.world_scratch[i],
                    p1_world: self.world_scratch[i + 1],
                    viewer_z: 0.5 * (self.vz_scratch[i] + self.vz_scratch[i + 1]),
                    stroke,
                    style,
                    allow_hover: !name.is_empty(),
                }));
        }
        if record_series {
            self.sidecar.push_series(Plot3DSeries::new(
                name,
                self.proj_scratch.clone(),
                self.world_scratch.clone(),
                stroke.color,
            ));
        }
    }

    /// Adds a 3D point cloud whose markers participate individually in `viewer_z` sorting and
    /// registers a unified logical [`Plot3DSeries`] when `record_series` is `true`.
    pub fn add_points(
        &mut self,
        name: impl Into<String>,
        points: &[Point3<f64>],
        radius: f32,
        color: egui::Color32,
        shape: MarkerShape,
        record_series: bool,
    ) {
        let world: Vec<[f64; 3]> = points.iter().map(|p| [p.x, p.y, p.z]).collect();
        self.add_points_with_world(name, (points, &world), radius, color, shape, record_series);
    }

    /// Adds a 3D point cloud with separate normalized `coords.0` and true `coords.1` world coordinates.
    pub fn add_points_with_world(
        &mut self,
        name: impl Into<String>,
        coords: (&[Point3<f64>], &[[f64; 3]]),
        radius: f32,
        color: egui::Color32,
        shape: MarkerShape,
        record_series: bool,
    ) {
        let (render_points, world_points) = coords;
        let count = render_points.len().min(world_points.len());
        if count == 0 {
            return;
        }
        let name = name.into();
        let mut proj_vec = Vec::with_capacity(count);
        let mut world_vec = Vec::with_capacity(count);
        self.mark_dirty();
        for i in 0..count {
            let (proj, viewer_z, _) = self.project_world(render_points[i]);
            let world = world_points[i];
            proj_vec.push(proj);
            world_vec.push(world);
            self.primitives.push(Drawable3D::Point(PointPrimitive3D {
                name: name.clone(),
                proj,
                world,
                viewer_z,
                radius,
                color,
                shape,
                allow_hover: true,
            }));
        }
        if record_series {
            self.sidecar
                .push_series(Plot3DSeries::new(name, proj_vec, world_vec, color));
        }
    }

    /// Explicitly records a logical 3D series (e.g. deduplicated surface mesh vertices) into the sidecar.
    pub fn record_series(
        &mut self,
        name: impl Into<String>,
        points: &[Point3<f64>],
        color: egui::Color32,
    ) {
        if points.is_empty() {
            return;
        }
        let mut proj_vec = Vec::with_capacity(points.len());
        let mut world_vec = Vec::with_capacity(points.len());
        for &pt in points {
            let (proj, _, world) = self.project_world(pt);
            proj_vec.push(proj);
            world_vec.push(world);
        }
        self.sidecar
            .push_series(Plot3DSeries::new(name, proj_vec, world_vec, color));
    }

    /// Explicitly records a logical 3D series pairing normalized `render_points` with true `world_points`.
    pub fn record_series_with_world(
        &mut self,
        name: impl Into<String>,
        render_points: &[Point3<f64>],
        world_points: &[[f64; 3]],
        color: egui::Color32,
    ) {
        let count = render_points.len().min(world_points.len());
        if count == 0 {
            return;
        }
        let mut proj_vec = Vec::with_capacity(count);
        for &pt in &render_points[..count] {
            let (proj, _, _) = self.project_world(pt);
            proj_vec.push(proj);
        }
        self.sidecar.push_series(Plot3DSeries::new(
            name,
            proj_vec,
            world_points[..count].to_vec(),
            color,
        ));
    }

    /// Sorts buffered primitives back-to-front by `viewer_z` and compiles the cached batch if dirty.
    pub fn ensure_sorted(&mut self) {
        if self.sorted {
            return;
        }
        self.order.clear();
        self.order.extend(0..self.primitives.len());
        let prims = &self.primitives;
        self.order
            .sort_by(|&a, &b| prims[a].viewer_z().total_cmp(&prims[b].viewer_z()));
        if let Some((compiled, world_verts)) =
            compile_sorted_scene(&self.primitives, &self.order, &self.labels, self.generation)
        {
            self.compiled = Some(compiled);
            self.hover_world_vertices = world_verts;
        } else {
            self.compiled = None;
            self.hover_world_vertices.clear();
        }
        self.registered_plot_id = None;
        self.sorted = true;
    }

    /// Returns projected 2D segments `([x0, y0], [x1, y1], viewer_z)` in back-to-front draw order.
    #[must_use]
    pub fn sorted_projected_segments(&mut self) -> Vec<([f64; 2], [f64; 2], f64)> {
        self.ensure_sorted();
        self.order
            .iter()
            .filter_map(|&idx| match &self.primitives[idx] {
                Drawable3D::Segment(s) => Some((s.p0_proj, s.p1_proj, s.viewer_z)),
                Drawable3D::Point(_) => None,
            })
            .collect()
    }

    /// Sorts all collected 3D primitives back-to-front by `viewer_z`, emits one cached batched
    /// [`PlotItem`](egui_plot::PlotItem) into `plot_ui`, and attaches the shared [`Plot3DSeriesSidecar`].
    pub fn finish(&mut self, plot_ui: &mut PlotUi<'_>) {
        self.ensure_sorted();
        let plot_id = plot_ui.plot_id();
        if self.registered_plot_id != Some(plot_id) {
            if !self.hover_world_vertices.is_empty() {
                self.sidecar.register_item_vertices(
                    batch_item_id(plot_id),
                    self.hover_world_vertices.clone(),
                );
            }
            self.registered_plot_id = Some(plot_id);
        }
        if let Some(ref compiled) = self.compiled {
            plot_ui.add(Scene3DBatchItem::new(plot_id, compiled.clone()));
        }
        plot_ui.set_3d_sidecar(self.sidecar.clone());
    }
}
