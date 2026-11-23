#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Depth-sorted 3D primitive collector and sidecar emitter for [`super::Plot3DViewport`].

#[path = "scene_batch.rs"]
mod scene_batch;

use eframe::egui;
use egui_plot::{LineStyle, MarkerShape, Plot3DSeries, Plot3DSeriesSidecar, PlotUi};
use nalgebra::Point3;
use scene_batch::{
    build_consolidated_batches, emit_consolidated_batches, CachedPointBatch, CachedSegmentBatch,
};

use super::Camera3D;

#[derive(Clone, Debug)]
pub(super) struct SegmentPrimitive3D {
    pub(super) name: String,
    pub(super) p0_proj: [f64; 2],
    pub(super) p1_proj: [f64; 2],
    pub(super) p0_world: [f64; 3],
    pub(super) p1_world: [f64; 3],
    pub(super) viewer_z: f64,
    pub(super) stroke: egui::Stroke,
    pub(super) style: LineStyle,
    pub(super) allow_hover: bool,
}

#[derive(Clone, Debug)]
pub(super) struct PointPrimitive3D {
    pub(super) name: String,
    pub(super) proj: [f64; 2],
    pub(super) world: [f64; 3],
    pub(super) viewer_z: f64,
    pub(super) radius: f32,
    pub(super) color: egui::Color32,
    pub(super) shape: MarkerShape,
    pub(super) allow_hover: bool,
}

#[derive(Clone, Debug)]
pub(super) enum Drawable3D {
    Segment(SegmentPrimitive3D),
    Point(PointPrimitive3D),
}

impl Drawable3D {
    fn viewer_z(&self) -> f64 {
        match self {
            Self::Segment(s) => s.viewer_z,
            Self::Point(p) => p.viewer_z,
        }
    }
}

#[derive(Clone, Debug)]
pub(super) struct LabelPrimitive3D {
    pub(super) proj: [f64; 2],
    pub(super) text: String,
    pub(super) color: egui::Color32,
}

/// Reusable 3D scene builder that projects world `Point3<f64>` primitives through [`Camera3D`],
/// splits polylines into 2-point segments, sorts primitives back-to-front by `viewer_z`,
/// and attaches a [`Plot3DSeriesSidecar`] with immutable world `(x, y, z)` coordinates.
#[derive(Clone, Debug, Default)]
pub struct Scene3DBuilder {
    camera: Camera3D,
    origin_offset: [f64; 3],
    primitives: Vec<Drawable3D>,
    order: Vec<usize>,
    sorted: bool,
    labels: Vec<LabelPrimitive3D>,
    sidecar: Plot3DSeriesSidecar,
    registered_plot_id: Option<egui::Id>,
    cached_segments: Option<CachedSegmentBatch>,
    cached_points: Option<CachedPointBatch>,
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
        self.registered_plot_id = None;
        self.cached_segments = None;
        self.cached_points = None;
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
        let name = name.into();
        let (p0_proj, vz0, p0_world) = self.project_world(endpoints[0]);
        let (p1_proj, vz1, p1_world) = self.project_world(endpoints[1]);
        if record_series {
            self.append_segment_to_series(
                &name,
                [p0_proj, p1_proj],
                [p0_world, p1_world],
                stroke.color,
            );
        } else if allow_hover {
            self.sidecar.register_projected_pair(p0_proj, p0_world);
            self.sidecar.register_projected_pair(p1_proj, p1_world);
        }
        self.sorted = false;
        self.primitives
            .push(Drawable3D::Segment(SegmentPrimitive3D {
                name,
                p0_proj,
                p1_proj,
                p0_world,
                p1_world,
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
        self.sorted = false;
        for i in 0..points.len().saturating_sub(1) {
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
        if points.is_empty() {
            return;
        }
        let name = name.into();
        let mut proj_vec = Vec::with_capacity(points.len());
        let mut world_vec = Vec::with_capacity(points.len());
        let allow_hover = true;
        self.sorted = false;
        for &pt in points {
            let (proj, viewer_z, world) = self.project_world(pt);
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
                allow_hover,
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

    /// Adds projected 3D X, Y, and Z coordinate reference axes with endpoint labels.
    pub fn add_axes(&mut self, length: f64, dark_mode: bool) {
        let palette = if dark_mode {
            &crate::app::palette::DARK_PALETTE
        } else {
            &crate::app::palette::LIGHT_PALETTE
        };
        let axes = [
            ("X Axis", "X", Point3::new(length, 0.0, 0.0), palette.axis_x),
            ("Y Axis", "Y", Point3::new(0.0, length, 0.0), palette.axis_y),
            ("Z Axis", "Z", Point3::new(0.0, 0.0, length), palette.axis_z),
        ];
        let origin = Point3::new(
            self.origin_offset[0],
            self.origin_offset[1],
            self.origin_offset[2],
        );
        for (axis_name, short, dir, color) in axes {
            let tip = Point3::new(origin.x + dir.x, origin.y + dir.y, origin.z + dir.z);
            self.add_styled_segment(
                axis_name,
                [origin, tip],
                egui::Stroke::new(1.75_f32, color),
                LineStyle::Solid,
                true,
                false,
            );
            let label_tip = Point3::new(
                origin.x + dir.x * 1.08,
                origin.y + dir.y * 1.08,
                origin.z + dir.z * 1.08,
            );
            let (proj, _, _) = self.project_world(label_tip);
            self.labels.push(LabelPrimitive3D {
                proj,
                text: short.to_owned(),
                color,
            });
        }
    }

    /// Adds a decorative projected `xOy` floor grid at `z_plane` sorted alongside scene geometry.
    pub fn add_floor_grid(
        &mut self,
        half_extent: f64,
        divisions: usize,
        z_plane: f64,
        stroke: egui::Stroke,
    ) {
        let divs = divisions.max(1);
        let step = (2.0 * half_extent) / (divs as f64);
        let ox = self.origin_offset[0];
        let oy = self.origin_offset[1];
        for i in 0..=divs {
            let offset = -half_extent + (i as f64) * step;
            self.add_styled_segment(
                "",
                [
                    Point3::new(ox - half_extent, oy + offset, z_plane),
                    Point3::new(ox + half_extent, oy + offset, z_plane),
                ],
                stroke,
                LineStyle::Solid,
                false,
                false,
            );
            self.add_styled_segment(
                "",
                [
                    Point3::new(ox + offset, oy - half_extent, z_plane),
                    Point3::new(ox + offset, oy + half_extent, z_plane),
                ],
                stroke,
                LineStyle::Solid,
                false,
                false,
            );
        }
    }

    /// Sorts buffered primitives back-to-front by `viewer_z` if not already sorted.
    pub fn ensure_sorted(&mut self) {
        if self.sorted {
            return;
        }
        self.order.clear();
        self.order.extend(0..self.primitives.len());
        let prims = &self.primitives;
        self.order
            .sort_by(|&a, &b| prims[a].viewer_z().total_cmp(&prims[b].viewer_z()));
        let (seg_batch, pt_batch) = build_consolidated_batches(&self.primitives, &self.order);
        self.cached_segments = seg_batch;
        self.cached_points = pt_batch;
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

    /// Sorts all collected 3D primitives back-to-front by `viewer_z`, emits consolidated items
    /// into `plot_ui`, and attaches the populated [`Plot3DSeriesSidecar`].
    pub fn finish(&mut self, plot_ui: &mut PlotUi<'_>) {
        self.ensure_sorted();
        let plot_id = plot_ui.plot_id();
        if self.registered_plot_id != Some(plot_id) {
            if let Some(batch) = self.cached_segments.as_ref().filter(|b| b.allow_hover) {
                self.sidecar.register_shared_item_vertices(
                    plot_id.with("scene3d_segments"),
                    std::sync::Arc::clone(&batch.world_vertices),
                );
            }
            if let Some(batch) = self.cached_points.as_ref().filter(|b| b.allow_hover) {
                self.sidecar.register_shared_item_vertices(
                    plot_id.with("scene3d_points"),
                    std::sync::Arc::clone(&batch.world_vertices),
                );
            }
            self.registered_plot_id = Some(plot_id);
        }
        emit_consolidated_batches(
            plot_ui,
            self.cached_segments.as_ref(),
            self.cached_points.as_ref(),
            &self.labels,
            &self.sidecar,
            true,
        );
    }

    /// Emits consolidated 3D items into `plot_ui` without attaching a 3D sidecar (allowing caller tabular series).
    pub fn finish_without_sidecar(&mut self, plot_ui: &mut PlotUi<'_>) {
        self.ensure_sorted();
        emit_consolidated_batches(
            plot_ui,
            self.cached_segments.as_ref(),
            self.cached_points.as_ref(),
            &self.labels,
            &self.sidecar,
            false,
        );
    }
}
