#![cfg_attr(any(), verified(opt_out = "gui_tool"))]
//! 3D viewport rendering (`MarchingCubes` facets, `Gaussian3D` radiance field, cutting plane, and click probe).
// [cite:isosurface_extraction]
// [cite:gaussian_splatting]

use super::state::{
    SliceAxis, VolumetricRenderMode, VolumetricStudioConfig, STUDIO_VIEWPORT_EXTENT,
};
use super::VolumetricFieldStudioTool;
use crate::accessibility::AccessibleTheoryHover;
use crate::camera::{Camera3D, Plot3DViewport};
use crate::volumetric_scene::SPLAT_GRID_SIZE;
use eframe::egui;
use egui_plot::{Line, PlotImage, PlotPoint, PlotPoints, Points, Polygon};

const VIEWPORT_HELPER: &str =
    "Drag or WASD to orbit, scroll or Q/E to zoom, click volume to probe/perturb, double-click or R to reset.";

impl VolumetricFieldStudioTool {
    pub(super) fn draw_viewport(&mut self, parent_ui: &mut egui::Ui) {
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            self.upload_splat_texture_if_needed(ui.ctx());
            let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
            let tex_id = if self.config.render_mode.uses_splats() {
                self.splat_texture.as_ref().map(egui::TextureHandle::id)
            } else {
                None
            };
            let snap_camera = self.config.camera;
            let mut camera = self.config.camera;
            let mut hover_plot_coord: Option<PlotPoint> = None;

            let response =
                Plot3DViewport::new("volumetric_field_studio_plot", STUDIO_VIEWPORT_EXTENT)
                    .helper_text(VIEWPORT_HELPER)
                    .show(ui, &mut camera, |plot_ui| {
                        hover_plot_coord = plot_ui.pointer_coordinate();
                        self.draw_plot_contents(plot_ui, &snap_camera, tex_id, &palette);
                    })
                    .response;

            let cam_changed = camera != self.config.camera;
            self.config.camera = camera;
            if cam_changed {
                self.engine.step_and_extract_zero_alloc(&self.config, false);
                self.splat_dirty = true;
            }
            if response.clicked() && !response.dragged() {
                if let Some(pt2d) = hover_plot_coord {
                    let world_pt = unproject_click_to_world([pt2d.x, pt2d.y], &self.config);
                    self.engine.apply_probe_or_pulse(&mut self.config, world_pt);
                    self.splat_dirty = true;
                    ui.ctx().request_repaint();
                }
            }
            response.accessible_theory_hover(self);
        });
    }

    fn upload_splat_texture_if_needed(&mut self, ctx: &egui::Context) {
        if !self.config.render_mode.uses_splats() {
            return;
        }
        if !self.splat_dirty && self.splat_texture.is_some() {
            return;
        }
        let image = egui::ColorImage {
            size: [SPLAT_GRID_SIZE, SPLAT_GRID_SIZE],
            source_size: egui::vec2(SPLAT_GRID_SIZE as f32, SPLAT_GRID_SIZE as f32),
            pixels: self.engine.splat_ws.pixel_buffer.clone(),
        };
        if let Some(tex) = &mut self.splat_texture {
            tex.set(image, egui::TextureOptions::LINEAR);
        } else {
            self.splat_texture = Some(ctx.load_texture(
                "volumetric_field_studio_radiance",
                image,
                egui::TextureOptions::LINEAR,
            ));
        }
        self.splat_dirty = false;
    }

    fn draw_plot_contents<'p>(
        &'p self,
        plot_ui: &mut egui_plot::PlotUi<'p>,
        camera: &Camera3D,
        tex_id: Option<egui::TextureId>,
        palette: &crate::app::SemanticPalette,
    ) {
        let extent_span = (2.0 * STUDIO_VIEWPORT_EXTENT) as f32;
        if let Some(tid) = tex_id {
            plot_ui.image(PlotImage::new(
                "Gaussian3D Radiance Cloud",
                tid,
                PlotPoint::new(0.0, 0.0),
                egui::vec2(extent_span, extent_span),
            ));
        }
        if self.config.render_mode.uses_mesh() {
            for facet in &self.engine.projected_facets {
                plot_ui.polygon(
                    Polygon::new("Isosurface Facet", &facet.pts[..])
                        .fill_color(facet.fill)
                        .stroke(egui::Stroke::new(0.55_f32, facet.stroke)),
                );
            }
        }
        if self.config.render_mode == VolumetricRenderMode::GaussianRadianceCloud {
            for (idx, (ring, color)) in self.engine.splat_ws.ellipse_rings.iter().enumerate() {
                plot_ui.line(
                    Line::new(
                        format!("Splat {idx} (1.5σ)"),
                        PlotPoints::new(ring.to_vec()),
                    )
                    .color(*color)
                    .width(1.1_f32),
                );
            }
        }
        draw_axes_and_overlays(self, plot_ui, camera, palette);
        plot_ui.set_3d_sidecar(build_volumetric_sidecar(self, camera, palette));
    }
}

fn build_volumetric_sidecar(
    tool: &VolumetricFieldStudioTool,
    camera: &Camera3D,
    palette: &crate::app::SemanticPalette,
) -> egui_plot::Plot3DSeriesSidecar {
    let mut sidecar = egui_plot::Plot3DSeriesSidecar::new();
    if tool.config.render_mode.uses_mesh() {
        push_mesh_series(
            &mut sidecar,
            "Isosurface (+)",
            &tool.engine.pos_mesh.vertices,
            camera,
            palette.series_color(0),
        );
        push_mesh_series(
            &mut sidecar,
            "Isosurface (-)",
            &tool.engine.neg_mesh.vertices,
            camera,
            palette.series_color(1),
        );
    }
    if tool.config.render_mode.uses_splats() {
        let world: Vec<[f64; 3]> = tool
            .engine
            .splat_ws
            .scene
            .gaussians
            .iter()
            .map(|g| [g.mean.x, g.mean.y, g.mean.z])
            .collect();
        let proj: Vec<[f64; 2]> = world.iter().map(|p| camera.project(p)).collect();
        sidecar.push_series(egui_plot::Plot3DSeries::new(
            "Gaussian3D Means",
            proj,
            world,
            palette.accent,
        ));
    }
    append_volumetric_overlay_series(&mut sidecar, tool, camera, palette);
    sidecar
}

fn push_mesh_series(
    sidecar: &mut egui_plot::Plot3DSeriesSidecar,
    name: &str,
    verts: &[oxidize_core::mesh::Point3D],
    camera: &Camera3D,
    color: egui::Color32,
) {
    if verts.is_empty() {
        return;
    }
    let stride = verts.len().div_ceil(256).max(1);
    let mut proj = Vec::with_capacity(256);
    let mut world = Vec::with_capacity(256);
    for v in verts.iter().step_by(stride) {
        let w = [f64::from(v.x), f64::from(v.y), f64::from(v.z)];
        proj.push(camera.project(&w));
        world.push(w);
    }
    sidecar.push_series(egui_plot::Plot3DSeries::new(name, proj, world, color));
}

fn append_volumetric_overlay_series(
    sidecar: &mut egui_plot::Plot3DSeriesSidecar,
    tool: &VolumetricFieldStudioTool,
    camera: &Camera3D,
    palette: &crate::app::SemanticPalette,
) {
    let origin = camera.project(&[0.0, 0.0, 0.0]);
    for (label, tip, color) in [
        ("X Axis", [2.1, 0.0, 0.0], palette.axis_x),
        ("Y Axis", [0.0, 2.1, 0.0], palette.axis_y),
        ("Z Axis", [0.0, 0.0, 2.1], palette.axis_z),
    ] {
        sidecar.push_series(egui_plot::Plot3DSeries::new(
            label,
            vec![origin, camera.project(&tip)],
            vec![[0.0, 0.0, 0.0], tip],
            color,
        ));
    }
    if tool.config.cut_enabled && !tool.engine.slice_contours.is_empty() {
        let mut c_proj = Vec::with_capacity(tool.engine.slice_contours.len() * 2);
        let mut c_world = Vec::with_capacity(tool.engine.slice_contours.len() * 2);
        for seg in &tool.engine.slice_contours {
            c_proj.push(camera.project(&seg[0]));
            c_proj.push(camera.project(&seg[1]));
            c_world.push(seg[0]);
            c_world.push(seg[1]);
        }
        sidecar.push_series(egui_plot::Plot3DSeries::new(
            "Slice Nodal Contour",
            c_proj,
            c_world,
            palette.warning,
        ));
    }
    if let Some(probe) = tool.engine.probe {
        sidecar.push_series(egui_plot::Plot3DSeries::new(
            "3D Probe",
            vec![camera.project(&probe.world_pos)],
            vec![probe.world_pos],
            palette.accent,
        ));
    }
}

fn draw_axes_and_overlays(
    tool: &VolumetricFieldStudioTool,
    plot_ui: &mut egui_plot::PlotUi<'_>,
    camera: &Camera3D,
    palette: &crate::app::SemanticPalette,
) {
    let origin = camera.project(&[0.0, 0.0, 0.0]);
    let axes = [
        (camera.project(&[2.1, 0.0, 0.0]), palette.axis_x, "X Axis"),
        (camera.project(&[0.0, 2.1, 0.0]), palette.axis_y, "Y Axis"),
        (camera.project(&[0.0, 0.0, 2.1]), palette.axis_z, "Z Axis"),
    ];
    for (end, color, label) in axes {
        plot_ui.line(
            Line::new(label, PlotPoints::new(vec![origin, end]))
                .color(color)
                .width(1.4_f32),
        );
    }
    if tool.config.cut_enabled {
        draw_cutting_plane_frame(plot_ui, camera, &tool.config, palette.accent);
        for seg in &tool.engine.slice_contours {
            let p0 = camera.project(&seg[0]);
            let p1 = camera.project(&seg[1]);
            plot_ui.line(
                Line::new("Slice Nodal Contour", PlotPoints::new(vec![p0, p1]))
                    .color(palette.warning)
                    .width(1.8_f32),
            );
        }
    }
    if let Some(probe) = tool.engine.probe {
        let p2 = camera.project(&probe.world_pos);
        plot_ui.points(
            Points::new("3D Probe", vec![p2])
                .radius(5.0_f32)
                .color(palette.accent),
        );
    }
}

fn draw_cutting_plane_frame(
    plot_ui: &mut egui_plot::PlotUi<'_>,
    camera: &Camera3D,
    cfg: &VolumetricStudioConfig,
    color: egui::Color32,
) {
    let [ex, ey, ez] = cfg.viewport_half_extents();
    let t = cfg.cut_offset.clamp(-1.0, 1.0);
    let corners_3d = match cfg.cut_axis {
        SliceAxis::X => {
            let c = t * ex;
            [
                [c, -ey, -ez],
                [c, ey, -ez],
                [c, ey, ez],
                [c, -ey, ez],
                [c, -ey, -ez],
            ]
        }
        SliceAxis::Y => {
            let c = t * ey;
            [
                [-ex, c, -ez],
                [ex, c, -ez],
                [ex, c, ez],
                [-ex, c, ez],
                [-ex, c, -ez],
            ]
        }
        SliceAxis::Z => {
            let c = t * ez;
            [
                [-ex, -ey, c],
                [ex, -ey, c],
                [ex, ey, c],
                [-ex, ey, c],
                [-ex, -ey, c],
            ]
        }
    };
    let projected: Vec<[f64; 2]> = corners_3d.iter().map(|p| camera.project(p)).collect();
    plot_ui.line(
        Line::new("Cutting Plane", PlotPoints::new(projected))
            .color(color)
            .width(1.5_f32),
    );
}

/// Unprojects a 2D plot click onto the active 3D cutting plane (or viewer-perpendicular origin plane).
pub(super) fn unproject_click_to_world(
    plot_xy: [f64; 2],
    cfg: &VolumetricStudioConfig,
) -> [f64; 3] {
    let z = (cfg.camera.zoom as f64).max(0.01);
    let cam_x = plot_xy[0] / z;
    let cam_y = plot_xy[1] / z;
    let r0 = cfg.camera.rotate_to_camera_space(&[1.0, 0.0, 0.0]);
    let r1 = cfg.camera.rotate_to_camera_space(&[0.0, 1.0, 0.0]);
    let r2 = cfg.camera.rotate_to_camera_space(&[0.0, 0.0, 1.0]);
    // Inverse of orthogonal rotation matrix R is R^T: world = R^T * [cam_x, cam_y, cam_z].
    let world_at_depth = |cam_z: f64| -> [f64; 3] {
        [
            r0[0] * cam_x + r0[1] * cam_y + r0[2] * cam_z,
            r1[0] * cam_x + r1[1] * cam_y + r1[2] * cam_z,
            r2[0] * cam_x + r2[1] * cam_y + r2[2] * cam_z,
        ]
    };
    let mut cam_z = 0.0;
    if let Some(cut) = cfg.active_cut_filter() {
        let base = world_at_depth(0.0);
        let dir = [r0[2], r1[2], r2[2]];
        let denom = dir[cut.axis.min(2)];
        if denom.abs() > 0.15 {
            cam_z = ((cut.max_coord - base[cut.axis.min(2)]) / denom)
                .clamp(-STUDIO_VIEWPORT_EXTENT, STUDIO_VIEWPORT_EXTENT);
        }
    }
    let ext = cfg.viewport_half_extents();
    let w = world_at_depth(cam_z);
    [
        w[0].clamp(-ext[0], ext[0]),
        w[1].clamp(-ext[1], ext[1]),
        w[2].clamp(-ext[2], ext[2]),
    ]
}
