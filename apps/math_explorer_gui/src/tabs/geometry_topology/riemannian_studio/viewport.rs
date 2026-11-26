#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Depth-sorted 3D manifold viewport rendering, `AccessKit` canvas wiring, and 3D geodesic
//! brush dispatch for [`super::RiemannianPdeStudio`].

use super::super::surface_brush::{ManifoldPickView, ViewportBrushMode, SCULPT_3D_HELPER_TEXT};
use super::mesh_gen::ManifoldGeometry;
use super::RiemannianPdeStudio;
use crate::app::SemanticPalette;
use crate::framework::{Plot3DViewport, CAMERA_3D_HELPER_TEXT};
use eframe::egui;
use egui_plot::Polygon;
use math_commons::math_kernel::colormap::mapped_gradient;
use nalgebra::Point3;

impl RiemannianPdeStudio {
    pub(super) fn draw_viewport(&mut self, parent_ui: &mut egui::Ui) {
        self.ensure_initialized();
        if self.config.running {
            self.step_once();
            parent_ui.ctx().request_repaint();
        }
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            let palette = SemanticPalette::from_visuals(ui.visuals());
            let is_orbit = self.config.brush_mode == ViewportBrushMode::Orbit;
            let helper = if is_orbit {
                CAMERA_3D_HELPER_TEXT
            } else {
                SCULPT_3D_HELPER_TEXT
            };
            let mut cam = self.config.camera;
            let cam_snap = cam;
            let mut pointer_xy = None;
            let (Some(geom), Some(eng)) = (&self.geometry, &self.engine) else {
                return;
            };
            record_manifold_scene_points(
                &mut self.scene_builder,
                cam,
                geom,
                ui.visuals().dark_mode,
                palette,
            );

            let summary = format!(
                "{} on {} ({} vertices, step {})",
                self.config.regime.label(),
                self.config.manifold.label(),
                geom.positions.len(),
                eng.step_count,
            );
            let plot_resp = Plot3DViewport::new("riemannian_pde_3d_plot", 3.3)
                .title("3D Riemannian PDE Manifold Viewport")
                .status_text(summary)
                .allow_primary_orbit(is_orbit)
                .helper_text(helper)
                .show(ui, &mut cam, |plot_ui| {
                    pointer_xy = plot_ui.pointer_coordinate().map(|p| [p.x, p.y]);
                    render_manifold_triangles(plot_ui, &cam_snap, geom, &eng.scalar_field, palette);
                    self.scene_builder.finish(plot_ui);
                    self.brush
                        .draw_plot_overlays(plot_ui, &cam_snap, &geom.positions, palette);
                });

            self.config.camera = cam;
            self.viewport_response_id = Some(plot_resp.response.id);
            self.last_canvas_id = plot_resp
                .response
                .has_focus()
                .then_some(plot_resp.response.id);
            self.process_viewport_brush(ui, &plot_resp.response, pointer_xy);
        });
    }

    fn process_viewport_brush(
        &mut self,
        ui: &egui::Ui,
        response: &egui::Response,
        pointer_xy: Option<[f64; 2]>,
    ) {
        let prev_cursor = self.brush.cursor_vertex;
        let stroke = if let Some(geom) = &self.geometry {
            self.brush.handle_viewport_response(
                ui,
                response,
                pointer_xy,
                ManifoldPickView {
                    camera: &self.config.camera,
                    laplacian: &geom.laplacian,
                    positions: &geom.positions,
                    triangles: &geom.triangles,
                    normals: &geom.normals,
                    two_sided: geom.two_sided,
                },
            )
        } else {
            None
        };
        self.config.brush_radius = self.brush.radius;
        if let (Some(s), Some(geom), Some(eng)) = (stroke, &mut self.geometry, &mut self.engine) {
            eng.apply_brush_stroke(geom, &self.config, s);
        } else if self.brush.cursor_vertex != prev_cursor {
            if let (Some(geom), Some(eng)) = (&self.geometry, &mut self.engine) {
                eng.refresh_cursor_probes(geom, &self.config, self.brush.cursor_vertex);
            }
        }
    }
}

fn record_manifold_scene_points(
    scene_builder: &mut crate::framework::Scene3DBuilder,
    cam: crate::camera::Camera3D,
    geom: &ManifoldGeometry,
    dark_mode: bool,
    palette: SemanticPalette,
) {
    scene_builder.begin(cam);
    scene_builder.add_axes(1.85, dark_mode);
    let world_pts: Vec<Point3<f64>> = geom
        .positions
        .iter()
        .map(|p| Point3::new(p[0], p[1], p[2]))
        .collect();
    scene_builder.record_series("Riemannian Manifold", &world_pts, palette.series_color(0));
}

fn field_min_max(field: &[f64]) -> (f64, f64) {
    let mut min_v = f64::INFINITY;
    let mut max_v = f64::NEG_INFINITY;
    for &val in field {
        if val.is_finite() {
            min_v = min_v.min(val);
            max_v = max_v.max(val);
        }
    }
    if min_v.is_finite() && max_v.is_finite() {
        (min_v, max_v)
    } else {
        (0.0, 1.0)
    }
}

fn render_manifold_triangles(
    plot_ui: &mut egui_plot::PlotUi<'_>,
    camera: &crate::camera::Camera3D,
    geom: &ManifoldGeometry,
    field: &[f64],
    palette: SemanticPalette,
) {
    let (min_v, max_v) = field_min_max(field);
    let span = max_v - min_v;
    let projected: Vec<([f64; 2], f64)> = geom
        .positions
        .iter()
        .map(|p| camera.project_with_depth(p))
        .collect();
    let mut order: Vec<(usize, f64)> = geom
        .triangles
        .iter()
        .enumerate()
        .map(|(idx, &[a, b, c])| {
            (
                idx,
                (projected[a].1 + projected[b].1 + projected[c].1) / 3.0,
            )
        })
        .collect();
    order.sort_by(|a, b| a.1.total_cmp(&b.1));

    let edge_color = palette.graph_edge.gamma_multiply(0.32);
    for (tri_idx, _) in order {
        let [i0, i1, i2] = geom.triangles[tri_idx];
        let u_avg = (field[i0] + field[i1] + field[i2]) / 3.0;
        let rgb = if !u_avg.is_finite() || span.abs() < 1e-10 {
            mapped_gradient(0.5, 0.0, 1.0)
        } else {
            mapped_gradient(u_avg, min_v, max_v)
        };
        let n_avg = [
            (geom.normals[i0][0] + geom.normals[i1][0] + geom.normals[i2][0]) / 3.0,
            (geom.normals[i0][1] + geom.normals[i1][1] + geom.normals[i2][1]) / 3.0,
            (geom.normals[i0][2] + geom.normals[i1][2] + geom.normals[i2][2]) / 3.0,
        ];
        let n_cam = camera.rotate_to_camera_space(&n_avg);
        let lambert = (0.52 + 0.48 * n_cam[2].abs()).clamp(0.42, 1.0) as f32;
        let fill = egui::Color32::from_rgb(
            (f32::from(rgb[0]) * lambert) as u8,
            (f32::from(rgb[1]) * lambert) as u8,
            (f32::from(rgb[2]) * lambert) as u8,
        );
        plot_ui.polygon(
            Polygon::new("", vec![projected[i0].0, projected[i1].0, projected[i2].0])
                .fill_color(fill)
                .stroke(egui::Stroke::new(0.45_f32, edge_color))
                .allow_hover(false),
        );
    }
}
