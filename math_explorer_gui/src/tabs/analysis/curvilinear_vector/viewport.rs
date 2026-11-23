//! 3D `Plot3DViewport` quiver, curvilinear `Domain` wireframe, and `StokesPatch` circulation
//! loop renderer plus dataset/SVG export helpers for [`super::CurvilinearVectorStudio`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::compute::curvilinear_vector_to_cartesian;
use super::{CoordinateFrameKind, CurvilinearVectorStudio};
use crate::app::SemanticPalette;
use crate::camera::{Plot3DViewport, Scene3DBuilder, CAMERA_3D_HELPER_TEXT};
use eframe::egui;
use egui_plot::{ExportPrimitive, LineStyle};
use math_explorer::pure_math::vector_calculus::coordinates::{
    Cartesian, Cylindrical, OrthogonalCoordinateSystem, Spherical,
};
use math_explorer::pure_math::vector_calculus::theorems::StokesPatch;
use nalgebra::{Point3, Vector3};

fn add_arrow_3d(
    scene: &mut Scene3DBuilder,
    series_label: &str,
    origin: [f64; 3],
    vec: [f64; 3],
    scale: f64,
    color: egui::Color32,
    width: f32,
) {
    let mag = (vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]).sqrt();
    if mag < 1e-9 || !mag.is_finite() {
        return;
    }
    let clamped_len = (mag * scale).clamp(0.03, 0.55);
    let dir = [vec[0] / mag, vec[1] / mag, vec[2] / mag];
    let tip = [
        origin[0] + dir[0] * clamped_len,
        origin[1] + dir[1] * clamped_len,
        origin[2] + dir[2] * clamped_len,
    ];
    scene.add_segment(
        series_label,
        Point3::from(origin),
        Point3::from(tip),
        egui::Stroke::new(width, color),
        !series_label.is_empty(),
    );

    let up = if dir[2].abs() < 0.85 {
        [0.0, 0.0, 1.0]
    } else {
        [1.0, 0.0, 0.0]
    };
    let side = [
        dir[1] * up[2] - dir[2] * up[1],
        dir[2] * up[0] - dir[0] * up[2],
        dir[0] * up[1] - dir[1] * up[0],
    ];
    let s_norm = (side[0] * side[0] + side[1] * side[1] + side[2] * side[2])
        .sqrt()
        .max(1e-9);
    let wing = 0.22 * clamped_len;
    let back = 0.28 * clamped_len;
    let w1 = Point3::new(
        tip[0] - dir[0] * back + (side[0] / s_norm) * wing,
        tip[1] - dir[1] * back + (side[1] / s_norm) * wing,
        tip[2] - dir[2] * back + (side[2] / s_norm) * wing,
    );
    let w2 = Point3::new(
        tip[0] - dir[0] * back - (side[0] / s_norm) * wing,
        tip[1] - dir[1] * back - (side[1] / s_norm) * wing,
        tip[2] - dir[2] * back - (side[2] / s_norm) * wing,
    );
    scene.add_polyline(
        "",
        &[w1, Point3::from(tip), w2],
        egui::Stroke::new((width * 0.9).max(1.0), color),
        LineStyle::Solid,
        false,
    );
}

fn build_curvilinear_domain_edges(
    coords: &dyn OrthogonalCoordinateSystem,
    min: [f64; 3],
    max: [f64; 3],
) -> Vec<Vec<Point3<f64>>> {
    let mut edges = Vec::with_capacity(12);
    let segs = 16_usize;
    for axis in 0..3 {
        let a = (axis + 1) % 3;
        let b = (axis + 2) % 3;
        for &va in &[min[a], max[a]] {
            for &vb in &[min[b], max[b]] {
                let mut curve = Vec::with_capacity(segs + 1);
                for s in 0..=segs {
                    let t = s as f64 / segs as f64;
                    let mut p = Vector3::zeros();
                    p[axis] = min[axis] + t * (max[axis] - min[axis]);
                    p[a] = va;
                    p[b] = vb;
                    let c = coords.local_to_cartesian(&p);
                    curve.push(Point3::new(c[0], c[1], c[2]));
                }
                edges.push(curve);
            }
        }
    }
    edges
}

fn build_stokes_loop_3d(
    coords: &dyn OrthogonalCoordinateSystem,
    patch: &StokesPatch,
) -> Vec<Point3<f64>> {
    let segs = 20_usize;
    let mut pts = Vec::with_capacity(4 * segs + 1);
    for s in 0..segs {
        let u = patch.u_range.0 + (s as f64 / segs as f64) * (patch.u_range.1 - patch.u_range.0);
        let c = coords.local_to_cartesian(&patch.point_at(u, patch.v_range.0));
        pts.push(Point3::new(c[0], c[1], c[2]));
    }
    for s in 0..segs {
        let v = patch.v_range.0 + (s as f64 / segs as f64) * (patch.v_range.1 - patch.v_range.0);
        let c = coords.local_to_cartesian(&patch.point_at(patch.u_range.1, v));
        pts.push(Point3::new(c[0], c[1], c[2]));
    }
    for s in 0..segs {
        let u = patch.u_range.1 - (s as f64 / segs as f64) * (patch.u_range.1 - patch.u_range.0);
        let c = coords.local_to_cartesian(&patch.point_at(u, patch.v_range.1));
        pts.push(Point3::new(c[0], c[1], c[2]));
    }
    for s in 0..=segs {
        let v = patch.v_range.1 - (s as f64 / segs as f64) * (patch.v_range.1 - patch.v_range.0);
        let c = coords.local_to_cartesian(&patch.point_at(patch.u_range.0, v));
        pts.push(Point3::new(c[0], c[1], c[2]));
    }
    pts
}

impl CurvilinearVectorStudio {
    pub(super) fn render_viewport_panel(&mut self, parent_ui: &mut egui::Ui) {
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            let dark = ui.visuals().dark_mode;
            let palette = SemanticPalette::from_visuals(ui.visuals());
            let mut camera = self.camera;
            let mut scene = Scene3DBuilder::new(camera);
            scene.add_axes(2.2, dark);
            self.populate_scene_3d(&mut scene, &palette);
            Plot3DViewport::new("curvilinear_vector_3d_viewport", 2.9)
                .helper_text(CAMERA_3D_HELPER_TEXT)
                .show(ui, &mut camera, |plot_ui| {
                    scene.finish(plot_ui);
                });
            self.camera = camera;
        });
    }

    fn populate_scene_3d(&self, scene: &mut Scene3DBuilder, palette: &SemanticPalette) {
        let col_total = palette.series_color(0);
        let col_irr = palette.series_color(1);
        let col_sol = palette.series_color(2);
        let col_dom = palette.graph_edge;
        let col_stokes = palette.warning;

        for (idx, q) in self.quivers.iter().enumerate() {
            if self.show_total_field {
                let lbl = if idx == 0 { "Total F" } else { "" };
                add_arrow_3d(
                    scene,
                    lbl,
                    q.pos,
                    q.total,
                    self.quiver_scale,
                    col_total,
                    1.5,
                );
            }
            if self.show_irrotational {
                let lbl = if idx == 0 {
                    "Irrotational (-grad Phi)"
                } else {
                    ""
                };
                add_arrow_3d(
                    scene,
                    lbl,
                    q.pos,
                    q.irrotational,
                    self.quiver_scale,
                    col_irr,
                    1.3,
                );
            }
            if self.show_solenoidal {
                let lbl = if idx == 0 { "Solenoidal F_sol" } else { "" };
                add_arrow_3d(
                    scene,
                    lbl,
                    q.pos,
                    q.solenoidal,
                    self.quiver_scale,
                    col_sol,
                    1.3,
                );
            }
        }

        let patch = StokesPatch::new(
            self.stokes_axis.min(2),
            self.stokes_fixed,
            (self.stokes_u[0], self.stokes_u[1]),
            (self.stokes_v[0], self.stokes_v[1]),
        );
        let coords_ref: &dyn OrthogonalCoordinateSystem = match self.frame {
            CoordinateFrameKind::Cartesian => &Cartesian,
            CoordinateFrameKind::Cylindrical => &Cylindrical,
            CoordinateFrameKind::Spherical => &Spherical,
        };
        self.add_geometry_overlays(coords_ref, &patch, scene, col_dom, col_stokes, palette);
    }

    fn add_geometry_overlays(
        &self,
        coords: &dyn OrthogonalCoordinateSystem,
        patch: &StokesPatch,
        scene: &mut Scene3DBuilder,
        col_dom: egui::Color32,
        col_stokes: egui::Color32,
        palette: &SemanticPalette,
    ) {
        if self.show_domain_wireframe {
            for (i, edge) in
                build_curvilinear_domain_edges(coords, self.domain_min, self.domain_max)
                    .into_iter()
                    .enumerate()
            {
                let lbl = if i == 0 { "Gauss Domain" } else { "" };
                scene.add_polyline(
                    lbl,
                    &edge,
                    egui::Stroke::new(1.15_f32, col_dom),
                    LineStyle::Solid,
                    i == 0,
                );
            }
        }
        if self.show_stokes_loop {
            let loop_pts = build_stokes_loop_3d(coords, patch);
            scene.add_polyline(
                "Stokes Loop",
                &loop_pts,
                egui::Stroke::new(2.4_f32, col_stokes),
                LineStyle::Solid,
                true,
            );
        }
        let probe_local = Vector3::from_row_slice(&self.probe_point);
        let p_cart = coords.local_to_cartesian(&probe_local);
        let grad_cart =
            curvilinear_vector_to_cartesian(coords, &probe_local, &self.diagnostics.probe_gradient);
        let curl_cart =
            curvilinear_vector_to_cartesian(coords, &probe_local, &self.diagnostics.probe_curl);
        let p_arr = [p_cart[0], p_cart[1], p_cart[2]];
        add_arrow_3d(
            scene,
            "Probe grad Phi",
            p_arr,
            [grad_cart[0], grad_cart[1], grad_cart[2]],
            self.quiver_scale * 1.25,
            palette.success,
            2.2,
        );
        add_arrow_3d(
            scene,
            "Probe curl F",
            p_arr,
            [curl_cart[0], curl_cart[1], curl_cart[2]],
            self.quiver_scale * 1.25,
            palette.error,
            2.2,
        );
    }

    pub(super) fn collect_export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let mut total_series = Vec::with_capacity(self.quivers.len());
        let mut irr_series = Vec::with_capacity(self.quivers.len());
        let mut sol_series = Vec::with_capacity(self.quivers.len());
        for (idx, q) in self.quivers.iter().enumerate() {
            let x = idx as f64;
            let norm3 = |v: [f64; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            total_series.push([x, norm3(q.total)]);
            irr_series.push([x, norm3(q.irrotational)]);
            sol_series.push([x, norm3(q.solenoidal)]);
        }
        vec![
            ("Total Field |F|".to_string(), total_series),
            ("Irrotational |F_irr|".to_string(), irr_series),
            ("Solenoidal |F_sol|".to_string(), sol_series),
        ]
    }

    pub(super) fn collect_export_primitives(&self) -> Vec<ExportPrimitive> {
        let mut out = Vec::with_capacity(self.quivers.len());
        for (idx, q) in self.quivers.iter().enumerate() {
            let p0 = self.camera.project_with_depth(&q.pos).0;
            let tip = [
                q.pos[0] + q.total[0] * self.quiver_scale,
                q.pos[1] + q.total[1] * self.quiver_scale,
                q.pos[2] + q.total[2] * self.quiver_scale,
            ];
            let p1 = self.camera.project_with_depth(&tip).0;
            out.push(ExportPrimitive::Polyline {
                name: if idx == 0 {
                    "Total Field Quivers".to_string()
                } else {
                    String::new()
                },
                points: vec![p0, p1].into(),
                color: crate::app::palette::DARK_PALETTE.series[0],
                stroke_width: 1.5_f32,
                closed: false,
                fill: None,
            });
        }
        out
    }
}
