//! Watertight 3D solid mesh exporter (`.stl`, `.ply`, `.obj`) and `Plot3DViewport`
//! colormapped mesh renderer for [`super::ManifoldFlowStudio`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::super::export_utils::DEFAULT_WALL_THICKNESS;
use super::{ManifoldFlowMode, ManifoldFlowStudio};
use crate::camera::{Plot3DViewport, Scene3DBuilder};
use eframe::egui;
use math_commons::math_kernel::colormap::mapped_gradient;
use math_commons::math_kernel::types::flatten_2d_index;
use nalgebra::{Point3, Vector3};
use oxidize_core::mesh::{ColoredMesh, Mesh, Point3D, VertexColor};

fn to_p3(p: Point3<f64>) -> Point3D {
    Point3D::new(p.x as f32, p.y as f32, p.z as f32)
}

fn to_n3(n: Vector3<f64>) -> Point3D {
    Point3D::new(n.x as f32, n.y as f32, n.z as f32)
}

fn map_scalars_to_colors(values: &[f64]) -> Vec<VertexColor> {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for &v in values {
        if v.is_finite() {
            lo = lo.min(v);
            hi = hi.max(v);
        }
    }
    if !lo.is_finite() || !hi.is_finite() {
        lo = 0.0;
        hi = 1.0;
    }
    let span = hi - lo;
    values
        .iter()
        .map(|&v| {
            let rgb = if !v.is_finite() || span.abs() < 1e-10 {
                mapped_gradient(0.5, 0.0, 1.0)
            } else {
                mapped_gradient(v, lo, hi)
            };
            VertexColor::new(rgb[0], rgb[1], rgb[2])
        })
        .collect()
}

fn compute_mesh_signed_volume(mesh: &Mesh) -> f64 {
    let mut vol6 = 0.0_f64;
    for tri in mesh.indices.as_chunks::<3>().0 {
        let a = mesh.vertices[tri[0]];
        let b = mesh.vertices[tri[1]];
        let c = mesh.vertices[tri[2]];
        let va = Vector3::new(a.x as f64, a.y as f64, a.z as f64);
        let vb = Vector3::new(b.x as f64, b.y as f64, b.z as f64);
        let vc = Vector3::new(c.x as f64, c.y as f64, c.z as f64);
        vol6 += va.dot(&vb.cross(&vc));
    }
    vol6 / 6.0
}

fn ensure_outward_winding(mesh: &mut Mesh) {
    if compute_mesh_signed_volume(mesh) < 0.0 {
        for tri in mesh.indices.as_chunks_mut::<3>().0 {
            tri.swap(1, 2);
        }
        for n in &mut mesh.normals {
            *n = Point3D::new(-n.x, -n.y, -n.z);
        }
    }
}

fn discrete_vertex_normal(
    pts: &[Point3<f64>],
    nu: usize,
    nv: usize,
    closed_u: bool,
    closed_v: bool,
    i: usize,
    j: usize,
) -> Vector3<f64> {
    let idx_u = |ii: i32| -> usize {
        if closed_u {
            ii.rem_euclid(nu as i32) as usize
        } else {
            ii.clamp(0, (nu - 1) as i32) as usize
        }
    };
    let idx_v = |jj: i32| -> usize {
        if closed_v {
            jj.rem_euclid(nv as i32) as usize
        } else {
            jj.clamp(0, (nv - 1) as i32) as usize
        }
    };
    let pu = pts[flatten_2d_index(j, idx_u(i as i32 + 1), nv)]
        - pts[flatten_2d_index(j, idx_u(i as i32 - 1), nv)];
    let pv = pts[flatten_2d_index(idx_v(j as i32 + 1), i, nv)]
        - pts[flatten_2d_index(idx_v(j as i32 - 1), i, nv)];
    let cross = pu.cross(&pv);
    let norm = cross.norm();
    if norm > 1e-12 && norm.is_finite() {
        cross / norm
    } else {
        Vector3::new(0.0, 0.0, 1.0)
    }
}

fn build_periodic_torus_mesh(
    pts: &[Point3<f64>],
    scalars: &[f64],
    nu: usize,
    nv: usize,
) -> ColoredMesh {
    let mut vertices = Vec::with_capacity(nu * nv);
    let mut normals = Vec::with_capacity(nu * nv);
    for i in 0..nu {
        for j in 0..nv {
            let p = pts[flatten_2d_index(j, i, nv)];
            let n = discrete_vertex_normal(pts, nu, nv, true, true, i, j);
            vertices.push(to_p3(p));
            normals.push(to_n3(n));
        }
    }
    let mut indices = Vec::with_capacity(6 * nu * nv);
    let at = |i: usize, j: usize| (i % nu) * nv + (j % nv);
    for i in 0..nu {
        for j in 0..nv {
            let (v00, v10, v11, v01) = (at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1));
            indices.extend_from_slice(&[v00, v10, v11, v00, v11, v01]);
        }
    }
    let mut mesh = Mesh {
        vertices,
        normals,
        indices,
    };
    ensure_outward_winding(&mut mesh);
    ColoredMesh::new(mesh, map_scalars_to_colors(scalars))
}

fn compute_polar_caps(
    pts: &[Point3<f64>],
    scalars: &[f64],
    nu: usize,
    nv: usize,
) -> (Vector3<f64>, Vector3<f64>, Vector3<f64>, f64, f64) {
    let mut north = Vector3::zeros();
    let mut south = Vector3::zeros();
    let (mut north_val, mut south_val) = (0.0, 0.0);
    for i in 0..nu {
        north += pts[flatten_2d_index(0, i, nv)].coords;
        south += pts[flatten_2d_index(nv - 1, i, nv)].coords;
        north_val += scalars[flatten_2d_index(0, i, nv)];
        south_val += scalars[flatten_2d_index(nv - 1, i, nv)];
    }
    north /= nu as f64;
    south /= nu as f64;
    let pole_axis = (north - south)
        .try_normalize(1e-9)
        .unwrap_or_else(|| Vector3::new(0.0, 0.0, 1.0));
    (
        north + pole_axis * 0.02,
        south - pole_axis * 0.02,
        pole_axis,
        north_val / nu as f64,
        south_val / nu as f64,
    )
}

fn build_polar_welded_mesh(
    pts: &[Point3<f64>],
    scalars: &[f64],
    nu: usize,
    nv: usize,
) -> ColoredMesh {
    let (north, south, pole_axis, north_val, south_val) = compute_polar_caps(pts, scalars, nu, nv);
    let mut vertices = vec![to_p3(Point3::from(north)), to_p3(Point3::from(south))];
    let mut normals = vec![to_n3(pole_axis), to_n3(-pole_axis)];
    let mut col_vals = vec![north_val, south_val];

    for j in 0..nv {
        for i in 0..nu {
            let idx = flatten_2d_index(j, i, nv);
            vertices.push(to_p3(pts[idx]));
            normals.push(to_n3(discrete_vertex_normal(
                pts, nu, nv, true, false, i, j,
            )));
            col_vals.push(scalars[idx]);
        }
    }

    let ring = |j: usize, i: usize| 2 + j * nu + (i % nu);
    let mut indices = Vec::with_capacity(6 * nv * nu);
    for i in 0..nu {
        indices.extend_from_slice(&[0, ring(0, i), ring(0, i + 1)]);
    }
    for j in 0..(nv - 1) {
        for i in 0..nu {
            let (r00, r10, r11, r01) = (
                ring(j, i),
                ring(j + 1, i),
                ring(j + 1, i + 1),
                ring(j, i + 1),
            );
            indices.extend_from_slice(&[r00, r10, r11, r00, r11, r01]);
        }
    }
    for i in 0..nu {
        indices.extend_from_slice(&[1, ring(nv - 1, i + 1), ring(nv - 1, i)]);
    }

    let mut mesh = Mesh {
        vertices,
        normals,
        indices,
    };
    ensure_outward_winding(&mut mesh);
    ColoredMesh::new(mesh, map_scalars_to_colors(&col_vals))
}

fn build_extruded_slab_mesh(
    pts: &[Point3<f64>],
    scalars: &[f64],
    nu: usize,
    nv: usize,
) -> ColoredMesh {
    let sheet_len = nu * nv;
    let mut vertices = vec![Point3D::new(0.0, 0.0, 0.0); 2 * sheet_len];
    let mut normals = vec![Point3D::new(0.0, 0.0, 1.0); 2 * sheet_len];
    let mut col_vals = Vec::with_capacity(2 * sheet_len);
    col_vals.extend_from_slice(scalars);
    col_vals.extend_from_slice(scalars);

    for i in 0..nu {
        for j in 0..nv {
            let k = flatten_2d_index(j, i, nv);
            let p = pts[k];
            let n = discrete_vertex_normal(pts, nu, nv, false, false, i, j);
            let p_in = Point3::from(p.coords - n * DEFAULT_WALL_THICKNESS);
            vertices[k] = to_p3(p);
            normals[k] = to_n3(n);
            vertices[sheet_len + k] = to_p3(p_in);
            normals[sheet_len + k] = to_n3(-n);
        }
    }

    let at = |i: usize, j: usize| flatten_2d_index(j, i, nv);
    let mut indices = Vec::with_capacity(12 * (nu - 1) * (nv - 1) + 12 * (nu + nv));
    for i in 0..(nu - 1) {
        for j in 0..(nv - 1) {
            let (o00, o10, o11, o01) = (at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1));
            indices.extend_from_slice(&[o00, o10, o11, o00, o11, o01]);
            let (w00, w10, w11, w01) = (
                sheet_len + o00,
                sheet_len + o10,
                sheet_len + o11,
                sheet_len + o01,
            );
            indices.extend_from_slice(&[w00, w11, w10, w00, w01, w11]);
        }
    }

    let mut perim = Vec::with_capacity(2 * (nu + nv - 2));
    perim.extend((0..(nu - 1)).map(|i| at(i, 0)));
    perim.extend((0..(nv - 1)).map(|j| at(nu - 1, j)));
    perim.extend((1..nu).rev().map(|i| at(i, nv - 1)));
    perim.extend((1..nv).rev().map(|j| at(0, j)));

    for m in 0..perim.len() {
        let a = perim[m];
        let b = perim[(m + 1) % perim.len()];
        indices.extend_from_slice(&[b, a, sheet_len + a, b, sheet_len + a, sheet_len + b]);
    }

    let mut mesh = Mesh {
        vertices,
        normals,
        indices,
    };
    ensure_outward_winding(&mut mesh);
    ColoredMesh::new(mesh, map_scalars_to_colors(&col_vals))
}

impl ManifoldFlowStudio {
    fn active_points_and_scalars(&self) -> (&[Point3<f64>], &[f64]) {
        match self.mode {
            ManifoldFlowMode::MeanCurvature => {
                (self.mcf.surface.as_slice(), &self.mcf.curvature_norms)
            }
            ManifoldFlowMode::HeatDiffusion => (&self.static_points, &self.temp_grid),
        }
    }

    pub(super) fn build_watertight_colored_mesh(&self) -> Result<ColoredMesh, String> {
        let (pts, scalars) = self.active_points_and_scalars();
        let (nu, nv) = (self.u_res, self.v_res);
        if pts.len() != nu * nv || scalars.len() != nu * nv {
            return Err("Manifold buffer dimensions out of sync".to_string());
        }
        if !pts.iter().all(|p| p.coords.iter().all(|c| c.is_finite())) {
            return Err("Manifold contains non-finite vertex coordinates".to_string());
        }
        let cm = match self.surface_type {
            super::SurfaceType::Torus => build_periodic_torus_mesh(pts, scalars, nu, nv),
            super::SurfaceType::Sphere => build_polar_welded_mesh(pts, scalars, nu, nv),
            super::SurfaceType::KleinBottle
            | super::SurfaceType::CustomHeightfield
            | super::SurfaceType::CustomParametric => {
                build_extruded_slab_mesh(pts, scalars, nu, nv)
            }
        };
        Ok(cm)
    }

    pub(super) fn render_viewport_panel(&mut self, parent_ui: &mut egui::Ui) {
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            let dark = ui.visuals().dark_mode;
            let mut camera = self.camera;
            let mut scene = Scene3DBuilder::new(camera);
            scene.add_axes(2.4, dark);
            let sorted_quads = self.populate_manifold_scene(&mut scene, camera, dark);
            Plot3DViewport::new("manifold_flow_3d_viewport", 3.1)
                .helper_text(crate::camera::CAMERA_3D_HELPER_TEXT)
                .show(ui, &mut camera, |plot_ui| {
                    for (_, corners, fill) in &sorted_quads {
                        plot_ui.polygon(
                            egui_plot::Polygon::new(
                                "",
                                egui_plot::PlotPoints::new(corners.to_vec()),
                            )
                            .fill_color(*fill)
                            .stroke(egui::Stroke::NONE)
                            .allow_hover(false),
                        );
                    }
                    scene.finish(plot_ui);
                });
            self.camera = camera;
        });
    }

    fn populate_manifold_scene(
        &self,
        scene: &mut Scene3DBuilder,
        camera: crate::camera::Camera3D,
        dark: bool,
    ) -> Vec<(f64, [[f64; 2]; 4], egui::Color32)> {
        let (pts, scalars) = self.active_points_and_scalars();
        let (nu, nv) = (self.u_res, self.v_res);
        if pts.len() != nu * nv || scalars.len() != nu * nv {
            return Vec::new();
        }
        let pal = if dark {
            crate::app::palette::DARK_PALETTE
        } else {
            crate::app::palette::LIGHT_PALETTE
        };
        scene.record_series("Manifold Mesh", pts, pal.series[0]);
        let colors = map_scalars_to_colors(scalars);
        let closed_u = matches!(
            self.surface_type,
            super::SurfaceType::Sphere | super::SurfaceType::Torus
        );
        let closed_v = matches!(self.surface_type, super::SurfaceType::Torus);
        let i_end = if closed_u { nu } else { nu - 1 };
        let j_end = if closed_v { nv } else { nv - 1 };
        let wire_stroke = egui::Stroke::new(
            0.7_f32,
            egui::Color32::from_gray(if dark { 80 } else { 145 }),
        );
        let mut quads = Vec::with_capacity(i_end * j_end);

        for i in 0..i_end {
            let i1 = (i + 1) % nu;
            for j in 0..j_end {
                let j1 = (j + 1) % nv;
                let k00 = flatten_2d_index(j, i, nv);
                let k10 = flatten_2d_index(j, i1, nv);
                let k11 = flatten_2d_index(j1, i1, nv);
                let k01 = flatten_2d_index(j1, i, nv);
                let c = colors[k00];
                let fill = egui::Color32::from_rgba_unmultiplied(c.r, c.g, c.b, 205);
                let w_pts = [pts[k00], pts[k10], pts[k11], pts[k01]];
                let mut proj = [[0.0_f64; 2]; 4];
                let mut vz_sum = 0.0_f64;
                for (m, wp) in w_pts.iter().enumerate() {
                    let (p2, vz) = camera.project_with_depth(&[wp.x, wp.y, wp.z]);
                    proj[m] = p2;
                    vz_sum += vz;
                }
                quads.push((0.25 * vz_sum, proj, fill));
                if self.show_wireframe {
                    scene.add_segment("", w_pts[0], w_pts[1], wire_stroke, false);
                    scene.add_segment("", w_pts[0], w_pts[3], wire_stroke, false);
                }
            }
        }
        quads.sort_by(|a, b| a.0.total_cmp(&b.0));
        quads
    }

    pub(super) fn collect_export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let label = match self.mode {
            ManifoldFlowMode::MeanCurvature => "Normalized Surface Area A(t)/A(0)",
            ManifoldFlowMode::HeatDiffusion => "Normalized Total Heat Q(t)/Q(0)",
        };
        vec![(label.to_string(), self.metric_history.clone())]
    }

    pub(super) fn collect_export_primitives(&self) -> Vec<egui_plot::ExportPrimitive> {
        let (pts, _) = self.active_points_and_scalars();
        let (nu, nv) = (self.u_res, self.v_res);
        let mut out = Vec::new();
        for i in (0..nu).step_by(2) {
            let mut row_line = Vec::with_capacity(nv);
            for j in 0..nv {
                let p = pts[flatten_2d_index(j, i, nv)];
                let proj = self.camera.project_with_depth(&[p.x, p.y, p.z]).0;
                row_line.push(proj);
            }
            out.push(egui_plot::ExportPrimitive::Polyline {
                name: if i == 0 {
                    "Manifold Isolines".to_string()
                } else {
                    String::new()
                },
                points: row_line.into(),
                color: crate::app::palette::DARK_PALETTE.series[0],
                stroke_width: 1.2_f32,
                closed: false,
                fill: None,
            });
        }
        out
    }
}
