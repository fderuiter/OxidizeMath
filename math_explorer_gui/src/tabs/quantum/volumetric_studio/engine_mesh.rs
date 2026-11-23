#![cfg_attr(any(), verified(opt_out = "gui_tool"))]
//! Zero-allocation 3D triangle projection, cutting-plane contour tracing, radial node counting,
//! and manifold Wavefront `.obj` export for [`super::VolumetricEngine`].
// [cite:isosurface_extraction]

use super::super::state::{VolumetricStudioConfig, STUDIO_VIEWPORT_EXTENT};
use super::VolumetricEngine;
use crate::camera::Camera3D;
use crate::volumetric_scene::VolumeCutFilter;
use eframe::egui;
use egui_plot::PlotPoint;
use math_explorer::applied::isosurface::{
    CentralDifferenceEstimator, IsosurfaceWorkspace, MarchingCubes, VoxelGrid,
};
use oxidize_core::mesh::{export_mesh_to_obj_string, Mesh, Point3D as CorePoint3D};

/// Maximum painter facets emitted per frame to keep `egui_plot::Polygon` crisp at 60 FPS.
pub const MAX_RENDERED_FACETS: usize = 2_400;

/// Depth-sorted 2D projected triangle facet with Lambertian shading.
#[derive(Clone, Copy, Debug)]
pub struct ProjectedFacet {
    /// Screen-projected vertices in `Plot3DViewport` coordinates.
    pub pts: [PlotPoint; 3],
    /// Mean viewer-space Z coordinate used for back-to-front painter sorting.
    pub depth: f64,
    /// Shaded translucent facet fill color.
    pub fill: egui::Color32,
    /// Subtle edge wireframe stroke color.
    pub stroke: egui::Color32,
}

/// Projects `pos_mesh` and `neg_mesh` into `out` and sorts back-to-front with zero heap allocations.
pub fn project_meshes_into(
    pos_mesh: &Mesh,
    neg_mesh: &Mesh,
    has_neg: bool,
    cut: Option<VolumeCutFilter>,
    camera: &Camera3D,
    out: &mut Vec<ProjectedFacet>,
) {
    out.clear();
    let cap = out.capacity().min(MAX_RENDERED_FACETS);
    if cap == 0 {
        return;
    }
    let total_tris = (pos_mesh.indices.len() / 3)
        + if has_neg {
            neg_mesh.indices.len() / 3
        } else {
            0
        };
    if total_tris == 0 {
        return;
    }
    let stride = (total_tris / cap).max(1);
    let pos_base = [58_u8, 170_u8, 250_u8];
    let neg_base = [248_u8, 118_u8, 62_u8];
    append_projected_mesh(pos_mesh, pos_base, false, stride, cut, camera, cap, out);
    if has_neg && out.len() < cap {
        append_projected_mesh(neg_mesh, neg_base, true, stride, cut, camera, cap, out);
    }
    out.sort_unstable_by(|a, b| a.depth.total_cmp(&b.depth));
}

#[allow(clippy::too_many_arguments)]
fn append_projected_mesh(
    mesh: &Mesh,
    rgb: [u8; 3],
    invert_normal: bool,
    stride: usize,
    cut: Option<VolumeCutFilter>,
    camera: &Camera3D,
    cap: usize,
    out: &mut Vec<ProjectedFacet>,
) {
    let tri_count = mesh.indices.len() / 3;
    let zoom = camera.zoom as f64;
    let light = [-0.38_f64, 0.48_f64, 0.79_f64];
    for tri_idx in (0..tri_count).step_by(stride.max(1)) {
        if out.len() >= cap {
            break;
        }
        let b = tri_idx * 3;
        let (i0, i1, i2) = (mesh.indices[b], mesh.indices[b + 1], mesh.indices[b + 2]);
        let (v0, v1, v2) = (mesh.vertices[i0], mesh.vertices[i1], mesh.vertices[i2]);
        if let Some(c) = cut {
            let cx = ((v0.x + v1.x + v2.x) as f64) / 3.0;
            let cy = ((v0.y + v1.y + v2.y) as f64) / 3.0;
            let cz = ((v0.z + v1.z + v2.z) as f64) / 3.0;
            if [cx, cy, cz][c.axis.min(2)] > c.max_coord {
                continue;
            }
        }
        let c0 = camera.rotate_to_camera_space(&[v0.x as f64, v0.y as f64, v0.z as f64]);
        let c1 = camera.rotate_to_camera_space(&[v1.x as f64, v1.y as f64, v1.z as f64]);
        let c2 = camera.rotate_to_camera_space(&[v2.x as f64, v2.y as f64, v2.z as f64]);
        let n_raw = mesh
            .normals
            .get(i0)
            .copied()
            .unwrap_or(CorePoint3D::new(0.0, 0.0, 1.0));
        let sign = if invert_normal { -1.0 } else { 1.0 };
        let nc = camera.rotate_to_camera_space(&[
            sign * (n_raw.x as f64),
            sign * (n_raw.y as f64),
            sign * (n_raw.z as f64),
        ]);
        let lambert = (nc[0] * light[0] + nc[1] * light[1] + nc[2] * light[2]).abs();
        let shade = (0.38 + 0.62 * lambert).clamp(0.35, 1.0) as f32;
        let fill = egui::Color32::from_rgba_unmultiplied(
            ((rgb[0] as f32) * shade) as u8,
            ((rgb[1] as f32) * shade) as u8,
            ((rgb[2] as f32) * shade) as u8,
            168,
        );
        let stroke = egui::Color32::from_rgba_unmultiplied(rgb[0], rgb[1], rgb[2], 96);
        out.push(ProjectedFacet {
            pts: [
                PlotPoint::new(c0[0] * zoom, c0[1] * zoom),
                PlotPoint::new(c1[0] * zoom, c1[1] * zoom),
                PlotPoint::new(c2[0] * zoom, c2[1] * zoom),
            ],
            depth: (c0[2] + c1[2] + c2[2]) / 3.0,
            fill,
            stroke,
        });
    }
}

/// Extracts marching-squares cross-section nodal contours on the active 3D cutting plane with zero heap allocations.
pub fn extract_slice_contours_into(
    grid: &VoxelGrid,
    cut: Option<VolumeCutFilter>,
    iso: f32,
    signed: bool,
    out: &mut Vec<[[f64; 3]; 2]>,
) {
    out.clear();
    let Some(c) = cut else {
        return;
    };
    let dim = grid.width();
    if dim < 2 || out.capacity() == 0 {
        return;
    }
    let frac =
        ((c.max_coord + STUDIO_VIEWPORT_EXTENT) / (2.0 * STUDIO_VIEWPORT_EXTENT)).clamp(0.0, 1.0);
    let slice_idx = ((frac * ((dim - 1) as f64)).round() as usize).min(dim - 1);
    if signed {
        trace_slice_level(grid, c.axis.min(2), slice_idx, c.max_coord, 0.0, out);
    }
    trace_slice_level(grid, c.axis.min(2), slice_idx, c.max_coord, iso, out);
    if signed {
        trace_slice_level(grid, c.axis.min(2), slice_idx, c.max_coord, -iso, out);
    }
}

fn trace_slice_level(
    grid: &VoxelGrid,
    axis: usize,
    slice_idx: usize,
    plane_coord: f64,
    level: f32,
    out: &mut Vec<[[f64; 3]; 2]>,
) {
    let dim = grid.width();
    let orig = grid.origin().x as f64;
    let step = grid.voxel_size().x as f64;
    let sample = |u: usize, v: usize| -> f32 {
        match axis {
            0 => grid.get(slice_idx, u, v),
            1 => grid.get(u, slice_idx, v),
            _ => grid.get(u, v, slice_idx),
        }
    };
    let to_world = |u: f64, v: f64| -> [f64; 3] {
        let cu = orig + u * step;
        let cv = orig + v * step;
        match axis {
            0 => [plane_coord, cu, cv],
            1 => [cu, plane_coord, cv],
            _ => [cu, cv, plane_coord],
        }
    };
    for v in 0..(dim - 1) {
        for u in 0..(dim - 1) {
            if out.len() >= out.capacity() {
                return;
            }
            let f00 = sample(u, v) - level;
            let f10 = sample(u + 1, v) - level;
            let f11 = sample(u + 1, v + 1) - level;
            let f01 = sample(u, v + 1) - level;
            let mut crossings = [[0.0_f64; 3]; 4];
            let mut n_pts = 0usize;
            if (f00 >= 0.0) != (f10 >= 0.0) {
                let t = (f00 / (f00 - f10)).clamp(0.0, 1.0) as f64;
                crossings[n_pts] = to_world((u as f64) + t, v as f64);
                n_pts += 1;
            }
            if (f10 >= 0.0) != (f11 >= 0.0) {
                let t = (f10 / (f10 - f11)).clamp(0.0, 1.0) as f64;
                crossings[n_pts] = to_world((u + 1) as f64, (v as f64) + t);
                n_pts += 1;
            }
            if (f01 >= 0.0) != (f11 >= 0.0) {
                let t = (f01 / (f01 - f11)).clamp(0.0, 1.0) as f64;
                crossings[n_pts] = to_world((u as f64) + t, (v + 1) as f64);
                n_pts += 1;
            }
            if (f00 >= 0.0) != (f01 >= 0.0) {
                let t = (f00 / (f00 - f01)).clamp(0.0, 1.0) as f64;
                crossings[n_pts] = to_world(u as f64, (v as f64) + t);
                n_pts += 1;
            }
            if n_pts >= 2 {
                out.push([crossings[0], crossings[1]]);
            }
        }
    }
}

impl VolumetricEngine {
    /// Counts sign changes along a radial ray inside the active 3D cutting plane (`2s -> 1`, `3s -> 2`).
    pub fn count_cutting_plane_radial_nodes(&self, cfg: &VolumetricStudioConfig) -> usize {
        let dim = self.voxel_grid.width();
        if dim < 4 {
            return 0;
        }
        let axis = cfg.cut_axis.index();
        let frac = ((cfg.cut_offset.clamp(-1.0, 1.0) + 1.0) * 0.5).clamp(0.0, 1.0);
        let plane_idx = ((frac * ((dim - 1) as f64)).round() as usize).min(dim - 1);
        let center = dim / 2;
        let mut prev_nonzero = 0.0_f32;
        let mut crossings = 0usize;
        for r_idx in center..dim {
            let val = match axis {
                0 => self.voxel_grid.get(plane_idx, r_idx, center),
                1 => self.voxel_grid.get(r_idx, plane_idx, center),
                _ => self.voxel_grid.get(r_idx, center, plane_idx),
            };
            if val.abs() > 1e-5 {
                if prev_nonzero != 0.0 && (val > 0.0) != (prev_nonzero > 0.0) {
                    crossings += 1;
                }
                prev_nonzero = val;
            }
        }
        crossings
    }

    /// Exports the active `MarchingCubes` surface(s) to Wavefront `.obj` text with finite unit normals.
    pub fn export_obj_string(&self, cfg: &VolumetricStudioConfig) -> Result<String, String> {
        let mut combined = Mesh::new();
        let cut = cfg.active_cut_filter();
        let iso = (cfg.iso_threshold as f32).clamp(0.001, 2.0);
        if self.pos_mesh.vertices.is_empty() && self.neg_mesh.vertices.is_empty() {
            let mut ws =
                IsosurfaceWorkspace::new_lazy(self.voxel_grid.width(), self.voxel_grid.height());
            let mut tmp_pos = Mesh::new();
            let _ = MarchingCubes::new(&self.voxel_grid, CentralDifferenceEstimator).extract_into(
                iso,
                &mut ws,
                &mut tmp_pos,
            );
            append_sanitized_mesh(&tmp_pos, false, cut, &mut combined);
        } else {
            append_sanitized_mesh(&self.pos_mesh, false, cut, &mut combined);
            append_sanitized_mesh(&self.neg_mesh, true, cut, &mut combined);
        }
        if combined.indices.is_empty() && cut.is_some() {
            append_sanitized_mesh(&self.pos_mesh, false, None, &mut combined);
            append_sanitized_mesh(&self.neg_mesh, true, None, &mut combined);
        }
        if combined.indices.is_empty() {
            return Err("No isosurface triangles at current iso_threshold to export.".into());
        }
        export_mesh_to_obj_string(&combined).map_err(|e| format!("OBJ export error: {e}"))
    }
}

fn append_sanitized_mesh(
    src: &Mesh,
    flip_winding: bool,
    cut: Option<VolumeCutFilter>,
    dst: &mut Mesh,
) {
    let mut remap = vec![usize::MAX; src.vertices.len()];
    for &[i0, i1, i2] in src.indices.as_chunks::<3>().0 {
        let (v0, v1, v2) = (src.vertices[i0], src.vertices[i1], src.vertices[i2]);
        if let Some(c) = cut {
            let cent = [
                ((v0.x + v1.x + v2.x) as f64) / 3.0,
                ((v0.y + v1.y + v2.y) as f64) / 3.0,
                ((v0.z + v1.z + v2.z) as f64) / 3.0,
            ];
            if cent[c.axis.min(2)] > c.max_coord {
                continue;
            }
        }
        let face_n = sanitize_normal((v1 - v0).cross(v2 - v0));
        let ordered = if flip_winding {
            [i0, i2, i1]
        } else {
            [i0, i1, i2]
        };
        for idx in ordered {
            if remap[idx] == usize::MAX {
                let new_idx = dst.vertices.len();
                remap[idx] = new_idx;
                let v = src.vertices[idx];
                dst.vertices.push(sanitize_point(v));
                let raw_n = src.normals.get(idx).copied().unwrap_or(face_n);
                let signed_n = if flip_winding { -raw_n } else { raw_n };
                dst.normals
                    .push(sanitize_normal_with_fallback(signed_n, face_n));
            }
            dst.indices.push(remap[idx]);
        }
    }
}

fn sanitize_point(p: CorePoint3D) -> CorePoint3D {
    CorePoint3D::new(
        if p.x.is_finite() { p.x } else { 0.0 },
        if p.y.is_finite() { p.y } else { 0.0 },
        if p.z.is_finite() { p.z } else { 0.0 },
    )
}

fn sanitize_normal(n: CorePoint3D) -> CorePoint3D {
    let clean = sanitize_point(n);
    let u = clean.normalize();
    if u.magnitude() > 0.5 {
        u
    } else {
        CorePoint3D::new(0.0, 0.0, 1.0)
    }
}

fn sanitize_normal_with_fallback(n: CorePoint3D, fallback: CorePoint3D) -> CorePoint3D {
    let clean = sanitize_point(n);
    let u = clean.normalize();
    if u.magnitude() > 0.5 {
        u
    } else {
        fallback
    }
}

impl VolumetricEngine {
    pub(super) fn extract_isosurfaces_zero_alloc(
        &mut self,
        cfg: &VolumetricStudioConfig,
        iso: f32,
        signed: bool,
        cut: Option<VolumeCutFilter>,
    ) {
        if cfg.render_mode.uses_mesh() {
            let mc = MarchingCubes::new(&self.voxel_grid, CentralDifferenceEstimator);
            let _ = mc.extract_into(iso, &mut self.iso_ws, &mut self.pos_mesh);
            if signed {
                let _ = mc.extract_into(-iso, &mut self.iso_ws, &mut self.neg_mesh);
            } else {
                self.neg_mesh.clear();
            }
            project_meshes_into(
                &self.pos_mesh,
                &self.neg_mesh,
                signed,
                cut,
                &cfg.camera,
                &mut self.projected_facets,
            );
        } else {
            self.pos_mesh.clear();
            self.neg_mesh.clear();
            self.projected_facets.clear();
        }
    }
}
