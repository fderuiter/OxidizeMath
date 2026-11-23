#![cfg_attr(any(), verified(opt_out = "gui_tool"))]
//! Zero-allocation 3D triangle projection, cutting-plane contour tracing, radial node counting,
//! and watertight millimeter-scaled 3D solid mesh export for [`super::VolumetricEngine`].
// [cite:isosurface_extraction]

use super::super::state::{FieldSourceMode, STUDIO_VIEWPORT_EXTENT, VolumetricStudioConfig};
use super::{VolumetricEngine, is_signed_field};
use crate::camera::Camera3D;
use crate::volumetric_scene::VolumeCutFilter;
use eframe::egui;
use egui_plot::PlotPoint;
use math_explorer::applied::isosurface::{
    CutPlaneCap, VoxelGrid, extract_capped_isosurface, extract_capped_isosurface_into,
};
use math_explorer::biology::diffusion::FiniteDifference3D;
use math_explorer::biology::morphogenesis::{SchnakenbergKinetics, TuringSystem};
use math_explorer::math_kernel::types::flatten_3d_index;
#[cfg(test)]
use oxidize_core::mesh::export_mesh_to_obj_string;
use oxidize_core::mesh::{Mesh, Point3D as CorePoint3D, scale_mesh_to_millimeters};

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
    let max_cap = out.capacity().min(MAX_RENDERED_FACETS);
    if max_cap == 0 {
        return;
    }
    let effective_cap = if let Some(c) = cut {
        let frac = (((c.max_coord / STUDIO_VIEWPORT_EXTENT.max(1e-6)) + 1.0) * 0.5).clamp(0.2, 0.9);
        ((max_cap as f64) * frac).round().max(1.0) as usize
    } else {
        max_cap
    };
    let total_tris = (pos_mesh.indices.len() / 3)
        + if has_neg {
            neg_mesh.indices.len() / 3
        } else {
            0
        };
    if total_tris == 0 {
        return;
    }
    let stride = total_tris.div_ceil(effective_cap).max(1);
    let pos_base = [58_u8, 170_u8, 250_u8];
    let neg_base = [248_u8, 118_u8, 62_u8];
    append_projected_mesh(pos_mesh, pos_base, stride, cut, camera, effective_cap, out);
    if has_neg && out.len() < effective_cap {
        append_projected_mesh(neg_mesh, neg_base, stride, cut, camera, effective_cap, out);
    }
    out.sort_unstable_by(|a, b| a.depth.total_cmp(&b.depth));
}

fn append_projected_mesh(
    mesh: &Mesh,
    rgb: [u8; 3],
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
            if [cx, cy, cz][c.axis.min(2)] > c.max_coord + 1.0e-3 {
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
        let nc = camera.rotate_to_camera_space(&[n_raw.x as f64, n_raw.y as f64, n_raw.z as f64]);
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
    let ax = c.axis.min(2);
    let orig = grid.origin();
    let vs = grid.voxel_size();
    let (min_a, step_a, len_a) = match ax {
        0 => (orig.x as f64, vs.x as f64, grid.width()),
        1 => (orig.y as f64, vs.y as f64, grid.height()),
        _ => (orig.z as f64, vs.z as f64, grid.depth()),
    };
    if len_a < 2 || out.capacity() == 0 {
        return;
    }
    let slice_idx = (((c.max_coord - min_a) / step_a.max(1e-6))
        .round()
        .clamp(0.0, (len_a - 1) as f64)) as usize;
    if signed {
        trace_slice_level(grid, ax, slice_idx, c.max_coord, 0.0, out);
    }
    trace_slice_level(grid, ax, slice_idx, c.max_coord, iso, out);
    if signed {
        trace_slice_level(grid, ax, slice_idx, c.max_coord, -iso, out);
    }
}

fn slice_plane_geometry(grid: &VoxelGrid, axis: usize) -> (usize, usize, f64, f64, f64, f64) {
    let (o, s) = (grid.origin(), grid.voxel_size());
    match axis {
        0 => (
            grid.height(),
            grid.depth(),
            o.y as f64,
            o.z as f64,
            s.y as f64,
            s.z as f64,
        ),
        1 => (
            grid.width(),
            grid.depth(),
            o.x as f64,
            o.z as f64,
            s.x as f64,
            s.z as f64,
        ),
        _ => (
            grid.width(),
            grid.height(),
            o.x as f64,
            o.y as f64,
            s.x as f64,
            s.y as f64,
        ),
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
    let (dim_u, dim_v, orig_u, orig_v, step_u, step_v) = slice_plane_geometry(grid, axis);
    let sample = |u: usize, v: usize| -> f32 {
        match axis {
            0 => grid.get(slice_idx, u, v),
            1 => grid.get(u, slice_idx, v),
            _ => grid.get(u, v, slice_idx),
        }
    };
    let to_world = |u: f64, v: f64| -> [f64; 3] {
        let cu = orig_u + u * step_u;
        let cv = orig_v + v * step_v;
        match axis {
            0 => [plane_coord, cu, cv],
            1 => [cu, plane_coord, cv],
            _ => [cu, cv, plane_coord],
        }
    };
    for v in 0..dim_v.saturating_sub(1) {
        for u in 0..dim_u.saturating_sub(1) {
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

    /// Extracts a closed, outward-wound 2-manifold [`Mesh`] scaled to `cfg.export_size_mm` millimeters.
    pub fn export_solid_mesh_3d(&self, cfg: &VolumetricStudioConfig) -> Result<Mesh, String> {
        let cut_cap = cfg.active_cut_filter().map(to_cut_plane_cap);
        let signed = is_signed_field(cfg);
        let mut combined = self.build_capped_export_mesh(cfg, signed, cut_cap)?;
        if combined.indices.is_empty() && cut_cap.is_some() {
            combined = self.build_capped_export_mesh(cfg, signed, None)?;
        }
        if combined.indices.is_empty() {
            return Err("No solid volume at current level / iso_threshold to export.".into());
        }
        scale_mesh_to_millimeters(&mut combined, cfg.export_size_mm as f32)
            .map_err(|e| format!("Millimeter mesh scaling error: {e}"))?;
        Ok(combined)
    }

    fn build_capped_export_mesh(
        &self,
        cfg: &VolumetricStudioConfig,
        signed: bool,
        cut_cap: Option<CutPlaneCap>,
    ) -> Result<Mesh, String> {
        let pos_thresh = if cfg.source == FieldSourceMode::CustomImplicit {
            0.0_f32
        } else {
            (cfg.iso_threshold as f32).clamp(0.001, 2.0)
        };
        let mut combined = extract_capped_isosurface(&self.voxel_grid, pos_thresh, false, cut_cap)
            .map_err(|e| format!("Isosurface export error: {e}"))?;
        if signed {
            let neg = extract_capped_isosurface(&self.voxel_grid, -pos_thresh, true, cut_cap)
                .map_err(|e| format!("Negative lobe export error: {e}"))?;
            append_mesh_component(&neg, &mut combined);
        }
        Ok(combined)
    }

    /// Exports the active watertight solid surface(s) to Wavefront `.obj` text with finite unit normals.
    #[cfg(test)]
    pub fn export_obj_string(&self, cfg: &VolumetricStudioConfig) -> Result<String, String> {
        let mesh = self.export_solid_mesh_3d(cfg)?;
        export_mesh_to_obj_string(&mesh).map_err(|e| format!("OBJ export error: {e}"))
    }

    pub(super) fn extract_isosurfaces_zero_alloc(
        &mut self,
        cfg: &VolumetricStudioConfig,
        iso: f32,
        signed: bool,
        cut: Option<VolumeCutFilter>,
    ) {
        if !cfg.render_mode.uses_mesh() {
            self.pos_mesh.clear();
            self.neg_mesh.clear();
            self.projected_facets.clear();
            return;
        }
        let cut_cap = cut.map(to_cut_plane_cap);
        let pos_thresh = if cfg.source == FieldSourceMode::CustomImplicit {
            0.0_f32
        } else {
            iso
        };
        let _ = extract_capped_isosurface_into(
            &self.voxel_grid,
            pos_thresh,
            false,
            cut_cap,
            &mut self.padded_grid,
            &mut self.iso_ws,
            &mut self.pos_mesh,
        );
        if signed {
            let _ = extract_capped_isosurface_into(
                &self.voxel_grid,
                -iso,
                true,
                cut_cap,
                &mut self.padded_grid,
                &mut self.iso_ws,
                &mut self.neg_mesh,
            );
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
    }
}

pub(crate) fn to_cut_plane_cap(c: VolumeCutFilter) -> CutPlaneCap {
    CutPlaneCap {
        axis: c.axis.min(2),
        max_coord: c.max_coord as f32,
    }
}

fn append_mesh_component(src: &Mesh, dst: &mut Mesh) {
    if src.indices.is_empty() {
        return;
    }
    let offset = dst.vertices.len();
    dst.vertices.extend_from_slice(&src.vertices);
    dst.normals.extend_from_slice(&src.normals);
    dst.indices
        .extend(src.indices.iter().map(|&idx| offset + idx));
}

pub(super) fn sample_voxel_nearest(grid: &VoxelGrid, world_pt: [f64; 3]) -> f32 {
    let orig = grid.origin();
    let vs = grid.voxel_size();
    let idx = |c: f64, o: f32, s: f32, len: usize| -> usize {
        let max_i = len.saturating_sub(1);
        (((c - f64::from(o)) / f64::from(s.max(1e-6)))
            .round()
            .clamp(0.0, max_i as f64)) as usize
    };
    grid.get(
        idx(world_pt[0], orig.x, vs.x, grid.width()),
        idx(world_pt[1], orig.y, vs.y, grid.height()),
        idx(world_pt[2], orig.z, vs.z, grid.depth()),
    )
}

pub(super) fn inject_turing_pulse(
    sys: &mut TuringSystem<2, SchnakenbergKinetics, FiniteDifference3D>,
    dim: usize,
    world_pt: [f64; 3],
) {
    let to_idx = |c: f64| {
        (((c + STUDIO_VIEWPORT_EXTENT) / (2.0 * STUDIO_VIEWPORT_EXTENT)) * ((dim - 1) as f64))
            .round()
            .clamp(0.0, (dim - 1) as f64) as isize
    };
    let (cx, cy, cz) = (
        to_idx(world_pt[0]),
        to_idx(world_pt[1]),
        to_idx(world_pt[2]),
    );
    let rad = (dim as isize / 7).clamp(2, 5);
    let sigma_sq = ((rad as f64) * 0.65).powi(2).max(0.5);
    for dz in -rad..=rad {
        let z = (cz + dz).clamp(0, (dim - 1) as isize) as usize;
        for dy in -rad..=rad {
            let y = (cy + dy).clamp(0, (dim - 1) as isize) as usize;
            for dx in -rad..=rad {
                let x = (cx + dx).clamp(0, (dim - 1) as isize) as usize;
                let r2 = ((dx * dx + dy * dy + dz * dz) as f64) / sigma_sq;
                let env = (-0.5 * r2).exp();
                let idx = flatten_3d_index(x, y, z, dim, dim);
                sys.u_mut()[idx] = (sys.u()[idx] + 0.85 * env).clamp(0.02, 6.0);
                sys.v_mut()[idx] = (sys.v()[idx] - 0.25 * env).clamp(0.02, 6.0);
            }
        }
    }
}
