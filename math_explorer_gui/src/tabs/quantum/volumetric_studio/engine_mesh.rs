#![cfg_attr(any(), verified(opt_out = "gui_tool"))]
//! Zero-allocation 3D triangle projection, cutting-plane contour tracing, radial node counting,
//! and solid `MarchingCubes` extraction for [`super::VolumetricEngine`].
// [cite:isosurface_extraction]

use super::super::state::{FieldSourceMode, VolumetricStudioConfig, STUDIO_VIEWPORT_EXTENT};
use super::{ProbeReadout, VolumetricEngine};
use crate::camera::Camera3D;
use crate::volumetric_scene::VolumeCutFilter;
use eframe::egui;
use egui_plot::PlotPoint;
use math_explorer::applied::isosurface::{
    CentralDifferenceEstimator, MarchingCubes, SolidClosureConfig, VoxelGrid,
};
use math_explorer::biology::diffusion::FiniteDifference3D;
use math_explorer::biology::morphogenesis::{SchnakenbergKinetics, TuringSystem};
use math_explorer::math_kernel::types::flatten_3d_index;
use math_explorer::physics::quantum::OrbitalQuantumNumbers;
use math_explorer::pure_math::special_functions::legendre::legendre_p_associated;
use oxidize_core::mesh::{Mesh, Point3D as CorePoint3D};

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
    let cap = match cut {
        Some(c) => {
            let frac = ((c.max_coord + STUDIO_VIEWPORT_EXTENT) / (2.0 * STUDIO_VIEWPORT_EXTENT))
                .clamp(0.15, 0.95);
            (((max_cap as f64) * frac).round() as usize).clamp(16.min(max_cap), max_cap)
        }
        None => max_cap,
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
    let stride = (total_tris / cap).max(1);
    let pos_base = [58_u8, 170_u8, 250_u8];
    let neg_base = [248_u8, 118_u8, 62_u8];
    append_projected_mesh(pos_mesh, pos_base, stride, camera, cap, out);
    if has_neg && out.len() < cap {
        append_projected_mesh(neg_mesh, neg_base, stride, camera, cap, out);
    }
    out.sort_unstable_by(|a, b| a.depth.total_cmp(&b.depth));
}

fn append_projected_mesh(
    mesh: &Mesh,
    rgb: [u8; 3],
    stride: usize,
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
        let cut_plane = cfg.active_cutting_plane();
        let mc = MarchingCubes::new(&self.voxel_grid, CentralDifferenceEstimator);
        let has_neg = if cfg.source == FieldSourceMode::CustomImplicitFormula {
            let closure = SolidClosureConfig::superlevel().with_cutting_plane(cut_plane);
            let _ = mc.extract_solid_into(0.0, closure, &mut self.iso_ws, &mut self.pos_mesh);
            self.neg_mesh.clear();
            false
        } else {
            let pos_closure = SolidClosureConfig::superlevel().with_cutting_plane(cut_plane);
            let _ = mc.extract_solid_into(iso, pos_closure, &mut self.iso_ws, &mut self.pos_mesh);
            if signed {
                let neg_closure = SolidClosureConfig::sublevel().with_cutting_plane(cut_plane);
                let _ =
                    mc.extract_solid_into(-iso, neg_closure, &mut self.iso_ws, &mut self.neg_mesh);
            } else {
                self.neg_mesh.clear();
            }
            signed
        };
        project_meshes_into(
            &self.pos_mesh,
            &self.neg_mesh,
            has_neg,
            cut,
            &cfg.camera,
            &mut self.projected_facets,
        );
    }

    /// Samples `world_pt`, evaluates `legendre_p_associated`, and optionally injects a 3D perturbation pulse.
    pub fn apply_probe_or_pulse(
        &mut self,
        cfg: &mut VolumetricStudioConfig,
        world_pt: [f64; 3],
    ) -> ProbeReadout {
        let clamped = [
            world_pt[0].clamp(-STUDIO_VIEWPORT_EXTENT, STUDIO_VIEWPORT_EXTENT),
            world_pt[1].clamp(-STUDIO_VIEWPORT_EXTENT, STUDIO_VIEWPORT_EXTENT),
            world_pt[2].clamp(-STUDIO_VIEWPORT_EXTENT, STUDIO_VIEWPORT_EXTENT),
        ];
        if cfg.probe_brush_enabled {
            match cfg.source {
                FieldSourceMode::TuringMorphogenesis => {
                    inject_turing_pulse(&mut self.turing, self.voxel_grid.width(), clamped);
                }
                FieldSourceMode::QuantumSuperposition => {
                    let wrapped = ((cfg.relative_phase + 0.45 + std::f64::consts::PI)
                        .rem_euclid(std::f64::consts::TAU))
                        - std::f64::consts::PI;
                    cfg.relative_phase = wrapped;
                    self.quantum
                        .set_superposition(cfg.superposition_mix, cfg.relative_phase);
                    self.cached_mix_phase = (
                        cfg.superposition_mix.to_bits(),
                        cfg.relative_phase.to_bits(),
                    );
                }
                FieldSourceMode::CustomImplicitFormula => {}
            }
            self.step_and_extract_zero_alloc(cfg, false);
        }
        let scalar_value = sample_voxel_nearest(&self.voxel_grid, clamped);
        let r =
            (clamped[0] * clamped[0] + clamped[1] * clamped[1] + clamped[2] * clamped[2]).sqrt();
        let cos_theta = if r > 1e-9 {
            (clamped[2] / r).clamp(-1.0, 1.0)
        } else {
            1.0
        };
        let q1 = OrbitalQuantumNumbers::clamped(cfg.n1, cfg.l1, cfg.m1);
        let legendre_polar = legendre_p_associated(q1.l as u64, q1.m as i64, cos_theta);
        let readout = ProbeReadout {
            world_pos: clamped,
            scalar_value,
            legendre_polar,
        };
        self.probe = Some(readout);
        readout
    }
}

fn sample_voxel_nearest(grid: &VoxelGrid, world_pt: [f64; 3]) -> f32 {
    let dim = grid.width().max(2);
    let norm = |c: f64| {
        (((c + STUDIO_VIEWPORT_EXTENT) / (2.0 * STUDIO_VIEWPORT_EXTENT)) * ((dim - 1) as f64))
            .round()
            .clamp(0.0, (dim - 1) as f64) as usize
    };
    grid.get(norm(world_pt[0]), norm(world_pt[1]), norm(world_pt[2]))
}

fn inject_turing_pulse(
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
