#![cfg_attr(any(), verified(opt_out = "gui_tool"))]
//! Cold-path watertight vertex-colored 3D solid mesh (`.ply`) and INRIA-standard 3DGS `.ply`
//! exporters plus zero-allocation implicit splat sampling for [`super::VolumetricFieldStudioTool`].
// [cite:isosurface_extraction]
// [cite:gaussian_splatting]

use super::VolumetricFieldStudioTool;
use super::engine::{is_signed_field, mesh_ops::to_cut_plane_cap};
use super::state::{FieldSourceMode, STUDIO_VIEWPORT_EXTENT, VolumetricStudioConfig};
use crate::volumetric_scene::{
    MAX_DYNAMIC_SPLATS, VolumeCutFilter, VolumetricSplatWorkspace, extract_voxel_splats_into,
    rotation_from_normal,
};
use math_explorer::ai::gaussian_splatting::Gaussian3D;
use math_explorer::applied::isosurface::{
    CentralDifferenceEstimator, CutPlaneCap, VoxelGrid, extract_capped_isosurface,
};
use nalgebra::{Point3, Vector3};
use oxidize_core::mesh::{
    ColoredMesh, GaussianSplatPly, Mesh, Point3D as CorePoint3D, VertexColor,
    export_splats_to_ply_binary, scale_mesh_to_millimeters,
};

const POS_LOW_RGB: [u8; 3] = [44, 152, 242];
const POS_HIGH_RGB: [u8; 3] = [88, 224, 252];
const NEG_LOW_RGB: [u8; 3] = [236, 98, 52];
const NEG_HIGH_RGB: [u8; 3] = [252, 168, 84];
const IMPLICIT_LOW_RGB: [u8; 3] = [38, 168, 236];
const IMPLICIT_HIGH_RGB: [u8; 3] = [84, 230, 176];

impl VolumetricFieldStudioTool {
    /// Builds a watertight, outward-oriented, millimeter-scaled [`ColoredMesh`] for `.ply` export.
    pub fn build_colored_solid_mesh_3d(&self) -> Result<ColoredMesh, String> {
        let cut = self.config.active_cut_filter().map(to_cut_plane_cap);
        let mut out = extract_colored_solid(&self.engine.voxel_grid, &self.config, cut)?;
        if out.mesh.indices.is_empty() && cut.is_some() {
            out = extract_colored_solid(&self.engine.voxel_grid, &self.config, None)?;
        }
        if out.mesh.indices.is_empty() {
            return Err("No solid volume at current level / iso_threshold to export.".into());
        }
        scale_mesh_to_millimeters(&mut out.mesh, self.config.export_size_mm as f32)
            .map_err(|e| format!("Millimeter mesh scaling error: {e}"))?;
        Ok(out)
    }

    /// Serializes the active `Gaussian3D` radiance cloud into an INRIA-standard 3DGS binary `.ply`.
    pub fn export_splat_ply_bytes(&self) -> Result<Vec<u8>, String> {
        if !self.config.splat_scale.is_finite() || self.config.splat_scale <= 0.0 {
            return Err("Gaussian splat scale must be finite and positive.".into());
        }
        let scale_factor = self.config.splat_scale.clamp(0.2, 4.0);
        if !self.engine.splat_ws.scene.gaussians.is_empty() {
            return serialize_gaussians(&self.engine.splat_ws.scene.gaussians, scale_factor);
        }
        let mut fallback_ws = VolumetricSplatWorkspace::with_capacity(MAX_DYNAMIC_SPLATS);
        let cut = self.config.active_cut_filter();
        if self.config.source == FieldSourceMode::CustomImplicit {
            extract_implicit_splats_into(&self.engine.voxel_grid, cut, &mut fallback_ws);
        } else {
            let iso = (self.config.iso_threshold as f32).clamp(0.001, 2.0);
            extract_voxel_splats_into(
                &self.engine.voxel_grid,
                iso * 0.45,
                is_signed_field(&self.config),
                cut,
                &mut fallback_ws,
            );
        }
        if fallback_ws.scene.gaussians.is_empty() {
            return Err("No volumetric Gaussian splats at current threshold to export.".into());
        }
        serialize_gaussians(&fallback_ws.scene.gaussians, scale_factor)
    }
}

/// Extracts gradient-aligned `Gaussian3D` splats strictly inside or along the implicit solid/shell (`val >= -skin`).
pub(super) fn extract_implicit_splats_into(
    grid: &VoxelGrid,
    cut: Option<VolumeCutFilter>,
    ws: &mut VolumetricSplatWorkspace,
) {
    ws.scene.gaussians.clear();
    ws.grad_magnitudes.clear();
    let cap = ws
        .scene
        .gaussians
        .capacity()
        .min(ws.grad_magnitudes.capacity());
    let (w, h, d) = (grid.width(), grid.height(), grid.depth());
    if cap == 0 || w * h * d == 0 {
        return;
    }
    let mut pos_peak = 1e-4_f32;
    let mut occupied = 0usize;
    let vs = grid.voxel_size();
    let skin = -0.05 * vs.x;
    for &v in grid.data() {
        if v >= skin {
            occupied += 1;
            pos_peak = pos_peak.max(v.abs());
        }
    }
    if occupied == 0 {
        return;
    }
    let step = (((occupied / cap.max(1)).max(1) as f64).cbrt().ceil() as usize).max(1);
    let orig = grid.origin();
    let base_r = ((vs.x.max(vs.y).max(vs.z)) as f64) * (step as f64) * 0.68;

    for z in (0..d).step_by(step) {
        let wz = (orig.z + (z as f32) * vs.z) as f64;
        for y in (0..h).step_by(step) {
            let wy = (orig.y + (y as f32) * vs.y) as f64;
            for x in (0..w).step_by(step) {
                if ws.scene.gaussians.len() >= cap {
                    return;
                }
                let val = grid.get(x, y, z);
                if val < skin {
                    continue;
                }
                let wx = (orig.x + (x as f32) * vs.x) as f64;
                if cut.is_some_and(|c| [wx, wy, wz][c.axis.min(2)] > c.max_coord) {
                    continue;
                }
                push_implicit_voxel_splat(grid, (x, y, z), [wx, wy, wz], val, pos_peak, base_r, ws);
            }
        }
    }
}

fn push_implicit_voxel_splat(
    grid: &VoxelGrid,
    ijk: (usize, usize, usize),
    world: [f64; 3],
    val: f32,
    pos_peak: f32,
    base_r: f64,
    ws: &mut VolumetricSplatWorkspace,
) {
    let (normal, grad_mag) =
        CentralDifferenceEstimator.world_normal_and_magnitude(grid, ijk.0, ijk.1, ijk.2);
    let rel = (val.max(0.0) / pos_peak).clamp(0.0, 1.0) as f64;
    let z_norm = ((world[2] + 2.5) * 0.2).clamp(0.0, 1.0);
    let color = Vector3::new(
        0.18 + 0.22 * z_norm,
        0.66 + 0.26 * rel,
        0.92 - 0.20 * z_norm,
    );
    ws.scene.gaussians.push(Gaussian3D {
        mean: Point3::new(world[0], world[1], world[2]),
        scale: Vector3::new(base_r, base_r, base_r * 0.45),
        rotation: rotation_from_normal(normal),
        opacity: (0.24 + 0.64 * (1.0 - 0.4 * rel)).clamp(0.08, 0.92),
        color,
    });
    ws.grad_magnitudes
        .push((grad_mag as f64) / (pos_peak as f64));
}

fn serialize_gaussians(gaussians: &[Gaussian3D], scale_factor: f64) -> Result<Vec<u8>, String> {
    let mut splats = Vec::with_capacity(gaussians.len());
    for g in gaussians {
        let record: GaussianSplatPly = g.to_splat_ply(scale_factor);
        splats.push(record);
    }
    export_splats_to_ply_binary(&splats).map_err(|e| e.to_string())
}

fn extract_colored_solid(
    grid: &VoxelGrid,
    cfg: &VolumetricStudioConfig,
    cut: Option<CutPlaneCap>,
) -> Result<ColoredMesh, String> {
    let mut combined = Mesh::new();
    let mut colors = Vec::new();
    if cfg.source == FieldSourceMode::CustomImplicit {
        let part = extract_capped_isosurface(grid, 0.0, false, cut)
            .map_err(|e| format!("Isosurface export error: {e}"))?;
        append_colored_component(
            &part,
            IMPLICIT_LOW_RGB,
            IMPLICIT_HIGH_RGB,
            &mut combined,
            &mut colors,
        );
    } else {
        let iso = (cfg.iso_threshold as f32).clamp(0.001, 2.0);
        let pos = extract_capped_isosurface(grid, iso, false, cut)
            .map_err(|e| format!("Isosurface export error: {e}"))?;
        append_colored_component(&pos, POS_LOW_RGB, POS_HIGH_RGB, &mut combined, &mut colors);
        if is_signed_field(cfg) {
            let neg = extract_capped_isosurface(grid, -iso, true, cut)
                .map_err(|e| format!("Negative lobe export error: {e}"))?;
            append_colored_component(&neg, NEG_LOW_RGB, NEG_HIGH_RGB, &mut combined, &mut colors);
        }
    }
    Ok(ColoredMesh {
        mesh: combined,
        colors,
    })
}

fn append_colored_component(
    src: &Mesh,
    low_rgb: [u8; 3],
    high_rgb: [u8; 3],
    dst_mesh: &mut Mesh,
    dst_colors: &mut Vec<VertexColor>,
) {
    if src.indices.is_empty() {
        return;
    }
    let base_vertex = dst_mesh.vertices.len();
    let ext = STUDIO_VIEWPORT_EXTENT as f32;
    let span = (2.0 * ext).max(1e-3);
    for (idx, &v) in src.vertices.iter().enumerate() {
        let clean_v = sanitize_point(v);
        let raw_n = src
            .normals
            .get(idx)
            .copied()
            .unwrap_or(CorePoint3D::new(0.0, 0.0, 1.0));
        dst_mesh.vertices.push(clean_v);
        dst_mesh.normals.push(sanitize_unit_normal(raw_n));
        let t = ((clean_v.z + ext) / span).clamp(0.0, 1.0);
        dst_colors.push(VertexColor::new(
            lerp_channel(low_rgb[0], high_rgb[0], t),
            lerp_channel(low_rgb[1], high_rgb[1], t),
            lerp_channel(low_rgb[2], high_rgb[2], t),
        ));
    }
    for &idx in &src.indices {
        dst_mesh.indices.push(base_vertex + idx);
    }
}

fn lerp_channel(low: u8, high: u8, t: f32) -> u8 {
    let lo = f32::from(low);
    let hi = f32::from(high);
    (lo + (hi - lo) * t).round().clamp(0.0, 255.0) as u8
}

fn sanitize_point(p: CorePoint3D) -> CorePoint3D {
    CorePoint3D::new(
        if p.x.is_finite() { p.x } else { 0.0 },
        if p.y.is_finite() { p.y } else { 0.0 },
        if p.z.is_finite() { p.z } else { 0.0 },
    )
}

fn sanitize_unit_normal(n: CorePoint3D) -> CorePoint3D {
    let clean = sanitize_point(n);
    let u = clean.normalize();
    if u.magnitude() > 0.5 {
        u
    } else {
        CorePoint3D::new(0.0, 0.0, 1.0)
    }
}
