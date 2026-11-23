#![cfg_attr(any(), verified(opt_out = "gui_tool"))]
//! Cold-path watertight 3D solid mesh (`.stl`, `.ply`, `.obj`) and INRIA-standard 3DGS `.ply`
//! exporters for [`super::VolumetricFieldStudioTool`].
// [cite:isosurface_extraction]
// [cite:gaussian_splatting]

use super::engine::is_signed_field;
use super::formula::extract_implicit_splats_into;
use super::state::{FieldSourceMode, VolumetricStudioConfig, STUDIO_VIEWPORT_EXTENT};
use super::VolumetricFieldStudioTool;
use crate::volumetric_scene::{
    extract_voxel_splats_into, VolumetricSplatWorkspace, MAX_DYNAMIC_SPLATS,
};
use math_explorer::applied::isosurface::{
    CentralDifferenceEstimator, CuttingPlane, IsosurfaceWorkspace, MarchingCubes,
    SolidClosureConfig,
};
use oxidize_core::mesh::{
    export_splats_to_ply_binary, ColoredMesh, GaussianSplatPly, Mesh, Point3D as CorePoint3D,
    VertexColor,
};

const POS_LOW_RGB: [u8; 3] = [44, 152, 242];
const POS_HIGH_RGB: [u8; 3] = [88, 224, 252];
const NEG_LOW_RGB: [u8; 3] = [236, 98, 52];
const NEG_HIGH_RGB: [u8; 3] = [252, 168, 84];
const IMPLICIT_LOW_RGB: [u8; 3] = [38, 168, 236];
const IMPLICIT_HIGH_RGB: [u8; 3] = [84, 230, 176];

impl VolumetricFieldStudioTool {
    /// Builds a watertight, outward-oriented [`ColoredMesh`] for `.stl`, `.ply`, and `.obj` exports.
    pub fn build_colored_solid_mesh_3d(&self) -> Result<ColoredMesh, String> {
        let cut = self.config.active_cutting_plane();
        let mut out = extract_colored_solid(&self.engine.voxel_grid, &self.config, cut)?;
        if out.mesh.indices.is_empty() && cut.is_some() {
            out = extract_colored_solid(&self.engine.voxel_grid, &self.config, None)?;
        }
        if out.mesh.indices.is_empty() {
            return Err("No solid isosurface triangles at current threshold to export.".into());
        }
        Ok(out)
    }

    /// Exports the active watertight solid mesh to a Wavefront `.obj` string.
    #[cfg(test)]
    pub fn export_obj_string(&self) -> Result<String, String> {
        let colored = self.build_colored_solid_mesh_3d()?;
        oxidize_core::mesh::export_mesh_to_obj_string(&colored.mesh)
            .map_err(|e| format!("OBJ export error: {e}"))
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
        if self.config.source == FieldSourceMode::CustomImplicitFormula {
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

fn serialize_gaussians(
    gaussians: &[math_explorer::ai::gaussian_splatting::Gaussian3D],
    scale_factor: f64,
) -> Result<Vec<u8>, String> {
    let mut splats = Vec::with_capacity(gaussians.len());
    for g in gaussians {
        let record: GaussianSplatPly = g.to_splat_ply(scale_factor);
        splats.push(record);
    }
    export_splats_to_ply_binary(&splats).map_err(|e| e.to_string())
}

fn extract_colored_solid(
    grid: &math_explorer::applied::isosurface::VoxelGrid,
    cfg: &VolumetricStudioConfig,
    cut: Option<CuttingPlane>,
) -> Result<ColoredMesh, String> {
    let mut ws = IsosurfaceWorkspace::new_lazy(grid.width() + 2, grid.height() + 2);
    let mc = MarchingCubes::new(grid, CentralDifferenceEstimator);
    let mut combined = Mesh::new();
    let mut colors = Vec::new();
    let mut part = Mesh::new();

    if cfg.source == FieldSourceMode::CustomImplicitFormula {
        let closure = SolidClosureConfig::superlevel().with_cutting_plane(cut);
        mc.extract_solid_into(0.0, closure, &mut ws, &mut part)
            .map_err(|e| format!("Solid MarchingCubes error: {e}"))?;
        append_colored_component(
            &part,
            IMPLICIT_LOW_RGB,
            IMPLICIT_HIGH_RGB,
            &mut combined,
            &mut colors,
        );
    } else {
        let iso = (cfg.iso_threshold as f32).clamp(0.001, 2.0);
        let pos_closure = SolidClosureConfig::superlevel().with_cutting_plane(cut);
        mc.extract_solid_into(iso, pos_closure, &mut ws, &mut part)
            .map_err(|e| format!("Solid MarchingCubes error: {e}"))?;
        append_colored_component(&part, POS_LOW_RGB, POS_HIGH_RGB, &mut combined, &mut colors);
        if is_signed_field(cfg) {
            let neg_closure = SolidClosureConfig::sublevel().with_cutting_plane(cut);
            mc.extract_solid_into(-iso, neg_closure, &mut ws, &mut part)
                .map_err(|e| format!("Solid MarchingCubes error: {e}"))?;
            append_colored_component(&part, NEG_LOW_RGB, NEG_HIGH_RGB, &mut combined, &mut colors);
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
