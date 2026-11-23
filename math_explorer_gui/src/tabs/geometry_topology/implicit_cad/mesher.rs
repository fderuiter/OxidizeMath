//! Pre-allocated voxel sampling, boundary-sealed Marching Cubes meshing, and 3D viewport shading.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::framework::Camera3D;
use eframe::egui::Color32;
use math_commons::ast::{Expr, parse_equation};
use math_explorer::applied::isosurface::marching_cubes::MarchingCubes;
use math_explorer::applied::isosurface::{
    CentralDifferenceEstimator, Mesh, Point3D, SliceEdgeCache, VoxelGrid,
};
use oxidize_core::mesh::{MeshDiagnostics, analyze_mesh};

/// Minimum interactive grid resolution per axis.
pub const MIN_GRID_RES: usize = 16;
/// Maximum interactive grid resolution per axis (`96^3 = 884,736` samples).
pub const MAX_GRID_RES: usize = 96;
/// Interactive viewport facet cap keeping immediate-mode `egui_plot` orbiting responsive.
const MAX_VIEWPORT_FACETS: usize = 14_000;

/// Physical meshing parameters for [`ImplicitCadWorkspace`].
#[derive(Debug, Clone, PartialEq)]
pub struct ImplicitCadParams {
    /// Implicit scalar field expression in `(x, y, z)`.
    pub equation: String,
    /// Cubic bounding box side length in physical millimeters.
    pub box_size_mm: f64,
    /// Sheet wall thickness `w` in millimeters (`0.0` extracts solid `{f < 0}`, `> 0` extracts `{|f| - w/2 < 0}`).
    pub shell_thickness_mm: f64,
    /// Samples per axis (clamped to `16..=96`).
    pub resolution: usize,
    /// Whether to clamp all six outer grid faces so clipped lattices form watertight solids.
    pub seal_boundaries: bool,
}

/// Projected 2D triangle ready for back-to-front painter submission in [`egui_plot`].
#[derive(Debug, Clone, Copy)]
pub struct ProjectedFacet {
    /// Camera-space depth centroid (`+z` closer to viewer).
    pub depth: f64,
    /// Projected 2D vertices in normalized viewport coordinates.
    pub pts: [[f64; 2]; 3],
    /// Shaded fill color.
    pub fill: Color32,
}

/// Reusable pre-allocated buffers for zero-allocation-inner-loop implicit meshing.
pub struct ImplicitCadWorkspace {
    grid: VoxelGrid,
    edge_cache: SliceEdgeCache,
    /// Latest extracted millimeter triangle mesh.
    pub mesh: Mesh,
    /// Cached manifold diagnostics for `mesh`.
    pub diagnostics: MeshDiagnostics,
    /// Latest equation parse error, if any.
    pub last_error: Option<String>,
    last_built: Option<ImplicitCadParams>,
    projected_buf: Vec<ProjectedFacet>,
}

impl Default for ImplicitCadWorkspace {
    fn default() -> Self {
        let res = 36usize;
        let grid = VoxelGrid::builder()
            .dimensions(res, res, res)
            .data(vec![1.0; res * res * res])
            .voxel_size(Point3D::new(1.0, 1.0, 1.0))
            .origin(Point3D::new(-20.0, -20.0, -20.0))
            .clamp_exterior(Some(0.0))
            .build()
            .expect("valid default voxel grid");
        Self {
            grid,
            edge_cache: SliceEdgeCache::preallocated(res, res),
            mesh: Mesh {
                vertices: Vec::with_capacity(16_384),
                normals: Vec::with_capacity(16_384),
                indices: Vec::with_capacity(65_536),
            },
            diagnostics: MeshDiagnostics::default(),
            last_error: None,
            last_built: None,
            projected_buf: Vec::with_capacity(MAX_VIEWPORT_FACETS),
        }
    }
}

impl ImplicitCadWorkspace {
    /// Returns `true` when the cached mesh already corresponds to `params`.
    pub fn matches_params(&self, params: &ImplicitCadParams) -> bool {
        let mut normalized = params.clone();
        normalized.resolution = normalized.resolution.clamp(MIN_GRID_RES, MAX_GRID_RES);
        self.last_built.as_ref() == Some(&normalized)
    }

    /// Rebuilds the voxel field and Marching Cubes mesh only when parameters have changed.
    pub fn ensure_up_to_date(&mut self, params: &ImplicitCadParams) {
        let mut normalized = params.clone();
        normalized.resolution = normalized.resolution.clamp(MIN_GRID_RES, MAX_GRID_RES);
        if self.last_built.as_ref() == Some(&normalized) {
            return;
        }
        self.rebuild(&normalized);
        self.last_built = Some(normalized);
    }

    fn rebuild(&mut self, params: &ImplicitCadParams) {
        let parsed = match parse_equation(&params.equation) {
            Ok((_lhs, expr)) => expr,
            Err(err) => {
                self.last_error = Some(err.user_message_for_input(&params.equation));
                return;
            }
        };
        self.last_error = None;

        let res = params.resolution.clamp(MIN_GRID_RES, MAX_GRID_RES);
        let box_mm = params.box_size_mm.clamp(4.0, 200.0);
        let half_mm = box_mm * 0.5;
        let step_mm = box_mm / (res.saturating_sub(1).max(1) as f64);
        let step_f32 = step_mm as f32;
        let half_f32 = half_mm as f32;

        let _ = self.grid.resize_dimensions(res, res, res);
        let _ = self
            .grid
            .set_voxel_size(Point3D::new(step_f32, step_f32, step_f32));
        self.grid
            .set_origin(Point3D::new(-half_f32, -half_f32, -half_f32));

        sample_ast_into_grid(
            &parsed,
            res,
            -half_mm,
            step_mm,
            params.shell_thickness_mm.max(0.0),
            self.grid.data_mut(),
        );
        self.grid
            .set_clamp_exterior(params.seal_boundaries.then_some(0.0));

        let extractor = MarchingCubes::new(&self.grid, CentralDifferenceEstimator);
        if let Err(e) = extractor.extract_into(0.0, &mut self.edge_cache, &mut self.mesh) {
            self.last_error = Some(format!("Isosurface extraction error: {e}"));
            self.diagnostics = MeshDiagnostics::default();
            return;
        }
        self.diagnostics = analyze_mesh(&self.mesh);
    }

    /// Projects and depth-sorts mesh facets for `Plot3DViewport`.
    /// Returns `(facets_slice, total_front_facets)`.
    pub fn project_facets(
        &mut self,
        camera: &Camera3D,
        box_size_mm: f64,
        cull_backfaces: bool,
    ) -> (&[ProjectedFacet], usize) {
        self.projected_buf.clear();
        let norm_scale = 1.15 / (box_size_mm.max(4.0) * 0.5);
        let zoom = f64::from(camera.zoom) * norm_scale;
        let total_tris = self.mesh.indices.len() / 3;
        let stride = (total_tris / MAX_VIEWPORT_FACETS).max(1);
        let mut front_count = 0usize;

        for (tri_idx, tri) in self.mesh.indices.as_chunks::<3>().0.iter().enumerate() {
            if let Some(facet) = build_projected_facet(
                &self.mesh,
                tri,
                camera,
                zoom,
                cull_backfaces,
                tri_idx % stride == 0,
                &mut front_count,
            ) {
                self.projected_buf.push(facet);
            }
        }

        self.projected_buf
            .sort_unstable_by(|a, b| a.depth.total_cmp(&b.depth));
        (&self.projected_buf, front_count)
    }
}

fn sample_ast_into_grid(
    expr: &Expr,
    res: usize,
    min_coord: f64,
    step: f64,
    shell_thickness_mm: f64,
    out: &mut [f32],
) {
    let half_shell = 0.5 * shell_thickness_mm;
    let use_shell = shell_thickness_mm > 0.0;
    let mut cursor = 0usize;

    for iz in 0..res {
        let z = min_coord + (iz as f64) * step;
        for iy in 0..res {
            let y = min_coord + (iy as f64) * step;
            for ix in 0..res {
                let x = min_coord + (ix as f64) * step;
                let raw = expr.eval_3d(x, y, z);
                let field = if !raw.is_finite() {
                    1.0
                } else if use_shell {
                    raw.abs() - half_shell
                } else {
                    raw
                };
                out[cursor] = field as f32;
                cursor += 1;
            }
        }
    }
}

fn build_projected_facet(
    mesh: &Mesh,
    tri: &[usize],
    camera: &Camera3D,
    zoom: f64,
    cull_backfaces: bool,
    keep_sample: bool,
    front_count: &mut usize,
) -> Option<ProjectedFacet> {
    let v0 = mesh.vertices[tri[0]];
    let v1 = mesh.vertices[tri[1]];
    let v2 = mesh.vertices[tri[2]];
    let c0 = camera.rotate_to_camera_space(&[f64::from(v0.x), f64::from(v0.y), f64::from(v0.z)]);
    let c1 = camera.rotate_to_camera_space(&[f64::from(v1.x), f64::from(v1.y), f64::from(v1.z)]);
    let c2 = camera.rotate_to_camera_space(&[f64::from(v2.x), f64::from(v2.y), f64::from(v2.z)]);

    let e1 = [c1[0] - c0[0], c1[1] - c0[1], c1[2] - c0[2]];
    let e2 = [c2[0] - c0[0], c2[1] - c0[1], c2[2] - c0[2]];
    let nx = e1[1] * e2[2] - e1[2] * e2[1];
    let ny = e1[2] * e2[0] - e1[0] * e2[2];
    let nz = e1[0] * e2[1] - e1[1] * e2[0];
    if cull_backfaces && nz <= 0.0 {
        return None;
    }
    *front_count += 1;
    if !keep_sample {
        return None;
    }

    let n_len = (nx * nx + ny * ny + nz * nz).sqrt();
    let (ux, uy, uz) = if n_len > 1.0e-12 {
        let sign = if !cull_backfaces && nz < 0.0 {
            -1.0
        } else {
            1.0
        };
        (sign * nx / n_len, sign * ny / n_len, sign * nz / n_len)
    } else {
        (0.0, 0.0, 1.0)
    };

    let fill = shade_lambertian(ux, uy, uz);
    let depth = (c0[2] + c1[2] + c2[2]) * (1.0 / 3.0);
    let pts = [
        [c0[0] * zoom, c0[1] * zoom],
        [c1[0] * zoom, c1[1] * zoom],
        [c2[0] * zoom, c2[1] * zoom],
    ];
    Some(ProjectedFacet { depth, pts, fill })
}

fn shade_lambertian(nx: f64, ny: f64, nz: f64) -> Color32 {
    // Normalized key light (0.37, 0.56, 0.74) + subtle cool sky fill
    let key = (nx * 0.37 + ny * 0.56 + nz * 0.74).max(0.0);
    let rim = (1.0 - nz.clamp(0.0, 1.0)).powi(2) * 0.14;
    let intensity = (0.24 + 0.68 * key + rim).clamp(0.15, 1.0) as f32;
    let r = (0.24 + 0.52 * intensity).clamp(0.0, 1.0);
    let g = (0.48 + 0.44 * intensity).clamp(0.0, 1.0);
    let b = (0.66 + 0.32 * intensity).clamp(0.0, 1.0);
    Color32::from(eframe::egui::Rgba::from_rgba_unmultiplied(r, g, b, 1.0))
}
