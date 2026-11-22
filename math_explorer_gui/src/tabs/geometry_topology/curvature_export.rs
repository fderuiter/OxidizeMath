//! Cold-path watertight per-vertex colored 3D mesh builder for [`super::CurvatureHeatmap`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::super::export_utils::{
    extrude_surface_wall_shell, periodic_surface_to_manifold_mesh, polar_surface_to_manifold_mesh,
    DEFAULT_WALL_THICKNESS,
};
use super::{CurvatureHeatmap, CurvatureType, SurfaceType};
use math_commons::math_kernel::colormap::mapped_gradient;
use math_explorer::pure_math::differential_geometry::surface::{
    KleinBottle, Sphere, SurfaceAnalysis, Torus,
};
use oxidize_core::mesh::{ColoredMesh, VertexColor};
use std::f64::consts::{PI, TAU};

fn map_curvatures_to_colors(values: &[f64]) -> Vec<VertexColor> {
    let mut min_c = f64::INFINITY;
    let mut max_c = f64::NEG_INFINITY;
    for &c in values {
        if c.is_finite() {
            min_c = min_c.min(c);
            max_c = max_c.max(c);
        }
    }
    if !min_c.is_finite() || !max_c.is_finite() {
        min_c = 0.0;
        max_c = 1.0;
    }
    let span = max_c - min_c;
    values
        .iter()
        .map(|&c| {
            let rgb = if !c.is_finite() || span.abs() < 1e-10 {
                mapped_gradient(0.5, 0.0, 1.0)
            } else {
                mapped_gradient(c, min_c, max_c)
            };
            VertexColor::new(rgb[0], rgb[1], rgb[2])
        })
        .collect()
}

fn eval_curvature<S: SurfaceAnalysis>(
    surface: &S,
    curvature_type: CurvatureType,
    u: f64,
    v: f64,
) -> f64 {
    let c = match curvature_type {
        CurvatureType::Gaussian => surface.gaussian_curvature(u, v),
        CurvatureType::Mean => surface.mean_curvature(u, v),
    };
    if c.is_finite() { c } else { 0.0 }
}

impl CurvatureHeatmap {
    pub(super) fn build_colored_mesh_3d(&self) -> Result<ColoredMesh, String> {
        let max_res = math_commons::registry::MAX_GRID_SIZE;
        let nu = self.u_resolution.clamp(4, max_res);
        let nv = self.v_resolution.clamp(4, max_res);
        match self.surface_type {
            SurfaceType::Sphere => self.build_sphere_colored_mesh(nu, nv),
            SurfaceType::Torus => self.build_torus_colored_mesh(nu, nv),
            SurfaceType::KleinBottle => self.build_klein_colored_mesh(nu, nv),
        }
    }

    fn build_sphere_colored_mesh(&self, nu: usize, nv: usize) -> Result<ColoredMesh, String> {
        let surface = Sphere {
            radius: self.sphere_radius,
        };
        let mesh = polar_surface_to_manifold_mesh(&surface, nu, nv)?;
        let mut curvatures = Vec::with_capacity(mesh.vertices.len());
        curvatures.push(eval_curvature(&surface, self.curvature_type, 0.0, 1e-3));
        curvatures.push(eval_curvature(
            &surface,
            self.curvature_type,
            0.0,
            PI - 1e-3,
        ));
        for j in 1..nv {
            let v = (PI * (j as f64 / nv as f64)).clamp(1e-3, PI - 1e-3);
            for i in 0..nu {
                let u = TAU * (i as f64 / nu as f64);
                curvatures.push(eval_curvature(&surface, self.curvature_type, u, v));
            }
        }
        let colors = map_curvatures_to_colors(&curvatures);
        Ok(ColoredMesh::new(mesh, colors))
    }

    fn build_torus_colored_mesh(&self, nu: usize, nv: usize) -> Result<ColoredMesh, String> {
        let surface = Torus {
            major_radius: self.torus_major_radius,
            minor_radius: self.torus_minor_radius,
        };
        let mesh = periodic_surface_to_manifold_mesh(&surface, nu, nv)?;
        let mut curvatures = Vec::with_capacity(nu * nv);
        for i in 0..nu {
            let u = TAU * (i as f64 / nu as f64);
            for j in 0..nv {
                let v = TAU * (j as f64 / nv as f64);
                curvatures.push(eval_curvature(&surface, self.curvature_type, u, v));
            }
        }
        let colors = map_curvatures_to_colors(&curvatures);
        Ok(ColoredMesh::new(mesh, colors))
    }

    fn build_klein_colored_mesh(&self, nu: usize, nv: usize) -> Result<ColoredMesh, String> {
        let surface = KleinBottle {
            radius: self.klein_radius,
        };
        let even_nv = if nv.is_multiple_of(2) {
            nv
        } else {
            (nv + 1).min(math_commons::registry::MAX_GRID_SIZE)
        };
        let mesh = extrude_surface_wall_shell(&surface, nu, even_nv, DEFAULT_WALL_THICKNESS)?;
        let mut sheet_curvatures = Vec::with_capacity(nu * even_nv);
        for i in 0..nu {
            let u = TAU * (i as f64 / nu as f64);
            for j in 0..even_nv {
                let v = TAU * (j as f64 / even_nv as f64);
                sheet_curvatures.push(eval_curvature(&surface, self.curvature_type, u, v));
            }
        }
        let mut all_curvatures = Vec::with_capacity(2 * sheet_curvatures.len());
        all_curvatures.extend_from_slice(&sheet_curvatures);
        all_curvatures.extend_from_slice(&sheet_curvatures);
        let colors = map_curvatures_to_colors(&all_curvatures);
        Ok(ColoredMesh::new(mesh, colors))
    }
}
