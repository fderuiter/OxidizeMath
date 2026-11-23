//! Cold-path watertight per-vertex colored 3D mesh builder for [`super::CurvatureHeatmap`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::super::export_utils::{
    clamp_grid_res, extrude_open_patch_wall_shell, extrude_surface_wall_shell,
    has_direct_periodic_seams, has_klein_seam_topology, has_polar_singularities,
    is_standard_polar_domain, is_standard_tau_domain, periodic_surface_to_manifold_mesh,
    polar_surface_to_manifold_mesh, DEFAULT_WALL_THICKNESS,
};
use super::{CurvatureHeatmap, CurvatureType, SurfaceType};
use math_commons::math_kernel::colormap::mapped_gradient;
use math_explorer::pure_math::differential_geometry::surface::{
    KleinBottle, ParametricSurface, Sphere, SurfaceAnalysis, Torus,
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
    if c.is_finite() {
        c.clamp(-1e4, 1e4)
    } else {
        0.0
    }
}

impl CurvatureHeatmap {
    pub(super) fn build_colored_mesh_3d(&self) -> Result<ColoredMesh, String> {
        let nu = clamp_grid_res(self.u_resolution);
        let nv = clamp_grid_res(self.v_resolution);
        let (u_range, v_range) = self.custom_sandbox.uv_domain_for(self.surface_type);
        match self.surface_type {
            SurfaceType::Sphere => self.build_surface_colored_mesh(
                &Sphere {
                    radius: self.sphere_radius,
                },
                u_range,
                v_range,
                nu,
                nv,
            ),
            SurfaceType::Torus => self.build_surface_colored_mesh(
                &Torus {
                    major_radius: self.torus_major_radius,
                    minor_radius: self.torus_minor_radius,
                },
                u_range,
                v_range,
                nu,
                nv,
            ),
            SurfaceType::KleinBottle => self.build_surface_colored_mesh(
                &KleinBottle {
                    radius: self.klein_radius,
                },
                u_range,
                v_range,
                nu,
                nv,
            ),
            SurfaceType::CustomHeightfield => self.build_surface_colored_mesh(
                &self.custom_sandbox.build_heightfield(),
                u_range,
                v_range,
                nu,
                nv,
            ),
            SurfaceType::CustomParametric => self.build_surface_colored_mesh(
                &self.custom_sandbox.build_parametric(),
                u_range,
                v_range,
                nu,
                nv,
            ),
        }
    }

    fn build_surface_colored_mesh<S: ParametricSurface>(
        &self,
        surface: &S,
        u_range: (f64, f64),
        v_range: (f64, f64),
        nu: usize,
        nv: usize,
    ) -> Result<ColoredMesh, String> {
        if is_standard_polar_domain(u_range, v_range) && has_polar_singularities(surface) {
            return self.build_polar_colored_mesh(surface, nu, nv);
        }
        if is_standard_tau_domain(u_range, v_range) {
            if has_direct_periodic_seams(surface) {
                return self.build_periodic_colored_mesh(surface, nu, nv);
            }
            if has_klein_seam_topology(surface) {
                return self.build_klein_colored_mesh(surface, nu, nv);
            }
        }
        self.build_open_patch_colored_mesh(surface, u_range, v_range, nu, nv)
    }

    fn build_polar_colored_mesh<S: ParametricSurface>(
        &self,
        surface: &S,
        nu: usize,
        nv: usize,
    ) -> Result<ColoredMesh, String> {
        let mesh = polar_surface_to_manifold_mesh(surface, nu, nv)?;
        let mut curvatures = Vec::with_capacity(mesh.vertices.len());
        curvatures.push(eval_curvature(surface, self.curvature_type, 0.0, 1e-3));
        curvatures.push(eval_curvature(surface, self.curvature_type, 0.0, PI - 1e-3));
        for j in 1..nv {
            let v = (PI * (j as f64 / nv as f64)).clamp(1e-3, PI - 1e-3);
            for i in 0..nu {
                let u = TAU * (i as f64 / nu as f64);
                curvatures.push(eval_curvature(surface, self.curvature_type, u, v));
            }
        }
        Ok(ColoredMesh::new(
            mesh,
            map_curvatures_to_colors(&curvatures),
        ))
    }

    fn build_periodic_colored_mesh<S: ParametricSurface>(
        &self,
        surface: &S,
        nu: usize,
        nv: usize,
    ) -> Result<ColoredMesh, String> {
        let mesh = periodic_surface_to_manifold_mesh(surface, nu, nv)?;
        let mut curvatures = Vec::with_capacity(nu * nv);
        for i in 0..nu {
            let u = TAU * (i as f64 / nu as f64);
            for j in 0..nv {
                let v = TAU * (j as f64 / nv as f64);
                curvatures.push(eval_curvature(surface, self.curvature_type, u, v));
            }
        }
        Ok(ColoredMesh::new(
            mesh,
            map_curvatures_to_colors(&curvatures),
        ))
    }

    fn build_klein_colored_mesh<S: ParametricSurface>(
        &self,
        surface: &S,
        nu: usize,
        nv: usize,
    ) -> Result<ColoredMesh, String> {
        let even_nv = if nv.is_multiple_of(2) {
            nv
        } else {
            (nv + 1).min(math_commons::registry::MAX_GRID_SIZE)
        };
        let mesh = extrude_surface_wall_shell(surface, nu, even_nv, DEFAULT_WALL_THICKNESS)?;
        let mut sheet = Vec::with_capacity(nu * even_nv);
        for i in 0..nu {
            let u = TAU * (i as f64 / nu as f64);
            for j in 0..even_nv {
                let v = TAU * (j as f64 / even_nv as f64);
                sheet.push(eval_curvature(surface, self.curvature_type, u, v));
            }
        }
        let mut all = Vec::with_capacity(2 * sheet.len());
        all.extend_from_slice(&sheet);
        all.extend_from_slice(&sheet);
        Ok(ColoredMesh::new(mesh, map_curvatures_to_colors(&all)))
    }

    fn build_open_patch_colored_mesh<S: ParametricSurface>(
        &self,
        surface: &S,
        u_range: (f64, f64),
        v_range: (f64, f64),
        nu: usize,
        nv: usize,
    ) -> Result<ColoredMesh, String> {
        let mesh = extrude_open_patch_wall_shell(
            surface,
            u_range,
            v_range,
            nu,
            nv,
            DEFAULT_WALL_THICKNESS,
        )?;
        let (u_min, u_max) = u_range;
        let (v_min, v_max) = v_range;
        let mut sheet = Vec::with_capacity((nu + 1) * (nv + 1));
        for i in 0..=nu {
            let u = u_min + (u_max - u_min) * (i as f64 / nu as f64);
            for j in 0..=nv {
                let v = v_min + (v_max - v_min) * (j as f64 / nv as f64);
                sheet.push(eval_curvature(surface, self.curvature_type, u, v));
            }
        }
        let mut all = Vec::with_capacity(2 * sheet.len());
        all.extend_from_slice(&sheet);
        all.extend_from_slice(&sheet);
        Ok(ColoredMesh::new(mesh, map_curvatures_to_colors(&all)))
    }
}
