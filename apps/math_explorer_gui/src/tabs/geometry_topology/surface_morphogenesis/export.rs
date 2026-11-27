// [cite:modular_polynomials_review]
//! Watertight buckled 3D `.stl`, `.ply`, and `.obj` mesh builders with vertex-colored overlays.

use super::solver::MorphogenesisSurfaceEngine;
use super::{SurfaceMorphogenesisTool, SurfaceOverlayMode};
use crate::tabs::geometry_topology::export_utils::{
    clamp_grid_res, extrude_open_patch_wall_shell, extrude_surface_wall_shell,
    periodic_surface_to_manifold_mesh, polar_surface_to_manifold_mesh,
    surface_to_mesh_with_domain_and_thickness,
};
use math_commons::math_kernel::colormap::mapped_gradient;
use math_commons::math_kernel::types::flatten_2d_index;
use math_explorer::pure_math::differential_geometry::buckled_flow::SurfaceTopology;
use math_explorer::pure_math::differential_geometry::surface::{
    ParametricSurface, SurfaceAnalysis,
};
use nalgebra::{Point3, Vector3};
use oxidize_core::mesh::{ColoredMesh, Mesh, VertexColor};
use std::f64::consts::{PI, TAU};

struct BuckledParametricSurface<'a> {
    base: &'a dyn ParametricSurface,
    topology: SurfaceTopology,
    range_u: (f64, f64),
    range_v: (f64, f64),
    nu: usize,
    nv: usize,
    seam_fwd: &'a [usize],
    displacements: &'a [f64],
    overlay: &'a [f64],
    outward_sign: f64,
    klein_normal_flips: bool,
    north_normal: Vector3<f64>,
    south_normal: Vector3<f64>,
}

impl<'a> BuckledParametricSurface<'a> {
    fn from_engine(
        base: &'a dyn ParametricSurface,
        eng: &'a MorphogenesisSurfaceEngine,
        mode: SurfaceOverlayMode,
    ) -> Self {
        let (nu, nv) = (eng.nu, eng.nv);
        let probe_i = (nu / 3).max(1).min(nu - 1);
        let probe_j = (nv / 3).max(1).min(nv - 2);
        let (pu, pv) = eng
            .topology
            .uv_at(eng.range_u, eng.range_v, nu, nv, probe_i, probe_j);
        let raw_n = base.unit_normal(pu, pv);
        let stored_n = eng.flow.buckling_normals()[flatten_2d_index(probe_j, probe_i, nv)];
        let outward_sign = if raw_n.dot(&stored_n) < 0.0 {
            -1.0
        } else {
            1.0
        };

        let klein_normal_flips = if eng.topology == SurfaceTopology::TwistedKlein {
            let dv = (eng.range_v.1 - eng.range_v.0) / nv.max(1) as f64;
            let pj = (nv / 5).max(1).min(nv - 1);
            let vs = eng.range_v.0 + pj as f64 * dv;
            let ve = eng.range_v.0 + eng.flow.seam_forward_map()[pj] as f64 * dv;
            base.unit_normal(eng.range_u.0, vs)
                .dot(&base.unit_normal(eng.range_u.1, ve))
                < 0.0
        } else {
            false
        };

        let north_normal = eng.flow.buckling_normals()[flatten_2d_index(0, 0, nv)];
        let south_normal = eng.flow.buckling_normals()[flatten_2d_index(nv - 1, 0, nv)];

        Self {
            base,
            topology: eng.topology,
            range_u: eng.range_u,
            range_v: eng.range_v,
            nu,
            nv,
            seam_fwd: eng.flow.seam_forward_map(),
            displacements: eng.flow.displacements(),
            overlay: eng.overlay_field(mode),
            outward_sign,
            klein_normal_flips,
            north_normal,
            south_normal,
        }
    }

    fn sample_grid_field(&self, field: &[f64], i_ext: usize, j_ext: usize) -> f64 {
        let (nu, nv) = (self.nu, self.nv);
        match self.topology {
            SurfaceTopology::PeriodicTorus => field[flatten_2d_index(j_ext % nv, i_ext % nu, nv)],
            SurfaceTopology::TwistedKlein => {
                let wj = j_ext % nv;
                if i_ext >= nu {
                    field[flatten_2d_index(self.seam_fwd[wj], i_ext % nu, nv)]
                } else {
                    field[flatten_2d_index(wj, i_ext, nv)]
                }
            }
            SurfaceTopology::PolarSphere => {
                let cj = j_ext.min(nv - 1);
                if cj == 0 || cj + 1 == nv {
                    field[flatten_2d_index(cj, 0, nv)]
                } else {
                    field[flatten_2d_index(cj, i_ext % nu, nv)]
                }
            }
            SurfaceTopology::OpenNeumann => {
                field[flatten_2d_index(j_ext.min(nv - 1), i_ext.min(nu - 1), nv)]
            }
        }
    }

    fn interp_scalar(&self, field: &[f64], u: f64, v: f64) -> f64 {
        let u_span = (self.range_u.1 - self.range_u.0).max(1e-9);
        let v_span = (self.range_v.1 - self.range_v.0).max(1e-9);
        let (fu, i0, i1) = match self.topology {
            SurfaceTopology::PeriodicTorus | SurfaceTopology::PolarSphere => {
                let s = ((u - self.range_u.0).rem_euclid(u_span) / u_span) * self.nu as f64;
                let base = (s.floor() as usize) % self.nu;
                (s.fract(), base, base + 1)
            }
            SurfaceTopology::TwistedKlein => {
                let t = ((u - self.range_u.0) / u_span).clamp(0.0, 1.0) * self.nu as f64;
                let base = (t.floor() as usize).min(self.nu - 1);
                let frac = if t >= self.nu as f64 {
                    1.0
                } else {
                    t - base as f64
                };
                (frac, base, base + 1)
            }
            SurfaceTopology::OpenNeumann => {
                let max_i = self.nu.saturating_sub(1).max(1);
                let t = ((u - self.range_u.0) / u_span).clamp(0.0, 1.0) * max_i as f64;
                let base = (t.floor() as usize).min(max_i - 1);
                (t - base as f64, base, base + 1)
            }
        };

        let (fv, j0, j1) = match self.topology {
            SurfaceTopology::PeriodicTorus | SurfaceTopology::TwistedKlein => {
                let s = ((v - self.range_v.0).rem_euclid(v_span) / v_span) * self.nv as f64;
                let base = (s.floor() as usize) % self.nv;
                (s.fract(), base, base + 1)
            }
            SurfaceTopology::PolarSphere | SurfaceTopology::OpenNeumann => {
                let max_j = self.nv.saturating_sub(1).max(1);
                let t = ((v - self.range_v.0) / v_span).clamp(0.0, 1.0) * max_j as f64;
                let base = (t.floor() as usize).min(max_j - 1);
                (t - base as f64, base, base + 1)
            }
        };

        let v00 = self.sample_grid_field(field, i0, j0);
        let v10 = self.sample_grid_field(field, i1, j0);
        let v01 = self.sample_grid_field(field, i0, j1);
        let v11 = self.sample_grid_field(field, i1, j1);
        let a = v00 + fu * (v10 - v00);
        let b = v01 + fu * (v11 - v01);
        a + fv * (b - a)
    }

    fn buckle_normal_at(&self, u: f64, v: f64) -> Vector3<f64> {
        if self.topology == SurfaceTopology::PolarSphere {
            if (v - self.range_v.0).abs() <= 1e-6 {
                return self.north_normal;
            }
            if (v - self.range_v.1).abs() <= 1e-6 {
                return self.south_normal;
            }
        }
        let raw = self.base.unit_normal(u, v);
        let clean = if raw.x.is_finite() && raw.y.is_finite() && raw.z.is_finite() {
            raw * self.outward_sign
        } else {
            Vector3::z()
        };
        if self.klein_normal_flips {
            let u_span = (self.range_u.1 - self.range_u.0).max(1e-9);
            let phase = PI * ((u - self.range_u.0) / u_span).clamp(0.0, 1.0);
            clean * phase.cos()
        } else {
            clean
        }
    }
}

impl ParametricSurface for BuckledParametricSurface<'_> {
    fn position(&self, u: f64, v: f64) -> Point3<f64> {
        let base_pt = self.base.position(u, v);
        let disp = self.interp_scalar(self.displacements, u, v);
        if disp.abs() <= 1e-14 {
            return base_pt;
        }
        base_pt + self.buckle_normal_at(u, v) * disp
    }
}

pub(super) fn map_scalars_to_vertex_colors(values: &[f64]) -> Vec<VertexColor> {
    let mut min_v = f64::INFINITY;
    let mut max_v = f64::NEG_INFINITY;
    for &v in values {
        if v.is_finite() {
            min_v = min_v.min(v);
            max_v = max_v.max(v);
        }
    }
    if !min_v.is_finite() || !max_v.is_finite() {
        min_v = 0.0;
        max_v = 1.0;
    }
    let span = max_v - min_v;
    values
        .iter()
        .map(|&v| {
            let rgb = if !v.is_finite() || span.abs() < 1e-10 {
                mapped_gradient(0.5, 0.0, 1.0)
            } else {
                mapped_gradient(v, min_v, max_v)
            };
            VertexColor::new(rgb[0], rgb[1], rgb[2])
        })
        .collect()
}

impl SurfaceMorphogenesisTool {
    pub(super) fn build_buckled_mesh_3d(&self) -> Result<Mesh, String> {
        let base = self.create_base_surface();
        let fallback_eng;
        let eng = if let Some(existing) = self.engine.as_ref() {
            existing
        } else {
            fallback_eng = MorphogenesisSurfaceEngine::build(self);
            &fallback_eng
        };
        let buckled = BuckledParametricSurface::from_engine(base.as_ref(), eng, self.overlay_mode);
        let (u_dom, v_dom) = match eng.topology {
            SurfaceTopology::PolarSphere
            | SurfaceTopology::PeriodicTorus
            | SurfaceTopology::TwistedKlein => ((0.0, TAU), (0.0, TAU)),
            SurfaceTopology::OpenNeumann => (eng.range_u, eng.range_v),
        };
        surface_to_mesh_with_domain_and_thickness(
            &buckled,
            u_dom,
            v_dom,
            clamp_grid_res(self.u_resolution),
            clamp_grid_res(self.v_resolution),
            self.wall_thickness,
        )
    }

    pub(super) fn build_buckled_colored_mesh_3d(&self) -> Result<ColoredMesh, String> {
        let base = self.create_base_surface();
        let fallback_eng;
        let eng = if let Some(existing) = self.engine.as_ref() {
            existing
        } else {
            fallback_eng = MorphogenesisSurfaceEngine::build(self);
            &fallback_eng
        };
        let buckled = BuckledParametricSurface::from_engine(base.as_ref(), eng, self.overlay_mode);
        let nu = clamp_grid_res(self.u_resolution);
        let nv = clamp_grid_res(self.v_resolution);

        match eng.topology {
            SurfaceTopology::PolarSphere => build_polar_colored(&buckled, nu, nv),
            SurfaceTopology::PeriodicTorus => build_periodic_colored(&buckled, nu, nv),
            SurfaceTopology::TwistedKlein => {
                build_klein_colored(&buckled, nu, nv, self.wall_thickness)
            }
            SurfaceTopology::OpenNeumann => build_open_colored(
                &buckled,
                eng.range_u,
                eng.range_v,
                nu,
                nv,
                self.wall_thickness,
            ),
        }
    }
}

fn build_polar_colored(
    surf: &BuckledParametricSurface<'_>,
    nu: usize,
    nv: usize,
) -> Result<ColoredMesh, String> {
    let mesh = polar_surface_to_manifold_mesh(surf, nu, nv)?;
    let mut vals = Vec::with_capacity(mesh.vertices.len());
    vals.push(surf.interp_scalar(surf.overlay, 0.0, 0.0));
    vals.push(surf.interp_scalar(surf.overlay, 0.0, PI));
    for j in 1..nv {
        let v = PI * (j as f64 / nv as f64);
        for i in 0..nu {
            let u = TAU * (i as f64 / nu as f64);
            vals.push(surf.interp_scalar(surf.overlay, u, v));
        }
    }
    Ok(ColoredMesh::new(mesh, map_scalars_to_vertex_colors(&vals)))
}

fn build_periodic_colored(
    surf: &BuckledParametricSurface<'_>,
    nu: usize,
    nv: usize,
) -> Result<ColoredMesh, String> {
    let mesh = periodic_surface_to_manifold_mesh(surf, nu, nv)?;
    let mut vals = Vec::with_capacity(nu * nv);
    for i in 0..nu {
        let u = TAU * (i as f64 / nu as f64);
        for j in 0..nv {
            let v = TAU * (j as f64 / nv as f64);
            vals.push(surf.interp_scalar(surf.overlay, u, v));
        }
    }
    Ok(ColoredMesh::new(mesh, map_scalars_to_vertex_colors(&vals)))
}

fn build_klein_colored(
    surf: &BuckledParametricSurface<'_>,
    nu: usize,
    nv: usize,
    thickness: f64,
) -> Result<ColoredMesh, String> {
    let even_nv = if nv.is_multiple_of(2) {
        nv
    } else {
        (nv + 1).min(math_commons::registry::MAX_GRID_SIZE)
    };
    let mesh = extrude_surface_wall_shell(surf, nu, even_nv, thickness)?;
    let mut sheet = Vec::with_capacity(nu * even_nv);
    for i in 0..nu {
        let u = TAU * (i as f64 / nu as f64);
        for j in 0..even_nv {
            let v = TAU * (j as f64 / even_nv as f64);
            sheet.push(surf.interp_scalar(surf.overlay, u, v));
        }
    }
    let mut all = Vec::with_capacity(2 * sheet.len());
    all.extend_from_slice(&sheet);
    all.extend_from_slice(&sheet);
    Ok(ColoredMesh::new(mesh, map_scalars_to_vertex_colors(&all)))
}

fn build_open_colored(
    surf: &BuckledParametricSurface<'_>,
    u_range: (f64, f64),
    v_range: (f64, f64),
    nu: usize,
    nv: usize,
    thickness: f64,
) -> Result<ColoredMesh, String> {
    let mesh = extrude_open_patch_wall_shell(surf, u_range, v_range, nu, nv, thickness)?;
    let mut sheet = Vec::with_capacity((nu + 1) * (nv + 1));
    for i in 0..=nu {
        let u = u_range.0 + (u_range.1 - u_range.0) * (i as f64 / nu as f64);
        for j in 0..=nv {
            let v = v_range.0 + (v_range.1 - v_range.0) * (j as f64 / nv as f64);
            sheet.push(surf.interp_scalar(surf.overlay, u, v));
        }
    }
    let mut all = Vec::with_capacity(2 * sheet.len());
    all.extend_from_slice(&sheet);
    all.extend_from_slice(&sheet);
    Ok(ColoredMesh::new(mesh, map_scalars_to_vertex_colors(&all)))
}
