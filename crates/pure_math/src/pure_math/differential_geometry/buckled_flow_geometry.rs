// [cite:modular_polynomials_review]
//! Geometry sampling, neighbor wrapping, and outward normal orientation helpers for `BuckledSurfaceFlow`.

use super::{SurfaceTopology, detect_twisted_seam_map};
use crate::pure_math::differential_geometry::surface::{ParametricSurface, SurfaceAnalysis};
use math_commons::math_kernel::types::flatten_2d_index;
use nalgebra::{Point3, Vector3};
use std::f64::consts::PI;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct StencilNeighbors {
    pub ip: usize,
    pub im: usize,
    pub jp: usize,
    pub jm: usize,
    pub pp: usize,
    pub pm: usize,
    pub mp: usize,
    pub mm: usize,
}

pub(super) struct SampledSurfaceGeometry {
    pub seam_fwd: Vec<usize>,
    pub neighbors: Vec<StencilNeighbors>,
    pub ref_points: Vec<Point3<f64>>,
    pub buckling_normals: Vec<Vector3<f64>>,
    pub min_area_element: f64,
    pub north_meridional_g: f64,
    pub south_meridional_g: f64,
}

pub(super) fn build_sampled_geometry<S: ParametricSurface + ?Sized>(
    surface: &S,
    topology: SurfaceTopology,
    range_u: (f64, f64),
    range_v: (f64, f64),
    nu: usize,
    nv: usize,
) -> SampledSurfaceGeometry {
    let (seam_fwd, seam_rev) = if topology == SurfaceTopology::TwistedKlein {
        detect_twisted_seam_map(surface, range_u, range_v, nv)
    } else {
        ((0..nv).collect(), (0..nv).collect())
    };
    let normal_flips = topology == SurfaceTopology::TwistedKlein
        && seam_normal_reverses(surface, range_u, range_v, nv, &seam_fwd);
    let neighbors = build_neighbor_table(topology, nu, nv, &seam_fwd, &seam_rev);
    let (ref_points, mut buckling_normals, min_area_element) =
        sample_reference_geometry(surface, topology, normal_flips, range_u, range_v, nu, nv);
    orient_normals_outward(topology, nu, nv, &ref_points, &mut buckling_normals);
    let (north_meridional_g, south_meridional_g) =
        polar_meridional_metrics(surface, topology, range_u, range_v);

    SampledSurfaceGeometry {
        seam_fwd,
        neighbors,
        ref_points,
        buckling_normals,
        min_area_element,
        north_meridional_g,
        south_meridional_g,
    }
}

fn wrap_neighbor(
    topology: SurfaceTopology,
    nu: usize,
    nv: usize,
    seam_fwd: &[usize],
    seam_rev: &[usize],
    i: isize,
    j: isize,
) -> usize {
    match topology {
        SurfaceTopology::PeriodicTorus => {
            let wi = i.rem_euclid(nu as isize) as usize;
            let wj = j.rem_euclid(nv as isize) as usize;
            flatten_2d_index(wj, wi, nv)
        }
        SurfaceTopology::TwistedKlein => {
            let base_j = j.rem_euclid(nv as isize) as usize;
            let (wi, wj) = if i < 0 {
                ((i + nu as isize) as usize, seam_rev[base_j])
            } else if i >= nu as isize {
                ((i - nu as isize) as usize, seam_fwd[base_j])
            } else {
                (i as usize, base_j)
            };
            flatten_2d_index(wj, wi, nv)
        }
        SurfaceTopology::PolarSphere => {
            let wi = i.rem_euclid(nu as isize) as usize;
            let wj = j.clamp(0, (nv - 1) as isize) as usize;
            flatten_2d_index(wj, wi, nv)
        }
        SurfaceTopology::OpenNeumann => {
            let wi = i.clamp(0, (nu - 1) as isize) as usize;
            let wj = j.clamp(0, (nv - 1) as isize) as usize;
            flatten_2d_index(wj, wi, nv)
        }
    }
}

fn build_neighbor_table(
    topology: SurfaceTopology,
    nu: usize,
    nv: usize,
    seam_fwd: &[usize],
    seam_rev: &[usize],
) -> Vec<StencilNeighbors> {
    let mut out = Vec::with_capacity(nu * nv);
    for i in 0..nu {
        let ii = i as isize;
        for j in 0..nv {
            let jj = j as isize;
            out.push(StencilNeighbors {
                ip: wrap_neighbor(topology, nu, nv, seam_fwd, seam_rev, ii + 1, jj),
                im: wrap_neighbor(topology, nu, nv, seam_fwd, seam_rev, ii - 1, jj),
                jp: wrap_neighbor(topology, nu, nv, seam_fwd, seam_rev, ii, jj + 1),
                jm: wrap_neighbor(topology, nu, nv, seam_fwd, seam_rev, ii, jj - 1),
                pp: wrap_neighbor(topology, nu, nv, seam_fwd, seam_rev, ii + 1, jj + 1),
                pm: wrap_neighbor(topology, nu, nv, seam_fwd, seam_rev, ii + 1, jj - 1),
                mp: wrap_neighbor(topology, nu, nv, seam_fwd, seam_rev, ii - 1, jj + 1),
                mm: wrap_neighbor(topology, nu, nv, seam_fwd, seam_rev, ii - 1, jj - 1),
            });
        }
    }
    out
}

fn seam_normal_reverses<S: ParametricSurface + ?Sized>(
    surface: &S,
    range_u: (f64, f64),
    range_v: (f64, f64),
    nv: usize,
    seam_fwd: &[usize],
) -> bool {
    let dv = (range_v.1 - range_v.0) / nv.max(1) as f64;
    let probe_j = (nv / 5).max(1).min(nv.saturating_sub(1));
    let v_start = range_v.0 + probe_j as f64 * dv;
    let v_end = range_v.0 + seam_fwd[probe_j] as f64 * dv;
    surface
        .unit_normal(range_u.0, v_start)
        .dot(&surface.unit_normal(range_u.1, v_end))
        < 0.0
}

fn sample_reference_geometry<S: ParametricSurface + ?Sized>(
    surface: &S,
    topology: SurfaceTopology,
    normal_flips: bool,
    range_u: (f64, f64),
    range_v: (f64, f64),
    nu: usize,
    nv: usize,
) -> (Vec<Point3<f64>>, Vec<Vector3<f64>>, f64) {
    let mut pts = Vec::with_capacity(nu * nv);
    let mut normals = Vec::with_capacity(nu * nv);
    let mut min_area = f64::INFINITY;
    let u_span = (range_u.1 - range_u.0).max(1e-9);
    let (du, dv) = topology.grid_spacings(range_u, range_v, nu, nv);

    for i in 0..nu {
        for j in 0..nv {
            let (u, v) = topology.uv_at(range_u, range_v, nu, nv, i, j);
            pts.push(surface.position(u, v));
            let is_pole = topology == SurfaceTopology::PolarSphere && (j == 0 || j + 1 == nv);
            let raw_n = if is_pole {
                let probe_v = if j == 0 { v + 0.25 * dv } else { v - 0.25 * dv };
                polar_axis_normal(surface, range_u, nu, v, probe_v)
            } else {
                let area = surface.area_element(u, v);
                let cell_area = area * du * dv;
                if cell_area.is_finite() && cell_area > 1e-9 {
                    min_area = min_area.min(cell_area);
                }
                let n = surface.unit_normal(u, v);
                if n.x.is_finite() && n.y.is_finite() && n.z.is_finite() {
                    n
                } else {
                    Vector3::z()
                }
            };
            let n_adj = if normal_flips {
                let phase = PI * (u - range_u.0) / u_span;
                raw_n * phase.cos()
            } else {
                raw_n
            };
            normals.push(n_adj);
        }
    }
    if !min_area.is_finite() {
        min_area = (du * dv).max(1e-4);
    }
    (pts, normals, min_area)
}

fn polar_axis_normal<S: ParametricSurface + ?Sized>(
    surface: &S,
    range_u: (f64, f64),
    nu: usize,
    pole_v: f64,
    probe_v: f64,
) -> Vector3<f64> {
    let p_pole = surface.position(range_u.0, pole_v);
    let mut ring_center = Vector3::zeros();
    let samples = nu.clamp(4, 8);
    let step_u = (range_u.1 - range_u.0) / samples as f64;
    for k in 0..samples {
        ring_center += surface
            .position(range_u.0 + k as f64 * step_u, probe_v)
            .coords;
    }
    ring_center /= samples as f64;
    let axis = p_pole.coords - ring_center;
    let norm = axis.norm();
    if norm > 1e-12 {
        axis / norm
    } else {
        Vector3::z()
    }
}

fn orient_normals_outward(
    topology: SurfaceTopology,
    nu: usize,
    nv: usize,
    pts: &[Point3<f64>],
    normals: &mut [Vector3<f64>],
) {
    if !matches!(
        topology,
        SurfaceTopology::PeriodicTorus | SurfaceTopology::PolarSphere
    ) {
        return;
    }
    let mut centroid = Vector3::zeros();
    for p in pts {
        centroid += p.coords;
    }
    centroid /= pts.len().max(1) as f64;

    let mut radial_dot = 0.0;
    let j_range = if topology == SurfaceTopology::PolarSphere {
        1..(nv - 1)
    } else {
        0..nv
    };
    for i in 0..nu {
        for j in j_range.clone() {
            let idx = flatten_2d_index(j, i, nv);
            radial_dot += (pts[idx].coords - centroid).dot(&normals[idx]);
        }
    }
    if radial_dot < 0.0 {
        for i in 0..nu {
            for j in j_range.clone() {
                let idx = flatten_2d_index(j, i, nv);
                normals[idx] = -normals[idx];
            }
        }
    }
}

fn polar_meridional_metrics<S: ParametricSurface + ?Sized>(
    surface: &S,
    topology: SurfaceTopology,
    range_u: (f64, f64),
    range_v: (f64, f64),
) -> (f64, f64) {
    if topology != SurfaceTopology::PolarSphere {
        return (1.0, 1.0);
    }
    let (_, _, g_north) = surface.first_fundamental_form(range_u.0, range_v.0 + 1e-4);
    let (_, _, g_south) = surface.first_fundamental_form(range_u.0, range_v.1 - 1e-4);
    (g_north.max(1e-6), g_south.max(1e-6))
}
