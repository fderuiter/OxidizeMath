// [cite:graph_parameters_rust]
//! Precomputed divergence-form Laplace-Beltrami 9-point stencils and Riemannian metric diagnostics.

use math_commons::math_kernel::types::flatten_2d_index;
use nalgebra::Matrix2;
use pure_math::pure_math::differential_geometry::buckled_flow::{
    SurfaceTopology, detect_twisted_seam_map,
};
use pure_math::pure_math::differential_geometry::surface::{ParametricSurface, SurfaceAnalysis};

#[derive(Debug, Clone, Copy)]
pub(super) struct CellStencil {
    pub idx: [usize; 8],
    pub weight: [f64; 8],
    pub diag: f64,
}

#[derive(Debug, Clone)]
pub(super) struct PrecomputedSurfaceMetrics {
    pub stencils: Vec<CellStencil>,
    pub area_elements: Vec<f64>,
    pub gaussian_curvature: Vec<f64>,
    pub metric_anisotropy: Vec<f64>,
    pub seam_fwd: Vec<usize>,
    pub min_area_element: f64,
    pub max_spectral_rate: f64,
    pub north_pole_rate: f64,
    pub south_pole_rate: f64,
}

#[derive(Clone, Copy)]
struct GridTopologyCtx<'a> {
    topology: SurfaceTopology,
    range_u: (f64, f64),
    range_v: (f64, f64),
    nu: usize,
    nv: usize,
    du: f64,
    dv: f64,
    seam_fwd: &'a [usize],
    seam_rev: &'a [usize],
    seam_v_sign: f64,
}

impl GridTopologyCtx<'_> {
    fn map_index(&self, i: isize, j: isize) -> usize {
        let (nu, nv) = (self.nu as isize, self.nv as isize);
        match self.topology {
            SurfaceTopology::PeriodicTorus => flatten_2d_index(
                j.rem_euclid(nv) as usize,
                i.rem_euclid(nu) as usize,
                self.nv,
            ),
            SurfaceTopology::TwistedKlein => {
                let base_j = j.rem_euclid(nv) as usize;
                let (wi, wj) = if i < 0 {
                    ((i + nu) as usize, self.seam_rev[base_j])
                } else if i >= nu {
                    ((i - nu) as usize, self.seam_fwd[base_j])
                } else {
                    (i as usize, base_j)
                };
                flatten_2d_index(wj, wi, self.nv)
            }
            SurfaceTopology::PolarSphere => flatten_2d_index(
                j.clamp(0, nv - 1) as usize,
                i.rem_euclid(nu) as usize,
                self.nv,
            ),
            SurfaceTopology::OpenNeumann => flatten_2d_index(
                j.clamp(0, nv - 1) as usize,
                i.clamp(0, nu - 1) as usize,
                self.nv,
            ),
        }
    }

    fn neighbor_indices(&self, i: usize, j: usize) -> [usize; 8] {
        let (ii, jj) = (i as isize, j as isize);
        [
            self.map_index(ii + 1, jj),
            self.map_index(ii - 1, jj),
            self.map_index(ii, jj + 1),
            self.map_index(ii, jj - 1),
            self.map_index(ii + 1, jj + 1),
            self.map_index(ii + 1, jj - 1),
            self.map_index(ii - 1, jj + 1),
            self.map_index(ii - 1, jj - 1),
        ]
    }
}

fn resolve_seam_maps<S: ParametricSurface + ?Sized>(
    surface: &S,
    topology: SurfaceTopology,
    range_u: (f64, f64),
    range_v: (f64, f64),
    nv: usize,
) -> (Vec<usize>, Vec<usize>, f64) {
    let (seam_fwd, seam_rev) = if topology == SurfaceTopology::TwistedKlein {
        detect_twisted_seam_map(surface, range_u, range_v, nv)
    } else {
        ((0..nv).collect(), (0..nv).collect())
    };
    let seam_v_sign = if topology == SurfaceTopology::TwistedKlein
        && nv >= 3
        && (seam_fwd[1] + nv - seam_fwd[0]) % nv == nv - 1
    {
        -1.0
    } else {
        1.0
    };
    (seam_fwd, seam_rev, seam_v_sign)
}

pub(super) fn precompute_surface_metrics<S: ParametricSurface + ?Sized>(
    surface: &S,
    topology: SurfaceTopology,
    range_u: (f64, f64),
    range_v: (f64, f64),
    nu: usize,
    nv: usize,
) -> PrecomputedSurfaceMetrics {
    let (du, dv) = topology.grid_spacings(range_u, range_v, nu, nv);
    let (seam_fwd, seam_rev, seam_v_sign) =
        resolve_seam_maps(surface, topology, range_u, range_v, nv);
    let ctx = GridTopologyCtx {
        topology,
        range_u,
        range_v,
        nu,
        nv,
        du,
        dv,
        seam_fwd: &seam_fwd,
        seam_rev: &seam_rev,
        seam_v_sign,
    };

    let len = nu * nv;
    let mut stencils = Vec::with_capacity(len);
    let mut area_elements = Vec::with_capacity(len);
    let mut gaussian_curvature = Vec::with_capacity(len);
    let mut metric_anisotropy = Vec::with_capacity(len);
    let mut min_area = f64::INFINITY;
    let mut max_rate = 1e-6_f64;

    for i in 0..nu {
        for j in 0..nv {
            let sample = build_cell_metrics(surface, &ctx, i, j);
            if sample.cell_area > 1e-10 && !sample.is_pole {
                min_area = min_area.min(sample.cell_area);
                max_rate = max_rate.max(sample.spectral_rate);
            }
            stencils.push(sample.stencil);
            area_elements.push(sample.area_elem);
            gaussian_curvature.push(sample.gauss_k);
            metric_anisotropy.push(sample.anisotropy);
        }
    }

    let (north_rate, south_rate) = polar_cap_rates(surface, topology, range_u, range_v, dv);
    if topology == SurfaceTopology::PolarSphere {
        max_rate = max_rate.max(north_rate).max(south_rate);
    }
    if !min_area.is_finite() {
        min_area = (du * dv).max(1e-5);
    }

    PrecomputedSurfaceMetrics {
        stencils,
        area_elements,
        gaussian_curvature,
        metric_anisotropy,
        seam_fwd,
        min_area_element: min_area,
        max_spectral_rate: max_rate,
        north_pole_rate: north_rate,
        south_pole_rate: south_rate,
    }
}

struct CellMetricSample {
    stencil: CellStencil,
    area_elem: f64,
    cell_area: f64,
    gauss_k: f64,
    anisotropy: f64,
    spectral_rate: f64,
    is_pole: bool,
}

fn build_cell_metrics<S: ParametricSurface + ?Sized>(
    surface: &S,
    ctx: &GridTopologyCtx<'_>,
    i: usize,
    j: usize,
) -> CellMetricSample {
    let (u, v) = ctx
        .topology
        .uv_at(ctx.range_u, ctx.range_v, ctx.nu, ctx.nv, i, j);
    let is_pole = ctx.topology == SurfaceTopology::PolarSphere && (j == 0 || j + 1 == ctx.nv);
    let eval_v = if is_pole {
        if j == 0 {
            v + 1e-4 * ctx.dv.max(1e-3)
        } else {
            v - 1e-4 * ctx.dv.max(1e-3)
        }
    } else {
        v
    };

    let (e, f, g) = surface.first_fundamental_form(u, eval_v);
    let g_cov = Matrix2::new(e, f, f, g);
    let anisotropy = principal_anisotropy(&g_cov);
    let gauss_k = sanitize_scalar(surface.gaussian_curvature(u, eval_v), 0.0);
    let area_elem = surface.area_element(u, eval_v).max(1e-8);
    let cell_area = area_elem * ctx.du * ctx.dv;
    let (_, g_inv) = surface.metric_tensor_inverse(u, eval_v);
    let spectral_rate = 4.0
        * (g_inv[(0, 0)].abs() / (ctx.du * ctx.du)
            + g_inv[(1, 1)].abs() / (ctx.dv * ctx.dv)
            + 2.0 * g_inv[(0, 1)].abs() / (ctx.du * ctx.dv));

    let idx = ctx.neighbor_indices(i, j);
    let weight = if is_pole {
        [0.0; 8]
    } else {
        assemble_divergence_weights(surface, ctx, i, j, area_elem)
    };
    let diag = -weight.iter().sum::<f64>();

    CellMetricSample {
        stencil: CellStencil { idx, weight, diag },
        area_elem,
        cell_area,
        gauss_k,
        anisotropy,
        spectral_rate: sanitize_scalar(spectral_rate, 1.0),
        is_pole,
    }
}

fn assemble_divergence_weights<S: ParametricSurface + ?Sized>(
    surface: &S,
    ctx: &GridTopologyCtx<'_>,
    i: usize,
    j: usize,
    area_elem: f64,
) -> [f64; 8] {
    // Neighbor order: [0: ip, 1: im, 2: jp, 3: jm, 4: pp, 5: pm, 6: mp, 7: mm]
    let mut w = [0.0_f64; 8];
    let (u, v) = ctx
        .topology
        .uv_at(ctx.range_u, ctx.range_v, ctx.nu, ctx.nv, i, j);
    let inv_area = 1.0 / area_elem.max(1e-8);
    let (inv_du2, inv_dv2, inv_4dudv) = (
        inv_area / (ctx.du * ctx.du),
        inv_area / (ctx.dv * ctx.dv),
        inv_area / (4.0 * ctx.du * ctx.dv),
    );

    if !(ctx.topology == SurfaceTopology::OpenNeumann && i + 1 >= ctx.nu) {
        let (w11, w12) = contravariant_flux_density(surface, u + 0.5 * ctx.du, v);
        let v_sign = if ctx.topology == SurfaceTopology::TwistedKlein && i + 1 >= ctx.nu {
            ctx.seam_v_sign
        } else {
            1.0
        };
        w[0] += w11 * inv_du2;
        w[2] += w12 * inv_4dudv;
        w[3] -= w12 * inv_4dudv;
        w[4] += v_sign * w12 * inv_4dudv;
        w[5] -= v_sign * w12 * inv_4dudv;
    }
    if !(ctx.topology == SurfaceTopology::OpenNeumann && i == 0) {
        let (w11, w12) = contravariant_flux_density(surface, u - 0.5 * ctx.du, v);
        let v_sign = if ctx.topology == SurfaceTopology::TwistedKlein && i == 0 {
            ctx.seam_v_sign
        } else {
            1.0
        };
        w[1] += w11 * inv_du2;
        w[2] -= w12 * inv_4dudv;
        w[3] += w12 * inv_4dudv;
        w[6] -= v_sign * w12 * inv_4dudv;
        w[7] += v_sign * w12 * inv_4dudv;
    }
    accumulate_v_face_weights(surface, ctx, j, (u, v), (inv_dv2, inv_4dudv), &mut w);
    w
}

fn accumulate_v_face_weights<S: ParametricSurface + ?Sized>(
    surface: &S,
    ctx: &GridTopologyCtx<'_>,
    j: usize,
    (u, v): (f64, f64),
    (inv_dv2, inv_4dudv): (f64, f64),
    w: &mut [f64; 8],
) {
    if !(ctx.topology == SurfaceTopology::OpenNeumann && j + 1 >= ctx.nv) {
        let (w22, w12) = contravariant_v_flux_density(surface, u, v + 0.5 * ctx.dv);
        w[2] += w22 * inv_dv2;
        w[0] += w12 * inv_4dudv;
        w[1] -= w12 * inv_4dudv;
        w[4] += w12 * inv_4dudv;
        w[6] -= w12 * inv_4dudv;
    }
    if !(ctx.topology == SurfaceTopology::OpenNeumann && j == 0) {
        let (w22, w12) = contravariant_v_flux_density(surface, u, v - 0.5 * ctx.dv);
        w[3] += w22 * inv_dv2;
        w[0] -= w12 * inv_4dudv;
        w[1] += w12 * inv_4dudv;
        w[5] -= w12 * inv_4dudv;
        w[7] += w12 * inv_4dudv;
    }
}

fn contravariant_flux_density<S: ParametricSurface + ?Sized>(
    surface: &S,
    u: f64,
    v: f64,
) -> (f64, f64) {
    let sqrt_g = surface.area_element(u, v).max(1e-8);
    let (det_g, g_inv) = surface.metric_tensor_inverse(u, v);
    if !det_g.is_finite() || det_g <= 1e-16 {
        return (1.0, 0.0);
    }
    (
        sanitize_scalar(sqrt_g * g_inv[(0, 0)], 1.0),
        sanitize_scalar(sqrt_g * g_inv[(0, 1)], 0.0),
    )
}

fn contravariant_v_flux_density<S: ParametricSurface + ?Sized>(
    surface: &S,
    u: f64,
    v: f64,
) -> (f64, f64) {
    let sqrt_g = surface.area_element(u, v).max(1e-8);
    let (det_g, g_inv) = surface.metric_tensor_inverse(u, v);
    if !det_g.is_finite() || det_g <= 1e-16 {
        return (1.0, 0.0);
    }
    (
        sanitize_scalar(sqrt_g * g_inv[(1, 1)], 1.0),
        sanitize_scalar(sqrt_g * g_inv[(1, 0)], 0.0),
    )
}

fn principal_anisotropy(g_mat: &Matrix2<f64>) -> f64 {
    let e = g_mat[(0, 0)];
    let f = g_mat[(0, 1)];
    let g = g_mat[(1, 1)];
    let trace = e + g;
    let disc = ((e - g) * (e - g) + 4.0 * f * f).max(0.0).sqrt();
    let lambda_max = 0.5 * (trace + disc);
    let lambda_min = (0.5 * (trace - disc)).max(1e-8);
    sanitize_scalar((lambda_max / lambda_min).max(1.0).sqrt(), 1.0)
}

fn polar_cap_rates<S: ParametricSurface + ?Sized>(
    surface: &S,
    topology: SurfaceTopology,
    range_u: (f64, f64),
    range_v: (f64, f64),
    dv: f64,
) -> (f64, f64) {
    if topology != SurfaceTopology::PolarSphere {
        return (0.0, 0.0);
    }
    let (_, _, g_n) = surface.first_fundamental_form(range_u.0, range_v.0 + 1e-4);
    let (_, _, g_s) = surface.first_fundamental_form(range_u.0, range_v.1 - 1e-4);
    let denom_n = (g_n.max(1e-6) * dv * dv).max(1e-8);
    let denom_s = (g_s.max(1e-6) * dv * dv).max(1e-8);
    (4.0 / denom_n, 4.0 / denom_s)
}

fn sanitize_scalar(val: f64, fallback: f64) -> f64 {
    if val.is_finite() { val } else { fallback }
}
