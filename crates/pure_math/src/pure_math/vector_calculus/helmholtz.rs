//! Zero-allocation curvilinear Helmholtz-Hodge decomposition on flat 3D slices.
//!
//! Splits a sampled 3D curvilinear vector field $\mathbf{F} = \mathbf{F}_{\text{irr}} + \mathbf{F}_{\text{sol}}$
//! into a curl-free irrotational component $\mathbf{F}_{\text{irr}} = -\nabla \Phi$ and a
//! divergence-free solenoidal component $\mathbf{F}_{\text{sol}}$ across any
//! [`OrthogonalCoordinateSystem`] (`Cartesian`, `Cylindrical`, or `Spherical`).

// [cite:vector_calculus]

use super::coordinates::OrthogonalCoordinateSystem;
use super::theorems::Domain;
use math_commons::math_kernel::types::flatten_3d_index;
use nalgebra::Vector3;

/// Pre-allocated contiguous 3D buffers for zero-allocation [`helmholtz_decompose`] execution.
#[derive(Debug, Clone)]
pub struct HelmholtzWorkspace {
    /// Grid resolution along curvilinear coordinate $u_1$.
    pub nx: usize,
    /// Grid resolution along curvilinear coordinate $u_2$.
    pub ny: usize,
    /// Grid resolution along curvilinear coordinate $u_3$.
    pub nz: usize,
    /// Sampled discrete curvilinear divergence $\nabla \cdot \mathbf{F}$ (`len = nx * ny * nz`).
    pub divergence: Vec<f64>,
    /// Solved scalar potential $\Phi$ satisfying $-\nabla^2 \Phi = \nabla \cdot \mathbf{F}$.
    pub potential: Vec<f64>,
    /// Pre-computed local scale factors $(h_1, h_2, h_3)$ at cell centers.
    pub scale_factors: Vec<Vector3<f64>>,
    /// Curl-free irrotational component $\mathbf{F}_{\text{irr}} = -\nabla \Phi$.
    pub irrotational: Vec<Vector3<f64>>,
    /// Divergence-free solenoidal component $\mathbf{F}_{\text{sol}} = \mathbf{F} - \mathbf{F}_{\text{irr}}$.
    pub solenoidal: Vec<Vector3<f64>>,
}

/// Diagnostic residuals returned by [`helmholtz_decompose`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HelmholtzSummary {
    /// Peak interior discrete divergence $|\nabla \cdot \mathbf{F}_{\text{sol}}|$.
    pub max_solenoidal_divergence: f64,
    /// Peak interior discrete curl norm $\|\nabla \times \mathbf{F}_{\text{irr}}\|$.
    pub max_irrotational_curl: f64,
    /// Peak pointwise reconstruction error $\|\mathbf{F}_{\text{irr}} + \mathbf{F}_{\text{sol}} - \mathbf{F}\|$.
    pub max_reconstruction_error: f64,
    /// Number of Gauss-Seidel SOR relaxation sweeps executed.
    pub iterations: usize,
}

impl HelmholtzWorkspace {
    /// Allocates contiguous flat 3D buffers for an `(nx, ny, nz)` curvilinear grid (`nx, ny, nz >= 5`).
    #[must_use]
    #[verified_engine::verified]
    pub fn new(nx: usize, ny: usize, nz: usize) -> Self {
        assert!(
            nx >= 5 && ny >= 5 && nz >= 5,
            "Helmholtz grid requires >= 5 cells per axis"
        );
        let total = nx * ny * nz;
        Self {
            nx,
            ny,
            nz,
            divergence: vec![0.0; total],
            potential: vec![0.0; total],
            scale_factors: vec![Vector3::new(1.0, 1.0, 1.0); total],
            irrotational: vec![Vector3::zeros(); total],
            solenoidal: vec![Vector3::zeros(); total],
        }
    }

    /// Returns total cell count `nx * ny * nz`.
    #[must_use]
    #[verified_engine::verified]
    pub fn len(&self) -> usize {
        let total = self.nx * self.ny * self.nz;
        assert_eq!(self.potential.len(), total);
        total
    }

    /// Returns `true` when the workspace contains zero cells.
    #[must_use]
    #[verified_engine::verified]
    pub fn is_empty(&self) -> bool {
        let empty = self.len() == 0;
        assert!(!empty);
        empty
    }

    /// Maps `(i, j, k)` cell coordinates to a flat contiguous slice index via [`flatten_3d_index`].
    #[must_use]
    #[inline]
    pub fn index(&self, i: usize, j: usize, k: usize) -> usize {
        flatten_3d_index(i, j, k, self.nx, self.ny)
    }

    /// Returns the curvilinear cell-center coordinate for `(i, j, k)` inside `domain`.
    #[must_use]
    #[inline]
    pub fn cell_center(&self, domain: &Domain, i: usize, j: usize, k: usize) -> Vector3<f64> {
        let du1 = (domain.max[0] - domain.min[0]) / self.nx as f64;
        let du2 = (domain.max[1] - domain.min[1]) / self.ny as f64;
        let du3 = (domain.max[2] - domain.min[2]) / self.nz as f64;
        Vector3::new(
            domain.min[0] + (i as f64 + 0.5) * du1,
            domain.min[1] + (j as f64 + 0.5) * du2,
            domain.min[2] + (k as f64 + 0.5) * du3,
        )
    }
}

#[verified_engine::verified]
fn fill_scale_factors<S: OrthogonalCoordinateSystem>(
    coords: &S,
    domain: &Domain,
    ws: &mut HelmholtzWorkspace,
) {
    assert_eq!(ws.scale_factors.len(), ws.nx * ws.ny * ws.nz);
    for k in 0..ws.nz {
        for j in 0..ws.ny {
            for i in 0..ws.nx {
                let p = ws.cell_center(domain, i, j, k);
                let raw = coords.scale_factors(&p);
                let idx = flatten_3d_index(i, j, k, ws.nx, ws.ny);
                ws.scale_factors[idx] = Vector3::new(
                    raw[0].abs().max(1e-6),
                    raw[1].abs().max(1e-6),
                    raw[2].abs().max(1e-6),
                );
            }
        }
    }
}

#[verified_engine::verified]
fn compute_discrete_divergence(
    field_samples: &[Vector3<f64>],
    du: Vector3<f64>,
    ws: &mut HelmholtzWorkspace,
) {
    assert_eq!(field_samples.len(), ws.nx * ws.ny * ws.nz);
    ws.divergence.fill(0.0);
    let (nx, ny, nz) = (ws.nx, ws.ny, ws.nz);
    for k in 1..(nz - 1) {
        for j in 1..(ny - 1) {
            for i in 1..(nx - 1) {
                let idx = flatten_3d_index(i, j, k, nx, ny);
                let h = ws.scale_factors[idx];
                let jac = h[0] * h[1] * h[2];

                let ip = flatten_3d_index(i + 1, j, k, nx, ny);
                let im = flatten_3d_index(i - 1, j, k, nx, ny);
                let jp = flatten_3d_index(i, j + 1, k, nx, ny);
                let jm = flatten_3d_index(i, j - 1, k, nx, ny);
                let kp = flatten_3d_index(i, j, k + 1, nx, ny);
                let km = flatten_3d_index(i, j, k - 1, nx, ny);

                let d1 = (ws.scale_factors[ip][1] * ws.scale_factors[ip][2] * field_samples[ip][0]
                    - ws.scale_factors[im][1] * ws.scale_factors[im][2] * field_samples[im][0])
                    / (2.0 * du[0]);
                let d2 = (ws.scale_factors[jp][0] * ws.scale_factors[jp][2] * field_samples[jp][1]
                    - ws.scale_factors[jm][0] * ws.scale_factors[jm][2] * field_samples[jm][1])
                    / (2.0 * du[1]);
                let d3 = (ws.scale_factors[kp][0] * ws.scale_factors[kp][1] * field_samples[kp][2]
                    - ws.scale_factors[km][0] * ws.scale_factors[km][1] * field_samples[km][2])
                    / (2.0 * du[2]);

                ws.divergence[idx] = (d1 + d2 + d3) / jac;
            }
        }
    }
}

#[inline]
fn metric_conductance(h: Vector3<f64>, axis: usize, du_axis: f64) -> f64 {
    let jac = h[0] * h[1] * h[2];
    jac / (h[axis] * h[axis] * du_axis * du_axis)
}

#[inline]
fn wide_axis_stencil(
    ws: &HelmholtzWorkspace,
    axis: usize,
    bounds: (usize, usize),
    nbrs: [usize; 4],
    du_axis: f64,
) -> (f64, f64) {
    let (coord, n_axis) = bounds;
    let wp = metric_conductance(ws.scale_factors[nbrs[0]], axis, du_axis);
    let wm = metric_conductance(ws.scale_factors[nbrs[1]], axis, du_axis);
    let phi_p2 = ws.potential[nbrs[2]];
    let phi_m2 = ws.potential[nbrs[3]];
    if coord == 1 {
        (0.25 * (wp + 2.0 * wm), 0.25 * wp * phi_p2)
    } else if coord + 2 == n_axis {
        (0.25 * (2.0 * wp + wm), 0.25 * wm * phi_m2)
    } else {
        (0.25 * (wp + wm), 0.25 * (wp * phi_p2 + wm * phi_m2))
    }
}

#[verified_engine::verified]
fn relax_compact_warmup(du: Vector3<f64>, ws: &mut HelmholtzWorkspace, sweeps: usize) {
    assert!(du[0] > 0.0 && du[1] > 0.0 && du[2] > 0.0);
    let (nx, ny, nz) = (ws.nx, ws.ny, ws.nz);
    for _ in 0..sweeps {
        for k in 1..(nz - 1) {
            for j in 1..(ny - 1) {
                for i in 1..(nx - 1) {
                    let idx = flatten_3d_index(i, j, k, nx, ny);
                    let hc = ws.scale_factors[idx];
                    let jac = hc[0] * hc[1] * hc[2];
                    let ip = flatten_3d_index(i + 1, j, k, nx, ny);
                    let im = flatten_3d_index(i - 1, j, k, nx, ny);
                    let jp = flatten_3d_index(i, j + 1, k, nx, ny);
                    let jm = flatten_3d_index(i, j - 1, k, nx, ny);
                    let kp = flatten_3d_index(i, j, k + 1, nx, ny);
                    let km = flatten_3d_index(i, j, k - 1, nx, ny);

                    let ap = 0.5
                        * (metric_conductance(hc, 0, du[0])
                            + metric_conductance(ws.scale_factors[ip], 0, du[0]));
                    let am = 0.5
                        * (metric_conductance(hc, 0, du[0])
                            + metric_conductance(ws.scale_factors[im], 0, du[0]));
                    let bp = 0.5
                        * (metric_conductance(hc, 1, du[1])
                            + metric_conductance(ws.scale_factors[jp], 1, du[1]));
                    let bm = 0.5
                        * (metric_conductance(hc, 1, du[1])
                            + metric_conductance(ws.scale_factors[jm], 1, du[1]));
                    let cp = 0.5
                        * (metric_conductance(hc, 2, du[2])
                            + metric_conductance(ws.scale_factors[kp], 2, du[2]));
                    let cm = 0.5
                        * (metric_conductance(hc, 2, du[2])
                            + metric_conductance(ws.scale_factors[km], 2, du[2]));

                    let diag = ap + am + bp + bm + cp + cm;
                    ws.potential[idx] = (jac * ws.divergence[idx]
                        + ap * ws.potential[ip]
                        + am * ws.potential[im]
                        + bp * ws.potential[jp]
                        + bm * ws.potential[jm]
                        + cp * ws.potential[kp]
                        + cm * ws.potential[km])
                        / diag;
                }
            }
        }
    }
}

#[verified_engine::verified]
fn relax_curvilinear_poisson(
    du: Vector3<f64>,
    ws: &mut HelmholtzWorkspace,
    max_iters: usize,
) -> usize {
    assert!(du[0] > 0.0 && du[1] > 0.0 && du[2] > 0.0);
    ws.potential.fill(0.0);
    let warmup = (max_iters / 5).clamp(4, 24);
    relax_compact_warmup(du, ws, warmup);

    let (nx, ny, nz) = (ws.nx, ws.ny, ws.nz);
    let omega = 1.45;
    let mut completed = 0;

    for iter in 0..max_iters {
        let mut max_delta = 0.0_f64;
        for k in 1..(nz - 1) {
            for j in 1..(ny - 1) {
                for i in 1..(nx - 1) {
                    let idx = flatten_3d_index(i, j, k, nx, ny);
                    let hc = ws.scale_factors[idx];
                    let jac = hc[0] * hc[1] * hc[2];

                    let (dx_diag, dx_off) = wide_axis_stencil(
                        ws,
                        0,
                        (i, nx),
                        [
                            flatten_3d_index(i + 1, j, k, nx, ny),
                            flatten_3d_index(i - 1, j, k, nx, ny),
                            flatten_3d_index((i + 2).min(nx - 1), j, k, nx, ny),
                            flatten_3d_index(i.saturating_sub(2), j, k, nx, ny),
                        ],
                        du[0],
                    );
                    let (dy_diag, dy_off) = wide_axis_stencil(
                        ws,
                        1,
                        (j, ny),
                        [
                            flatten_3d_index(i, j + 1, k, nx, ny),
                            flatten_3d_index(i, j - 1, k, nx, ny),
                            flatten_3d_index(i, (j + 2).min(ny - 1), k, nx, ny),
                            flatten_3d_index(i, j.saturating_sub(2), k, nx, ny),
                        ],
                        du[1],
                    );
                    let (dz_diag, dz_off) = wide_axis_stencil(
                        ws,
                        2,
                        (k, nz),
                        [
                            flatten_3d_index(i, j, k + 1, nx, ny),
                            flatten_3d_index(i, j, k - 1, nx, ny),
                            flatten_3d_index(i, j, (k + 2).min(nz - 1), nx, ny),
                            flatten_3d_index(i, j, k.saturating_sub(2), nx, ny),
                        ],
                        du[2],
                    );

                    let gs_val = (jac * ws.divergence[idx] + dx_off + dy_off + dz_off)
                        / (dx_diag + dy_diag + dz_diag);
                    let old_val = ws.potential[idx];
                    let updated = (1.0 - omega) * old_val + omega * gs_val;
                    ws.potential[idx] = updated;
                    max_delta = max_delta.max((updated - old_val).abs());
                }
            }
        }
        completed = iter + 1;
        if max_delta < 1e-10 {
            break;
        }
    }
    completed
}

#[verified_engine::verified]
fn extract_components(
    field_samples: &[Vector3<f64>],
    du: Vector3<f64>,
    ws: &mut HelmholtzWorkspace,
) -> f64 {
    assert_eq!(field_samples.len(), ws.irrotational.len());
    let (nx, ny, nz) = (ws.nx, ws.ny, ws.nz);
    let mut max_recon_err = 0.0_f64;

    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let idx = flatten_3d_index(i, j, k, nx, ny);
                let h = ws.scale_factors[idx];

                let dphi1 = if i > 0 && i + 1 < nx {
                    (ws.potential[flatten_3d_index(i + 1, j, k, nx, ny)]
                        - ws.potential[flatten_3d_index(i - 1, j, k, nx, ny)])
                        / (2.0 * du[0])
                } else if i == 0 {
                    (ws.potential[flatten_3d_index(1, j, k, nx, ny)] - ws.potential[idx]) / du[0]
                } else {
                    (ws.potential[idx] - ws.potential[flatten_3d_index(nx - 2, j, k, nx, ny)])
                        / du[0]
                };

                let dphi2 = if j > 0 && j + 1 < ny {
                    (ws.potential[flatten_3d_index(i, j + 1, k, nx, ny)]
                        - ws.potential[flatten_3d_index(i, j - 1, k, nx, ny)])
                        / (2.0 * du[1])
                } else if j == 0 {
                    (ws.potential[flatten_3d_index(i, 1, k, nx, ny)] - ws.potential[idx]) / du[1]
                } else {
                    (ws.potential[idx] - ws.potential[flatten_3d_index(i, ny - 2, k, nx, ny)])
                        / du[1]
                };

                let dphi3 = if k > 0 && k + 1 < nz {
                    (ws.potential[flatten_3d_index(i, j, k + 1, nx, ny)]
                        - ws.potential[flatten_3d_index(i, j, k - 1, nx, ny)])
                        / (2.0 * du[2])
                } else if k == 0 {
                    (ws.potential[flatten_3d_index(i, j, 1, nx, ny)] - ws.potential[idx]) / du[2]
                } else {
                    (ws.potential[idx] - ws.potential[flatten_3d_index(i, j, nz - 2, nx, ny)])
                        / du[2]
                };

                let f_irr = Vector3::new(-dphi1 / h[0], -dphi2 / h[1], -dphi3 / h[2]);
                let f_sol = field_samples[idx] - f_irr;
                ws.irrotational[idx] = f_irr;
                ws.solenoidal[idx] = f_sol;
                max_recon_err = max_recon_err.max((f_irr + f_sol - field_samples[idx]).norm());
            }
        }
    }
    max_recon_err
}

#[verified_engine::verified]
fn measure_interior_residuals(du: Vector3<f64>, ws: &HelmholtzWorkspace) -> (f64, f64) {
    assert!(ws.nx >= 5 && ws.ny >= 5 && ws.nz >= 5);
    let (nx, ny, nz) = (ws.nx, ws.ny, ws.nz);
    let mut max_div_sol = 0.0_f64;
    let mut max_curl_irr = 0.0_f64;

    for k in 2..(nz - 2) {
        for j in 2..(ny - 2) {
            for i in 2..(nx - 2) {
                let idx = flatten_3d_index(i, j, k, nx, ny);
                let h = ws.scale_factors[idx];
                let jac = h[0] * h[1] * h[2];

                let ip = flatten_3d_index(i + 1, j, k, nx, ny);
                let im = flatten_3d_index(i - 1, j, k, nx, ny);
                let jp = flatten_3d_index(i, j + 1, k, nx, ny);
                let jm = flatten_3d_index(i, j - 1, k, nx, ny);
                let kp = flatten_3d_index(i, j, k + 1, nx, ny);
                let km = flatten_3d_index(i, j, k - 1, nx, ny);

                let d1 = (ws.scale_factors[ip][1] * ws.scale_factors[ip][2] * ws.solenoidal[ip][0]
                    - ws.scale_factors[im][1] * ws.scale_factors[im][2] * ws.solenoidal[im][0])
                    / (2.0 * du[0]);
                let d2 = (ws.scale_factors[jp][0] * ws.scale_factors[jp][2] * ws.solenoidal[jp][1]
                    - ws.scale_factors[jm][0] * ws.scale_factors[jm][2] * ws.solenoidal[jm][1])
                    / (2.0 * du[1]);
                let d3 = (ws.scale_factors[kp][0] * ws.scale_factors[kp][1] * ws.solenoidal[kp][2]
                    - ws.scale_factors[km][0] * ws.scale_factors[km][1] * ws.solenoidal[km][2])
                    / (2.0 * du[2]);
                max_div_sol = max_div_sol.max(((d1 + d2 + d3) / jac).abs());

                let c1 = ((ws.scale_factors[jp][2] * ws.irrotational[jp][2]
                    - ws.scale_factors[jm][2] * ws.irrotational[jm][2])
                    / (2.0 * du[1])
                    - (ws.scale_factors[kp][1] * ws.irrotational[kp][1]
                        - ws.scale_factors[km][1] * ws.irrotational[km][1])
                        / (2.0 * du[2]))
                    / (h[1] * h[2]);
                let c2 = ((ws.scale_factors[kp][0] * ws.irrotational[kp][0]
                    - ws.scale_factors[km][0] * ws.irrotational[km][0])
                    / (2.0 * du[2])
                    - (ws.scale_factors[ip][2] * ws.irrotational[ip][2]
                        - ws.scale_factors[im][2] * ws.irrotational[im][2])
                        / (2.0 * du[0]))
                    / (h[2] * h[0]);
                let c3 = ((ws.scale_factors[ip][1] * ws.irrotational[ip][1]
                    - ws.scale_factors[im][1] * ws.irrotational[im][1])
                    / (2.0 * du[0])
                    - (ws.scale_factors[jp][0] * ws.irrotational[jp][0]
                        - ws.scale_factors[jm][0] * ws.irrotational[jm][0])
                        / (2.0 * du[1]))
                    / (h[0] * h[1]);
                max_curl_irr = max_curl_irr.max(Vector3::new(c1, c2, c3).norm());
            }
        }
    }
    (max_div_sol, max_curl_irr)
}

/// Decomposes a sampled curvilinear vector field `field_samples` on `domain` into curl-free
/// (`workspace.irrotational`) and divergence-free (`workspace.solenoidal`) components with
/// **zero heap allocations** after [`HelmholtzWorkspace`] initialization.
#[verified_engine::verified]
pub fn helmholtz_decompose<S: OrthogonalCoordinateSystem>(
    coords: &S,
    field_samples: &[Vector3<f64>],
    domain: &Domain,
    workspace: &mut HelmholtzWorkspace,
    max_iters: usize,
) -> HelmholtzSummary {
    let total = workspace.nx * workspace.ny * workspace.nz;
    assert_eq!(
        field_samples.len(),
        total,
        "field_samples length must match workspace grid"
    );
    assert!(
        domain.max[0] > domain.min[0]
            && domain.max[1] > domain.min[1]
            && domain.max[2] > domain.min[2],
        "Domain bounds must satisfy max > min on every axis"
    );

    let du = Vector3::new(
        (domain.max[0] - domain.min[0]) / workspace.nx as f64,
        (domain.max[1] - domain.min[1]) / workspace.ny as f64,
        (domain.max[2] - domain.min[2]) / workspace.nz as f64,
    );

    fill_scale_factors(coords, domain, workspace);
    compute_discrete_divergence(field_samples, du, workspace);
    let iterations = relax_curvilinear_poisson(du, workspace, max_iters.max(1));
    let max_reconstruction_error = extract_components(field_samples, du, workspace);
    let (max_solenoidal_divergence, max_irrotational_curl) =
        measure_interior_residuals(du, workspace);

    assert!(max_reconstruction_error.is_finite());
    assert!(max_solenoidal_divergence.is_finite());
    assert!(max_irrotational_curl.is_finite());

    HelmholtzSummary {
        max_solenoidal_divergence,
        max_irrotational_curl,
        max_reconstruction_error,
        iterations,
    }
}
