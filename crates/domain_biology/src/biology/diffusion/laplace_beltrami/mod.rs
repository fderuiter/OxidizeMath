// [cite:graph_parameters_rust]
//! Intrinsic Laplace-Beltrami diffusion operator on 2D Riemannian manifolds.

mod stencil;

use super::SpatialDiffusion;
use crate::biology::morphogenesis::reaction::ReactionKinetics;
use crate::biology::reaction_diffusion::{ChemicalState, DiffusionModel};
use math_commons::math_kernel::types::{StepSize, flatten_2d_index};
use pure_math::pure_math::analysis::pde::fused_stepper::FusedStencilStepper;
use pure_math::pure_math::differential_geometry::buckled_flow::SurfaceTopology;
use pure_math::pure_math::differential_geometry::surface::ParametricSurface;
use stencil::{CellStencil, precompute_surface_metrics};

/// Flat, pre-allocated 2D Laplace-Beltrami diffusion strategy $\Delta_g f = \frac{1}{\sqrt{\det g}} \partial_a (\sqrt{\det g}\, g^{ab} \partial_b f)$.
///
/// Precomputes contravariant metric flux weights, topology-aware seam/polar neighbor tables,
/// Gaussian curvature $K$, and principal metric anisotropy at construction time so every
/// reaction-diffusion step executes with zero heap allocations.
#[derive(Debug, Clone)]
pub struct LaplaceBeltramiDiffusion2D {
    nu: usize,
    nv: usize,
    topology: SurfaceTopology,
    range_u: (f64, f64),
    range_v: (f64, f64),
    du: StepSize,
    dv: StepSize,
    min_area_element: f64,
    max_spectral_rate: f64,
    north_pole_rate: f64,
    south_pole_rate: f64,
    seam_fwd: Vec<usize>,
    stencils: Vec<CellStencil>,
    area_elements: Vec<f64>,
    gaussian_curvature: Vec<f64>,
    metric_anisotropy: Vec<f64>,
}

impl LaplaceBeltramiDiffusion2D {
    /// Precomputes the intrinsic Laplace-Beltrami operator for `surface` over `(nu, nv)` cells.
    #[must_use]
    #[verified_engine::verified]
    pub fn from_surface<S: ParametricSurface + ?Sized>(
        surface: &S,
        topology: SurfaceTopology,
        range_u: (f64, f64),
        range_v: (f64, f64),
        nu: usize,
        nv: usize,
    ) -> Self {
        let nu = nu.max(4);
        let nv = nv.max(4);
        let (raw_du, raw_dv) = topology.grid_spacings(range_u, range_v, nu, nv);
        let du = StepSize(raw_du.abs().max(1e-6));
        let dv = StepSize(raw_dv.abs().max(1e-6));
        let pre = precompute_surface_metrics(surface, topology, range_u, range_v, nu, nv);

        Self {
            nu,
            nv,
            topology,
            range_u,
            range_v,
            du,
            dv,
            min_area_element: pre.min_area_element,
            max_spectral_rate: pre.max_spectral_rate,
            north_pole_rate: pre.north_pole_rate,
            south_pole_rate: pre.south_pole_rate,
            seam_fwd: pre.seam_fwd,
            stencils: pre.stencils,
            area_elements: pre.area_elements,
            gaussian_curvature: pre.gaussian_curvature,
            metric_anisotropy: pre.metric_anisotropy,
        }
    }

    /// Grid resolution `(nu, nv)`.
    #[must_use]
    #[verified_engine::verified]
    pub fn dimensions(&self) -> (usize, usize) {
        (self.nu, self.nv)
    }

    /// Total flat cell count `nu * nv`.
    #[must_use]
    #[verified_engine::verified]
    pub fn grid_size(&self) -> usize {
        self.nu * self.nv
    }

    /// Active manifold boundary [`SurfaceTopology`].
    #[must_use]
    #[verified_engine::verified]
    pub fn topology(&self) -> SurfaceTopology {
        self.topology
    }

    /// Parameter domain bounds `(range_u, range_v)`.
    #[must_use]
    #[verified_engine::verified]
    pub fn parameter_bounds(&self) -> ((f64, f64), (f64, f64)) {
        (self.range_u, self.range_v)
    }

    /// Minimum physical metric cell area $\min (\sqrt{EG - F^2}\, \Delta u \Delta v)$.
    #[must_use]
    #[verified_engine::verified]
    pub fn min_area_element(&self) -> f64 {
        self.min_area_element
    }

    /// Maximum explicit time step satisfying the metric CFL condition for `max_diffusivity`.
    #[must_use]
    #[verified_engine::verified]
    pub fn max_stable_dt(&self, max_diffusivity: f64) -> f64 {
        let diff = max_diffusivity.abs().max(1e-9);
        let rate = (diff * self.max_spectral_rate).max(1e-9);
        (0.85 / rate).clamp(1e-7, 0.25)
    }

    /// Clamps a requested `dt` to [`Self::max_stable_dt`] so explicit stepping remains stable.
    #[must_use]
    #[verified_engine::verified]
    pub fn clamp_dt(&self, dt: f64, max_diffusivity: f64) -> f64 {
        dt.clamp(1e-7, self.max_stable_dt(max_diffusivity))
    }

    /// Precomputed differential area element $\sqrt{EG - F^2}$ per flat cell.
    #[must_use]
    #[verified_engine::verified]
    pub fn area_elements(&self) -> &[f64] {
        &self.area_elements
    }

    /// Precomputed Gaussian curvature $K(u, v)$ per flat cell.
    #[must_use]
    #[verified_engine::verified]
    pub fn gaussian_curvature(&self) -> &[f64] {
        &self.gaussian_curvature
    }

    /// Precomputed principal metric stretch ratio $\sqrt{\lambda_{\max} / \lambda_{\min}} \ge 1$ per cell.
    #[must_use]
    #[verified_engine::verified]
    pub fn metric_anisotropy(&self) -> &[f64] {
        &self.metric_anisotropy
    }

    /// Twisted seam index map `j -> j_seam` across $u_{\min} \leftrightarrow u_{\max}$.
    #[must_use]
    #[verified_engine::verified]
    pub fn seam_forward_map(&self) -> &[usize] {
        &self.seam_fwd
    }

    /// Computes area-weighted surface integral $\iint f(u, v)\, \sqrt{\det g}\, du\, dv$.
    #[must_use]
    #[verified_engine::verified]
    pub fn surface_integral(&self, field: &[f64]) -> f64 {
        let cell_du_dv = *self.du * *self.dv;
        field
            .iter()
            .zip(&self.area_elements)
            .map(|(&val, &area)| val * area * cell_du_dv)
            .sum()
    }

    #[inline]
    fn eval_cell_laplacian(
        &self,
        src: &[f64],
        idx: usize,
        j: usize,
        north_ring_mean: f64,
        south_ring_mean: f64,
    ) -> f64 {
        if self.topology == SurfaceTopology::PolarSphere {
            if j == 0 {
                let pole_val = src[flatten_2d_index(0, 0, self.nv)];
                return self.north_pole_rate * (north_ring_mean - pole_val);
            }
            if j + 1 == self.nv {
                let pole_val = src[flatten_2d_index(self.nv - 1, 0, self.nv)];
                return self.south_pole_rate * (south_ring_mean - pole_val);
            }
        }
        let st = &self.stencils[idx];
        let mut sum = st.diag * src[idx];
        for k in 0..8 {
            sum += st.weight[k] * src[st.idx[k]];
        }
        sum
    }

    fn polar_ring_means<const N: usize>(&self, state: [&[f64]; N]) -> ([f64; N], [f64; N]) {
        let mut north = [0.0_f64; N];
        let mut south = [0.0_f64; N];
        if self.topology != SurfaceTopology::PolarSphere || self.nv < 3 {
            return (north, south);
        }
        let inv_nu = 1.0 / self.nu as f64;
        for i in 0..self.nu {
            let n_idx = flatten_2d_index(1, i, self.nv);
            let s_idx = flatten_2d_index(self.nv - 2, i, self.nv);
            for s in 0..N {
                north[s] += state[s][n_idx];
                south[s] += state[s][s_idx];
            }
        }
        for s in 0..N {
            north[s] *= inv_nu;
            south[s] *= inv_nu;
        }
        (north, south)
    }
}

impl DiffusionModel for LaplaceBeltramiDiffusion2D {
    #[verified_engine::verified]
    fn apply(&self, state: &ChemicalState, out: &mut ChemicalState, coeffs: &[f64]) {
        let n_species = state.num_species();
        let n_grid = self.grid_size();
        assert_eq!(state.grid_size(), n_grid, "ChemicalState grid mismatch");
        assert_eq!(out.grid_size(), n_grid, "Output ChemicalState mismatch");
        assert_eq!(coeffs.len(), n_species, "Coeff count mismatch");

        for (s, &coeff) in coeffs.iter().enumerate().take(n_species) {
            let src = state.species(s);
            let dst = out.species_mut(s);
            let ([n_mean], [s_mean]) = self.polar_ring_means([src]);
            for i in 0..self.nu {
                for j in 0..self.nv {
                    let idx = flatten_2d_index(j, i, self.nv);
                    dst[idx] = coeff * self.eval_cell_laplacian(src, idx, j, n_mean, s_mean);
                }
            }
        }
    }
}

impl<const N: usize> SpatialDiffusion<N> for LaplaceBeltramiDiffusion2D {
    fn stepper(&self) -> FusedStencilStepper {
        FusedStencilStepper::new_2d(self.du, self.dv)
    }

    #[verified_engine::verified]
    fn map_diffusion<F>(&self, state: [&[f64]; N], coeffs: [f64; N], mut op: F)
    where
        F: FnMut(usize, [f64; N], [f64; N]),
    {
        if N == 0 {
            return;
        }
        let n_grid = self.grid_size();
        assert!(
            state.iter().all(|s| s.len() >= n_grid),
            "Buffer size mismatch in LaplaceBeltramiDiffusion2D"
        );
        let (north_means, south_means) = self.polar_ring_means(state);

        for i in 0..self.nu {
            for j in 0..self.nv {
                let idx = flatten_2d_index(j, i, self.nv);
                let mut curr = [0.0_f64; N];
                let mut diffs = [0.0_f64; N];
                for s in 0..N {
                    curr[s] = if self.topology == SurfaceTopology::PolarSphere
                        && (j == 0 || j + 1 == self.nv)
                    {
                        state[s][flatten_2d_index(j, 0, self.nv)]
                    } else {
                        state[s][idx]
                    };
                    let lap =
                        self.eval_cell_laplacian(state[s], idx, j, north_means[s], south_means[s]);
                    diffs[s] = coeffs[s] * lap;
                }
                op(idx, curr, diffs);
            }
        }
    }

    #[verified_engine::verified]
    fn step_fused<K: ReactionKinetics<N>>(
        &self,
        state: [&[f64]; N],
        next_state: [&mut [f64]; N],
        dt: f64,
        coeffs: [f64; N],
        kinetics: &K,
    ) {
        if N == 0 {
            return;
        }
        let max_d = coeffs.iter().fold(0.0_f64, |acc, &c| acc.max(c.abs()));
        let eff_dt = self.clamp_dt(dt, max_d);
        self.map_diffusion(state, coeffs, |idx, curr, diffs| {
            let rates = kinetics.reaction(curr);
            for s in 0..N {
                next_state[s][idx] = curr[s] + eff_dt * (diffs[s] + rates[s]);
            }
        });
    }
}

#[cfg(test)]
mod tests;
