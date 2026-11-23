use super::SpatialDiffusion;

#[allow(missing_docs)]
pub mod boundary;
#[allow(missing_docs)]
pub mod geometry;
#[allow(missing_docs)]
pub mod iteration;
pub mod mask;

pub use mask::{CellMask2D, MaskedFiniteDifference2D, ReactionCellRole};

use boundary::NeumannBoundary;
use geometry::{Cartesian2D, GeometryStrategy};
use iteration::{IterationStrategy, LoopSplittingIteration};
use math_commons::math_kernel::types::{Dimension, StepSize};

/// A 2D Finite Difference implementation using a 5-point stencil.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FiniteDifference2D {
    #[allow(missing_docs)]
    pub geometry: Cartesian2D,
    #[allow(missing_docs)]
    pub boundary: NeumannBoundary,
    #[allow(missing_docs)]
    pub iteration: LoopSplittingIteration,
}

impl FiniteDifference2D {
    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn new(width: Dimension, height: Dimension, dx: StepSize, dy: StepSize) -> Self {
        Self {
            geometry: Cartesian2D {
                width,
                height,
                dx,
                dy,
            },
            boundary: NeumannBoundary,
            iteration: LoopSplittingIteration,
        }
    }

    /// Wraps this 2D stencil operator in a [`MaskedFiniteDifference2D`] with an empty cell mask.
    #[verified_engine::verified]
    pub fn with_mask(self) -> MaskedFiniteDifference2D {
        MaskedFiniteDifference2D::new(self)
    }
}

impl crate::biology::reaction_diffusion::DiffusionModel for FiniteDifference2D {
    #[verified_engine::verified]
    fn apply(
        &self,
        state: &crate::biology::reaction_diffusion::ChemicalState,
        out: &mut crate::biology::reaction_diffusion::ChemicalState,
        coeffs: &[f64],
    ) {
        let n_species = state.num_species();
        if n_species == 0 {
            return;
        }

        let n_grid = self.geometry.size();

        assert_eq!(
            state.grid_size(),
            n_grid,
            "ChemicalState grid size mismatch with FiniteDifference2D"
        );
        assert_eq!(out.grid_size(), n_grid, "Output state grid size mismatch");
        assert_eq!(
            coeffs.len(),
            n_species,
            "Diffusion coefficients count mismatch"
        );

        for (s, coeff) in coeffs.iter().enumerate().take(n_species) {
            let src = state.species(s);
            let dst = out.species_mut(s);
            let coeff = *coeff;

            let inv_dx_sq = 1.0 / (*self.geometry.dx() * *self.geometry.dx());
            let inv_dy_sq = 1.0 / (*self.geometry.dy() * *self.geometry.dy());
            let cx = coeff * inv_dx_sq;
            let cy = coeff * inv_dy_sq;
            let c_center = -2.0 * (cx + cy);

            self.iteration.iterate(
                &self.geometry,
                &self.boundary,
                |idx, idx_l, idx_r, idx_u, idx_d| {
                    let u_curr = src[*idx];
                    let u_l = src[*idx_l];
                    let u_r = src[*idx_r];
                    let u_u = src[*idx_u];
                    let u_d = src[*idx_d];

                    let diff = (u_r + u_l) * cx + (u_d + u_u) * cy + u_curr * c_center;
                    dst[*idx] = diff;
                },
            );
        }
    }
}

use pure_math::pure_math::analysis::pde::fused_stepper::FusedStencilStepper;

impl<const N: usize> SpatialDiffusion<N> for FiniteDifference2D {
    fn stepper(&self) -> FusedStencilStepper {
        FusedStencilStepper::new_2d(self.geometry.dx, self.geometry.dy)
    }

    #[verified_engine::verified]
    fn step_fused<K: crate::biology::morphogenesis::reaction::ReactionKinetics<N>>(
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
        let n = self.geometry.size();
        if n == 0 {
            return;
        }

        SpatialDiffusion::<N>::stepper(self).step_2d_coupled_neumann(
            (*self.geometry.width, *self.geometry.height),
            state,
            next_state,
            dt,
            1.0, // Forward time
            |_i, curr, left, right, up, down, ops| {
                let mut rhs = [0.0; N];
                let rates = kinetics.reaction(curr);
                for s in 0..N {
                    let d2u = ops.central_diff_2nd_2d(curr[s], left[s], right[s], up[s], down[s]);
                    rhs[s] = coeffs[s] * d2u + rates[s];
                }
                rhs
            },
        );
    }

    #[verified_engine::verified]
    fn map_diffusion<F>(&self, state: [&[f64]; N], coeffs: [f64; N], mut op: F)
    where
        F: FnMut(usize, [f64; N], [f64; N]),
    {
        if N == 0 {
            return;
        }

        let n = self.geometry.size();
        if n == 0 {
            return;
        }

        if state[0].len() < n {
            panic!("Buffer size mismatch in FiniteDifference2D");
        }

        let inv_dx_sq = 1.0 / (*self.geometry.dx() * *self.geometry.dx());
        let inv_dy_sq = 1.0 / (*self.geometry.dy() * *self.geometry.dy());

        let mut cx = [0.0; N];
        let mut cy = [0.0; N];
        let mut c_center = [0.0; N];

        for (s, coeff) in coeffs.iter().enumerate().take(N) {
            cx[s] = coeff * inv_dx_sq;
            cy[s] = coeff * inv_dy_sq;
            c_center[s] = -2.0 * (cx[s] + cy[s]);
        }

        for (s, buffer) in state.iter().enumerate().take(N) {
            assert!(
                buffer.len() >= n,
                "Buffer too small for diffusion (species {})",
                s
            );
        }

        self.iteration.iterate(
            &self.geometry,
            &self.boundary,
            |idx, idx_l, idx_r, idx_u, idx_d| {
                let mut current_vals = [0.0; N];
                let mut diff_vals = [0.0; N];

                for s in 0..N {
                    let u = state[s];
                    let u_curr = u[*idx];
                    let u_l = u[*idx_l];
                    let u_r = u[*idx_r];
                    let u_u = u[*idx_u];
                    let u_d = u[*idx_d];

                    let diff = (u_r + u_l) * cx[s] + (u_d + u_u) * cy[s] + u_curr * c_center[s];

                    current_vals[s] = u_curr;
                    diff_vals[s] = diff;
                }
                op(*idx, current_vals, diff_vals);
            },
        );
    }
}

#[cfg(test)]
mod tests_2d {
    use super::*;

    #[test]
    #[verified_engine::verified]
    fn test_laplacian_uniform() {
        let width = 10;
        let height = 10;
        let diff = FiniteDifference2D::new(
            math_commons::math_kernel::types::Dimension(width),
            math_commons::math_kernel::types::Dimension(height),
            math_commons::math_kernel::types::StepSize(1.0),
            math_commons::math_kernel::types::StepSize(1.0),
        );

        let n = width * height;
        let u = vec![1.0; n];
        let v = vec![2.0; n];
        let mut out_u = vec![0.0; n];
        let mut out_v = vec![0.0; n];

        // Use array arguments for N=2
        diff.apply(
            [u.as_slice(), v.as_slice()],
            [out_u.as_mut_slice(), out_v.as_mut_slice()],
            [1.0, 1.0],
        );

        for val in out_u {
            assert_eq!(val, 0.0);
        }
        for val in out_v {
            assert_eq!(val, 0.0);
        }
    }

    #[test]
    #[verified_engine::verified]
    fn test_laplacian_parabolic() {
        let width = 5;
        let height = 5;
        let diff = FiniteDifference2D::new(
            math_commons::math_kernel::types::Dimension(width),
            math_commons::math_kernel::types::Dimension(height),
            math_commons::math_kernel::types::StepSize(1.0),
            math_commons::math_kernel::types::StepSize(1.0),
        );

        let n = width * height;
        let mut u = vec![0.0; n];
        let v = vec![0.0; n]; // unused
        let mut out_u = vec![0.0; n];
        let mut out_v = vec![0.0; n];

        use math_commons::math_kernel::types::flatten_2d_index;
        // u = x^2 + y^2
        for y in 0..height {
            for x in 0..width {
                let idx = flatten_2d_index(x, y, width);
                u[idx] = (x as f64).powi(2) + (y as f64).powi(2);
            }
        }

        diff.apply(
            [u.as_slice(), v.as_slice()],
            [out_u.as_mut_slice(), out_v.as_mut_slice()],
            [1.0, 1.0],
        );

        // Interior points should be exactly 4.0
        // (1,1) is index 1*5 + 1 = 6.
        // (3,3) is index 3*5 + 3 = 18.
        // Interior range: x in 1..3, y in 1..3
        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let idx = flatten_2d_index(x, y, width);
                assert!(
                    (out_u[idx] - 4.0).abs() < math_commons::registry::TOLERANCE_HIGH,
                    "Failed at ({}, {}): {}",
                    x,
                    y,
                    out_u[idx]
                );
            }
        }
    }

    #[test]
    #[verified_engine::verified]
    fn test_map_diffusion_equivalence() {
        let width = 5;
        let height = 5;
        let diff = FiniteDifference2D::new(
            math_commons::math_kernel::types::Dimension(width),
            math_commons::math_kernel::types::Dimension(height),
            math_commons::math_kernel::types::StepSize(1.0),
            math_commons::math_kernel::types::StepSize(1.0),
        );

        let n = width * height;
        let mut u = vec![0.0; n];
        let mut v = vec![0.0; n];

        // Randomish initialization
        for i in 0..n {
            u[i] = (i as f64) * 0.1;
            v[i] = (n - i) as f64 * 0.1;
        }

        let mut out_u_1 = vec![0.0; n];
        let mut out_v_1 = vec![0.0; n];
        let mut out_u_2 = vec![0.0; n];
        let mut out_v_2 = vec![0.0; n];

        let dt = 0.01;
        let d_u = 0.5;
        let d_v = 0.1;

        // Method 1: Manual step using apply
        diff.apply(
            [u.as_slice(), v.as_slice()],
            [out_u_1.as_mut_slice(), out_v_1.as_mut_slice()],
            [d_u, d_v],
        );
        for i in 0..n {
            out_u_1[i] = u[i] + dt * (out_u_1[i] + 1.0); // Dummy reaction +1
            out_v_1[i] = v[i] + dt * (out_v_1[i] + 2.0); // Dummy reaction +2
        }

        // Method 2: map_diffusion fused step
        diff.map_diffusion(
            [u.as_slice(), v.as_slice()],
            [d_u, d_v],
            |i, vals, diffs| {
                let u_curr = vals[0];
                let v_curr = vals[1];
                let diff_u = diffs[0];
                let diff_v = diffs[1];

                let (reac_u, reac_v) = (1.0, 2.0);
                out_u_2[i] = u_curr + dt * (diff_u + reac_u);
                out_v_2[i] = v_curr + dt * (diff_v + reac_v);
            },
        );

        for i in 0..n {
            assert!((out_u_1[i] - out_u_2[i]).abs() < math_commons::registry::TOLERANCE_HIGH);
            assert!((out_v_1[i] - out_v_2[i]).abs() < math_commons::registry::TOLERANCE_HIGH);
        }
    }

    #[test]
    #[verified_engine::verified]
    fn test_diffusion_model_2d() {
        use crate::biology::reaction_diffusion::{ChemicalState, DiffusionModel};

        let width = 5;
        let height = 5;
        let dx = 1.0;
        let dy = 1.0;
        let diff = FiniteDifference2D::new(
            math_commons::math_kernel::types::Dimension(width),
            math_commons::math_kernel::types::Dimension(height),
            math_commons::math_kernel::types::StepSize(dx),
            math_commons::math_kernel::types::StepSize(dy),
        );

        let n = width * height;
        let mut state = ChemicalState::new(2, n);
        let mut out = ChemicalState::new(2, n);

        // Initialize with a simple pattern: center point high
        let center_idx = 2 * width + 2; // (2, 2)
        state.species_mut(0)[center_idx] = 1.0;

        // Apply diffusion
        let coeffs = [0.1, 0.2];
        DiffusionModel::apply(&diff, &state, &mut out, &coeffs);

        // Check center point diffusion
        // Laplacian at center: (0+0+0+0 - 4*1) = -4
        // D*Lap = 0.1 * -4 = -0.4
        let expected_center = -4.0 * coeffs[0];
        let val = out.species(0)[center_idx];
        assert!(
            (val - expected_center).abs() < math_commons::registry::TOLERANCE_HIGH,
            "Expected {}, got {}",
            expected_center,
            val
        );

        // Check neighbor points (should receive flux)
        // Neighbor Laplacian: (1+0+0+0 - 4*0) = 1
        // D*Lap = 0.1 * 1 = 0.1
        let neighbor_idx = 2 * width + 3; // (3, 2)
        let expected_neighbor = 1.0 * coeffs[0];
        let val_neighbor = out.species(0)[neighbor_idx];
        assert!(
            (val_neighbor - expected_neighbor).abs() < math_commons::registry::TOLERANCE_HIGH,
            "Expected {}, got {}",
            expected_neighbor,
            val_neighbor
        );

        // Verify 2nd species works independently
        assert_eq!(out.species(1)[center_idx], 0.0);
    }
}
// [cite:essay]
