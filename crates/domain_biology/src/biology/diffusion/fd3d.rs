//! 3D 7-Point Finite Difference Diffusion Strategy with Zero-Flux Neumann Boundaries.
// [cite:favorite_child]

use super::SpatialDiffusion;
use crate::biology::morphogenesis::reaction::ReactionKinetics;
use crate::biology::reaction_diffusion::{ChemicalState, DiffusionModel};
use math_commons::math_kernel::types::{Dimension, StepSize, flatten_3d_index};
use pure_math::pure_math::analysis::pde::fused_stepper::FusedStencilStepper;

/// 3D Cartesian Finite Difference diffusion operator using a 7-point Laplacian stencil
/// and zero-flux Neumann boundary clamping over [`FusedStencilStepper`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FiniteDifference3D {
    /// Grid extent along the X axis.
    pub width: Dimension,
    /// Grid extent along the Y axis.
    pub height: Dimension,
    /// Grid extent along the Z axis.
    pub depth: Dimension,
    /// Spatial step along X.
    pub dx: StepSize,
    /// Spatial step along Y.
    pub dy: StepSize,
    /// Spatial step along Z.
    pub dz: StepSize,
}

impl FiniteDifference3D {
    /// Constructs an anisotropic 3D 7-point finite-difference operator.
    #[verified_engine::verified]
    pub fn new(
        width: Dimension,
        height: Dimension,
        depth: Dimension,
        dx: StepSize,
        dy: StepSize,
        dz: StepSize,
    ) -> Self {
        assert!(
            *dx > 0.0 && *dy > 0.0 && *dz > 0.0,
            "3D step sizes must be positive"
        );
        Self {
            width,
            height,
            depth,
            dx,
            dy,
            dz,
        }
    }

    /// Constructs an isotropic 3D 7-point finite-difference operator (`dx = dy = dz = h`).
    #[verified_engine::verified]
    pub fn isotropic(width: Dimension, height: Dimension, depth: Dimension, h: StepSize) -> Self {
        Self::new(width, height, depth, h, h, h)
    }

    /// Total number of voxels `width * height * depth`.
    #[inline]
    #[verified_engine::verified]
    pub fn grid_len(&self) -> usize {
        (*self.width) * (*self.height) * (*self.depth)
    }
}

impl DiffusionModel for FiniteDifference3D {
    #[verified_engine::verified]
    fn apply(&self, state: &ChemicalState, out: &mut ChemicalState, coeffs: &[f64]) {
        let n_species = state.num_species();
        if n_species == 0 {
            return;
        }
        let n_grid = self.grid_len();
        assert_eq!(
            state.grid_size(),
            n_grid,
            "ChemicalState grid size mismatch with FiniteDifference3D"
        );
        assert_eq!(out.grid_size(), n_grid, "Output state grid size mismatch");
        assert_eq!(
            coeffs.len(),
            n_species,
            "Diffusion coefficients count mismatch"
        );

        let w = *self.width;
        let h = *self.height;
        let d = *self.depth;
        let slice = w * h;
        let ops = FusedStencilStepper::new_3d(self.dx, self.dy, self.dz).ops;

        for (s, &coeff) in coeffs.iter().enumerate().take(n_species) {
            let src = state.species(s);
            let dst = out.species_mut(s);
            for z in 0..d {
                for y in 0..h {
                    for x in 0..w {
                        let idx = flatten_3d_index(x, y, z, w, h);
                        let neighbors = [
                            src[if x > 0 { idx - 1 } else { idx }],
                            src[if x + 1 < w { idx + 1 } else { idx }],
                            src[if y > 0 { idx - w } else { idx }],
                            src[if y + 1 < h { idx + w } else { idx }],
                            src[if z > 0 { idx - slice } else { idx }],
                            src[if z + 1 < d { idx + slice } else { idx }],
                        ];
                        dst[idx] = coeff * ops.central_diff_2nd_3d(src[idx], neighbors);
                    }
                }
            }
        }
    }
}

impl<const N: usize> SpatialDiffusion<N> for FiniteDifference3D {
    fn stepper(&self) -> FusedStencilStepper {
        FusedStencilStepper::new_3d(self.dx, self.dy, self.dz)
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
        if N == 0 || self.grid_len() == 0 {
            return;
        }
        let stepper = <Self as SpatialDiffusion<N>>::stepper(self);
        stepper.step_3d_coupled_neumann(
            (*self.width, *self.height, *self.depth),
            state,
            next_state,
            dt,
            1.0,
            |idx, curr, nb, ops| {
                let react = kinetics.reaction_at(idx, curr);
                let mut rhs = [0.0; N];
                for s in 0..N {
                    let lap = ops.central_diff_2nd_3d(
                        curr[s],
                        [nb.xm[s], nb.xp[s], nb.ym[s], nb.yp[s], nb.zm[s], nb.zp[s]],
                    );
                    rhs[s] = coeffs[s] * lap + react[s];
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
        let w = *self.width;
        let h = *self.height;
        let d = *self.depth;
        let slice = w * h;
        if N == 0 || slice * d == 0 {
            return;
        }
        let ops = FusedStencilStepper::new_3d(self.dx, self.dy, self.dz).ops;
        let get_species = |idx: usize| {
            let mut vals = [0.0; N];
            for s in 0..N {
                if idx < state[s].len() {
                    vals[s] = state[s][idx];
                }
            }
            vals
        };
        for z in 0..d {
            for y in 0..h {
                for x in 0..w {
                    let idx = flatten_3d_index(x, y, z, w, h);
                    let curr = get_species(idx);
                    let xm = get_species(if x > 0 { idx - 1 } else { idx });
                    let xp = get_species(if x + 1 < w { idx + 1 } else { idx });
                    let ym = get_species(if y > 0 { idx - w } else { idx });
                    let yp = get_species(if y + 1 < h { idx + w } else { idx });
                    let zm = get_species(if z > 0 { idx - slice } else { idx });
                    let zp = get_species(if z + 1 < d { idx + slice } else { idx });
                    let mut diffs = [0.0; N];
                    for s in 0..N {
                        diffs[s] = coeffs[s]
                            * ops.central_diff_2nd_3d(
                                curr[s],
                                [xm[s], xp[s], ym[s], yp[s], zm[s], zp[s]],
                            );
                    }
                    op(idx, curr, diffs);
                }
            }
        }
    }
}
