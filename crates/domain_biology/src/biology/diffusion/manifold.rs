//! Intrinsic Cotangent-Mesh Spatial Diffusion Strategy for Curved 2-Manifolds.
//!
//! Couples [`CotangentLaplacian`] in Compressed Sparse Row (CSR) format to [`SpatialDiffusion`]
//! and [`DiffusionModel`] with zero heap allocations inside hot `step_fused` and `map_diffusion` loops.
// [cite:freesurfer]

use super::SpatialDiffusion;
use crate::biology::morphogenesis::reaction::ReactionKinetics;
use crate::biology::reaction_diffusion::{ChemicalState, DiffusionModel};
use math_commons::math_kernel::types::StepSize;
use oxidize_core::mesh::Mesh;
use pure_math::pure_math::analysis::pde::fused_stepper::FusedStencilStepper;
use pure_math::pure_math::differential_geometry::cotangent_mesh::CotangentLaplacian;

/// Intrinsic Riemannian diffusion operator over a 3D triangulated 2-manifold backed by [`CotangentLaplacian`].
#[derive(Debug, Clone, PartialEq)]
pub struct CotangentMeshDiffusion {
    laplacian: CotangentLaplacian,
}

impl CotangentMeshDiffusion {
    /// Wraps an existing [`CotangentLaplacian`] operator for reaction-diffusion stepping.
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub fn new(laplacian: CotangentLaplacian) -> Self {
        Self { laplacian }
    }

    /// Builds a cotangent manifold diffusion operator from indexed 3D vertex positions and triangles.
    #[must_use]
    pub fn from_indexed(positions: &[[f64; 3]], triangles: &[[usize; 3]]) -> Self {
        Self {
            laplacian: CotangentLaplacian::from_indexed(positions, triangles),
        }
    }

    /// Builds a cotangent manifold diffusion operator directly from an indexed [`Mesh`].
    #[must_use]
    pub fn from_mesh(mesh: &Mesh) -> Self {
        Self {
            laplacian: CotangentLaplacian::from_mesh(mesh),
        }
    }

    /// Underlying CSR [`CotangentLaplacian`] operator.
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub fn laplacian(&self) -> &CotangentLaplacian {
        &self.laplacian
    }

    /// Number of vertices in the manifold mesh.
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub fn num_vertices(&self) -> usize {
        self.laplacian.num_vertices()
    }
}

impl DiffusionModel for CotangentMeshDiffusion {
    #[verified_engine::verified]
    fn apply(&self, state: &ChemicalState, out: &mut ChemicalState, coeffs: &[f64]) {
        let n_species = state.num_species();
        if n_species == 0 {
            return;
        }
        let n_verts = self.num_vertices();
        assert_eq!(
            state.grid_size(),
            n_verts,
            "ChemicalState vertex count mismatch"
        );
        assert_eq!(out.grid_size(), n_verts, "Output vertex count mismatch");
        assert_eq!(
            coeffs.len(),
            n_species,
            "Diffusion coefficients count mismatch"
        );

        for (s, &coeff) in coeffs.iter().enumerate().take(n_species) {
            let src = state.species(s);
            let dst = out.species_mut(s);
            for (v, slot) in dst.iter_mut().enumerate().take(n_verts) {
                *slot = coeff * self.laplacian.apply_at(src, v);
            }
        }
    }
}

impl<const N: usize> SpatialDiffusion<N> for CotangentMeshDiffusion {
    fn stepper(&self) -> FusedStencilStepper {
        FusedStencilStepper::new(StepSize(self.laplacian.mean_edge_length().max(1e-6)))
    }

    #[verified_engine::verified]
    fn map_diffusion<F>(&self, state: [&[f64]; N], coeffs: [f64; N], mut op: F)
    where
        F: FnMut(usize, [f64; N], [f64; N]),
    {
        let n_verts = self.num_vertices();
        if N == 0 || n_verts == 0 {
            return;
        }
        for species in &state {
            assert_eq!(species.len(), n_verts, "Species buffer length mismatch");
        }
        for v in 0..n_verts {
            let mut curr = [0.0; N];
            let mut diffs = [0.0; N];
            for s in 0..N {
                curr[s] = state[s][v];
                diffs[s] = coeffs[s] * self.laplacian.apply_at(state[s], v);
            }
            op(v, curr, diffs);
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
        let n_verts = self.num_vertices();
        if N == 0 || n_verts == 0 {
            return;
        }
        for s in 0..N {
            assert_eq!(state[s].len(), n_verts, "State slice length mismatch");
            assert_eq!(
                next_state[s].len(),
                n_verts,
                "Next-state slice length mismatch"
            );
        }
        for v in 0..n_verts {
            let mut curr = [0.0; N];
            let mut diffs = [0.0; N];
            for s in 0..N {
                curr[s] = state[s][v];
                diffs[s] = coeffs[s] * self.laplacian.apply_at(state[s], v);
            }
            let react = kinetics.reaction(curr);
            for s in 0..N {
                next_state[s][v] = curr[s] + dt * (diffs[s] + react[s]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::biology::morphogenesis::{
        DiffusionCoeff, GrayScottKinetics, SchnakenbergKinetics, TuringSystem,
    };
    use math_commons::math_kernel::types::Dimension;
    use std::f64::consts::PI;
    use verified_engine::engine::VerifiedEngine;

    fn sphere_diffusion(nu: usize, nv: usize) -> (CotangentMeshDiffusion, Vec<[f64; 3]>) {
        let mut positions = Vec::with_capacity(2 + (nv - 2) * nu);
        positions.push([0.0, 0.0, 1.0]);
        positions.push([0.0, 0.0, -1.0]);
        for j in 1..(nv - 1) {
            let v = PI * (j as f64) / ((nv - 1) as f64);
            let (sv, cv) = v.sin_cos();
            for i in 0..nu {
                let u = 2.0 * PI * (i as f64) / (nu as f64);
                let (su, cu) = u.sin_cos();
                positions.push([sv * cu, sv * su, cv]);
            }
        }
        let ring = |i: usize, j: usize| 2 + (j - 1) * nu + (i % nu);
        let mut tris = Vec::new();
        for i in 0..nu {
            tris.push([0, ring(i, 1), ring(i + 1, 1)]);
        }
        for j in 1..(nv - 2) {
            for i in 0..nu {
                tris.push([ring(i, j), ring(i, j + 1), ring(i + 1, j + 1)]);
                tris.push([ring(i, j), ring(i + 1, j + 1), ring(i + 1, j)]);
            }
        }
        for i in 0..nu {
            tris.push([ring(i, nv - 2), 1, ring(i + 1, nv - 2)]);
        }
        (
            CotangentMeshDiffusion::from_indexed(&positions, &tris),
            positions,
        )
    }

    #[test]
    fn test_cotangent_mesh_diffusion_schnakenberg_and_gray_scott_100_steps_zero_alloc() {
        let (diff_s, pos) = sphere_diffusion(16, 12);
        let diff_g = diff_s.clone();
        let n = pos.len();

        let schnak = SchnakenbergKinetics::new(0.1, 0.9);
        let mut sys_s = TuringSystem::new_with_kinetics(
            Dimension(n),
            DiffusionCoeff(0.005),
            DiffusionCoeff(0.1),
            schnak,
            diff_s,
        );
        for (i, p) in pos.iter().enumerate() {
            sys_s.u_mut()[i] = 1.0 + 0.05 * p[2];
            sys_s.v_mut()[i] = 0.9 - 0.05 * p[0];
        }

        let gs = GrayScottKinetics::new(0.037, 0.06);
        let mut sys_g = TuringSystem::new_with_kinetics(
            Dimension(n),
            DiffusionCoeff(0.002),
            DiffusionCoeff(0.001),
            gs,
            diff_g,
        );
        for (i, p) in pos.iter().enumerate() {
            sys_g.u_mut()[i] = 1.0;
            sys_g.v_mut()[i] = if p[2] > 0.7 { 0.25 } else { 0.0 };
        }

        let ((), _metrics) = VerifiedEngine::run_verified(|| {
            for _ in 0..100 {
                sys_s.step(1e-3);
                sys_g.step(1e-2);
            }
        });

        assert!(sys_s.u().iter().all(|v| v.is_finite()));
        assert!(sys_s.v().iter().all(|v| v.is_finite()));
        assert!(sys_g.u().iter().all(|v| v.is_finite()));
        assert!(sys_g.v().iter().all(|v| v.is_finite()));
    }
}
