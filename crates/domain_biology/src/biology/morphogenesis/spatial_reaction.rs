//! Spatially heterogeneous Schnakenberg reaction kinetics with preallocated cell masks.

use super::reaction::ReactionKinetics;
use crate::biology::reaction_diffusion::{ChemicalState, ReactionModel};

/// Spatially varying Schnakenberg kinetics with preallocated per-cell $(a_i, b_i)$ masks,
/// evaluating local reaction rates $R_u(u_i, v_i) = a_i - u_i + u_i^2 v_i$ and
/// $R_v(u_i, v_i) = b_i - u_i^2 v_i$ at grid cell $i$.
///
/// Unmasked cells evaluate the scalar baseline $(a, b)$ rates, while masked cells
/// evaluate their preallocated $(a_i, b_i)$ overrides with zero heap allocation during
/// reaction-diffusion integration steps.
///
/// Reference: Schnakenberg (1979), DOI: `10.1016/0022-5193(79)90002-4`; see
/// [`docs/math_specs/domain_biology.md`](../../../../../docs/math_specs/domain_biology.md).
///
/// # Stability
/// Explicit reaction-diffusion stepping requires bounded $\Delta t$ against cubic $u_i^2 v_i$ stiffness
/// across $\max_i (a_i + b_i)$ alongside the diffusive parabolic CFL bound.
///
/// # Domain of Validity
/// Non-negative species concentrations $u_i, v_i \ge 0$, feed rates $a_i \ge 0, b_i > 0$ with $a_i + b_i > 0$,
/// and concentration slice lengths matching `grid_size`.
#[derive(Debug, Clone)]
pub struct SpatialSchnakenbergKinetics {
    /// Baseline activator production rate ($a$).
    pub a: f64,
    /// Baseline inhibitor production rate ($b$).
    pub b: f64,
    a_mask: Vec<f64>,
    b_mask: Vec<f64>,
    mask_override: Vec<bool>,
    active_overrides: usize,
}

impl SpatialSchnakenbergKinetics {
    /// Creates a spatially heterogeneous Schnakenberg model with preallocated mask buffers.
    #[verified_engine::verified]
    pub fn new(a: f64, b: f64, grid_size: usize) -> Self {
        Self {
            a,
            b,
            a_mask: vec![a; grid_size],
            b_mask: vec![b; grid_size],
            mask_override: vec![false; grid_size],
            active_overrides: 0,
        }
    }

    /// Returns the preallocated grid cell capacity.
    #[inline]
    #[verified_engine::verified]
    pub fn grid_size(&self) -> usize {
        self.mask_override.len()
    }

    /// Returns `true` when at least one cell holds an active $(a_i, b_i)$ override.
    #[inline]
    #[verified_engine::verified]
    pub fn has_spatial_mask(&self) -> bool {
        self.active_overrides > 0
    }

    /// Returns whether cell `index` overrides the baseline $(a, b)$ parameters.
    #[inline]
    #[verified_engine::verified]
    pub fn is_overridden(&self, index: usize) -> bool {
        index < self.mask_override.len() && self.mask_override[index]
    }

    /// Returns the effective $(a_i, b_i)$ rates at linear cell `index`.
    #[inline]
    #[verified_engine::verified]
    pub fn rates_at(&self, index: usize) -> (f64, f64) {
        if self.active_overrides > 0
            && index < self.mask_override.len()
            && self.mask_override[index]
        {
            (self.a_mask[index], self.b_mask[index])
        } else {
            (self.a, self.b)
        }
    }

    /// Assigns a local $(a_i, b_i)$ reaction override at `index` without reallocating.
    #[verified_engine::verified]
    pub fn set_cell_rates(&mut self, index: usize, a: f64, b: f64) {
        if index >= self.mask_override.len() {
            return;
        }
        self.a_mask[index] = a;
        self.b_mask[index] = b;
        if !self.mask_override[index] {
            self.mask_override[index] = true;
            self.active_overrides += 1;
        }
    }

    /// Clears any local override at `index`, reverting that cell to baseline $(a, b)$.
    #[verified_engine::verified]
    pub fn clear_cell_override(&mut self, index: usize) {
        if index < self.mask_override.len() && self.mask_override[index] {
            self.mask_override[index] = false;
            self.active_overrides = self.active_overrides.saturating_sub(1);
        }
    }

    /// Clears all spatial overrides in place without dropping preallocated capacity.
    #[verified_engine::verified]
    pub fn clear_mask(&mut self) {
        self.mask_override.fill(false);
        self.active_overrides = 0;
    }
}

impl ReactionKinetics<2> for SpatialSchnakenbergKinetics {
    #[inline]
    #[verified_engine::verified]
    fn reaction(&self, concentrations: [f64; 2]) -> [f64; 2] {
        let u = concentrations[0];
        let v = concentrations[1];
        let uv_sq = u * u * v;
        [self.a - u + uv_sq, self.b - uv_sq]
    }

    #[inline]
    #[verified_engine::verified]
    fn reaction_at(&self, index: usize, concentrations: [f64; 2]) -> [f64; 2] {
        let (rate_a, rate_b) = self.rates_at(index);
        let u = concentrations[0];
        let v = concentrations[1];
        let uv_sq = u * u * v;
        [rate_a - u + uv_sq, rate_b - uv_sq]
    }
}

impl ReactionModel for SpatialSchnakenbergKinetics {
    #[verified_engine::verified]
    fn reaction(&self, concentrations: &[f64], rates: &mut [f64]) {
        if concentrations.len() < 2 || rates.len() < 2 {
            return;
        }
        let out =
            <Self as ReactionKinetics<2>>::reaction(self, [concentrations[0], concentrations[1]]);
        rates[0] = out[0];
        rates[1] = out[1];
    }

    #[verified_engine::verified]
    fn add_reaction_batch(&self, state: &ChemicalState, rates: &mut ChemicalState) {
        if state.num_species() < 2 {
            return;
        }
        let u_vec = state.species(0);
        let v_vec = state.species(1);
        let n = state.grid_size();
        let (rates_u, rates_v) = rates.grid.as_mut_slice().split_at_mut(n);
        for i in 0..n {
            let out = self.reaction_at(i, [u_vec[i], v_vec[i]]);
            rates_u[i] += out[0];
            rates_v[i] += out[1];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::biology::diffusion::{FiniteDifference2D, LaplaceBeltramiDiffusion2D};
    use crate::biology::morphogenesis::{DiffusionCoeff, TuringSystem};
    use math_commons::math_kernel::types::{Dimension, StepSize};
    use pure_math::pure_math::differential_geometry::buckled_flow::SurfaceTopology;
    use pure_math::pure_math::differential_geometry::surface::Torus;
    use std::f64::consts::TAU;

    #[test]
    #[verified_engine::verified]
    fn test_spatial_schnakenberg_mask_overrides_and_clear() {
        let mut kin = SpatialSchnakenbergKinetics::new(0.1, 0.9, 16);
        assert!(!kin.has_spatial_mask());
        assert_eq!(kin.rates_at(5), (0.1, 0.9));

        kin.set_cell_rates(5, 0.14, 0.86);
        assert!(kin.has_spatial_mask());
        assert!(kin.is_overridden(5));
        assert!(!kin.is_overridden(6));

        let unmasked = kin.reaction_at(6, [1.0, 0.9]);
        let masked = kin.reaction_at(5, [1.0, 0.9]);
        assert!((unmasked[0] - 0.0).abs() < 1e-12);
        assert!((masked[0] - 0.04).abs() < 1e-12);
        assert!((masked[1] - (0.86 - 0.9)).abs() < 1e-12);

        kin.clear_cell_override(5);
        assert!(!kin.has_spatial_mask());
        assert_eq!(kin.rates_at(5), (0.1, 0.9));

        kin.set_cell_rates(2, 0.2, 0.8);
        kin.clear_mask();
        assert!(!kin.has_spatial_mask());
    }

    #[test]
    fn test_spatial_schnakenberg_step_zero_heap_allocation() {
        let w = 16;
        let h = 16;
        let n = w * h;
        let mut kin = SpatialSchnakenbergKinetics::new(0.1, 0.9, n);
        for i in 0..(n / 2) {
            kin.set_cell_rates(i, 0.14, 0.86);
        }
        let diff =
            FiniteDifference2D::new(Dimension(w), Dimension(h), StepSize(1.0), StepSize(1.0));
        let mut sys = TuringSystem::new_with_kinetics(
            Dimension(n),
            DiffusionCoeff(1.0),
            DiffusionCoeff(50.0),
            kin,
            diff,
        );
        for i in 0..n {
            sys.u_mut()[i] = 1.0 + (i as f64) * 1e-3;
            sys.v_mut()[i] = 0.9;
        }
        let (_, metrics) = verified_engine::engine::VerifiedEngine::run_verified(|| {
            sys.step(0.01);
        });
        assert!(metrics.function_calls > 0);
        assert!(sys.u()[0] != sys.u()[n - 1]);
    }

    #[test]
    fn test_spatial_schnakenberg_uniform_state_breaks_symmetry_fd2d_and_lb() {
        let (w, h) = (12, 12);
        let n = w * h;
        let mut kin_fd = SpatialSchnakenbergKinetics::new(0.1, 0.9, n);
        let mut kin_lb = SpatialSchnakenbergKinetics::new(0.1, 0.9, n);
        for i in 0..(n / 2) {
            kin_fd.set_cell_rates(i, 0.16, 0.84);
            kin_lb.set_cell_rates(i, 0.16, 0.84);
        }

        let diff_fd =
            FiniteDifference2D::new(Dimension(w), Dimension(h), StepSize(1.0), StepSize(1.0));
        let mut sys_fd = TuringSystem::new_with_kinetics(
            Dimension(n),
            DiffusionCoeff(1.0),
            DiffusionCoeff(20.0),
            kin_fd,
            diff_fd,
        );
        sys_fd.u_mut().fill(1.0);
        sys_fd.v_mut().fill(0.9);
        sys_fd.step(0.01);
        assert!((sys_fd.u()[0] - sys_fd.u()[n - 1]).abs() > 1e-6);

        let torus = Torus {
            major_radius: 2.2,
            minor_radius: 0.7,
        };
        let diff_lb = LaplaceBeltramiDiffusion2D::from_surface(
            &torus,
            SurfaceTopology::PeriodicTorus,
            (0.0, TAU),
            (0.0, TAU),
            w,
            h,
        );
        let mut sys_lb = TuringSystem::new_with_kinetics(
            Dimension(n),
            DiffusionCoeff(0.05),
            DiffusionCoeff(1.0),
            kin_lb,
            diff_lb,
        );
        sys_lb.u_mut().fill(1.0);
        sys_lb.v_mut().fill(0.9);
        sys_lb.step(0.01);
        assert!((sys_lb.u()[0] - sys_lb.u()[n - 1]).abs() > 1e-6);
    }
}
