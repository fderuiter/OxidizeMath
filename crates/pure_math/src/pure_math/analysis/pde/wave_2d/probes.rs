//! Zero-allocation bilinear point-probe readers, detector profiles, and pulse injection
//! for [`crate::pure_math::analysis::pde::wave_2d::Wave2DSolver`].
//!
//! Samples signed displacement $u(x, y, t)$ and backward-difference Störmer-Verlet velocity
//! $u_t(x, y, t) = (u^n - u^{n-1}) / \Delta t_{\text{sub}}$ at arbitrary normalized coordinates
//! $(x, y) \in [0, 1]^2$ without dynamic heap allocation under `VerifiedAllocator::lock_allocations()`.
//!
//! # Example
//!
//! ```
//! # use pure_math::__doctest_facade as math_explorer;
//! use math_explorer::pure_math::analysis::pde::wave_2d::{Wave2DParams, Wave2DSolver};
//!
//! let mut solver = Wave2DSolver::new_unit_square(32);
//! solver.inject_gaussian_pulse(0.5, 0.5, 0.08, 1.0);
//! let initial = solver.sample_probe(0.5, 0.5);
//! assert!(initial.displacement > 0.5);
//!
//! solver.step(&Wave2DParams::default(), 0.002);
//! let stepped = solver.sample_probe(0.5, 0.5);
//! assert!(stepped.displacement.is_finite() && stepped.velocity.is_finite());
//! ```

// [cite:advanced_linear_algebra]

use math_commons::math_kernel::types::flatten_2d_index;

use super::{Wave2DSolver, masks};

/// Signed displacement and velocity sampled at a continuous 2D acoustic pickup probe.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct WaveProbeSample {
    /// Bilinearly interpolated signed displacement $u(x, y, t)$.
    pub displacement: f64,
    /// Bilinearly interpolated signed velocity $\partial_t u(x, y, t)$.
    pub velocity: f64,
}

#[derive(Clone, Copy)]
struct BilinearCoords {
    i00: usize,
    i10: usize,
    i01: usize,
    i11: usize,
    tx: f64,
    ty: f64,
}

#[verified_engine::verified]
fn bilinear_grid_coords(width: usize, height: usize, x_norm: f64, y_norm: f64) -> BilinearCoords {
    assert!(width >= 2 && height >= 2, "Grid dimensions must be >= 2");
    let xn = if x_norm.is_finite() {
        x_norm.clamp(0.0, 1.0)
    } else {
        0.5
    };
    let yn = if y_norm.is_finite() {
        y_norm.clamp(0.0, 1.0)
    } else {
        0.5
    };
    let fx = xn * ((width - 1) as f64);
    let fy = yn * ((height - 1) as f64);
    let x0 = (fx.floor() as usize).min(width - 2);
    let y0 = (fy.floor() as usize).min(height - 2);
    let tx = (fx - (x0 as f64)).clamp(0.0, 1.0);
    let ty = (fy - (y0 as f64)).clamp(0.0, 1.0);
    BilinearCoords {
        i00: flatten_2d_index(x0, y0, width),
        i10: flatten_2d_index(x0 + 1, y0, width),
        i01: flatten_2d_index(x0, y0 + 1, width),
        i11: flatten_2d_index(x0 + 1, y0 + 1, width),
        tx,
        ty,
    }
}

#[verified_engine::verified]
fn bilerp(field: &[f64], c: BilinearCoords) -> f64 {
    let top = field[c.i00] * (1.0 - c.tx) + field[c.i10] * c.tx;
    let bot = field[c.i01] * (1.0 - c.tx) + field[c.i11] * c.tx;
    let val = top * (1.0 - c.ty) + bot * c.ty;
    if val.is_finite() { val } else { 0.0 }
}

impl Wave2DSolver {
    /// Samples bilinearly interpolated signed displacement $u(x, y, t)$ at normalized `(x_norm, y_norm)`.
    #[verified_engine::verified]
    pub fn sample_displacement_bilinear(&self, x_norm: f64, y_norm: f64) -> f64 {
        if !x_norm.is_finite() || !y_norm.is_finite() {
            return 0.0;
        }
        let coords = bilinear_grid_coords(self.width, self.height, x_norm, y_norm);
        bilerp(&self.u_curr, coords)
    }

    /// Samples bilinearly interpolated signed velocity $\partial_t u(x, y, t)$ at normalized `(x_norm, y_norm)`.
    #[verified_engine::verified]
    pub fn sample_velocity_bilinear(&self, x_norm: f64, y_norm: f64) -> f64 {
        if !x_norm.is_finite()
            || !y_norm.is_finite()
            || self.stationary_start
            || !self.last_dt.is_finite()
            || self.last_dt <= 1.0e-12
        {
            return 0.0;
        }
        let coords = bilinear_grid_coords(self.width, self.height, x_norm, y_norm);
        let curr = bilerp(&self.u_curr, coords);
        let prev = bilerp(&self.u_prev, coords);
        let vel = (curr - prev) / self.last_dt;
        if vel.is_finite() { vel } else { 0.0 }
    }

    /// Samples both signed displacement and signed velocity at normalized pickup coordinates `(x_norm, y_norm)`.
    #[verified_engine::verified]
    pub fn sample_probe(&self, x_norm: f64, y_norm: f64) -> WaveProbeSample {
        if !x_norm.is_finite() || !y_norm.is_finite() {
            return WaveProbeSample::default();
        }
        let coords = bilinear_grid_coords(self.width, self.height, x_norm, y_norm);
        let displacement = bilerp(&self.u_curr, coords);
        let velocity =
            if self.stationary_start || !self.last_dt.is_finite() || self.last_dt <= 1.0e-12 {
                0.0
            } else {
                let prev = bilerp(&self.u_prev, coords);
                let vel = (displacement - prev) / self.last_dt;
                if vel.is_finite() { vel } else { 0.0 }
            };
        WaveProbeSample {
            displacement,
            velocity,
        }
    }

    /// Writes the time-averaged intensity profile $I(y)$ along the detector screen into `out`.
    #[verified_engine::verified]
    pub fn read_detector_profile(&self, det_x: usize, out: &mut [f64]) {
        masks::read_detector_profile_into(
            &self.detector_accum,
            self.detector_samples,
            &self.energy_ema,
            self.width,
            self.height,
            det_x,
            out,
        );
    }

    /// Injects a smooth 2D Gaussian displacement pulse centered at normalized `(cx_norm, cy_norm)`.
    #[verified_engine::verified]
    pub fn inject_gaussian_pulse(
        &mut self,
        cx_norm: f64,
        cy_norm: f64,
        sigma_norm: f64,
        amplitude: f64,
    ) {
        masks::inject_gaussian_pulse_into(
            &mut self.u_curr,
            &mut self.u_prev,
            &self.wall_mask,
            self.width,
            self.height,
            cx_norm,
            cy_norm,
            sigma_norm,
            amplitude,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pure_math::analysis::pde::wave_2d::{BoundaryMode2D, Wave2DMedium, Wave2DParams};

    #[test]
    #[verified_engine::verified]
    fn test_sample_displacement_bilinear_off_grid_and_pulse_strike() {
        let n = 32_usize;
        let mut solver = Wave2DSolver::new_unit_square(n);
        let mut plane = vec![0.0; n * n];
        let denom = (n - 1) as f64;
        for y in 0..n {
            let yn = (y as f64) / denom;
            for x in 0..n {
                let xn = (x as f64) / denom;
                plane[flatten_2d_index(x, y, n)] = 2.5 * xn - 1.75 * yn + 0.4;
            }
        }
        solver.set_stationary_displacement(&plane);

        for (qx, qy) in [(0.137, 0.842), (0.5, 0.5), (0.913, 0.219), (0.0, 1.0)] {
            let expected = 2.5 * qx - 1.75 * qy + 0.4;
            let actual = solver.sample_displacement_bilinear(qx, qy);
            assert!(
                (actual - expected).abs() < 1.0e-12,
                "Bilinear readout mismatch at ({qx}, {qy}): got {actual}, expected {expected}"
            );
        }
        assert_eq!(solver.sample_displacement_bilinear(f64::NAN, 0.5), 0.0);
        assert_eq!(solver.sample_displacement_bilinear(0.5, f64::INFINITY), 0.0);

        solver.clear_wave_state();
        solver.inject_gaussian_pulse(0.5, 0.5, 0.06, 1.0);
        let params = Wave2DParams {
            medium: Wave2DMedium::BiharmonicPlate,
            boundary: BoundaryMode2D::Clamped,
            wave_speed: 0.0,
            plate_stiffness: 1.0e-3,
            damping: 0.02,
        };
        let (peak, _metrics) = verified_engine::engine::VerifiedEngine::run_verified(|| {
            let mut max_abs = 0.0_f64;
            for _ in 0..64 {
                solver.step(&params, 0.002);
                let l = solver.sample_displacement_bilinear(0.33, 0.41);
                let r = solver.sample_displacement_bilinear(0.67, 0.59);
                max_abs = max_abs.max(l.abs()).max(r.abs());
            }
            max_abs
        });
        assert!(peak > 0.0);
        let sym_a = solver.sample_displacement_bilinear(0.31, 0.43);
        let sym_b = solver.sample_displacement_bilinear(0.69, 0.57);
        assert!((sym_a - sym_b).abs() < 1.0e-11);
    }
}
