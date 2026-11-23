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
        let coords = bilinear_grid_coords(self.width, self.height, x_norm, y_norm);
        bilerp(&self.u_curr, coords)
    }

    /// Samples bilinearly interpolated signed velocity $\partial_t u(x, y, t)$ at normalized `(x_norm, y_norm)`.
    #[verified_engine::verified]
    pub fn sample_velocity_bilinear(&self, x_norm: f64, y_norm: f64) -> f64 {
        if self.stationary_start || !self.last_dt.is_finite() || self.last_dt <= 1.0e-12 {
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
        assert!(
            out.len() >= self.height,
            "Output buffer shorter than grid height"
        );
        if self.detector_samples > 0 {
            let inv = 1.0 / (self.detector_samples as f64);
            for (dst, &acc) in out.iter_mut().zip(self.detector_accum.iter()) {
                *dst = acc * inv;
            }
        } else {
            let col = det_x.min(self.width - 1);
            for (y, dst) in out.iter_mut().enumerate().take(self.height) {
                *dst = self.energy_ema[flatten_2d_index(col, y, self.width)];
            }
        }
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
