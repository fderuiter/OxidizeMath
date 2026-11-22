//! # Spectral Lenia Continuous Cellular Automata
//!
//! Continuous spatial-temporal generalization of Conway's Game of Life introduced by
//! Bert Wang-Chak Chan (2019), governed by periodic ring-kernel convolution and smooth
//! Gaussian growth dynamics:
//!
//! $$ U^t = K * A^t = \mathcal{F}^{-1}\!\left(\mathcal{F}(K) \odot \mathcal{F}(A^t)\right) $$
//! $$ G(u; \mu, \sigma) = 2 \exp\!\left(-\frac{(u - \mu)^2}{2 \sigma^2}\right) - 1 $$
//! $$ A^{t + \Delta t}(x) = \mathrm{clamp}\!\left(A^t(x) + \Delta t \, G(U^t(x)),\, 0,\, 1\right) $$
//!
//! `LeniaSystem` preplans 1D row and column `rustfft` transforms and preallocates all
//! complex spectral, transpose, scratch, and state buffers at initialization so that
//! `LeniaSystem::step` executes with **zero heap allocations**.
//!
//! ## Quick Start
//!
//! ```rust
//! # use domain_biology::__doctest_facade as math_explorer;
//! use math_explorer::biology::lenia::{LeniaPreset, LeniaSystem};
//!
//! let mut system = LeniaSystem::from_preset(64, 64, LeniaPreset::Orbium)
//!     .expect("valid Lenia grid");
//! for _ in 0..10 {
//!     system.step();
//! }
//! assert!(system.telemetry().total_mass > 10.0);
//! ```

/// Concentric ring kernel builders and spectral 2D FFT helpers.
pub mod kernel;
/// Canonical Lenia lifeform presets (`Orbium`, `Hydrogeminium`).
pub mod presets;
/// Live toroidal mass, velocity, and rotational symmetry telemetry.
pub mod telemetry;

#[cfg(test)]
mod tests;

pub use kernel::{LeniaKernelCore, build_spatial_kernel, evaluate_shell_kernel};
pub use presets::LeniaPreset;
pub use telemetry::{LeniaTelemetry, shortest_toroidal_delta};

use kernel::{forward_fft_2d_transposed, transpose_complex};
use num_complex::Complex;
use rustfft::{Fft, FftPlanner};
use std::collections::HashMap;
use std::sync::Arc;
use telemetry::ToroidalBasis;
use thiserror::Error;
use verified_engine::scientific_metadata::theory::TheoryDescribable;

/// Errors raised when configuring or resizing a [`LeniaSystem`].
#[derive(Debug, Error, Clone, PartialEq)]
pub enum LeniaError {
    /// Grid width and height must both be at least 8 cells.
    #[error("Lenia grid dimensions ({width}x{height}) must each be >= 8")]
    InvalidDimensions {
        /// Requested grid width.
        width: usize,
        /// Requested grid height.
        height: usize,
    },
    /// Growth width `sigma`, step `dt`, or kernel parameters are invalid or non-finite.
    #[error("Invalid Lenia parameter `{parameter}`: {reason}")]
    InvalidParameter {
        /// Parameter name.
        parameter: &'static str,
        /// Explanation of failure.
        reason: &'static str,
    },
}

/// Dynamical and kernel parameters governing a [`LeniaSystem`].
#[derive(Debug, Clone, PartialEq)]
pub struct LeniaParams {
    /// Optimal neighborhood potential center $\mu \in (0, 1)$.
    pub mu: f64,
    /// Growth bell width $\sigma > 0$.
    pub sigma: f64,
    /// Explicit integration step $\Delta t \in (0, 1]$.
    pub dt: f64,
    /// Spatial interaction radius $R$ in grid cells.
    pub kernel_radius: usize,
    /// Concentric shell weights $\beta = [\beta_0, \dots, \beta_{B-1}]$.
    pub ring_weights: Vec<f64>,
    /// Radial bump kernel core profile.
    pub kernel_core: LeniaKernelCore,
}

impl Default for LeniaParams {
    #[verified_engine::verified]
    fn default() -> Self {
        LeniaPreset::Orbium.params()
    }
}

impl LeniaParams {
    /// Validates that all dynamical parameters are finite and strictly positive where required.
    #[verified_engine::verified]
    pub fn validate(&self) -> Result<(), LeniaError> {
        if !self.mu.is_finite() || self.mu <= 0.0 || self.mu >= 1.0 {
            return Err(LeniaError::InvalidParameter {
                parameter: "mu",
                reason: "must be finite in (0.0, 1.0)",
            });
        }
        if !self.sigma.is_finite() || self.sigma <= 1e-5 || self.sigma > 1.0 {
            return Err(LeniaError::InvalidParameter {
                parameter: "sigma",
                reason: "must be finite in (1e-5, 1.0]",
            });
        }
        if !self.dt.is_finite() || self.dt <= 0.0 || self.dt > 1.0 {
            return Err(LeniaError::InvalidParameter {
                parameter: "dt",
                reason: "must be finite in (0.0, 1.0]",
            });
        }
        if self.kernel_radius < 2 {
            return Err(LeniaError::InvalidParameter {
                parameter: "kernel_radius",
                reason: "must be >= 2 cells",
            });
        }
        if self.ring_weights.is_empty()
            || self.ring_weights.iter().any(|w| !w.is_finite() || *w < 0.0)
            || self.ring_weights.iter().all(|w| *w <= 0.0)
        {
            return Err(LeniaError::InvalidParameter {
                parameter: "ring_weights",
                reason: "must contain finite non-negative weights with positive sum",
            });
        }
        Ok(())
    }
}

impl TheoryDescribable for LeniaParams {
    fn theory_description(&self) -> String {
        "Continuous cellular automaton governed by spectral ring-kernel convolution U = F^-1(F(K) * F(A)) and smooth Gaussian growth G(u) = 2 exp(-(u - mu)^2 / (2 sigma^2)) - 1.".to_string()
    }

    fn phonetic_description(&self) -> String {
        "Continuous cellular automaton with spectral FFT ring-kernel convolution and Gaussian growth mapping mu and sigma.".to_string()
    }

    fn theory_citation(&self) -> String {
        "Chan, B. W.-C. (2019). Lenia: Biology of artificial life. Complex Systems, 28(3), 251-286."
            .to_string()
    }

    fn available_descriptions(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert(
            "en".to_string(),
            "Continuous cellular automaton governed by spectral ring-kernel convolution and Gaussian growth mapping.".to_string(),
        );
        map
    }
}

/// Evaluates the smooth Gaussian growth mapping $G(u; \mu, \sigma) \in [-1, 1]$.
#[verified_engine::verified]
pub fn growth_mapping(potential: f64, mu: f64, sigma: f64) -> f64 {
    let z = (potential - mu) / sigma;
    2.0 * (-0.5 * z * z).exp() - 1.0
}

/// Spectral 2D Lenia solver with preallocated FFT and telemetry buffers.
pub struct LeniaSystem {
    width: usize,
    height: usize,
    params: LeniaParams,
    grid: Vec<f64>,
    next_grid: Vec<f64>,
    potential: Vec<f64>,
    kernel_spatial: Vec<f64>,
    kernel_spectrum: Vec<Complex<f64>>,
    spatial_buf: Vec<Complex<f64>>,
    transposed_buf: Vec<Complex<f64>>,
    scratch_buf: Vec<Complex<f64>>,
    fft_row_fwd: Arc<dyn Fft<f64>>,
    fft_col_fwd: Arc<dyn Fft<f64>>,
    fft_row_inv: Arc<dyn Fft<f64>>,
    fft_col_inv: Arc<dyn Fft<f64>>,
    basis: ToroidalBasis,
    telemetry: LeniaTelemetry,
    has_prev_com: bool,
    step_count: u64,
}

impl LeniaSystem {
    /// Constructs a preplanned spectral Lenia solver on a `width x height` periodic torus.
    #[verified_engine::verified]
    pub fn new(width: usize, height: usize, params: LeniaParams) -> Result<Self, LeniaError> {
        if width < 8 || height < 8 {
            return Err(LeniaError::InvalidDimensions { width, height });
        }
        params.validate()?;

        let len = width * height;
        let mut planner = FftPlanner::<f64>::new();
        let fft_row_fwd = planner.plan_fft_forward(width);
        let fft_col_fwd = planner.plan_fft_forward(height);
        let fft_row_inv = planner.plan_fft_inverse(width);
        let fft_col_inv = planner.plan_fft_inverse(height);

        let scratch_len = [
            fft_row_fwd.get_inplace_scratch_len(),
            fft_col_fwd.get_inplace_scratch_len(),
            fft_row_inv.get_inplace_scratch_len(),
            fft_col_inv.get_inplace_scratch_len(),
        ]
        .into_iter()
        .max()
        .unwrap_or(0);

        let mut system = Self {
            width,
            height,
            params,
            grid: vec![0.0; len],
            next_grid: vec![0.0; len],
            potential: vec![0.0; len],
            kernel_spatial: vec![0.0; len],
            kernel_spectrum: vec![Complex::new(0.0, 0.0); len],
            spatial_buf: vec![Complex::new(0.0, 0.0); len],
            transposed_buf: vec![Complex::new(0.0, 0.0); len],
            scratch_buf: vec![Complex::new(0.0, 0.0); scratch_len.max(1)],
            fft_row_fwd,
            fft_col_fwd,
            fft_row_inv,
            fft_col_inv,
            basis: ToroidalBasis::new(width, height),
            telemetry: LeniaTelemetry::default(),
            has_prev_com: false,
            step_count: 0,
        };
        system.rebuild_kernel_spectrum();
        Ok(system)
    }

    /// Creates a `LeniaSystem` initialized with a centered canonical [`LeniaPreset`].
    #[verified_engine::verified]
    pub fn from_preset(
        width: usize,
        height: usize,
        preset: LeniaPreset,
    ) -> Result<Self, LeniaError> {
        let params = preset.params();
        let mut system = Self::new(width, height, params)?;
        system.apply_preset(preset);
        Ok(system)
    }

    /// Resets parameters and stamps `preset` at the center of the periodic grid.
    #[verified_engine::verified]
    pub fn apply_preset(&mut self, preset: LeniaPreset) {
        self.params = preset.params();
        let max_r = (self.width.min(self.height) / 2).saturating_sub(1).max(2);
        if self.params.kernel_radius > max_r {
            self.params.kernel_radius = max_r;
        }
        self.rebuild_kernel_spectrum();
        preset.stamp_into_grid(
            &mut self.grid,
            self.width,
            self.height,
            self.params.kernel_radius,
            self.width / 2,
            self.height / 2,
        );
        self.has_prev_com = false;
        self.step_count = 0;
        self.refresh_telemetry_in_place(0.0);
    }

    /// Updates parameters; recomputes the preallocated kernel spectrum if kernel geometry changed.
    #[verified_engine::verified]
    pub fn set_params(&mut self, params: LeniaParams) -> Result<(), LeniaError> {
        params.validate()?;
        let kernel_changed = self.params.kernel_radius != params.kernel_radius
            || self.params.kernel_core != params.kernel_core
            || self.params.ring_weights != params.ring_weights;
        self.params = params;
        if kernel_changed {
            self.rebuild_kernel_spectrum();
        }
        Ok(())
    }

    /// Updates scalar dynamical parameters and kernel radius without allocating ring weight vectors.
    #[verified_engine::verified]
    pub fn update_scalar_params(
        &mut self,
        mu: f64,
        sigma: f64,
        dt: f64,
        kernel_radius: usize,
    ) -> Result<(), LeniaError> {
        if !mu.is_finite() || mu <= 0.0 || mu >= 1.0 {
            return Err(LeniaError::InvalidParameter {
                parameter: "mu",
                reason: "must be finite in (0.0, 1.0)",
            });
        }
        if !sigma.is_finite() || sigma <= 1e-5 || sigma > 1.0 {
            return Err(LeniaError::InvalidParameter {
                parameter: "sigma",
                reason: "must be finite in (1e-5, 1.0]",
            });
        }
        if !dt.is_finite() || dt <= 0.0 || dt > 1.0 {
            return Err(LeniaError::InvalidParameter {
                parameter: "dt",
                reason: "must be finite in (0.0, 1.0]",
            });
        }
        if kernel_radius < 2 {
            return Err(LeniaError::InvalidParameter {
                parameter: "kernel_radius",
                reason: "must be >= 2 cells",
            });
        }
        let radius_changed = self.params.kernel_radius != kernel_radius;
        self.params.mu = mu;
        self.params.sigma = sigma;
        self.params.dt = dt;
        self.params.kernel_radius = kernel_radius;
        if radius_changed {
            self.rebuild_kernel_spectrum();
        }
        Ok(())
    }

    /// Recomputes `kernel_spatial` and its transposed 2D DFT `kernel_spectrum` in-place.
    #[verified_engine::verified]
    pub fn rebuild_kernel_spectrum(&mut self) {
        build_spatial_kernel(
            self.width,
            self.height,
            self.params.kernel_radius,
            &self.params.ring_weights,
            self.params.kernel_core,
            &mut self.kernel_spatial,
        );
        forward_fft_2d_transposed(
            &self.kernel_spatial,
            (self.width, self.height),
            &self.fft_row_fwd,
            &self.fft_col_fwd,
            &mut self.spatial_buf,
            &mut self.transposed_buf,
            &mut self.scratch_buf,
        );
        self.kernel_spectrum.copy_from_slice(&self.transposed_buf);
    }

    /// Advances the continuous cellular automaton by one step with **zero heap allocations**.
    #[verified_engine::verified]
    pub fn step(&mut self) {
        forward_fft_2d_transposed(
            &self.grid,
            (self.width, self.height),
            &self.fft_row_fwd,
            &self.fft_col_fwd,
            &mut self.spatial_buf,
            &mut self.transposed_buf,
            &mut self.scratch_buf,
        );

        for (freq, &k_hat) in self
            .transposed_buf
            .iter_mut()
            .zip(self.kernel_spectrum.iter())
        {
            *freq *= k_hat;
        }

        self.fft_col_inv
            .process_with_scratch(&mut self.transposed_buf, &mut self.scratch_buf);
        transpose_complex(
            &self.transposed_buf,
            &mut self.spatial_buf,
            self.width,
            self.height,
        );
        self.fft_row_inv
            .process_with_scratch(&mut self.spatial_buf, &mut self.scratch_buf);

        let inv_n = 1.0 / ((self.width * self.height) as f64);
        let mu = self.params.mu;
        let sigma = self.params.sigma;
        let dt = self.params.dt;

        for i in 0..self.grid.len() {
            let u = self.spatial_buf[i].re * inv_n;
            self.potential[i] = u;
            let g = growth_mapping(u, mu, sigma);
            self.next_grid[i] = (self.grid[i] + dt * g).clamp(0.0, 1.0);
        }

        std::mem::swap(&mut self.grid, &mut self.next_grid);
        self.step_count += 1;
        self.refresh_telemetry_in_place(dt);
    }

    #[verified_engine::verified]
    fn refresh_telemetry_in_place(&mut self, dt: f64) {
        let prev_com = self.telemetry.center_of_mass;
        let (mass, com) = self.basis.compute_mass_and_com(&self.grid);
        let (symmetry, fold) = self.basis.compute_angular_symmetry(&self.grid, mass, com);

        let velocity = if self.has_prev_com && dt > 0.0 && mass > 1e-9 {
            let dx = shortest_toroidal_delta(com[0], prev_com[0], self.width as f64);
            let dy = shortest_toroidal_delta(com[1], prev_com[1], self.height as f64);
            [dx / dt, dy / dt]
        } else {
            [0.0, 0.0]
        };

        self.telemetry = LeniaTelemetry {
            total_mass: mass,
            center_of_mass: com,
            velocity,
            speed: velocity[0].hypot(velocity[1]),
            rotational_symmetry: symmetry,
            dominant_symmetry_fold: fold,
        };
        self.has_prev_com = mass > 1e-9;
    }

    /// Grid width in cells.
    #[verified_engine::verified]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Grid height in cells.
    #[verified_engine::verified]
    pub fn height(&self) -> usize {
        self.height
    }

    /// Current Lenia dynamical and kernel parameters.
    #[verified_engine::verified]
    pub fn params(&self) -> &LeniaParams {
        &self.params
    }

    /// Immutable slice of cell state activations $A(x, y) \in [0, 1]$.
    #[verified_engine::verified]
    pub fn grid(&self) -> &[f64] {
        &self.grid
    }

    /// Mutable slice of cell state activations; call [`Self::recompute_telemetry`] after manual edits.
    #[verified_engine::verified]
    pub fn grid_mut(&mut self) -> &mut [f64] {
        &mut self.grid
    }

    /// Last convolved potential field $U = K * A$.
    #[verified_engine::verified]
    pub fn potential(&self) -> &[f64] {
        &self.potential
    }

    /// Spatial convolution kernel $K(x, y)$ normalized to unit sum.
    #[verified_engine::verified]
    pub fn kernel_spatial(&self) -> &[f64] {
        &self.kernel_spatial
    }

    /// Latest zero-allocation kinematic and symmetry telemetry.
    #[verified_engine::verified]
    pub fn telemetry(&self) -> LeniaTelemetry {
        self.telemetry
    }

    /// Number of completed integration steps since initialization or preset reset.
    #[verified_engine::verified]
    pub fn step_count(&self) -> u64 {
        self.step_count
    }

    /// Recomputes telemetry from the current grid without advancing time.
    #[verified_engine::verified]
    pub fn recompute_telemetry(&mut self) {
        self.has_prev_com = false;
        self.refresh_telemetry_in_place(0.0);
    }
}

// [cite:biology]
