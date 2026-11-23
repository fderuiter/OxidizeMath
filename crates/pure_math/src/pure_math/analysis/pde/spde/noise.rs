//! Pre-allocated Gaussian white-noise and conservative staggered-flux ($\nabla \cdot \boldsymbol{\xi}$)
//! spatial buffers driven by [`oxidize_core::rng::OxidizeRng`] without dynamic heap allocations.

// [cite:spde]

use math_commons::math_kernel::types::flatten_2d_index;
use oxidize_core::rng::OxidizeRng;
use rand_distr::{Distribution, Normal};

/// Pre-allocated workspace for non-conserved Gaussian noise and conserved flux-divergence fields.
///
/// Once constructed for a maximum 1D length or 2D `(width, height)` grid, all `fill_*` methods
/// populate slices in place with zero heap allocations inside `VerifiedEngine::run_verified`.
#[derive(Clone, Debug)]
pub struct SpatialNoiseBuffer {
    output: Vec<f64>,
    flux_x: Vec<f64>,
    flux_y: Vec<f64>,
    standard_normal: Normal<f64>,
}

impl SpatialNoiseBuffer {
    /// Allocates a 1D spatial noise workspace capable of serving grids up to `max_len` cells.
    #[verified_engine::verified]
    pub fn new_1d(max_len: usize) -> Self {
        assert!(max_len >= 1, "Noise buffer capacity must be >= 1");
        let standard_normal = Normal::new(0.0, 1.0).expect("Standard normal parameters are valid");
        Self {
            output: vec![0.0; max_len],
            flux_x: vec![0.0; max_len + 1],
            flux_y: Vec::new(),
            standard_normal,
        }
    }

    /// Allocates a 2D spatial noise workspace capable of serving grids up to `max_width * max_height`.
    #[verified_engine::verified]
    pub fn new_2d(max_width: usize, max_height: usize) -> Self {
        assert!(
            max_width >= 1 && max_height >= 1,
            "2D noise buffer dimensions must be >= 1"
        );
        let cap = max_width * max_height;
        let face_cap = (max_width + 1) * (max_height + 1);
        let standard_normal = Normal::new(0.0, 1.0).expect("Standard normal parameters are valid");
        Self {
            output: vec![0.0; cap],
            flux_x: vec![0.0; face_cap],
            flux_y: vec![0.0; face_cap],
            standard_normal,
        }
    }

    /// Ensures the internal workspace can hold a 2D grid of `width * height` cells (may allocate if grown).
    pub fn resize_2d(&mut self, width: usize, height: usize) {
        assert!(width >= 1 && height >= 1, "2D dimensions must be >= 1");
        let cap = width * height;
        let face_cap = (width + 1) * (height + 1);
        if self.output.len() < cap {
            self.output.resize(cap, 0.0);
        }
        if self.flux_x.len() < face_cap {
            self.flux_x.resize(face_cap, 0.0);
        }
        if self.flux_y.len() < face_cap {
            self.flux_y.resize(face_cap, 0.0);
        }
    }

    /// Ensures the internal workspace can hold a 1D grid of `len` cells (may allocate if grown).
    pub fn resize_1d(&mut self, len: usize) {
        assert!(len >= 1, "1D length must be >= 1");
        if self.output.len() < len {
            self.output.resize(len, 0.0);
        }
        if self.flux_x.len() < len + 1 {
            self.flux_x.resize(len + 1, 0.0);
        }
    }

    /// Returns an immutable slice of the most recently generated `len` noise samples.
    #[verified_engine::verified]
    pub fn slice(&self, len: usize) -> &[f64] {
        assert!(
            len <= self.output.len(),
            "Requested slice exceeds buffer capacity"
        );
        &self.output[..len]
    }

    /// Fills an external slice `out` with independent $\mathcal{N}(0, \text{scale}^2)$ samples without heap allocation.
    #[verified_engine::verified]
    pub fn fill_gaussian_into(&self, rng: &mut OxidizeRng, out: &mut [f64], scale: f64) {
        assert!(
            scale.is_finite() && scale >= 0.0,
            "Noise scale must be finite and non-negative"
        );
        sample_scaled_normals(&self.standard_normal, rng, out, scale);
    }

    /// Fills the first `len` internal entries with independent $\mathcal{N}(0, \text{scale}^2)$ samples.
    #[verified_engine::verified]
    pub fn fill_gaussian(&mut self, rng: &mut OxidizeRng, len: usize, scale: f64) -> &[f64] {
        assert!(
            len <= self.output.len(),
            "Requested length exceeds pre-allocated Gaussian buffer"
        );
        assert!(
            scale.is_finite() && scale >= 0.0,
            "Noise scale must be finite and non-negative"
        );
        sample_scaled_normals(&self.standard_normal, rng, &mut self.output[..len], scale);
        &self.output[..len]
    }

    /// Fills `len` entries with periodic 1D conservative noise $(\xi_{i+1/2} - \xi_{i-1/2}) / \Delta x$.
    ///
    /// Every cell flux $\xi_{i+1/2} \sim \mathcal{N}(0, \text{scale}^2)$ is shared between adjacent cells,
    /// guaranteeing $\sum_{i=0}^{\text{len}-1} (\nabla \cdot \xi)_i = 0$ up to machine precision.
    #[verified_engine::verified]
    pub fn fill_conserved_flux_1d_periodic(
        &mut self,
        rng: &mut OxidizeRng,
        len: usize,
        dx: f64,
        scale: f64,
    ) -> &[f64] {
        assert!(
            len >= 2 && len <= self.output.len() && len <= self.flux_x.len(),
            "Invalid or unallocated 1D periodic conserved-noise length"
        );
        assert!(
            dx.is_finite() && dx > 0.0 && scale.is_finite() && scale >= 0.0,
            "dx must be positive and scale must be non-negative"
        );
        sample_scaled_normals(&self.standard_normal, rng, &mut self.flux_x[..len], scale);
        compute_1d_periodic_divergence(&self.flux_x[..len], &mut self.output[..len], 1.0 / dx);
        remove_residual_mean(&mut self.output[..len]);
        &self.output[..len]
    }

    /// Fills `len` entries with zero-flux Neumann 1D conservative noise ($\xi_{-1/2} = \xi_{N-1/2} = 0$).
    #[verified_engine::verified]
    pub fn fill_conserved_flux_1d_neumann(
        &mut self,
        rng: &mut OxidizeRng,
        len: usize,
        dx: f64,
        scale: f64,
    ) -> &[f64] {
        assert!(
            len >= 2 && len <= self.output.len() && len < self.flux_x.len(),
            "Invalid or unallocated 1D Neumann conserved-noise length"
        );
        assert!(
            dx.is_finite() && dx > 0.0 && scale.is_finite() && scale >= 0.0,
            "dx must be positive and scale must be non-negative"
        );
        self.flux_x[0] = 0.0;
        self.flux_x[len] = 0.0;
        sample_scaled_normals(&self.standard_normal, rng, &mut self.flux_x[1..len], scale);
        let inv_dx = 1.0 / dx;
        for i in 0..len {
            self.output[i] = (self.flux_x[i + 1] - self.flux_x[i]) * inv_dx;
        }
        remove_residual_mean(&mut self.output[..len]);
        &self.output[..len]
    }

    /// Fills `width * height` entries with periodic 2D staggered conservative flux divergence $\nabla_h \cdot \boldsymbol{\xi}$.
    ///
    /// Samples independent horizontal face fluxes $\xi^x_{x+1/2, y}$ and vertical face fluxes $\xi^y_{x, y+1/2}$
    /// from $\mathcal{N}(0, \text{scale}^2)$ and evaluates $(\xi^x_{x+1/2,y} - \xi^x_{x-1/2,y})/\Delta x
    /// + (\xi^y_{x,y+1/2} - \xi^y_{x,y-1/2})/\Delta y$ with zero heap allocations.
    #[verified_engine::verified]
    pub fn fill_conserved_flux_2d_periodic(
        &mut self,
        rng: &mut OxidizeRng,
        width: usize,
        height: usize,
        dx: f64,
        dy: f64,
        scale: f64,
    ) -> &[f64] {
        let total = width * height;
        assert!(
            width >= 2 && height >= 2 && total <= self.output.len(),
            "Invalid or unallocated 2D periodic conserved-noise dimensions"
        );
        assert!(
            total <= self.flux_x.len() && total <= self.flux_y.len(),
            "Staggered flux buffers are smaller than width * height"
        );
        assert!(
            dx.is_finite() && dx > 0.0 && dy.is_finite() && dy > 0.0 && scale >= 0.0,
            "Grid spacings must be positive and scale non-negative"
        );
        sample_scaled_normals(&self.standard_normal, rng, &mut self.flux_x[..total], scale);
        sample_scaled_normals(&self.standard_normal, rng, &mut self.flux_y[..total], scale);
        compute_2d_periodic_divergence(
            &self.flux_x[..total],
            &self.flux_y[..total],
            &mut self.output[..total],
            width,
            height,
            1.0 / dx,
            1.0 / dy,
        );
        remove_residual_mean(&mut self.output[..total]);
        &self.output[..total]
    }
}

fn sample_scaled_normals(normal: &Normal<f64>, rng: &mut OxidizeRng, out: &mut [f64], scale: f64) {
    if scale == 0.0 {
        out.fill(0.0);
        return;
    }
    for slot in out.iter_mut() {
        *slot = normal.sample(rng) * scale;
    }
}

fn compute_1d_periodic_divergence(flux: &[f64], out: &mut [f64], inv_dx: f64) {
    let n = out.len();
    for i in 0..n {
        let prev = if i == 0 { n - 1 } else { i - 1 };
        out[i] = (flux[i] - flux[prev]) * inv_dx;
    }
}

fn compute_2d_periodic_divergence(
    flux_x: &[f64],
    flux_y: &[f64],
    out: &mut [f64],
    width: usize,
    height: usize,
    inv_dx: f64,
    inv_dy: f64,
) {
    for y in 0..height {
        let ym = if y == 0 { height - 1 } else { y - 1 };
        for x in 0..width {
            let xm = if x == 0 { width - 1 } else { x - 1 };
            let idx = flatten_2d_index(x, y, width);
            let idx_xm = flatten_2d_index(xm, y, width);
            let idx_ym = flatten_2d_index(x, ym, width);
            let div_x = (flux_x[idx] - flux_x[idx_xm]) * inv_dx;
            let div_y = (flux_y[idx] - flux_y[idx_ym]) * inv_dy;
            out[idx] = div_x + div_y;
        }
    }
}

fn remove_residual_mean(values: &mut [f64]) {
    if values.is_empty() {
        return;
    }
    let mean = values.iter().copied().sum::<f64>() / (values.len() as f64);
    for v in values.iter_mut() {
        *v -= mean;
    }
}
