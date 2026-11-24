//! Pre-planned 2D `rustfft` semi-implicit Laplacian and biharmonic solver with radial $S(k, t)$ telemetry.
//!
//! Uses exact 5-point periodic finite-difference Fourier symbols
//! $q^2(k_x, k_y) = \frac{4\sin^2(\pi k_x / W)}{\Delta x^2} + \frac{4\sin^2(\pi k_y / H)}{\Delta y^2}$
//! and $q^4 = (q^2)^2$ so Eyre convex-splitting updates match discrete Ginzburg-Landau free-energy
//! variations to machine precision while executing with zero dynamic heap allocations.

// [cite:spde]
// [cite:stochastic_pde]

use std::sync::Arc;

use math_commons::math_kernel::types::flatten_2d_index;
use num_complex::Complex;
use rustfft::{Fft, FftPlanner};

use super::spectral_tables::{
    accumulate_radial_bins, fill_spectral_tables_in_place, precompute_spectral_tables,
};

/// Semi-implicit step coefficients for non-conserved reaction-diffusion-biharmonic SPDEs:
/// $(1 + \Delta t (\alpha_0 + \alpha_2 q^2 + \alpha_4 q^4)) \hat{u}^{n+1} = \hat{u}^n + \Delta t \hat{g} + \widehat{\delta W}$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReactionBiharmonicParams {
    /// Time step $\Delta t > 0$.
    pub dt: f64,
    /// Implicit zero-order linear damping weight $\alpha_0 \ge 0$ (e.g., Eyre stabilizer $M \cdot S$).
    pub alpha_0: f64,
    /// Implicit negative-Laplacian weight $\alpha_2 \ge 0$ multiplying $q^2$.
    pub alpha_2: f64,
    /// Implicit biharmonic weight $\alpha_4 \ge 0$ multiplying $q^4$.
    pub alpha_4: f64,
}

/// Semi-implicit Eyre convex-splitting coefficients for conserved Cahn-Hilliard SPDEs:
/// $(1 + \Delta t M (S q^2 + \kappa q^4)) \hat{\phi}^{n+1} = \hat{\phi}^n - \Delta t M q^2 \widehat{\mu_{\text{nl}}} + \widehat{\delta W_{\text{cons}}}$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConservedBiharmonicParams {
    /// Time step $\Delta t > 0$.
    pub dt: f64,
    /// Diffusive mobility $M > 0$.
    pub mobility: f64,
    /// Eyre convex-splitting linear stabilizer $S \ge 0$.
    pub stabilizer_s: f64,
    /// Interfacial gradient-energy coefficient $\kappa \ge 0$.
    pub kappa: f64,
}

/// Summary metrics extracted from the circularly averaged radial structure factor $S(k, t)$.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct RadialSpectrumSummary {
    /// Number of populated radial wavenumber bins written to output slices.
    pub num_bins: usize,
    /// Wavenumber $k_{\max}(t)$ at which $S(k, t)$ attains its maximum over $k > 0$.
    pub k_max: f64,
    /// Peak structure-factor value $S(k_{\max}, t)$.
    pub s_max: f64,
    /// First spectral moment $k_1(t) = \sum_{k>0} k S(k, t) / \sum_{k>0} S(k, t)$ tracking domain coarsening.
    pub k_first_moment: f64,
}

/// Pre-planned 2D semi-implicit spectral PDE/SPDE stepper with zero post-initialization allocations.
#[derive(Clone)]
pub struct SemiImplicitSpectral2D {
    width: usize,
    height: usize,
    dx: f64,
    dy: f64,
    row_fft_fwd: Arc<dyn Fft<f64>>,
    col_fft_fwd: Arc<dyn Fft<f64>>,
    row_fft_inv: Arc<dyn Fft<f64>>,
    col_fft_inv: Arc<dyn Fft<f64>>,
    spatial_a: Vec<Complex<f64>>,
    transposed_a: Vec<Complex<f64>>,
    spatial_b: Vec<Complex<f64>>,
    transposed_b: Vec<Complex<f64>>,
    scratch: Vec<Complex<f64>>,
    /// Discrete negative-Laplacian symbol $q^2(k_y, k_x)$ in transposed `(ky, kx)` layout.
    q_sq: Vec<f64>,
    /// Discrete biharmonic symbol $q^4(k_y, k_x) = (q^2)^2$ in transposed `(ky, kx)` layout.
    q_fourth: Vec<f64>,
    /// Precomputed radial bin index for each `(ky, kx)` entry.
    radial_bin_index: Vec<usize>,
    /// Physical radial wavenumber centers $k_b$ for each radial shell bin.
    radial_bin_k: Vec<f64>,
}

impl std::fmt::Debug for SemiImplicitSpectral2D {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SemiImplicitSpectral2D")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("dx", &self.dx)
            .field("dy", &self.dy)
            .finish()
    }
}

impl SemiImplicitSpectral2D {
    /// Plans forward/inverse 1D `rustfft` passes and precomputes 2D discrete spectral operators.
    pub fn new(width: usize, height: usize, dx: f64, dy: f64) -> Self {
        assert!(width >= 2 && height >= 2, "2D spectral grid must be >= 2x2");
        assert!(
            dx.is_finite() && dx > 0.0 && dy.is_finite() && dy > 0.0,
            "Grid spacings dx and dy must be finite and positive"
        );
        let mut planner = FftPlanner::new();
        let row_fft_fwd = planner.plan_fft_forward(width);
        let col_fft_fwd = planner.plan_fft_forward(height);
        let row_fft_inv = planner.plan_fft_inverse(width);
        let col_fft_inv = planner.plan_fft_inverse(height);
        let scratch_len = [
            row_fft_fwd.get_inplace_scratch_len(),
            col_fft_fwd.get_inplace_scratch_len(),
            row_fft_inv.get_inplace_scratch_len(),
            col_fft_inv.get_inplace_scratch_len(),
            1,
        ]
        .into_iter()
        .max()
        .unwrap_or(1);

        let total = width * height;
        let (q_sq, q_fourth, radial_bin_index, radial_bin_k) =
            precompute_spectral_tables(width, height, dx, dy);

        Self {
            width,
            height,
            dx,
            dy,
            row_fft_fwd,
            col_fft_fwd,
            row_fft_inv,
            col_fft_inv,
            spatial_a: vec![Complex::new(0.0, 0.0); total],
            transposed_a: vec![Complex::new(0.0, 0.0); total],
            spatial_b: vec![Complex::new(0.0, 0.0); total],
            transposed_b: vec![Complex::new(0.0, 0.0); total],
            scratch: vec![Complex::new(0.0, 0.0); scratch_len],
            q_sq,
            q_fourth,
            radial_bin_index,
            radial_bin_k,
        }
    }

    /// Grid width $W$.
    #[verified_engine::verified]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Grid height $H$.
    #[verified_engine::verified]
    pub fn height(&self) -> usize {
        self.height
    }

    /// Grid spacing $\Delta x$.
    #[verified_engine::verified]
    pub fn dx(&self) -> f64 {
        self.dx
    }

    /// Grid spacing $\Delta y$.
    #[verified_engine::verified]
    pub fn dy(&self) -> f64 {
        self.dy
    }

    /// Number of radial wavenumber shell bins up to the Nyquist cutoff.
    #[verified_engine::verified]
    pub fn radial_bin_count(&self) -> usize {
        self.radial_bin_k.len()
    }

    /// Updates `(dx, dy)` spectral operator tables in place without heap allocation.
    #[verified_engine::verified]
    pub fn set_grid_spacing(&mut self, dx: f64, dy: f64) {
        assert!(
            dx.is_finite() && dx > 0.0 && dy.is_finite() && dy > 0.0,
            "Grid spacings dx and dy must be finite and positive"
        );
        self.dx = dx;
        self.dy = dy;
        fill_spectral_tables_in_place(
            self.width,
            self.height,
            dx,
            dy,
            &mut self.q_sq,
            &mut self.q_fourth,
            &mut self.radial_bin_index,
            &mut self.radial_bin_k,
        );
    }

    /// Advances a non-conserved 2D field $u$ by one semi-implicit reaction-diffusion-biharmonic step.
    #[verified_engine::verified]
    pub fn step_reaction_biharmonic(
        &mut self,
        u: &mut [f64],
        explicit_rhs: &[f64],
        noise_increment: Option<&[f64]>,
        params: ReactionBiharmonicParams,
    ) {
        let total = self.width * self.height;
        assert_eq!(u.len(), total, "State length must equal width * height");
        assert_eq!(
            explicit_rhs.len(),
            total,
            "RHS length must equal width * height"
        );
        assert!(
            params.dt.is_finite()
                && params.dt > 0.0
                && params.alpha_0 >= 0.0
                && params.alpha_2 >= 0.0
                && params.alpha_4 >= 0.0,
            "ReactionBiharmonicParams must have dt > 0 and non-negative implicit weights"
        );
        if let Some(noise) = noise_increment {
            assert_eq!(noise.len(), total, "Noise length must equal width * height");
            for i in 0..total {
                let rhs_val = u[i] + params.dt * explicit_rhs[i] + noise[i];
                self.spatial_a[i] = Complex::new(rhs_val, 0.0);
            }
        } else {
            for i in 0..total {
                let rhs_val = u[i] + params.dt * explicit_rhs[i];
                self.spatial_a[i] = Complex::new(rhs_val, 0.0);
            }
        }

        forward_2d_fft(
            self.width,
            self.height,
            &self.row_fft_fwd,
            &self.col_fft_fwd,
            &mut self.spatial_a,
            &mut self.transposed_a,
            &mut self.scratch,
        );
        apply_reaction_symbol(&mut self.transposed_a, &self.q_sq, &self.q_fourth, params);
        inverse_2d_fft_to_real(
            self.width,
            self.height,
            &self.row_fft_inv,
            &self.col_fft_inv,
            &mut self.transposed_a,
            &mut self.spatial_a,
            &mut self.scratch,
            u,
        );
    }

    /// Advances a conserved 2D order parameter $\phi$ by one Eyre convex-splitting Cahn-Hilliard step.
    #[verified_engine::verified]
    pub fn step_conserved_biharmonic(
        &mut self,
        phi: &mut [f64],
        chem_potential_nonlinear: &[f64],
        conserved_noise_increment: Option<&[f64]>,
        params: ConservedBiharmonicParams,
    ) {
        let total = self.width * self.height;
        assert_eq!(phi.len(), total, "phi length must equal width * height");
        assert_eq!(
            chem_potential_nonlinear.len(),
            total,
            "Nonlinear chemical potential length must equal width * height"
        );
        assert!(
            params.dt.is_finite()
                && params.dt > 0.0
                && params.mobility >= 0.0
                && params.stabilizer_s >= 0.0
                && params.kappa >= 0.0,
            "ConservedBiharmonicParams must have dt > 0 and non-negative mobility/stabilizer/kappa"
        );
        load_real_plus_optional(phi, conserved_noise_increment, &mut self.spatial_a);
        load_real_plus_optional(chem_potential_nonlinear, None, &mut self.spatial_b);

        forward_2d_fft(
            self.width,
            self.height,
            &self.row_fft_fwd,
            &self.col_fft_fwd,
            &mut self.spatial_a,
            &mut self.transposed_a,
            &mut self.scratch,
        );
        forward_2d_fft(
            self.width,
            self.height,
            &self.row_fft_fwd,
            &self.col_fft_fwd,
            &mut self.spatial_b,
            &mut self.transposed_b,
            &mut self.scratch,
        );
        apply_conserved_symbol(
            &mut self.transposed_a,
            &self.transposed_b,
            &self.q_sq,
            &self.q_fourth,
            params,
        );
        inverse_2d_fft_to_real(
            self.width,
            self.height,
            &self.row_fft_inv,
            &self.col_fft_inv,
            &mut self.transposed_a,
            &mut self.spatial_a,
            &mut self.scratch,
            phi,
        );
    }

    /// Computes circularly averaged structure factor $S(k, t) = \frac{1}{WH}\langle |\hat{\phi}(\mathbf{k})|^2 \rangle$
    /// into pre-allocated slices without heap allocation.
    #[verified_engine::verified]
    pub fn compute_radial_structure_factor(
        &mut self,
        field: &[f64],
        k_bins_out: &mut [f64],
        s_k_out: &mut [f64],
        bin_counts_scratch: &mut [usize],
    ) -> RadialSpectrumSummary {
        let total = self.width * self.height;
        let n_bins = self
            .radial_bin_k
            .len()
            .min(k_bins_out.len())
            .min(s_k_out.len())
            .min(bin_counts_scratch.len());
        assert_eq!(field.len(), total, "Field length must equal width * height");
        assert!(
            n_bins >= 2,
            "Output radial spectrum slices must hold >= 2 bins"
        );

        let mean = field.iter().copied().sum::<f64>() / (total as f64);
        for (dst, &v) in self.spatial_a.iter_mut().zip(field.iter()) {
            *dst = Complex::new(v - mean, 0.0);
        }
        forward_2d_fft(
            self.width,
            self.height,
            &self.row_fft_fwd,
            &self.col_fft_fwd,
            &mut self.spatial_a,
            &mut self.transposed_a,
            &mut self.scratch,
        );
        accumulate_radial_bins(
            &self.transposed_a,
            &self.radial_bin_index,
            &self.radial_bin_k[..n_bins],
            total as f64,
            &mut k_bins_out[..n_bins],
            &mut s_k_out[..n_bins],
            &mut bin_counts_scratch[..n_bins],
        )
    }
}

fn load_real_plus_optional(base: &[f64], extra: Option<&[f64]>, dst: &mut [Complex<f64>]) {
    if let Some(inc) = extra {
        assert_eq!(inc.len(), base.len(), "Noise slice length mismatch");
        for i in 0..base.len() {
            dst[i] = Complex::new(base[i] + inc[i], 0.0);
        }
    } else {
        for i in 0..base.len() {
            dst[i] = Complex::new(base[i], 0.0);
        }
    }
}

fn forward_2d_fft(
    width: usize,
    height: usize,
    row_fft: &Arc<dyn Fft<f64>>,
    col_fft: &Arc<dyn Fft<f64>>,
    spatial: &mut [Complex<f64>],
    transposed: &mut [Complex<f64>],
    scratch: &mut [Complex<f64>],
) {
    row_fft.process_with_scratch(spatial, scratch);
    for y in 0..height {
        for x in 0..width {
            let src = flatten_2d_index(x, y, width);
            let dst = flatten_2d_index(y, x, height);
            transposed[dst] = spatial[src];
        }
    }
    col_fft.process_with_scratch(transposed, scratch);
}

#[allow(clippy::too_many_arguments)]
fn inverse_2d_fft_to_real(
    width: usize,
    height: usize,
    row_fft_inv: &Arc<dyn Fft<f64>>,
    col_fft_inv: &Arc<dyn Fft<f64>>,
    transposed: &mut [Complex<f64>],
    spatial: &mut [Complex<f64>],
    scratch: &mut [Complex<f64>],
    out_real: &mut [f64],
) {
    col_fft_inv.process_with_scratch(transposed, scratch);
    for x in 0..width {
        for y in 0..height {
            let src = flatten_2d_index(y, x, height);
            let dst = flatten_2d_index(x, y, width);
            spatial[dst] = transposed[src];
        }
    }
    row_fft_inv.process_with_scratch(spatial, scratch);
    let inv_norm = 1.0 / ((width * height) as f64);
    for (dst, val) in out_real.iter_mut().zip(spatial.iter()) {
        *dst = val.re * inv_norm;
    }
}

fn apply_reaction_symbol(
    spec: &mut [Complex<f64>],
    q_sq: &[f64],
    q_fourth: &[f64],
    params: ReactionBiharmonicParams,
) {
    for i in 0..spec.len() {
        let denom = 1.0
            + params.dt
                * (params.alpha_0 + params.alpha_2 * q_sq[i] + params.alpha_4 * q_fourth[i]);
        spec[i] /= denom;
    }
}

fn apply_conserved_symbol(
    phi_hat: &mut [Complex<f64>],
    mu_nl_hat: &[Complex<f64>],
    q_sq: &[f64],
    q_fourth: &[f64],
    params: ConservedBiharmonicParams,
) {
    let dt_m = params.dt * params.mobility;
    for i in 0..phi_hat.len() {
        let q2 = q_sq[i];
        let denom = 1.0 + dt_m * (params.stabilizer_s * q2 + params.kappa * q_fourth[i]);
        phi_hat[i] = (phi_hat[i] - mu_nl_hat[i] * (dt_m * q2)) / denom;
    }
}
