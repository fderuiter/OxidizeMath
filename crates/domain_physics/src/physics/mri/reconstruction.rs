//! Image Reconstruction algorithms and pre-planned zero-allocation 2D FFT workspaces.

use nalgebra::DMatrix;
use num_complex::Complex;
use rustfft::{Fft, FftDirection, FftPlanner};
use std::sync::Arc;

/// Pre-planned 2D Fast Fourier Transform workspace for zero-allocation MRI simulation and reconstruction.
///
/// Caches forward and inverse 1D `rustfft` row and column plans alongside a reusable
/// inplace scratch slice and an intermediate transposed matrix buffer (`cols x rows`).
/// Once constructed for `(rows, cols)`, [`Self::simulate_signal_2d_into`],
/// [`Self::inverse_dft_2d_into`], and [`Self::reconstruct_normalized_2d_into`] execute
/// with zero dynamic heap allocations inside `VerifiedEngine::run_verified`.
///
/// # Example
///
/// ```rust
/// # use domain_physics::__doctest_facade as math_explorer;
/// use math_explorer::nalgebra::DMatrix;
/// use math_explorer::num_complex::Complex;
/// use math_explorer::physics::mri::reconstruction::Fft2dWorkspace;
///
/// let mut ws = Fft2dWorkspace::new(16, 16);
/// let density = DMatrix::from_element(16, 16, Complex::new(1.0, 0.0));
/// let mut k_space = DMatrix::zeros(16, 16);
/// let mut image = DMatrix::zeros(16, 16);
///
/// ws.simulate_signal_2d_into(&density, &mut k_space);
/// ws.reconstruct_normalized_2d_into(&k_space, &mut image);
/// assert!((image[(8, 8)].re - 1.0).abs() < 1e-12);
/// ```
#[derive(Clone)]
pub struct Fft2dWorkspace {
    rows: usize,
    cols: usize,
    fwd_row_fft: Option<Arc<dyn Fft<f64>>>,
    fwd_col_fft: Option<Arc<dyn Fft<f64>>>,
    inv_row_fft: Option<Arc<dyn Fft<f64>>>,
    inv_col_fft: Option<Arc<dyn Fft<f64>>>,
    scratch: Vec<Complex<f64>>,
    transposed: DMatrix<Complex<f64>>,
}

impl std::fmt::Debug for Fft2dWorkspace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Fft2dWorkspace")
            .field("rows", &self.rows)
            .field("cols", &self.cols)
            .field("scratch_len", &self.scratch.len())
            .finish_non_exhaustive()
    }
}

impl Fft2dWorkspace {
    /// Creates a pre-planned 2D FFT workspace for `rows x cols` complex matrices.
    #[must_use]
    pub fn new(rows: usize, cols: usize) -> Self {
        if rows == 0 || cols == 0 {
            return Self {
                rows,
                cols,
                fwd_row_fft: None,
                fwd_col_fft: None,
                inv_row_fft: None,
                inv_col_fft: None,
                scratch: Vec::new(),
                transposed: DMatrix::zeros(cols, rows),
            };
        }

        let mut planner = FftPlanner::new();
        let fwd_row = planner.plan_fft(cols, FftDirection::Forward);
        let fwd_col = planner.plan_fft(rows, FftDirection::Forward);
        let inv_row = planner.plan_fft(cols, FftDirection::Inverse);
        let inv_col = planner.plan_fft(rows, FftDirection::Inverse);

        let scratch_len = fwd_row
            .get_inplace_scratch_len()
            .max(fwd_col.get_inplace_scratch_len())
            .max(inv_row.get_inplace_scratch_len())
            .max(inv_col.get_inplace_scratch_len());

        Self {
            rows,
            cols,
            fwd_row_fft: Some(fwd_row),
            fwd_col_fft: Some(fwd_col),
            inv_row_fft: Some(inv_row),
            inv_col_fft: Some(inv_col),
            scratch: vec![Complex::new(0.0, 0.0); scratch_len],
            transposed: DMatrix::zeros(cols, rows),
        }
    }

    /// Returns the configured row count `R` of this workspace.
    #[must_use]
    #[verified_engine::verified]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the configured column count `C` of this workspace.
    #[must_use]
    #[verified_engine::verified]
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Rebuilds cached FFT plans and scratch buffers when `(rows, cols)` changes.
    pub fn resize(&mut self, rows: usize, cols: usize) {
        if self.rows != rows || self.cols != cols {
            *self = Self::new(rows, cols);
        }
    }

    #[verified_engine::verified]
    fn execute_in_place(
        &mut self,
        input: &DMatrix<Complex<f64>>,
        output: &mut DMatrix<Complex<f64>>,
        inverse: bool,
    ) {
        assert_eq!(
            input.shape(),
            (self.rows, self.cols),
            "Fft2dWorkspace input shape mismatch"
        );
        assert_eq!(
            output.shape(),
            (self.rows, self.cols),
            "Fft2dWorkspace output shape mismatch"
        );
        if self.rows == 0 || self.cols == 0 {
            return;
        }

        let (row_plan, col_plan) = if inverse {
            (self.inv_row_fft.as_ref(), self.inv_col_fft.as_ref())
        } else {
            (self.fwd_row_fft.as_ref(), self.fwd_col_fft.as_ref())
        };

        transpose_blocked(
            input.as_slice(),
            self.transposed.as_mut_slice(),
            self.rows,
            self.cols,
        );
        if let Some(fft_r) = row_plan {
            fft_r.process_with_scratch(self.transposed.as_mut_slice(), &mut self.scratch);
        }
        transpose_blocked(
            self.transposed.as_slice(),
            output.as_mut_slice(),
            self.cols,
            self.rows,
        );
        if let Some(fft_c) = col_plan {
            fft_c.process_with_scratch(output.as_mut_slice(), &mut self.scratch);
        }
    }

    /// Computes the unnormalized 2D forward DFT $S(k_x, k_y)$ into `out_k_space` without heap allocation.
    #[verified_engine::verified]
    pub fn simulate_signal_2d_into(
        &mut self,
        density: &DMatrix<Complex<f64>>,
        out_k_space: &mut DMatrix<Complex<f64>>,
    ) {
        self.execute_in_place(density, out_k_space, false);
    }

    /// Computes the unnormalized 2D inverse DFT into `out_image` without heap allocation.
    #[verified_engine::verified]
    pub fn inverse_dft_2d_into(
        &mut self,
        k_space: &DMatrix<Complex<f64>>,
        out_image: &mut DMatrix<Complex<f64>>,
    ) {
        self.execute_in_place(k_space, out_image, true);
    }

    /// Computes the $1 / (R \cdot C)$-normalized 2D inverse DFT into `out_image` without heap allocation.
    #[verified_engine::verified]
    pub fn reconstruct_normalized_2d_into(
        &mut self,
        k_space: &DMatrix<Complex<f64>>,
        out_image: &mut DMatrix<Complex<f64>>,
    ) {
        self.execute_in_place(k_space, out_image, true);
        let total = self.rows * self.cols;
        if total > 0 {
            scale_complex_slice(out_image.as_mut_slice(), 1.0 / (total as f64));
        }
    }

    /// Executes a fused zero-allocation 2D forward DFT, frequency-domain mask/spike modulation,
    /// and $1/(R \cdot C)$-scaled inverse 2D DFT.
    ///
    /// `dft_mask` and `dft_spikes` match the transposed k-space layout produced after `fwd_col` ->
    /// transpose -> `fwd_row` (i.e., `r * cols + c` for column-major `density`). When `identity_pass`
    /// is `true` (all `dft_mask == 1.0` and `dft_spikes == 0`), computes `out_kspace` via a single
    /// 2D forward DFT and copies `density` into `out_image` directly by Fourier inversion identity.
    #[verified_engine::verified]
    #[allow(clippy::too_many_arguments)]
    pub fn filter_kspace_2d_into(
        &mut self,
        density: &DMatrix<Complex<f64>>,
        dft_mask: &[f64],
        dft_spikes: &[Complex<f64>],
        identity_pass: bool,
        out_kspace: &mut [Complex<f64>],
        out_image: &mut DMatrix<Complex<f64>>,
    ) {
        let total = self.rows * self.cols;
        assert_eq!(density.shape(), (self.rows, self.cols));
        assert_eq!(out_image.shape(), (self.rows, self.cols));
        assert_eq!(dft_mask.len(), total);
        assert_eq!(dft_spikes.len(), total);
        assert_eq!(out_kspace.len(), total);
        if total == 0 {
            return;
        }
        if identity_pass {
            self.forward_transposed_into(density, out_kspace);
            out_image.copy_from(density);
            return;
        }
        out_image.copy_from(density);
        if let Some(ref fwd_c) = self.fwd_col_fft {
            fwd_c.process_with_scratch(out_image.as_mut_slice(), &mut self.scratch);
        }
        transpose_blocked(
            out_image.as_slice(),
            self.transposed.as_mut_slice(),
            self.rows,
            self.cols,
        );
        if let Some(ref fwd_r) = self.fwd_row_fft {
            fwd_r.process_with_scratch(self.transposed.as_mut_slice(), &mut self.scratch);
        }
        apply_kspace_modulation(
            self.transposed.as_mut_slice(),
            dft_mask,
            dft_spikes,
            out_kspace,
        );
        if let Some(ref inv_r) = self.inv_row_fft {
            inv_r.process_with_scratch(self.transposed.as_mut_slice(), &mut self.scratch);
        }
        transpose_blocked(
            self.transposed.as_slice(),
            out_image.as_mut_slice(),
            self.cols,
            self.rows,
        );
        if let Some(ref inv_c) = self.inv_col_fft {
            inv_c.process_with_scratch(out_image.as_mut_slice(), &mut self.scratch);
        }
        scale_complex_slice(out_image.as_mut_slice(), 1.0 / (total as f64));
    }

    /// Transforms `density` in-place along columns, transposes directly into `out_kspace`
    /// (`cols x rows` layout), and transforms rows in-place with zero extra buffer copies.
    #[verified_engine::verified]
    pub fn forward_transposed_in_place(
        &mut self,
        density: &mut DMatrix<Complex<f64>>,
        out_kspace: &mut [Complex<f64>],
    ) {
        let total = self.rows * self.cols;
        assert_eq!(density.shape(), (self.rows, self.cols));
        assert_eq!(out_kspace.len(), total);
        if total == 0 {
            return;
        }
        if let Some(ref fwd_c) = self.fwd_col_fft {
            fwd_c.process_with_scratch(density.as_mut_slice(), &mut self.scratch);
        }
        transpose_blocked(density.as_slice(), out_kspace, self.rows, self.cols);
        if let Some(ref fwd_r) = self.fwd_row_fft {
            fwd_r.process_with_scratch(out_kspace, &mut self.scratch);
        }
    }

    fn forward_transposed_into(
        &mut self,
        density: &DMatrix<Complex<f64>>,
        out_kspace: &mut [Complex<f64>],
    ) {
        self.transposed
            .as_mut_slice()
            .copy_from_slice(density.as_slice());
        if let Some(ref fwd_c) = self.fwd_col_fft {
            fwd_c.process_with_scratch(self.transposed.as_mut_slice(), &mut self.scratch);
        }
        transpose_blocked(self.transposed.as_slice(), out_kspace, self.rows, self.cols);
        if let Some(ref fwd_r) = self.fwd_row_fft {
            fwd_r.process_with_scratch(out_kspace, &mut self.scratch);
        }
    }
}

fn apply_kspace_modulation(
    buf: &mut [Complex<f64>],
    mask: &[f64],
    spikes: &[Complex<f64>],
    out_kspace: &mut [Complex<f64>],
) {
    for (((b, &m), &sp), out) in buf.iter_mut().zip(mask).zip(spikes).zip(out_kspace) {
        let v = *b * m + sp;
        *b = v;
        *out = v;
    }
}

fn scale_complex_slice(slice: &mut [Complex<f64>], scale: f64) {
    for elem in slice {
        *elem *= scale;
    }
}

/// Cache-blocked `8x8` transpose of a column-major `rows x cols` slice into a column-major `cols x rows` slice.
fn transpose_blocked(src: &[Complex<f64>], dst: &mut [Complex<f64>], rows: usize, cols: usize) {
    if rows == 0 || cols == 0 {
        return;
    }
    const TILE: usize = 8;
    let mut r0 = 0;
    for dst_band in dst.chunks_mut(cols * TILE) {
        let r_len = dst_band.len() / cols;
        for (c, src_col) in src.chunks_exact(rows).enumerate() {
            for (&val, dst_row) in src_col[r0..r0 + r_len]
                .iter()
                .zip(dst_band.chunks_exact_mut(cols))
            {
                dst_row[c] = val;
            }
        }
        r0 += r_len;
    }
}

fn shift_2d_columns(
    src: &DMatrix<Complex<f64>>,
    dst: &mut DMatrix<Complex<f64>>,
    shift_r: usize,
    shift_c: usize,
) {
    let rows = src.nrows();
    let cols = src.ncols();
    let split_r = rows - shift_r;
    let src_slice = src.as_slice();
    let dst_slice = dst.as_mut_slice();
    for c in 0..cols {
        let dc = (c + shift_c) % cols;
        let src_col = &src_slice[c * rows..(c + 1) * rows];
        let dst_col = &mut dst_slice[dc * rows..(dc + 1) * rows];
        dst_col[shift_r..].copy_from_slice(&src_col[..split_r]);
        dst_col[..shift_r].copy_from_slice(&src_col[split_r..]);
    }
}

/// Shifts zero-frequency (DC) k-space component `(0, 0)` to grid center `(R/2, C/2)` without allocation.
#[verified_engine::verified]
pub fn fftshift_2d_into(src: &DMatrix<Complex<f64>>, dst: &mut DMatrix<Complex<f64>>) {
    let rows = src.nrows();
    let cols = src.ncols();
    assert_eq!(dst.shape(), (rows, cols), "fftshift_2d_into shape mismatch");
    if rows == 0 || cols == 0 {
        return;
    }
    shift_2d_columns(src, dst, rows / 2, cols / 2);
}

/// Inverse-shifts centered k-space `(R/2, C/2)` back to standard DFT layout `(0, 0)` without allocation.
#[verified_engine::verified]
pub fn ifftshift_2d_into(src: &DMatrix<Complex<f64>>, dst: &mut DMatrix<Complex<f64>>) {
    let rows = src.nrows();
    let cols = src.ncols();
    assert_eq!(
        dst.shape(),
        (rows, cols),
        "ifftshift_2d_into shape mismatch"
    );
    if rows == 0 || cols == 0 {
        return;
    }
    shift_2d_columns(src, dst, rows - (rows / 2), cols - (cols / 2));
}

#[verified_engine::verified]
fn separable_transform(matrix: &DMatrix<Complex<f64>>, inverse: bool) -> DMatrix<Complex<f64>> {
    let rows = matrix.nrows();
    let cols = matrix.ncols();
    let mut out = DMatrix::zeros(rows, cols);
    if rows == 0 || cols == 0 {
        return out;
    }
    let mut workspace = Fft2dWorkspace::new(rows, cols);
    workspace.execute_in_place(matrix, &mut out, inverse);
    out
}

/// Simulates the raw signal $S(k)$ measured from a 2D slice of spin density.
///
/// Computes $S(k_x, k_y) = \sum_{x,y} \rho(x,y) e^{-i 2\pi (k_x x / R + k_y y / C)}$
///
/// **Optimization:** Uses transposed unit-stride 1D FFT passes ($O(R \cdot C \log(R \cdot C))$).
///
/// # Arguments
/// * `density` - 2D matrix representing the spin density $\rho(x,y)$.
///
/// # Returns
/// * 2D matrix of k-space samples (raw signal), same dimensions as density.
#[verified_engine::verified]
pub fn simulate_signal_2d(density: &DMatrix<Complex<f64>>) -> DMatrix<Complex<f64>> {
    separable_transform(density, false)
}

/// Performs a 2D Inverse Discrete Fourier Transform (IDFT) to reconstruct the image.
///
/// Computes $\rho(x,y) = \sum_{k_x, k_y} S(k_x, k_y) e^{+i 2\pi (k_x x / R + k_y y / C)}$
///
/// **Optimization:** Uses transposed unit-stride 1D IFFT passes ($O(R \cdot C \log(R \cdot C))$).
///
/// # Arguments
/// * `k_space` - 2D matrix of k-space samples $S(k_x, k_y)$.
///
/// # Returns
/// * Reconstructed image density matrix $\rho(x,y)$.
/// * Note: This implementation does not normalize by $1/(R \cdot C)$. Scale depends on definition.
#[verified_engine::verified]
pub fn inverse_dft_2d(k_space: &DMatrix<Complex<f64>>) -> DMatrix<Complex<f64>> {
    separable_transform(k_space, true)
}
