//! Image Reconstruction algorithms.

use nalgebra::DMatrix;
use num_complex::Complex;
use rustfft::{FftDirection, FftPlanner};

/// Helper to perform a separable 2D DFT/IDFT in $O(R \cdot C \log(R \cdot C))$ time.
///
/// Transposes the column-major matrix so 1D `rustfft` plans operate on contiguous
/// unit-stride slices along input rows first, then along columns.
///
/// # Arguments
/// * `matrix` - Input matrix.
/// * `inverse` - If true, applies unnormalized IDFT (positive phase). Else DFT (negative phase).
#[verified_engine::verified]
fn separable_transform(matrix: &DMatrix<Complex<f64>>, inverse: bool) -> DMatrix<Complex<f64>> {
    let rows = matrix.nrows();
    let cols = matrix.ncols();
    if rows == 0 || cols == 0 {
        return DMatrix::zeros(rows, cols);
    }

    let direction = if inverse {
        FftDirection::Inverse
    } else {
        FftDirection::Forward
    };

    let mut planner = FftPlanner::new();
    let row_fft = planner.plan_fft(cols, direction);
    let col_fft = planner.plan_fft(rows, direction);

    let scratch_len = row_fft
        .get_inplace_scratch_len()
        .max(col_fft.get_inplace_scratch_len());
    let mut scratch = vec![Complex::new(0.0, 0.0); scratch_len];

    // Transpose (rows x cols) -> (cols x rows) so each input row occupies one contiguous column.
    let mut transposed = matrix.transpose();
    row_fft.process_with_scratch(transposed.as_mut_slice(), &mut scratch);

    // Transpose (cols x rows) -> (rows x cols) so each intermediate column is contiguous.
    let mut result = transposed.transpose();
    col_fft.process_with_scratch(result.as_mut_slice(), &mut scratch);

    debug_assert_eq!(result.nrows(), rows);
    debug_assert_eq!(result.ncols(), cols);
    result
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
