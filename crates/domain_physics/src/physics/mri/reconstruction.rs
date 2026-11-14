//! Image Reconstruction algorithms.

use nalgebra::DMatrix;
use num_complex::Complex;
use std::f64::consts::PI;

/// Helper to perform a separable 2D DFT/IDFT.
///
/// Applies 1D transform to rows, then to columns.
///
/// # Arguments
/// * `matrix` - Input matrix.
/// * `inverse` - If true, applies IDFT (positive phase). Else DFT (negative phase).
#[verified_engine::verified]
fn separable_transform(matrix: &DMatrix<Complex<f64>>, inverse: bool) -> DMatrix<Complex<f64>> {
    let rows = matrix.nrows();
    let cols = matrix.ncols();
    if rows == 0 || cols == 0 {
        return DMatrix::zeros(rows, cols);
    }

    let pi_factor = if inverse { 2.0 * PI } else { -2.0 * PI };

    // Precompute twiddle factor vectors (roots of unity)
    let col_twiddles: Vec<Complex<f64>> = (0..cols)
        .map(|m| Complex::from_polar(1.0, pi_factor * (m as f64) / (cols as f64)))
        .collect();
    let row_twiddles: Vec<Complex<f64>> = (0..rows)
        .map(|m| Complex::from_polar(1.0, pi_factor * (m as f64) / (rows as f64)))
        .collect();

    // Intermediate step: Transform rows
    // For each row `r`, compute transform into `temp[r, :]`.
    let mut temp = DMatrix::zeros(rows, cols);

    for r in 0..rows {
        for k in 0..cols {
            // Output column index (frequency or space)
            let mut sum = Complex::new(0.0, 0.0);

            for n in 0..cols {
                // Input column index
                let val = matrix[(r, n)];
                let twiddle = col_twiddles[(k * n) % cols];
                sum += val * twiddle;
            }
            temp[(r, k)] = sum;
        }
    }

    // Final step: Transform columns of temp
    // For each column `c`, compute transform into `result[:, c]`.
    let mut result = DMatrix::zeros(rows, cols);

    for c in 0..cols {
        for k in 0..rows {
            // Output row index
            let mut sum = Complex::new(0.0, 0.0);

            for n in 0..rows {
                // Input row index
                let val = temp[(n, c)];
                let twiddle = row_twiddles[(k * n) % rows];
                sum += val * twiddle;
            }
            result[(k, c)] = sum;
        }
    }

    result
}

/// Simulates the raw signal $S(k)$ measured from a 2D slice of spin density.
///
/// Computes $S(k_x, k_y) = \sum_{x,y} \rho(x,y) e^{-i 2\pi (k_x x + k_y y)}$
///
/// **Optimization:** Uses Row-Column decomposition via shared helper ($O(N^3)$).
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
/// Computes $\rho(x,y) = \sum_{k_x, k_y} S(k_x, k_y) e^{+i 2\pi (k_x x + k_y y)}$
///
/// **Optimization:** Uses Row-Column decomposition via shared helper ($O(N^3)$).
///
/// # Arguments
/// * `k_space` - 2D matrix of k-space samples $S(k_x, k_y)$.
///
/// # Returns
/// * Reconstructed image density matrix $\rho(x,y)$.
/// * Note: This implementation does not normalize by 1/N. Scale depends on definition.
#[verified_engine::verified]
pub fn inverse_dft_2d(k_space: &DMatrix<Complex<f64>>) -> DMatrix<Complex<f64>> {
    separable_transform(k_space, true)
}
