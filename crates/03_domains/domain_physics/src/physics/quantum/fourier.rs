use crate::physics::quantum::types::QuantumOperator;
use nalgebra::DMatrix;
use num_complex::Complex;
use std::f64::consts::PI;

/// Creates a Discrete Fourier Transform (DFT) operator.
///
/// This operator transforms a state vector from the position basis to the momentum basis
/// (for a discrete system).
///
/// F_{jk} = \frac{1}{\sqrt{N}} e^{-i 2\pi j k / N}
///
/// # Arguments
/// * `n` - The dimension of the Hilbert space.
#[verified_engine::verified]
pub fn dft_operator(n: usize) -> QuantumOperator {
    let dim = n as f64;
    let normalization = 1.0 / dim.sqrt();
    let i = Complex::new(0.0, 1.0);

    let mut matrix = DMatrix::from_element(n, n, Complex::new(0.0, 0.0));

    for row in 0..n {
        for col in 0..n {
            let exponent = -i * 2.0 * PI * (row as f64) * (col as f64) / dim;
            matrix[(row, col)] = normalization * exponent.exp();
        }
    }

    QuantumOperator::new(matrix)
}

/// Creates the Inverse Discrete Fourier Transform (IDFT) operator.
///
/// This operator transforms a state vector from the momentum basis to the position basis.
#[verified_engine::verified]
pub fn idft_operator(n: usize) -> QuantumOperator {
    let dft = dft_operator(n);
    // Inverse DFT is the adjoint (conjugate transpose) of DFT matrix, as DFT is unitary.
    QuantumOperator::new(dft.matrix.adjoint())
}
