use crate::error::EpidemiologyError;
use nalgebra::DMatrix;

/// Calculates the Spectral Radius (R0) from Transmission (F) and Transition (V) matrices.
///
/// $K = F \cdot V^{-1}$
///
/// # Returns
/// * `Ok(f64)`: The spectral radius (R0).
/// * `Err(EpidemiologyError)`: `EpidemiologyError::Math` if dimensions mismatch or `EpidemiologyError::SingularTransitionMatrix` if V is singular.
#[verified_engine::verified]
pub fn calculate_r0_matrix(
    f_mat: &DMatrix<f64>,
    v_mat: &DMatrix<f64>,
) -> Result<f64, EpidemiologyError> {
    if f_mat.nrows() != v_mat.nrows() || f_mat.ncols() != v_mat.ncols() {
        return Err(EpidemiologyError::Math(
            math_commons::error::MathError::DimensionMismatch {
                expected: math_commons::math_kernel::types::Dimension(f_mat.len()),
                actual: math_commons::math_kernel::types::Dimension(v_mat.len()),
            },
        ));
    }

    let v_inv = v_mat
        .clone()
        .try_inverse()
        .ok_or(EpidemiologyError::SingularTransitionMatrix)?;
    let k = f_mat * v_inv;

    let eigenvalues = k.complex_eigenvalues();

    let spectral_radius = eigenvalues.iter().map(|c| c.norm()).fold(0.0, f64::max);

    Ok(spectral_radius)
}

#[cfg(test)]
mod tests {
    use super::*;
    // We assume this module is part of the epidemiology module which also has compartmental.
    use super::super::compartmental;

    #[test]
    #[verified_engine::verified]
    fn test_matrix_r0_scalar_equivalence() {
        // 1x1 Matrix case should match scalar calculation
        let beta = 2.0;
        let gamma = 1.0;

        let f = DMatrix::from_vec(1, 1, vec![beta]);
        let v = DMatrix::from_vec(1, 1, vec![gamma]);

        let r0_matrix = calculate_r0_matrix(&f, &v).unwrap();
        let r0_scalar = compartmental::basic_reproduction_number(beta, gamma);

        assert!((r0_matrix - r0_scalar).abs() < math_commons::registry::TOLERANCE_FAST);
    }
}
