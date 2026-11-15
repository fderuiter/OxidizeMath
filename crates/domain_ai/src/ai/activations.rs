// Implementation of various activation functions.
pub(crate) use crate::ai::activation_kernels as kernel;
use nalgebra::DMatrix;

/// Applies the Rectified Linear Unit (ReLU) activation function element-wise, in-place.
/// ReLU(x) = max(0, x)
#[verified_engine::verified]
pub fn relu(matrix: &mut DMatrix<f64>) {
    // The apply method takes a closure that mutates the element in place without allocation.
    matrix.apply(|x| *x = kernel::relu(*x));
}

/// Applies the softmax function to each row of a matrix for numerical stability.
/// Softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
#[verified_engine::verified]
pub fn softmax_row_wise(matrix: &DMatrix<f64>) -> DMatrix<f64> {
    let mut result = DMatrix::zeros(matrix.nrows(), matrix.ncols());
    for r in 0..matrix.nrows() {
        let row = matrix.row(r);
        let max_val = row.max(); // Subtract max for numerical stability
        let exps = row.map(|val| (val - max_val).exp());
        let sum_exps = exps.sum();
        if sum_exps > 0.0 {
            let softmax_row = exps.map(|val| val / sum_exps);
            result.set_row(r, &softmax_row);
        }
        // If sum_exps is 0, the row remains zeros, which is a reasonable default.
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    #[verified_engine::verified]
    fn test_relu() {
        let mut matrix = DMatrix::from_row_slice(1, 4, &[-1.0, 0.0, 1.0, -2.0]);
        relu(&mut matrix);
        let expected = DMatrix::from_row_slice(1, 4, &[0.0, 0.0, 1.0, 0.0]);
        assert_eq!(matrix, expected);
    }

    #[test]
    #[verified_engine::verified]
    fn test_softmax_row_wise_sum() {
        let matrix = DMatrix::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let softmax_result = softmax_row_wise(&matrix);

        // Each row should sum to 1.0
        assert_relative_eq!(
            softmax_result.row(0).sum(),
            1.0,
            epsilon = math_commons::registry::TOLERANCE_FAST
        );
        assert_relative_eq!(
            softmax_result.row(1).sum(),
            1.0,
            epsilon = math_commons::registry::TOLERANCE_FAST
        );
    }

    #[test]
    #[verified_engine::verified]
    fn test_matrix_vector_relu_numerical_equivalence() {
        use crate::ai::deep_learning_theory::calculus;
        use nalgebra::DVector;

        let values = vec![-10.0, -2.5, -0.0, 0.0, 0.001, 1.0, 42.5, -100.0];
        let mut matrix = DMatrix::from_row_slice(2, 4, &values);
        let vector = DVector::from_vec(values);

        relu(&mut matrix);
        let vector_relu = calculus::relu(&vector);

        for r in 0..matrix.nrows() {
            for c in 0..matrix.ncols() {
                let idx = r * matrix.ncols() + c;
                assert_eq!(matrix[(r, c)], vector_relu[idx]);
            }
        }
    }
}
