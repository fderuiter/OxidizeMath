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
    let nrows = matrix.nrows();
    let ncols = matrix.ncols();
    let mut result = DMatrix::zeros(nrows, ncols);

    if nrows == 0 || ncols == 0 {
        return result;
    }

    for r in 0..nrows {
        let mut max_val = matrix[(r, 0)];
        for c in 1..ncols {
            let val = matrix[(r, c)];
            if val > max_val {
                max_val = val;
            }
        }

        let mut sum_exps = 0.0;
        for c in 0..ncols {
            let exp_val = (matrix[(r, c)] - max_val).exp();
            result[(r, c)] = exp_val;
            sum_exps += exp_val;
        }

        if sum_exps > 0.0 {
            for c in 0..ncols {
                result[(r, c)] /= sum_exps;
            }
        } else {
            // When sum_exps equals zero or NaN, preserve an all-zero row output
            for c in 0..ncols {
                result[(r, c)] = 0.0;
            }
        }
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
    fn test_softmax_row_wise_numerical_accuracy() {
        let matrix = DMatrix::from_row_slice(2, 3, &[0.0, 1.0, 2.0, 3.0, 1.0, 2.0]);
        let result = softmax_row_wise(&matrix);

        let e0_0 = (-2.0_f64).exp();
        let e0_1 = (-1.0_f64).exp();
        let e0_2 = 0.0_f64.exp();
        let sum0 = e0_0 + e0_1 + e0_2;

        assert_relative_eq!(
            result[(0, 0)],
            e0_0 / sum0,
            epsilon = math_commons::registry::TOLERANCE_FAST
        );
        assert_relative_eq!(
            result[(0, 1)],
            e0_1 / sum0,
            epsilon = math_commons::registry::TOLERANCE_FAST
        );
        assert_relative_eq!(
            result[(0, 2)],
            e0_2 / sum0,
            epsilon = math_commons::registry::TOLERANCE_FAST
        );
        assert_relative_eq!(
            result.row(0).sum(),
            1.0,
            epsilon = math_commons::registry::TOLERANCE_FAST
        );
        assert_relative_eq!(
            result.row(1).sum(),
            1.0,
            epsilon = math_commons::registry::TOLERANCE_FAST
        );

        // Single element row must normalize to 1.0
        let single = DMatrix::from_row_slice(1, 1, &[42.0]);
        let res_single = softmax_row_wise(&single);
        assert_relative_eq!(
            res_single[(0, 0)],
            1.0,
            epsilon = math_commons::registry::TOLERANCE_FAST
        );
    }

    #[test]
    #[verified_engine::verified]
    fn test_softmax_row_wise_uniform_normalization() {
        // Uniform identical values across rectangular matrices
        let matrix = DMatrix::from_element(3, 4, 3.7);
        let result = softmax_row_wise(&matrix);
        for r in 0..3 {
            for c in 0..4 {
                assert_relative_eq!(
                    result[(r, c)],
                    0.25,
                    epsilon = math_commons::registry::TOLERANCE_FAST
                );
            }
            assert_relative_eq!(
                result.row(r).sum(),
                1.0,
                epsilon = math_commons::registry::TOLERANCE_FAST
            );
        }

        // All-zero rectangular matrix
        let zeros_matrix = DMatrix::zeros(2, 5);
        let res_zeros = softmax_row_wise(&zeros_matrix);
        for r in 0..2 {
            for c in 0..5 {
                assert_relative_eq!(
                    res_zeros[(r, c)],
                    0.2,
                    epsilon = math_commons::registry::TOLERANCE_FAST
                );
            }
            assert_relative_eq!(
                res_zeros.row(r).sum(),
                1.0,
                epsilon = math_commons::registry::TOLERANCE_FAST
            );
        }
    }

    #[test]
    #[verified_engine::verified]
    fn test_softmax_row_wise_large_negative_numbers() {
        let matrix = DMatrix::from_row_slice(2, 2, &[-1000.0, -1000.0, -500.0, -501.0]);
        let result = softmax_row_wise(&matrix);

        assert_relative_eq!(
            result[(0, 0)],
            0.5,
            epsilon = math_commons::registry::TOLERANCE_FAST
        );
        assert_relative_eq!(
            result[(0, 1)],
            0.5,
            epsilon = math_commons::registry::TOLERANCE_FAST
        );
        assert_relative_eq!(
            result.row(0).sum(),
            1.0,
            epsilon = math_commons::registry::TOLERANCE_FAST
        );

        let exp_diff = (-1.0_f64).exp();
        let expected_row1_0 = 1.0 / (1.0 + exp_diff);
        let expected_row1_1 = exp_diff / (1.0 + exp_diff);
        assert_relative_eq!(
            result[(1, 0)],
            expected_row1_0,
            epsilon = math_commons::registry::TOLERANCE_FAST
        );
        assert_relative_eq!(
            result[(1, 1)],
            expected_row1_1,
            epsilon = math_commons::registry::TOLERANCE_FAST
        );
        assert_relative_eq!(
            result.row(1).sum(),
            1.0,
            epsilon = math_commons::registry::TOLERANCE_FAST
        );
    }

    #[test]
    #[verified_engine::verified]
    fn test_softmax_row_wise_zero_sum_and_empty() {
        // Empty matrices
        let empty_matrix = DMatrix::<f64>::zeros(0, 0);
        let res = softmax_row_wise(&empty_matrix);
        assert_eq!(res.nrows(), 0);
        assert_eq!(res.ncols(), 0);

        let empty_cols = DMatrix::<f64>::zeros(3, 0);
        let res_cols = softmax_row_wise(&empty_cols);
        assert_eq!(res_cols.nrows(), 3);
        assert_eq!(res_cols.ncols(), 0);

        // NaN leading to non-positive sum should preserve all-zero row output
        let nan_matrix = DMatrix::from_row_slice(1, 2, &[f64::NAN, 1.0]);
        let res_nan = softmax_row_wise(&nan_matrix);
        assert_eq!(res_nan[(0, 0)], 0.0);
        assert_eq!(res_nan[(0, 1)], 0.0);
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
