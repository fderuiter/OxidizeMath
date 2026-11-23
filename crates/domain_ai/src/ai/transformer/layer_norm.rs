use crate::ai::transformer::traits::NormalizationLayer;
use nalgebra::{DMatrix, RowDVector};

/// Layer Normalization.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerNorm {
    /// Small constant for numerical stability.
    epsilon: f64,
    /// Learnable scale parameter.
    gamma: RowDVector<f64>,
    /// Learnable shift parameter.
    beta: RowDVector<f64>,
}

impl LayerNorm {
    /// Creates a new `LayerNorm` instance.
    ///
    /// # Arguments
    ///
    /// * `d_model`: The dimension of the model.
    ///
    /// # Returns
    ///
    /// A new `LayerNorm` instance.
    #[verified_engine::verified]
    pub fn new(d_model: usize) -> Self {
        Self {
            epsilon: math_commons::registry::TOLERANCE_FAST,
            gamma: RowDVector::from_element(d_model, 1.0),
            beta: RowDVector::from_element(d_model, 0.0),
        }
    }

    /// Applies layer normalization to the input.
    #[verified_engine::verified]
    pub fn forward_impl(&self, x: &DMatrix<f64>) -> DMatrix<f64> {
        if x.nrows() == 0 || x.ncols() == 0 {
            return DMatrix::zeros(x.nrows(), x.ncols());
        }

        let mut output = DMatrix::zeros(x.nrows(), x.ncols());
        for r in 0..x.nrows() {
            let row = x.row(r);
            let mean = row.mean();
            let variance = row.variance();
            let inv_std = 1.0 / (variance + self.epsilon).sqrt();

            for c in 0..x.ncols() {
                output[(r, c)] = (x[(r, c)] - mean) * inv_std * self.gamma[c] + self.beta[c];
            }
        }
        output
    }

    /// Applies layer normalization to the input.
    #[verified_engine::verified]
    pub fn forward(&self, x: &DMatrix<f64>) -> DMatrix<f64> {
        self.forward_impl(x)
    }
}

impl NormalizationLayer for LayerNorm {
    #[verified_engine::verified]
    fn forward(&self, x: &DMatrix<f64>) -> DMatrix<f64> {
        self.forward_impl(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    #[verified_engine::verified]
    fn test_layer_norm() {
        let d_model = 4;
        let layer_norm = LayerNorm::new(d_model);
        let mut input = DMatrix::from_row_slice(1, 4, &[1.0, 2.0, 3.0, 4.0]);
        input.apply(|x| *x *= 10.0); // Scale up to make mean/variance more meaningful

        let output = layer_norm.forward(&input);

        // After normalization (before gamma/beta), mean should be ~0 and std dev ~1.
        // Since gamma=1 and beta=0 by default, the output should be normalized.
        let output_row = output.row(0);
        assert_relative_eq!(
            output_row.mean(),
            0.0,
            epsilon = math_commons::registry::TOLERANCE_FAST
        );
        assert_relative_eq!(
            output_row.variance().sqrt(),
            1.0,
            epsilon = math_commons::registry::TOLERANCE_FAST
        );
    }

    #[test]
    #[verified_engine::verified]
    fn test_layer_norm_empty() {
        let layer_norm = LayerNorm::new(4);

        let empty_rows = DMatrix::<f64>::zeros(0, 4);
        let out_empty_rows = layer_norm.forward(&empty_rows);
        assert_eq!(out_empty_rows.nrows(), 0);
        assert_eq!(out_empty_rows.ncols(), 4);

        let empty_cols = DMatrix::<f64>::zeros(3, 0);
        let out_empty_cols = layer_norm.forward(&empty_cols);
        assert_eq!(out_empty_cols.nrows(), 3);
        assert_eq!(out_empty_cols.ncols(), 0);

        let empty_both = DMatrix::<f64>::zeros(0, 0);
        let out_empty_both = layer_norm.forward(&empty_both);
        assert_eq!(out_empty_both.nrows(), 0);
        assert_eq!(out_empty_both.ncols(), 0);
    }

    #[test]
    #[verified_engine::verified]
    fn test_layer_norm_multi_row() {
        let d_model = 3;
        let layer_norm = LayerNorm::new(d_model);
        let input = DMatrix::from_row_slice(2, 3, &[1.0, 5.0, 9.0, -2.0, 0.0, 2.0]);

        let output = layer_norm.forward(&input);
        assert_eq!(output.nrows(), 2);
        assert_eq!(output.ncols(), 3);

        for r in 0..2 {
            let row = output.row(r);
            assert_relative_eq!(
                row.mean(),
                0.0,
                epsilon = math_commons::registry::TOLERANCE_FAST
            );
            assert_relative_eq!(
                row.variance().sqrt(),
                1.0,
                epsilon = math_commons::registry::TOLERANCE_FAST
            );
        }
    }

    #[test]
    #[verified_engine::verified]
    fn test_layer_norm_zero_variance_stability() {
        let d_model = 4;
        let layer_norm = LayerNorm::new(d_model);
        // Constant row has 0 variance
        let input = DMatrix::from_row_slice(1, 4, &[7.0, 7.0, 7.0, 7.0]);
        let output = layer_norm.forward(&input);

        // Since x_i - mean = 0, output elements should be 0.0 without NaN
        for c in 0..4 {
            assert_eq!(output[(0, c)], 0.0);
        }
    }

    #[test]
    #[verified_engine::verified]
    fn test_layer_norm_custom_gamma_beta() {
        let ln = LayerNorm {
            epsilon: 1e-5,
            gamma: RowDVector::from_row_slice(&[2.0, 0.5, 3.0]),
            beta: RowDVector::from_row_slice(&[1.0, -1.0, 2.0]),
        };
        let input = DMatrix::from_row_slice(1, 3, &[2.0, 4.0, 6.0]);
        let output = ln.forward(&input);

        let row = input.row(0);
        let mean = row.mean();
        let var = row.variance();
        let inv_std = 1.0 / (var + 1e-5).sqrt();

        let expected_0 = (2.0 - mean) * inv_std * 2.0 + 1.0;
        let expected_1 = (4.0 - mean) * inv_std * 0.5 + (-1.0);
        let expected_2 = (6.0 - mean) * inv_std * 3.0 + 2.0;

        assert_relative_eq!(output[(0, 0)], expected_0, epsilon = 1e-7);
        assert_relative_eq!(output[(0, 1)], expected_1, epsilon = 1e-7);
        assert_relative_eq!(output[(0, 2)], expected_2, epsilon = 1e-7);
    }
}
