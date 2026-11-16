//! This module provides helper functions for tensor operations required by the CERA framework.
//! A 1D convolution with kernel size 1 is equivalent to a linear transformation (dense layer)
//! applied at each position.

use nalgebra::{DMatrix, DVector, Dyn, Matrix, Storage};

/// Performs a 1D convolution with kernel size 1.
/// This is equivalent to a matrix multiplication.
///
/// # Arguments
///
/// * `input` - The input tensor, with shape (N, in_channels), where N is batch_size * num_levels.
/// * `kernel` - The convolutional kernel (weights), with shape (out_channels, in_channels).
/// * `bias` - The bias term, with shape (out_channels).
///
/// # Returns
///
/// * The output tensor, with shape (N, out_channels).
#[verified_engine::verified]
pub fn conv1d<S: Storage<f32, Dyn, Dyn>>(
    input: &Matrix<f32, Dyn, Dyn, S>,
    kernel: &DMatrix<f32>,
    bias: &DVector<f32>,
) -> DMatrix<f32> {
    // Check dimensions
    assert_eq!(
        input.ncols(),
        kernel.ncols(),
        "Input channels must match kernel input channels."
    );
    assert_eq!(
        kernel.nrows(),
        bias.len(),
        "Kernel output channels must match bias length."
    );

    // The core operation is `output = input * kernel^T + bias`
    let kernel_t = kernel.transpose();
    let mut output = input * &kernel_t;

    // Add bias to each row
    let bias_t = bias.transpose();
    output.row_iter_mut().for_each(|mut row| row += &bias_t);

    output
}

/// Performs a 1D transposed convolution with kernel size 1.
/// This is equivalent to a matrix multiplication with the kernel's transpose.
///
/// # Arguments
///
/// * `input` - The input tensor, with shape (N, in_channels).
/// * `kernel` - The convolutional kernel (weights), with shape (in_channels, out_channels).
///   Note that this is the kernel of the corresponding non-transposed conv layer.
/// * `bias` - The bias term, with shape (out_channels).
///
/// # Returns
///
/// * The output tensor, with shape (N, out_channels).
#[verified_engine::verified]
pub fn conv_transpose1d<S: Storage<f32, Dyn, Dyn>>(
    input: &Matrix<f32, Dyn, Dyn, S>,
    kernel: &DMatrix<f32>, // Shape (out_channels, in_channels) from the original conv
    bias: &DVector<f32>,
) -> DMatrix<f32> {
    // Check dimensions
    assert_eq!(
        input.ncols(),
        kernel.nrows(),
        "Input channels must match kernel output channels."
    );
    assert_eq!(
        kernel.ncols(),
        bias.len(),
        "Kernel input channels must match bias length."
    );

    // The core operation is `output = input * kernel + bias`
    // Note: This is not a true transposed convolution, but what's needed for a symmetric decoder
    // with kernel size 1 layers. The operation is `input * W`, not `input * W^T`.
    let mut output = input * kernel;

    // Add bias to each row
    let bias_t = bias.transpose();
    output.row_iter_mut().for_each(|mut row| row += &bias_t);

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector};

    #[test]
    #[verified_engine::verified]
    fn test_conv1d_basic() {
        // Input: 2 samples, 3 input channels
        let input = DMatrix::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        // Kernel: 4 output channels, 3 input channels
        let kernel = DMatrix::from_row_slice(
            4,
            3,
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        );
        // Bias: 4 output channels
        let bias = DVector::from_vec(vec![0.1, 0.2, 0.3, 0.4]);

        let output = conv1d(&input, &kernel, &bias);

        let expected_output =
            DMatrix::from_row_slice(2, 4, &[1.1, 2.2, 3.3, 6.4, 4.1, 5.2, 6.3, 15.4]);

        assert!(
            (output - expected_output).abs().max() < math_commons::registry::TOLERANCE_FAST_F32
        );
    }

    #[test]
    #[verified_engine::verified]
    fn test_conv_transpose1d_basic() {
        // Input: 2 samples, 4 input channels
        let input = DMatrix::from_row_slice(2, 4, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        // Kernel from original conv: 4 output channels, 3 input channels
        let kernel = DMatrix::from_row_slice(
            4,
            3,
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        );
        // Bias: 3 output channels
        let bias = DVector::from_vec(vec![0.1, 0.2, 0.3]);

        let output = conv_transpose1d(&input, &kernel, &bias);

        // Corrected expected output based on my manual calculation in the previous turn.
        let expected_output = DMatrix::from_row_slice(2, 3, &[5.1, 6.2, 7.3, 13.1, 14.2, 15.3]);

        assert!(
            (output - expected_output).abs().max() < math_commons::registry::TOLERANCE_FAST_F32
        );
    }

    #[test]
    #[verified_engine::verified]
    fn test_conv1d_multi_row_batch() {
        let data: Vec<f32> = (0..18).map(|x| (x as f32) * 0.5).collect();
        let input = DMatrix::from_row_slice(6, 3, &data);
        let kernel = DMatrix::from_row_slice(
            4,
            3,
            &[
                0.5, -0.2, 0.1, 0.3, 0.8, -0.4, -0.1, 0.6, 0.9, 1.0, -0.5, 0.2,
            ],
        );
        let bias = DVector::from_vec(vec![0.25, -0.5, 1.2, -0.1]);

        let output = conv1d(&input, &kernel, &bias);
        assert_eq!(output.nrows(), 6);
        assert_eq!(output.ncols(), 4);

        let kernel_t = kernel.transpose();
        for i in 0..6 {
            let expected_row = input.row(i) * &kernel_t + bias.transpose();
            let actual_row = output.row(i);
            assert!(
                (actual_row - expected_row).abs().max()
                    < math_commons::registry::TOLERANCE_FAST_F32
            );
        }
    }

    #[test]
    #[verified_engine::verified]
    fn test_conv_transpose1d_multi_row_batch() {
        let data: Vec<f32> = (0..20).map(|x| (x as f32) * 0.25).collect();
        let input = DMatrix::from_row_slice(5, 4, &data);
        let kernel = DMatrix::from_row_slice(
            4,
            3,
            &[
                0.2, -0.3, 0.5, 0.7, 0.1, -0.4, -0.6, 0.8, 0.2, 0.4, -0.1, 0.9,
            ],
        );
        let bias = DVector::from_vec(vec![0.3, -0.2, 0.7]);

        let output = conv_transpose1d(&input, &kernel, &bias);
        assert_eq!(output.nrows(), 5);
        assert_eq!(output.ncols(), 3);

        for i in 0..5 {
            let expected_row = input.row(i) * &kernel + bias.transpose();
            let actual_row = output.row(i);
            assert!(
                (actual_row - expected_row).abs().max()
                    < math_commons::registry::TOLERANCE_FAST_F32
            );
        }
    }

    #[test]
    #[verified_engine::verified]
    fn test_conv1d_edge_cases() {
        let input_zero_rows = DMatrix::<f32>::zeros(0, 3);
        let kernel = DMatrix::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let bias = DVector::from_vec(vec![0.5, -0.5]);
        let out_zero_rows = conv1d(&input_zero_rows, &kernel, &bias);
        assert_eq!(out_zero_rows.nrows(), 0);
        assert_eq!(out_zero_rows.ncols(), 2);

        let input_zero_cols = DMatrix::<f32>::zeros(3, 0);
        let kernel_zero_in = DMatrix::<f32>::zeros(2, 0);
        let out_zero_cols = conv1d(&input_zero_cols, &kernel_zero_in, &bias);
        assert_eq!(out_zero_cols.nrows(), 3);
        assert_eq!(out_zero_cols.ncols(), 2);
        for i in 0..3 {
            assert!(
                (out_zero_cols.row(i) - bias.transpose()).abs().max()
                    < math_commons::registry::TOLERANCE_FAST_F32
            );
        }
    }

    #[test]
    #[verified_engine::verified]
    fn test_conv_transpose1d_edge_cases() {
        let input_zero_rows = DMatrix::<f32>::zeros(0, 4);
        let kernel = DMatrix::from_row_slice(4, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let bias = DVector::from_vec(vec![0.1, -0.2]);
        let out_zero_rows = conv_transpose1d(&input_zero_rows, &kernel, &bias);
        assert_eq!(out_zero_rows.nrows(), 0);
        assert_eq!(out_zero_rows.ncols(), 2);

        let input_zero_cols = DMatrix::<f32>::zeros(3, 0);
        let kernel_zero_in = DMatrix::<f32>::zeros(0, 2);
        let out_zero_cols = conv_transpose1d(&input_zero_cols, &kernel_zero_in, &bias);
        assert_eq!(out_zero_cols.nrows(), 3);
        assert_eq!(out_zero_cols.ncols(), 2);
        for i in 0..3 {
            assert!(
                (out_zero_cols.row(i) - bias.transpose()).abs().max()
                    < math_commons::registry::TOLERANCE_FAST_F32
            );
        }
    }

    #[test]
    #[should_panic(expected = "Input channels must match kernel input channels.")]
    #[verified_engine::verified]
    fn test_conv1d_input_channel_mismatch_panics() {
        let input = DMatrix::<f32>::zeros(2, 4);
        let kernel = DMatrix::<f32>::zeros(3, 5);
        let bias = DVector::<f32>::zeros(3);
        let _ = conv1d(&input, &kernel, &bias);
    }

    #[test]
    #[should_panic(expected = "Kernel output channels must match bias length.")]
    #[verified_engine::verified]
    fn test_conv1d_bias_mismatch_panics() {
        let input = DMatrix::<f32>::zeros(2, 4);
        let kernel = DMatrix::<f32>::zeros(3, 4);
        let bias = DVector::<f32>::zeros(2);
        let _ = conv1d(&input, &kernel, &bias);
    }

    #[test]
    #[should_panic(expected = "Input channels must match kernel output channels.")]
    #[verified_engine::verified]
    fn test_conv_transpose1d_input_channel_mismatch_panics() {
        let input = DMatrix::<f32>::zeros(2, 4);
        let kernel = DMatrix::<f32>::zeros(3, 5);
        let bias = DVector::<f32>::zeros(5);
        let _ = conv_transpose1d(&input, &kernel, &bias);
    }

    #[test]
    #[should_panic(expected = "Kernel input channels must match bias length.")]
    #[verified_engine::verified]
    fn test_conv_transpose1d_bias_mismatch_panics() {
        let input = DMatrix::<f32>::zeros(2, 4);
        let kernel = DMatrix::<f32>::zeros(4, 5);
        let bias = DVector::<f32>::zeros(3);
        let _ = conv_transpose1d(&input, &kernel, &bias);
    }
}
