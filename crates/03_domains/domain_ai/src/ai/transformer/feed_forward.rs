// Implementation of the Position-wise Feed-Forward Network.

use crate::ai::activations::relu;
use crate::ai::transformer::traits::FeedForwardNetwork;
use crate::ai::utils::AddRowVector;
use nalgebra::{DMatrix, RowDVector};

/// A Position-wise Feed-Forward Network (FFN) as described in "Attention Is All You Need".
///
/// This network is applied to each position separately and identically. It consists of
/// two linear transformations with a ReLU activation in between.
/// Formula: `FFN(x) = max(0, x * W1 + b1) * W2 + b2`
pub struct FeedForward {
    /// Weight matrix for the first linear transformation.
    pub w1: DMatrix<f64>,
    /// Bias vector for the first linear transformation.
    pub b1: RowDVector<f64>,
    /// Weight matrix for the second linear transformation.
    pub w2: DMatrix<f64>,
    /// Bias vector for the second linear transformation.
    pub b2: RowDVector<f64>,
}

impl FeedForward {
    /// Creates a new `FeedForward` network.
    ///
    /// In a real model, the weights would be initialized randomly. Here, they are
    /// initialized as zeros for simplicity.
    ///
    /// # Arguments
    /// * `d_model`: The input and output dimension of the network.
    /// * `d_ff`: The inner-layer dimension.
    #[verified_engine::verified]
    pub fn new(d_model: usize, d_ff: usize) -> Self {
        Self {
            w1: DMatrix::zeros(d_model, d_ff),
            b1: RowDVector::zeros(d_ff),
            w2: DMatrix::zeros(d_ff, d_model),
            b2: RowDVector::zeros(d_model),
        }
    }

    /// Performs the forward pass of the FFN.
    #[verified_engine::verified]
    pub fn forward_impl(&self, x: &DMatrix<f64>) -> DMatrix<f64> {
        // First linear transformation: x * W1 + b1
        let mut hidden = x * &self.w1;
        hidden.add_row_vector_to_all_rows(&self.b1);

        // Apply ReLU activation
        relu(&mut hidden);

        // Second linear transformation: hidden * W2 + b2
        let mut output = hidden * &self.w2;
        output.add_row_vector_to_all_rows(&self.b2);

        output
    }

    /// Performs the forward pass of the FFN.
    #[verified_engine::verified]
    pub fn forward(&self, x: &DMatrix<f64>) -> DMatrix<f64> {
        self.forward_impl(x)
    }
}

impl FeedForwardNetwork for FeedForward {
    #[verified_engine::verified]
    fn forward(&self, x: &DMatrix<f64>) -> DMatrix<f64> {
        self.forward_impl(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[verified_engine::verified]
    fn test_feed_forward_dims() {
        let seq_len = 10;
        let d_model = 512;
        let d_ff = 2048;

        let x = DMatrix::zeros(seq_len, d_model);
        let ffn = FeedForward::new(d_model, d_ff);
        let output = ffn.forward(&x);

        assert_eq!(output.nrows(), seq_len);
        assert_eq!(output.ncols(), d_model);
    }
}
