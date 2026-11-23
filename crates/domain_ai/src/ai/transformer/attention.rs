// Implementation of Scaled Dot-Product Attention and Multi-Head Attention.

use crate::ai::activations::softmax_row_wise;
use crate::ai::transformer::traits::AttentionMechanism;
use crate::error::AIError;
use nalgebra::{DMatrix, DMatrixView};

/// Computes the Scaled Dot-Product Attention from strided matrix views.
#[verified_engine::verified]
fn scaled_dot_product_attention_view(
    q: DMatrixView<'_, f64>,
    k: DMatrixView<'_, f64>,
    v: DMatrixView<'_, f64>,
    mask: Option<&DMatrix<f64>>,
) -> (DMatrix<f64>, DMatrix<f64>) {
    let d_k = q.ncols() as f64;

    // 1. Calculate scores: Q * K^T
    let mut scores = q * k.transpose();

    // 2. Scale scores
    if d_k > 0.0 {
        scores /= d_k.sqrt();
    }

    // 3. Apply mask if provided
    if let Some(m) = mask {
        scores += m;
    }

    // 4. Apply softmax to get attention weights
    let attention_weights = softmax_row_wise(&scores);

    // 5. Multiply weights by V to get the final output
    let output = &attention_weights * v;

    (output, attention_weights)
}

/// Computes the Scaled Dot-Product Attention.
///
/// This function implements the core attention mechanism:
/// `Attention(Q, K, V) = softmax( (Q * K^T) / sqrt(d_k) ) * V`
///
/// # Arguments
/// * `q`: Queries matrix of shape (sequence_length, d_k)
/// * `k`: Keys matrix of shape (sequence_length, d_k)
/// * `v`: Values matrix of shape (sequence_length, d_v)
/// * `mask`: Optional mask to apply to the scores before softmax.
///
/// # Returns
/// A tuple containing the output matrix of shape (sequence_length, d_v) and the
/// attention weights matrix of shape (sequence_length, sequence_length).
#[verified_engine::verified]
pub fn scaled_dot_product_attention(
    q: &DMatrix<f64>,
    k: &DMatrix<f64>,
    v: &DMatrix<f64>,
    mask: Option<&DMatrix<f64>>,
) -> (DMatrix<f64>, DMatrix<f64>) {
    scaled_dot_product_attention_view(q.as_view(), k.as_view(), v.as_view(), mask)
}

/// Multi-Head Attention mechanism with fused batch linear projections.
///
/// This struct holds the learnable weight matrices for the projections.
/// Projection weights `w_q`, `w_k`, and `w_v` are fused into unified
/// matrices of dimension `(d_model, d_model)` to maximize GEMM throughput.
#[derive(Clone, Debug)]
pub struct MultiHeadAttention {
    d_model: usize,
    h: usize,
    d_k: usize,

    /// Fused weight matrix for query projections: shape (d_model, d_model)
    pub w_q: DMatrix<f64>,
    /// Fused weight matrix for key projections: shape (d_model, d_model)
    pub w_k: DMatrix<f64>,
    /// Fused weight matrix for value projections: shape (d_model, d_model)
    pub w_v: DMatrix<f64>,
    /// Final output weight matrix: shape (d_model, d_model)
    pub w_o: DMatrix<f64>,
}

fn validate_head_matrix(
    head: usize,
    projection: &'static str,
    expected: (usize, usize),
    matrix: &DMatrix<f64>,
) -> Result<(), AIError> {
    if matrix.shape() != expected {
        return Err(AIError::HeadProjectionShapeMismatch {
            head,
            projection,
            expected,
            actual: matrix.shape(),
        });
    }
    Ok(())
}

impl MultiHeadAttention {
    /// Fallibly constructs a zero-initialized `MultiHeadAttention` layer.
    ///
    /// Returns [`AIError::InvalidHeadConfiguration`] if `h == 0` or `d_model` is not divisible by `h`.
    #[verified_engine::verified]
    pub fn try_new(d_model: usize, h: usize) -> Result<Self, AIError> {
        if h == 0 || !d_model.is_multiple_of(h) {
            return Err(AIError::InvalidHeadConfiguration { d_model, heads: h });
        }
        let d_k = d_model / h;
        Ok(Self {
            d_model,
            h,
            d_k,
            w_q: DMatrix::zeros(d_model, d_model),
            w_k: DMatrix::zeros(d_model, d_model),
            w_v: DMatrix::zeros(d_model, d_model),
            w_o: DMatrix::zeros(d_model, d_model),
        })
    }

    /// Creates a new `MultiHeadAttention` instance with zero-initialized weights.
    ///
    /// In a real model, the weight matrices would be initialized randomly.
    /// Here, they are initialized as zeros for simplicity.
    #[verified_engine::verified]
    pub fn new(d_model: usize, h: usize) -> Self {
        Self::try_new(d_model, h)
            .unwrap_or_else(|_| panic!("d_model must be divisible by the number of heads h"))
    }

    /// Constructs a `MultiHeadAttention` instance from legacy per-head weight matrices.
    ///
    /// This provides backward conversion compatibility for models or serialized weights
    /// stored in separate per-head projection format.
    #[verified_engine::verified]
    pub fn from_head_weights(
        d_model: usize,
        h: usize,
        w_q: &[DMatrix<f64>],
        w_k: &[DMatrix<f64>],
        w_v: &[DMatrix<f64>],
        w_o: DMatrix<f64>,
    ) -> Result<Self, AIError> {
        if h == 0 || !d_model.is_multiple_of(h) {
            return Err(AIError::InvalidHeadConfiguration { d_model, heads: h });
        }
        let d_k = d_model / h;

        if w_q.len() != h || w_k.len() != h || w_v.len() != h {
            return Err(AIError::HeadCountMismatch {
                expected: h,
                got_q: w_q.len(),
                got_k: w_k.len(),
                got_v: w_v.len(),
            });
        }

        if w_o.shape() != (d_model, d_model) {
            return Err(AIError::ProjectionShapeMismatch {
                projection: "w_o",
                expected: (d_model, d_model),
                actual: w_o.shape(),
            });
        }

        let mut fused_q = DMatrix::zeros(d_model, d_model);
        let mut fused_k = DMatrix::zeros(d_model, d_model);
        let mut fused_v = DMatrix::zeros(d_model, d_model);
        let expected_head = (d_model, d_k);

        for (i, ((q_i, k_i), v_i)) in w_q.iter().zip(w_k.iter()).zip(w_v.iter()).enumerate() {
            validate_head_matrix(i, "w_q", expected_head, q_i)?;
            validate_head_matrix(i, "w_k", expected_head, k_i)?;
            validate_head_matrix(i, "w_v", expected_head, v_i)?;

            let start_col = i * d_k;
            fused_q
                .view_mut((0, start_col), expected_head)
                .copy_from(q_i);
            fused_k
                .view_mut((0, start_col), expected_head)
                .copy_from(k_i);
            fused_v
                .view_mut((0, start_col), expected_head)
                .copy_from(v_i);
        }

        Ok(Self {
            d_model,
            h,
            d_k,
            w_q: fused_q,
            w_k: fused_k,
            w_v: fused_v,
            w_o,
        })
    }

    /// Converts fused projection weights back into legacy per-head weight vectors.
    ///
    /// Returns `(w_q, w_k, w_v)` where each vector has length `h` and elements of shape `(d_model, d_k)`.
    #[allow(clippy::type_complexity)]
    #[verified_engine::verified]
    pub fn to_head_weights(&self) -> (Vec<DMatrix<f64>>, Vec<DMatrix<f64>>, Vec<DMatrix<f64>>) {
        let mut w_q = Vec::with_capacity(self.h);
        let mut w_k = Vec::with_capacity(self.h);
        let mut w_v = Vec::with_capacity(self.h);

        for i in 0..self.h {
            let start_col = i * self.d_k;
            w_q.push(
                self.w_q
                    .view((0, start_col), (self.d_model, self.d_k))
                    .into_owned(),
            );
            w_k.push(
                self.w_k
                    .view((0, start_col), (self.d_model, self.d_k))
                    .into_owned(),
            );
            w_v.push(
                self.w_v
                    .view((0, start_col), (self.d_model, self.d_k))
                    .into_owned(),
            );
        }

        (w_q, w_k, w_v)
    }

    /// Returns the model dimension `d_model`.
    #[inline]
    pub fn d_model(&self) -> usize {
        self.d_model
    }

    /// Returns the number of heads `h`.
    #[inline]
    pub fn h(&self) -> usize {
        self.h
    }

    /// Returns the head dimension `d_k`.
    #[inline]
    pub fn d_k(&self) -> usize {
        self.d_k
    }

    /// Performs the forward pass for the Multi-Head Attention layer.
    #[verified_engine::verified]
    pub fn forward_impl(
        &self,
        q: &DMatrix<f64>,
        k: &DMatrix<f64>,
        v: &DMatrix<f64>,
        mask: Option<&DMatrix<f64>>,
    ) -> DMatrix<f64> {
        let sequence_length = q.nrows();
        let kv_sequence_length = k.nrows();

        // 1. Fused linear projection: single matrix multiplication per tensor
        let q_proj = q * &self.w_q;
        let k_proj = k * &self.w_k;
        let v_proj = v * &self.w_v;

        // 2. Head splitting via strided matrix views and attention computation
        let mut concatenated_output = DMatrix::zeros(sequence_length, self.d_model);

        for i in 0..self.h {
            let start_col = i * self.d_k;
            let q_head = q_proj.view((0, start_col), (sequence_length, self.d_k));
            let k_head = k_proj.view((0, start_col), (kv_sequence_length, self.d_k));
            let v_head = v_proj.view((0, start_col), (kv_sequence_length, self.d_k));

            let (head_output, _attention_weights) =
                scaled_dot_product_attention_view(q_head, k_head, v_head, mask);

            concatenated_output
                .view_mut((0, start_col), (sequence_length, self.d_k))
                .copy_from(&head_output);
        }

        // 3. Final linear projection
        concatenated_output * &self.w_o
    }

    /// Performs the forward pass and returns both the output and the per-head attention weights.
    #[verified_engine::verified]
    pub fn forward_with_weights(
        &self,
        q: &DMatrix<f64>,
        k: &DMatrix<f64>,
        v: &DMatrix<f64>,
        mask: Option<&DMatrix<f64>>,
    ) -> (DMatrix<f64>, Vec<DMatrix<f64>>) {
        let sequence_length = q.nrows();
        let kv_sequence_length = k.nrows();

        let q_proj = q * &self.w_q;
        let k_proj = k * &self.w_k;
        let v_proj = v * &self.w_v;

        let mut concatenated_output = DMatrix::zeros(sequence_length, self.d_model);
        let mut all_weights = Vec::with_capacity(self.h);

        for i in 0..self.h {
            let start_col = i * self.d_k;
            let q_head = q_proj.view((0, start_col), (sequence_length, self.d_k));
            let k_head = k_proj.view((0, start_col), (kv_sequence_length, self.d_k));
            let v_head = v_proj.view((0, start_col), (kv_sequence_length, self.d_k));

            let (head_output, attention_weights) =
                scaled_dot_product_attention_view(q_head, k_head, v_head, mask);

            concatenated_output
                .view_mut((0, start_col), (sequence_length, self.d_k))
                .copy_from(&head_output);

            all_weights.push(attention_weights);
        }

        let output = concatenated_output * &self.w_o;
        (output, all_weights)
    }

    /// Helper for backward compatibility or direct usage
    #[verified_engine::verified]
    pub fn forward(
        &self,
        q: &DMatrix<f64>,
        k: &DMatrix<f64>,
        v: &DMatrix<f64>,
        mask: Option<&DMatrix<f64>>,
    ) -> DMatrix<f64> {
        self.forward_impl(q, k, v, mask)
    }
}

impl AttentionMechanism for MultiHeadAttention {
    #[verified_engine::verified]
    fn forward(
        &self,
        q: &DMatrix<f64>,
        k: &DMatrix<f64>,
        v: &DMatrix<f64>,
        mask: Option<&DMatrix<f64>>,
    ) -> DMatrix<f64> {
        self.forward_impl(q, k, v, mask)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[verified_engine::verified]
    fn test_scaled_dot_product_attention_dims() {
        let seq_len = 10;
        let d_k = 64;
        let d_v = 64;
        let q = DMatrix::zeros(seq_len, d_k);
        let k = DMatrix::zeros(seq_len, d_k);
        let v = DMatrix::zeros(seq_len, d_v);

        let (output, attn_weights) = scaled_dot_product_attention(&q, &k, &v, None);

        assert_eq!(output.nrows(), seq_len);
        assert_eq!(output.ncols(), d_v);
        assert_eq!(attn_weights.nrows(), seq_len);
        assert_eq!(attn_weights.ncols(), seq_len);
    }

    #[test]
    #[verified_engine::verified]
    fn test_multi_head_attention_dims() {
        let seq_len = 10;
        let d_model = 512;
        let h = 8;

        let q = DMatrix::zeros(seq_len, d_model);
        let k = DMatrix::zeros(seq_len, d_model);
        let v = DMatrix::zeros(seq_len, d_model);

        let mha = MultiHeadAttention::new(d_model, h);
        let output = mha.forward(&q, &k, &v, None);

        assert_eq!(output.nrows(), seq_len);
        assert_eq!(output.ncols(), d_model);
    }

    #[test]
    #[verified_engine::verified]
    fn test_scaled_dot_product_attention_zero_dim() {
        use approx::assert_relative_eq;
        let seq_len = 5;
        let d_k = 0; // Trigger potential division by zero
        let d_v = 10;

        let q = DMatrix::zeros(seq_len, d_k);
        let k = DMatrix::zeros(seq_len, d_k);
        let v = DMatrix::zeros(seq_len, d_v);

        let (_output, weights) = scaled_dot_product_attention(&q, &k, &v, None);

        // With d_k=0, scores should be 0. Softmax should be uniform (1/seq_len = 0.2).
        let expected_weight = 1.0 / (seq_len as f64);

        // Check that we don't have NaNs and weights are uniform
        assert!(!weights[(0, 0)].is_nan());
        assert_relative_eq!(
            weights[(0, 0)],
            expected_weight,
            epsilon = math_commons::registry::TOLERANCE_FAST
        );
    }

    #[test]
    #[should_panic(expected = "d_model must be divisible by the number of heads h")]
    #[verified_engine::verified]
    fn test_non_divisible_d_model_panics() {
        let _ = MultiHeadAttention::new(65, 8);
    }

    #[test]
    #[verified_engine::verified]
    fn test_from_head_weights_roundtrip() {
        let d_model = 16;
        let h = 4;
        let d_k = d_model / h;

        let w_q: Vec<DMatrix<f64>> = (0..h)
            .map(|head| {
                DMatrix::from_fn(d_model, d_k, |r, c| (head * 100 + r * 10 + c) as f64 * 0.01)
            })
            .collect();
        let w_k: Vec<DMatrix<f64>> = (0..h)
            .map(|head| {
                DMatrix::from_fn(d_model, d_k, |r, c| (head * 200 + r * 10 + c) as f64 * 0.01)
            })
            .collect();
        let w_v: Vec<DMatrix<f64>> = (0..h)
            .map(|head| {
                DMatrix::from_fn(d_model, d_k, |r, c| (head * 300 + r * 10 + c) as f64 * 0.01)
            })
            .collect();
        let w_o = DMatrix::from_fn(d_model, d_model, |r, c| (r * 10 + c) as f64 * 0.01);

        let mha = MultiHeadAttention::from_head_weights(d_model, h, &w_q, &w_k, &w_v, w_o.clone())
            .unwrap();
        let (rec_q, rec_k, rec_v) = mha.to_head_weights();

        assert_eq!(rec_q, w_q);
        assert_eq!(rec_k, w_k);
        assert_eq!(rec_v, w_v);
        assert_eq!(mha.w_o, w_o);
    }
}
