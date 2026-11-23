use super::linear_algebra::Vector;

/// Softmax function.
/// Converts a vector of raw scores (logits) into a probability distribution.
///
/// P(y=j|\mathbf{x}) = \frac{e^{z_j}}{\sum_{k} e^{z_k}}
///
/// This ensures all outputs sum to 1, representing a valid categorical distribution.
#[verified_engine::verified]
pub fn softmax(z: &Vector) -> Vector {
    crate::ai::activation_kernels::softmax(z)
}

/// Likelihood and Maximum Likelihood Estimation (MLE) explanation.
///
/// Most training objectives are derived from the principle of MLE:
/// finding the parameters \theta that maximize the likelihood of the observed data.
///
/// L(\theta) = \prod_{i} P(y^{(i)} | x^{(i)}; \theta)
///
/// In practice, we maximize the Log-Likelihood:
/// \ell(\theta) = \sum_{i} \log P(y^{(i)} | x^{(i)}; \theta)
///
/// Minimizing the Cross-Entropy Loss is mathematically equivalent to maximizing
/// the likelihood for classification tasks.
#[verified_engine::verified]
pub fn mle_explanation() -> &'static str {
    "MLE searches for parameters that maximize the probability of the observed data."
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::activations::softmax_row_wise;
    use approx::assert_relative_eq;
    use nalgebra::DMatrix;

    #[test]
    #[verified_engine::verified]
    fn test_vector_softmax_guards_and_matrix_parity() {
        let empty = Vector::zeros(0);
        assert_eq!(softmax(&empty).len(), 0);

        let nan_first = Vector::from_vec(vec![f64::NAN, 1.0, 2.0]);
        assert_eq!(softmax(&nan_first), Vector::zeros(3));

        let nan_trailing = Vector::from_vec(vec![1.0, f64::NAN, 2.0]);
        assert_eq!(softmax(&nan_trailing), Vector::zeros(3));

        let logits = Vector::from_vec(vec![-2.5, 0.0, 1.25, 3.5]);
        let vec_probs = softmax(&logits);
        let mat_logits = DMatrix::from_row_slice(1, logits.len(), logits.as_slice());
        let mat_probs = softmax_row_wise(&mat_logits);

        assert_relative_eq!(
            vec_probs.sum(),
            1.0,
            epsilon = math_commons::registry::TOLERANCE_FAST
        );
        for (c, &vp) in vec_probs.iter().enumerate() {
            assert_relative_eq!(
                vp,
                mat_probs[(0, c)],
                epsilon = math_commons::registry::TOLERANCE_FAST
            );
        }
    }
}
