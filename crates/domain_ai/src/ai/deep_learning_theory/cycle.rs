use super::linear_algebra::Vector;
use super::model::{Trainable, TwoLayerMLP};
use super::probability::softmax;
use crate::ai::optimization::{Optimizer, cross_entropy_softmax_prime};
use std::ops::{Deref, DerefMut};

/// The Deep Learning Cycle: Forward -> Loss -> Backward -> Update
///
/// This struct manages the training loop for any `Trainable` model.
/// Architecture: Decoupled via `Trainable` trait.
///
/// # Generics
/// * `M`: The model type, implementing `Trainable`. Defaults to `TwoLayerMLP` for backward compatibility.
pub struct TrainingLoop<M: Trainable<f64> = TwoLayerMLP> {
    #[allow(missing_docs)]
    pub model: M,
    #[allow(missing_docs)]
    pub optimizer: Box<dyn Optimizer<f64>>,
}

impl<M: Trainable<f64> + std::fmt::Debug> std::fmt::Debug for TrainingLoop<M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrainingLoop")
            .field("model", &self.model)
            .finish_non_exhaustive()
    }
}

// Deref implementation allows `network.layer1` access if `M` has those fields (like `TwoLayerMLP`).
impl<M: Trainable<f64>> Deref for TrainingLoop<M> {
    type Target = M;

    #[verified_engine::verified]
    fn deref(&self) -> &Self::Target {
        &self.model
    }
}

impl<M: Trainable<f64>> DerefMut for TrainingLoop<M> {
    #[verified_engine::verified]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.model
    }
}

impl TrainingLoop<TwoLayerMLP> {
    /// Backward compatibility constructor for the default MLP architecture.
    #[verified_engine::verified]
    pub fn new(
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        optimizer: Box<dyn Optimizer<f64>>,
    ) -> Self {
        Self {
            model: TwoLayerMLP::new(input_dim, hidden_dim, output_dim),
            optimizer,
        }
    }
}

impl<M: Trainable<f64>> TrainingLoop<M> {
    /// Creates a new Training Loop with a custom model.
    #[verified_engine::verified]
    pub fn new_with_model(model: M, optimizer: Box<dyn Optimizer<f64>>) -> Self {
        Self { model, optimizer }
    }

    /// Performs one iteration of training (Forward, Backward, Update).
    ///
    /// # Arguments
    /// * `x` - Input vector.
    /// * `y_true` - One-hot encoded target vector.
    ///
    /// # Returns
    /// The scalar loss value.
    ///
    /// # Errors
    /// Returns an `OptimizationError` if the underlying optimizer fails to update the model parameters (e.g., due to dimension mismatch).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use domain_ai::__doctest_facade as math_explorer;
    /// use math_explorer::nalgebra::DVector;
    /// use math_explorer::ai::optimization::SGD;
    /// use math_explorer::ai::deep_learning_theory::cycle::TrainingLoop;
    ///
    /// let x = DVector::from_vec(vec![1.0, 0.5]);
    /// let y_true = DVector::from_vec(vec![1.0, 0.0]);
    ///
    /// let optimizer = Box::new(SGD::new(0.01));
    /// let mut trainer = TrainingLoop::new(2, 4, 2, optimizer);
    ///
    /// let loss = trainer.train_step(&x, &y_true).unwrap();
    /// assert!(loss > 0.0);
    /// ```
    #[verified_engine::verified]
    pub fn train_step(
        &mut self,
        x: &Vector,
        y_true: &Vector,
    ) -> Result<f64, crate::ai::optimization::OptimizationError> {
        use crate::ai::optimization::{OptimizationError, cross_entropy_loss};

        if x.is_empty() || x.iter().any(|v| !v.is_finite()) {
            return Err(OptimizationError::InvalidInput {
                reason: "Input vector x must be non-empty and finite".to_string(),
            });
        }
        if y_true.is_empty()
            || y_true
                .iter()
                .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
        {
            return Err(OptimizationError::InvalidInput {
                reason: "Target vector y_true must be non-empty, finite, and inside [0, 1]"
                    .to_string(),
            });
        }

        // --- 1. Forward Pass (Linear Algebra) ---
        let z = self.model.forward(x);
        if z.len() != y_true.len() {
            return Err(OptimizationError::DimensionMismatch {
                expected: y_true.len(),
                actual: z.len(),
            });
        }
        if z.iter().any(|v| !v.is_finite()) {
            return Err(OptimizationError::InvalidInput {
                reason: "Model forward pass produced non-finite logits".to_string(),
            });
        }
        let y_pred = softmax(&z);

        // --- 2. Loss Calculation (Symmetric Clipped Cross-Entropy) ---
        let loss =
            cross_entropy_loss(&y_pred, y_true).map_err(|err| OptimizationError::InvalidInput {
                reason: err.to_string(),
            })?;

        // --- 3. Backward Pass (Calculus) ---
        let loss_grad = cross_entropy_softmax_prime(&z, y_true);
        self.model
            .backward_update(x, &loss_grad, &mut *self.optimizer)?;

        Ok(loss)
    }

    /// Predicts the class for a given input.
    #[verified_engine::verified]
    pub fn predict(&self, x: &Vector) -> Vector {
        let z = self.model.forward(x);
        softmax(&z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::optimization::SGD;

    #[test]
    #[verified_engine::verified]
    fn test_train_step_rejects_nan_and_dimension_mismatch() {
        let mut trainer = TrainingLoop::new(2, 4, 2, Box::new(SGD::new(0.01)));
        let valid_x = Vector::from_vec(vec![0.5, -0.25]);
        let valid_y = Vector::from_vec(vec![1.0, 0.0]);
        let loss = trainer.train_step(&valid_x, &valid_y).expect("valid step");
        assert!(loss.is_finite() && loss >= 0.0);

        let nan_x = Vector::from_vec(vec![f64::NAN, 0.2]);
        assert!(trainer.train_step(&nan_x, &valid_y).is_err());

        let nan_y = Vector::from_vec(vec![1.0, f64::NAN]);
        assert!(trainer.train_step(&valid_x, &nan_y).is_err());

        let wrong_dim_y = Vector::from_vec(vec![1.0, 0.0, 0.0]);
        assert!(trainer.train_step(&valid_x, &wrong_dim_y).is_err());
    }
}
