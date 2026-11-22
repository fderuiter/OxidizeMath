use super::types::LoraStateDict;
use crate::error::LoraError;

/// Strategy for combining multiple LoRA state dictionaries.
pub trait CombinationStrategy {
    #[verified_engine::verified]
    #[allow(missing_docs)]
    fn combine(
        &self,
        modules: &[LoraStateDict],
        weights: &[f64],
    ) -> Result<LoraStateDict, LoraError>;
}

/// Strategy for calculating the objective score.
pub trait ObjectiveStrategy {
    #[verified_engine::verified]
    #[allow(missing_docs)]
    fn evaluate(&self, weights: &[f64], mock_loss: f64) -> f64;
}

/// The standard linear combination strategy.
#[derive(Debug, Clone, Copy, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LinearCombinationStrategy;

impl CombinationStrategy for LinearCombinationStrategy {
    #[verified_engine::verified]
    fn combine(
        &self,
        modules: &[LoraStateDict],
        weights: &[f64],
    ) -> Result<LoraStateDict, LoraError> {
        if modules.is_empty() {
            return Err(LoraError::EmptyEnsemble);
        }
        if weights.is_empty() {
            return Err(LoraError::EmptyWeights);
        }
        if modules.len() != weights.len() {
            return Err(LoraError::WeightModuleMismatch);
        }

        let first_lora = &modules[0];
        // Initialize with the first weighted LoRA
        let mut final_state_dict: LoraStateDict = first_lora
            .iter()
            .map(|(key, tensor)| (key.clone(), tensor * weights[0]))
            .collect();

        // Accumulate the rest
        for (i, lora_state_dict) in modules.iter().enumerate().skip(1) {
            for (key, final_tensor) in &mut final_state_dict {
                if let Some(tensor) = lora_state_dict.get(key) {
                    if final_tensor.shape() != tensor.shape() {
                        return Err(LoraError::TensorShapeMismatch);
                    }
                    // Optimized in-place addition to avoid allocating intermediate matrix
                    let weight = weights[i];
                    let final_slice = final_tensor.as_mut_slice();
                    let other_slice = tensor.as_slice();
                    for (f, t) in final_slice.iter_mut().zip(other_slice.iter()) {
                        *f += *t * weight;
                    }
                } else {
                    return Err(LoraError::KeyMismatch);
                }
            }
        }

        Ok(final_state_dict)
    }
}

/// L1 Regularization strategy.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct L1RegularizationStrategy {
    #[allow(missing_docs)]
    pub alpha: f64,
}

impl L1RegularizationStrategy {
    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn new(alpha: f64) -> Self {
        Self { alpha }
    }
}

impl ObjectiveStrategy for L1RegularizationStrategy {
    #[verified_engine::verified]
    fn evaluate(&self, weights: &[f64], mock_loss: f64) -> f64 {
        if weights.is_empty() {
            return mock_loss;
        }
        let sum_abs = weights.iter().map(|w| w.abs()).sum::<f64>();
        let reg_term = self.alpha * sum_abs / (weights.len() as f64);
        mock_loss + reg_term
    }
}
