use super::strategies::{
    CombinationStrategy, L1RegularizationStrategy, LinearCombinationStrategy, ObjectiveStrategy,
};
use super::types::LoraStateDict;
use crate::error::{AppliedError, LoraError};
use nalgebra::{DMatrix, DVector};
use pure_math::pure_math::analysis::optimization::{
    BoxBounds, L1RegularizedLeastSquares, LbfgsBConfig, LbfgsBSolver, LbfgsBWorkspace, NelderMead,
    NelderMeadConfig, NelderMeadWorkspace, ProximalConfig, ProximalSolution,
};

/// Result of optimizing LoRA combination weights via `LoraEnsemble`.
#[derive(Debug, Clone, PartialEq)]
pub struct LoraWeightSolution {
    /// Converged module combination weights $w^* \in \mathbb{R}^m$.
    pub weights: DVector<f64>,
    /// Final regularized objective score.
    pub objective: f64,
    /// Solver iterations executed.
    pub iterations: usize,
}

/// A specialized ensemble of LoRA modules that can be combined dynamically.
///
/// `LoraEnsemble` encapsulates the collection of LoRA state dictionaries
/// and provides methods to compute weighted combinations and objective scores.
///
/// # Architecture
/// This struct uses the Strategy Pattern to decouple the combination logic
/// and objective evaluation from the data structure.
pub struct LoraEnsemble<C = LinearCombinationStrategy, O = L1RegularizationStrategy> {
    modules: Vec<LoraStateDict>,
    combination_strategy: C,
    objective_strategy: O,
}

impl LoraEnsemble<LinearCombinationStrategy, L1RegularizationStrategy> {
    /// Creates a new `LoraEnsemble` with default strategies (Linear Combination, L1 Regularization).
    ///
    /// # Arguments
    /// * `modules` - A vector of LoRA state dictionaries.
    #[verified_engine::verified]
    pub fn new(modules: Vec<LoraStateDict>) -> Self {
        Self {
            modules,
            combination_strategy: LinearCombinationStrategy,
            // Default alpha=0.0 since it wasn't stored before.
            // But evaluate_objective overrides this anyway for legacy calls.
            objective_strategy: L1RegularizationStrategy::new(0.0),
        }
    }
}

fn sorted_module_keys(module: &LoraStateDict) -> Vec<String> {
    let mut keys: Vec<String> = module.iter().map(|(k, _)| k.clone()).collect();
    keys.sort();
    keys
}

fn build_flattened_ensemble_system(
    modules: &[LoraStateDict],
    target: &LoraStateDict,
) -> Result<(DMatrix<f64>, DVector<f64>), LoraError> {
    if modules.is_empty() {
        return Err(LoraError::EmptyEnsemble);
    }
    let keys = sorted_module_keys(&modules[0]);
    let mut total_rows = 0;
    for k in &keys {
        let Some(t_mat) = target.get(k) else {
            return Err(LoraError::KeyMismatch);
        };
        let Some(m0_mat) = modules[0].get(k) else {
            return Err(LoraError::KeyMismatch);
        };
        if t_mat.shape() != m0_mat.shape() {
            return Err(LoraError::TensorShapeMismatch);
        }
        total_rows += t_mat.len();
    }
    let mut a = DMatrix::zeros(total_rows, modules.len());
    let mut y = DVector::zeros(total_rows);
    let mut offset = 0;
    for k in &keys {
        let Some(t_mat) = target.get(k) else {
            return Err(LoraError::KeyMismatch);
        };
        let t_slice = t_mat.as_slice();
        for (r, &tv) in t_slice.iter().enumerate() {
            y[offset + r] = tv;
        }
        for (col, m_dict) in modules.iter().enumerate() {
            let Some(m_mat) = m_dict.get(k) else {
                return Err(LoraError::KeyMismatch);
            };
            if m_mat.shape() != t_mat.shape() {
                return Err(LoraError::TensorShapeMismatch);
            }
            for (r, &mv) in m_mat.as_slice().iter().enumerate() {
                a[(offset + r, col)] = mv;
            }
        }
        offset += t_slice.len();
    }
    Ok((a, y))
}

impl<C: CombinationStrategy, O: ObjectiveStrategy> LoraEnsemble<C, O> {
    /// Creates a new `LoraEnsemble` with custom strategies.
    #[verified_engine::verified]
    pub fn with_strategies(
        modules: Vec<LoraStateDict>,
        combination_strategy: C,
        objective_strategy: O,
    ) -> Self {
        Self {
            modules,
            combination_strategy,
            objective_strategy,
        }
    }

    /// Returns the number of LoRA modules in the ensemble.
    #[verified_engine::verified]
    pub fn module_count(&self) -> usize {
        self.modules.len()
    }

    /// Combines the encapsulated LoRA modules using the configured strategy.
    #[verified_engine::verified]
    pub fn combine(&self, weights: &[f64]) -> Result<LoraStateDict, LoraError> {
        self.combination_strategy.combine(&self.modules, weights)
    }

    /// Calculates the objective score using the provided alpha (Legacy).
    #[verified_engine::verified]
    pub fn evaluate_objective(&self, weights: &[f64], mock_loss: f64, alpha: f64) -> f64 {
        let strategy = L1RegularizationStrategy::new(alpha);
        strategy.evaluate(weights, mock_loss)
    }

    /// Calculates the objective score using the configured strategy.
    #[verified_engine::verified]
    pub fn evaluate(&self, weights: &[f64], mock_loss: f64) -> f64 {
        self.objective_strategy.evaluate(weights, mock_loss)
    }

    /// Optimizes ensemble combination weights in-place via derivative-free [`NelderMead`]
    /// using a preallocated [`NelderMeadWorkspace`] and clamping candidate weights to `[-bound, bound]`.
    #[verified_engine::verified]
    pub fn optimize_weights_in_place<F>(
        &self,
        mut task_loss: F,
        weights: &mut DVector<f64>,
        bound: f64,
        config: &NelderMeadConfig,
        workspace: &mut NelderMeadWorkspace,
    ) -> Result<LoraWeightSolution, AppliedError>
    where
        F: FnMut(&[f64]) -> f64,
    {
        if self.modules.is_empty() {
            return Err(LoraError::EmptyEnsemble.into());
        }
        if weights.is_empty() {
            return Err(LoraError::EmptyWeights.into());
        }
        if weights.len() != self.modules.len() {
            return Err(LoraError::WeightModuleMismatch.into());
        }
        let lim = bound.abs().max(1e-6);
        let mut clamped = DVector::zeros(weights.len());
        let solver = NelderMead::new(*config);
        let (_, iterations) = solver.minimize_in_place(
            |candidate| {
                let mut overflow_pen = 0.0;
                for (i, &w) in candidate.as_slice().iter().enumerate() {
                    let cw = w.clamp(-lim, lim);
                    clamped[i] = cw;
                    let excess = (w - cw).abs();
                    overflow_pen += 100.0 * excess * excess;
                }
                let raw_loss = task_loss(clamped.as_slice());
                self.evaluate(clamped.as_slice(), raw_loss) + overflow_pen
            },
            weights,
            workspace,
        )?;
        for w in weights.as_mut_slice() {
            *w = w.clamp(-lim, lim);
        }
        let final_loss = task_loss(weights.as_slice());
        let objective = self.evaluate(weights.as_slice(), final_loss);
        Ok(LoraWeightSolution {
            weights: weights.clone(),
            objective,
            iterations,
        })
    }

    /// Convenience derivative-free weight optimizer allocating its own [`NelderMeadWorkspace`].
    #[verified_engine::verified]
    pub fn optimize_weights<F>(
        &self,
        task_loss: F,
        initial_weights: &[f64],
        bound: f64,
        config: &NelderMeadConfig,
    ) -> Result<LoraWeightSolution, AppliedError>
    where
        F: FnMut(&[f64]) -> f64,
    {
        let mut weights = DVector::from_column_slice(initial_weights);
        let mut ws = NelderMeadWorkspace::new(weights.len().max(1), config.max_iterations);
        self.optimize_weights_in_place(task_loss, &mut weights, bound, config, &mut ws)
    }

    /// Optimizes smooth differentiable task objectives over box bounds `[-bound, bound]^m`
    /// using preallocated [`LbfgsBSolver`].
    #[verified_engine::verified]
    pub fn optimize_weights_box<F>(
        &self,
        mut loss_and_grad: F,
        initial_weights: &[f64],
        bound: f64,
        config: &LbfgsBConfig,
    ) -> Result<LoraWeightSolution, AppliedError>
    where
        F: FnMut(&DVector<f64>, &mut DVector<f64>) -> f64,
    {
        if self.modules.is_empty() {
            return Err(LoraError::EmptyEnsemble.into());
        }
        if initial_weights.is_empty() {
            return Err(LoraError::EmptyWeights.into());
        }
        if initial_weights.len() != self.modules.len() {
            return Err(LoraError::WeightModuleMismatch.into());
        }
        let lim = bound.abs().max(1e-6);
        let bounds = BoxBounds::uniform(self.modules.len(), -lim, lim)?;
        let mut weights = DVector::from_column_slice(initial_weights);
        let mut ws = LbfgsBWorkspace::new(weights.len(), config.memory, config.max_iterations);
        let solver = LbfgsBSolver::new(*config);
        let (objective, _, iterations) =
            solver.solve_in_place(&mut loss_and_grad, &bounds, &mut weights, &mut ws)?;
        Ok(LoraWeightSolution {
            weights,
            objective,
            iterations,
        })
    }

    /// Solves sparse $\ell_1$-regularized linear target matching
    /// $\min_w \frac{1}{2}\|\sum_k w_k M_k - T\|_F^2 + \lambda \|w\|_1$ via accelerated FISTA.
    #[verified_engine::verified]
    pub fn optimize_target_fista(
        &self,
        target: &LoraStateDict,
        lambda: f64,
        config: &ProximalConfig,
    ) -> Result<ProximalSolution, AppliedError> {
        let (a, y) = build_flattened_ensemble_system(&self.modules, target)?;
        let lasso = L1RegularizedLeastSquares::new(lambda);
        Ok(lasso.solve_fista(&a, &y, config)?)
    }
}
