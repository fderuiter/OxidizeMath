//! Gradient-free zero-allocation Nelder-Mead simplex solver for LoraHub weight composition.
//!
//! Implements the bounded black-box simplex optimization used by LoraHub (Huang et al., 2023,
//! `arXiv:2307.13269`) to discover sparse linear combinations of LoRA adapters under
//! [`super::strategies::L1RegularizationStrategy`] without computing backpropagation gradients.

use math_commons::math_kernel::types::flatten_2d_index;

/// Outcome of a single Nelder-Mead simplex transition step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimplexStepOutcome {
    /// Objective spread fell at or below `config.tolerance`.
    Converged,
    /// Worst vertex was replaced by its centroid reflection.
    Reflected,
    /// Reflection improved on the current best and expanded further.
    Expanded,
    /// Outside contraction improved upon the reflected candidate.
    ContractedOutside,
    /// Inside contraction improved upon the worst vertex.
    ContractedInside,
    /// Entire simplex shrunk toward the current best vertex.
    Shrunk,
}

/// Summary statistics returned by [`NelderMeadSimplex::optimize_in_place`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SimplexOptimizationSummary {
    /// Number of simplex iterations executed.
    pub steps: usize,
    /// Lowest objective value achieved across all simplex vertices.
    pub best_value: f64,
    /// Whether convergence was reached within `max_steps`.
    pub converged: bool,
}

/// Hyperparameters governing bounded Nelder-Mead simplex updates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NelderMeadConfig {
    /// Reflection factor $\alpha > 0$ (standard `1.0`).
    pub reflection: f64,
    /// Expansion factor $\gamma > 1$ (standard `2.0`).
    pub expansion: f64,
    /// Contraction factor $\rho \in (0, 1)$ (standard `0.5`).
    pub contraction: f64,
    /// Shrink factor $\sigma \in (0, 1)$ (standard `0.5`).
    pub shrink: f64,
    /// Coordinate perturbation step for initial simplex construction.
    pub initial_step: f64,
    /// Symmetric box bound $[-B, B]$ applied to adapter weights (`1.5` in LoraHub).
    pub weight_bound: f64,
    /// Stopping threshold on objective value spread $|f_{\max} - f_{\min}|$.
    pub tolerance: f64,
    /// Maximum number of simplex iterations in [`NelderMeadSimplex::optimize_in_place`].
    pub max_steps: usize,
}

impl Default for NelderMeadConfig {
    fn default() -> Self {
        Self {
            reflection: 1.0,
            expansion: 2.0,
            contraction: 0.5,
            shrink: 0.5,
            initial_step: 0.35,
            weight_bound: 1.5,
            tolerance: 1e-6,
            max_steps: 200,
        }
    }
}

/// Pre-allocated workspace for zero-allocation Nelder-Mead optimization in $\mathbb{R}^N$.
#[derive(Debug, Clone)]
pub struct NelderMeadSimplex {
    dim: usize,
    vertices: Vec<f64>,
    values: Vec<f64>,
    order: Vec<usize>,
    centroid: Vec<f64>,
    reflected: Vec<f64>,
    expanded: Vec<f64>,
    contracted: Vec<f64>,
    shrink_trial: Vec<f64>,
    steps_taken: usize,
}

#[inline]
fn clamp_weight(val: f64, bound: f64) -> f64 {
    let b = bound.abs().max(1e-6);
    val.clamp(-b, b)
}

#[inline]
fn vertex_offset(vertex_idx: usize, dim: usize) -> usize {
    flatten_2d_index(0, vertex_idx, dim)
}

impl NelderMeadSimplex {
    /// Allocates a reusable Nelder-Mead workspace for `dim` adapter weights ($N \ge 1$).
    #[verified_engine::verified]
    pub fn new(dim: usize) -> Self {
        assert!(dim >= 1, "NelderMeadSimplex requires dim >= 1");
        let n_verts = dim + 1;
        Self {
            dim,
            vertices: vec![0.0; n_verts * dim],
            values: vec![0.0; n_verts],
            order: (0..n_verts).collect(),
            centroid: vec![0.0; dim],
            reflected: vec![0.0; dim],
            expanded: vec![0.0; dim],
            contracted: vec![0.0; dim],
            shrink_trial: vec![0.0; dim],
            steps_taken: 0,
        }
    }

    /// Number of adapter weights $N$ optimized by this simplex.
    pub fn dimension(&self) -> usize {
        self.dim
    }

    /// Total simplex transition steps executed since initialization.
    pub fn steps_taken(&self) -> usize {
        self.steps_taken
    }

    /// Current minimum objective score across the $N + 1$ simplex vertices.
    pub fn best_value(&self) -> f64 {
        self.values[self.order[0]]
    }

    /// Current maximum objective score across the $N + 1$ simplex vertices.
    pub fn worst_value(&self) -> f64 {
        self.values[self.order[self.dim]]
    }

    /// Spread $|f_{\text{worst}} - f_{\text{best}}|$ across the current simplex.
    pub fn value_spread(&self) -> f64 {
        (self.worst_value() - self.best_value()).abs()
    }

    /// Returns the adapter weights of the `rank`-th best vertex (`0` = best, `dim` = worst).
    pub fn vertex(&self, rank: usize) -> &[f64] {
        let clamped_rank = rank.min(self.dim);
        let v_idx = self.order[clamped_rank];
        let start = vertex_offset(v_idx, self.dim);
        &self.vertices[start..start + self.dim]
    }

    /// Objective value of the `rank`-th best vertex (`0` = best, `dim` = worst).
    pub fn vertex_value(&self, rank: usize) -> f64 {
        self.values[self.order[rank.min(self.dim)]]
    }

    /// Slice of current best adapter weights.
    pub fn best_weights(&self) -> &[f64] {
        self.vertex(0)
    }

    /// Centroid of the best $N$ vertices (excluding the worst vertex).
    pub fn centroid(&self) -> &[f64] {
        &self.centroid
    }

    /// Copies the best adapter weights into `out` without allocating.
    pub fn copy_best_weights_into(&self, out: &mut [f64]) {
        let best = self.best_weights();
        let count = best.len().min(out.len());
        out[..count].copy_from_slice(&best[..count]);
    }

    fn sort_order_in_place(&mut self) {
        let n_verts = self.dim + 1;
        for i in 1..n_verts {
            let key_idx = self.order[i];
            let key_val = self.values[key_idx];
            let mut j = i;
            while j > 0 && self.values[self.order[j - 1]] > key_val {
                self.order[j] = self.order[j - 1];
                j -= 1;
            }
            self.order[j] = key_idx;
        }
    }

    fn update_centroid_in_place(&mut self) {
        self.centroid.fill(0.0);
        for rank in 0..self.dim {
            let v_idx = self.order[rank];
            let start = vertex_offset(v_idx, self.dim);
            for c in 0..self.dim {
                self.centroid[c] += self.vertices[start + c];
            }
        }
        let inv_n = 1.0 / (self.dim as f64);
        for c in &mut self.centroid {
            *c *= inv_n;
        }
    }

    fn replace_worst_with_reflected(&mut self, value: f64) {
        let worst_idx = self.order[self.dim];
        let start = vertex_offset(worst_idx, self.dim);
        self.vertices[start..start + self.dim].copy_from_slice(&self.reflected);
        self.values[worst_idx] = value;
    }

    fn shrink_toward_best<F: FnMut(&[f64]) -> f64>(
        &mut self,
        config: &NelderMeadConfig,
        eval: &mut F,
    ) {
        let best_idx = self.order[0];
        let best_start = vertex_offset(best_idx, self.dim);
        for rank in 1..=self.dim {
            let v_idx = self.order[rank];
            let v_start = vertex_offset(v_idx, self.dim);
            for c in 0..self.dim {
                let best_c = self.vertices[best_start + c];
                let cur_c = self.vertices[v_start + c];
                let shrunk = best_c + config.shrink * (cur_c - best_c);
                self.shrink_trial[c] = clamp_weight(shrunk, config.weight_bound);
            }
            self.vertices[v_start..v_start + self.dim].copy_from_slice(&self.shrink_trial);
            self.values[v_idx] = eval(&self.shrink_trial);
        }
    }

    /// Initializes the $N + 1$ simplex vertices around `initial` and evaluates each vertex in place.
    #[verified_engine::verified]
    pub fn initialize_in_place<F: FnMut(&[f64]) -> f64>(
        &mut self,
        initial: &[f64],
        config: &NelderMeadConfig,
        eval: &mut F,
    ) {
        assert_eq!(
            initial.len(),
            self.dim,
            "initial slice must match simplex dim"
        );
        self.steps_taken = 0;
        for idx in 0..=self.dim {
            self.order[idx] = idx;
            let start = vertex_offset(idx, self.dim);
            for c in 0..self.dim {
                let mut val = clamp_weight(initial[c], config.weight_bound);
                if idx > 0 && (idx - 1) == c {
                    let step = if val + config.initial_step <= config.weight_bound.abs() {
                        config.initial_step
                    } else {
                        -config.initial_step
                    };
                    val = clamp_weight(val + step, config.weight_bound);
                }
                self.vertices[start + c] = val;
            }
            let v_slice = &self.vertices[start..start + self.dim];
            self.values[idx] = eval(v_slice);
        }
        self.sort_order_in_place();
        self.update_centroid_in_place();
    }

    /// Executes one zero-allocation Nelder-Mead simplex step (reflect, expand, contract, or shrink).
    #[verified_engine::verified]
    pub fn step_in_place<F: FnMut(&[f64]) -> f64>(
        &mut self,
        config: &NelderMeadConfig,
        eval: &mut F,
    ) -> SimplexStepOutcome {
        debug_assert_eq!(self.values.len(), self.dim + 1);
        if self.value_spread() <= config.tolerance {
            return SimplexStepOutcome::Converged;
        }
        self.update_centroid_in_place();

        let worst_idx = self.order[self.dim];
        let worst_start = vertex_offset(worst_idx, self.dim);
        for c in 0..self.dim {
            let r = self.centroid[c]
                + config.reflection * (self.centroid[c] - self.vertices[worst_start + c]);
            self.reflected[c] = clamp_weight(r, config.weight_bound);
        }
        let f_reflected = eval(&self.reflected);
        let f_best = self.values[self.order[0]];
        let f_second_worst = self.values[self.order[self.dim - 1]];
        let f_worst = self.values[worst_idx];

        let outcome = if f_reflected < f_best {
            self.try_expansion(config, f_reflected, eval)
        } else if f_reflected < f_second_worst {
            self.replace_worst_with_reflected(f_reflected);
            SimplexStepOutcome::Reflected
        } else {
            self.try_contraction_or_shrink(config, f_reflected, f_worst, worst_start, eval)
        };

        self.steps_taken += 1;
        self.sort_order_in_place();
        self.update_centroid_in_place();
        outcome
    }

    fn try_expansion<F: FnMut(&[f64]) -> f64>(
        &mut self,
        config: &NelderMeadConfig,
        f_reflected: f64,
        eval: &mut F,
    ) -> SimplexStepOutcome {
        for c in 0..self.dim {
            let e = self.centroid[c] + config.expansion * (self.reflected[c] - self.centroid[c]);
            self.expanded[c] = clamp_weight(e, config.weight_bound);
        }
        let f_expanded = eval(&self.expanded);
        let worst_idx = self.order[self.dim];
        let worst_start = vertex_offset(worst_idx, self.dim);
        if f_expanded < f_reflected {
            self.vertices[worst_start..worst_start + self.dim].copy_from_slice(&self.expanded);
            self.values[worst_idx] = f_expanded;
            SimplexStepOutcome::Expanded
        } else {
            self.vertices[worst_start..worst_start + self.dim].copy_from_slice(&self.reflected);
            self.values[worst_idx] = f_reflected;
            SimplexStepOutcome::Reflected
        }
    }

    fn try_contraction_or_shrink<F: FnMut(&[f64]) -> f64>(
        &mut self,
        config: &NelderMeadConfig,
        f_reflected: f64,
        f_worst: f64,
        worst_start: usize,
        eval: &mut F,
    ) -> SimplexStepOutcome {
        if f_reflected < f_worst {
            for c in 0..self.dim {
                let oc =
                    self.centroid[c] + config.contraction * (self.reflected[c] - self.centroid[c]);
                self.contracted[c] = clamp_weight(oc, config.weight_bound);
            }
            let f_contracted = eval(&self.contracted);
            if f_contracted <= f_reflected {
                let worst_idx = self.order[self.dim];
                self.vertices[worst_start..worst_start + self.dim]
                    .copy_from_slice(&self.contracted);
                self.values[worst_idx] = f_contracted;
                return SimplexStepOutcome::ContractedOutside;
            }
        } else {
            for c in 0..self.dim {
                let ic = self.centroid[c]
                    + config.contraction * (self.vertices[worst_start + c] - self.centroid[c]);
                self.contracted[c] = clamp_weight(ic, config.weight_bound);
            }
            let f_contracted = eval(&self.contracted);
            if f_contracted < f_worst {
                let worst_idx = self.order[self.dim];
                self.vertices[worst_start..worst_start + self.dim]
                    .copy_from_slice(&self.contracted);
                self.values[worst_idx] = f_contracted;
                return SimplexStepOutcome::ContractedInside;
            }
        }
        self.shrink_toward_best(config, eval);
        SimplexStepOutcome::Shrunk
    }

    /// Runs Nelder-Mead iterations in place until `tolerance` or `max_steps` is reached.
    #[verified_engine::verified]
    pub fn optimize_in_place<F: FnMut(&[f64]) -> f64>(
        &mut self,
        config: &NelderMeadConfig,
        eval: &mut F,
    ) -> SimplexOptimizationSummary {
        assert!(config.max_steps >= 1, "max_steps must be at least 1");
        let mut converged = false;
        let mut executed = 0;
        for _ in 0..config.max_steps {
            let outcome = self.step_in_place(config, eval);
            if outcome == SimplexStepOutcome::Converged {
                converged = true;
                break;
            }
            executed += 1;
            if self.value_spread() <= config.tolerance {
                converged = true;
                break;
            }
        }
        SimplexOptimizationSummary {
            steps: executed,
            best_value: self.best_value(),
            converged,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[verified_engine::verified]
    fn test_nelder_mead_quadratic_convergence() {
        let mut simplex = NelderMeadSimplex::new(2);
        let config = NelderMeadConfig {
            tolerance: 1e-9,
            max_steps: 250,
            ..NelderMeadConfig::default()
        };
        let mut eval = |w: &[f64]| (w[0] - 0.6).powi(2) + 2.0 * (w[1] + 0.35).powi(2);
        simplex.initialize_in_place(&[0.0, 0.0], &config, &mut eval);
        let summary = simplex.optimize_in_place(&config, &mut eval);
        assert!(summary.converged);
        assert!(summary.best_value < 1e-7);
        let best = simplex.best_weights();
        assert!((best[0] - 0.6).abs() < 1e-3);
        assert!((best[1] + 0.35).abs() < 1e-3);
    }
}
