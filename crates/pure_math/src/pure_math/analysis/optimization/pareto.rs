//! Multi-objective Pareto frontier extraction, weighted-sum scalarization sweeps,
//! and $\epsilon$-constraint trade-off solvers.

use super::OptimizationError;
use super::augmented_lagrangian::{
    AugmentedLagrangianConfig, AugmentedLagrangianSolver, AugmentedLagrangianWorkspace,
};
use super::lbfgsb::{LbfgsBConfig, LbfgsBSolver, LbfgsBWorkspace};
use super::projected_gradient::BoxBounds;
use nalgebra::{DMatrix, DVector};

/// Candidate multi-objective point pairing a decision vector with its evaluated objectives
/// (all interpreted in minimization sense).
#[derive(Debug, Clone, PartialEq)]
pub struct ParetoPoint {
    /// Decision variable vector $x \in \mathbb{R}^n$.
    pub decision: DVector<f64>,
    /// Evaluated objective vector $[f_1(x), f_2(x), \dots, f_m(x)]$.
    pub objectives: Vec<f64>,
    /// Scalarization weight $w \in [0, 1]$ or $\epsilon$ bound that generated this point.
    pub parameter: f64,
}

/// Returns `true` iff objective vector `a` Pareto-dominates `b` (minimization sense) within `eps`:
/// $\forall k,\; a_k \le b_k + \epsilon$ and $\exists k,\; a_k < b_k - \epsilon$.
#[verified_engine::verified]
pub fn dominates(a: &[f64], b: &[f64], eps: f64) -> bool {
    if a.len() != b.len() || a.is_empty() {
        return false;
    }
    let mut strictly_better = false;
    for (&ak, &bk) in a.iter().zip(b.iter()) {
        if ak > bk + eps {
            return false;
        }
        if ak < bk - eps {
            strictly_better = true;
        }
    }
    strictly_better
}

/// Preallocated index workspace for zero-allocation non-dominated frontier filtering.
#[derive(Debug, Clone)]
pub struct ParetoWorkspace {
    keep_flags: Vec<bool>,
    frontier_indices: Vec<usize>,
}

impl ParetoWorkspace {
    /// Preallocates non-dominated filtering buffers for up to `max_candidates` rows.
    #[verified_engine::verified]
    pub fn new(max_candidates: usize) -> Self {
        Self {
            keep_flags: vec![true; max_candidates],
            frontier_indices: Vec::with_capacity(max_candidates),
        }
    }

    /// Extracts row indices of non-dominated points from `objectives` (`num_candidates x num_obj`),
    /// sorted ascending by objective column 0, with zero heap allocations.
    #[verified_engine::verified]
    pub fn extract_indices_in_place(&mut self, objectives: &DMatrix<f64>, eps: f64) -> &[usize] {
        let count = objectives.nrows();
        let m = objectives.ncols();
        assert!(count <= self.keep_flags.len());
        for flag in &mut self.keep_flags[..count] {
            *flag = true;
        }
        for i in 0..count {
            if !self.keep_flags[i] {
                continue;
            }
            for j in 0..count {
                if i == j || !self.keep_flags[j] {
                    continue;
                }
                let mut le_all = true;
                let mut lt_any = false;
                for c in 0..m {
                    let vj = objectives[(j, c)];
                    let vi = objectives[(i, c)];
                    if vj > vi + eps {
                        le_all = false;
                        break;
                    }
                    if vj < vi - eps {
                        lt_any = true;
                    }
                }
                if le_all && lt_any {
                    self.keep_flags[i] = false;
                    break;
                }
            }
        }
        self.frontier_indices.clear();
        for i in 0..count {
            if self.keep_flags[i] {
                // Deduplicate nearly identical objective vectors:
                let dup = self.frontier_indices.iter().any(|&prev| {
                    (0..m).all(|c| (objectives[(prev, c)] - objectives[(i, c)]).abs() <= eps)
                });
                if !dup {
                    self.frontier_indices.push(i);
                }
            }
        }
        // Insertion-sort frontier indices by column 0 ascending:
        for k in 1..self.frontier_indices.len() {
            let key = self.frontier_indices[k];
            let val = objectives[(key, 0)];
            let mut p = k;
            while p > 0 && objectives[(self.frontier_indices[p - 1], 0)] > val {
                self.frontier_indices[p] = self.frontier_indices[p - 1];
                p -= 1;
            }
            self.frontier_indices[p] = key;
        }
        &self.frontier_indices
    }
}

/// Filters `candidates` to the non-dominated Pareto frontier sorted by `objectives[0]` ascending.
#[verified_engine::verified]
pub fn extract_pareto_frontier(candidates: &[ParetoPoint], eps: f64) -> Vec<ParetoPoint> {
    if candidates.is_empty() {
        return Vec::new();
    }
    let m = candidates[0].objectives.len();
    let mut mat = DMatrix::zeros(candidates.len(), m);
    for (r, pt) in candidates.iter().enumerate() {
        for (c, &v) in pt.objectives.iter().enumerate() {
            mat[(r, c)] = v;
        }
    }
    let mut ws = ParetoWorkspace::new(candidates.len());
    let indices = ws.extract_indices_in_place(&mat, eps);
    indices.iter().map(|&idx| candidates[idx].clone()).collect()
}

/// Sweeps `num_weights` convex combinations $w f_1(x) + (1 - w) f_2(x)$ over `bounds`
/// using a single preallocated `L-BFGS-B` workspace and returns `(sampled_points, pareto_frontier)`.
#[verified_engine::verified]
pub fn weighted_sum_sweep<F>(
    mut eval_biobjective: F,
    bounds: &BoxBounds,
    x0: &DVector<f64>,
    num_weights: usize,
) -> Result<(Vec<ParetoPoint>, Vec<ParetoPoint>), OptimizationError>
where
    F: FnMut(&DVector<f64>, &mut DVector<f64>, &mut DVector<f64>) -> (f64, f64),
{
    if num_weights == 0 {
        return Err(OptimizationError::InvalidParameter(
            "num_weights must be >= 1",
        ));
    }
    let n = x0.len();
    let solver = LbfgsBSolver::new(LbfgsBConfig {
        memory: 8,
        max_iterations: 150,
        tolerance: 1e-7,
        max_line_search: 24,
    });
    let mut ws = LbfgsBWorkspace::new(n, 8, 150);
    let mut g1 = DVector::zeros(n);
    let mut g2 = DVector::zeros(n);
    let mut x = x0.clone();
    let mut all_points = Vec::with_capacity(num_weights);

    for idx in 0..num_weights {
        let w = if num_weights == 1 {
            0.5
        } else {
            (idx as f64) / ((num_weights - 1) as f64)
        };
        let _ = solver.solve_in_place(
            |xk, grad_out| {
                let (f1, f2) = eval_biobjective(xk, &mut g1, &mut g2);
                for j in 0..n {
                    grad_out[j] = w * g1[j] + (1.0 - w) * g2[j];
                }
                w * f1 + (1.0 - w) * f2
            },
            bounds,
            &mut x,
            &mut ws,
        )?;
        let (f1, f2) = eval_biobjective(&x, &mut g1, &mut g2);
        all_points.push(ParetoPoint {
            decision: x.clone(),
            objectives: vec![f1, f2],
            parameter: w,
        });
    }
    let frontier = extract_pareto_frontier(&all_points, 1e-6);
    Ok((all_points, frontier))
}

/// Sweeps `num_steps` upper bounds $\epsilon \in [\epsilon_{\min}, \epsilon_{\max}]$ solving
/// $\min_{x \in [l, u]} f_1(x)$ subject to $f_2(x) \le \epsilon$ via [`AugmentedLagrangianSolver`],
/// recovering both convex and non-convex Pareto fronts.
#[verified_engine::verified]
pub fn epsilon_constraint_sweep<F>(
    mut eval_biobjective: F,
    bounds: &BoxBounds,
    x0: &DVector<f64>,
    eps_range: (f64, f64),
    num_steps: usize,
) -> Result<(Vec<ParetoPoint>, Vec<ParetoPoint>), OptimizationError>
where
    F: FnMut(&DVector<f64>, &mut DVector<f64>, &mut DVector<f64>) -> (f64, f64),
{
    let (eps_min, eps_max) = eps_range;
    if num_steps == 0 || !eps_min.is_finite() || !eps_max.is_finite() || eps_min > eps_max {
        return Err(OptimizationError::InvalidParameter(
            "epsilon_constraint_sweep requires valid eps_range and num_steps >= 1",
        ));
    }
    let n = x0.len();
    let solver = AugmentedLagrangianSolver::new(AugmentedLagrangianConfig {
        max_outer_iterations: 20,
        max_inner_iterations: 80,
        constraint_tolerance: 1e-5,
        gradient_tolerance: 1e-6,
        ..AugmentedLagrangianConfig::default()
    });
    let mut ws = AugmentedLagrangianWorkspace::new(n, 0, 1, 80);
    let mut g1_buf = DVector::zeros(n);
    let mut g2_buf = DVector::zeros(n);
    let mut x = x0.clone();
    let mut candidates = Vec::with_capacity(num_steps);

    for k in 0..num_steps {
        let eps = if num_steps == 1 {
            0.5 * (eps_min + eps_max)
        } else {
            eps_max - (k as f64) * (eps_max - eps_min) / ((num_steps - 1) as f64)
        };
        let _ = solver.solve_in_place(
            |xk: &DVector<f64>,
             gf: &mut DVector<f64>,
             _h: &mut DVector<f64>,
             _jh: &mut DMatrix<f64>,
             g: &mut DVector<f64>,
             jg: &mut DMatrix<f64>| {
                let (f1, f2) = eval_biobjective(xk, gf, &mut g2_buf);
                g[0] = f2 - eps;
                for j in 0..n {
                    jg[(0, j)] = g2_buf[j];
                }
                f1
            },
            bounds,
            &mut x,
            &mut ws,
        )?;
        let (_, viol) = ws.summary();
        let (f1, f2) = eval_biobjective(&x, &mut g1_buf, &mut g2_buf);
        if viol <= 1e-3 && f2 <= eps + 1e-3 {
            candidates.push(ParetoPoint {
                decision: x.clone(),
                objectives: vec![f1, f2],
                parameter: eps,
            });
        }
    }
    let frontier = extract_pareto_frontier(&candidates, 1e-5);
    Ok((candidates, frontier))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pareto_extraction_and_sweeps() {
        let bounds = BoxBounds::uniform(1, 0.0, 2.0).expect("bounds");
        let x0 = DVector::from_vec(vec![1.0]);
        // f1(x) = x^2, f2(x) = (x - 2)^2 on [0, 2]
        let (_, ws_front) = weighted_sum_sweep(
            |x, g1, g2| {
                g1[0] = 2.0 * x[0];
                g2[0] = 2.0 * (x[0] - 2.0);
                (x[0] * x[0], (x[0] - 2.0).powi(2))
            },
            &bounds,
            &x0,
            9,
        )
        .expect("weighted sweep");
        assert!(ws_front.len() >= 7);

        let (_, eps_front) = epsilon_constraint_sweep(
            |x, g1, g2| {
                g1[0] = 2.0 * x[0];
                g2[0] = 2.0 * (x[0] - 2.0);
                (x[0] * x[0], (x[0] - 2.0).powi(2))
            },
            &bounds,
            &x0,
            (0.0, 4.0),
            7,
        )
        .expect("epsilon sweep");
        assert!(eps_front.len() >= 5);
    }
}
