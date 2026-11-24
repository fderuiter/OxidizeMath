//! Ohtsuki-Nowak Replicator Dynamics on Graphs.
//!
//! Implements pair-approximation fitness transformation $A_{\text{eff}} = A + B(A, k)$
//! for evolutionary games on contact graphs of mean active degree $k$, following
//! Ohtsuki & Nowak (2006), *Nature* 441, 502–505.

use super::traits::FitnessStrategy;
use crate::error::GameTheoryError;
use nalgebra::{DMatrix, DVector};
use verified_engine::Theory;

/// Minimum clamped active degree above `2.0` used in the Ohtsuki-Nowak denominator $(k - 2)$
/// so sparse paths, dyads, and cycles remain continuous and finite.
pub const MIN_REGULARIZED_GRAPH_DEGREE: f64 = 2.05;

/// Microscopic replacement rule governing Ohtsuki-Nowak pair-approximation assortment $B(A, k)$.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum GraphUpdateRule {
    /// Death-Birth updating: a random focal individual dies and neighbors compete proportional to fitness ($b/c > k$).
    #[default]
    DeathBirth,
    /// Birth-Death updating: an individual reproduces proportional to fitness into a random neighbor.
    BirthDeath,
    /// Imitation updating: a focal individual compares its own fitness with neighbors ($b/c > k + 2$).
    Imitation,
}

/// Computes the mean active degree across nodes with at least one incident CSR edge.
///
/// Returns `0.0` when `n_nodes == 0` or every node is isolated.
#[must_use]
#[verified_engine::verified]
pub fn mean_active_degree_from_csr(n_nodes: usize, row_ptr: &[usize]) -> f64 {
    if n_nodes == 0 || row_ptr.len() < n_nodes + 1 {
        return 0.0;
    }
    let mut active_nodes = 0usize;
    let mut degree_sum = 0usize;
    for i in 0..n_nodes {
        let deg = row_ptr[i + 1].saturating_sub(row_ptr[i]);
        if deg > 0 {
            active_nodes += 1;
            degree_sum += deg;
        }
    }
    if active_nodes == 0 {
        0.0
    } else {
        (degree_sum as f64) / (active_nodes as f64)
    }
}

#[verified_engine::verified]
fn build_ohtsuki_nowak_matrix(
    a: &DMatrix<f64>,
    mean_active_degree: f64,
    rule: GraphUpdateRule,
) -> DMatrix<f64> {
    let n = a.nrows();
    if !mean_active_degree.is_finite() || mean_active_degree <= 0.0 || n == 0 {
        return a.clone();
    }
    let k = mean_active_degree.max(MIN_REGULARIZED_GRAPH_DEGREE);
    let mut eff = a.clone();
    for i in 0..n {
        let a_ii = a[(i, i)];
        for j in 0..n {
            let a_ij = a[(i, j)];
            let a_ji = a[(j, i)];
            let a_jj = a[(j, j)];
            let b_ij = match rule {
                GraphUpdateRule::DeathBirth => {
                    ((k + 1.0) * (a_ii - a_jj) + a_ij - a_ji) / ((k + 1.0) * (k - 2.0))
                }
                GraphUpdateRule::BirthDeath => (a_ii + a_ij - a_ji - a_jj) / (k - 2.0),
                GraphUpdateRule::Imitation => {
                    ((k + 3.0) * (a_ii - a_jj) + 3.0 * (a_ij - a_ji)) / ((k + 3.0) * (k - 2.0))
                }
            };
            eff[(i, j)] = a_ij + b_ij;
        }
    }
    eff
}

/// Pair-approximation fitness landscape $f(x) = (A + B(A, k))x$ for replicator dynamics on graphs.
///
/// Precomputes the Ohtsuki-Nowak effective payoff matrix $A_{\text{eff}} = A + B(A, k)$ at construction
/// so [`FitnessStrategy::fitness`] executes a single zero-allocation `mul_to` matrix-vector product.
///
/// Reference: Ohtsuki & Nowak (2006), *J. Theor. Biol.* 243(1), 86–97, DOI: 10.1016/j.jtbi.2006.06.004; Ohtsuki et al. (2006), *Nature* 441, 502–505, DOI: 10.1038/nature04605; [`docs/math_specs/domain_applied.md`](../../../../../../docs/math_specs/domain_applied.md).
///
/// # Stability
/// Clamps active degree $k \le 2$ away from the pair-approximation pole $(k - 2)^{-1}$ so `effective_payoff` remains finite for sparse or degenerate graphs.
///
/// # Domain of Validity
/// Requires a square payoff matrix $A \in \mathbb{R}^{n \times n}$ with finite entries and finite mean active degree $k > 2$ for exact pair-approximation validity.
///
/// # Examples
///
/// ```
/// # use domain_applied::__doctest_facade as math_explorer;
/// use math_explorer::applied::game_theory::evolutionary::{
///     FitnessStrategy, GraphReplicatorFitness, GraphUpdateRule,
/// };
/// use math_explorer::nalgebra::{DMatrix, DVector};
///
/// // Donation Prisoner's Dilemma with benefit b = 4.0, cost c = 1.0 on degree k = 3.0 (< b/c):
/// let payoff = DMatrix::from_row_slice(2, 2, &[3.0, -1.0, 4.0, 0.0]);
/// let fit = GraphReplicatorFitness::new(payoff, 3.0, GraphUpdateRule::DeathBirth).unwrap();
/// let x = DVector::from_vec(vec![0.5, 0.5]);
/// let mut out = DVector::zeros(2);
/// fit.fitness(&x, &mut out);
/// // Because b/c = 4 > k = 3 under Death-Birth updating, Cooperators outperform Defectors!
/// assert!(out[0] > out[1]);
/// ```
#[derive(Debug, Clone, Theory)]
#[theory(
    description = "Ohtsuki-Nowak pair-approximation replicator equation transforming payoff matrix A into A + B(A, k) on graphs of mean active degree k.",
    citation = "The replicator equation on graphs (Ohtsuki & Nowak, Nature 441, 502-505, 2006)"
)]
pub struct GraphReplicatorFitness {
    base_payoff: DMatrix<f64>,
    effective_payoff: DMatrix<f64>,
    mean_active_degree: f64,
    rule: GraphUpdateRule,
}

impl GraphReplicatorFitness {
    /// Creates a new [`GraphReplicatorFitness`] from a square payoff matrix and mean active degree $k$.
    ///
    /// # Errors
    /// Returns [`GameTheoryError::NonSquarePayoffMatrix`] if `payoff_matrix` is not square.
    #[verified_engine::verified]
    pub fn new(
        payoff_matrix: DMatrix<f64>,
        mean_active_degree: f64,
        rule: GraphUpdateRule,
    ) -> Result<Self, GameTheoryError> {
        if payoff_matrix.nrows() != payoff_matrix.ncols() {
            return Err(GameTheoryError::NonSquarePayoffMatrix {
                rows: payoff_matrix.nrows(),
                cols: payoff_matrix.ncols(),
            });
        }
        let sanitized_k = if mean_active_degree.is_finite() && mean_active_degree > 0.0 {
            mean_active_degree
        } else {
            0.0
        };
        let effective_payoff = build_ohtsuki_nowak_matrix(&payoff_matrix, sanitized_k, rule);
        Ok(Self {
            base_payoff: payoff_matrix,
            effective_payoff,
            mean_active_degree: sanitized_k,
            rule,
        })
    }

    /// Constructs [`GraphReplicatorFitness`] directly from a CSR `row_ptr` slice.
    ///
    /// # Errors
    /// Returns [`GameTheoryError::NonSquarePayoffMatrix`] if `payoff_matrix` is not square.
    #[verified_engine::verified]
    pub fn from_csr(
        payoff_matrix: DMatrix<f64>,
        n_nodes: usize,
        row_ptr: &[usize],
        rule: GraphUpdateRule,
    ) -> Result<Self, GameTheoryError> {
        let k = mean_active_degree_from_csr(n_nodes, row_ptr);
        Self::new(payoff_matrix, k, rule)
    }

    /// Updates the graph's mean active degree $k$ and recomputes $A + B(A, k)$.
    #[verified_engine::verified]
    pub fn set_mean_active_degree(&mut self, mean_active_degree: f64) {
        let sanitized_k = if mean_active_degree.is_finite() && mean_active_degree > 0.0 {
            mean_active_degree
        } else {
            0.0
        };
        self.mean_active_degree = sanitized_k;
        self.effective_payoff =
            build_ohtsuki_nowak_matrix(&self.base_payoff, sanitized_k, self.rule);
    }

    /// Returns the underlying well-mixed payoff matrix $A$.
    #[must_use]
    #[verified_engine::verified]
    pub fn base_payoff_matrix(&self) -> &DMatrix<f64> {
        &self.base_payoff
    }

    /// Returns the Ohtsuki-Nowak effective payoff matrix $A + B(A, k)$.
    #[must_use]
    #[verified_engine::verified]
    pub fn effective_payoff_matrix(&self) -> &DMatrix<f64> {
        &self.effective_payoff
    }

    /// Returns the stored mean active degree $k$.
    #[must_use]
    #[verified_engine::verified]
    pub fn mean_active_degree(&self) -> f64 {
        self.mean_active_degree
    }

    /// Returns the microscopic replacement rule.
    #[must_use]
    #[verified_engine::verified]
    pub fn update_rule(&self) -> GraphUpdateRule {
        self.rule
    }
}

impl FitnessStrategy for GraphReplicatorFitness {
    #[verified_engine::verified]
    fn fitness(&self, x: &DVector<f64>, out: &mut DVector<f64>) {
        self.effective_payoff.mul_to(x, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use verified_engine::engine::VerifiedEngine;

    #[test]
    fn test_ohtsuki_nowak_b_over_c_threshold_and_edgeless_fallback() {
        // Donation game: b = 4, c = 1 => A = [[3, -1], [4, 0]]
        let payoff = DMatrix::from_row_slice(2, 2, &[3.0, -1.0, 4.0, 0.0]);
        let x = DVector::from_vec(vec![0.5, 0.5]);
        let mut out = DVector::zeros(2);

        // Sparse degree k = 3 < b/c = 4 => Cooperators win under Death-Birth
        let sparse =
            GraphReplicatorFitness::new(payoff.clone(), 3.0, GraphUpdateRule::DeathBirth).unwrap();
        sparse.fitness(&x, &mut out);
        assert!(out[0] > out[1], "Expected f_C > f_D at k=3, got {out:?}");

        // Dense degree k = 8 > b/c = 4 => Defectors win under Death-Birth
        let dense =
            GraphReplicatorFitness::new(payoff.clone(), 8.0, GraphUpdateRule::DeathBirth).unwrap();
        dense.fitness(&x, &mut out);
        assert!(out[1] > out[0], "Expected f_D > f_C at k=8, got {out:?}");

        // Edgeless graph k = 0 => reduces to well-mixed A * x
        let edgeless =
            GraphReplicatorFitness::new(payoff, 0.0, GraphUpdateRule::DeathBirth).unwrap();
        edgeless.fitness(&x, &mut out);
        assert!((out[0] - 1.0).abs() < 1e-12);
        assert!((out[1] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_graph_replicator_fitness_zero_alloc_1000_steps() {
        let payoff = DMatrix::from_row_slice(2, 2, &[3.0, -1.0, 4.0, 0.0]);
        let strategy =
            GraphReplicatorFitness::new(payoff, 3.2, GraphUpdateRule::DeathBirth).unwrap();
        let x = DVector::from_vec(vec![0.4, 0.6]);
        let mut out = DVector::zeros(2);

        let (diff, metrics) = VerifiedEngine::run_verified(|| {
            for _ in 0..1_000 {
                strategy.fitness(&x, &mut out);
            }
            out[0] - out[1]
        });
        assert!(diff > 0.0);
        assert!(metrics.function_calls >= 1_000);
    }
}
