// [cite:pure_math]
use super::traits::{FitnessStrategy, GameTheoryError};
use nalgebra::{DMatrix, DVector};

/// Linear bimatrix/symmetric game fitness landscape $\mathbf{f}(\mathbf{x}) = \mathbf{A}\mathbf{x}$.
///
/// Evaluates expected strategy payoffs against population frequencies $\mathbf{x} \in \Delta^{n-1}$:
///
/// $$f_i(\mathbf{x}) = (\mathbf{A}\mathbf{x})_i = \sum_{j=1}^{n} A_{ij} x_j$$
///
/// Reference: Taylor & Jonker (1978), *Math. Biosci.* 40(1–2), 145–156, DOI: 10.1016/0025-5564(78)90077-9; see
/// [`docs/math_specs/pure_math.md`](../../../../../../../docs/math_specs/pure_math.md).
///
/// # Stability
/// Uses `DMatrix::mul_to` in-place matrix-vector multiplication with zero dynamic heap allocation.
///
/// # Domain of Validity
/// Square payoff matrix $\mathbf{A} \in \mathbb{R}^{n \times n}$ with conformable state/output vectors of length $n$.
#[derive(Debug, Clone)]
pub struct MatrixPayoff {
    payoff_matrix: DMatrix<f64>,
}

impl MatrixPayoff {
    /// Creates a validated square [`MatrixPayoff`] landscape.
    ///
    /// # Errors
    /// Returns [`GameTheoryError::NonSquarePayoffMatrix`] if `payoff_matrix` is not square.
    #[verified_engine::verified]
    pub fn new(payoff_matrix: DMatrix<f64>) -> Result<Self, GameTheoryError> {
        if payoff_matrix.nrows() != payoff_matrix.ncols() {
            return Err(GameTheoryError::NonSquarePayoffMatrix {
                rows: payoff_matrix.nrows(),
                cols: payoff_matrix.ncols(),
            });
        }
        Ok(Self { payoff_matrix })
    }

    /// Returns a reference to the underlying $(n \times n)$ payoff matrix $\mathbf{A}$.
    #[must_use]
    #[verified_engine::verified]
    pub fn payoff_matrix(&self) -> &DMatrix<f64> {
        &self.payoff_matrix
    }
}

impl FitnessStrategy for MatrixPayoff {
    #[verified_engine::verified]
    fn fitness(&self, x: &DVector<f64>, out: &mut DVector<f64>) {
        self.payoff_matrix.mul_to(x, out);
    }
}
