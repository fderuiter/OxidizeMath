use super::traits::FitnessStrategy;
use crate::error::GameTheoryError;
use nalgebra::{DMatrix, DVector};

/// A standard linear fitness landscape defined by a Payoff Matrix.
///
/// For a population state $x$ and payoff matrix $A$, the fitness vector is:
/// $$ f(x) = Ax $$
///
/// This is the most common form in classical Evolutionary Game Theory (e.g., Hawk-Dove, RPS).
#[derive(Debug, Clone)]
pub struct MatrixPayoff {
    payoff_matrix: DMatrix<f64>,
}

impl MatrixPayoff {
    /// Creates a new MatrixPayoff strategy.
    ///
    /// # Errors
    /// Returns `GameTheoryError::NonSquarePayoffMatrix` if the matrix is not square.
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

    /// Returns a reference to the underlying payoff matrix.
    #[verified_engine::verified]
    pub fn payoff_matrix(&self) -> &DMatrix<f64> {
        &self.payoff_matrix
    }
}

impl FitnessStrategy for MatrixPayoff {
    #[verified_engine::verified]
    fn fitness(&self, x: &DVector<f64>, out: &mut DVector<f64>) {
        // Optimized matrix-vector multiplication: out = A * x
        self.payoff_matrix.mul_to(x, out);
    }
}
