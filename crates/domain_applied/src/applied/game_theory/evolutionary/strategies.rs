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
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct MatrixPayoff {
    payoff_matrix: DMatrix<f64>,
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for MatrixPayoff {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct MatrixPayoffRaw {
            payoff_matrix: DMatrix<f64>,
        }
        let raw = MatrixPayoffRaw::deserialize(deserializer)?;
        if raw.payoff_matrix.iter().any(|v| !v.is_finite()) {
            return Err(serde::de::Error::custom(
                "MatrixPayoff entries must be finite",
            ));
        }
        Self::new(raw.payoff_matrix).map_err(|e| serde::de::Error::custom(format!("{e:?}")))
    }
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
