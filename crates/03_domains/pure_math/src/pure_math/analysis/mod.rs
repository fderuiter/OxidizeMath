/// Submodule providing `complex` types, traits, and algorithms.
pub mod complex;
pub mod evolution;
/// Evolutionary game theory (`FitnessStrategy`, `MatrixPayoff`, `ReplicatorDynamics`).
pub mod evolutionary;
pub mod integration;
/// Discrete-time linear and Extended Kalman Filter (`EKF`) state estimation.
pub mod kalman;
pub mod momentum_dynamics;
pub mod ode;
pub mod optimization;
/// Submodule providing `pde` types, traits, and algorithms.
pub mod pde;
pub mod roots;
pub mod stochastic;
/// Submodule providing `transforms` types, traits, and algorithms.
pub mod transforms;

pub use evolution::EvolutionError;
pub use evolutionary::{FitnessStrategy, GameTheoryError, MatrixPayoff, ReplicatorDynamics};
pub use kalman::{KalmanError, KalmanFilter, KalmanFilterBuilder, KalmanModel, KalmanSystem};
pub use optimization::OptimizationError;
pub use roots::AnalysisError;
pub use stochastic::StochasticError;
pub use transforms::TransformError;

// [cite:graph_parameters_rust]
