use nalgebra::DVector;

/// Defines the strategy for calculating the fitness of a population.
///
/// In Evolutionary Game Theory, the fitness of a strategy depends on the composition
/// of the population. This trait abstracts the fitness calculation, allowing for
/// both linear (matrix-based) and non-linear (frequency-dependent) games.
pub trait FitnessStrategy {
    /// Computes the fitness vector $f(x)$ for the population state $x$.
    ///
    /// # Arguments
    /// * `x` - The current population state (proportions of each strategy).
    /// * `out` - The output vector to store the calculated fitness values.
    #[verified_engine::verified]
    fn fitness(&self, x: &DVector<f64>, out: &mut DVector<f64>);
}
