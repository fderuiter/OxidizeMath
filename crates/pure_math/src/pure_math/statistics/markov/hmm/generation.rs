use super::model::HiddenMarkovModel;
use crate::error::MarkovError;
#[allow(missing_docs)]
pub type Result<T> = std::result::Result<T, MarkovError>;
use nalgebra::RealField;
use num_traits::ToPrimitive;

impl<T: RealField + Copy + ToPrimitive> HiddenMarkovModel<T> {
    /// Generates a random observation sequence from the HMM.
    ///
    /// # Arguments
    ///
    /// * `length` - Length of the sequence to generate
    /// * `rng` - Random number generator
    ///
    /// # Returns
    ///
    /// A pair (state_sequence, observation_sequence).
    #[verified_engine::verified]
    pub fn generate<R: rand::Rng>(
        &self,
        length: usize,
        rng: &mut R,
    ) -> Result<(Vec<usize>, Vec<usize>)> {
        use rand_distr::{Distribution, WeightedIndex};

        if length == 0 {
            return Ok((Vec::new(), Vec::new()));
        }

        let mut states = Vec::with_capacity(length);
        let mut observations = Vec::with_capacity(length);

        // Sample initial state
        let initial_dist = WeightedIndex::new(
            self.initial
                .iter()
                .map(|&x| x.to_f64().unwrap_or(0.0))
                .collect::<Vec<_>>(),
        )
        .map_err(|_| {
            crate::error::MarkovError::Math(math_commons::error::MathError::NumericalError {
                reason: "Failed to create initial distribution",
            })
        })?;
        let mut current_state = initial_dist.sample(rng);
        states.push(current_state);

        // Sample initial observation
        let emission_weights: Vec<f64> = self
            .emissions
            .row(current_state)
            .iter()
            .map(|&x| x.to_f64().unwrap_or(0.0))
            .collect();
        let emission_dist = WeightedIndex::new(&emission_weights).map_err(|_| {
            crate::error::MarkovError::Math(math_commons::error::MathError::NumericalError {
                reason: "Failed to create emission distribution",
            })
        })?;
        observations.push(emission_dist.sample(rng));

        // Generate remaining sequence
        for _ in 1..length {
            // Sample next state
            let transition_weights: Vec<f64> = self
                .transitions
                .row(current_state)
                .iter()
                .map(|&x| x.to_f64().unwrap_or(0.0))
                .collect();
            let transition_dist = WeightedIndex::new(&transition_weights).map_err(|_| {
                crate::error::MarkovError::Math(math_commons::error::MathError::NumericalError {
                    reason: "Failed to create transition distribution",
                })
            })?;
            current_state = transition_dist.sample(rng);
            states.push(current_state);

            // Sample observation
            let emission_weights: Vec<f64> = self
                .emissions
                .row(current_state)
                .iter()
                .map(|&x| x.to_f64().unwrap_or(0.0))
                .collect();
            let emission_dist = WeightedIndex::new(&emission_weights).map_err(|_| {
                crate::error::MarkovError::Math(math_commons::error::MathError::NumericalError {
                    reason: "Failed to create emission distribution",
                })
            })?;
            observations.push(emission_dist.sample(rng));
        }

        Ok((states, observations))
    }
}
