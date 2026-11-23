use super::model::HiddenMarkovModel;
use crate::error::MarkovError;
#[allow(missing_docs)]
pub type Result<T> = std::result::Result<T, MarkovError>;
use engine_prob::Categorical;
use nalgebra::RealField;
use num_traits::ToPrimitive;

fn sample_weights<R: rand::RngCore + ?Sized>(
    weights: &[f64],
    label: &str,
    rng: &mut R,
) -> Result<usize> {
    let dist = Categorical::new(weights).map_err(|_| {
        MarkovError::Math(math_commons::error::MathError::NumericalError {
            reason: format!("Failed to create {label} distribution"),
        })
    })?;
    Ok(dist.sample_index(rng))
}

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
        if length == 0 {
            return Ok((Vec::new(), Vec::new()));
        }

        let mut states = Vec::with_capacity(length);
        let mut observations = Vec::with_capacity(length);

        let initial_weights: Vec<f64> = self
            .initial
            .iter()
            .map(|&x| x.to_f64().unwrap_or(0.0))
            .collect();
        let mut current_state = sample_weights(&initial_weights, "initial", rng)?;
        states.push(current_state);

        let emission_weights: Vec<f64> = self
            .emissions
            .row(current_state)
            .iter()
            .map(|&x| x.to_f64().unwrap_or(0.0))
            .collect();
        observations.push(sample_weights(&emission_weights, "emission", rng)?);

        for _ in 1..length {
            let transition_weights: Vec<f64> = self
                .transitions
                .row(current_state)
                .iter()
                .map(|&x| x.to_f64().unwrap_or(0.0))
                .collect();
            current_state = sample_weights(&transition_weights, "transition", rng)?;
            states.push(current_state);

            let emission_weights: Vec<f64> = self
                .emissions
                .row(current_state)
                .iter()
                .map(|&x| x.to_f64().unwrap_or(0.0))
                .collect();
            observations.push(sample_weights(&emission_weights, "emission", rng)?);
        }

        Ok((states, observations))
    }
}
