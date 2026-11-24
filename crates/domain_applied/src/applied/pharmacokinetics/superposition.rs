use super::traits::PharmacokineticModel;

/// A model that calculates the concentration resulting from multiple doses using the principle of superposition.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SuperpositionModel<M> {
    /// The underlying model for a single dose.
    pub base_model: M,
    /// The times at which doses were administered.
    pub dose_times: Vec<f64>,
}

impl<M> SuperpositionModel<M> {
    /// Creates a new `SuperpositionModel`.
    #[verified_engine::verified]
    pub fn new(base_model: M, dose_times: Vec<f64>) -> Self {
        Self {
            base_model,
            dose_times,
        }
    }
}

impl<M: PharmacokineticModel> PharmacokineticModel for SuperpositionModel<M> {
    #[verified_engine::verified]
    fn concentration(&self, t: f64) -> f64 {
        self.dose_times
            .iter()
            .map(|&dose_time| {
                if t >= dose_time {
                    self.base_model.concentration(t - dose_time)
                } else {
                    0.0
                }
            })
            .sum()
    }
}
