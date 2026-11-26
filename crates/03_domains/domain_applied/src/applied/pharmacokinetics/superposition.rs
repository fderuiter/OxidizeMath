use super::traits::PharmacokineticModel;

/// Multi-dose linear pharmacokinetic superposition model.
///
/// $$
/// C(t) = \sum_{k:\, t_k \le t} C_1(t - t_k)
/// $$
///
/// Reference: Bateman (1910), DOI: 10.1017/S0305004100019988;
/// [`docs/math_specs/domain_applied.md`](../../../../../../docs/math_specs/domain_applied.md).
///
/// # Stability
/// Causal summation skips future doses ($t_k > t$) so exponential elimination terms remain bounded in $(0, 1]$.
///
/// # Domain of Validity
/// Valid for linear (dose-proportional, time-invariant) ADME kinetics with finite administration times $t_k$.
#[derive(Debug, Clone)]
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
