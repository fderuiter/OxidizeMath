use super::traits::PharmacokineticModel;

/// A model representing an Extended Release (XR) formulation using a two-pulse approach.
/// C_XR(t) = f1 * C_IR(t) + f2 * C_IR(t - lag_time)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TwoPulseModel<M> {
    /// The base (Immediate Release) model.
    pub base_model: M,
    /// The time delay for the second pulse.
    pub lag_time: f64,
    /// The fraction of the dose in the first pulse.
    pub f1: f64,
    /// The fraction of the dose in the second pulse.
    pub f2: f64,
}

impl<M> TwoPulseModel<M> {
    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn new(base_model: M, lag_time: f64, f1: f64, f2: f64) -> Self {
        Self {
            base_model,
            lag_time,
            f1,
            f2,
        }
    }
}

impl<M: PharmacokineticModel> PharmacokineticModel for TwoPulseModel<M> {
    #[verified_engine::verified]
    fn concentration(&self, t: f64) -> f64 {
        let c1 = self.base_model.concentration(t);
        let c2 = if t >= self.lag_time {
            self.base_model.concentration(t - self.lag_time)
        } else {
            0.0
        };
        self.f1 * c1 + self.f2 * c2
    }
}
