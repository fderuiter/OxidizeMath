use super::bateman::BatemanModel;
use super::parameters::PKParameters;
use super::superposition::SuperpositionModel;
use super::traits::PharmacokineticModel;
use super::two_pulse::TwoPulseModel;

use crate::error::PharmacokineticsError;

/// A model for a drug composed of two enantiomers (e.g., d- and l-amphetamine).
///
/// Constructed via [`EnantiomerModel::new`] to ensure valid parameter states.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EnantiomerModel {
    d_model: BatemanModel,
    l_model: BatemanModel,
}

impl EnantiomerModel {
    /// Creates a new `EnantiomerModel` from the given parameters and fractions.
    ///
    /// # Arguments
    /// * `d_params` - Parameters for the d-enantiomer. The dose `d` should be the total dose of the mixture.
    /// * `l_params` - Parameters for the l-enantiomer. The dose `d` should be the total dose of the mixture.
    /// * `f_d` - Fraction of d-enantiomer in the dose (e.g., 0.75 for Adderall).
    /// * `f_l` - Fraction of l-enantiomer in the dose (e.g., 0.25 for Adderall).
    ///
    /// # Returns
    /// - `Ok(EnantiomerModel)` if fractions and resulting doses are valid.
    /// - `Err(PharmacokineticsError)` if any parameter or fraction is invalid.
    #[verified_engine::verified]
    pub fn new(
        d_params: PKParameters,
        l_params: PKParameters,
        f_d: f64,
        f_l: f64,
    ) -> Result<Self, PharmacokineticsError> {
        if !(0.0..=1.0).contains(&f_d) {
            return Err(PharmacokineticsError::InvalidParameter(format!(
                "Fraction of d-enantiomer f_d={} must be between 0.0 and 1.0",
                f_d
            )));
        }
        if !(0.0..=1.0).contains(&f_l) {
            return Err(PharmacokineticsError::InvalidParameter(format!(
                "Fraction of l-enantiomer f_l={} must be between 0.0 and 1.0",
                f_l
            )));
        }

        let d_dose = d_params.d() * f_d;
        let l_dose = l_params.d() * f_l;

        let d_model = BatemanModel::new(d_params.with_dose(d_dose)?);
        let l_model = BatemanModel::new(l_params.with_dose(l_dose)?);

        Ok(Self { d_model, l_model })
    }
}

impl PharmacokineticModel for EnantiomerModel {
    #[verified_engine::verified]
    fn concentration(&self, t: f64) -> f64 {
        self.d_model.concentration(t) + self.l_model.concentration(t)
    }
}

impl EnantiomerModel {
    /// Calculates the total concentration of both enantiomers at time `t` for a single IR dose.
    #[verified_engine::verified]
    pub fn concentration_ir_single_dose(&self, t: f64) -> f64 {
        self.concentration(t)
    }

    /// Calculates the total concentration for multiple IR doses using superposition.
    #[verified_engine::verified]
    pub fn concentration_ir_multiple_doses(&self, dose_times: &[f64], t: f64) -> f64 {
        let model = SuperpositionModel::new(*self, dose_times.to_vec());
        model.concentration(t)
    }

    /// Calculates the total concentration for a single XR dose using the two-pulse model.
    #[verified_engine::verified]
    pub fn concentration_xr_single_dose(&self, lag_time: f64, f1: f64, f2: f64, t: f64) -> f64 {
        let model = TwoPulseModel::new(*self, lag_time, f1, f2);
        model.concentration(t)
    }

    /// Calculates the total concentration for multiple XR doses using superposition.
    #[verified_engine::verified]
    pub fn concentration_xr_multiple_doses(
        &self,
        dose_times: &[f64],
        lag_time: f64,
        f1: f64,
        f2: f64,
        t: f64,
    ) -> f64 {
        let xr_model = TwoPulseModel::new(*self, lag_time, f1, f2);
        let model = SuperpositionModel::new(xr_model, dose_times.to_vec());
        model.concentration(t)
    }
}
