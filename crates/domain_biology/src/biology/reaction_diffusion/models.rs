//! Concrete zero-allocation biochemical [`ReactionModel`] implementations coupling
//! cooperative Hill enzyme kinetics with multi-species spatial reaction-diffusion.

use super::state::ChemicalState;
use super::traits::ReactionModel;
use crate::biology::kinetics::{HillKinetics, KineticsError, KineticsModel};

/// Two-species activator-substrate biochemical reaction coupling cooperative [`HillKinetics`]
/// autocatalysis with substrate replenishment and linear clearance.
///
/// Species `0` ($u$, activator/product) and species `1` ($s$, substrate) evolve locally as:
/// $$\frac{\partial u}{\partial t} = s \cdot \frac{V_{\max} u^n}{K_m^n + u^n} - k_{\mathrm{deg}} u + k_{\mathrm{basal}},$$
/// $$\frac{\partial s}{\partial t} = k_{\mathrm{feed}} (s_0 - s) - s \cdot \frac{V_{\max} u^n}{K_m^n + u^n}.$$
#[derive(Debug, Clone, PartialEq)]
pub struct HillSubstrateReaction {
    /// Cooperative Hill enzyme model governing autocatalytic turnover.
    pub hill: HillKinetics,
    /// Ambient substrate reservoir concentration $s_0 > 0$.
    pub substrate_reservoir: f64,
    /// Substrate replenishment rate $k_{\mathrm{feed}} \ge 0$.
    pub feed_rate: f64,
    /// Activator clearance rate $k_{\mathrm{deg}} \ge 0$.
    pub degradation_rate: f64,
    /// Basal constitutive synthesis rate $k_{\mathrm{basal}} \ge 0$.
    pub basal_rate: f64,
}

impl HillSubstrateReaction {
    /// Constructs a validated [`HillSubstrateReaction`] from positive kinetic constants.
    #[verified_engine::verified]
    pub fn new(
        v_max: f64,
        k_m: f64,
        n_hill: f64,
        substrate_reservoir: f64,
        feed_rate: f64,
        degradation_rate: f64,
        basal_rate: f64,
    ) -> Result<Self, KineticsError> {
        if !v_max.is_finite()
            || v_max <= 0.0
            || !k_m.is_finite()
            || k_m <= 0.0
            || !n_hill.is_finite()
            || n_hill <= 0.0
            || !substrate_reservoir.is_finite()
            || substrate_reservoir <= 0.0
            || !feed_rate.is_finite()
            || feed_rate < 0.0
            || !degradation_rate.is_finite()
            || degradation_rate < 0.0
            || !basal_rate.is_finite()
            || basal_rate < 0.0
        {
            return Err(KineticsError::InvalidParameters);
        }
        Ok(Self {
            hill: HillKinetics::new(v_max, k_m, n_hill)?,
            substrate_reservoir,
            feed_rate,
            degradation_rate,
            basal_rate,
        })
    }

    /// Evaluates the cooperative turnover flux $s \cdot V_{\max} u^n / (K_m^n + u^n)$ safely for $u, s \ge 0$.
    #[inline]
    #[verified_engine::verified]
    pub fn turnover_flux(&self, activator: f64, substrate: f64) -> f64 {
        let u_pos = activator.max(0.0);
        let s_pos = substrate.max(0.0);
        let hill_rate = self.hill.reaction_velocity(u_pos).unwrap_or(0.0);
        s_pos * hill_rate
    }
}

impl ReactionModel for HillSubstrateReaction {
    #[verified_engine::verified]
    fn reaction(&self, concentrations: &[f64], rates: &mut [f64]) {
        if concentrations.len() < 2 || rates.len() < 2 {
            return;
        }
        let u = concentrations[0];
        let s = concentrations[1];
        let flux = self.turnover_flux(u, s);
        rates[0] = flux - self.degradation_rate * u.max(0.0) + self.basal_rate;
        rates[1] = self.feed_rate * (self.substrate_reservoir - s) - flux;
    }

    #[verified_engine::verified]
    fn add_reaction_batch(&self, state: &ChemicalState, out_rates: &mut ChemicalState) {
        if state.num_species() < 2 || out_rates.num_species() < 2 {
            return;
        }
        let n_grid = state.grid_size();
        let u_slice = state.species(0);
        let s_slice = state.species(1);
        for i in 0..n_grid {
            let u = u_slice[i];
            let s = s_slice[i];
            let flux = self.turnover_flux(u, s);
            out_rates.species_mut(0)[i] +=
                flux - self.degradation_rate * u.max(0.0) + self.basal_rate;
            out_rates.species_mut(1)[i] += self.feed_rate * (self.substrate_reservoir - s) - flux;
        }
    }

    #[verified_engine::verified]
    fn add_reaction_batch_at(&self, _t: f64, state: &ChemicalState, out_rates: &mut ChemicalState) {
        self.add_reaction_batch(state, out_rates);
    }
}
