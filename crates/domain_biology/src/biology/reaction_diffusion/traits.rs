use super::state::ChemicalState;

const MAX_STACK_SPECIES: usize = 32;

#[verified_engine::verified]
fn accumulate_reaction_grid<F>(state: &ChemicalState, out_rates: &mut ChemicalState, mut react: F)
where
    F: FnMut(&[f64], &mut [f64]),
{
    let n_species = state.num_species();
    if n_species == 0 {
        return;
    }
    let n_grid = state.grid_size();

    if n_species <= MAX_STACK_SPECIES {
        let mut local_concs = [0.0_f64; MAX_STACK_SPECIES];
        let mut local_rates = [0.0_f64; MAX_STACK_SPECIES];
        let concs = &mut local_concs[..n_species];
        let rates = &mut local_rates[..n_species];
        for i in 0..n_grid {
            for (s, conc) in concs.iter_mut().enumerate() {
                *conc = state.species(s)[i];
            }
            rates.fill(0.0);
            react(concs, rates);
            for (s, rate) in rates.iter().enumerate() {
                out_rates.species_mut(s)[i] += rate;
            }
        }
    } else {
        let mut local_concs = vec![0.0; n_species];
        let mut local_rates = vec![0.0; n_species];
        for i in 0..n_grid {
            for (s, conc) in local_concs.iter_mut().enumerate().take(n_species) {
                *conc = state.species(s)[i];
            }
            local_rates.fill(0.0);
            react(&local_concs, &mut local_rates);
            for (s, rate) in local_rates.iter().enumerate().take(n_species) {
                out_rates.species_mut(s)[i] += rate;
            }
        }
    }
}

/// Defines the local reaction kinetics for N species.
pub trait ReactionModel {
    /// Computes the reaction rates for a single spatial point.
    ///
    /// # Arguments
    /// * `concentrations`: The current concentrations of all species at this point.
    /// * `rates`: Output buffer for the computed reaction rates (dC/dt).
    #[verified_engine::verified]
    fn reaction(&self, concentrations: &[f64], rates: &mut [f64]);

    /// Computes time-dependent reaction rates at simulation time `t`.
    ///
    /// Autonomous models delegate to [`Self::reaction`] by default.
    #[verified_engine::verified]
    fn reaction_at(&self, _t: f64, concentrations: &[f64], rates: &mut [f64]) {
        self.reaction(concentrations, rates);
    }

    /// Computes and accumulates reaction rates for the entire grid without heap allocation
    /// when `num_species() <= 32`.
    #[verified_engine::verified]
    fn add_reaction_batch(&self, state: &ChemicalState, out_rates: &mut ChemicalState) {
        accumulate_reaction_grid(state, out_rates, |concs, rates| self.reaction(concs, rates));
    }

    /// Computes and accumulates reaction rates at explicit simulation time `t` for the entire grid.
    #[verified_engine::verified]
    fn add_reaction_batch_at(&self, t: f64, state: &ChemicalState, out_rates: &mut ChemicalState) {
        accumulate_reaction_grid(state, out_rates, |concs, rates| {
            self.reaction_at(t, concs, rates)
        });
    }
}

/// Defines the spatial diffusion strategy for N species.
pub trait DiffusionModel {
    /// Applies the diffusion operator (Laplacian) to the full state.
    ///
    /// # Arguments
    /// * `state`: Current chemical state.
    /// * `out`: Output buffer for the diffusion term (D * Laplacian).
    /// * `coeffs`: Diffusion coefficients for each species.
    #[verified_engine::verified]
    fn apply(&self, state: &ChemicalState, out: &mut ChemicalState, coeffs: &[f64]);
}
