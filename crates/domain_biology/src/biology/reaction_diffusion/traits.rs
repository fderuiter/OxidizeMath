use super::state::ChemicalState;

/// Defines the local pointwise reaction source term $\mathbf{R}(\mathbf{C})$ for $N$ species:
/// $$\left(\frac{\partial C_s}{\partial t}\right)_{\text{react}} = R_s(C_0, \dots, C_{N-1})$$
///
/// Reference: Turing (1952), DOI: `10.1098/rstb.1952.0012`; see
/// [`docs/math_specs/domain_biology.md`](../../../../../docs/math_specs/domain_biology.md).
///
/// # Stability
/// Local Lipschitz stiffness $\|\partial \mathbf{R}/\partial \mathbf{C}\|$ bounds explicit time steps
/// $\Delta t$ alongside the spatial diffusion CFL condition.
///
/// # Domain of Validity
/// Finite species concentrations $C_s \in \mathbb{R}$ (typically $C_s \ge 0$) with slice length $N$.
pub trait ReactionModel {
    /// Computes the reaction rates for a single spatial point.
    ///
    /// # Arguments
    /// * `concentrations`: The current concentrations of all species at this point.
    /// * `rates`: Output buffer for the computed reaction rates (dC/dt).
    #[verified_engine::verified]
    fn reaction(&self, concentrations: &[f64], rates: &mut [f64]);

    /// Computes and accumulates reaction rates for the entire grid.
    ///
    /// This method allows for vectorized implementations that process multiple grid points
    /// efficiently. The default implementation iterates over grid points and calls `reaction`.
    ///
    /// # Arguments
    /// * `state`: Current chemical state (concentrations).
    /// * `out_rates`: Chemical state buffer to accumulate reaction rates into.
    #[verified_engine::verified]
    fn add_reaction_batch(&self, state: &ChemicalState, out_rates: &mut ChemicalState) {
        let n_species = state.num_species();
        if n_species == 0 {
            return;
        }
        let n_grid = state.grid_size();

        let mut local_concs = vec![0.0; n_species];
        let mut local_rates = vec![0.0; n_species];

        for i in 0..n_grid {
            // Gather
            for (s, conc) in local_concs.iter_mut().enumerate().take(n_species) {
                *conc = state.species(s)[i];
            }

            self.reaction(&local_concs, &mut local_rates);

            // Scatter (Accumulate)
            for (s, rate) in local_rates.iter().enumerate().take(n_species) {
                out_rates.species_mut(s)[i] += rate;
            }
        }
    }
}

/// Defines the spatial diffusion operator $D_s \nabla^2 C_s$ for $N$ species:
/// $$\left(\frac{\partial C_s}{\partial t}\right)_{\text{diff}} = D_s \nabla^2 C_s$$
///
/// Reference: Turing (1952), DOI: `10.1098/rstb.1952.0012`; see
/// [`docs/math_specs/domain_biology.md`](../../../../../docs/math_specs/domain_biology.md).
///
/// # Stability
/// Second-order central difference stencils yield negative-semidefinite discrete Laplacians requiring
/// $\Delta t \le h^2 / (2 d \max_s D_s)$ under explicit forward-Euler integration.
///
/// # Domain of Validity
/// Non-negative diffusivities $D_s \ge 0$ (`coeffs.len() == state.num_species()`) on matching spatial grids.
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
