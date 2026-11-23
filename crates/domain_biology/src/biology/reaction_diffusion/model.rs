use pure_math::pure_math::analysis::ode::solvers::Euler;
use pure_math::pure_math::analysis::ode::traits::{OdeSystem, Solver};
use verified_engine::Theory;

use super::{ChemicalState, DiffusionModel, ReactionDiffusionError, ReactionModel};

/// A pure physics definition of a Reaction-Diffusion system.
///
/// This struct encapsulates the immutable laws of the system (reaction kinetics, diffusion strategy,
/// and coefficients), separating them from the mutable simulation state.
///
/// It implements `OdeSystem` to calculate the time derivative ($dC/dt$) given a state.
#[derive(Debug, Clone, PartialEq, Theory)]
#[theory(
    description = "Turing patterns arise from reaction-diffusion systems where local auto-activation is coupled with long-range inhibition, leading to spontaneous symmetry breaking and the formation of stable spatial patterns.",
    citation = "The Chemical Basis of Morphogenesis (Turing, 1952)"
)]
pub struct ReactionDiffusionModel<R, D> {
    #[allow(missing_docs)]
    pub reaction: R,
    #[allow(missing_docs)]
    pub diffusion: D,
    #[allow(missing_docs)]
    pub diffusion_coeffs: Vec<f64>,
}

impl<R: ReactionModel, D: DiffusionModel> OdeSystem<ChemicalState>
    for ReactionDiffusionModel<R, D>
{
    #[verified_engine::verified]
    fn derivative(&self, t: f64, state: &ChemicalState) -> ChemicalState {
        let mut out = ChemicalState::new(state.num_species(), state.grid_size());
        self.derivative_in_place(t, state, &mut out);
        out
    }

    #[verified_engine::verified]
    fn derivative_in_place(&self, t: f64, state: &ChemicalState, out: &mut ChemicalState) {
        // Compute Diffusion
        self.diffusion.apply(state, out, &self.diffusion_coeffs);

        // Add Reaction
        self.reaction.add_reaction_batch_at(t, state, out);
    }
}

/// A generic Reaction-Diffusion system for N species.
///
/// This struct manages the simulation state and the integration strategy.
/// By default, it uses the Forward Euler method, but can be configured with other solvers.
#[derive(Debug, Clone, PartialEq)]
pub struct ReactionDiffusionSystem<
    R: ReactionModel,
    D: DiffusionModel,
    S: Solver<ChemicalState> = Euler<ChemicalState>,
> {
    #[allow(missing_docs)]
    pub model: ReactionDiffusionModel<R, D>,
    #[allow(missing_docs)]
    pub state: ChemicalState,
    #[allow(missing_docs)]
    pub solver: S,
    time: f64,
}

/// Builder for `ReactionDiffusionSystem`.
///
/// Ensures that parameters are physically valid before creating the system.
#[derive(Debug, Clone, PartialEq)]
pub struct ReactionDiffusionSystemBuilder<R, D, S> {
    num_species: Option<usize>,
    grid_size: Option<usize>,
    reaction: Option<R>,
    diffusion: Option<D>,
    diffusion_coeffs: Option<Vec<f64>>,
    solver: Option<S>,
}

impl<R, D, S> Default for ReactionDiffusionSystemBuilder<R, D, S> {
    #[verified_engine::verified]
    fn default() -> Self {
        Self {
            num_species: None,
            grid_size: None,
            reaction: None,
            diffusion: None,
            diffusion_coeffs: None,
            solver: None,
        }
    }
}

impl<R: ReactionModel, D: DiffusionModel>
    ReactionDiffusionSystemBuilder<R, D, Euler<ChemicalState>>
{
    /// Starts a new builder with default type parameters.
    #[verified_engine::verified]
    pub fn new() -> Self {
        Self::default()
    }
}

impl<R: ReactionModel, D: DiffusionModel, S: Solver<ChemicalState>>
    ReactionDiffusionSystemBuilder<R, D, S>
{
    /// Sets the number of chemical species.
    #[verified_engine::verified]
    pub fn num_species(mut self, num_species: usize) -> Self {
        self.num_species = Some(num_species);
        self
    }

    /// Sets the spatial grid size.
    #[verified_engine::verified]
    pub fn grid_size(mut self, grid_size: usize) -> Self {
        self.grid_size = Some(grid_size);
        self
    }

    /// Sets the reaction kinetics model.
    #[verified_engine::verified]
    pub fn reaction(mut self, reaction: R) -> Self {
        self.reaction = Some(reaction);
        self
    }

    /// Sets the spatial diffusion model.
    #[verified_engine::verified]
    pub fn diffusion(mut self, diffusion: D) -> Self {
        self.diffusion = Some(diffusion);
        self
    }

    /// Sets the diffusion coefficients for each species.
    #[verified_engine::verified]
    pub fn diffusion_coeffs(mut self, coeffs: Vec<f64>) -> Self {
        self.diffusion_coeffs = Some(coeffs);
        self
    }

    /// Sets a custom solver strategy.
    pub fn solver<NewS: Solver<ChemicalState>>(
        self,
        solver: NewS,
    ) -> ReactionDiffusionSystemBuilder<R, D, NewS> {
        ReactionDiffusionSystemBuilder {
            num_species: self.num_species,
            grid_size: self.grid_size,
            reaction: self.reaction,
            diffusion: self.diffusion,
            diffusion_coeffs: self.diffusion_coeffs,
            solver: Some(solver),
        }
    }
}

impl<R: ReactionModel, D: DiffusionModel>
    ReactionDiffusionSystemBuilder<R, D, Euler<ChemicalState>>
{
    /// Builds the `ReactionDiffusionSystem` with the default Euler solver.
    #[verified_engine::verified]
    pub fn build(
        self,
    ) -> Result<ReactionDiffusionSystem<R, D, Euler<ChemicalState>>, ReactionDiffusionError> {
        let num_species = self
            .num_species
            .ok_or(ReactionDiffusionError::MissingParameter("num_species"))?;
        let grid_size = self
            .grid_size
            .ok_or(ReactionDiffusionError::MissingParameter("grid_size"))?;
        let reaction = self
            .reaction
            .ok_or(ReactionDiffusionError::MissingParameter("reaction"))?;
        let diffusion = self
            .diffusion
            .ok_or(ReactionDiffusionError::MissingParameter("diffusion"))?;
        let diffusion_coeffs = self
            .diffusion_coeffs
            .ok_or(ReactionDiffusionError::MissingParameter("diffusion_coeffs"))?;

        if num_species == 0 {
            return Err(ReactionDiffusionError::ZeroSpecies);
        }

        if grid_size == 0 {
            return Err(ReactionDiffusionError::ZeroGridSize);
        }

        if diffusion_coeffs.len() != num_species {
            return Err(ReactionDiffusionError::Math(
                math_commons::error::MathError::DimensionMismatch {
                    expected: math_commons::math_kernel::types::Dimension(num_species),
                    actual: math_commons::math_kernel::types::Dimension(diffusion_coeffs.len()),
                },
            ));
        }

        let state = ChemicalState::new(num_species, grid_size);
        let solver = Euler::new(&state);

        Ok(ReactionDiffusionSystem {
            model: ReactionDiffusionModel {
                reaction,
                diffusion,
                diffusion_coeffs,
            },
            state,
            solver,
            time: 0.0,
        })
    }
}

impl<R: ReactionModel, D: DiffusionModel, S: Solver<ChemicalState>>
    ReactionDiffusionSystemBuilder<R, D, S>
{
    /// Builds the `ReactionDiffusionSystem` with a custom solver.
    #[verified_engine::verified]
    pub fn build_with_solver(
        self,
    ) -> Result<ReactionDiffusionSystem<R, D, S>, ReactionDiffusionError> {
        let num_species = self
            .num_species
            .ok_or(ReactionDiffusionError::MissingParameter("num_species"))?;
        let grid_size = self
            .grid_size
            .ok_or(ReactionDiffusionError::MissingParameter("grid_size"))?;
        let reaction = self
            .reaction
            .ok_or(ReactionDiffusionError::MissingParameter("reaction"))?;
        let diffusion = self
            .diffusion
            .ok_or(ReactionDiffusionError::MissingParameter("diffusion"))?;
        let diffusion_coeffs = self
            .diffusion_coeffs
            .ok_or(ReactionDiffusionError::MissingParameter("diffusion_coeffs"))?;
        let solver = self
            .solver
            .ok_or(ReactionDiffusionError::MissingParameter("solver"))?;

        if num_species == 0 {
            return Err(ReactionDiffusionError::ZeroSpecies);
        }

        if grid_size == 0 {
            return Err(ReactionDiffusionError::ZeroGridSize);
        }

        if diffusion_coeffs.len() != num_species {
            return Err(ReactionDiffusionError::Math(
                math_commons::error::MathError::DimensionMismatch {
                    expected: math_commons::math_kernel::types::Dimension(num_species),
                    actual: math_commons::math_kernel::types::Dimension(diffusion_coeffs.len()),
                },
            ));
        }

        Ok(ReactionDiffusionSystem {
            model: ReactionDiffusionModel {
                reaction,
                diffusion,
                diffusion_coeffs,
            },
            state: ChemicalState::new(num_species, grid_size),
            solver,
            time: 0.0,
        })
    }
}

impl<R: ReactionModel, D: DiffusionModel> ReactionDiffusionSystem<R, D, Euler<ChemicalState>> {
    /// Creates a new builder for a Reaction-Diffusion system.
    #[verified_engine::verified]
    pub fn builder() -> ReactionDiffusionSystemBuilder<R, D, Euler<ChemicalState>> {
        ReactionDiffusionSystemBuilder::new()
    }
}

impl<R: ReactionModel, D: DiffusionModel, S: Solver<ChemicalState>>
    ReactionDiffusionSystem<R, D, S>
{
    /// Returns the elapsed simulation time `t`.
    #[verified_engine::verified]
    pub fn time(&self) -> f64 {
        self.time
    }

    /// Sets the current simulation time `t`.
    #[verified_engine::verified]
    pub fn set_time(&mut self, time: f64) {
        self.time = time;
    }

    /// Advances the system at explicit simulation time `t` by `dt` without mutating [`Self::time`].
    #[verified_engine::verified]
    pub fn step_at(&mut self, t: f64, dt: f64) {
        self.solver.step(&self.model, t, &mut self.state, dt);
    }

    /// Advances the system by a time step `dt` using the configured solver.
    #[verified_engine::verified]
    pub fn step(&mut self, dt: f64) {
        let t = self.time;
        self.solver.step(&self.model, t, &mut self.state, dt);
        self.time += dt;
    }

    /// Accessor for the reaction model.
    #[verified_engine::verified]
    pub fn reaction(&self) -> &R {
        &self.model.reaction
    }

    /// Accessor for the diffusion model.
    #[verified_engine::verified]
    pub fn diffusion(&self) -> &D {
        &self.model.diffusion
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pure_math::pure_math::analysis::ode::RungeKutta4;

    #[derive(Debug, Clone, Copy, PartialEq)]
    struct TimeForcedReaction;

    impl ReactionModel for TimeForcedReaction {
        fn reaction(&self, _u: &[f64], out: &mut [f64]) {
            out.fill(0.0);
        }

        fn reaction_at(&self, t: f64, _u: &[f64], out: &mut [f64]) {
            for val in out.iter_mut() {
                *val = 2.0 * t;
            }
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq)]
    struct ZeroDiffusion;

    impl DiffusionModel for ZeroDiffusion {
        fn apply(&self, _state: &ChemicalState, out: &mut ChemicalState, _coeffs: &[f64]) {
            for s in 0..out.num_species() {
                out.species_mut(s).fill(0.0);
            }
        }
    }

    #[test]
    fn test_reaction_diffusion_non_autonomous_time_and_derives() {
        let state_template = ChemicalState::new(2, 4);
        let mut sys = ReactionDiffusionSystem::builder()
            .num_species(2)
            .grid_size(4)
            .reaction(TimeForcedReaction)
            .diffusion(ZeroDiffusion)
            .diffusion_coeffs(vec![0.0, 0.0])
            .solver(RungeKutta4::new(&state_template))
            .build_with_solver()
            .unwrap();

        assert_eq!(sys, sys.clone());
        assert!(!format!("{sys:?}").is_empty());

        for _ in 0..10 {
            sys.step(0.1);
        }

        assert!((sys.time() - 1.0).abs() < 1e-12);
        for &c in sys.state.species(0) {
            assert!((c - 1.0).abs() < 1e-12);
        }
        for &c in sys.state.species(1) {
            assert!((c - 1.0).abs() < 1e-12);
        }
    }
}
