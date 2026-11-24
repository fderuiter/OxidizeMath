use pure_math::pure_math::analysis::ode::solvers::Euler;
use pure_math::pure_math::analysis::ode::traits::{OdeSystem, Solver};
use verified_engine::Theory;

use super::{ChemicalState, DiffusionModel, ReactionDiffusionError, ReactionModel};

/// Pure operator definition of an $N$-species reaction-diffusion PDE $\partial_t \mathbf{C} = \mathbf{D}\nabla^2 \mathbf{C} + \mathbf{R}(\mathbf{C})$ (`@turing1952chemical`, `spec:biology_v1`).
///
/// - **Numerical stability**: Explicit spatial Laplacians require the parabolic CFL step bound $\Delta t \le \Delta x^2 / (2 d \max_i D_i)$.
/// - **Domain of validity**: Requires `diffusion_coeffs.len() == state.num_species()`, non-negative diffusivities $D_i \ge 0$, and non-empty spatial grids.
#[derive(Theory)]
#[theory(
    description = "Turing patterns arise from reaction-diffusion systems where local auto-activation is coupled with long-range inhibition, leading to spontaneous symmetry breaking and the formation of stable spatial patterns.",
    citation = "turing1952chemical"
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
    fn derivative(&self, _t: f64, state: &ChemicalState) -> ChemicalState {
        let mut out = ChemicalState::new(state.num_species(), state.grid_size());
        self.derivative_in_place(_t, state, &mut out);
        out
    }

    #[verified_engine::verified]
    fn derivative_in_place(&self, _t: f64, state: &ChemicalState, out: &mut ChemicalState) {
        // Compute Diffusion
        self.diffusion.apply(state, out, &self.diffusion_coeffs);

        // Add Reaction
        // Optimized: Use batch processing to allow vectorization and avoid gather/scatter
        self.reaction.add_reaction_batch(state, out);
    }
}

/// Coupled state-and-solver container integrating $\partial_t \mathbf{C} = \mathbf{D}\nabla^2 \mathbf{C} + \mathbf{R}(\mathbf{C})$ (`@turing1952chemical`, `spec:biology_v1`).
/// Numerical stability: governed by the parabolic CFL bound $\Delta t \le \Delta x^2 / (2 d \max_i D_i)$ of the active solver `S`. Domain of validity: `num_species >= 1`, `grid_size >= 1`, and finite $D_i \ge 0$.
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
}

/// Validated builder assembling [`ReactionDiffusionSystem`] for $\partial_t \mathbf{C} = \mathbf{D}\nabla^2 \mathbf{C} + \mathbf{R}(\mathbf{C})$ (`@turing1952chemical`, `spec:biology_v1`).
/// Numerical stability: enforces dimensional alignment prior to time-stepping with zero floating-point drift. Domain of validity: requires positive `num_species`, positive `grid_size`, and matching `diffusion_coeffs` length.
#[derive(Debug, Clone)]
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
    /// Starts a builder for $\partial_t \mathbf{C} = \mathbf{D}\nabla^2 \mathbf{C} + \mathbf{R}(\mathbf{C})$ with empty slots (`@turing1952chemical`, `spec:biology_v1`).
    /// Numerical stability: exact `None` initialization without rounding. Domain of validity: all required fields must be populated before calling `build`.
    #[verified_engine::verified]
    pub fn new() -> Self {
        Self::default()
    }
}

impl<R: ReactionModel, D: DiffusionModel, S: Solver<ChemicalState>>
    ReactionDiffusionSystemBuilder<R, D, S>
{
    /// Sets the species count $N$ for state vector $\mathbf{C} \in \mathbb{R}^{N \times M}$ (`@turing1952chemical`, `spec:biology_v1`).
    /// Numerical stability: exact integer assignment. Domain of validity: requires $N \ge 1$ at `build` validation.
    #[verified_engine::verified]
    pub fn num_species(mut self, num_species: usize) -> Self {
        self.num_species = Some(num_species);
        self
    }

    /// Sets the flattened spatial node count $M$ for $\mathbf{C} \in \mathbb{R}^{N \times M}$ (`@turing1952chemical`, `spec:biology_v1`).
    /// Numerical stability: exact integer assignment. Domain of validity: requires $M \ge 1$ matching the discretised domain mesh.
    #[verified_engine::verified]
    pub fn grid_size(mut self, grid_size: usize) -> Self {
        self.grid_size = Some(grid_size);
        self
    }

    /// Binds the nonlinear reaction operator $\mathbf{R}(\mathbf{C})$ (`@turing1952chemical`, `spec:biology_v1`).
    /// Numerical stability: lossless move without numerical rounding. Domain of validity: `reaction` must support `num_species` channels.
    #[verified_engine::verified]
    pub fn reaction(mut self, reaction: R) -> Self {
        self.reaction = Some(reaction);
        self
    }

    /// Binds the spatial diffusion operator $\mathbf{D}\nabla^2 \mathbf{C}$ (`@turing1952chemical`, `spec:biology_v1`).
    /// Numerical stability: lossless move without numerical rounding. Domain of validity: `diffusion` stencil must match `grid_size`.
    #[verified_engine::verified]
    pub fn diffusion(mut self, diffusion: D) -> Self {
        self.diffusion = Some(diffusion);
        self
    }

    /// Sets species diffusivities $\mathbf{D} = [D_1, \dots, D_N]$ (`@turing1952chemical`, `spec:biology_v1`).
    /// Numerical stability: exact `f64` vector capture. Domain of validity: requires `coeffs.len() == num_species` and finite $D_i \ge 0$.
    #[verified_engine::verified]
    pub fn diffusion_coeffs(mut self, coeffs: Vec<f64>) -> Self {
        self.diffusion_coeffs = Some(coeffs);
        self
    }

    /// Substitutes time-integrator `solver` advancing $\mathbf{C}(t + \Delta t)$ (`@turing1952chemical`, `spec:biology_v1`).
    /// Numerical stability: inherits truncation order and stability region of `NewS`. Domain of validity: `solver` buffers must match `(num_species, grid_size)`.
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
    /// Validates dimensions and constructs a Forward-Euler system $\mathbf{C}^{n+1} = \mathbf{C}^n + \Delta t \dot{\mathbf{C}}^n$ (`@turing1952chemical`, `spec:biology_v1`).
    /// Numerical stability: first-order $O(\Delta t)$ Euler requires $\Delta t \le \Delta x^2 / (2 d \max_i D_i)$. Domain of validity: `num_species >= 1`, `grid_size >= 1`, and `diffusion_coeffs.len() == num_species`.
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
        })
    }
}

impl<R: ReactionModel, D: DiffusionModel, S: Solver<ChemicalState>>
    ReactionDiffusionSystemBuilder<R, D, S>
{
    /// Validates dimensions and constructs a [`ReactionDiffusionSystem`] with custom integrator $\Phi_{\Delta t}$ (`@turing1952chemical`, `spec:biology_v1`).
    /// Numerical stability: governed by the CFL and stiffness stability region of `S`. Domain of validity: `num_species >= 1`, `grid_size >= 1`, `diffusion_coeffs.len() == num_species`, and `solver` present.
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
        })
    }
}

impl<R: ReactionModel, D: DiffusionModel> ReactionDiffusionSystem<R, D, Euler<ChemicalState>> {
    /// Returns a fresh builder for $\partial_t \mathbf{C} = \mathbf{D}\nabla^2 \mathbf{C} + \mathbf{R}(\mathbf{C})$ (`@turing1952chemical`, `spec:biology_v1`).
    /// Numerical stability: exact `None` initialization. Domain of validity: configure species, grid, operators, and diffusivities prior to `build`.
    #[verified_engine::verified]
    pub fn builder() -> ReactionDiffusionSystemBuilder<R, D, Euler<ChemicalState>> {
        ReactionDiffusionSystemBuilder::new()
    }
}

impl<R: ReactionModel, D: DiffusionModel, S: Solver<ChemicalState>>
    ReactionDiffusionSystem<R, D, S>
{
    /// Advances $\mathbf{C}(t)$ to $\mathbf{C}(t + \Delta t)$ under $\partial_t \mathbf{C} = \mathbf{D}\nabla^2 \mathbf{C} + \mathbf{R}(\mathbf{C})$ (`@turing1952chemical`, `spec:biology_v1`).
    /// Numerical stability: requires finite $\Delta t > 0$ within the solver's parabolic CFL stability bound. Domain of validity: autonomous reaction-diffusion state with finite concentrations.
    #[verified_engine::verified]
    pub fn step(&mut self, dt: f64) {
        // The solver manages the integration logic.
        // We pass 0.0 as the current time since most RD systems are autonomous (time-invariant).
        self.solver.step(&self.model, 0.0, &mut self.state, dt);
    }

    /// Borrows the reaction operator $\mathbf{R}(\mathbf{C})$ (`@turing1952chemical`, `spec:biology_v1`).
    /// Numerical stability: zero-cost reference borrow without arithmetic error. Domain of validity: valid for the lifetime of `self`.
    #[verified_engine::verified]
    pub fn reaction(&self) -> &R {
        &self.model.reaction
    }

    /// Borrows the spatial diffusion operator $\mathbf{D}\nabla^2$ (`@turing1952chemical`, `spec:biology_v1`).
    /// Numerical stability: zero-cost reference borrow without arithmetic error. Domain of validity: valid for the lifetime of `self`.
    #[verified_engine::verified]
    pub fn diffusion(&self) -> &D {
        &self.model.diffusion
    }
}
