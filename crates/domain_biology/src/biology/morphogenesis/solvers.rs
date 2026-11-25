use super::reaction::ReactionKinetics;
use super::state::TuringState;
use crate::biology::diffusion::SpatialDiffusion;
use pure_math::pure_math::analysis::ode::{OdeSystem, Solver, VectorOperations};

/// Strategy for time-stepping $N$-species reaction-diffusion systems:
/// $$\frac{\partial \mathbf{u}}{\partial t} = \mathbf{D} \nabla^2 \mathbf{u} + \mathbf{R}(\mathbf{u})$$
///
/// Reference: Turing (1952), DOI: `10.1098/rstb.1952.0012`; see
/// [`docs/math_specs/domain_biology.md`](../../../../../docs/math_specs/domain_biology.md).
///
/// # Stability
/// Explicit time-steppers must satisfy the parabolic CFL bound
/// $\Delta t \le h^2 / (2 d \max_i D_i)$ in $d$ spatial dimensions to remain $L^\infty$-stable.
///
/// # Domain of Validity
/// Uniform finite-difference grids with matching buffer lengths, $D_i \ge 0$, and $\Delta t > 0$.
pub trait TuringSolverStrategy<const N: usize = 2> {
    /// Performs a single time step of the simulation.
    ///
    /// # Arguments
    /// * `state` - Current system state.
    /// * `next_state` - Buffer for the next system state.
    /// * `dynamics` - The physics model (kinetics + diffusion + coefficients).
    /// * `dt` - Time step size.
    fn step<K: ReactionKinetics<N>, D: SpatialDiffusion<N>>(
        &mut self,
        state: &TuringState<N>,
        next_state: &mut TuringState<N>,
        dynamics: &TuringDynamics<N, K, D>,
        dt: f64,
    );

    /// Performs a single time step starting at explicit simulation time `t`.
    fn step_at<K: ReactionKinetics<N>, D: SpatialDiffusion<N>>(
        &mut self,
        _t: f64,
        state: &TuringState<N>,
        next_state: &mut TuringState<N>,
        dynamics: &TuringDynamics<N, K, D>,
        dt: f64,
    ) {
        self.step(state, next_state, dynamics, dt);
    }
}

/// A descriptor for the Turing system's physics (OdeSystem).
///
/// This struct implements `OdeSystem` to allow standard solvers to integrate
/// the reaction-diffusion equations.
pub struct TuringDynamics<'a, const N: usize, K, D> {
    /// Local reaction kinetics operator $\mathbf{R}(\mathbf{u})$.
    pub kinetics: &'a K,
    /// Spatial Laplacian operator $\nabla^2 \mathbf{u}$.
    pub diffusion: &'a D,
    /// Per-species scalar diffusivities $[D_0, \dots, D_{N-1}]$.
    pub diffusion_coeffs: [f64; N],
}

impl<'a, const N: usize, K: ReactionKinetics<N>, D: SpatialDiffusion<N>> OdeSystem<TuringState<N>>
    for TuringDynamics<'a, N, K, D>
{
    #[verified_engine::verified]
    fn derivative(&self, t: f64, state: &TuringState<N>) -> TuringState<N> {
        let mut out = TuringState::new(state.len());
        self.derivative_in_place(t, state, &mut out);
        out
    }

    #[verified_engine::verified]
    fn derivative_in_place(&self, t: f64, state: &TuringState<N>, out: &mut TuringState<N>) {
        let n = state.len();
        if n == 0 {
            return;
        }

        // SAFETY: Check vector lengths to avoid UB in unsafe block
        for (i, vec) in state.concentrations.iter().enumerate() {
            assert_eq!(vec.len(), n, "State vector {} length mismatch", i);
        }

        // Ensure output buffer is the right size
        if out.len() != n {
            *out = TuringState::new(n);
        } else {
            for (i, vec) in out.concentrations.iter().enumerate() {
                assert_eq!(vec.len(), n, "Output vector {} length mismatch", i);
            }
        }

        // Prepare slices for diffusion application
        let state_slices: [&[f64]; N] = std::array::from_fn(|i| state.concentrations[i].as_slice());

        // Create an array of mutable slices from the concentrations Vec without using unwrap or unsafe.
        let mut out_slices: [&mut [f64]; N] = std::array::from_fn(|_| &mut [] as &mut [f64]);

        // Safe disjoint mutable borrowing of array elements:
        let mut iter = out.concentrations.iter_mut();
        for slice in &mut out_slices {
            if let Some(vec) = iter.next() {
                *slice = vec.as_mut_slice();
            }
        }

        // 1. Compute Diffusion
        self.diffusion
            .apply(state_slices, out_slices, self.diffusion_coeffs);

        // 2. Compute Reaction and Accumulate
        for i in 0..n {
            // Gather inputs
            // We use standard indexing here. The compiler can often elide bounds checks
            // because we asserted lengths at the start of the function.
            let inputs: [f64; N] = std::array::from_fn(|s| state.concentrations[s][i]);

            // Compute Reaction
            let rates = self.kinetics.reaction_at_time(t, i, inputs);

            // Accumulate results
            for (s, rate) in rates.iter().enumerate().take(N) {
                out.concentrations[s][i] += rate;
            }
        }
    }
}

/// A fused Euler integration solver.
///
/// This solver combines diffusion, reaction, and time integration into a single pass
/// using `SpatialDiffusion::map_diffusion` to maximize data locality.
///
/// # Optimization
///
/// Traditional solvers often compute the derivative into a temporary buffer and then add it to the state:
/// `state_next = state + dt * (diffusion + reaction)`.
/// This requires writing the derivative to memory and reading it back.
///
/// The "Fused" solver uses a closure passed to the diffusion engine to compute the reaction
/// and the time update *immediately* after the Laplacian is computed for a grid point (or block).
/// This keeps the data in L1/L2 cache, significantly improving memory bandwidth efficiency.
///
/// # Stability Warning
///
/// This solver implements the **Forward Euler** method, which is an explicit first-order method.
/// It is conditionally stable. If the time step `dt` is too large relative to the diffusion coefficients
/// and grid spacing, the simulation will explode (values go to infinity).
///
/// The stability condition for 1D diffusion is approximately:
/// $$ \Delta t \le \frac{\Delta x^2}{2 D} $$
/// and for a 2D isotropic 5-point Laplacian stencil:
/// $$ \Delta t \le \frac{\Delta x^2}{4 \max(D_u, D_v)} $$
///
/// Reference: Turing (1952), DOI: `10.1098/rstb.1952.0012`; see
/// [`docs/math_specs/domain_biology.md`](../../../../../docs/math_specs/domain_biology.md).
///
/// # Stability
/// Conditionally stable explicit forward-Euler update requiring
/// $\Delta t \le h^2 / (2 d \max_i D_i)$ in $d$ spatial dimensions; executes with zero heap allocations.
///
/// # Domain of Validity
/// Non-stiff or CFL-bounded parabolic reaction-diffusion grids with $D_i \ge 0$, $h > 0$, and $\Delta t > 0$.
///
/// # Examples
///
/// Manually stepping a system:
///
/// ```rust
/// # use domain_biology::__doctest_facade as math_explorer;
/// use math_explorer::biology::morphogenesis::{FusedEulerSolver, TuringSolverStrategy, TuringState, SchnakenbergKinetics, TuringDynamics};
/// use math_explorer::biology::diffusion::FiniteDifference1D;
///
/// // 1. Setup System Components
/// let mut solver = FusedEulerSolver::new();
/// let n = 100;
/// let state = TuringState::new(n);
/// let mut next_state = TuringState::new(n);
/// let kinetics = SchnakenbergKinetics::default();
/// let diffusion = FiniteDifference1D::new(math_explorer::math_kernel::types::StepSize(1.0));
/// let dynamics = TuringDynamics {
///    kinetics: &kinetics,
///    diffusion: &diffusion,
///    diffusion_coeffs: [1.0, 40.0],
/// };
///
/// // 2. Perform a single time step
/// // dt = 0.01
/// solver.step(&state, &mut next_state, &dynamics, 0.01);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct FusedEulerSolver;

impl FusedEulerSolver {
    /// Creates a new FusedEulerSolver.
    ///
    /// Since the solver holds no state (it is a pure strategy), this is effectively a no-op
    /// provided for API consistency.
    #[verified_engine::verified]
    pub fn new() -> Self {
        Self
    }

    /// Computes the damped 2D Forward Euler CFL time-step bound (`0.8 * dx^2 / (4.0 * max(|d_u|, |d_v|))`)
    /// for a 5-point isotropic finite-difference Laplacian on grid spacing `dx`.
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub fn cfl_dt_2d(dx: f64, d_u: f64, d_v: f64) -> f64 {
        let max_d = d_u.abs().max(d_v.abs());
        if !dx.is_finite() || dx <= 0.0 || !max_d.is_finite() || max_d <= 1e-15 {
            return f64::INFINITY;
        }
        0.8 * (dx * dx) / (4.0 * max_d)
    }

    /// Splits `target_dt` into `(substeps, sub_dt)` bounded by `max_substeps` (`>= 1`) so each
    /// substep satisfies `sub_dt <= Self::cfl_dt_2d(dx, d_u, d_v)`.
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub fn cfl_substeps_2d(
        target_dt: f64,
        dx: f64,
        d_u: f64,
        d_v: f64,
        max_substeps: usize,
    ) -> (usize, f64) {
        let cap = max_substeps.max(1);
        if !target_dt.is_finite() || target_dt <= 0.0 {
            return (1, 0.0);
        }
        let cfl = Self::cfl_dt_2d(dx, d_u, d_v);
        if !cfl.is_finite() || target_dt <= cfl {
            return (1, target_dt);
        }
        let needed = (target_dt / cfl).ceil();
        let substeps = if needed.is_finite() && needed < (cap as f64) {
            (needed as usize).max(1)
        } else {
            cap
        };
        let sub_dt = (target_dt / (substeps as f64)).min(cfl);
        (substeps, sub_dt)
    }
}

impl<const N: usize> TuringSolverStrategy<N> for FusedEulerSolver {
    fn step<K: ReactionKinetics<N>, D: SpatialDiffusion<N>>(
        &mut self,
        state: &TuringState<N>,
        next_state: &mut TuringState<N>,
        dynamics: &TuringDynamics<N, K, D>,
        dt: f64,
    ) {
        let n = state.len();
        if n == 0 {
            return;
        }

        if next_state.len() != n {
            return;
        }

        let concentrations_slices: [&[f64]; N] =
            std::array::from_fn(|i| state.concentrations[i].as_slice());

        let mut next_slices_arr: [&mut [f64]; N] = std::array::from_fn(|_| &mut [] as &mut [f64]);

        let mut iter = next_state.concentrations.iter_mut();
        for slice in &mut next_slices_arr {
            if let Some(vec) = iter.next() {
                *slice = vec.as_mut_slice();
            }
        }

        dynamics.diffusion.step_fused(
            concentrations_slices,
            next_slices_arr,
            dt,
            dynamics.diffusion_coeffs,
            dynamics.kinetics,
        );
    }

    fn step_at<K: ReactionKinetics<N>, D: SpatialDiffusion<N>>(
        &mut self,
        t: f64,
        state: &TuringState<N>,
        next_state: &mut TuringState<N>,
        dynamics: &TuringDynamics<N, K, D>,
        dt: f64,
    ) {
        if !dynamics.kinetics.is_time_dependent() {
            self.step(state, next_state, dynamics, dt);
            return;
        }
        let n = state.len();
        if n == 0 || next_state.len() != n {
            return;
        }
        dynamics.derivative_in_place(t, state, next_state);
        for s in 0..N {
            for i in 0..n {
                next_state.concentrations[s][i] =
                    state.concentrations[s][i] + dt * next_state.concentrations[s][i];
            }
        }
    }
}

/// Adapts a generic ODE [`Solver`] (e.g. `RungeKutta4`) to [`TuringSolverStrategy`] for
/// $$\frac{\partial \mathbf{u}}{\partial t} = \mathbf{D} \nabla^2 \mathbf{u} + \mathbf{R}(\mathbf{u})$$
///
/// Reference: Turing (1952), DOI: `10.1098/rstb.1952.0012`; see
/// [`docs/math_specs/domain_biology.md`](../../../../../docs/math_specs/domain_biology.md).
///
/// # Stability
/// Inherits the stability region of the wrapped explicit `Solver` (e.g., RK4 extends the real-axis
/// parabolic stability limit to $\approx 2.78\, h^2 / (4 d \max_i D_i)$).
///
/// # Domain of Validity
/// Method-of-lines discretizations over finite grids with $D_i \ge 0$ and matching `TuringState` dimensions.
pub struct StandardSolverAdapter<S> {
    /// Wrapped ODE time-integrator strategy.
    pub solver: S,
    time: f64,
}

impl<S> StandardSolverAdapter<S> {
    /// Wraps an [`OdeSystem`] `solver` as a [`TuringSolverStrategy`].
    #[verified_engine::verified]
    pub fn new(solver: S) -> Self {
        Self { solver, time: 0.0 }
    }

    /// Returns the elapsed simulation clock $t$ tracked by this adapter.
    #[must_use]
    #[verified_engine::verified]
    pub const fn time(&self) -> f64 {
        self.time
    }

    /// Sets the simulation clock to `t`.
    #[verified_engine::verified]
    pub fn set_time(&mut self, t: f64) {
        self.time = t;
    }
}

impl<const N: usize, S: Solver<TuringState<N>>> TuringSolverStrategy<N>
    for StandardSolverAdapter<S>
{
    fn step<K: ReactionKinetics<N>, D: SpatialDiffusion<N>>(
        &mut self,
        state: &TuringState<N>,
        next_state: &mut TuringState<N>,
        dynamics: &TuringDynamics<N, K, D>,
        dt: f64,
    ) {
        let t = self.time;
        self.step_at(t, state, next_state, dynamics, dt);
    }

    fn step_at<K: ReactionKinetics<N>, D: SpatialDiffusion<N>>(
        &mut self,
        t: f64,
        state: &TuringState<N>,
        next_state: &mut TuringState<N>,
        dynamics: &TuringDynamics<N, K, D>,
        dt: f64,
    ) {
        next_state.copy_from(state);
        self.solver.step(dynamics, t, next_state, dt);
        self.time = t + dt;
    }
}

use engine_pde::{BoundaryCondition, inject_additive_noise_1d};
use math_commons::math_kernel::types::SolverWorkspace;
use pure_math::pure_math::analysis::evolution::{EvolutionEngine, EvolutionError};
use rand::RngCore;

/// A wrapper to use [`FusedEulerSolver`] within the unified [`EvolutionEngine`] interface,
/// forwarding caller `rng` streams into [`engine_pde`] spatio-temporal noise when `noise_intensity > 0.0`.
pub struct FusedEulerEvolution<'a, const N: usize, K, D> {
    /// Underlying deterministic fused reaction-diffusion stepper.
    pub solver: FusedEulerSolver,
    /// Borrowed reaction-diffusion kinetics and spatial diffusion operator.
    pub dynamics: &'a TuringDynamics<'a, N, K, D>,
    /// Optional spatio-temporal Gaussian white-noise intensity $D_{\text{noise}} \ge 0$.
    pub noise_intensity: f64,
    /// Effective spatial cell measure $\Delta V > 0$ scaling stochastic concentration fluctuations.
    pub cell_measure: f64,
}

impl<'a, const N: usize, K, D> FusedEulerEvolution<'a, N, K, D> {
    /// Creates a deterministic (`noise_intensity = 0.0`) [`FusedEulerEvolution`] adapter.
    #[verified_engine::verified]
    pub fn new(solver: FusedEulerSolver, dynamics: &'a TuringDynamics<'a, N, K, D>) -> Self {
        Self {
            solver,
            dynamics,
            noise_intensity: 0.0,
            cell_measure: 1.0,
        }
    }

    /// Fallibly creates a stochastic [`FusedEulerEvolution`] adapter injecting white noise of `noise_intensity`
    /// scaled by `cell_measure` ($\Delta x$ or $\Delta x \Delta y$) via [`engine_pde`].
    #[verified_engine::verified]
    pub fn try_with_stochastic_noise(
        solver: FusedEulerSolver,
        dynamics: &'a TuringDynamics<'a, N, K, D>,
        noise_intensity: f64,
        cell_measure: f64,
    ) -> Result<Self, EvolutionError> {
        if !(noise_intensity >= 0.0
            && noise_intensity.is_finite()
            && cell_measure > 0.0
            && cell_measure.is_finite())
        {
            return Err(EvolutionError::InvalidState);
        }
        debug_assert!(noise_intensity >= 0.0 && cell_measure > 0.0);
        Ok(Self {
            solver,
            dynamics,
            noise_intensity,
            cell_measure,
        })
    }

    /// Creates a stochastic [`FusedEulerEvolution`] adapter injecting white noise of `noise_intensity`
    /// scaled by `cell_measure`; prefer [`Self::try_with_stochastic_noise`].
    #[verified_engine::verified]
    pub fn with_stochastic_noise(
        solver: FusedEulerSolver,
        dynamics: &'a TuringDynamics<'a, N, K, D>,
        noise_intensity: f64,
        cell_measure: f64,
    ) -> Self {
        Self::try_with_stochastic_noise(solver, dynamics, noise_intensity, cell_measure)
            .expect("noise_intensity must be >= 0 and cell_measure > 0")
    }
}

impl<'a, const N: usize, K: ReactionKinetics<N>, D: SpatialDiffusion<N>>
    EvolutionEngine<TuringState<N>, TuringState<N>> for FusedEulerEvolution<'a, N, K, D>
{
    #[verified_engine::verified]
    fn step<R: RngCore + ?Sized>(
        &mut self,
        state: &mut TuringState<N>,
        aux: &mut TuringState<N>,
        rng: &mut R,
        dt: f64,
    ) -> Result<(), EvolutionError> {
        if !dt.is_finite() || dt < 0.0 {
            return Err(EvolutionError::InvalidState);
        }
        self.solver.step(state, aux, self.dynamics, dt);
        state.copy_from(aux);
        if self.noise_intensity > 0.0 && dt > 0.0 {
            let dx = self.cell_measure.max(1e-12);
            let mut stack_scratch = [0.0_f64; 64];
            for species in &mut state.concentrations {
                for chunk in species.chunks_mut(64) {
                    let mut ws = SolverWorkspace::new(&mut stack_scratch);
                    inject_additive_noise_1d(
                        chunk,
                        dx,
                        dt,
                        self.noise_intensity,
                        BoundaryCondition::Neumann,
                        &mut ws,
                        rng,
                    );
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "solvers_tests.rs"]
mod stochastic_evolution_tests;
