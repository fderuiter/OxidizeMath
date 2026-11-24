use super::reaction::ReactionKinetics;
use super::state::TuringState;
use crate::biology::diffusion::SpatialDiffusion;
use pure_math::pure_math::analysis::ode::{OdeSystem, Solver, VectorOperations};

/// Strategy for time-stepping the Turing System.
///
/// This trait allows decoupling the numerical integration method from the system definition.
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
}

/// A descriptor for the Turing system's physics (OdeSystem).
///
/// This struct implements `OdeSystem` to allow standard solvers to integrate
/// the reaction-diffusion equations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TuringDynamics<'a, const N: usize, K, D> {
    #[allow(missing_docs)]
    pub kinetics: &'a K,
    #[allow(missing_docs)]
    pub diffusion: &'a D,
    #[allow(missing_docs)]
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
            let rates = self.kinetics.reaction_at(t, inputs);

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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
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
}

/// Adapts a standard ODE Solver to the TuringSolverStrategy trait.
///
/// This adapter allows using any generic `Solver` (like RungeKutta4) within the TuringSystem.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StandardSolverAdapter<S> {
    #[allow(missing_docs)]
    pub solver: S,
    time: f64,
}

impl<S> StandardSolverAdapter<S> {
    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn new(solver: S) -> Self {
        Self { solver, time: 0.0 }
    }

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
        next_state.copy_from(state);
        self.solver.step(dynamics, t, next_state, dt);
        self.time += dt;
    }
}

use engine_pde::{BoundaryCondition, inject_additive_noise_1d};
use math_commons::math_kernel::types::SolverWorkspace;
use pure_math::pure_math::analysis::evolution::{EvolutionEngine, EvolutionError};
use rand::RngCore;

/// A wrapper to use [`FusedEulerSolver`] within the unified [`EvolutionEngine`] interface,
/// forwarding caller `rng` streams into [`engine_pde`] spatio-temporal noise when `noise_intensity > 0.0`.
#[derive(Debug, Clone, Copy, PartialEq)]
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

    /// Creates a stochastic [`FusedEulerEvolution`] adapter injecting white noise of `noise_intensity`
    /// scaled by `cell_measure` ($\Delta x$ or $\Delta x \Delta y$) via [`engine_pde`].
    #[verified_engine::verified]
    pub fn with_stochastic_noise(
        solver: FusedEulerSolver,
        dynamics: &'a TuringDynamics<'a, N, K, D>,
        noise_intensity: f64,
        cell_measure: f64,
    ) -> Self {
        assert!(noise_intensity >= 0.0 && cell_measure > 0.0);
        Self {
            solver,
            dynamics,
            noise_intensity,
            cell_measure,
        }
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
mod tests {
    use super::*;
    use crate::biology::diffusion::FiniteDifference1D;
    use crate::biology::morphogenesis::{DiffusionCoeff, SchnakenbergKinetics, TuringSystem};
    use math_commons::math_kernel::types::{Dimension, StepSize};
    use oxidize_core::rng::OxidizeRng;
    use pure_math::pure_math::analysis::ode::{RungeKutta4, TimeStepper};

    #[derive(Debug, Clone, Copy, PartialEq)]
    struct TimeForcedKinetics;

    impl ReactionKinetics<2> for TimeForcedKinetics {
        fn reaction(&self, _state: [f64; 2]) -> [f64; 2] {
            [0.0, 0.0]
        }

        fn reaction_at(&self, t: f64, _state: [f64; 2]) -> [f64; 2] {
            [2.0 * t, 3.0 * t * t]
        }
    }

    #[test]
    #[verified_engine::verified]
    fn test_standard_solver_adapter_tracks_non_autonomous_time() {
        let init_state = TuringState::<2>::new(4);
        let adapter = StandardSolverAdapter::new(RungeKutta4::new(&init_state));
        let mut sys = TuringSystem::new_with_solver(
            Dimension(4),
            [DiffusionCoeff(0.0), DiffusionCoeff(0.0)],
            TimeForcedKinetics,
            FiniteDifference1D::new(StepSize(1.0)),
            adapter,
        );
        assert_eq!(sys, sys.clone());

        for _ in 0..10 {
            TimeStepper::step(&mut sys, 0.1);
        }

        assert!((sys.solver.time() - 1.0).abs() < 1e-12);
        for &u in &sys.state.concentrations[0] {
            assert!((u - 1.0).abs() < 1e-12);
        }
        for &v in &sys.state.concentrations[1] {
            assert!((v - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    #[verified_engine::verified]
    fn test_fused_euler_evolution_forwards_rng_to_engine_pde() {
        let kinetics = SchnakenbergKinetics::default();
        let diffusion = FiniteDifference1D::new(StepSize(1.0));
        let dynamics = TuringDynamics {
            kinetics: &kinetics,
            diffusion: &diffusion,
            diffusion_coeffs: [1.0, 20.0],
        };
        let mut state_a = TuringState::<2>::new(32);
        let mut state_b = TuringState::<2>::new(32);
        let mut aux = TuringState::<2>::new(32);
        state_a.concentrations[0].fill(1.0);
        state_a.concentrations[1].fill(0.5);
        state_b.copy_from(&state_a);

        let mut evo_det = FusedEulerEvolution::new(FusedEulerSolver::new(), &dynamics);
        let mut evo_stoch = FusedEulerEvolution::with_stochastic_noise(
            FusedEulerSolver::new(),
            &dynamics,
            0.05,
            1.0,
        );
        let mut rng = OxidizeRng::new(77);
        evo_det
            .step(&mut state_a, &mut aux, &mut rng, 0.01)
            .unwrap();
        evo_stoch
            .step(&mut state_b, &mut aux, &mut rng, 0.01)
            .unwrap();
        // Homogeneous deterministic step stays spatially uniform, whereas stochastic step breaks symmetry!
        let u_min = state_b.concentrations[0]
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let u_max = state_b.concentrations[0]
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(u_max - u_min > 1e-4);
    }
}
