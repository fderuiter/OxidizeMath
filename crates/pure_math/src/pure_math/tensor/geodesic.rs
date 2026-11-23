//! Zero-allocation geodesic ODE system $\ddot{x}^\mu + \Gamma^\mu_{\alpha\beta}\dot{x}^\alpha\dot{x}^\beta = 0$.
//!
//! Bridges [`StaticMetric`] manifolds to [`OdeSystem`] and [`RungeKutta4`] over stack-allocated
//! [`ArrayState`] state vectors (`ArrayState<4>` for 2D, `ArrayState<6>` for 3D, and
//! `ArrayState<8>` for 4D spacetimes).

use nalgebra::SVector;

use crate::pure_math::analysis::ode::{
    ArrayState, OdeSystem, RungeKutta4, Solver, VectorOperations,
};

use super::static_metric::StaticMetric;
use super::types::TensorError;

/// Stack-allocated phase-space state `(x^0..x^{N-1}, v^0..v^{N-1})` for an `N`-dimensional geodesic.
pub trait GeodesicState<const N: usize>: VectorOperations {
    /// Packs position $x^\mu$ and four-velocity $\dot{x}^\mu$ into a phase-space state vector.
    fn from_pos_vel(pos: &SVector<f64, N>, vel: &SVector<f64, N>) -> Self;

    /// Splits a phase-space state vector into `(position, velocity)`.
    fn split_pos_vel(&self) -> (SVector<f64, N>, SVector<f64, N>);

    /// Fills all state components with `0.0` in place.
    fn zero_all(&mut self);
}

macro_rules! impl_geodesic_state {
    ($dim:literal, $state_dim:literal) => {
        impl GeodesicState<$dim> for ArrayState<$state_dim> {
            #[verified_engine::verified]
            fn from_pos_vel(pos: &SVector<f64, $dim>, vel: &SVector<f64, $dim>) -> Self {
                let mut data = [0.0; $state_dim];
                let mut i = 0;
                while i < $dim {
                    data[i] = pos[i];
                    data[$dim + i] = vel[i];
                    i += 1;
                }
                ArrayState(data)
            }

            #[verified_engine::verified]
            fn split_pos_vel(&self) -> (SVector<f64, $dim>, SVector<f64, $dim>) {
                let mut pos = SVector::<f64, $dim>::zeros();
                let mut vel = SVector::<f64, $dim>::zeros();
                let mut i = 0;
                while i < $dim {
                    pos[i] = self.0[i];
                    vel[i] = self.0[$dim + i];
                    i += 1;
                }
                (pos, vel)
            }

            #[verified_engine::verified]
            fn zero_all(&mut self) {
                self.0 = [0.0; $state_dim];
            }
        }
    };
}

impl_geodesic_state!(2, 4);
impl_geodesic_state!(3, 6);
impl_geodesic_state!(4, 8);

/// Outcome of a single singularity-checked geodesic integration step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GeodesicStepStatus {
    /// Step succeeded cleanly; reports updated quadratic norm $g_{\mu\nu}\dot{x}^\mu\dot{x}^\nu$.
    Advanced {
        /// Quadratic tangent norm $g_{\mu\nu}(x)\dot{x}^\mu\dot{x}^\nu$ after the step.
        metric_norm: f64,
    },
    /// Integration halted cleanly before a horizon or coordinate singularity.
    HaltedSingularity,
}

/// First-order ODE system for affine-parameterized geodesics on a [`StaticMetric`].
///
/// Integrates the coupled first-order system
/// $$\frac{dx^\mu}{d\lambda} = v^\mu, \qquad
///   \frac{dv^\mu}{d\lambda} = -\Gamma^\mu_{\alpha\beta}(x) v^\alpha v^\beta$$
/// with zero heap allocations. Whenever `x` enters an irregular region
/// (`!metric.is_regular_at(&x)`), derivatives evaluate to zero so trajectories halt cleanly
/// without generating `NaN` or `Inf`.
///
/// # Examples
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::nalgebra::SVector;
/// use math_explorer::pure_math::analysis::ode::{ArrayState, RungeKutta4};
/// use math_explorer::pure_math::tensor::{
///     GeodesicState, GeodesicStepStatus, GeodesicSystem, TwoSphereMetric,
/// };
///
/// let sphere = TwoSphereMetric::new(1.0).unwrap();
/// let system = GeodesicSystem::new(sphere);
/// let mut state = ArrayState::<4>::from_pos_vel(
///     &SVector::<f64, 2>::new(std::f64::consts::FRAC_PI_2, 0.0),
///     &SVector::<f64, 2>::new(0.0, 1.0),
/// );
/// let mut solver = RungeKutta4::new(&state);
/// let status = system.step_rk4(&mut solver, 0.0, &mut state, 0.01);
/// assert!(matches!(status, GeodesicStepStatus::Advanced { .. }));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeodesicSystem<M, const N: usize> {
    metric: M,
}

impl<M: StaticMetric<N>, const N: usize> GeodesicSystem<M, N> {
    /// Wraps `metric` in a stack-allocated geodesic ODE system.
    #[verified_engine::verified]
    pub fn new(metric: M) -> Self {
        Self { metric }
    }

    /// Borrows the underlying [`StaticMetric`].
    #[verified_engine::verified]
    pub fn metric(&self) -> &M {
        &self.metric
    }

    /// Consumes the geodesic system and returns the underlying metric.
    #[verified_engine::verified]
    pub fn into_metric(self) -> M {
        self.metric
    }

    /// Evaluates the quadratic tangent norm $g_{\mu\nu}(x)\dot{x}^\mu\dot{x}^\nu$ for `state`.
    ///
    /// # Errors
    /// Returns [`TensorError::SingularMetric`] if the metric cannot be evaluated at `state`.
    #[verified_engine::verified]
    pub fn metric_norm<S: GeodesicState<N>>(&self, state: &S) -> Result<f64, TensorError> {
        let (pos, vel) = state.split_pos_vel();
        self.metric.metric_norm(&pos, &vel)
    }

    /// Returns `true` if `state` has reached a horizon/coordinate guard or contains non-finite values.
    #[verified_engine::verified]
    pub fn is_halted<S: GeodesicState<N>>(&self, state: &S) -> bool {
        let (pos, vel) = state.split_pos_vel();
        !vel.iter().all(|v| v.is_finite()) || !self.metric.is_regular_at(&pos)
    }

    /// Advances `state` by one RK4 step `d_lambda`, rolling back cleanly if the trial step
    /// enters a horizon or coordinate singularity guard.
    #[verified_engine::verified]
    pub fn step_rk4<S>(
        &self,
        solver: &mut RungeKutta4<S>,
        lambda: f64,
        state: &mut S,
        d_lambda: f64,
    ) -> GeodesicStepStatus
    where
        S: GeodesicState<N>,
        Self: OdeSystem<S>,
    {
        if !lambda.is_finite() || !d_lambda.is_finite() || self.is_halted(state) {
            return GeodesicStepStatus::HaltedSingularity;
        }
        let previous = state.clone();
        solver.step(self, lambda, state, d_lambda);
        if self.is_halted(state) {
            *state = previous;
            return GeodesicStepStatus::HaltedSingularity;
        }
        match self.metric_norm(state) {
            Ok(metric_norm) if metric_norm.is_finite() => {
                GeodesicStepStatus::Advanced { metric_norm }
            }
            _ => {
                *state = previous;
                GeodesicStepStatus::HaltedSingularity
            }
        }
    }
}

#[verified_engine::verified]
fn evaluate_geodesic_rhs<M: StaticMetric<N>, S: GeodesicState<N>, const N: usize>(
    metric: &M,
    state: &S,
    out: &mut S,
) {
    let (pos, vel) = state.split_pos_vel();
    if !vel.iter().all(|v| v.is_finite()) || !metric.is_regular_at(&pos) {
        out.zero_all();
        return;
    }
    let Ok(gamma) = metric.christoffel_at(&pos) else {
        out.zero_all();
        return;
    };

    let mut acc = SVector::<f64, N>::zeros();
    for mu in 0..N {
        let mut sum = 0.0;
        for alpha in 0..N {
            for beta in 0..N {
                sum += gamma[mu][(alpha, beta)] * vel[alpha] * vel[beta];
            }
        }
        if !sum.is_finite() {
            out.zero_all();
            return;
        }
        acc[mu] = -sum;
    }
    *out = S::from_pos_vel(&vel, &acc);
}

macro_rules! impl_ode_system_for_geodesic {
    ($dim:literal, $state_dim:literal) => {
        impl<M: StaticMetric<$dim>> OdeSystem<ArrayState<$state_dim>> for GeodesicSystem<M, $dim> {
            #[verified_engine::verified]
            fn derivative(
                &self,
                _t: f64,
                state: &ArrayState<$state_dim>,
            ) -> ArrayState<$state_dim> {
                let mut out = ArrayState([0.0; $state_dim]);
                evaluate_geodesic_rhs(&self.metric, state, &mut out);
                out
            }

            #[verified_engine::verified]
            fn derivative_in_place(
                &self,
                _t: f64,
                state: &ArrayState<$state_dim>,
                out: &mut ArrayState<$state_dim>,
            ) {
                evaluate_geodesic_rhs(&self.metric, state, out);
            }
        }
    };
}

impl_ode_system_for_geodesic!(2, 4);
impl_ode_system_for_geodesic!(3, 6);
impl_ode_system_for_geodesic!(4, 8);
