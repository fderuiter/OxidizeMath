//! Zero-allocation symplectic numerical integrators for conservative Hamiltonian ODEs.
//!
//! Provides explicit splitting solvers ([`StormerVerlet`] and [`Yoshida4`]) for
//! [`SeparableHamiltonian`] systems alongside implicit Gauss-Legendre collocation solvers
//! ([`ImplicitMidpoint`] and [`GaussLegendre4`]) for general (including non-separable)
//! [`OdeSystem`] and [`super::traits::HamiltonianSystem`] dynamics.

// [cite:advanced_linear_algebra]

use serde::{Deserialize, Serialize};

pub use super::implicit_midpoint::{ImplicitMidpoint, MAX_FIXED_POINT_ITERS};
use super::state::ArrayState;
use super::traits::{OdeSystem, SeparableHamiltonian, Solver, VectorOperations};

/// Symplectic numerical integration schemes for conservative Hamiltonian systems.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash, Serialize, Deserialize)]
pub enum SymplecticMethod {
    /// 2nd-order explicit velocity Störmer-Verlet (leapfrog) splitting for separable Hamiltonians.
    StormerVerlet,
    /// 4th-order explicit Yoshida (1990) symmetric triple-Verlet composition for separable Hamiltonians.
    #[default]
    Yoshida4,
    /// 2nd-order 1-stage implicit Gauss-Legendre midpoint rule for general Hamiltonians.
    ImplicitMidpoint,
    /// 4th-order 2-stage implicit Gauss-Legendre Runge-Kutta for general Hamiltonians.
    GaussLegendre4,
}

impl SymplecticMethod {
    /// All supported symplectic integration methods.
    pub const ALL: [Self; 4] = [
        Self::StormerVerlet,
        Self::Yoshida4,
        Self::ImplicitMidpoint,
        Self::GaussLegendre4,
    ];

    /// Human-readable display label for UI selectors and telemetry.
    #[must_use]
    #[verified_engine::verified]
    pub fn name(&self) -> &'static str {
        match self {
            Self::StormerVerlet => "Störmer-Verlet (2nd)",
            Self::Yoshida4 => "Yoshida 4th Order",
            Self::ImplicitMidpoint => "Implicit Midpoint (2nd)",
            Self::GaussLegendre4 => "Gauss-Legendre 4th",
        }
    }
}

/// Second-order symplectic Stormer-Verlet (velocity-kick / drift / kick) integrator preserving $d\mathbf{q} \wedge d\mathbf{p}$ for separable Hamiltonians $H(\mathbf{q}, \mathbf{p}) = T(\mathbf{p}) + V(\mathbf{q})$.
///
/// Reference: Hairer, Lubich & Wanner (2006), *Geometric Numerical Integration*, DOI: 10.1007/3-540-30666-8; [`docs/math_specs/pure_math.md`](../../../../../../docs/math_specs/pure_math.md).
///
/// # Stability
/// Time-reversible and symplectic; conserves a shadow Hamiltonian $\tilde{H} = H + O(\Delta t^2)$ over exponentially long horizons when $\omega_{\max} \Delta t < 2$.
///
/// # Domain of Validity
/// Valid for separable Hamiltonian systems with smooth potential forces $-\nabla_{\mathbf{q}} V$ and step size $\Delta t < 2 / \omega_{\max}$.
#[derive(Debug, Clone)]
pub struct StormerVerlet<Coord> {
    q: Coord,
    p: Coord,
    dq: Coord,
    dp: Coord,
}

impl<Coord: Clone> StormerVerlet<Coord> {
    /// Constructs a new `StormerVerlet` solver with pre-allocated canonical scratch buffers.
    #[must_use]
    #[verified_engine::verified]
    pub fn new(example_coord: &Coord) -> Self {
        Self {
            q: example_coord.clone(),
            p: example_coord.clone(),
            dq: example_coord.clone(),
            dp: example_coord.clone(),
        }
    }
}

impl<Coord: VectorOperations> StormerVerlet<Coord> {
    /// Advances canonical coordinates `(q, p)` in-place by one kick-drift-kick step `dt`.
    #[verified_engine::verified]
    pub fn step_qp<State, S>(&mut self, system: &S, q: &mut Coord, p: &mut Coord, dt: f64)
    where
        State: VectorOperations,
        S: SeparableHamiltonian<State, Coord = Coord> + ?Sized,
    {
        let half_dt = 0.5 * dt;
        system.neg_grad_potential_in_place(q, &mut self.dp);
        p.scale_add(&self.dp, half_dt);

        system.grad_kinetic_in_place(p, &mut self.dq);
        q.scale_add(&self.dq, dt);

        system.neg_grad_potential_in_place(q, &mut self.dp);
        p.scale_add(&self.dp, half_dt);
    }

    /// Advances the full phase-space `state` of a [`SeparableHamiltonian`] in-place by `dt`.
    #[verified_engine::verified]
    pub fn step_separable<State, S>(&mut self, system: &S, state: &mut State, dt: f64)
    where
        State: VectorOperations,
        S: SeparableHamiltonian<State, Coord = Coord> + ?Sized,
    {
        system.split_canonical(state, &mut self.q, &mut self.p);
        let half_dt = 0.5 * dt;

        system.neg_grad_potential_in_place(&self.q, &mut self.dp);
        self.p.scale_add(&self.dp, half_dt);

        system.grad_kinetic_in_place(&self.p, &mut self.dq);
        self.q.scale_add(&self.dq, dt);

        system.neg_grad_potential_in_place(&self.q, &mut self.dp);
        self.p.scale_add(&self.dp, half_dt);

        system.join_canonical(&self.q, &self.p, state);
    }
}

/// Real cube root of 2 ($2^{1/3}$) used in Yoshida's 4th-order symmetric composition weights.
const CBRT_2: f64 = 1.259_921_049_894_873_2;
const YOSHIDA_W1: f64 = 1.0 / (2.0 - CBRT_2);
const YOSHIDA_W0: f64 = -CBRT_2 / (2.0 - CBRT_2);
const YOSHIDA_KICK_OUTER: f64 = 0.5 * YOSHIDA_W1;
const YOSHIDA_KICK_INNER: f64 = 0.5 * (YOSHIDA_W0 + YOSHIDA_W1);

/// Fourth-order symmetric symplectic composition solver $S_4(\Delta t) = S_2(w_1 \Delta t) \circ S_2(w_0 \Delta t) \circ S_2(w_1 \Delta t)$ with $w_1 = 1/(2 - 2^{1/3})$.
///
/// Reference: Yoshida (1990), DOI: 10.1016/0375-9601(90)90092-3; [`docs/math_specs/pure_math.md`](../../../../../../docs/math_specs/pure_math.md).
///
/// # Stability
/// Exactly preserves canonical Poincaré invariants with bounded $O(\Delta t^4)$ energy oscillation and zero secular drift below the linear stability threshold.
///
/// # Domain of Validity
/// Valid for $C^4$ separable Hamiltonian systems with $\omega_{\max} |w_0| \Delta t < 2$.
#[derive(Debug, Clone)]
pub struct Yoshida4<Coord> {
    q: Coord,
    p: Coord,
    dq: Coord,
    dp: Coord,
}

impl<Coord: Clone> Yoshida4<Coord> {
    /// Constructs a new `Yoshida4` solver with pre-allocated canonical scratch buffers.
    #[must_use]
    #[verified_engine::verified]
    pub fn new(example_coord: &Coord) -> Self {
        Self {
            q: example_coord.clone(),
            p: example_coord.clone(),
            dq: example_coord.clone(),
            dp: example_coord.clone(),
        }
    }
}

#[verified_engine::verified]
fn yoshida4_kernel<State, Coord, S>(
    system: &S,
    q: &mut Coord,
    p: &mut Coord,
    dq: &mut Coord,
    dp: &mut Coord,
    dt: f64,
) where
    State: VectorOperations,
    Coord: VectorOperations,
    S: SeparableHamiltonian<State, Coord = Coord> + ?Sized,
{
    let k_outer = YOSHIDA_KICK_OUTER * dt;
    let k_inner = YOSHIDA_KICK_INNER * dt;
    let d_outer = YOSHIDA_W1 * dt;
    let d_mid = YOSHIDA_W0 * dt;

    system.neg_grad_potential_in_place(q, dp);
    p.scale_add(dp, k_outer);
    system.grad_kinetic_in_place(p, dq);
    q.scale_add(dq, d_outer);

    system.neg_grad_potential_in_place(q, dp);
    p.scale_add(dp, k_inner);
    system.grad_kinetic_in_place(p, dq);
    q.scale_add(dq, d_mid);

    system.neg_grad_potential_in_place(q, dp);
    p.scale_add(dp, k_inner);
    system.grad_kinetic_in_place(p, dq);
    q.scale_add(dq, d_outer);

    system.neg_grad_potential_in_place(q, dp);
    p.scale_add(dp, k_outer);
}

impl<Coord: VectorOperations> Yoshida4<Coord> {
    /// Advances canonical coordinates `(q, p)` in-place by one 4th-order Yoshida step `dt`.
    #[verified_engine::verified]
    pub fn step_qp<State, S>(&mut self, system: &S, q: &mut Coord, p: &mut Coord, dt: f64)
    where
        State: VectorOperations,
        S: SeparableHamiltonian<State, Coord = Coord> + ?Sized,
    {
        yoshida4_kernel(system, q, p, &mut self.dq, &mut self.dp, dt);
    }

    /// Advances the full phase-space `state` of a [`SeparableHamiltonian`] in-place by `dt`.
    #[verified_engine::verified]
    pub fn step_separable<State, S>(&mut self, system: &S, state: &mut State, dt: f64)
    where
        State: VectorOperations,
        S: SeparableHamiltonian<State, Coord = Coord> + ?Sized,
    {
        system.split_canonical(state, &mut self.q, &mut self.p);
        yoshida4_kernel(
            system,
            &mut self.q,
            &mut self.p,
            &mut self.dq,
            &mut self.dp,
            dt,
        );
        system.join_canonical(&self.q, &self.p, state);
    }
}

const SQRT_3_OVER_6: f64 = 0.288_675_134_594_812_87;
const GL4_C1: f64 = 0.5 - SQRT_3_OVER_6;
const GL4_C2: f64 = 0.5 + SQRT_3_OVER_6;
const GL4_A11: f64 = 0.25;
const GL4_A12: f64 = 0.25 - SQRT_3_OVER_6;
const GL4_A21: f64 = 0.25 + SQRT_3_OVER_6;
const GL4_A22: f64 = 0.25;

/// Two-stage 4th-order implicit Gauss-Legendre collocation Runge-Kutta solver $\mathbf{y}_{n+1} = \mathbf{y}_n + \frac{\Delta t}{2}(\mathbf{k}_1 + \mathbf{k}_2)$.
///
/// Reference: Hairer, Lubich & Wanner (2006), DOI: 10.1007/3-540-30666-8; [`docs/math_specs/pure_math.md`](../../../../../../docs/math_specs/pure_math.md).
///
/// # Stability
/// Unconditionally A-stable, algebraically B-stable, and symplectic on arbitrary (including non-separable) Hamiltonian systems; fixed-point stage iteration requires Lipschitz contraction $L \Delta t < 2$.
///
/// # Domain of Validity
/// Valid for general stiff or Hamiltonian ODEs where stage fixed-point iteration converges within the configured tolerance.
#[derive(Debug, Clone)]
pub struct GaussLegendre4<State> {
    initial_state: State,
    k1: State,
    k2: State,
    k1_next: State,
    k2_next: State,
    stage: State,
    max_iters: usize,
    substeps: usize,
}

impl<State: Clone> GaussLegendre4<State> {
    /// Default fixed-point iteration budget per substep (converges to `f64` machine precision).
    pub const DEFAULT_MAX_ITERS: usize = 10;

    /// Default symmetric symplectic substep count (`2` reduces the $O(h^4)$ shadow error constant $16\times$).
    pub const DEFAULT_SUBSTEPS: usize = 2;

    /// Upper cap on fixed-point stage iterations (`12`) to enforce bounded loop execution.
    pub const MAX_ALLOWED_ITERS: usize = MAX_FIXED_POINT_ITERS;

    /// Upper cap on internal symplectic substeps per `step` call.
    pub const MAX_ALLOWED_SUBSTEPS: usize = 8;

    /// Constructs a new `GaussLegendre4` solver with default iteration and substep budgets.
    #[must_use]
    #[verified_engine::verified]
    pub fn new(example_state: &State) -> Self {
        Self::with_config(
            example_state,
            Self::DEFAULT_MAX_ITERS,
            Self::DEFAULT_SUBSTEPS,
        )
    }

    /// Constructs a new `GaussLegendre4` solver with an explicit fixed-point iteration bound (`1..=12`).
    #[must_use]
    #[verified_engine::verified]
    pub fn with_max_iters(example_state: &State, max_iters: usize) -> Self {
        Self::with_config(example_state, max_iters, Self::DEFAULT_SUBSTEPS)
    }

    /// Constructs a new `GaussLegendre4` solver with explicit fixed-point iteration and substep bounds.
    #[must_use]
    #[verified_engine::verified]
    pub fn with_config(example_state: &State, max_iters: usize, substeps: usize) -> Self {
        Self {
            initial_state: example_state.clone(),
            k1: example_state.clone(),
            k2: example_state.clone(),
            k1_next: example_state.clone(),
            k2_next: example_state.clone(),
            stage: example_state.clone(),
            max_iters: max_iters.clamp(1, Self::MAX_ALLOWED_ITERS),
            substeps: substeps.clamp(1, Self::MAX_ALLOWED_SUBSTEPS),
        }
    }

    /// Returns the configured fixed-point iteration cap (`1..=12`).
    #[must_use]
    #[verified_engine::verified]
    pub const fn max_iters(&self) -> usize {
        self.max_iters
    }
}

impl<const N: usize> GaussLegendre4<ArrayState<N>> {
    /// Advances `state` by `dt`, rolling back to `y_n` and returning `false` if any component becomes non-finite.
    #[verified_engine::verified]
    pub fn try_step_array<S>(
        &mut self,
        system: &S,
        t: f64,
        state: &mut ArrayState<N>,
        dt: f64,
    ) -> bool
    where
        S: OdeSystem<ArrayState<N>> + ?Sized,
    {
        if !dt.is_finite() || !state.is_finite() {
            return false;
        }
        self.initial_state.copy_from(state);
        self.step(system, t, state, dt);
        if state.is_finite() {
            true
        } else {
            state.copy_from(&self.initial_state);
            false
        }
    }
}

impl<State: VectorOperations> GaussLegendre4<State> {
    #[verified_engine::verified]
    fn step_single<S>(&mut self, system: &S, t: f64, state: &mut State, dt: f64)
    where
        S: OdeSystem<State> + ?Sized,
    {
        system.derivative_in_place(t, state, &mut self.k1);
        self.k2.copy_from(&self.k1);

        let t1 = t + GL4_C1 * dt;
        let t2 = t + GL4_C2 * dt;
        let iters = self.max_iters.clamp(1, Self::MAX_ALLOWED_ITERS);

        for _ in 0..iters {
            self.stage.copy_from_scaled(state, &self.k1, GL4_A11 * dt);
            self.stage.scale_add(&self.k2, GL4_A12 * dt);
            system.derivative_in_place(t1, &self.stage, &mut self.k1_next);

            self.stage.copy_from_scaled(state, &self.k1, GL4_A21 * dt);
            self.stage.scale_add(&self.k2, GL4_A22 * dt);
            system.derivative_in_place(t2, &self.stage, &mut self.k2_next);

            self.k1.copy_from(&self.k1_next);
            self.k2.copy_from(&self.k2_next);
        }

        let half_dt = 0.5 * dt;
        state.scale_add(&self.k1, half_dt);
        state.scale_add(&self.k2, half_dt);
    }
}

impl<State: VectorOperations> Solver<State> for GaussLegendre4<State> {
    #[verified_engine::verified]
    fn step<S>(&mut self, system: &S, t: f64, state: &mut State, dt: f64)
    where
        S: OdeSystem<State> + ?Sized,
    {
        let n_sub = self.substeps.clamp(1, Self::MAX_ALLOWED_SUBSTEPS);
        let sub_dt = dt / (n_sub as f64);
        for sub in 0..n_sub {
            self.step_single(system, t + (sub as f64) * sub_dt, state, sub_dt);
        }
    }
}

#[cfg(test)]
#[path = "symplectic_tests.rs"]
mod tests;
