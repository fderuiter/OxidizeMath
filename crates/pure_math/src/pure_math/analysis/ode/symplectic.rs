//! Zero-allocation symplectic numerical integrators for conservative Hamiltonian ODEs.
//!
//! Provides explicit splitting solvers ([`StormerVerlet`] and [`Yoshida4`]) for
//! [`SeparableHamiltonian`] systems alongside the implicit 2-stage 4th-order
//! [`GaussLegendre4`] Runge-Kutta solver for general (including non-separable)
//! [`OdeSystem`] and [`super::traits::HamiltonianSystem`] dynamics.

// [cite:advanced_linear_algebra]

use serde::{Deserialize, Serialize};

use super::traits::{OdeSystem, SeparableHamiltonian, Solver, VectorOperations};

/// Symplectic numerical integration schemes for conservative Hamiltonian systems.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash, Serialize, Deserialize)]
pub enum SymplecticMethod {
    /// 2nd-order explicit velocity Störmer-Verlet (leapfrog) splitting for separable Hamiltonians.
    StormerVerlet,
    /// 4th-order explicit Yoshida (1990) symmetric triple-Verlet composition for separable Hamiltonians.
    #[default]
    Yoshida4,
    /// 4th-order 2-stage implicit Gauss-Legendre Runge-Kutta for general Hamiltonians.
    GaussLegendre4,
}

impl SymplecticMethod {
    /// All supported symplectic integration methods.
    pub const ALL: [Self; 3] = [Self::StormerVerlet, Self::Yoshida4, Self::GaussLegendre4];

    /// Human-readable display label for UI selectors and telemetry.
    #[must_use]
    #[verified_engine::verified]
    pub fn name(&self) -> &'static str {
        match self {
            Self::StormerVerlet => "Störmer-Verlet (2nd)",
            Self::Yoshida4 => "Yoshida 4th Order",
            Self::GaussLegendre4 => "Gauss-Legendre 4th",
        }
    }
}

/// Second-order symplectic velocity Störmer-Verlet (leapfrog) integrator.
///
/// Pre-allocates canonical coordinate and momentum scratch buffers at construction so
/// stepping triggers zero dynamic heap allocations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
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

/// Fourth-order explicit symplectic integrator via Yoshida (1990) triple-Verlet composition.
///
/// Composes three velocity-Verlet stages with substeps $(w_1 \Delta t, w_0 \Delta t, w_1 \Delta t)$,
/// fusing adjacent half-kicks so each step requires only four force evaluations and zero heap allocations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
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

/// Fourth-order 2-stage implicit Gauss-Legendre Runge-Kutta symplectic integrator.
///
/// Preserves canonical symplectic 2-forms for **arbitrary** (including non-separable)
/// Hamiltonian systems such as the chaotic double pendulum. Solves the coupled stage
/// equations via bounded fixed-point iteration in pre-allocated stack/state buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GaussLegendre4<State> {
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

    /// Upper cap on fixed-point stage iterations to enforce bounded loop execution.
    pub const MAX_ALLOWED_ITERS: usize = 64;

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

    /// Constructs a new `GaussLegendre4` solver with an explicit fixed-point iteration bound.
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
            k1: example_state.clone(),
            k2: example_state.clone(),
            k1_next: example_state.clone(),
            k2_next: example_state.clone(),
            stage: example_state.clone(),
            max_iters: max_iters.clamp(1, Self::MAX_ALLOWED_ITERS),
            substeps: substeps.clamp(1, Self::MAX_ALLOWED_SUBSTEPS),
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
mod tests {
    use super::*;
    use crate::pure_math::analysis::ode::{ArrayState, HamiltonianSystem, RungeKutta4};

    struct QuarticOscillator;

    impl OdeSystem<ArrayState<2>> for QuarticOscillator {
        #[verified_engine::verified]
        fn derivative(&self, _t: f64, s: &ArrayState<2>) -> ArrayState<2> {
            let q = s.0[0];
            let p = s.0[1];
            ArrayState([p, -(q + q * q * q)])
        }

        #[verified_engine::verified]
        fn derivative_in_place(&self, _t: f64, s: &ArrayState<2>, out: &mut ArrayState<2>) {
            let q = s.0[0];
            out.0[0] = s.0[1];
            out.0[1] = -(q + q * q * q);
        }
    }

    impl HamiltonianSystem<ArrayState<2>> for QuarticOscillator {
        #[verified_engine::verified]
        fn hamiltonian(&self, s: &ArrayState<2>) -> f64 {
            let q = s.0[0];
            let p = s.0[1];
            0.5 * p * p + 0.5 * q * q + 0.25 * q * q * q * q
        }
    }

    impl SeparableHamiltonian<ArrayState<2>> for QuarticOscillator {
        type Coord = ArrayState<1>;

        #[verified_engine::verified]
        fn split_canonical(&self, s: &ArrayState<2>, q: &mut Self::Coord, p: &mut Self::Coord) {
            q.0[0] = s.0[0];
            p.0[0] = s.0[1];
        }

        #[verified_engine::verified]
        fn join_canonical(&self, q: &Self::Coord, p: &Self::Coord, s: &mut ArrayState<2>) {
            s.0[0] = q.0[0];
            s.0[1] = p.0[0];
        }

        #[verified_engine::verified]
        fn kinetic_energy(&self, p: &Self::Coord) -> f64 {
            0.5 * p.0[0] * p.0[0]
        }

        #[verified_engine::verified]
        fn potential_energy(&self, q: &Self::Coord) -> f64 {
            let x = q.0[0];
            0.5 * x * x + 0.25 * x * x * x * x
        }

        #[verified_engine::verified]
        fn grad_kinetic_in_place(&self, p: &Self::Coord, dq_dt: &mut Self::Coord) {
            dq_dt.0[0] = p.0[0];
        }

        #[verified_engine::verified]
        fn neg_grad_potential_in_place(&self, q: &Self::Coord, dp_dt: &mut Self::Coord) {
            let x = q.0[0];
            dp_dt.0[0] = -(x + x * x * x);
        }
    }

    #[test]
    #[verified_engine::verified]
    fn test_symplectic_order_hierarchy_and_energy_bounds() {
        let sys = QuarticOscillator;
        let y0 = ArrayState([1.4, 0.0]);
        let e0 = sys.hamiltonian(&y0);
        let dt = 0.05;
        let steps = 4_000;

        let mut s_verlet = y0;
        let mut s_yoshida = y0;
        let mut s_gl4 = y0;
        let mut s_rk4 = y0;

        let mut verlet = StormerVerlet::new(&ArrayState([0.0]));
        let mut yoshida = Yoshida4::new(&ArrayState([0.0]));
        let mut gl4 = GaussLegendre4::new(&y0);
        let mut rk4 = RungeKutta4::new(&y0);

        for i in 0..steps {
            let t = i as f64 * dt;
            verlet.step_separable(&sys, &mut s_verlet, dt);
            yoshida.step_separable(&sys, &mut s_yoshida, dt);
            gl4.step(&sys, t, &mut s_gl4, dt);
            rk4.step(&sys, t, &mut s_rk4, dt);
        }

        let err_verlet = sys.relative_energy_error(&s_verlet, e0);
        let err_yoshida = sys.relative_energy_error(&s_yoshida, e0);
        let err_gl4 = sys.relative_energy_error(&s_gl4, e0);
        let err_rk4 = sys.relative_energy_error(&s_rk4, e0);

        assert!(
            err_verlet < 5e-3,
            "Verlet bounded shadow error: {err_verlet}"
        );
        assert!(err_yoshida < 5e-6, "Yoshida4 bounded error: {err_yoshida}");
        assert!(err_gl4 < 5e-7, "GaussLegendre4 bounded error: {err_gl4}");
        assert!(
            err_gl4 * 50.0 < err_rk4,
            "GL4 ({err_gl4}) must outperform secular RK4 drift ({err_rk4})"
        );
    }

    #[test]
    #[verified_engine::verified]
    fn test_symplectic_method_metadata_and_array_helpers() {
        assert_eq!(SymplecticMethod::ALL.len(), 3);
        for method in SymplecticMethod::ALL {
            assert!(!method.name().is_empty());
        }
        let a = ArrayState::zeros();
        let b = ArrayState([1.5, -2.25]);
        assert!(a.is_finite() && b.is_finite());
        assert!((a.max_abs_diff(&b) - 2.25).abs() < 1e-12);
    }
}
