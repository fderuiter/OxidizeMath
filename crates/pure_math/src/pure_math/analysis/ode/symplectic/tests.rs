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
    let mut s_midpoint = y0;
    let mut s_yoshida = y0;
    let mut s_gl4 = y0;
    let mut s_rk4 = y0;

    let mut verlet = StormerVerlet::new(&ArrayState([0.0]));
    let mut midpoint = ImplicitMidpoint::new(&y0);
    let mut yoshida = Yoshida4::new(&ArrayState([0.0]));
    let mut gl4 = GaussLegendre4::new(&y0);
    let mut rk4 = RungeKutta4::new(&y0);

    for i in 0..steps {
        let t = i as f64 * dt;
        verlet.step_separable(&sys, &mut s_verlet, dt);
        midpoint.step(&sys, t, &mut s_midpoint, dt);
        yoshida.step_separable(&sys, &mut s_yoshida, dt);
        gl4.step(&sys, t, &mut s_gl4, dt);
        rk4.step(&sys, t, &mut s_rk4, dt);
    }

    let err_verlet = sys.relative_energy_error(&s_verlet, e0);
    let err_midpoint = sys.relative_energy_error(&s_midpoint, e0);
    let err_yoshida = sys.relative_energy_error(&s_yoshida, e0);
    let err_gl4 = sys.relative_energy_error(&s_gl4, e0);
    let err_rk4 = sys.relative_energy_error(&s_rk4, e0);

    assert!(
        err_verlet < 5e-3,
        "Verlet bounded shadow error: {err_verlet}"
    );
    assert!(
        err_midpoint < 2e-3,
        "ImplicitMidpoint bounded shadow error: {err_midpoint}"
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
    assert_eq!(SymplecticMethod::ALL.len(), 4);
    for method in SymplecticMethod::ALL {
        assert!(!method.name().is_empty());
    }
    let a = ArrayState::zeros();
    let b = ArrayState([1.5, -2.25]);
    assert!(a.is_finite() && b.is_finite());
    assert!((a.max_abs_diff(&b) - 2.25).abs() < 1e-12);
}
