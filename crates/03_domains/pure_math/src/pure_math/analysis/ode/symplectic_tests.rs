use super::*;
use crate::pure_math::analysis::ode::{ArrayState, HamiltonianSystem, RungeKutta4};

pub(super) struct QuarticOscillator;

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

struct ExplosiveOde;

impl OdeSystem<ArrayState<2>> for ExplosiveOde {
    #[verified_engine::verified]
    fn derivative(&self, _t: f64, _s: &ArrayState<2>) -> ArrayState<2> {
        ArrayState([f64::NAN, f64::INFINITY])
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

    let mut s_mid = y0;
    let mut s_verlet = y0;
    let mut s_yoshida = y0;
    let mut s_gl4 = y0;
    let mut s_rk4 = y0;

    let mut mid = ImplicitMidpoint::new(&y0);
    let mut verlet = StormerVerlet::new(&ArrayState([0.0]));
    let mut yoshida = Yoshida4::new(&ArrayState([0.0]));
    let mut gl4 = GaussLegendre4::new(&y0);
    let mut rk4 = RungeKutta4::new(&y0);

    for i in 0..steps {
        let t = i as f64 * dt;
        mid.step(&sys, t, &mut s_mid, dt);
        verlet.step_separable(&sys, &mut s_verlet, dt);
        yoshida.step_separable(&sys, &mut s_yoshida, dt);
        gl4.step(&sys, t, &mut s_gl4, dt);
        rk4.step(&sys, t, &mut s_rk4, dt);
    }

    let err_mid = sys.relative_energy_error(&s_mid, e0);
    let err_verlet = sys.relative_energy_error(&s_verlet, e0);
    let err_yoshida = sys.relative_energy_error(&s_yoshida, e0);
    let err_gl4 = sys.relative_energy_error(&s_gl4, e0);
    let err_rk4 = sys.relative_energy_error(&s_rk4, e0);

    assert!(
        err_mid < 5e-3,
        "ImplicitMidpoint bounded shadow error: {err_mid}"
    );
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
fn test_symplectic_method_metadata_iteration_caps_and_rollback() {
    assert_eq!(SymplecticMethod::ALL.len(), 4);
    for method in SymplecticMethod::ALL {
        assert!(!method.name().is_empty());
    }

    let seed = ArrayState([1.5, -2.25]);
    let mut mid = ImplicitMidpoint::with_max_iters(&seed, 99);
    let mut gl4 = GaussLegendre4::with_max_iters(&seed, 99);
    assert_eq!(mid.max_iters(), MAX_FIXED_POINT_ITERS);
    assert_eq!(gl4.max_iters(), MAX_FIXED_POINT_ITERS);

    let mut s_mid = seed;
    let mut s_gl4 = seed;
    assert!(!mid.try_step_array(&ExplosiveOde, 0.0, &mut s_mid, 0.02));
    assert!(!gl4.try_step_array(&ExplosiveOde, 0.0, &mut s_gl4, 0.02));
    assert_eq!(s_mid, seed);
    assert_eq!(s_gl4, seed);
}

#[test]
#[verified_engine::verified]
fn test_symplectic_dt_refinement_convergence_orders() {
    let sys = QuarticOscillator;
    let y0 = ArrayState([1.1, 0.2]);
    let t_end = 1.0;
    let mut exact = y0;
    let mut ref_gl4 = GaussLegendre4::new(&y0);
    let ref_steps = 2_000usize;
    let ref_dt = t_end / ref_steps as f64;
    for step in 0..ref_steps {
        ref_gl4.step(&sys, step as f64 * ref_dt, &mut exact, ref_dt);
    }

    let eval = |steps: usize| -> [f64; 4] {
        let dt = t_end / steps as f64;
        let (mut im_s, mut sv_s, mut yo_s, mut gl_s) = (y0, y0, y0, y0);
        let mut im = ImplicitMidpoint::new(&y0);
        let mut sv = StormerVerlet::new(&ArrayState([0.0]));
        let mut yo = Yoshida4::new(&ArrayState([0.0]));
        let mut gl = GaussLegendre4::new(&y0);
        for k in 0..steps {
            let t = k as f64 * dt;
            im.step(&sys, t, &mut im_s, dt);
            sv.step_separable(&sys, &mut sv_s, dt);
            yo.step_separable(&sys, &mut yo_s, dt);
            gl.step(&sys, t, &mut gl_s, dt);
        }
        [
            im_s.max_abs_diff(&exact),
            sv_s.max_abs_diff(&exact),
            yo_s.max_abs_diff(&exact),
            gl_s.max_abs_diff(&exact),
        ]
    };

    let [e_im_1, e_sv_1, e_yo_1, e_gl_1] = eval(25);
    let [e_im_2, e_sv_2, e_yo_2, e_gl_2] = eval(50);
    let ln2 = 2.0_f64.ln();
    assert!((e_im_1 / e_im_2).ln() / ln2 > 1.95);
    assert!((e_sv_1 / e_sv_2).ln() / ln2 > 1.95);
    assert!((e_yo_1 / e_yo_2).ln() / ln2 > 3.90);
    assert!((e_gl_1 / e_gl_2).ln() / ln2 > 3.95);
}
