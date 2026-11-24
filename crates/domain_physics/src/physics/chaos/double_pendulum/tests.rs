use super::*;
use pure_math::pure_math::analysis::ode::GaussLegendre4;

#[test]
#[verified_engine::verified]
fn test_double_pendulum_gauss_legendre4_beats_rk4_over_10000_steps() {
    let pendulum = DoublePendulum::default();
    let initial = DoublePendulum::default_chaotic_state();
    let e0 = pendulum.hamiltonian(&initial);
    let dt = 0.02;

    let mut s_gl4 = initial;
    let mut s_rk4 = initial;
    let mut gl4 = GaussLegendre4::new(&initial);
    let mut rk4 = RungeKutta4::new(&initial);

    let mut max_gl4_pct = 0.0_f64;
    for step in 0..10_000 {
        let t = step as f64 * dt;
        gl4.step(&pendulum, t, &mut s_gl4, dt);
        rk4.step(&pendulum, t, &mut s_rk4, dt);
        let step_gl4_pct = pendulum.relative_energy_error(&s_gl4, e0) * 100.0;
        if step_gl4_pct > max_gl4_pct {
            max_gl4_pct = step_gl4_pct;
        }
    }

    let rel_gl4_pct = pendulum.relative_energy_error(&s_gl4, e0) * 100.0;
    let rel_rk4_pct = pendulum.relative_energy_error(&s_rk4, e0) * 100.0;

    assert!(
        max_gl4_pct < 0.001 && rel_gl4_pct < 0.001,
        "GaussLegendre4 peak ({max_gl4_pct:.6}%) and final ({rel_gl4_pct:.6}%) error must stay < 0.001%"
    );
    assert!(
        rel_rk4_pct > 3.0,
        "RungeKutta4 expected secular energy loss (>3%), got {rel_rk4_pct:.4}%"
    );
}

#[test]
#[verified_engine::verified]
fn test_double_pendulum_poincare_seed_energy_shell() {
    let pendulum = DoublePendulum::default();
    let target_energy = 15.0;
    let seeded = pendulum
        .seed_on_poincare_section(target_energy, 0.35, 0.8)
        .expect("valid state inside energy shell");
    assert!(
        (pendulum.hamiltonian(&seeded) - target_energy).abs() < 1e-12,
        "Seeded state must lie on exact energy shell"
    );
    let (dth1, _) = pendulum.angular_velocities(&seeded);
    assert!(dth1 >= 0.0, "Positive crossing requires dtheta1 >= 0");
    assert!(pendulum.seed_on_poincare_section(1.0, 2.8, 10.0).is_none());
}
