//! VerifiedAllocator lock tests confirming zero dynamic heap allocations across
//! 1,000 symplectic integration and cubic-Hermite Poincaré section detection steps.

use domain_physics::physics::astrophysics::nbody::FigureEightThreeBody;
use domain_physics::physics::chaos::double_pendulum::DoublePendulum;
use domain_physics::physics::chaos::henon_heiles::HenonHeiles;
use domain_physics::physics::chaos::poincare::PoincareSection;
use pure_math::pure_math::analysis::ode::{
    ArrayState, GaussLegendre4, HamiltonianSystem, OdeSystem, Solver, StormerVerlet,
    VectorOperations, Yoshida4,
};
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

fn step_double_pendulum_and_poincare(
    pendulum: &DoublePendulum,
    sec: &PoincareSection<4>,
    gl4: &mut GaussLegendre4<ArrayState<4>>,
    state: &mut ArrayState<4>,
    t: f64,
    dt: f64,
) -> usize {
    let mut dy0 = ArrayState::<4>::zeros();
    let mut dy1 = ArrayState::<4>::zeros();
    let mut next = *state;
    pendulum.derivative_in_place(t, state, &mut dy0);
    gl4.step(pendulum, t, &mut next, dt);
    pendulum.derivative_in_place(t + dt, &next, &mut dy1);
    let hit = usize::from(
        sec.detect_crossing(t, dt, state, &next, &dy0, &dy1)
            .is_some(),
    );
    state.copy_from(&next);
    hit
}

#[test]
fn test_1000_symplectic_and_poincare_steps_zero_heap_allocations() {
    let pendulum = DoublePendulum::default();
    let mut dp_state = DoublePendulum::default_chaotic_state();
    let mut gl4 = GaussLegendre4::new(&dp_state);
    let dp_sec = PoincareSection::<4>::double_pendulum();
    let dp_e0 = pendulum.hamiltonian(&dp_state);

    let hh = HenonHeiles::default();
    let mut hh_state = hh.seed_on_poincare_section(0.125, 0.12, 0.04).unwrap();
    let mut yoshida_hh = Yoshida4::new(&ArrayState::<2>::zeros());
    let hh_sec = PoincareSection::<4>::henon_heiles();

    let nbody = FigureEightThreeBody::default();
    let mut nb_state = FigureEightThreeBody::figure_eight_initial_state();
    let mut verlet_nb = StormerVerlet::new(&ArrayState::<6>::zeros());
    let mut yoshida_nb = Yoshida4::new(&ArrayState::<6>::zeros());
    let nb_sec = PoincareSection::<12>::figure_eight();

    let ((dp_hits, hh_hits, nb_hits), _metrics) = VerifiedEngine::run_verified(|| {
        let mut counts = (0_usize, 0_usize, 0_usize);
        let dt = 0.02;
        let mut hh_dy0 = ArrayState::<4>::zeros();
        let mut hh_dy1 = ArrayState::<4>::zeros();
        let mut nb_dy0 = ArrayState::<12>::zeros();
        let mut nb_dy1 = ArrayState::<12>::zeros();

        for step in 0..1_000 {
            let t = step as f64 * dt;
            counts.0 += step_double_pendulum_and_poincare(
                &pendulum,
                &dp_sec,
                &mut gl4,
                &mut dp_state,
                t,
                dt,
            );

            let hh_prev = hh_state;
            hh.derivative_in_place(t, &hh_prev, &mut hh_dy0);
            yoshida_hh.step_separable(&hh, &mut hh_state, dt);
            hh.derivative_in_place(t + dt, &hh_state, &mut hh_dy1);
            counts.1 += usize::from(
                hh_sec
                    .detect_crossing(t, dt, &hh_prev, &hh_state, &hh_dy0, &hh_dy1)
                    .is_some(),
            );

            let nb_prev = nb_state;
            nbody.derivative_in_place(t, &nb_prev, &mut nb_dy0);
            if step % 2 == 0 {
                yoshida_nb.step_separable(&nbody, &mut nb_state, dt);
            } else {
                verlet_nb.step_separable(&nbody, &mut nb_state, dt);
            }
            nbody.derivative_in_place(t + dt, &nb_state, &mut nb_dy1);
            counts.2 += usize::from(
                nb_sec
                    .detect_crossing(t, dt, &nb_prev, &nb_state, &nb_dy0, &nb_dy1)
                    .is_some(),
            );
        }
        counts
    });

    assert!(dp_hits > 0 && hh_hits > 0 && nb_hits > 0);
    assert!(pendulum.relative_energy_error(&dp_state, dp_e0) < 1e-5);
}
