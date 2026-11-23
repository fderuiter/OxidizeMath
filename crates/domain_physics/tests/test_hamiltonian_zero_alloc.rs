//! VerifiedAllocator lock tests confirming zero dynamic heap allocations across
//! 1,000 symplectic integration, Chirikov standard map, cubic-Hermite Poincaré section,
//! and flat Euclidean billiard collision steps.

use domain_physics::physics::astrophysics::nbody::FigureEightThreeBody;
use domain_physics::physics::chaos::billiards::{
    BilliardRay, BilliardTable, CircularBilliard, EllipticalBilliard, SinaiBilliard,
    StadiumBilliard,
};
use domain_physics::physics::chaos::double_pendulum::DoublePendulum;
use domain_physics::physics::chaos::hamiltonian::ChirikovStandardMap;
use domain_physics::physics::chaos::henon_heiles::HenonHeiles;
use domain_physics::physics::chaos::poincare::PoincareSection;
use pure_math::pure_math::analysis::ode::{
    ArrayState, GaussLegendre4, HamiltonianSystem, ImplicitMidpoint, OdeSystem, Solver,
    StormerVerlet, VectorOperations, Yoshida4,
};
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

#[test]
fn test_1000_double_pendulum_and_chirikov_zero_heap_allocations() {
    let pendulum = DoublePendulum::default();
    let mut dp_state = DoublePendulum::default_chaotic_state();
    let mut dp_mid_state = dp_state;
    let mut gl4 = GaussLegendre4::new(&dp_state);
    let mut midpoint = ImplicitMidpoint::new(&dp_mid_state);
    let dp_sec = PoincareSection::<4>::double_pendulum();
    let dp_e0 = pendulum.hamiltonian(&dp_state);

    let chirikov = ChirikovStandardMap::default();
    let mut ch_state = ChirikovStandardMap::seed_on_torus(1.1, 2.2).unwrap();

    let (dp_hits, _metrics) = VerifiedEngine::run_verified(|| {
        let mut hits = 0_usize;
        let dt = 0.02;
        let mut dy0 = ArrayState::<4>::zeros();
        let mut dy1 = ArrayState::<4>::zeros();
        for step in 0..1_000 {
            let t = step as f64 * dt;
            let mut next = dp_state;
            pendulum.derivative_in_place(t, &dp_state, &mut dy0);
            gl4.step(&pendulum, t, &mut next, dt);
            pendulum.derivative_in_place(t + dt, &next, &mut dy1);
            if dp_sec
                .detect_crossing(t, dt, &dp_state, &next, &dy0, &dy1)
                .is_some()
            {
                hits += 1;
            }
            dp_state.copy_from(&next);
            midpoint.step(&pendulum, t, &mut dp_mid_state, 0.5 * dt);
            chirikov.step_in_place(&mut ch_state);
        }
        hits
    });

    assert!(dp_hits > 0);
    assert!(pendulum.relative_energy_error(&dp_state, dp_e0) < 1e-5);
    assert!(pendulum.relative_energy_error(&dp_mid_state, dp_e0) < 2e-3);
    assert!(ch_state.is_finite());
}

#[test]
fn test_1000_separable_hamiltonians_and_poincare_zero_heap_allocations() {
    let hh = HenonHeiles::default();
    let mut hh_state = hh.seed_on_poincare_section(0.125, 0.12, 0.04).unwrap();
    let mut yoshida_hh = Yoshida4::new(&ArrayState::<2>::zeros());
    let hh_sec = PoincareSection::<4>::henon_heiles();

    let nbody = FigureEightThreeBody::default();
    let mut nb_state = FigureEightThreeBody::figure_eight_initial_state();
    let mut verlet_nb = StormerVerlet::new(&ArrayState::<6>::zeros());
    let mut yoshida_nb = Yoshida4::new(&ArrayState::<6>::zeros());
    let nb_sec = PoincareSection::<12>::figure_eight();

    let ((hh_hits, nb_hits), _metrics) = VerifiedEngine::run_verified(|| {
        let mut counts = (0_usize, 0_usize);
        let dt = 0.02;
        let mut hh_dy0 = ArrayState::<4>::zeros();
        let mut hh_dy1 = ArrayState::<4>::zeros();
        let mut nb_dy0 = ArrayState::<12>::zeros();
        let mut nb_dy1 = ArrayState::<12>::zeros();

        for step in 0..1_000 {
            let t = step as f64 * dt;
            let hh_prev = hh_state;
            hh.derivative_in_place(t, &hh_prev, &mut hh_dy0);
            yoshida_hh.step_separable(&hh, &mut hh_state, dt);
            hh.derivative_in_place(t + dt, &hh_state, &mut hh_dy1);
            counts.0 += usize::from(
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
            counts.1 += usize::from(
                nb_sec
                    .detect_crossing(t, dt, &nb_prev, &nb_state, &nb_dy0, &nb_dy1)
                    .is_some(),
            );
        }
        counts
    });

    assert!(hh_hits > 0 && nb_hits > 0);
}

#[test]
fn test_1000_euclidean_billiard_collisions_zero_heap_allocations() {
    let tables = [
        BilliardTable::Circle(CircularBilliard::new(1.0)),
        BilliardTable::Ellipse(EllipticalBilliard::new(1.45, 1.0)),
        BilliardTable::Stadium(StadiumBilliard::new(0.8, 1.0)),
        BilliardTable::Sinai(SinaiBilliard::new(1.0, 0.35)),
    ];
    let mut rays = [
        BilliardRay::from_angle([0.25, -0.15], 0.62).unwrap(),
        BilliardRay::from_angle([0.2, 0.3], 1.05).unwrap(),
        BilliardRay::from_angle([0.1, 0.25], 0.74).unwrap(),
        BilliardRay::from_angle([0.65, 0.2], 2.41).unwrap(),
    ];

    let (total_hits, _metrics) = VerifiedEngine::run_verified(|| {
        let mut hits = 0_usize;
        for _ in 0..1_000 {
            for (table, ray) in tables.iter().zip(rays.iter_mut()) {
                if table.collide(ray).is_some() {
                    hits += 1;
                }
            }
        }
        hits
    });

    assert_eq!(total_hits, 4_000);
    for ray in &rays {
        assert!((ray.speed() - 1.0).abs() < 1e-14);
    }
}
