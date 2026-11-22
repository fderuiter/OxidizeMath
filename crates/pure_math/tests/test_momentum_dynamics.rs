//! Integration tests verifying zero heap allocations, symplectic energy conservation,
//! Rosenbrock slingshots, and Rastrigin thermal hopping for 2D momentum engines.

use nalgebra::Vector2;
use oxidize_core::rng::OxidizeRng;
use pure_math::pure_math::analysis::evolution::EvolutionEngine;
use pure_math::pure_math::analysis::momentum_dynamics::{
    HamiltonianLeapfrog2D, LangevinBaoab2D, MAX_MASS, MAX_TEMPERATURE, MIN_DAMPING, MIN_DT,
    MIN_MASS, PhasePoint2D, clamp_damping, clamp_dt, clamp_mass, clamp_temperature,
};
use pure_math::pure_math::analysis::optimization::{Adam, AdamState2D, SGD};
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

#[test]
#[verified_engine::verified]
fn test_2d_momentum_and_optimizer_engines_zero_allocations() {
    let mut rng = OxidizeRng::new(0xCAFE_F00D);
    let mut sgd = SGD::new(0.01_f64);
    let mut adam = Adam::<f64, ()>::new(0.02).expect("valid Adam");
    let mut adam_state = AdamState2D::new();
    let mut leapfrog = HamiltonianLeapfrog2D::new(1.2, 0.25);
    let mut langevin = LangevinBaoab2D::new(1.2, 1.0, 2.5);

    let mut q_sgd = Vector2::new(-1.0, 1.5);
    let mut q_adam = Vector2::new(-1.0, 1.5);
    let mut phase_h = PhasePoint2D::from_velocity(q_sgd, Vector2::new(0.8, -0.4), 1.2);
    let mut phase_l = PhasePoint2D::at_rest(q_sgd);

    let (_res, metrics) = VerifiedEngine::run_verified(|| {
        let mut grad_fn = |q: Vector2<f64>| {
            Vector2::new(
                -2.0 * (1.0 - q.x) - 40.0 * q.x * (q.y - q.x * q.x),
                20.0 * (q.y - q.x * q.x),
            )
        };
        for _ in 0..64 {
            let mut g_s = grad_fn(q_sgd);
            sgd.step(&mut q_sgd, &mut g_s, &mut rng, 0.005).unwrap();

            let g_a = grad_fn(q_adam);
            let mut aux_a = (g_a, &mut adam_state);
            adam.step(&mut q_adam, &mut aux_a, &mut rng, 0.01).unwrap();

            leapfrog
                .step(&mut phase_h, &mut grad_fn, &mut rng, 0.005)
                .unwrap();
            langevin
                .step(&mut phase_l, &mut grad_fn, &mut rng, 0.005)
                .unwrap();

            phase_h.reflect_bounds(Vector2::new(-4.0, -4.0), Vector2::new(4.0, 4.0), 0.5);
            phase_l.reflect_bounds(Vector2::new(-4.0, -4.0), Vector2::new(4.0, 4.0), 0.5);
        }
    });

    assert!(q_sgd.x.is_finite() && q_adam.x.is_finite());
    assert!(phase_h.is_finite() && phase_l.is_finite());
    assert_eq!(metrics.memory_loads, 0);
}

#[test]
#[verified_engine::verified]
fn test_symplectic_leapfrog_conserves_harmonic_energy() {
    let mut state = PhasePoint2D::new(Vector2::new(1.5, -0.5), Vector2::new(0.2, 1.1));
    let mut solver = HamiltonianLeapfrog2D::new(2.0, 0.0);
    let mut rng = OxidizeRng::new(7);
    let mut grad = |q: Vector2<f64>| q;
    let e0 = 0.5 * state.q.norm_squared() + state.kinetic_energy(solver.mass);

    let (_res, metrics) = VerifiedEngine::run_verified(|| {
        solver
            .evolve::<_, 500>(&mut state, &mut grad, &mut rng, 0.01)
            .expect("leapfrog evolve");
    });
    let e1 = 0.5 * state.q.norm_squared() + state.kinetic_energy(solver.mass);
    assert!((e1 - e0).abs() < 1e-3);
    assert_eq!(metrics.memory_loads, 0);
}

#[test]
#[verified_engine::verified]
fn test_rosenbrock_slingshot_beats_sgd() {
    let start = Vector2::new(-1.2, 1.0);
    let target = Vector2::new(1.0, 1.0);
    let rosen_grad = |q: Vector2<f64>| {
        let dw = -2.0 * (1.0 - q.x) - 400.0 * q.x * (q.y - q.x * q.x);
        let db = 200.0 * (q.y - q.x * q.x);
        Vector2::new(dw, db)
    };

    let mut sgd = SGD::new(0.002);
    let mut q_sgd = start;
    let mut rng = OxidizeRng::new(11);
    let mut leap = HamiltonianLeapfrog2D::new(1.0, 1.2);
    let mut phase = PhasePoint2D::from_velocity(start, Vector2::new(1.8, -0.5), 1.0);

    let (_res, metrics) = VerifiedEngine::run_verified(|| {
        for _ in 0..250 {
            let mut g = rosen_grad(q_sgd);
            sgd.step(&mut q_sgd, &mut g, &mut rng, 0.002).expect("sgd");
        }
        let mut g_fn = rosen_grad;
        leap.evolve::<_, 250>(&mut phase, &mut g_fn, &mut rng, 0.004)
            .expect("leapfrog");
    });

    assert!((phase.q - target).norm() < (q_sgd - target).norm());
    assert_eq!(metrics.memory_loads, 0);
}

#[test]
#[verified_engine::verified]
fn test_langevin_hops_rastrigin_local_minimum() {
    let local_min = Vector2::new(0.994_958_6, 0.994_958_6);
    let rastrigin_loss = |q: Vector2<f64>| {
        let pi2 = std::f64::consts::TAU;
        6.0 + (q.x * q.x - 3.0 * (pi2 * q.x).cos()) + (q.y * q.y - 3.0 * (pi2 * q.y).cos())
    };
    let rastrigin_grad = |q: Vector2<f64>| {
        let pi2 = std::f64::consts::TAU;
        Vector2::new(
            2.0 * q.x + 3.0 * pi2 * (pi2 * q.x).sin(),
            2.0 * q.y + 3.0 * pi2 * (pi2 * q.y).sin(),
        )
    };

    let mut sgd = SGD::new(0.01);
    let mut q_sgd = local_min;
    let mut adam = Adam::<f64, ()>::new(0.01).expect("adam");
    let mut q_adam = local_min;
    let mut adam_state = AdamState2D::new();
    let mut langevin = LangevinBaoab2D::new(0.8, 1.5, 8.0);
    let mut phase = PhasePoint2D::at_rest(local_min);
    let mut rng = OxidizeRng::new(2026);
    let mut best_langevin_loss = rastrigin_loss(phase.q);

    let (_res, metrics) = VerifiedEngine::run_verified(|| {
        for _ in 0..400 {
            let mut gs = rastrigin_grad(q_sgd);
            sgd.step(&mut q_sgd, &mut gs, &mut rng, 0.01).expect("sgd");
            let ga = rastrigin_grad(q_adam);
            let mut aux_a = (ga, &mut adam_state);
            adam.step(&mut q_adam, &mut aux_a, &mut rng, 0.01)
                .expect("adam");
            let mut gl = rastrigin_grad;
            langevin
                .step(&mut phase, &mut gl, &mut rng, 0.02)
                .expect("langevin");
            best_langevin_loss = best_langevin_loss.min(rastrigin_loss(phase.q));
        }
    });

    assert!((q_sgd - local_min).norm() < 0.05);
    assert!(best_langevin_loss + 0.5 < rastrigin_loss(q_sgd));
    assert!(best_langevin_loss + 0.5 < rastrigin_loss(q_adam));
    assert_eq!(metrics.memory_loads, 0);
}

#[test]
#[verified_engine::verified]
fn test_parameter_clamping_and_inelastic_reflection() {
    assert_eq!(clamp_mass(0.0), MIN_MASS);
    assert_eq!(clamp_mass(500.0), MAX_MASS);
    assert_eq!(clamp_damping(-2.0), MIN_DAMPING);
    assert_eq!(clamp_temperature(999.0), MAX_TEMPERATURE);
    assert_eq!(clamp_dt(0.0), MIN_DT);

    let mut pt = PhasePoint2D::new(Vector2::new(-1.5, 4.2), Vector2::new(-3.0, 2.0));
    pt.reflect_bounds(Vector2::new(-1.0, -1.0), Vector2::new(3.0, 3.0), 0.5);
    assert_eq!(pt.q, Vector2::new(-1.0, 3.0));
    assert_eq!(pt.p, Vector2::new(1.5, -1.0));
}
