//! VerifiedAllocator lock tests confirming zero dynamic heap allocations across
//! 1,000 symplectic integration and cubic-Hermite Poincaré section detection steps.

use domain_physics::physics::chaos::metrics::{lyapunov_exponent_4d, lyapunov_exponent_6d};
use domain_physics::physics::chaos::poincare::{PoincareRingBuffer, PoincareSection};
use domain_physics::physics::chaos::{DoublePendulum, HenonHeilesSystem, ThreeBodyChoreography};
use pure_math::pure_math::analysis::ode::{
    ArrayState, GaussLegendre4, HamiltonianSystem, ImplicitMidpoint, OdeSystem, Solver,
    StormerVerlet, VectorOperations, Yoshida4,
};
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

struct Harmonic6D;

impl OdeSystem<ArrayState<6>> for Harmonic6D {
    fn derivative(&self, t: f64, s: &ArrayState<6>) -> ArrayState<6> {
        let mut out = ArrayState::zeros();
        self.derivative_in_place(t, s, &mut out);
        out
    }

    fn derivative_in_place(&self, _t: f64, s: &ArrayState<6>, out: &mut ArrayState<6>) {
        out.0 = [s.0[3], s.0[4], s.0[5], -s.0[0], -s.0[1], -s.0[2]];
    }
}

fn step_double_pendulum_and_poincare(
    pendulum: &DoublePendulum,
    sec: &PoincareSection<4>,
    gl4: &mut GaussLegendre4<ArrayState<4>>,
    ring: &mut PoincareRingBuffer<32>,
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
    let hit = if let Some(crossing) = sec.detect_crossing(t, dt, state, &next, &dy0, &dy1) {
        ring.push_crossing(&crossing);
        1
    } else {
        0
    };
    state.copy_from(&next);
    hit
}

fn step_separable_and_poincare<const N: usize, const M: usize, S>(
    sys: &S,
    sec: &PoincareSection<N>,
    yoshida: &mut Yoshida4<ArrayState<M>>,
    state: &mut ArrayState<N>,
    t: f64,
    dt: f64,
) -> usize
where
    S: pure_math::pure_math::analysis::ode::SeparableHamiltonian<
            ArrayState<N>,
            Coord = ArrayState<M>,
        >,
{
    let prev = *state;
    let mut dy0 = ArrayState::<N>::zeros();
    let mut dy1 = ArrayState::<N>::zeros();
    sys.derivative_in_place(t, &prev, &mut dy0);
    yoshida.step_separable(sys, state, dt);
    sys.derivative_in_place(t + dt, state, &mut dy1);
    usize::from(
        sec.detect_crossing(t, dt, &prev, state, &dy0, &dy1)
            .is_some(),
    )
}

#[test]
fn test_1000_symplectic_and_poincare_steps_zero_heap_allocations() {
    let pendulum = DoublePendulum::default();
    let mut dp_state = DoublePendulum::default_chaotic_state();
    let mut gl4 = GaussLegendre4::new(&dp_state);
    let dp_sec = PoincareSection::<4>::double_pendulum();
    let dp_e0 = pendulum.hamiltonian(&dp_state);

    let hh = HenonHeilesSystem::default();
    let mut hh_state = hh.seed_on_poincare_section(0.125, 0.12, 0.04).unwrap();
    let hh_e0 = hh.hamiltonian(&hh_state);
    let mut yoshida_hh = Yoshida4::new(&ArrayState::<2>::zeros());
    let hh_sec = PoincareSection::<4>::henon_heiles();

    let nbody = ThreeBodyChoreography::default();
    let mut nb_state = ThreeBodyChoreography::figure_eight_initial_state();
    let nb_e0 = nbody.hamiltonian(&nb_state);
    let nb_l0 = nbody.angular_momentum_z(&nb_state);
    let mut verlet_nb = StormerVerlet::new(&ArrayState::<6>::zeros());
    let mut yoshida_nb = Yoshida4::new(&ArrayState::<6>::zeros());
    let nb_sec = PoincareSection::<12>::figure_eight();

    let ((dp_hits, hh_hits, nb_hits, ring_len), _metrics) = VerifiedEngine::run_verified(|| {
        let mut counts = (0_usize, 0_usize, 0_usize, 0_usize);
        let dt = 0.01;
        let mut ring = PoincareRingBuffer::<32>::new();
        for step in 0..1_000 {
            let t = step as f64 * dt;
            counts.0 += step_double_pendulum_and_poincare(
                &pendulum,
                &dp_sec,
                &mut gl4,
                &mut ring,
                &mut dp_state,
                t,
                dt,
            );
            counts.1 +=
                step_separable_and_poincare(&hh, &hh_sec, &mut yoshida_hh, &mut hh_state, t, dt);
            counts.2 +=
                step_separable_and_poincare(&nbody, &nb_sec, &mut yoshida_nb, &mut nb_state, t, dt);
        }
        let mut nb_verlet_probe = ThreeBodyChoreography::figure_eight_initial_state();
        verlet_nb.step_separable(&nbody, &mut nb_verlet_probe, dt);
        counts.3 = ring.len();
        counts
    });

    assert!(dp_hits > 0 && hh_hits > 0 && nb_hits > 0 && ring_len > 0);
    assert!(pendulum.relative_energy_error(&dp_state, dp_e0) < 1e-5);
    assert!(hh.relative_energy_error(&hh_state, hh_e0) < 1e-8);
    assert!(nbody.relative_energy_error(&nb_state, nb_e0) < 1e-7);
    let lin_p = nbody.total_linear_momentum(&nb_state);
    assert!(lin_p[0].hypot(lin_p[1]) < 1e-12);
    assert!((nbody.angular_momentum_z(&nb_state) - nb_l0).abs() < 1e-11);
}

#[test]
fn test_implicit_midpoint_and_lyapunov_4d_6d_zero_heap_allocations() {
    let pendulum = DoublePendulum::default();
    let dp_seed = DoublePendulum::default_chaotic_state();
    let mut mid4 = ImplicitMidpoint::new(&dp_seed);

    let osc6 = Harmonic6D;
    let seed6 = ArrayState([1.0, 0.5, -0.5, 0.0, 0.2, -0.1]);
    let mut mid6 = ImplicitMidpoint::new(&seed6);

    let ((lam4, lam6), _metrics) = VerifiedEngine::run_verified(|| {
        let l4 = lyapunov_exponent_4d(&pendulum, &mut mid4, dp_seed, 0.02, 20, 0.2).unwrap();
        let l6 = lyapunov_exponent_6d(&osc6, &mut mid6, seed6, 0.05, 15, 0.25).unwrap();
        (l4, l6)
    });

    assert!(lam4.is_finite());
    assert!(lam6.abs() < 0.05);
}
