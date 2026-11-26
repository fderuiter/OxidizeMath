use super::*;

#[test]
#[should_panic(expected = "State vector 1 length mismatch")]
#[verified_engine::verified]
fn test_derivative_in_place_safety_check() {
    let n = 10;
    let system = TuringSystem::new(
        Dimension(n),
        DiffusionCoeff(1.0),
        DiffusionCoeff(1.0),
        StepSize(1.0),
    );
    let mut state = TuringState::new(n);
    state.concentrations[1].pop();
    let mut out = TuringState::new(n);
    system.derivative_in_place(0.0, &state, &mut out);
}

#[test]
#[verified_engine::verified]
fn test_turing_system_logic_preservation() {
    let n = 10;
    let mut system = TuringSystem::new(
        Dimension(n),
        DiffusionCoeff(1.0),
        DiffusionCoeff(0.5),
        StepSize(1.0),
    );
    for i in 0..n {
        system.state.u_mut()[i] = 1.0 + 0.1 * (i as f64);
        system.state.v_mut()[i] = 0.5 - 0.05 * (i as f64);
    }
    let dt = 0.01;
    for _ in 0..5 {
        system.step(dt);
    }
    let expected_u = [
        0.9798926377401955,
        1.0722504645444493,
        1.1685990805783317,
        1.2642647090938448,
        1.359028327357602,
        1.4527705800845148,
        1.5453811790730032,
        1.6367576303434268,
        1.7267186541725483,
        1.8109737170223916,
    ];
    let expected_v = [
        0.47709091921002866,
        0.4263770084483741,
        0.3750152156844884,
        0.32443296992262166,
        0.2747722006954079,
        0.22615405798594523,
        0.1786914141249832,
        0.1324883911255509,
        0.08765106523936222,
        0.04531981611374585,
    ];
    let tolerance = math_commons::registry::TOLERANCE_HIGH;
    for i in 0..n {
        assert!((system.u()[i] - expected_u[i]).abs() < tolerance);
        assert!((system.v()[i] - expected_v[i]).abs() < tolerance);
    }
    assert!((system.time() - 0.05).abs() < 1e-15);
}

#[test]
#[verified_engine::verified]
fn test_step_with_kinetics_reuses_buffers_and_advances_time_zero_alloc() {
    struct TimeForcedKinetics;
    impl ReactionKinetics<2> for TimeForcedKinetics {
        fn reaction(&self, _c: [f64; 2]) -> [f64; 2] {
            [0.0, 0.0]
        }
        fn reaction_at_time(&self, t: f64, _index: usize, _c: [f64; 2]) -> [f64; 2] {
            [2.0 * t, t]
        }
        fn is_time_dependent(&self) -> bool {
            true
        }
    }

    let (dim, du, dv, dx) = (
        Dimension(16),
        DiffusionCoeff(0.16),
        DiffusionCoeff(0.08),
        StepSize(1.0),
    );
    let gs = GrayScottKinetics::mitosis();
    let mut alt_sys = TuringSystem::new(dim, du, dv, dx);
    let mut dedicated =
        TuringSystem::new_with_kinetics(dim, du, dv, gs, FiniteDifference1D::new(dx));
    alt_sys.u_mut().fill(1.0);
    dedicated.u_mut().fill(1.0);
    for i in 6..=9 {
        alt_sys.u_mut()[i] = 0.5;
        alt_sys.v_mut()[i] = 0.25;
        dedicated.u_mut()[i] = 0.5;
        dedicated.v_mut()[i] = 0.25;
    }
    verified_engine::engine::VerifiedEngine::run_verified(|| {
        alt_sys.step_with_kinetics(&gs, 0.5);
        dedicated.step(0.5);
    });
    assert_eq!(alt_sys.u(), dedicated.u());
    assert_eq!(alt_sys.v(), dedicated.v());
    assert!((alt_sys.time() - 0.5).abs() < 1e-15);

    let forced = TimeForcedKinetics;
    alt_sys.set_time(1.5);
    let u0 = alt_sys.u()[0];
    alt_sys.step_with_kinetics(&forced, 0.25);
    assert!((alt_sys.time() - 1.75).abs() < 1e-15);
    assert!((alt_sys.u()[0] - (u0 + 0.25 * 3.0)).abs() < 0.2);
}
