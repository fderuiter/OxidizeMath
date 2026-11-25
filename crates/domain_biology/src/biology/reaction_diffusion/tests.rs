use super::*;
use crate::biology::diffusion::FiniteDifference1D;
use crate::biology::morphogenesis::SchnakenbergKinetics;
use pure_math::pure_math::analysis::ode::solvers::RungeKutta4;

#[test]
#[verified_engine::verified]
fn test_reaction_diffusion_system_rk4() {
    // Setup a small system with RK4 solver
    let n = 10;
    let d_u = 1.0;
    let d_v = 0.5;
    let dx = 1.0;

    let kinetics = SchnakenbergKinetics::default();
    let diffusion = FiniteDifference1D::new(math_commons::math_kernel::types::StepSize(dx));
    let diffusion_coeffs = vec![d_u, d_v];

    // Explicitly use RK4
    let dummy_state = ChemicalState::new(2, n);
    let solver = RungeKutta4::new(&dummy_state);

    let mut system = ReactionDiffusionSystem::builder()
        .num_species(2)
        .grid_size(n)
        .reaction(kinetics)
        .diffusion(diffusion)
        .diffusion_coeffs(diffusion_coeffs)
        .solver(solver)
        .build_with_solver()
        .unwrap();

    // Initialize with same pattern
    for i in 0..n {
        system.state.species_mut(0)[i] = 1.0 + 0.1 * (i as f64);
        system.state.species_mut(1)[i] = 0.5 - 0.05 * (i as f64);
    }

    // Run for a few steps
    let dt = 0.01;
    for _ in 0..5 {
        system.step(dt);
    }

    // Check values are reasonable (not NaN and changed)
    let u_val = system.state.species(0)[0];
    assert!(!u_val.is_nan());
    assert!((u_val - 1.0).abs() > 1e-3);
}

#[test]
#[verified_engine::verified]
fn test_reaction_diffusion_system_equivalence() {
    // Setup a small system
    let n = 10;
    let d_u = 1.0;
    let d_v = 0.5;
    let dx = 1.0;

    let kinetics = SchnakenbergKinetics::default();
    let diffusion = FiniteDifference1D::new(math_commons::math_kernel::types::StepSize(dx));
    let diffusion_coeffs = vec![d_u, d_v];

    let mut system = ReactionDiffusionSystem::builder()
        .num_species(2)
        .grid_size(n)
        .reaction(kinetics)
        .diffusion(diffusion)
        .diffusion_coeffs(diffusion_coeffs)
        .build()
        .unwrap();

    // Initialize with same pattern as in morphogenesis test
    // u = 1.0 + 0.1 * i
    // v = 0.5 - 0.05 * i
    for i in 0..n {
        system.state.species_mut(0)[i] = 1.0 + 0.1 * (i as f64);
        system.state.species_mut(1)[i] = 0.5 - 0.05 * (i as f64);
    }

    // Run for a few steps
    let dt = 0.01;
    for _ in 0..5 {
        system.step(dt);
    }

    // Capture output
    let u_out = system.state.species(0).to_vec();
    let v_out = system.state.species(1).to_vec();

    // Expected values (same as in morphogenesis.rs)
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

    // Assert with tolerance
    let tolerance = math_commons::registry::TOLERANCE_HIGH;
    for i in 0..n {
        assert!(
            (u_out[i] - expected_u[i]).abs() < tolerance,
            "U mismatch at {}: {} vs {}",
            i,
            u_out[i],
            expected_u[i]
        );
        assert!(
            (v_out[i] - expected_v[i]).abs() < tolerance,
            "V mismatch at {}: {} vs {}",
            i,
            v_out[i],
            expected_v[i]
        );
    }
}

#[test]
fn cell_mask_2d_set_roles_slice_returns_typed_error_without_allocating() {
    use crate::biology::diffusion::CellMask2D;
    use verified_engine::engine::VerifiedEngine;

    let mut mask = CellMask2D::new(4, 3);
    let valid_bytes = [0_u8; 12];
    let short_bytes = [0_u8; 5];

    let (ok, _) = VerifiedEngine::run_verified(|| {
        let err = mask
            .set_roles_slice(&short_bytes)
            .expect_err("mismatched slice length must fail");
        assert_eq!(
            err,
            ReactionDiffusionError::CellMaskLengthMismatch {
                expected: 12,
                actual: 5,
            }
        );
        mask.set_roles_slice(&valid_bytes).is_ok()
    });
    assert!(ok);
}
