//! Integration tests for test turing regression.
use domain_biology::biology::diffusion::FiniteDifference1D;
use domain_biology::biology::morphogenesis::{GrayScottKinetics, TuringSystem};

#[test]
#[verified_engine::verified]
fn test_turing_regression() {
    let size = 100;
    let iterations = 100;
    // Default uses Schnakenberg kinetics (backward compatible)
    let mut system = TuringSystem::new(
        math_commons::math_kernel::types::Dimension(size),
        domain_biology::biology::morphogenesis::DiffusionCoeff(0.1),
        domain_biology::biology::morphogenesis::DiffusionCoeff(0.05),
        math_commons::math_kernel::types::StepSize(1.0),
    );

    // Seed it deterministically
    for i in 0..size {
        if i % 10 == 0 {
            system.u_mut()[i] = 1.0;
        }
    }

    for _ in 0..iterations {
        system.step(0.01);
    }

    // Verify exactness
    assert!((system.u()[50] - 0.3116871030).abs() < math_commons::registry::TOLERANCE_STANDARD);
    assert!((system.v()[50] - 0.0447023450).abs() < math_commons::registry::TOLERANCE_STANDARD);
}

#[test]
#[verified_engine::verified]
fn test_custom_kinetics_strategy() {
    let size = 50;
    let iterations = 10;

    // Use Gray-Scott kinetics via the Strategy Pattern
    let kinetics = GrayScottKinetics { f: 0.055, k: 0.062 };
    let diffusion = FiniteDifference1D::new(math_commons::math_kernel::types::StepSize(1.0));
    let mut system = TuringSystem::new_with_kinetics(
        math_commons::math_kernel::types::Dimension(size),
        domain_biology::biology::morphogenesis::DiffusionCoeff(0.2),
        domain_biology::biology::morphogenesis::DiffusionCoeff(0.1),
        kinetics,
        diffusion,
    );

    // Seed
    system.u_mut()[25] = 0.5;
    system.v_mut()[25] = 0.25;

    // Run simulation
    for _ in 0..iterations {
        system.step(0.1);
    }

    // Check that values changed (diffusion + reaction happened)
    // We don't check exact values as this is a new model, but we ensure it didn't panic and produced finite numbers
    assert!(system.u()[25] != 0.5);
    assert!(system.u()[25].is_finite());
    assert!(system.v()[25].is_finite());
}
