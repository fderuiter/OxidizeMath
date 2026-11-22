#![allow(missing_docs)]
use domain_biology::biology::diffusion::{FiniteDifference1D, FiniteDifference2D};
use domain_biology::biology::morphogenesis::{DiffusionCoeff, GrayScottKinetics, TuringSystem};
use math_commons::math_kernel::types::{Dimension, StepSize, flatten_2d_index};
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

static ALLOC_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn run_gray_scott_2d_periodic(f: f64, k: f64, steps: usize) -> (Vec<f64>, Vec<f64>) {
    let (width, height) = (24_usize, 24_usize);
    let n = width * height;
    let diffusion = FiniteDifference2D::new_periodic(
        Dimension(width),
        Dimension(height),
        StepSize(1.0),
        StepSize(1.0),
    );
    let kinetics = GrayScottKinetics::new(f, k);
    let mut system = TuringSystem::new_with_kinetics(
        Dimension(n),
        DiffusionCoeff(0.16),
        DiffusionCoeff(0.08),
        kinetics,
        diffusion,
    );

    system.u_mut().fill(1.0);
    system.v_mut().fill(0.0);
    for y in 9..15 {
        for x in 9..15 {
            let idx = flatten_2d_index(x, y, width);
            system.u_mut()[idx] = 0.50;
            system.v_mut()[idx] = 0.25 + 0.01 * ((x + y) % 3) as f64;
        }
    }

    for _ in 0..steps {
        system.step(1.0);
    }
    (system.u().to_vec(), system.v().to_vec())
}

#[test]
#[verified_engine::verified]
fn test_turing_regression() {
    let _guard = ALLOC_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let size = 100;
    let iterations = 100;
    let mut system = TuringSystem::new(
        Dimension(size),
        DiffusionCoeff(0.1),
        DiffusionCoeff(0.05),
        StepSize(1.0),
    );

    for i in 0..size {
        if i % 10 == 0 {
            system.u_mut()[i] = 1.0;
        }
    }

    for _ in 0..iterations {
        system.step(0.01);
    }

    assert!((system.u()[50] - 0.3116871030).abs() < math_commons::registry::TOLERANCE_STANDARD);
    assert!((system.v()[50] - 0.0447023450).abs() < math_commons::registry::TOLERANCE_STANDARD);
}

#[test]
#[verified_engine::verified]
fn test_custom_kinetics_strategy() {
    let _guard = ALLOC_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let size = 50;
    let iterations = 10;

    let kinetics = GrayScottKinetics::new(0.055, 0.062);
    let diffusion = FiniteDifference1D::new(StepSize(1.0));
    let mut system = TuringSystem::new_with_kinetics(
        Dimension(size),
        DiffusionCoeff(0.2),
        DiffusionCoeff(0.1),
        kinetics,
        diffusion,
    );

    system.u_mut()[25] = 0.5;
    system.v_mut()[25] = 0.25;

    for _ in 0..iterations {
        system.step(0.1);
    }

    assert!(system.u()[25] != 0.5);
    assert!(system.u()[25].is_finite());
    assert!(system.v()[25].is_finite());
}

#[test]
#[verified_engine::verified]
fn test_gray_scott_spots_and_stripes_regimes_and_periodic_zero_alloc() {
    let _guard = ALLOC_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    // Mitosis / spots regime (Pearson 1993: f = 0.035, k = 0.065)
    let (u_spots, v_spots) = run_gray_scott_2d_periodic(0.035, 0.065, 250);
    // Coral / stripes regime (Pearson 1993: f = 0.055, k = 0.062)
    let (u_stripes, v_stripes) = run_gray_scott_2d_periodic(0.055, 0.062, 250);

    for (&u, &v) in u_spots.iter().zip(v_spots.iter()) {
        assert!(u.is_finite() && (0.0..=1.05).contains(&u));
        assert!(v.is_finite() && (0.0..=1.05).contains(&v));
    }
    for (&u, &v) in u_stripes.iter().zip(v_stripes.iter()) {
        assert!(u.is_finite() && (0.0..=1.05).contains(&u));
        assert!(v.is_finite() && (0.0..=1.05).contains(&v));
    }

    let v_spots_mean = v_spots.iter().sum::<f64>() / v_spots.len() as f64;
    let v_stripes_mean = v_stripes.iter().sum::<f64>() / v_stripes.len() as f64;
    assert!(v_spots.iter().copied().fold(0.0_f64, f64::max) > 0.05);
    assert!(v_stripes.iter().copied().fold(0.0_f64, f64::max) > 0.05);
    // Higher feed rate f=0.055 sustains higher mean activator mass than localized spots f=0.035
    assert!(v_stripes_mean > v_spots_mean);

    // Verify 0 heap allocations and exact mass conservation when reaction rates are zero
    let (width, height) = (16_usize, 16_usize);
    let n = width * height;
    let diffusion = FiniteDifference2D::new_periodic(
        Dimension(width),
        Dimension(height),
        StepSize(1.0),
        StepSize(1.0),
    );
    let mut system = TuringSystem::new_with_kinetics(
        Dimension(n),
        DiffusionCoeff(0.16),
        DiffusionCoeff(0.08),
        GrayScottKinetics::new(0.0, 0.0),
        diffusion,
    );
    system.u_mut()[0] = 2.0;
    system.v_mut()[flatten_2d_index(width - 1, height - 1, width)] = 1.0;
    let initial_total: f64 = system.u().iter().sum::<f64>() + system.v().iter().sum::<f64>();

    let ((), metrics) = VerifiedEngine::run_verified(|| {
        for _ in 0..16 {
            system.step(0.25);
        }
    });
    assert!(metrics.function_calls > 0);
    // f=0, k=0 converts u -> v via uv^2 while conserving u + v across the closed torus!
    let final_total: f64 = system.u().iter().sum::<f64>() + system.v().iter().sum::<f64>();
    assert!((final_total - initial_total).abs() < 1e-11);
}
