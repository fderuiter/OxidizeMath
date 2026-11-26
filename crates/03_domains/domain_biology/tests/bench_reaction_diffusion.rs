//! Integration tests for bench reaction diffusion.
#[cfg(not(target_arch = "wasm32"))]
use domain_biology::biology::diffusion::FiniteDifference2D;
#[cfg(not(target_arch = "wasm32"))]
use domain_biology::biology::morphogenesis::{SchnakenbergKinetics, TuringSystem};
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(not(target_arch = "wasm32"))]
use verified_engine::allocator::VerifiedAllocator;
#[cfg(not(target_arch = "wasm32"))]
use verified_engine::engine::VerifiedEngine;
#[cfg(not(target_arch = "wasm32"))]
use verified_engine::metrics::ComplexityMetrics;

#[cfg(not(target_arch = "wasm32"))]
#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

#[cfg(not(target_arch = "wasm32"))]
fn assert_turing_invariants(
    u: &[f64],
    v: &[f64],
    expected_len: usize,
    metrics: &ComplexityMetrics,
) {
    assert_eq!(u.len(), expected_len);
    assert_eq!(v.len(), expected_len);
    assert!(u.iter().all(|&val| val.is_finite() && val > 0.0));
    assert!(v.iter().all(|&val| val.is_finite() && val >= 0.0));
    assert!(metrics.function_calls > 0);
    assert!(metrics.arithmetic_ops > 0);
    assert!(metrics.memory_loads > 0);
}

#[cfg(not(target_arch = "wasm32"))]
#[allow(clippy::disallowed_methods)]
#[test]
#[verified_engine::verified]
fn bench_turing_2d_step() {
    let width = 300;
    let height = 300;
    let diff = FiniteDifference2D::new(
        math_commons::math_kernel::types::Dimension(width),
        math_commons::math_kernel::types::Dimension(height),
        math_commons::math_kernel::types::StepSize(1.0),
        math_commons::math_kernel::types::StepSize(1.0),
    );
    let kinetics = SchnakenbergKinetics::default();

    // Create system with 90k elements
    let mut system = TuringSystem::new_with_kinetics(
        math_commons::math_kernel::types::Dimension(width * height),
        domain_biology::biology::morphogenesis::DiffusionCoeff(1.0),
        domain_biology::biology::morphogenesis::DiffusionCoeff(40.0),
        kinetics,
        diff,
    );

    // Initialize with some values
    use oxidize_core::rng::OxidizeRng;
    use rand::Rng;
    let mut rng = OxidizeRng::default();
    for i in 0..width * height {
        system.u_mut()[i] = 1.0 + rng.gen_range(-0.1..0.1);
        system.v_mut()[i] = 0.5 + rng.gen_range(-0.1..0.1);
    }

    let iterations = 100;
    // 2D explicit Euler CFL stability requires 2 * D_max * dt / dx^2 <= 0.5 (40.0 * 2 * 0.005 = 0.4).
    let dt = 0.005;

    // Warmup
    for _ in 0..10 {
        system.step(dt);
    }

    let start = Instant::now();
    let ((), metrics) = VerifiedEngine::run_verified(|| {
        for _ in 0..iterations {
            system.step(dt);
        }
    });
    let duration = start.elapsed();

    assert_turing_invariants(system.u(), system.v(), width * height, &metrics);
    println!(
        "Turing 2D ({}x{}) - {} iterations: {:?}",
        width, height, iterations, duration
    );
    println!(
        "Average time per iteration: {:?}",
        duration / iterations as u32
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn bench_turing_aborts_if_hot_loop_allocates() {
    const PROBE_ENV: &str = "OXIDIZE_TURING_ALLOC_ABORT_PROBE";
    if std::env::var_os(PROBE_ENV).is_some() {
        let mut system = TuringSystem::new(
            math_commons::math_kernel::types::Dimension(32),
            domain_biology::biology::morphogenesis::DiffusionCoeff(0.1),
            domain_biology::biology::morphogenesis::DiffusionCoeff(0.05),
            math_commons::math_kernel::types::StepSize(1.0),
        );
        VerifiedEngine::run_verified(|| {
            system.step(0.01);
            let snapshot = system.u().to_vec();
            std::hint::black_box(snapshot.len());
        });
        return;
    }
    let output = std::process::Command::new(std::env::current_exe().expect("test binary"))
        .env(PROBE_ENV, "1")
        .args(["--exact", "bench_turing_aborts_if_hot_loop_allocates"])
        .output()
        .expect("spawn Turing allocation probe");
    assert!(
        !output.status.success(),
        "VerifiedAllocator must abort when reaction-diffusion hot loop allocates on the heap"
    );
}
