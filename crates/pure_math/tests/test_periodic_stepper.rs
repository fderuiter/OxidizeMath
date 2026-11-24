//! Integration test enforcing zero heap allocations in `step_2d_coupled_periodic`.

use math_commons::math_kernel::types::StepSize;
use pure_math::pure_math::analysis::pde::fused_stepper::FusedStencilStepper;
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

#[test]
#[verified_engine::verified]
fn test_step_2d_coupled_periodic_zero_allocations() {
    let (width, height) = (16_usize, 16_usize);
    let n = width * height;
    let stepper = FusedStencilStepper::new_2d(StepSize(1.0), StepSize(1.0));
    let mut u = vec![0.25_f64; n];
    let mut v = vec![0.5_f64; n];
    u[0] = 1.0;
    let mut next_u = vec![0.0_f64; n];
    let mut next_v = vec![0.0_f64; n];

    let ((), metrics) = VerifiedEngine::run_verified(|| {
        for _ in 0..8 {
            stepper.step_2d_coupled_periodic(
                (width, height),
                [u.as_slice(), v.as_slice()],
                [next_u.as_mut_slice(), next_v.as_mut_slice()],
                0.05,
                1.0,
                |_idx, curr, left, right, up, down, ops| {
                    [
                        0.2 * ops.central_diff_2nd_2d(curr[0], left[0], right[0], up[0], down[0]),
                        0.1 * ops.central_diff_2nd_2d(curr[1], left[1], right[1], up[1], down[1]),
                    ]
                },
            );
            std::mem::swap(&mut u, &mut next_u);
            std::mem::swap(&mut v, &mut next_v);
        }
    });

    assert!(metrics.function_calls > 0);
    assert!((u.iter().sum::<f64>() - (0.25 * (n as f64) + 0.75)).abs() < 1e-11);
}
