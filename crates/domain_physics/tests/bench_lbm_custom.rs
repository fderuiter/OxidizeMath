#![allow(missing_docs)]
use domain_physics::physics::fluid_dynamics::lattice_boltzmann::LatticeBoltzmannD2Q9;
use std::time::Instant;
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

#[test]
#[verified_engine::verified]
fn bench_lbm_step() {
    let width = 100;
    let height = 50;
    let tau = 0.6;
    let mut solver = LatticeBoltzmannD2Q9::new(width, height, tau);
    for y in 20..=30 {
        for x in 28..=32 {
            solver.set_obstacle(x, y, true);
        }
    }

    // Warmup outside allocation lock
    for _ in 0..10 {
        solver.set_inlet(0, 0, 2, height, 0.1, 0.0);
        solver.step();
    }

    let start = Instant::now();
    let iterations = 100;
    let ((fx, fy, curl, solid), _metrics) = VerifiedEngine::run_verified(|| {
        for _ in 0..iterations {
            solver.set_inlet(0, 0, 2, height, 0.1, 0.0);
            solver.step();
        }
        let (fx, fy) = solver.net_obstacle_force();
        let curl = solver.vorticity_curl(36, 22);
        let solid = solver.obstacle_cell_count();
        (fx, fy, curl, solid)
    });
    let duration = start.elapsed();

    assert_eq!(solid, 55);
    assert!(
        fx > 0.0,
        "Expected positive bounce-back drag force, got {fx}"
    );
    assert!(fy.is_finite() && curl.is_finite());

    println!("LBM Step Benchmark (100x50, 100 iters): {:?}", duration);
    println!("Average time per step: {:?}", duration / iterations as u32);
}
