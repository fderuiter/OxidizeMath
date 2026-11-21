#![allow(missing_docs)]
use math_explorer::epidemiology::compartmental::SIRModel;
use math_explorer::epidemiology::stochastic::GillespieSolver;
use math_explorer::rng::OxidizeRng;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: verified_engine::allocator::VerifiedAllocator =
    verified_engine::allocator::VerifiedAllocator;

fn main() {
    let steps = 500_000;
    println!("Benchmarking Gillespie Solver with {} steps...", steps);

    // Setup
    let rng = OxidizeRng::default();
    let mut solver = GillespieSolver::new(rng);

    // Large population to prevent extinction early
    let n = 1_000_000.0;
    let i0 = 1000.0;
    let model = SIRModel::new(n, i0, 2.0, 0.1).unwrap();
    let mut state = *model.state();

    // Warmup
    for _ in 0..100 {
        solver.step(&model, &mut state).unwrap();
    }

    // Reset state for actual bench
    state = *model.state();
    let rng = OxidizeRng::default();
    solver = GillespieSolver::new(rng);

    let (_, metrics) = VerifiedEngine::run_verified(|| {
        for _ in 0..steps {
            let dt = solver.step(&model, &mut state).unwrap();
            if dt.is_infinite() {
                println!("Simulation ended early (extinction)!");
                break;
            }
        }
    });

    println!("Deterministic Metrics:");
    println!("{:#?}", metrics);
}
