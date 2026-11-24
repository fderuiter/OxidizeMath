//! Criterion benchmark suite for `domain_applied` pharmacokinetics, evolutionary game theory,
//! and battery degradation solvers accessed strictly through the `math_explorer` facade.

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use math_explorer::nalgebra::{DMatrix, DVector};
use math_explorer::prelude::{
    Cycles, DepthOfDischarge, PKParameters, PowerLawModel, ReplicatorDynamics,
    concentration_bateman,
};

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("DomainApplied");

    let pk = PKParameters::new(0.85, 100.0, 1.2, 0.12, 40.0).expect("valid pk parameters");
    group.bench_function("bateman_pk_trajectory_200", |b| {
        b.iter(|| {
            let mut acc = 0.0_f64;
            for step in 0..200 {
                let t = (step as f64) * 0.25;
                acc += concentration_bateman(black_box(&pk), t);
            }
            black_box(acc)
        })
    });

    let payoff = DMatrix::from_row_slice(3, 3, &[0.0, -1.0, 1.0, 1.0, 0.0, -1.0, -1.0, 1.0, 0.0]);
    let replicator = ReplicatorDynamics::new(payoff).expect("valid replicator dynamics");
    let init_pop = DVector::from_vec(vec![0.5, 0.3, 0.2]);
    group.bench_function("replicator_dynamics_rps_200_steps", |b| {
        b.iter(|| replicator.simulate(black_box(init_pop.clone()), 10.0, 0.05))
    });

    let battery = PowerLawModel::standard();
    let dod = DepthOfDischarge::new(80.0).expect("valid DoD");
    group.bench_function("battery_power_law_curve_100", |b| {
        b.iter(|| {
            let mut sum = 0.0_f64;
            for i in 0..=100 {
                let cyc = Cycles::new((i as f64) * 25.0).expect("valid cycle");
                sum += battery.capacity(cyc, black_box(dod)).as_f64();
            }
            black_box(sum)
        })
    });

    group.finish();
}

mod bench_group {
    use super::{criterion_benchmark, criterion_group};
    criterion_group!(benches, criterion_benchmark);
}

criterion_main!(bench_group::benches);
