//! Criterion benchmark suite for `pure_math` ODE integration and Vietoris-Rips TDA
//! accessed strictly through the `math_explorer` facade.

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use math_explorer::prelude::{
    ArrayState, OdeSystem, RungeKutta4, Solver, compute_rips_persistence,
};
use math_explorer::pure_math::statistics::tda::core::{Point2D, PointCloud};
use std::f64::consts::TAU;

struct HarmonicOscillator {
    omega_sq: f64,
}

impl OdeSystem<ArrayState<2>> for HarmonicOscillator {
    fn derivative(&self, _t: f64, state: &ArrayState<2>) -> ArrayState<2> {
        ArrayState([state.0[1], -self.omega_sq * state.0[0]])
    }

    fn derivative_in_place(&self, _t: f64, state: &ArrayState<2>, out: &mut ArrayState<2>) {
        out.0[0] = state.0[1];
        out.0[1] = -self.omega_sq * state.0[0];
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("PureMath");

    let system = HarmonicOscillator { omega_sq: 4.0 };
    let initial = ArrayState([1.0_f64, 0.0_f64]);
    let mut solver = RungeKutta4::new(&initial);

    group.bench_function("rk4_harmonic_oscillator_250_steps", |b| {
        b.iter(|| {
            let mut state = black_box(initial);
            let mut t = 0.0;
            for _ in 0..250 {
                solver.step(&system, t, &mut state, 0.01);
                t += 0.01;
            }
            black_box(state.0)
        })
    });

    let pts: Vec<Point2D> = (0..24)
        .map(|i| {
            let theta = TAU * (i as f64) / 24.0;
            Point2D::new(theta.cos(), theta.sin())
        })
        .collect();
    let cloud = PointCloud::new(pts).expect("valid circle cloud");

    group.bench_function("vietoris_rips_persistence_24_pts", |b| {
        b.iter(|| compute_rips_persistence(black_box(&cloud)))
    });

    group.finish();
}

mod bench_group {
    use super::{criterion_benchmark, criterion_group};
    criterion_group!(benches, criterion_benchmark);
}

criterion_main!(bench_group::benches);
