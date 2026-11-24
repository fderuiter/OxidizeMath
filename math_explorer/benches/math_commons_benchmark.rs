//! Criterion benchmark suite for `math_commons` curve fitting and sensitivity sweeps
//! accessed strictly through the `math_explorer` facade.

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use math_explorer::prelude::{
    FitModel, ParameterUncertainty, SensitivityEngine, TornadoBounds, fit_model,
    levenberg_marquardt_numeric,
};

fn bench_curve_fitting(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
) {
    let mut xs = [0.0_f64; 64];
    let mut ys = [0.0_f64; 64];
    for i in 0..64 {
        let x = 0.1 * (i as f64);
        xs[i] = x;
        ys[i] = 4.2 * (-0.35 * x).exp() + 0.05;
    }

    group.bench_function("fit_exponential_64", |b| {
        b.iter(|| {
            let series = (black_box(&xs[..]), black_box(&ys[..]));
            fit_model(&series, FitModel::Exponential, false)
        })
    });

    let seeds = [2.0_f64, 0.2_f64];
    group.bench_function("levenberg_marquardt_numeric_64", |b| {
        b.iter(|| {
            let series = (black_box(&xs[..]), black_box(&ys[..]));
            levenberg_marquardt_numeric(&series, black_box(&seeds), |x, p| {
                let val = p[0].clamp(0.1, 20.0) * (-p[1].clamp(0.01, 5.0) * x).exp() + 0.05;
                val.is_finite().then_some(val)
            })
        })
    });
}

fn bench_sensitivity_sweeps(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
) {
    let params = [
        ("dose", 100.0_f64),
        ("ka", 1.2_f64),
        ("ke", 0.12_f64),
        ("volume", 40.0_f64),
    ];
    let t_bounds = [
        TornadoBounds::new(50.0, 200.0),
        TornadoBounds::new(0.4, 3.0),
        TornadoBounds::new(0.03, 0.4),
        TornadoBounds::new(20.0, 80.0),
    ];
    let unc = [
        ParameterUncertainty::new(10.0, 50.0, 200.0),
        ParameterUncertainty::new(0.15, 0.4, 3.0),
        ParameterUncertainty::new(0.02, 0.03, 0.4),
        ParameterUncertainty::new(4.0, 20.0, 80.0),
    ];
    let mut engine = SensitivityEngine::new();
    engine.prepare_tornado(&params, &t_bounds);
    engine.prepare_monte_carlo(&params, &unc, 42, 64);

    group.bench_function("sensitivity_tornado_and_mc_64", |b| {
        b.iter(|| {
            let mut local = engine.clone();
            while !local.is_tornado_converged() {
                local.step_tornado_slice(black_box(&params), &t_bounds, 8, |v| {
                    (v[0] / v[3]) * (v[1] / (v[1] + v[2]))
                });
            }
            while !local.is_monte_carlo_converged() {
                local.step_monte_carlo_slice(black_box(&params), &unc, 42, 64, 8, |v| {
                    (v[0] / v[3]) * (v[1] / (v[1] + v[2]))
                });
            }
            black_box(local.monte_carlo_summary().copied())
        })
    });
}

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("MathCommons");
    bench_curve_fitting(&mut group);
    bench_sensitivity_sweeps(&mut group);
    group.finish();
}

mod bench_group {
    use super::{criterion_benchmark, criterion_group};
    criterion_group!(benches, criterion_benchmark);
}

criterion_main!(bench_group::benches);
