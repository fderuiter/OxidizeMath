#![allow(missing_docs)]
use criterion::{Criterion, black_box, criterion_group, criterion_main};
use math_explorer::applied::algorithms::sorting::divide_conquer::{merge_sort, quick_sort};
use math_explorer::applied::algorithms::sorting::linear::radix_sort;
use math_explorer::rand::Rng;
use math_explorer::rng::OxidizeRng;

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Sorting");

    // Generate a random array of 1000 integers
    let mut data = Vec::with_capacity(1000);
    for i in 0..1000 {
        data.push(1000 - i); // worst case or randomish
    }

    group.bench_function("merge_sort", |b| b.iter(|| merge_sort(black_box(&data))));
    group.bench_function("quick_sort", |b| b.iter(|| quick_sort(black_box(&data))));

    let mut rng = OxidizeRng::default();
    let mut radix_data = Vec::with_capacity(100_000);
    for _ in 0..100_000 {
        radix_data.push(rng.r#gen::<u32>() as u64); // avoid overflow
    }
    group.bench_function("radix_sort", |b| {
        b.iter(|| radix_sort(black_box(&radix_data)))
    });

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
