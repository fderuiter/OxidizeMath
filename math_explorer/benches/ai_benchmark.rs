//! Criterion benchmark suite for `domain_ai` Transformer attention and encoder layers
//! accessed strictly through the `math_explorer` facade.

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use math_explorer::nalgebra::DMatrix;
use math_explorer::prelude::{Encoder, scaled_dot_product_attention};

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("DomainAi");

    let seq_len = 16;
    let d_k = 32;
    let q = DMatrix::from_fn(seq_len, d_k, |r, col| {
        ((r * 7 + col * 3) as f64 * 0.01).sin()
    });
    let k = DMatrix::from_fn(seq_len, d_k, |r, col| {
        ((r * 5 + col * 11) as f64 * 0.01).cos()
    });
    let v = DMatrix::from_fn(seq_len, d_k, |r, col| (r + col) as f64 * 0.02);

    group.bench_function("scaled_dot_product_attention_16x32", |b| {
        b.iter(|| scaled_dot_product_attention(black_box(&q), black_box(&k), black_box(&v), None))
    });

    let encoder = Encoder::new(2, 32, 4, 64);
    let tokens = DMatrix::from_fn(12, 32, |r, col| ((r + col) as f64) * 0.01);
    group.bench_function("transformer_encoder_2L_12x32", |b| {
        b.iter(|| encoder.forward(black_box(tokens.clone()), None))
    });

    group.finish();
}

mod bench_group {
    use super::{criterion_benchmark, criterion_group};
    criterion_group!(benches, criterion_benchmark);
}

criterion_main!(bench_group::benches);
