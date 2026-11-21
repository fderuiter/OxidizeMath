#![allow(missing_docs)]
use math_explorer::num_complex::Complex;
use math_explorer::physics::medical::radar_gating::super_resolution::MusicEstimator;
use std::time::Instant;

fn main() {
    let samples = 64;
    let snapshots = 20;
    let targets = 2;
    let mut estimator = MusicEstimator::new(samples, snapshots, targets).unwrap();

    // Feed dummy data
    for _ in 0..snapshots {
        let mut data = Vec::with_capacity(samples);
        for i in 0..samples {
            let t = i as f64;
            let val = Complex::new(0.0, 0.1 * t).exp() + Complex::new(0.0, 0.3 * t).exp();
            data.push(val);
        }
        estimator.add_snapshot(&data).unwrap();
    }

    println!("Benchmarking MusicEstimator::compute_spectrum...");

    // Warmup
    for _ in 0..10 {
        let _ = estimator
            .compute_spectrum(0.0, 10.0, 0.05, 4.0e9, 3.0e8)
            .unwrap();
    }

    let start = Instant::now();
    let iterations = 100;
    for _ in 0..iterations {
        // This specific call was causing the off-by-one panic
        let spectrum = estimator
            .compute_spectrum(0.0, 10.0, 0.001, 4.0e9, 3.0e8)
            .unwrap();

        // Strictly verify the length dynamically
        let expected_len = ((10.0_f64 - 0.0_f64) / 0.001_f64).round() as usize + 1;
        if spectrum.len() != expected_len {
            panic!(
                "Unexpected spectrum length: expected {}, got {}",
                expected_len,
                spectrum.len()
            );
        }
    }
    let duration = start.elapsed();

    println!("Total time for {} iterations: {:?}", iterations, duration);
    println!(
        "Average time per iteration: {:?}",
        duration / iterations as u32
    );
}
