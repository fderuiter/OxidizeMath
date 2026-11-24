//! Integration test verifying continuous fractal potentials and zero heap allocations.

use domain_physics::physics::chaos::fractals::{smooth_escape_julia, smooth_escape_mandelbrot};
use num_complex::Complex;
use std::collections::HashSet;
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

#[test]
#[verified_engine::verified]
fn test_smooth_fractal_potentials_continuous_and_zero_alloc() {
    let max_iter = 256_u32;
    assert_eq!(
        smooth_escape_mandelbrot(Complex::new(0.0, 0.0), max_iter),
        256.0
    );
    assert_eq!(
        smooth_escape_julia(Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), max_iter),
        256.0
    );

    let mut quantized_16bit = HashSet::new();
    let ((), metrics) = VerifiedEngine::run_verified(|| {
        let mut accum = 0.0_f64;
        for idx in 0..512 {
            let t = idx as f64 / 511.0;
            let c = Complex::new(-2.1 + t * 1.3, 0.25 + t * 0.6);
            let m_val = smooth_escape_mandelbrot(c, max_iter);
            let j_val = smooth_escape_julia(c * 0.5, Complex::new(-0.7, 0.27015), max_iter);
            assert!(m_val.is_finite() && (0.0..=max_iter as f64).contains(&m_val));
            assert!(j_val.is_finite() && (0.0..=max_iter as f64).contains(&j_val));
            accum += m_val + j_val;
        }
        assert!(accum > 0.0);
    });
    assert!(metrics.function_calls > 0);

    let low_iter = 100_u32;
    let samples: Vec<f64> = (0..1024)
        .map(|idx| {
            let t = idx as f64 / 1023.0;
            let c = Complex::new(-0.745, 0.12 + t * 1.25);
            smooth_escape_mandelbrot(c, low_iter)
        })
        .collect();
    let min_v = samples.iter().copied().fold(f64::INFINITY, f64::min);
    let max_v = samples.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    for val in samples {
        let norm = ((val - min_v) / (max_v - min_v)).clamp(0.0, 1.0);
        quantized_16bit.insert((norm * 65535.0).round() as u16);
    }
    assert!(
        quantized_16bit.len() > 256,
        "Smooth Mandelbrot potential (max_iter={low_iter}) must yield > 256 distinct 16-bit levels, got {}",
        quantized_16bit.len()
    );
}
