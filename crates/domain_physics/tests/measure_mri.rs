#![allow(missing_docs)]
use domain_physics::physics::mri::reconstruction::{inverse_dft_2d, simulate_signal_2d};
use nalgebra::DMatrix;
use num_complex::Complex;
use std::f64::consts::PI;
#[cfg(not(target_arch = "wasm32"))]
use std::time::{Duration, Instant};

#[test]
#[verified_engine::verified]
fn measure_mri_dft_performance() {
    let size = 64; // 64x64 matrix
    let density = DMatrix::from_fn(size, size, |r, c| {
        let x = r as f64 - size as f64 / 2.0;
        let y = c as f64 - size as f64 / 2.0;
        let val = if (x * x + y * y).sqrt() < size as f64 / 4.0 {
            1.0
        } else {
            0.0
        };
        Complex::new(val, 0.0)
    });

    println!("Running simulate_signal_2d for {}x{} matrix...", size, size);
    #[cfg(not(target_arch = "wasm32"))]
    #[allow(clippy::disallowed_methods)]
    let start_sim = Instant::now();
    let signal = simulate_signal_2d(&density);
    #[cfg(not(target_arch = "wasm32"))]
    println!("simulate_signal_2d took: {:?}", start_sim.elapsed());

    println!("Running inverse_dft_2d for {}x{} matrix...", size, size);
    #[cfg(not(target_arch = "wasm32"))]
    #[allow(clippy::disallowed_methods)]
    let start_inv = Instant::now();
    let reconstructed = inverse_dft_2d(&signal);
    #[cfg(not(target_arch = "wasm32"))]
    println!("inverse_dft_2d took: {:?}", start_inv.elapsed());

    // Verify basic correctness (round trip)
    // The naive implementation in mri.rs doesn't normalize by 1/(N*M).
    // Forward DFT sums unnormalized. Inverse DFT sums unnormalized.
    // Total scale factor is (N*M).

    let scale = (size * size) as f64;

    let center = (size / 2, size / 2);
    let orig_val = density[center].re;
    let recon_val = reconstructed[center].re / scale;

    println!(
        "Center pixel original: {}, reconstructed (scaled): {}",
        orig_val, recon_val
    );

    assert!(
        (orig_val - recon_val).abs() < 1e-5,
        "Reconstruction failed correctness check. Expected {}, got {}",
        orig_val,
        recon_val
    );
}

#[test]
#[verified_engine::verified]
fn test_mri_rectangular_matrix_reconstruction() {
    let rows = 32;
    let cols = 16;
    let density = DMatrix::from_fn(rows, cols, |r, c| {
        let x = r as f64 - rows as f64 / 2.0;
        let y = c as f64 - cols as f64 / 2.0;
        let val = if (x * x + y * y).sqrt() < 6.0 {
            1.0
        } else {
            0.0
        };
        Complex::new(val, 0.0)
    });

    let signal = simulate_signal_2d(&density);
    let reconstructed = inverse_dft_2d(&signal);

    let scale = (rows * cols) as f64;
    for r in 0..rows {
        for c in 0..cols {
            let orig = density[(r, c)].re;
            let recon = reconstructed[(r, c)].re / scale;
            assert!(
                (orig - recon).abs() < 1e-5,
                "Mismatch at ({}, {}): orig {}, recon {}",
                r,
                c,
                orig,
                recon
            );
            assert!(
                reconstructed[(r, c)].im.abs() / scale < 1e-5,
                "Imaginary component non-zero at ({}, {}): {}",
                r,
                c,
                reconstructed[(r, c)].im / scale
            );
        }
    }
}

#[test]
#[verified_engine::verified]
fn test_mri_empty_matrix() {
    for (rows, cols) in [(0, 0), (0, 9), (9, 0)] {
        let empty = DMatrix::zeros(rows, cols);
        let sig = simulate_signal_2d(&empty);
        assert_eq!(sig.shape(), (rows, cols));
        let recon = inverse_dft_2d(&sig);
        assert_eq!(recon.shape(), (rows, cols));
    }
}

#[verified_engine::verified]
fn direct_dft_2d(matrix: &DMatrix<Complex<f64>>, inverse: bool) -> DMatrix<Complex<f64>> {
    let rows = matrix.nrows();
    let cols = matrix.ncols();
    if rows == 0 || cols == 0 {
        return DMatrix::zeros(rows, cols);
    }
    let sign = if inverse { 2.0 * PI } else { -2.0 * PI };
    DMatrix::from_fn(rows, cols, |kr, kc| {
        let mut sum = Complex::new(0.0, 0.0);
        for r in 0..rows {
            for c in 0..cols {
                let phase = sign * ((kr * r) as f64 / rows as f64 + (kc * c) as f64 / cols as f64);
                sum += matrix[(r, c)] * Complex::from_polar(1.0, phase);
            }
        }
        sum
    })
}

#[verified_engine::verified]
fn assert_matches_direct_dft(rows: usize, cols: usize) {
    let input = DMatrix::from_fn(rows, cols, |r, c| {
        let re = ((r * 7 + c * 3 + 1) as f64 * 0.37).sin();
        let im = ((r * 5 + c * 11 + 2) as f64 * 0.23).cos();
        Complex::new(re, im)
    });

    let fwd_fft = simulate_signal_2d(&input);
    let fwd_ref = direct_dft_2d(&input, false);
    assert_eq!(fwd_fft.shape(), (rows, cols));

    let inv_fft = inverse_dft_2d(&fwd_fft);
    let inv_ref = direct_dft_2d(&fwd_fft, true);
    assert_eq!(inv_fft.shape(), (rows, cols));

    let scale = (rows * cols) as f64;
    for r in 0..rows {
        for c in 0..cols {
            assert!(
                (fwd_fft[(r, c)] - fwd_ref[(r, c)]).norm() < 1e-9,
                "Forward mismatch at ({r}, {c}) for {rows}x{cols}"
            );
            assert!(
                (inv_fft[(r, c)] - inv_ref[(r, c)]).norm() < 1e-8,
                "Inverse mismatch at ({r}, {c}) for {rows}x{cols}"
            );
            assert!(
                (inv_fft[(r, c)] / scale - input[(r, c)]).norm() < 1e-10,
                "Round-trip mismatch at ({r}, {c}) for {rows}x{cols}"
            );
        }
    }
}

#[test]
#[verified_engine::verified]
fn test_mri_direct_dft_prime_and_rectangular() {
    for (rows, cols) in [(7, 11), (13, 17), (19, 5), (6, 15), (14, 8)] {
        assert_matches_direct_dft(rows, cols);
    }
}

#[test]
#[verified_engine::verified]
fn test_mri_single_line_matrices() {
    for (rows, cols) in [(1, 1), (1, 13), (11, 1), (1, 16), (16, 1)] {
        assert_matches_direct_dft(rows, cols);
    }
}

#[test]
#[verified_engine::verified]
fn test_mri_256x256_roundtrip_scaling_guard() {
    let size = 256;
    let density = DMatrix::from_fn(size, size, |r, c| {
        let x = r as f64 - size as f64 / 2.0;
        let y = c as f64 - size as f64 / 2.0;
        let re = if x * x + y * y < (size as f64 / 4.0).powi(2) {
            1.0
        } else {
            0.125
        };
        let im = ((r * 3 + c * 5) as f64 * 0.05).sin() * 0.25;
        Complex::new(re, im)
    });

    #[cfg(not(target_arch = "wasm32"))]
    #[allow(clippy::disallowed_methods)]
    let start = Instant::now();
    let k_space = simulate_signal_2d(&density);
    let reconstructed = inverse_dft_2d(&k_space);
    #[cfg(not(target_arch = "wasm32"))]
    {
        let elapsed = start.elapsed();
        println!("256x256 MRI FFT round-trip took: {:?}", elapsed);
        assert!(
            elapsed < Duration::from_secs(2),
            "256x256 round-trip exceeded latency budget: {:?}",
            elapsed
        );
    }

    let scale = (size * size) as f64;
    for (orig, recon) in density.iter().zip(reconstructed.iter()) {
        assert!(
            (*recon / scale - *orig).norm() < 1e-9,
            "256x256 round-trip reconstruction error exceeded tolerance"
        );
    }
}
