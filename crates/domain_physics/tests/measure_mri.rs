#![allow(missing_docs)]
use domain_physics::physics::mri::reconstruction::{inverse_dft_2d, simulate_signal_2d};
use nalgebra::DMatrix;
use num_complex::Complex;
use std::time::Instant;

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
    let start_sim = Instant::now();
    let signal = simulate_signal_2d(&density);
    let duration_sim = start_sim.elapsed();
    println!("simulate_signal_2d took: {:?}", duration_sim);

    println!("Running inverse_dft_2d for {}x{} matrix...", size, size);
    let start_inv = Instant::now();
    let reconstructed = inverse_dft_2d(&signal);
    let duration_inv = start_inv.elapsed();
    println!("inverse_dft_2d took: {:?}", duration_inv);

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
    let empty = DMatrix::zeros(0, 0);
    let sig = simulate_signal_2d(&empty);
    assert_eq!(sig.nrows(), 0);
    assert_eq!(sig.ncols(), 0);
    let recon = inverse_dft_2d(&sig);
    assert_eq!(recon.nrows(), 0);
    assert_eq!(recon.ncols(), 0);
}
