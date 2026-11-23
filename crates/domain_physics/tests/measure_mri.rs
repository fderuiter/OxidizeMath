#![allow(missing_docs)]
use domain_physics::physics::mri::phantom::fill_shepp_logan_grid;
use domain_physics::physics::mri::reconstruction::{
    Fft2dWorkspace, fftshift_2d_into, ifftshift_2d_into, inverse_dft_2d, simulate_signal_2d,
};
use nalgebra::DMatrix;
use num_complex::Complex;
use std::f64::consts::PI;
#[cfg(not(target_arch = "wasm32"))]
use std::time::{Duration, Instant};
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

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

#[verified_engine::verified]
fn verify_zero_alloc_workspace_grid(size: usize) {
    let mut rho = vec![0.0; size * size];
    let mut t1 = vec![0.0; size * size];
    let mut t2 = vec![0.0; size * size];
    fill_shepp_logan_grid(size, size, &mut rho, &mut t1, &mut t2);

    let density = DMatrix::from_fn(size, size, |r, c| {
        Complex::new(
            rho[math_commons::math_kernel::types::flatten_2d_index(c, r, size)],
            0.0,
        )
    });
    let mut ws = Fft2dWorkspace::new(size, size);
    let mut k_raw = DMatrix::zeros(size, size);
    let mut k_shifted = DMatrix::zeros(size, size);
    let mut k_unshifted = DMatrix::zeros(size, size);
    let mut recon = DMatrix::zeros(size, size);

    let ((), _metrics) = VerifiedEngine::run_verified(|| {
        ws.simulate_signal_2d_into(&density, &mut k_raw);
        fftshift_2d_into(&k_raw, &mut k_shifted);
        ifftshift_2d_into(&k_shifted, &mut k_unshifted);
        ws.reconstruct_normalized_2d_into(&k_unshifted, &mut recon);
    });

    assert_eq!((ws.rows(), ws.cols()), (size, size));
    for (orig, rec) in density.iter().zip(recon.iter()) {
        assert!(
            (*rec - *orig).norm() < 1e-10,
            "Zero-alloc {size}x{size} round-trip mismatch"
        );
    }
}

#[test]
#[verified_engine::verified]
fn test_fft2d_workspace_zero_allocation_64_and_128() {
    verify_zero_alloc_workspace_grid(64);
    verify_zero_alloc_workspace_grid(128);

    let mut ws = Fft2dWorkspace::new(0, 0);
    let empty = DMatrix::zeros(0, 0);
    let mut out = DMatrix::zeros(0, 0);
    ws.simulate_signal_2d_into(&empty, &mut out);
    ws.resize(7, 5);
    let odd = DMatrix::from_fn(7, 5, |r, c| Complex::new((r * 5 + c) as f64, -0.5));
    let mut shifted = DMatrix::zeros(7, 5);
    let mut unshifted = DMatrix::zeros(7, 5);
    fftshift_2d_into(&odd, &mut shifted);
    ifftshift_2d_into(&shifted, &mut unshifted);
    assert_eq!(odd, unshifted);
}

#[test]
#[verified_engine::verified]
fn test_mri_128x128_pass_sub_millisecond() {
    let size = 128;
    let total = size * size;
    let mut bloch =
        domain_physics::physics::mri::bloch::BlochSimulator::new(nalgebra::Vector3::zeros(), 1.0);
    let mut rho = vec![0.0; total];
    let mut t1 = vec![1.0; total];
    let mut t2 = vec![0.1; total];
    fill_shepp_logan_grid(size, size, &mut rho, &mut t1, &mut t2);
    let exp_t1: Vec<f64> = t1.iter().map(|&v| (-0.002 / v.max(1e-4)).exp()).collect();
    let exp_t2: Vec<f64> = t2.iter().map(|&v| (-0.002 / v.max(1e-4)).exp()).collect();
    let delta_b0 = vec![0.0; total];
    let mut spins = vec![[nalgebra::Vector3::new(0.0, 0.8, 0.2); 3]; total];
    let mut trans = DMatrix::zeros(size, size);
    let mut out_kspace = vec![Complex::new(0.0, 0.0); total];
    let mut recon_mag = vec![0.0; total];
    let mut ws128 = Fft2dWorkspace::new(size, size);

    let ((), _metrics) = VerifiedEngine::run_verified(|| {
        bloch.advance_isochromat_grid(&mut spins, &rho, &exp_t1, &exp_t2, &delta_b0, 0.85, 0.002);
        for ((ens, out), mag) in spins
            .iter()
            .zip(trans.as_mut_slice().iter_mut())
            .zip(recon_mag.iter_mut())
        {
            let mx = (ens[0].x + ens[1].x + ens[2].x) * (1.0 / 3.0);
            let my = (ens[0].y + ens[1].y + ens[2].y) * (1.0 / 3.0);
            *out = Complex::new(mx, my);
            *mag = (mx * mx + my * my).sqrt();
        }
        ws128.forward_transposed_in_place(&mut trans, &mut out_kspace);
    });
    assert!(recon_mag.iter().any(|&v| v > 0.1));
}
