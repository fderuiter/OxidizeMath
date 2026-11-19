#![allow(missing_docs)]
use domain_physics::error::RadarError;
use domain_physics::physics::medical::radar_gating::super_resolution::MusicEstimator;
use nalgebra::{DMatrix, DVector};
use num_complex::Complex;
use std::f64::consts::PI;

#[test]
#[verified_engine::verified]
fn test_music_invalid_construction() {
    // 0 samples
    assert!(matches!(
        MusicEstimator::new(0, 10, 1),
        Err(RadarError::InvalidConfiguration(_))
    ));

    // 0 smoothing
    assert!(matches!(
        MusicEstimator::new(64, 0, 1),
        Err(RadarError::InvalidConfiguration(_))
    ));

    // 0 targets
    assert!(matches!(
        MusicEstimator::new(64, 10, 0),
        Err(RadarError::InvalidConfiguration(_))
    ));

    // targets >= samples
    assert!(matches!(
        MusicEstimator::new(64, 10, 64),
        Err(RadarError::InvalidConfiguration(_))
    ));
}

#[test]
#[verified_engine::verified]
fn test_music_nan_handling() {
    let mut estimator = MusicEstimator::new(64, 10, 1).unwrap();

    // Inject NaN into snapshots
    let nan_chirp = vec![Complex::new(f64::NAN, 0.0); 64];

    // Fill with NaNs
    for _ in 0..10 {
        estimator.add_snapshot(&nan_chirp).unwrap();
    }

    // This should detect NaNs in eigenvalues and return error
    let result = estimator.compute_spectrum(0.0, 1.0, 0.1, 1e9, 3e8);

    match result {
        Err(RadarError::Math(math_commons::error::MathError::NumericalError { .. })) => (), // Pass
        Err(e) => panic!("Expected Math::NumericalError, got {:?}", e),
        Ok(_) => panic!("Expected Math::NumericalError, got Ok"),
    }
}

#[test]
#[verified_engine::verified]
fn test_music_invalid_compute_args() {
    let mut estimator = MusicEstimator::new(64, 10, 1).unwrap();
    // Add dummy data
    for _ in 0..10 {
        estimator
            .add_snapshot(&vec![Complex::new(0.0, 0.0); 64])
            .unwrap();
    }

    // Step <= 0
    assert!(matches!(
        estimator.compute_spectrum(0.0, 1.0, 0.0, 1e9, 3e8),
        Err(RadarError::InvalidConfiguration(_))
    ));

    // Start > End
    assert!(matches!(
        estimator.compute_spectrum(1.0, 0.0, 0.1, 1e9, 3e8),
        Err(RadarError::InvalidConfiguration(_))
    ));
}

#[verified_engine::verified]
fn build_dense_noise_projector(
    snapshots: &[Vec<Complex<f64>>],
    num_targets: usize,
) -> DMatrix<Complex<f64>> {
    let n = snapshots[0].len();
    assert!(num_targets < n);
    let mut r_xx = DMatrix::<Complex<f64>>::zeros(n, n);
    for snap in snapshots {
        let vec = DVector::from_iterator(n, snap.iter().cloned());
        r_xx += &vec * vec.adjoint();
    }
    r_xx /= Complex::new(snapshots.len() as f64, 0.0);

    let eigen = r_xx.symmetric_eigen();
    let mut pairs: Vec<(f64, usize)> = eigen
        .eigenvalues
        .iter()
        .enumerate()
        .map(|(i, &v)| (v, i))
        .collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    let mut p_noise = DMatrix::<Complex<f64>>::zeros(n, n);
    for &(_, idx) in pairs.iter().skip(num_targets) {
        let col = eigen.eigenvectors.column(idx);
        p_noise += col * col.adjoint();
    }
    p_noise
}

#[verified_engine::verified]
fn dense_reference_spectrum(
    snapshots: &[Vec<Complex<f64>>],
    num_targets: usize,
    range_bounds: (f64, f64, f64),
    rf_params: (f64, f64),
) -> Vec<(f64, f64)> {
    let n = snapshots[0].len();
    let (start_range, end_range, step_range) = range_bounds;
    let (bandwidth, c) = rf_params;
    let p_noise = build_dense_noise_projector(snapshots, num_targets);

    let steps = ((end_range - start_range) / step_range).round() as usize;
    let constant_factor = (4.0 * PI * bandwidth) / (c * n as f64);
    let mut a_vec = DVector::from_element(n, Complex::new(0.0, 0.0));
    let mut tmp = DVector::from_element(n, Complex::new(0.0, 0.0));
    let mut spectrum = Vec::with_capacity(steps + 1);
    let mut r = start_range;

    for _ in 0..=steps {
        let (sin_a, cos_a) = (constant_factor * r).sin_cos();
        let step_phasor = Complex::new(cos_a, sin_a);
        a_vec[0] = Complex::new(1.0, 0.0);
        for k in 1..n {
            a_vec[k] = a_vec[k - 1] * step_phasor;
        }
        p_noise.mul_to(&a_vec, &mut tmp);
        let den = a_vec.dotc(&tmp).norm();
        spectrum.push((r, 1.0 / (den + 1e-12)));
        r += step_range;
    }
    spectrum
}

#[verified_engine::verified]
fn synthesize_snapshots(
    n: usize,
    num_snapshots: usize,
    target_ranges: &[f64],
    bandwidth: f64,
    c: f64,
    background_scale: f64,
) -> Vec<Vec<Complex<f64>>> {
    let factor = (4.0 * PI * bandwidth) / (c * n as f64);
    let mut out = Vec::with_capacity(num_snapshots);
    for m in 0..num_snapshots {
        let mut chirp = Vec::with_capacity(n);
        for k in 0..n {
            let mut sample = Complex::new(0.0, 0.0);
            for (t_idx, &r_target) in target_ranges.iter().enumerate() {
                let snap_phase = 0.37 * (m as f64 + 1.0) * (t_idx as f64 + 1.0);
                let phase = factor * r_target * k as f64 + snap_phase;
                sample += Complex::new(phase.cos(), phase.sin());
            }
            if background_scale > 0.0 {
                let bg_phase = 0.73 * (m as f64 + 1.0) * (k as f64 + 1.0);
                sample += Complex::new(bg_phase.cos(), bg_phase.sin()) * background_scale;
            }
            chirp.push(sample);
        }
        out.push(chirp);
    }
    assert_eq!(out.len(), num_snapshots);
    out
}

#[verified_engine::verified]
fn assert_spectrum_parity(actual: &[(f64, f64)], expected: &[(f64, f64)], rel_tol: f64) {
    assert_eq!(actual.len(), expected.len());
    for (&(r_act, p_act), &(r_exp, p_exp)) in actual.iter().zip(expected.iter()) {
        assert_eq!(r_act.to_bits(), r_exp.to_bits());
        assert!(p_act.is_finite() && p_act > 0.0);
        let rel_err = (p_act - p_exp).abs() / p_exp.max(1.0);
        assert!(
            rel_err < rel_tol,
            "Mismatch at range {r_act}: actual={p_act}, expected={p_exp}, rel_err={rel_err}"
        );
    }
}

#[test]
#[verified_engine::verified]
fn test_music_parity_signal_smaller_subspace() {
    // P = 2 < N - P = 14 exercises the signal-complement basis path
    let (n, m, p) = (16, 12, 2);
    let (bandwidth, c) = (4.0e9, 3.0e8);
    let snapshots = synthesize_snapshots(n, m, &[0.85, 2.15], bandwidth, c, 0.05);

    let mut estimator = MusicEstimator::new(n, m, p).unwrap();
    for snap in &snapshots {
        estimator.add_snapshot(snap).unwrap();
    }

    let actual = estimator
        .compute_spectrum(0.2, 3.0, 0.01, bandwidth, c)
        .unwrap();
    let expected = dense_reference_spectrum(&snapshots, p, (0.2, 3.0, 0.01), (bandwidth, c));
    assert_spectrum_parity(&actual, &expected, 1e-9);
}

#[test]
#[verified_engine::verified]
fn test_music_parity_noise_smaller_and_balanced_subspace() {
    let (bandwidth, c) = (2.0e9, 3.0e8);

    // Case 1: P = 7 > N - P = 3 exercises the direct noise-basis path
    let snaps_noise_smaller =
        synthesize_snapshots(10, 14, &[0.4, 0.9, 1.4, 1.9], bandwidth, c, 0.1);
    let mut est_noise = MusicEstimator::new(10, 14, 7).unwrap();
    for snap in &snaps_noise_smaller {
        est_noise.add_snapshot(snap).unwrap();
    }
    let act_noise = est_noise
        .compute_spectrum(0.1, 2.5, 0.02, bandwidth, c)
        .unwrap();
    let exp_noise =
        dense_reference_spectrum(&snaps_noise_smaller, 7, (0.1, 2.5, 0.02), (bandwidth, c));
    assert_spectrum_parity(&act_noise, &exp_noise, 1e-11);

    // Case 2: P = 4 == N - P = 4 exercises the boundary tie-breaker
    let snaps_tie = synthesize_snapshots(8, 12, &[0.6, 1.7], bandwidth, c, 0.08);
    let mut est_tie = MusicEstimator::new(8, 12, 4).unwrap();
    for snap in &snaps_tie {
        est_tie.add_snapshot(snap).unwrap();
    }
    let act_tie = est_tie
        .compute_spectrum(0.1, 2.5, 0.02, bandwidth, c)
        .unwrap();
    let exp_tie = dense_reference_spectrum(&snaps_tie, 4, (0.1, 2.5, 0.02), (bandwidth, c));
    assert_spectrum_parity(&act_tie, &exp_tie, 1e-11);
}

#[test]
#[verified_engine::verified]
fn test_music_exact_peak_clamping_and_localization() {
    // Noiseless 2-target signal on exact grid steps (P = 2 < N - P = 30)
    // Near exact peaks, ||a||^2 - ||E_s^H a||^2 is ~0 up to rounding residuals.
    let (n, m, p) = (32, 16, 2);
    let (bandwidth, c) = (1.5e9, 3.0e8);
    let targets = [0.25, 0.65];
    let snapshots = synthesize_snapshots(n, m, &targets, bandwidth, c, 0.0);

    let mut estimator = MusicEstimator::new(n, m, p).unwrap();
    for snap in &snapshots {
        estimator.add_snapshot(snap).unwrap();
    }

    let spectrum = estimator
        .compute_spectrum(0.0, 0.9, 0.01, bandwidth, c)
        .unwrap();
    let dense = dense_reference_spectrum(&snapshots, p, (0.0, 0.9, 0.01), (bandwidth, c));
    assert_eq!(spectrum.len(), dense.len());

    for &(r, power) in &spectrum {
        assert!(
            power.is_finite() && power > 0.0 && power <= 1.000_001e12,
            "Invalid power {power} at range {r}"
        );
    }

    // Both exact target bins (indices 25 and 65) must exhibit sharp super-resolution peaks
    assert!(spectrum[25].1 > 1.0e10 && dense[25].1 > 1.0e10);
    assert!(spectrum[65].1 > 1.0e10 && dense[65].1 > 1.0e10);
    assert!(spectrum[25].1 > spectrum[24].1 * 100.0 && spectrum[25].1 > spectrum[26].1 * 100.0);
    assert!(spectrum[65].1 > spectrum[64].1 * 100.0 && spectrum[65].1 > spectrum[66].1 * 100.0);
}
