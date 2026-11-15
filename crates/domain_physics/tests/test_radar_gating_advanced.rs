#![allow(missing_docs)]
use domain_physics::physics::medical::radar_gating::{
    clutter::EllipticalFilter, czt::SpatialCztConfig, mimo::Beamformer, phase::PhaseUnwrapper,
    super_resolution::MusicEstimator,
};
use num_complex::Complex;
use std::f64::consts::PI;

#[test]
#[verified_engine::verified]
fn test_phase_unwrapping_with_wraps() {
    let wavelength = 1.0; // Simplify math
    let mut unwrapper = PhaseUnwrapper::new(wavelength);

    // Simulate a displacement that moves 0.8 wavelengths in one step (phase change > PI)
    // 0.8 * 2PI = 1.6PI (> PI). Should wrap.
    // However, if we jump too fast (undersampling), we can't recover.
    // The unwrapper assumes jumps are < PI relative to the "nearest" multiple.
    // Let's simulate a smooth movement that accumulates to > 2PI total, but steps are small.

    let steps = 100;
    let max_displacement = 2.5 * wavelength; // 2.5 wavelengths total

    let mut final_disp = 0.0;

    for i in 0..=steps {
        let true_disp = max_displacement * (i as f64 / steps as f64);
        // phase = 4pi * d / lambda
        let phase = 4.0 * PI * true_disp / wavelength;

        let signal = Complex::new(0.0, phase).exp();
        final_disp = unwrapper.process(signal);
    }

    // Tolerance
    assert!(
        (final_disp - max_displacement).abs() < 1e-4,
        "Displacement {} != {}",
        final_disp,
        max_displacement
    );
}

#[test]
#[verified_engine::verified]
fn test_mimo_beamforming() {
    let lambda = 0.004; // 4mm
    let spacing = lambda / 2.0;
    let beamformer = Beamformer::new_ula(4, spacing, lambda);

    let target_angle = 30.0_f64.to_radians();

    // Simulate signal coming from target_angle
    // x_k = exp(-j * k_wave * d_k * sin(theta))
    let k_wave = 2.0 * PI / lambda;
    let mut signals = Vec::new();

    for k in 0..4 {
        let d_k = k as f64 * spacing;
        // Generate signal with positive phase to match the user's beamforming formula
        // which applies a negative phase shift e^{-j ...}
        let phase = k_wave * d_k * target_angle.sin();
        signals.push(Complex::new(0.0, phase).exp());
    }

    // Steer towards target
    let steered_signal = beamformer.steer(&signals, target_angle);
    // Constructive interference -> Amplitude should be approx 4.0
    assert!((steered_signal.norm() - 4.0).abs() < 1e-4);

    // Steer away (e.g., -30 degrees)
    let bad_signal = beamformer.steer(&signals, -30.0_f64.to_radians());
    // Destructive interference -> Amplitude should be small
    assert!(bad_signal.norm() < 4.0);
}

#[test]
#[verified_engine::verified]
fn test_elliptical_filter_clutter_removal() {
    let mut filter = EllipticalFilter::new(100.0);

    let center = Complex::new(5.0, -3.0); // Static Clutter
    let radius = 1.0;

    // Feed data: circle around center
    for i in 0..200 {
        let angle = 2.0 * PI * (i as f64 / 50.0);
        let signal = center + Complex::new(0.0, angle).exp() * radius;
        filter.filter(signal);
    }

    let estimate = filter.get_clutter_estimate();
    // Bounding box of a circle centers perfectly on the circle center
    assert!((estimate.re - center.re).abs() < 0.1);
    assert!((estimate.im - center.im).abs() < 0.1);
}

#[test]
#[verified_engine::verified]
fn test_czt_config_smoke() {
    let config = SpatialCztConfig {
        start_distance: 0.5,
        step_distance: 0.001,
        output_bins: 10,
        bandwidth: 4.0e9,
        chirp_time: 50.0e-6,
        c: 3.0e8,
    };

    let signal = vec![Complex::new(1.0, 0.0); 64];
    let spectrum = config.process(&signal, 1.0e6);

    assert_eq!(spectrum.len(), 10);
}

#[test]
#[verified_engine::verified]
fn test_czt_radar_peak_matching() {
    let start_dist = 0.5;
    let step_dist = 0.001; // 1mm resolution
    let bins = 50;
    let bandwidth = 4.0e9;
    let chirp_time = 50.0e-6;
    let c = 3.0e8;
    let sample_rate = 2.0e6;
    let n_samples = 256;

    let config = SpatialCztConfig {
        start_distance: start_dist,
        step_distance: step_dist,
        output_bins: bins,
        bandwidth,
        chirp_time,
        c,
    };

    // Target at 0.525 m (bin 25)
    let target_range = 0.525;
    // Beat frequency: f_b = (2 * B * R) / (c * T_c)
    let beat_freq = (2.0 * bandwidth * target_range) / (c * chirp_time);

    let signal: Vec<Complex<f64>> = (0..n_samples)
        .map(|n| {
            let t = n as f64 / sample_rate;
            Complex::new(0.0, 2.0 * PI * beat_freq * t).exp()
        })
        .collect();

    let spectrum = config.process(&signal, sample_rate);
    assert_eq!(spectrum.len(), bins);

    let mut max_mag = -1.0;
    let mut peak_bin = 0;
    for (i, val) in spectrum.iter().enumerate() {
        let mag = val.norm();
        if mag > max_mag {
            max_mag = mag;
            peak_bin = i;
        }
    }

    let estimated_range = start_dist + step_dist * peak_bin as f64;
    assert!(
        (estimated_range - target_range).abs() <= step_dist,
        "Estimated range {} differed from target range {}",
        estimated_range,
        target_range
    );
}

#[test]
#[verified_engine::verified]
fn test_music_smoke() {
    // 64 samples per chirp, 10 snapshots, 1 target
    let mut estimator = MusicEstimator::new(64, 10, 1).unwrap();

    // Add 10 dummy snapshots
    for _ in 0..10 {
        let chirp = vec![Complex::new(0.0, 0.0); 64];
        estimator.add_snapshot(&chirp).unwrap();
    }

    // Compute spectrum (should not panic)
    // Range 0.5 to 1.0m, step 0.01m
    let result = estimator.compute_spectrum(0.5, 1.0, 0.01, 4.0e9, 3.0e8);
    assert!(result.is_ok());
    let spectrum = result.unwrap();
    assert!(!spectrum.is_empty());
}
