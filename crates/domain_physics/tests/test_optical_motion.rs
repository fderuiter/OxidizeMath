#![allow(missing_docs)]
use domain_physics::physics::medical::{
    evaluation::{
        pearson_correlation as eval_pearson_correlation,
        root_mean_square_error as eval_root_mean_square_error, spatial_accuracy_error,
    },
    optical_motion::{
        calibration::LinearCalibrator,
        physics::{
            calculate_lambertian_order, calculate_photocurrent, calculate_received_power,
            calculate_tia_output,
        },
        processing::{
            LockInAmplifier, calculate_time_delay, snr_improvement_factor, weighted_average_height,
        },
        validation::{
            dice_similarity_coefficient, pearson_correlation, percentage_error,
            root_mean_square_error, time_shift_error,
        },
    },
    radar_gating::surface::weighted_average_height as radar_weighted_average_height,
};
use nalgebra::Point3;
use std::f64::consts::PI;
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

#[test]
#[verified_engine::verified]
fn test_physical_principles() {
    // Lambertian Order
    let phi_half = PI / 3.0; // 60 degrees
    let n = calculate_lambertian_order(phi_half);
    // cos(60) = 0.5. -ln(2) / ln(0.5) = -ln(2) / -ln(2) = 1.0.
    assert!((n - 1.0).abs() < math_commons::registry::TOLERANCE_FAST);

    // Received Power
    let a = 1e-4; // 1 cm^2 = 1e-4 m^2
    let p_t = 0.1; // 100 mW
    let d = 1.0; // 1 meter
    let phi = 0.0; // 0 degrees
    let theta = 0.0; // 0 degrees
    let gamma = 2.0; // free space
    // n = 1.
    // Pd = (2 * A * Pt) / (2 * pi * 1) * 1 * 1 = (A * Pt) / pi
    // Pd = 1e-5 / pi
    let pd = calculate_received_power(n, a, p_t, d, phi, theta, gamma);
    let expected_pd = (a * p_t) / PI;
    assert!((pd - expected_pd).abs() < math_commons::registry::TOLERANCE_STANDARD);

    // Photocurrent
    // i_d = math_commons::registry::TOLERANCE_STANDARD (1 nA), R_pd = 0.5 A/W, Pr = math_commons::registry::TOLERANCE_FAST (1 uW)
    let i_d = math_commons::registry::TOLERANCE_STANDARD;
    let r_pd = 0.5;
    let p_r = math_commons::registry::TOLERANCE_FAST;
    let i_pd = calculate_photocurrent(p_r, i_d, r_pd);
    assert!((i_pd - (math_commons::registry::TOLERANCE_STANDARD + 0.5e-6)).abs() < 1e-12);

    // TIA Output
    let g_pd = 1e5; // 100 kOhm
    let v_sig = calculate_tia_output(i_pd, g_pd);
    assert!((v_sig - (i_pd * 1e5)).abs() < math_commons::registry::TOLERANCE_STANDARD);
}

#[test]
#[verified_engine::verified]
fn test_wah_and_snr() {
    let vertices = vec![
        Point3::new(0.0, 0.0, 10.0),
        Point3::new(1.0, 0.0, 12.0),
        Point3::new(0.0, 1.0, 8.0),
    ];
    let wah = weighted_average_height(&vertices);
    assert!((wah - 10.0).abs() < math_commons::registry::TOLERANCE_FAST);

    let snr_boost = snr_improvement_factor(400);
    assert!((snr_boost - 20.0).abs() < math_commons::registry::TOLERANCE_FAST);
}

#[test]
#[verified_engine::verified]
fn test_lock_in_amplifier() {
    let lia = LockInAmplifier::new(1.0, 10.0, 1.0);
    // Ideal case: Signal in phase, phase_diff = 0
    let v_s = 2.0;
    let _v_r = 1.0; // stored in LIA as reference_amplitude
    let (vx, vy) = lia.mix_and_filter(v_s, 0.0);
    // Vx = (2*1)/2 * cos(0) = 1.0
    // Vy = (2*1)/2 * sin(0) = 0.0
    assert!((vx - 1.0).abs() < math_commons::registry::TOLERANCE_FAST);
    assert!(vy.abs() < math_commons::registry::TOLERANCE_FAST);

    let (r, theta) = lia.calculate_magnitude_phase(vx, vy);
    assert!((r - 1.0).abs() < math_commons::registry::TOLERANCE_FAST);
    assert!(theta.abs() < math_commons::registry::TOLERANCE_FAST);

    // Scaled Output
    // R_scaled = (Vfs/S) * R = (10/1) * 1 = 10.0
    let output = lia.scale_output(r);
    assert!((output - 10.0).abs() < math_commons::registry::TOLERANCE_FAST);
}

#[test]
#[verified_engine::verified]
fn test_time_delay() {
    // Create a signal and a delayed version
    let sample_rate = 100.0;
    let n = 100;
    let mut signal1 = Vec::new();
    let mut signal2 = Vec::new();

    // Signal: sin wave
    for i in 0..n {
        let t = i as f64 / sample_rate;
        signal1.push((2.0 * PI * 5.0 * t).sin());
        // Delay by 10 samples (0.1s)
        let t_delayed = (i as f64 - 10.0) / sample_rate;
        signal2.push((2.0 * PI * 5.0 * t_delayed).sin());
    }

    // The delay calculation assumes circular shift or padding.
    // For pure sin wave, phase shift is detected.
    // FFT correlation is circular correlation.
    let delay = calculate_time_delay(&signal1, &signal2, sample_rate);
    // Expected delay is 0.1s.
    // Note: Due to noise or edge effects, it might be close.
    assert!((delay - 0.1).abs() < 0.02); // 0.02s tolerance (2 samples)
}

#[test]
#[verified_engine::verified]
fn test_validation_metrics() {
    let measured = vec![1.0, 2.0, 3.0];
    let reference = vec![1.1, 1.9, 3.1];

    // Percentage Error (using single value function for demonstration)
    let pe = percentage_error(measured[0], reference[0]);
    // (1.0 - 1.1) / 1.0 * 100 = -10%
    assert!((pe + 10.0).abs() < math_commons::registry::TOLERANCE_FAST);

    // RMSE
    // Errors: -0.1, 0.1, -0.1
    // Sq Errors: 0.01, 0.01, 0.01. Mean: 0.01. Sqrt: 0.1
    let rmse = root_mean_square_error(&measured, &reference);
    assert!((rmse - 0.1).abs() < math_commons::registry::TOLERANCE_FAST);

    // Pearson
    // Perfect correlation despite scaling/shifting?
    // x: 1, 2, 3. y: 1.1, 1.9, 3.1. Line is y = x + noise.
    // It should be high.
    let r = pearson_correlation(&measured, &reference);
    assert!(r > 0.9);

    // Dice
    let mask_a = vec![true, true, false, false];
    let mask_b = vec![true, false, true, false];
    // A: {0, 1}, B: {0, 2}. Int: {0}.
    // Size A: 2, Size B: 2.
    // DSC = 2*1 / (2+2) = 0.5.
    let dsc = dice_similarity_coefficient(&mask_a, &mask_b);
    assert!((dsc - 0.5).abs() < math_commons::registry::TOLERANCE_FAST);

    // Time Shift Error
    let err = time_shift_error(&measured, &reference);
    // Sum sq diffs: 0.01 + 0.01 + 0.01 = 0.03
    assert!((err - 0.03).abs() < math_commons::registry::TOLERANCE_FAST);
}

#[test]
#[verified_engine::verified]
fn test_calibration() {
    let voltages = vec![1.0, 2.0, 3.0];
    let distances = vec![10.0, 20.0, 30.0];
    // y = 10x
    let calibrator = LinearCalibrator::fit(&voltages, &distances).unwrap();
    let d = calibrator.calibrate(1.5);
    assert!((d - 15.0).abs() < math_commons::registry::TOLERANCE_FAST);
}

#[test]
#[verified_engine::verified]
fn test_kernel_parity_truncation_and_dc_offset() {
    let offset = 1.0e9;
    let x_eq = [offset + 1.0, offset + 2.0, offset + 3.0, offset + 4.0];
    let y_eq = [offset + 1.5, offset + 2.8, offset + 4.2, offset + 5.7];

    let r_val = pearson_correlation(&x_eq, &y_eq);
    let r_eval = eval_pearson_correlation(&x_eq, &y_eq).unwrap();
    assert_eq!(r_val, r_eval);
    assert!(r_val > 0.99);

    let rmse_val = root_mean_square_error(&x_eq, &y_eq);
    let rmse_eval = eval_root_mean_square_error(&x_eq, &y_eq).unwrap();
    assert_eq!(rmse_val, rmse_eval);

    let x_long = [
        offset + 1.0,
        offset + 2.0,
        offset + 3.0,
        offset + 4.0,
        -999.0,
    ];
    assert_eq!(pearson_correlation(&x_long, &y_eq), r_eval);
    assert_eq!(eval_pearson_correlation(&x_long, &y_eq), None);
    assert_eq!(root_mean_square_error(&x_long, &y_eq), rmse_eval);
    assert_eq!(eval_root_mean_square_error(&x_long, &y_eq), None);

    assert_eq!(pearson_correlation(&[], &[]), 0.0);
    assert_eq!(pearson_correlation(&[2.0], &[2.0]), 0.0);
    assert_eq!(pearson_correlation(&[4.0, 4.0, 4.0], &[1.0, 2.0, 3.0]), 0.0);
    assert_eq!(root_mean_square_error(&[], &[1.0, 2.0]), 0.0);

    assert_eq!(percentage_error(0.0, 1.0), f64::INFINITY);
    assert_eq!(
        percentage_error(50.0, 45.0),
        spatial_accuracy_error(50.0, 45.0)
    );
    assert!((percentage_error(1.0e-12, 0.0) - 100.0).abs() < 1.0e-12);
    assert_eq!(spatial_accuracy_error(1.0e-12, 0.0), 0.0);
}

#[test]
#[verified_engine::verified]
fn test_optical_motion_kernels_zero_allocation() {
    let vertices = [
        Point3::new(-1.0, 0.0, 8.0),
        Point3::new(0.0, 1.0, 10.0),
        Point3::new(1.0, -1.0, 12.0),
    ];
    let x = [1.0e8 + 2.0, 1.0e8 + 4.0, 1.0e8 + 6.0, 1.0e8 + 8.0];
    let y = [1.0e8 + 1.0, 1.0e8 + 2.0, 1.0e8 + 3.0, 1.0e8 + 4.0, 99.0];

    let ((wah_opt, wah_rad, pe, rmse, corr), metrics) = VerifiedEngine::run_verified(|| {
        let wah_opt = weighted_average_height(&vertices);
        let wah_rad = radar_weighted_average_height(&vertices);
        let pe = percentage_error(25.0, 20.0);
        let rmse = root_mean_square_error(&x, &y);
        let corr = pearson_correlation(&x, &y);
        (wah_opt, wah_rad, pe, rmse, corr)
    });

    assert!(metrics.function_calls >= 10);
    assert_eq!(wah_opt, wah_rad);
    assert!((wah_opt - 10.0).abs() < math_commons::registry::TOLERANCE_FAST);
    assert_eq!(weighted_average_height(&[]), 0.0);
    assert!((pe - 20.0).abs() < 1.0e-12);
    assert!(rmse > 2.0);
    assert!((corr - 1.0).abs() < 1.0e-12);
}
