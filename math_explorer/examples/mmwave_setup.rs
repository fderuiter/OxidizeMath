//! Verified Algorithm for mmWave Radiotherapy Setup (Bressler et al. 2024).
//!
//! This example simulates a radar signal reflected from a target and runs the verified
//! processing pipeline: Coarse FFT -> Fine CZT -> Phase Extraction -> Displacement.

use math_explorer::num_complex::Complex;
use math_explorer::physics::medical::radar_gating::{C, FmcwConfig, chirp_z_transform};
use std::f64::consts::PI;

fn simulate_signal(
    config: &FmcwConfig,
    sample_rate: f64,
    n_samples: usize,
    true_range: f64,
    displacement_mm: f64,
) -> (f64, Vec<Complex<f64>>) {
    let true_displacement = displacement_mm / 1000.0;
    println!("\n--- Simulation ---");
    println!("True Range: {:.6} m", true_range);
    println!("True Displacement: {:.6} mm", displacement_mm);

    let sim_chirp_time = config.chirp_time;
    let beat_freq = (2.0 * config.bandwidth * true_range) / (C * sim_chirp_time);
    println!("Expected Beat Frequency: {:.2} Hz", beat_freq);

    let lambda = config.wavelength();
    let phase_shift = (4.0 * PI * true_displacement) / lambda;
    println!("Expected Phase Shift: {:.6} rad", phase_shift);

    let signal: Vec<Complex<f64>> = (0..n_samples)
        .map(|n| {
            let t = n as f64 / sample_rate;
            let phase = 2.0 * PI * beat_freq * t + phase_shift;
            Complex::new(0.0, phase).exp()
        })
        .collect();

    (beat_freq, signal)
}

fn coarse_search(
    config: &FmcwConfig,
    signal: &[Complex<f64>],
    sample_rate: f64,
    n_samples: usize,
    true_range: f64,
) -> f64 {
    println!("\n--- Step 1: Coarse Search (FFT) ---");
    let fft_output = chirp_z_transform(signal, 0.0, sample_rate, sample_rate, n_samples);

    let mut max_mag = -1.0;
    let mut peak_idx = 0;
    for (i, val) in fft_output.iter().enumerate() {
        if val.norm() > max_mag {
            max_mag = val.norm();
            peak_idx = i;
        }
    }

    let coarse_freq = peak_idx as f64 * sample_rate / n_samples as f64;
    let coarse_range = config.range_from_beat_frequency(coarse_freq);

    println!("Peak Index: {}", peak_idx);
    println!("Coarse Frequency: {:.2} Hz", coarse_freq);
    println!("Coarse Range: {:.4} m", coarse_range);
    println!("Error: {:.4} m", (coarse_range - true_range).abs());

    coarse_freq
}

fn fine_search(
    config: &FmcwConfig,
    signal: &[Complex<f64>],
    sample_rate: f64,
    coarse_freq: f64,
    true_range: f64,
) -> (f64, Complex<f64>) {
    println!("\n--- Step 2: Fine Search (CZT Zoom) ---");
    let zoom_bandwidth = 20_000.0;
    let start_freq = coarse_freq - zoom_bandwidth / 2.0;
    let zoom_bins = 100;

    let czt_output = chirp_z_transform(signal, start_freq, zoom_bandwidth, sample_rate, zoom_bins);

    let mut zoom_max_mag = -1.0;
    let mut zoom_peak_idx = 0;
    for (i, val) in czt_output.iter().enumerate() {
        if val.norm() > zoom_max_mag {
            zoom_max_mag = val.norm();
            zoom_peak_idx = i;
        }
    }

    let refined_freq = start_freq + zoom_bandwidth * (zoom_peak_idx as f64 / zoom_bins as f64);
    let refined_range = config.range_from_beat_frequency(refined_freq);

    println!("Zoom Peak Index: {}", zoom_peak_idx);
    println!("Refined Frequency: {:.2} Hz", refined_freq);
    println!("Refined Range: {:.6} m", refined_range);
    println!("Error: {:.6} m", (refined_range - true_range).abs());

    (refined_freq, czt_output[zoom_peak_idx])
}

fn extract_displacement(config: &FmcwConfig, peak_complex: Complex<f64>, displacement_mm: f64) {
    println!("\n--- Step 3: Phase & Displacement ---");
    let extracted_phase = peak_complex.arg();
    println!("Extracted Phase: {:.6} rad", extracted_phase);

    let calculated_displacement = config.displacement_from_phase(extracted_phase);

    println!(
        "Calculated Displacement: {:.6} mm",
        calculated_displacement * 1000.0
    );
    println!(
        "Displacement Error: {:.6} mm",
        (calculated_displacement * 1000.0 - displacement_mm).abs()
    );
}

fn main() {
    println!("=== mmWave Radiotherapy Setup Verification ===");
    let config = FmcwConfig::iwr6843_default();
    println!("Config: {:#?}", config);
    println!("Range Resolution (FFT): {:.4} m", config.range_resolution());

    let sample_rate = 10.0e6;
    let n_samples = (sample_rate * config.chirp_time).round() as usize;
    println!("Sample Rate: {:.2} MHz", sample_rate / 1.0e6);
    println!("Samples per Chirp: {}", n_samples);

    let true_range = 0.5;
    let displacement_mm = 0.1;

    let (_beat_freq, signal) =
        simulate_signal(&config, sample_rate, n_samples, true_range, displacement_mm);
    let coarse_freq = coarse_search(&config, &signal, sample_rate, n_samples, true_range);
    let (_refined_freq, peak_complex) =
        fine_search(&config, &signal, sample_rate, coarse_freq, true_range);
    extract_displacement(&config, peak_complex, displacement_mm);

    println!("\n=== Verification Complete ===");
}
