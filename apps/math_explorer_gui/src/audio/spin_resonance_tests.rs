#![cfg_attr(any(), verified(opt_out = "tests"))]

use num_complex::Complex;
use rustfft::FftPlanner;
use verified_engine::engine::VerifiedEngine;

use super::backend::SonificationBackend;
use super::frame::LIMITER_CEILING_LINEAR;
use super::mappings::spin_resonance::larmor_carrier_hz;
use super::mappings::{
    bloch_fid_frame, hahn_echo_isochromat_frame, kspace_voxel_ensemble_frame, zeeman_bloch_frame,
};
use super::synth::OfflinePcmSynthesizer;

fn compute_mono_fft_magnitudes(interleaved_stereo: &[f32], fft_len: usize) -> Vec<f32> {
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(fft_len);
    let mut buf: Vec<Complex<f32>> = (0..fft_len)
        .map(|i| {
            let l = interleaved_stereo.get(i * 2).copied().unwrap_or(0.0);
            let r = interleaved_stereo.get(i * 2 + 1).copied().unwrap_or(0.0);
            let hann = 0.5 * (1.0 - (std::f32::consts::TAU * (i as f32) / (fft_len as f32)).cos());
            Complex::new(0.5 * (l + r) * hann, 0.0)
        })
        .collect();
    fft.process(&mut buf);
    buf[..fft_len / 2].iter().map(|c| c.norm()).collect()
}

fn peak_magnitude_around(mags: &[f32], sample_rate: u32, fft_len: usize, target_hz: f32) -> f32 {
    let center = ((target_hz * (fft_len as f32)) / (sample_rate as f32)).round() as usize;
    let lo = center.saturating_sub(2);
    let hi = (center + 2).min(mags.len().saturating_sub(1));
    mags[lo..=hi].iter().copied().fold(0.0_f32, f32::max)
}

#[test]
fn test_zero_alloc_1000_steps_spin_resonance_mappings_and_pcm_synth() {
    let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut synth = OfflinePcmSynthesizer::new(44_100, 64);
    let _ = synth.set_muted_from_user_gesture(false);
    let offsets = [-42.0, -31.5, -21.0, -10.5, 0.0, 10.5, 21.0, 31.5, 42.0];
    let probe = [nalgebra::Vector3::new(0.7, 0.3, 0.4); 8];
    let hahn = [nalgebra::Vector3::new(0.8, 0.2, 0.1); 9];

    let (peak, _metrics) = VerifiedEngine::run_verified(|| {
        let mut max_abs = 0.0_f32;
        for step in 0..1_000usize {
            let ph = (step as f64) * 0.04;
            let frame = match step % 4 {
                0 => bloch_fid_frame(
                    nalgebra::Vector3::new(ph.cos(), ph.sin(), 0.2),
                    nalgebra::Vector3::new(1e-5, 0.0, 1.5),
                    1.0,
                ),
                1 => hahn_echo_isochromat_frame(&hahn, &offsets, 1.5, 1.0),
                2 => kspace_voxel_ensemble_frame(&probe, 2.2e-7, ph, [24.0, -12.0], 1.0),
                _ => zeeman_bloch_frame([ph.cos(), ph.sin(), 0.25], [0.5, 0.0, 1.5]),
            };
            synth.apply_frame(&frame);
            for &s in synth.render_frames(32) {
                max_abs = max_abs.max(s.abs());
            }
        }
        max_abs
    });
    assert!(peak > 0.01 && peak <= LIMITER_CEILING_LINEAR + 1e-6);
}

#[test]
fn test_rustfft_larmor_scaling_dephasing_decay_and_echo_recovery() {
    let sr = 16_384_u32;
    let fft_len = 4_096_usize;
    let mut synth = OfflinePcmSynthesizer::new(sr, fft_len + 1024);
    let _ = synth.set_muted_from_user_gesture(false);

    let f_low = bloch_fid_frame(
        nalgebra::Vector3::new(1.0, 0.0, 0.0),
        nalgebra::Vector3::new(0.0, 0.0, 0.5),
        1.0,
    );
    let f_high = bloch_fid_frame(
        nalgebra::Vector3::new(1.0, 0.0, 0.0),
        nalgebra::Vector3::new(0.0, 0.0, 2.5),
        1.0,
    );
    assert!(f_high.voices[0].freq_hz > f_low.voices[0].freq_hz + 250.0);

    synth.apply_frame(&f_low);
    let _ = synth.render_frames(1024);
    let mags_low = compute_mono_fft_magnitudes(synth.render_frames(fft_len), fft_len);
    let p_low = peak_magnitude_around(&mags_low, sr, fft_len, larmor_carrier_hz(0.5));

    synth.apply_frame(&f_high);
    let _ = synth.render_frames(1024);
    let mags_high = compute_mono_fft_magnitudes(synth.render_frames(fft_len), fft_len);
    let p_high = peak_magnitude_around(&mags_high, sr, fft_len, larmor_carrier_hz(2.5));
    assert!(p_low > 20.0 && p_high > 20.0);

    // Hahn 9-isochromat destructive T2* dephasing mid-interval vs coherent recovery at t = TE
    let offsets = [-42.0, -31.5, -21.0, -10.5, 0.0, 10.5, 21.0, 31.5, 42.0];
    let dephased_spins: [nalgebra::Vector3<f64>; 9] = std::array::from_fn(|i| {
        let th = std::f64::consts::TAU * offsets[i] * 0.012;
        nalgebra::Vector3::new(0.85 * th.cos(), 0.85 * th.sin(), 0.1)
    });
    let refocused_spins = [nalgebra::Vector3::new(-0.78, 0.0, 0.15); 9];

    let mid_frame = hahn_echo_isochromat_frame(&dephased_spins, &offsets, 1.5, 1.0);
    let echo_frame = hahn_echo_isochromat_frame(&refocused_spins, &offsets, 1.5, 1.0);
    assert!(echo_frame.master_gain > mid_frame.master_gain * 2.0);

    synth.apply_frame(&mid_frame);
    let _ = synth.render_frames(1024);
    let mid_mags = compute_mono_fft_magnitudes(synth.render_frames(fft_len), fft_len);
    let mid_peak = peak_magnitude_around(&mid_mags, sr, fft_len, larmor_carrier_hz(1.5));

    synth.apply_frame(&echo_frame);
    let _ = synth.render_frames(1024);
    let echo_mags = compute_mono_fft_magnitudes(synth.render_frames(fft_len), fft_len);
    let echo_peak = peak_magnitude_around(&echo_mags, sr, fft_len, larmor_carrier_hz(1.5));
    assert!(
        echo_peak > mid_peak * 2.5,
        "Hahn spin-echo recovery at t=TE ({echo_peak}) must exceed dephased mid-interval ({mid_peak})"
    );

    assert!(!zeeman_bloch_frame([f64::NAN, 0.0, 1.0], [0.0, 0.0, 1.0]).is_audible());
}
