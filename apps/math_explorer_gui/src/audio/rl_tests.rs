//! Zero-allocation and `rustfft` spectral/panning tests for RL temporal-difference sonification.

#![cfg_attr(any(), verified(opt_out = "tests"))]

use verified_engine::engine::VerifiedEngine;

use super::backend::SonificationBackend;
use super::frame::{LIMITER_CEILING_DBFS, LIMITER_CEILING_LINEAR};
use super::mappings::rl_td_surprise_frame;
use super::synth::OfflinePcmSynthesizer;
use super::tests::{compute_mono_fft_magnitudes, peak_magnitude_around};

#[test]
fn test_rl_td_surprise_zero_alloc_1000_steps() {
    let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut synth = OfflinePcmSynthesizer::new(44_100, 64);
    let _ = synth.set_muted_from_user_gesture(false);

    let (max_sample, _metrics) = VerifiedEngine::run_verified(|| {
        let mut peak = 0.0_f32;
        for step in 0..1_000usize {
            let v = -8.0 + 16.0 * ((step % 25) as f64 / 24.0);
            let td = ((step % 11) as f64 - 5.0) * 1.4;
            let reward = match step % 20 {
                0 => 10.0,
                10 => -10.0,
                _ => -0.1,
            };
            let pan = if step % 2 == 0 { -0.75 } else { 0.75 };
            let frame = rl_td_surprise_frame(v, td, reward, pan);
            synth.apply_frame(&frame);
            for &s in synth.render_frames(32) {
                peak = peak.max(s.abs());
            }
        }
        peak
    });
    assert!(max_sample > 0.02);
    assert!(max_sample <= LIMITER_CEILING_LINEAR + 1e-6);
    assert!(synth.meter_reading().peak_dbfs <= LIMITER_CEILING_DBFS);
    assert!(!rl_td_surprise_frame(f64::NAN, 0.0, 0.0, 0.0).is_audible());
}

#[test]
fn test_rl_td_surprise_rustfft_value_pitch_and_wind_panning() {
    let sr = 16_384_u32;
    let fft_len = 4_096_usize;
    let mut synth = OfflinePcmSynthesizer::new(sr, fft_len + 1024);
    let _ = synth.set_muted_from_user_gesture(false);
    synth.set_master_volume(1.0);

    let low_frame = rl_td_surprise_frame(-6.0, 0.0, -0.1, -0.75);
    synth.apply_frame(&low_frame);
    let _ = synth.render_frames(1024);
    let low_pcm = synth.render_frames(fft_len).to_vec();
    let low_mags = compute_mono_fft_magnitudes(&low_pcm, fft_len);

    let mut left_sq = 0.0_f32;
    let mut right_sq = 0.0_f32;
    for &[left, right] in low_pcm.as_chunks::<2>().0 {
        left_sq += left * left;
        right_sq += right * right;
    }
    assert!(left_sq > right_sq * 1.5, "WindLeft pan must bias left RMS");

    let high_frame = rl_td_surprise_frame(8.0, 0.0, -0.1, 0.75);
    synth.apply_frame(&high_frame);
    let _ = synth.render_frames(1024);
    let high_mags = compute_mono_fft_magnitudes(synth.render_frames(fft_len), fft_len);

    let low_f0 = low_frame.voices[0].freq_hz;
    let high_f0 = high_frame.voices[0].freq_hz;
    assert!(high_f0 > low_f0 + 250.0);
    assert!(peak_magnitude_around(&low_mags, sr, fft_len, low_f0) > 20.0);
    assert!(peak_magnitude_around(&high_mags, sr, fft_len, high_f0) > 20.0);
}
