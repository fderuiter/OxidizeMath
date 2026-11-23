//! Zero-allocation, `rustfft` spectral, periodicity, mute-ramp, and `.wav` tests
//! for Ulam spiral, FRACTRAN register machine, and elliptic curve torsion sonification.

#![cfg_attr(any(), verified(opt_out = "tests"))]

use num_complex::Complex;
use rustfft::FftPlanner;
use verified_engine::engine::VerifiedEngine;

use super::backend::SonificationBackend;
use super::frame::{
    SonificationFrame, WaveformPreset, LIMITER_CEILING_DBFS, LIMITER_CEILING_LINEAR,
};
use super::mappings::{
    elliptic_torsion_frame, fractran_register_frame, just_intonation_ratio, ulam_spiral_frame,
    EllipticTorsionSample, UlamPolySample,
};
use super::synth::OfflinePcmSynthesizer;
use super::wav::synthesize_frames_to_wav;

fn mono_fft_magnitudes(interleaved_stereo: &[f32], fft_len: usize) -> Vec<f32> {
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

fn bin_peak(mags: &[f32], sample_rate: u32, fft_len: usize, target_hz: f32) -> f32 {
    let center = ((target_hz * (fft_len as f32)) / (sample_rate as f32)).round() as usize;
    let lo = center.saturating_sub(2);
    let hi = (center + 2).min(mags.len().saturating_sub(1));
    mags[lo..=hi].iter().copied().fold(0.0_f32, f32::max)
}

const fn sample(value: u64, is_prime: bool, prime_gap: u32) -> UlamPolySample {
    UlamPolySample {
        value,
        is_prime,
        prime_gap,
    }
}

#[test]
fn test_zero_alloc_1000_steps_number_theory_mappings_and_pcm_synthesis() {
    let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut synth = OfflinePcmSynthesizer::new(44_100, 64);
    let _ = synth.set_muted_from_user_gesture(false);

    let primes: [u64; 8] = [2, 3, 5, 7, 11, 13, 17, 19];
    let exps: [u32; 8] = [3, 2, 0, 1, 4, 0, 1, 2];
    let ulam_samples = [
        sample(41, true, 1),
        sample(43, true, 2),
        sample(47, true, 4),
        sample(53, true, 6),
    ];

    let (peak, _metrics) = VerifiedEngine::run_verified(|| {
        let mut max_abs = 0.0_f32;
        for step in 0..1_000_usize {
            let frame = match step % 3 {
                0 => ulam_spiral_frame(&ulam_samples, 140.0, step % 2 == 0),
                1 => fractran_register_frame(
                    &primes,
                    &exps,
                    if step % 25 == 0 { Some(5) } else { None },
                    28.0 + (step % 40) as f64,
                    18.0 + (step % 30) as f64,
                    false,
                ),
                _ => elliptic_torsion_frame(EllipticTorsionSample {
                    modulus_p: 97,
                    is_singular: false,
                    kp_coords: Some(((step as i64 * 7) % 97, (step as i64 * 13) % 97)),
                    p_order: 20,
                    scalar_k: (step as i64 % 20) + 1,
                    frobenius_trace: -10,
                    torus_wraps: (step % 4) + 1,
                }),
            };
            synth.apply_frame(&frame);
            for &s in synth.render_frames(32) {
                max_abs = max_abs.max(s.abs());
            }
        }
        max_abs
    });

    assert!(peak > 0.02);
    assert!(peak <= LIMITER_CEILING_LINEAR + 1e-6);
    assert!(synth.meter_reading().peak_dbfs <= LIMITER_CEILING_DBFS);
}

#[test]
fn test_rustfft_ulam_prime_diagonal_vs_composite_and_polyrhythm() {
    let euler_primes = [
        sample(41, true, 1),
        sample(47, true, 4),
        sample(61, true, 16),
    ];
    let composites = [
        sample(4, false, 8),
        sample(12, false, 8),
        sample(28, false, 8),
    ];

    let f_prime = ulam_spiral_frame(&euler_primes, 120.0, true);
    let f_comp = ulam_spiral_frame(&composites, 120.0, true);
    assert_eq!(f_prime.waveform, WaveformPreset::JustHarmonic);
    assert_eq!(f_comp.waveform, WaveformPreset::Sine);
    assert!(f_prime.voices[0].transient_impulse > 0.7);
    assert_eq!(f_comp.voices[0].transient_impulse, 0.0);
    assert!(f_prime.voices[0].fm_mod_hz > f_prime.voices[1].fm_mod_hz);
    assert!(f_prime.voices[1].fm_mod_hz > f_prime.voices[2].fm_mod_hz);

    let sr = 16_384_u32;
    let fft_len = 4_096_usize;
    let mut synth = OfflinePcmSynthesizer::new(sr, fft_len + 1024);
    let _ = synth.set_muted_from_user_gesture(false);

    synth.apply_frame(&f_prime);
    let _ = synth.render_frames(1024);
    let prime_mags = mono_fft_magnitudes(synth.render_frames(fft_len), fft_len);
    let f0 = f_prime.voices[0].freq_hz;
    let fundamental = bin_peak(&prime_mags, sr, fft_len, f0);
    let harmonic2 = bin_peak(&prime_mags, sr, fft_len, f0 * 2.0);

    synth.apply_frame(&f_comp);
    let _ = synth.render_frames(1024);
    let comp_mags = mono_fft_magnitudes(synth.render_frames(fft_len), fft_len);
    let comp_energy: f32 = comp_mags.iter().sum();
    let prime_energy: f32 = prime_mags.iter().sum();

    assert!(fundamental > 12.0 && harmonic2 > 2.5);
    assert!(
        prime_energy > comp_energy * 2.0,
        "Euler prime diagonal must carry stronger harmonic energy than composite ray"
    );
}

#[test]
fn test_rustfft_fractran_just_intonation_pow2_chime_complexity_and_30ms_halt_mute() {
    let primes = [2_u64, 3, 5, 7, 11, 13];
    let regs = [1_u32, 4, 2, 0, 1, 0];
    let low_c = fractran_register_frame(&primes, &regs, None, 8.0, 4.0, false);
    let high_c = fractran_register_frame(&primes, &regs, None, 180.0, 96.0, false);
    assert!(high_c.voices[0].filter_cutoff_hz > low_c.voices[0].filter_cutoff_hz + 1500.0);
    assert!(high_c.voices[0].fm_index > low_c.voices[0].fm_index + 0.5);

    let mut pure_chord = low_c;
    pure_chord.waveform = WaveformPreset::Sine;
    for v in &mut pure_chord.voices {
        v.fm_index = 0.0;
    }

    let sr = 16_384_u32;
    let fft_len = 4_096_usize;
    let mut synth = OfflinePcmSynthesizer::new(sr, fft_len + 1024);
    let _ = synth.set_muted_from_user_gesture(false);
    synth.apply_frame(&pure_chord);
    let _ = synth.render_frames(1024);
    let mags = mono_fft_magnitudes(synth.render_frames(fft_len), fft_len);

    // Dominant exponent is p = 3 (e = 4) -> 220 * 1.5 = 330 Hz; secondary p = 5 -> 275 Hz
    let p3_hz = 220.0 * just_intonation_ratio(3);
    let p5_hz = 220.0 * just_intonation_ratio(5);
    let m3 = bin_peak(&mags, sr, fft_len, p3_hz);
    let m5 = bin_peak(&mags, sr, fft_len, p5_hz);
    let m_void = bin_peak(&mags, sr, fft_len, 512.0);
    assert!(m3 > m5 && m5 > m_void * 6.0);

    // Pure 2^k prime hit emits high-frequency chime transient
    let pow2_regs = [7_u32, 0, 0, 0, 0, 0];
    let chime = fractran_register_frame(&primes, &pow2_regs, Some(7), 24.0, 12.0, false);
    assert!(chime.voices[0].transient_impulse >= 0.85);
    assert!(chime.voices[1].freq_hz > chime.voices[0].freq_hz);

    // Terminal halt or Floyd cycle decays output to silence within <= 30 ms at 44.1 kHz
    let mut live_synth = OfflinePcmSynthesizer::new(44_100, 2048);
    let _ = live_synth.set_muted_from_user_gesture(false);
    live_synth.apply_frame(&high_c);
    let _ = live_synth.render_frames(1024);

    let halted_frame = fractran_register_frame(&primes, &regs, None, 180.0, 96.0, true);
    assert_eq!(halted_frame, SonificationFrame::silent());
    live_synth.apply_frame(&halted_frame);
    let tail = live_synth.render_frames(1350);
    let after_30ms = tail[2640..]
        .iter()
        .copied()
        .fold(0.0_f32, |acc, s| acc.max(s.abs()));
    assert!(
        after_30ms < 0.001,
        "Halted FRACTRAN frame must decay to silence within <= 30 ms, got {after_30ms}"
    );
}

#[test]
fn test_elliptic_torsion_exact_periodicity_singular_silence_and_wav_export() {
    let singular = elliptic_torsion_frame(EllipticTorsionSample {
        modulus_p: 17,
        is_singular: true,
        kp_coords: Some((3, 5)),
        p_order: 9,
        scalar_k: 2,
        frobenius_trace: 0,
        torus_wraps: 2,
    });
    assert_eq!(singular, SonificationFrame::silent());

    // Exact torsion subgroup periodicity: frame(k) == frame(k + p_order)
    let order = 7_usize;
    let subgroup_coords: [Option<(i64, i64)>; 7] = [
        None,
        Some((1, 5)),
        Some((4, 19)),
        Some((8, 12)),
        Some((8, 17)),
        Some((4, 10)),
        Some((1, 24)),
    ];
    let sample_at = |k: i64| {
        let residue = k.rem_euclid(order as i64) as usize;
        elliptic_torsion_frame(EllipticTorsionSample {
            modulus_p: 29,
            is_singular: false,
            kp_coords: subgroup_coords[residue],
            p_order: order,
            scalar_k: k,
            frobenius_trace: -6,
            torus_wraps: (residue % 3) + 1,
        })
    };

    for k in 1..=(order as i64) {
        assert_eq!(
            sample_at(k),
            sample_at(k + order as i64),
            "Torsion sonification must repeat with exact period p_order = {order} at k = {k}"
        );
    }
    assert_ne!(sample_at(1), sample_at(2));
    assert_ne!(sample_at(1), sample_at(order as i64));

    let frames: Vec<SonificationFrame> = (1..=(order as i64)).map(sample_at).collect();
    let wav = synthesize_frames_to_wav(&frames, 44_100, 512, 0.85);
    assert_eq!(&wav[0..4], b"RIFF");
    assert_eq!(&wav[8..12], b"WAVE");
    assert!(
        wav[44..].iter().any(|&b| b != 0),
        "Synthesized stereo WAV payload must contain non-zero PCM audio"
    );
}
