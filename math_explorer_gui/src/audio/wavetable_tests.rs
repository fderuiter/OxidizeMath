#![cfg_attr(any(), verified(opt_out = "tests"))]

use num_complex::Complex;
use rustfft::FftPlanner;
use verified_engine::engine::VerifiedEngine;

use super::backend::MuteReason;
use super::studio_preview::{
    snapshot_studio_preview, trigger_studio_ir_strike_from_gesture, StudioPreviewController,
};
use super::wav::{
    encode_stereo_ir_wav, encode_wav_samples, encode_wavetable_wav, inspect_wav_riff,
    WavChannelLayout, WavEncodeSpec, WavSampleFormat, WavSampleRate,
};
use super::wavetable::{
    condition_stereo_ir_buffers, max_nyquist_harmonic, morph_wavetable_frame,
    WavetableSpectralProcessor, MAX_WAVETABLE_FRAMES, PREVIEW_PERIODIC_WAVE_HARMONICS,
    WAVETABLE_FRAME_SAMPLES,
};

#[test]
fn test_riff_headers_all_rates_channels_and_pcm16_float32_formats() {
    let base_samples = [0.5_f32, -0.5, 0.25, -0.25, 0.125, -0.125, 0.0, 0.75];
    for rate in WavSampleRate::ALL {
        for channels in WavChannelLayout::ALL {
            for format in WavSampleFormat::ALL {
                let spec = WavEncodeSpec {
                    sample_rate_hz: rate.hz(),
                    channels,
                    format,
                    peak_ceiling: 1.0,
                    wavetable_chunks: None,
                };
                let wav = encode_wav_samples(&spec, &base_samples);
                let summary = inspect_wav_riff(&wav).expect("valid RIFF WAVE");
                let ch = channels.channels();
                let bps = format.bytes_per_sample();
                assert_eq!(summary.format_tag, format.format_tag());
                assert_eq!(summary.channels, ch);
                assert_eq!(summary.sample_rate_hz, rate.hz());
                assert_eq!(summary.bits_per_sample, format.bits_per_sample());
                assert_eq!(summary.block_align, ch * (format.bits_per_sample() / 8));
                assert_eq!(summary.byte_rate, rate.hz() * u32::from(ch) * bps);
                assert_eq!(summary.data_bytes, (base_samples.len() as u32) * bps);
                assert!(summary.extra_chunks.is_empty());

                if format == WavSampleFormat::Float32 {
                    let p = summary.data_offset;
                    let first = f32::from_le_bytes([wav[p], wav[p + 1], wav[p + 2], wav[p + 3]]);
                    assert!((first - 0.5).abs() < 1.0e-6);
                }
            }
        }
    }
}

#[test]
fn test_wavetable_wav_chunks_frame_counts_1_to_256_zero_dc_and_finite() {
    let mut proc = WavetableSpectralProcessor::new();
    for frame_count in [1_u16, 16, 64, MAX_WAVETABLE_FRAMES as u16] {
        let total_samples = WAVETABLE_FRAME_SAMPLES * usize::from(frame_count);
        let mut bank = vec![0.0_f32; total_samples];
        let mut frame = [0.0_f32; WAVETABLE_FRAME_SAMPLES];
        for f_idx in 0..usize::from(frame_count) {
            let morph = (f_idx as f32) / (usize::from(frame_count).max(1) as f32);
            for (i, slot) in frame.iter_mut().enumerate() {
                let phase = std::f32::consts::TAU * (i as f32) / (WAVETABLE_FRAME_SAMPLES as f32);
                // Intentional DC bias (+0.37) + high harmonic content to verify conditioning.
                *slot = 0.37 + phase.sin() + (0.4 + 0.5 * morph) * (7.0 * phase).sin();
            }
            proc.condition_frame_in_place(&mut frame, 64, 0.98);
            let mean = frame.iter().copied().sum::<f32>() / (WAVETABLE_FRAME_SAMPLES as f32);
            assert!(
                mean.abs() < 1.0e-6,
                "frame DC mean must be < 1e-6, got {mean}"
            );
            assert!(frame.iter().all(|s| s.is_finite() && s.abs() <= 0.980_01));
            let dst_start = f_idx * WAVETABLE_FRAME_SAMPLES;
            bank[dst_start..dst_start + WAVETABLE_FRAME_SAMPLES].copy_from_slice(&frame);
        }

        let wav = encode_wavetable_wav(
            WavSampleRate::Hz48000,
            WavSampleFormat::Float32,
            WAVETABLE_FRAME_SAMPLES as u16,
            frame_count,
            &bank,
        );
        let summary = inspect_wav_riff(&wav).expect("valid wavetable RIFF");
        assert_eq!(summary.format_tag, 3);
        assert_eq!(summary.channels, 1);
        assert_eq!(summary.sample_rate_hz, 48_000);
        assert_eq!(summary.data_bytes, (total_samples * 4) as u32);
        assert_eq!(summary.extra_chunks.len(), 2);

        let clm = &summary.extra_chunks[0];
        assert_eq!(&clm.id, b"clm ");
        let clm_str = std::str::from_utf8(&clm.payload).expect("utf8 clm");
        assert!(clm_str.starts_with("<!>2048 "));

        let uhwt = &summary.extra_chunks[1];
        assert_eq!(&uhwt.id, b"uhWT");
        assert_eq!(uhwt.size, 12);
        let parsed_frames = u32::from_le_bytes(uhwt.payload[4..8].try_into().unwrap());
        let parsed_samples = u32::from_le_bytes(uhwt.payload[8..12].try_into().unwrap());
        assert_eq!(parsed_frames, u32::from(frame_count));
        assert_eq!(parsed_samples, WAVETABLE_FRAME_SAMPLES as u32);
    }
}

#[test]
fn test_nyquist_brickwall_bandlimiting_and_zero_alloc_spectral_loop() {
    let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut proc = WavetableSpectralProcessor::new();
    let max_h = max_nyquist_harmonic(44_100, 1_000.0); // ~21 harmonics below Nyquist
    assert!((20..=22).contains(&max_h));

    let mut frame = [0.0_f32; WAVETABLE_FRAME_SAMPLES];
    let mut real = [0.0_f32; PREVIEW_PERIODIC_WAVE_HARMONICS + 1];
    let mut imag = [0.0_f32; PREVIEW_PERIODIC_WAVE_HARMONICS + 1];
    let mut mags = [0.0_f32; PREVIEW_PERIODIC_WAVE_HARMONICS];
    let bank = vec![0.25_f32; WAVETABLE_FRAME_SAMPLES * 4];
    let mut morphed = [0.0_f32; WAVETABLE_FRAME_SAMPLES];

    let (peak, _metrics) = VerifiedEngine::run_verified(|| {
        for (i, slot) in frame.iter_mut().enumerate() {
            let phase = std::f32::consts::TAU * (i as f32) / (WAVETABLE_FRAME_SAMPLES as f32);
            // Fundamental (k=1) + ultrasonic harmonic (k=48 > max_h) + DC offset.
            *slot = 0.5 + phase.sin() + 0.9 * (48.0 * phase).sin();
        }
        proc.condition_frame_in_place(&mut frame, max_h, 0.95);
        proc.extract_periodic_wave_and_spectrum(&frame, &mut real, &mut imag, &mut mags);
        morph_wavetable_frame(&bank, 4, 0.35, &mut morphed);
        frame.iter().copied().fold(0.0_f32, |a, b| a.max(b.abs()))
    });

    assert!((peak - 0.95).abs() < 1.0e-4);
    // Verify k=48 ultrasonic partial was brickwall-removed while k=1 survived.
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(WAVETABLE_FRAME_SAMPLES);
    let mut spec: Vec<Complex<f32>> = frame.iter().map(|&x| Complex::new(x, 0.0)).collect();
    fft.process(&mut spec);
    assert!(spec[0].norm() < 1.0e-3, "DC bin must be zeroed");
    assert!(spec[1].norm() > 800.0, "fundamental must remain strong");
    assert!(
        spec[48].norm() < 1.0e-3,
        "above-Nyquist k=48 must be suppressed"
    );
    assert!((imag[1].abs() - 0.95).abs() < 0.02);
}

#[test]
fn test_stereo_ir_conditioning_and_studio_preview_gesture_gate() {
    let mut left = vec![0.35_f32; 4096];
    let mut right = vec![-0.25_f32; 4096];
    for i in 0..4096 {
        let t = (i as f32) / 48_000.0;
        let env = (-8.0 * t).exp();
        left[i] += 0.8 * (std::f32::consts::TAU * 440.0 * t).sin() * env;
        right[i] += 0.4 * (std::f32::consts::TAU * 660.0 * t).sin() * env;
    }
    condition_stereo_ir_buffers(&mut left, &mut right, 0.707);
    let mean_l = left.iter().copied().sum::<f32>() / (left.len() as f32);
    let mean_r = right.iter().copied().sum::<f32>() / (right.len() as f32);
    assert!(mean_l.abs() < 1.0e-6 && mean_r.abs() < 1.0e-6);
    let wav = encode_stereo_ir_wav(
        WavSampleRate::Hz48000,
        WavSampleFormat::Pcm16,
        &left,
        &right,
    );
    let summary = inspect_wav_riff(&wav).expect("valid stereo IR WAV");
    assert_eq!(summary.channels, 2);
    assert_eq!(summary.format_tag, 1);

    let mut ctrl = StudioPreviewController::new();
    assert!(!ctrl.has_audio_context());
    assert!(!ctrl.snapshot().unlocked);
    // Updates before user gesture must remain strictly locked and silent.
    ctrl.update_wavetable_audition(&[0.0, 0.0], &[0.0, 1.0], 220.0, 0.8)
        .expect("locked update is no-op");
    assert!(!ctrl.has_audio_context());

    let active = ctrl
        .toggle_wavetable_audition_from_user_gesture(&[0.0, 0.0], &[0.0, 1.0], 220.0, 0.8)
        .expect("gesture unlocks audition");
    assert!(active && ctrl.has_audio_context());
    ctrl.suspend_all(MuteReason::Paused);
    assert!(!ctrl.snapshot().wavetable_auditioning);

    trigger_studio_ir_strike_from_gesture(48_000, &left[..256], &right[..256], 0.8)
        .expect("gesture strike");
    assert_eq!(snapshot_studio_preview().last_strike_frames, 256);
    super::notify_playback_paused();
    assert!(!snapshot_studio_preview().unlocked);
}

#[test]
fn test_framework_wav_16bit_riff_parity_and_webgl_loss_silences_studio_preview() {
    use super::mappings::attractor_frame;
    use super::{
        silence_all_audio_on_webgl_loss, synthesize_wav_16bit, AcousticSettings,
        AudioEngineController, AutoSilenceReason, DEFAULT_WAV_SAMPLE_RATE, WAV_HEADER_BYTES,
    };

    let frames = [attractor_frame([8.0, -5.0, 26.0], [12.0, 18.0, -9.0], false); 4];
    let wav = synthesize_wav_16bit(
        &frames,
        AcousticSettings {
            gain: 0.75,
            pitch_scale: 1.25,
        },
        DEFAULT_WAV_SAMPLE_RATE,
        0.2,
        42,
    );
    let summary = inspect_wav_riff(&wav).expect("synthesize_wav_16bit must emit valid RIFF WAVE");
    assert_eq!(summary.format_tag, 1);
    assert_eq!(summary.channels, 2);
    assert_eq!(summary.sample_rate_hz, DEFAULT_WAV_SAMPLE_RATE);
    assert_eq!(summary.bits_per_sample, 16);
    assert_eq!(summary.data_offset, WAV_HEADER_BYTES);
    assert!(summary.extra_chunks.is_empty());

    let mut scoped = AudioEngineController::new();
    scoped.unmute_from_user_gesture("Wavetable Studio");
    super::studio_preview::toggle_studio_wavetable_audition_from_gesture(
        &[0.0, 0.0, 0.1],
        &[0.0, 0.9, 0.2],
        110.0,
        0.7,
    )
    .expect("gesture unlocks studio preview");
    assert!(snapshot_studio_preview().wavetable_auditioning);

    silence_all_audio_on_webgl_loss();
    scoped.update_frame(Some(frames[0]));
    assert!(scoped.is_muted());
    assert_eq!(
        scoped.last_silence_reason(),
        AutoSilenceReason::WebGlContextLost
    );
    let snap = snapshot_studio_preview();
    assert!(!snap.unlocked && !snap.wavetable_auditioning);
    assert_eq!(snap.last_mute_reason, MuteReason::WebGlContextLost);
}
