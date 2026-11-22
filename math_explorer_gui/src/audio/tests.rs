#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;

#[test]
fn test_sonification_frame_sanitizes_non_finite_and_clamps_bounds() {
    let nan_frame = SonificationFrame {
        active: true,
        carrier_hz: f32::NAN,
        amplitude: 0.8,
        ..SonificationFrame::silent()
    };
    let clean_nan = nan_frame.sanitized();
    assert!(!clean_nan.active);
    assert_eq!(clean_nan.amplitude, 0.0);

    let extreme = SonificationFrame {
        active: true,
        carrier_hz: 99_999.0,
        amplitude: 5.0,
        pan: -9.0,
        fm_modulator_hz: -10.0,
        fm_index_hz: 50_000.0,
        filter_cutoff_hz: 5.0,
        filter_q: 99.0,
        filter_bandpass: true,
        harmonics_real: [1.0, 4.0, -4.0, f32::INFINITY, 0.5, 0.0, 0.0, 0.0],
        harmonics_imag: [1.0, -2.0, 2.0, f32::NAN, 0.25, 0.0, 0.0, 0.0],
        use_periodic_wave: true,
        click_impulse: 3.0,
        bell_decay_rate: 999.0,
        poly_voices_hz: [5.0, 440.0, 99_999.0, f32::NAN],
        poly_voices_amp: [0.9, 2.0, 0.5, 0.8],
    }
    .sanitized();

    assert_eq!(extreme.carrier_hz, MAX_CARRIER_HZ);
    assert_eq!(extreme.amplitude, 1.0);
    assert_eq!(extreme.pan, -1.0);
    assert_eq!(extreme.fm_modulator_hz, 0.0);
    assert_eq!(extreme.fm_index_hz, 4_000.0);
    assert_eq!(extreme.filter_cutoff_hz, 40.0);
    assert_eq!(extreme.filter_q, 18.0);
    assert_eq!(extreme.harmonics_real[0], 0.0);
    assert_eq!(extreme.harmonics_real[1], 1.0);
    assert_eq!(extreme.harmonics_real[3], 0.0);
    assert_eq!(extreme.harmonics_imag[3], 0.0);
    assert_eq!(extreme.click_impulse, 1.0);
    assert_eq!(extreme.poly_voices_amp[0], 0.0);
    assert_eq!(extreme.poly_voices_amp[1], 1.0);
    assert_eq!(extreme.poly_voices_amp[3], 0.0);
}

#[test]
fn test_soft_clip_tanh_limiter_and_20ms_silence_ramp() {
    assert_eq!(soft_clip_sample(f32::NAN), 0.0);
    assert_eq!(soft_clip_sample(f32::INFINITY), 0.0);
    assert!(soft_clip_sample(10.0) <= 1.0);
    assert!(soft_clip_sample(-10.0) >= -1.0);
    assert!((soft_clip_sample(0.5) - 0.5_f32.tanh()).abs() < 1e-6);

    let mut env = GainRampEnvelope::new_muted();
    assert_eq!(env.advance(), 0.0);
    env.ramp_to(0.8);
    assert!((env.sample_after_elapsed(std::time::Duration::from_millis(20)) - 0.8).abs() < 1e-5);

    std::thread::sleep(std::time::Duration::from_millis(18));
    let _ = env.advance();
    env.ramp_to_zero();
    assert_eq!(
        env.sample_after_elapsed(std::time::Duration::from_millis(20)),
        0.0,
        "GainRampEnvelope must reach exact 0.0 within 20 ms"
    );
}

#[test]
fn test_pcm_synth_chunk_zero_allocation_and_riff_wav_energy() {
    let frame = SonificationFrame {
        active: true,
        carrier_hz: 330.0,
        amplitude: 0.75,
        pan: -0.35,
        fm_modulator_hz: 165.0,
        fm_index_hz: 80.0,
        filter_cutoff_hz: 2_800.0,
        filter_q: 2.0,
        click_impulse: 0.6,
        ..SonificationFrame::silent()
    };
    let settings = AcousticSettings {
        gain: 0.7,
        pitch_scale: 1.0,
    };
    let mut synth = PcmSynthState::new(42);
    let mut chunk = [0_u8; 4 * 256];
    {
        let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        verified_engine::engine::VerifiedEngine::run_verified(|| {
            let clean = frame.sanitized();
            synth.render_chunk_into(clean, settings, DEFAULT_WAV_SAMPLE_RATE, &mut chunk);
        });
    }

    let wav = synthesize_wav_16bit(&[frame; 16], settings, DEFAULT_WAV_SAMPLE_RATE, 0.25, 42);
    assert!(wav.len() > WAV_HEADER_BYTES);
    assert_eq!(&wav[0..4], b"RIFF");
    assert_eq!(&wav[8..12], b"WAVE");
    assert_eq!(&wav[12..16], b"fmt ");
    assert_eq!(&wav[36..40], b"data");
    assert_eq!(u16::from_le_bytes([wav[20], wav[21]]), 1);
    assert_eq!(u16::from_le_bytes([wav[22], wav[23]]), 2);
    assert_eq!(u16::from_le_bytes([wav[34], wav[35]]), 16);

    let mut energy = 0.0_f64;
    for sample_bytes in wav[WAV_HEADER_BYTES..].as_chunks::<2>().0 {
        let s = f64::from(i16::from_le_bytes(*sample_bytes)) / 32_768.0;
        energy += s * s;
    }
    assert!(
        energy > 1.0,
        "Synthesized 16-bit WAV must have non-zero acoustic energy"
    );
}

#[test]
fn test_controller_default_muted_divergence_and_webgl_loss_guards() {
    crate::accessibility::clear_test_announcements();
    let mut ctrl = AudioEngineController::new();
    assert!(ctrl.is_muted());
    assert!(!ctrl.has_user_gesture());
    assert!(!ctrl.has_audio_context());

    ctrl.unmute_from_user_gesture("Test Tool");
    assert!(!ctrl.is_muted());
    assert!(ctrl.has_user_gesture());

    ctrl.auto_mute_on_divergence("Test Tool");
    assert!(ctrl.is_muted());
    assert_eq!(
        ctrl.last_silence_reason(),
        AutoSilenceReason::SolverDiverged
    );
    assert_eq!(
        ctrl.current_gain_after_elapsed(std::time::Duration::from_millis(20)),
        0.0
    );
    assert!(crate::accessibility::all_announcements()
        .iter()
        .any(|(msg, prio)| msg.contains("solver divergence") && prio == "assertive"));

    ctrl.unmute_from_user_gesture("Test Tool");
    assert!(!ctrl.is_muted());
    silence_all_audio_on_webgl_loss();
    ctrl.update_frame(Some(SonificationFrame {
        active: true,
        amplitude: 0.7,
        ..SonificationFrame::silent()
    }));
    assert!(ctrl.is_muted());
    assert_eq!(
        ctrl.last_silence_reason(),
        AutoSilenceReason::WebGlContextLost
    );
}
