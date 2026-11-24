use wasm_bindgen_test::wasm_bindgen_test;
use web_sys::AudioContextState;

use super::{create_periodic_wave, WebAudioNodeBackend};
use crate::audio::backend::{MuteReason, SonificationBackend};
use crate::audio::fourier::compute_preset_harmonics;
use crate::audio::frame::{WaveformPreset, LIMITER_CEILING_DBFS};
use crate::audio::mappings::{hodgkin_huxley_frame, prime_chord_frame};

#[wasm_bindgen_test]
fn wasm_web_audio_backend_starts_muted_without_creating_audio_context() {
    let backend = WebAudioNodeBackend::new();
    assert!(backend.is_muted());
    assert!(!backend.has_audio_context());
    assert_eq!(backend.context_state(), None);
    assert_eq!(backend.last_mute_reason(), MuteReason::InitialDefault);
    assert_eq!(backend.meter_reading().peak_dbfs, -60.0);
}

#[wasm_bindgen_test]
fn wasm_web_audio_backend_unlocks_applies_fourier_waves_and_suspends_on_faults() {
    let mut backend = WebAudioNodeBackend::new();
    backend
        .set_muted_from_user_gesture(false)
        .expect("explicit user gesture unlock should build Web Audio node graph");
    assert!(!backend.is_muted());
    assert!(backend.has_audio_context());
    assert_eq!(backend.analyser_bin_count(), Some(128));

    let hh_frame = hodgkin_huxley_frame(28.0, -5.0, 0.85, 0.45, 0.35, 12.0);
    backend.apply_frame(&hh_frame);
    let meter = backend.meter_reading();
    assert!(meter.peak_dbfs <= LIMITER_CEILING_DBFS);
    assert!(meter.dominant_hz >= 20.0 && meter.dominant_hz <= 8000.0);

    let chord = prime_chord_frame(&[(2, 2), (3, 2), (5, 1)], WaveformPreset::JustHarmonic);
    backend.apply_frame(&chord);
    assert_eq!(backend.meter_reading().active_voices, 3);

    backend.mute_and_suspend(MuteReason::WebGlContextLost);
    assert!(backend.is_muted());
    assert_eq!(backend.last_mute_reason(), MuteReason::WebGlContextLost);
    assert_eq!(backend.meter_reading().peak_dbfs, -60.0);
    let state = backend.context_state();
    assert!(
        matches!(
            state,
            Some(AudioContextState::Suspended | AudioContextState::Running)
        ),
        "unexpected AudioContextState after suspend: {state:?}"
    );
}

#[wasm_bindgen_test]
fn wasm_periodic_wave_builds_from_fourier_coefficients() {
    let ctx = web_sys::AudioContext::new().expect("AudioContext should instantiate in test");
    let weights = compute_preset_harmonics(WaveformPreset::FourierWarm);
    let wave = create_periodic_wave(&ctx, &weights);
    assert!(wave.is_ok());
    let _ = ctx.close();
}
