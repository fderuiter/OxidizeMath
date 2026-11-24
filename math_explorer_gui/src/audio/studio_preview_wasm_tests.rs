#![cfg_attr(any(), verified(opt_out = "tests"))]

use wasm_bindgen_test::wasm_bindgen_test;

use super::StudioPreviewController;
use crate::audio::backend::MuteReason;

#[wasm_bindgen_test]
fn wasm_studio_preview_gates_on_gesture_and_plays_ir_and_periodic_wave() {
    let mut preview = StudioPreviewController::new();
    assert!(!preview.has_audio_context());
    preview
        .update_wavetable_audition(&[0.0, 0.0, 0.0], &[0.0, 1.0, 0.25], 110.0, 0.6)
        .expect("locked update must not instantiate AudioContext");
    assert!(!preview.has_audio_context());

    let left = [0.0_f32, 0.35, -0.25, 0.12, -0.05, 0.02, 0.0, 0.0];
    let right = [0.0_f32, 0.28, -0.30, 0.15, -0.08, 0.03, 0.0, 0.0];
    preview
        .trigger_ir_strike_from_user_gesture(48_000, &left, &right, 0.8)
        .expect("AudioBufferSourceNode strike should succeed on user gesture");
    assert!(preview.has_audio_context());
    assert_eq!(preview.snapshot().last_strike_frames, 8);

    let playing = preview
        .toggle_wavetable_audition_from_user_gesture(
            &[0.0, 0.0, 0.15, 0.0],
            &[0.0, 0.85, 0.25, 0.1],
            220.0,
            0.7,
        )
        .expect("PeriodicWave audition should start on user gesture");
    assert!(playing);
    preview
        .update_wavetable_audition(&[0.0, 0.2, 0.1], &[0.0, 0.7, 0.35], 330.0, 0.7)
        .expect("morph update while unlocked should succeed");
    preview.suspend_all(MuteReason::WindowBlurred);
    assert!(!preview.snapshot().wavetable_auditioning);
}
