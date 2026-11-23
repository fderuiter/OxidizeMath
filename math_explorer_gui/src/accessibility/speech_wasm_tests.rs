#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use wasm_bindgen_test::wasm_bindgen_test;

#[wasm_bindgen_test]
fn wasm_speech_stays_silent_until_enabled_and_supports_volume_and_cancel() {
    reset_test_speech_state();
    assert!(!is_speech_enabled());
    assert!((speech_volume() - DEFAULT_SPEECH_VOLUME).abs() < f32::EPSILON);

    speak_if_enabled("Should remain silent when disabled");
    let (spoken_disabled, _) = test_speech_log();
    assert!(spoken_disabled.is_empty());

    sync_speech_settings(true, 0.45);
    assert!(is_speech_enabled());
    assert!((speech_volume() - 0.45).abs() < 1e-6);

    speak_if_enabled("Opt-in narration active");
    let (spoken_enabled, _) = test_speech_log();
    assert_eq!(spoken_enabled.len(), 1);
    assert_eq!(spoken_enabled[0].0, "Opt-in narration active");
    assert!((spoken_enabled[0].1 - 0.45).abs() < 1e-6);

    stop_speech();
    let (_, cancels_after_stop) = test_speech_log();
    assert_eq!(cancels_after_stop, 1);
    assert!(take_frame_speech_suppression());
    assert!(!take_frame_speech_suppression());

    sync_speech_settings(false, 0.45);
    assert!(!is_speech_enabled());
    let (_, cancels_after_disable) = test_speech_log();
    assert_eq!(cancels_after_disable, 2);
}
