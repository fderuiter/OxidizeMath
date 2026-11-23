use wasm_bindgen_test::wasm_bindgen_test;

use super::{
    export_current_tool_wav, notify_playback_paused, notify_solver_diverged, on_webgl_context_lost,
    snapshot_ui_state, submit_tool_frame, toggle_mute_from_user_gesture, WaveformPreset,
};
use crate::audio::mappings::attractor_frame;

#[wasm_bindgen_test]
fn wasm_audio_controller_lifecycle_and_wav_export() {
    let initial = snapshot_ui_state();
    assert!(initial.muted);

    let frame = attractor_frame([8.5, -6.0, 27.0], [15.0, 22.0, -10.0], false);
    submit_tool_frame("attractors", frame, false);
    toggle_mute_from_user_gesture();
    let unlocked = snapshot_ui_state();
    assert!(!unlocked.muted);
    assert!(unlocked.meter.dominant_hz >= 20.0);

    export_current_tool_wav("attractors");

    notify_playback_paused();
    assert!(snapshot_ui_state().muted);

    toggle_mute_from_user_gesture();
    assert!(!snapshot_ui_state().muted);
    on_webgl_context_lost();
    assert!(snapshot_ui_state().muted);

    toggle_mute_from_user_gesture();
    notify_solver_diverged();
    assert!(snapshot_ui_state().muted);
    super::set_waveform_override(Some(WaveformPreset::Sine));
    assert_eq!(
        snapshot_ui_state().waveform_override,
        Some(WaveformPreset::Sine)
    );
}
