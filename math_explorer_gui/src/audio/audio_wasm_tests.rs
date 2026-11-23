use wasm_bindgen_test::wasm_bindgen_test;

use super::{
    export_current_tool_wav, notify_playback_paused, notify_solver_diverged, on_webgl_context_lost,
    snapshot_ui_state, submit_tool_frame, toggle_mute_from_user_gesture, WaveformPreset,
};
use crate::audio::mappings::{
    attractor_frame, elliptic_torsion_frame, fractran_register_frame, ulam_spiral_frame,
    EllipticTorsionSample, UlamPolySample,
};

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

    let ulam = ulam_spiral_frame(
        &[UlamPolySample {
            value: 41,
            is_prime: true,
            prime_gap: 1,
        }],
        120.0,
        true,
    );
    submit_tool_frame("prime_spiral", ulam, false);
    export_current_tool_wav("prime_spiral");

    let fractran = fractran_register_frame(&[2, 3, 5], &[1, 3, 2], None, 32.0, 16.0, false);
    submit_tool_frame("fractran_machine", fractran, false);
    export_current_tool_wav("fractran_machine");

    let ec = elliptic_torsion_frame(EllipticTorsionSample {
        modulus_p: 29,
        is_singular: false,
        kp_coords: Some((1, 5)),
        p_order: 7,
        scalar_k: 2,
        frobenius_trace: -6,
        torus_wraps: 1,
    });
    submit_tool_frame("elliptic_curve_playground", ec, false);
    export_current_tool_wav("elliptic_curve_playground");

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
