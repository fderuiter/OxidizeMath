#![cfg(feature = "physics")]
use wasm_bindgen_test::wasm_bindgen_test;

use super::{
    export_current_tool_wav, notify_playback_paused, notify_solver_diverged, on_webgl_context_lost,
    silence_all_audio_on_webgl_loss, snapshot_ui_state, submit_tool_frame,
    toggle_mute_from_user_gesture, AudioEngineController, AutoSilenceReason, WaveformPreset,
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
    super::set_waveform_override(None);
}

#[wasm_bindgen_test]
fn wasm_scoped_controller_divergence_and_webgl_epoch_guard() {
    let mut ctrl = AudioEngineController::new();
    assert!(ctrl.is_muted());
    ctrl.unmute_from_user_gesture("Attractor Plotter");
    assert!(!ctrl.is_muted());

    silence_all_audio_on_webgl_loss();
    let frame = attractor_frame([8.5, -6.0, 27.0], [15.0, 22.0, -10.0], false);
    ctrl.update_frame(Some(frame));
    assert!(ctrl.is_muted());
    assert_eq!(
        ctrl.last_silence_reason(),
        AutoSilenceReason::WebGlContextLost
    );
}

#[cfg(all(feature = "physics", feature = "engineering"))]
#[wasm_bindgen_test]
fn wasm_mri_kspace_and_60ghz_radar_sonification_lifecycle() {
    use math_explorer::physics::medical::radar_gating::BeamState;
    use nalgebra::Vector3;

    use crate::audio::mappings::{
        medical_radar_frame, mri_bloch_frame, mri_kspace_frame, MriKSpaceAudioInput,
        RadarDopplerAudioInput,
    };

    if !snapshot_ui_state().muted {
        toggle_mute_from_user_gesture();
    }

    let bloch = mri_bloch_frame(
        Vector3::new(0.8, 0.2, 0.3),
        1.0,
        Vector3::new(1.0e-5, 0.0, 1.5),
        400.0,
    );
    submit_tool_frame("mri_bloch", bloch, false);
    toggle_mute_from_user_gesture();
    assert!(!snapshot_ui_state().muted);
    assert!(snapshot_ui_state().meter.dominant_hz > 700.0);

    let kspace = mri_kspace_frame(&MriKSpaceAudioInput {
        spins: [
            Vector3::new(0.6, 0.2, 0.1),
            Vector3::new(0.5, 0.3, 0.1),
            Vector3::new(0.4, 0.4, 0.1),
        ],
        rho: 0.9,
        t2: 0.08,
        delta_b0: 2.2e-7,
        turing_uv: [1.1, 0.8],
        current_time: 0.04,
        echo_tau: Some(0.02),
        k_coord_cycles_per_m: [80.0, -60.0],
        selected_grad_phase_rad: 0.9,
        rf_spike_triggered: true,
    });
    submit_tool_frame("mri_kspace", kspace, false);
    assert!(snapshot_ui_state().meter.dominant_hz >= 20.0);

    let radar = medical_radar_frame(&RadarDopplerAudioInput {
        velocity_m_s: 0.015,
        amplitude_m: -0.002,
        bandwidth_ghz: 4.0,
        coarse_peak_count: 1,
        music_peak_count: 2,
        primary_beam_gain: 0.95,
        secondary_beam_gain: 0.85,
        cough_timer_s: 0.25,
        beam_state: BeamState::BeamActive,
    });
    submit_tool_frame("radar_gating", radar, false);
    assert!(snapshot_ui_state().meter.dominant_hz >= 20.0);

    notify_solver_diverged();
    assert!(snapshot_ui_state().muted);
}
