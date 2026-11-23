#![cfg_attr(any(), verified(opt_out = "tests"))]

use std::sync::{Arc, RwLock};

use eframe::egui;
use math_explorer::physics::medical::radar_gating::{BeamState, C};
use nalgebra::Vector3;
use num_complex::Complex;
use rustfft::FftPlanner;
use verified_engine::engine::VerifiedEngine;

use super::backend::{MuteReason, SonificationBackend};
use super::frame::{LIMITER_CEILING_DBFS, LIMITER_CEILING_LINEAR};
use super::mappings::{
    doppler_shift_60ghz_hz, medical_radar_frame, mri_bloch_frame, mri_kspace_frame,
    mri_pulse_playhead_frame, MriKSpaceAudioInput, RadarDopplerAudioInput, RADAR_CARRIER_60GHZ_HZ,
};
use super::synth::OfflinePcmSynthesizer;
use crate::async_sim::{SimCommand, SimulationRunner};
use crate::framework::InteractiveTool;
use crate::tabs::medical::radar_gating::{RadarGatingTool, STARTER_PRESET_JSON};
use crate::tabs::mri::kspace_runner::MriKSpaceRunner;
use crate::tabs::mri::kspace_types::{
    MriKSpaceSharedParams, MriKSpaceToolState, PhantomPreset, StudioBrushMode,
};
use crate::tabs::mri::pulse::{
    MriPulseMode, RfPulseConfig, CHALLENGE_BLOCH_HAHN_ECHO, CHALLENGE_BLOCH_PI_INVERSION,
    MRI_PULSE_SAMPLES,
};
use crate::tabs::mri::{MriKSpaceTool, MriTool};

fn mono_fft_mags(interleaved: &[f32], fft_len: usize) -> Vec<f32> {
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(fft_len);
    let mut buf: Vec<Complex<f32>> = (0..fft_len)
        .map(|i| {
            let l = interleaved.get(i * 2).copied().unwrap_or(0.0);
            let r = interleaved.get(i * 2 + 1).copied().unwrap_or(0.0);
            let hann = 0.5 * (1.0 - (std::f32::consts::TAU * (i as f32) / (fft_len as f32)).cos());
            Complex::new(0.5 * (l + r) * hann, 0.0)
        })
        .collect();
    fft.process(&mut buf);
    buf[..fft_len / 2].iter().map(|c| c.norm()).collect()
}

fn bin_peak(mags: &[f32], sr: u32, fft_len: usize, hz: f32) -> f32 {
    let idx = ((hz * (fft_len as f32)) / (sr as f32)).round() as usize;
    let lo = idx.saturating_sub(2);
    let hi = (idx + 2).min(mags.len().saturating_sub(1));
    mags[lo..=hi].iter().copied().fold(0.0_f32, f32::max)
}

fn rms_stereo(interleaved: &[f32]) -> f32 {
    if interleaved.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = interleaved.iter().map(|&s| s * s).sum();
    (sum_sq / (interleaved.len() as f32)).sqrt()
}

fn ensure_controller_muted() {
    if !super::snapshot_ui_state().muted {
        super::toggle_mute_from_user_gesture();
    }
    super::set_waveform_override(None);
}

fn sample_zero_alloc_step_frame(
    step: usize,
    pulse_cfg: &RfPulseConfig,
) -> super::frame::SonificationFrame {
    let ph = (step as f64) * 0.04;
    match step % 4 {
        0 => mri_bloch_frame(
            Vector3::new(ph.cos(), ph.sin(), -0.3),
            1.0,
            Vector3::new(1.0e-5, 0.0, 1.5),
            180.0 * ph.sin(),
        ),
        1 => mri_pulse_playhead_frame(
            MriPulseMode::HahnSpinEcho,
            0.020 + 0.020 * ph.sin(),
            Vector3::new(0.6 * ph.cos(), 0.2 * ph.sin(), -0.4),
            0.5 * (1.0 + ph.cos()),
            1.0,
            1.5,
            pulse_cfg,
        ),
        2 => mri_kspace_frame(&MriKSpaceAudioInput {
            spins: [
                Vector3::new(0.5, 0.2, 0.1),
                Vector3::new(0.4, 0.3, 0.1),
                Vector3::new(0.3, 0.4, 0.1),
            ],
            rho: 0.85,
            t2: 0.09,
            delta_b0: 2.2e-7,
            turing_uv: [1.05, 0.82],
            current_time: 0.04,
            echo_tau: Some(0.02),
            k_coord_cycles_per_m: [120.0 * ph.cos(), 95.0 * ph.sin()],
            selected_grad_phase_rad: ph,
            rf_spike_triggered: step.is_multiple_of(64),
        }),
        _ => medical_radar_frame(&RadarDopplerAudioInput {
            velocity_m_s: 0.018 * ph.sin(),
            amplitude_m: 0.004 * ph.cos(),
            bandwidth_ghz: 4.0,
            coarse_peak_count: 1,
            music_peak_count: 2,
            primary_beam_gain: 0.92,
            secondary_beam_gain: 0.81,
            cough_timer_s: if step.is_multiple_of(100) { 0.30 } else { 0.0 },
            beam_state: if step.is_multiple_of(8) {
                BeamState::BeamHold
            } else {
                BeamState::BeamActive
            },
        }),
    }
}

#[test]
fn test_zero_alloc_1000_steps_mri_kspace_and_radar_extractors_and_pcm() {
    let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut synth = OfflinePcmSynthesizer::new(44_100, 64);
    let _ = synth.set_muted_from_user_gesture(false);
    let pulse_cfg = RfPulseConfig {
        mode: MriPulseMode::HahnSpinEcho,
        b1_amp: 2.9e-5,
        duration_tau: 0.002,
        phase_deg: 0.0,
        detuning_hz: 120.0,
        echo_time_te: 0.040,
    };

    let (peak, _) = VerifiedEngine::run_verified(|| {
        let mut max_abs = 0.0_f32;
        for step in 0..1_000_usize {
            let frame = sample_zero_alloc_step_frame(step, &pulse_cfg);
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
fn test_mri_detuning_meter_hz_and_hahn_echo_rustfft_peak_preserving_scores() {
    ensure_controller_muted();
    let mut tool = MriTool::default();
    let f0 = tool.current_sonification_frame();
    super::submit_tool_frame("mri_bloch", f0, false);
    super::toggle_mute_from_user_gesture();
    let hz_base = super::snapshot_ui_state().meter.dominant_hz;

    tool.pulse_config.detuning_hz = 800.0;
    let f_detuned = tool.current_sonification_frame();
    super::submit_tool_frame("mri_bloch", f_detuned, false);
    let hz_detuned = super::snapshot_ui_state().meter.dominant_hz;
    assert!(
        (hz_detuned - (hz_base + 200.0)).abs() < 1.0,
        "detuning_hz=+800 must shift MeterReading dominant_hz by +200 Hz ({hz_base} -> {hz_detuned})"
    );
    ensure_controller_muted();

    // Verify HahnSpinEcho challenge score invariance and rustfft echo peak at t = TE
    tool.pulse_config.detuning_hz = 0.0;
    tool.pulse_config.duration_tau = 0.002;
    tool.pulse_config.b1_amp =
        std::f64::consts::PI / (math_explorer::physics::mri::proton::GYROMAGNETIC_RATIO * 0.002);
    tool.apply_challenge(Some(CHALLENGE_BLOCH_HAHN_ECHO))
        .expect("Hahn echo challenge");
    let score_before = tool.evaluate_challenge().expect("echo eval");
    assert!(score_before.passed);

    let sr = 16_384_u32;
    let fft_len = 2_048_usize;
    let mut synth = OfflinePcmSynthesizer::new(sr, fft_len + 1024);
    let _ = synth.set_muted_from_user_gesture(false);

    tool.pulse_workspace.playhead_idx = 55; // Dephased trough before 180 deg refocusing
    let trough_frame = tool.current_sonification_frame();
    synth.apply_frame(&trough_frame);
    let _ = synth.render_frames(1024);
    let trough_mags = mono_fft_mags(synth.render_frames(fft_len), fft_len);
    let trough_peak = bin_peak(&trough_mags, sr, fft_len, trough_frame.voices[0].freq_hz);

    tool.pulse_workspace.playhead_idx = MRI_PULSE_SAMPLES - 1; // t = TE spin echo peak
    let echo_frame = tool.current_sonification_frame();
    synth.apply_frame(&echo_frame);
    let _ = synth.render_frames(1024);
    let echo_mags = mono_fft_mags(synth.render_frames(fft_len), fft_len);
    let echo_peak = bin_peak(&echo_mags, sr, fft_len, echo_frame.voices[0].freq_hz);
    assert!(
        echo_peak > trough_peak * 1.45,
        "Hahn spin echo peak at t=TE ({echo_peak:.2}) must exceed dephased trough ({trough_peak:.2})"
    );

    let score_after = tool.evaluate_challenge().expect("echo eval unchanged");
    assert_eq!(score_before.score.to_bits(), score_after.score.to_bits());

    tool.apply_challenge(Some(CHALLENGE_BLOCH_PI_INVERSION))
        .expect("pi inversion challenge");
    let inv_before = tool.evaluate_challenge().unwrap().score;
    tool.pulse_workspace.playhead_idx = 25;
    let _ = tool.current_sonification_frame();
    assert_eq!(
        inv_before.to_bits(),
        tool.evaluate_challenge().unwrap().score.to_bits()
    );
}

#[test]
fn test_kspace_turing_delta_b0_detuning_echo_recovery_chirp_and_rf_spike() {
    let params = Arc::new(RwLock::new(MriKSpaceSharedParams {
        config: MriKSpaceToolState {
            phantom: PhantomPreset::TuringSpots,
            te: 0.006,
            selected_voxel: [32, 32],
            brush_mode: StudioBrushMode::SpatialDeltaB0Pocket,
            ..Default::default()
        },
        high_contrast: false,
    }));
    let mut runner = MriKSpaceRunner::new(Arc::clone(&params));
    let base_frame = runner.get_snapshot().sonification;
    assert!(base_frame.custom_harmonics.is_some());
    let base_spread = (base_frame.voices[2].freq_hz - base_frame.voices[0].freq_hz).abs();
    assert!(base_spread < 1.0);

    // Paint SpatialDeltaB0Pocket on Turing phantom and evolve under T2* dephasing
    runner.process_command(SimCommand::ApplyBrush {
        cx: 32,
        cy: 32,
        r: 4,
        is_obstacle: true,
    });
    for _ in 0..17 {
        runner.step();
    }
    let dephased = runner.get_snapshot().sonification;
    let detuned_spread = (dephased.voices[2].freq_hz - dephased.voices[0].freq_hz).abs();
    assert!(
        detuned_spread > 20.0 && dephased.master_gain < base_frame.master_gain,
        "SpatialDeltaB0Pocket must detune chord ({detuned_spread} Hz) and drop T2* gain"
    );

    // Fire 180 deg refocusing pulse at tau = 40 ms and step 20 ticks to t = 2*tau = 80 ms
    runner.process_command(SimCommand::Custom("refocus_180".to_string()));
    for _ in 0..20 {
        runner.step();
    }
    let echo = runner.get_snapshot().sonification;
    assert!(
        echo.master_gain > dephased.master_gain * 1.20
            && echo.voices[1].gain > dephased.voices[1].gain * 1.20,
        "apply_refocus_180_pulse must restore peak echo gain at t = 2*tau"
    );

    // Apply readout/phase-encoding gradients and verify FM chirp activation
    {
        let mut guard = params.write().unwrap();
        guard.config.grad_gx_integral = 2.5e-5;
        guard.config.grad_gy_integral = -1.8e-5;
    }
    runner.process_command(SimCommand::Custom("sync_view".to_string()));
    let chirped = runner.get_snapshot().sonification;
    assert_eq!(chirped.active_voices, 8);
    assert!(chirped.voices[0].fm_mod_hz > echo.voices[0].fm_mod_hz);

    // Paint KSpaceRfSpike stroke and verify transient_impulse click
    params.write().unwrap().config.brush_mode = StudioBrushMode::KSpaceRfSpike;
    runner.process_command(SimCommand::ApplyBrush {
        cx: 40,
        cy: 24,
        r: 1,
        is_obstacle: true,
    });
    let spiked = runner.get_snapshot().sonification;
    assert!(spiked.voices[1].transient_impulse >= 0.9);
}

#[test]
fn test_radar_60ghz_doppler_music_scatterers_cough_and_30ms_interlock_mute() {
    let expected_fd = (2.0 * RADAR_CARRIER_60GHZ_HZ * 0.015) / C;
    assert!((doppler_shift_60ghz_hz(0.015) - expected_fd).abs() < 1e-9);

    let mut tool = RadarGatingTool::default();
    tool.import_preset_json(STARTER_PRESET_JSON).unwrap();
    assert_eq!(tool.pipeline.spectra.coarse_peak_count, 1);
    assert_eq!(tool.pipeline.spectra.music_peak_count, 2);
    let sub_rayleigh_frame = tool.sonification_frame();
    assert_eq!(sub_rayleigh_frame.active_voices, 4);
    assert!(sub_rayleigh_frame.voices[2].pan < -0.5 && sub_rayleigh_frame.voices[3].pan > 0.5);

    // Positive vs negative Kalman velocity shifts Doppler pitch
    tool.pipeline.fast_kalman.state[1] = -0.015;
    let f_exhale = tool.sonification_frame();
    tool.pipeline.fast_kalman.state[1] = 0.015;
    let f_inhale = tool.sonification_frame();
    assert!(f_inhale.voices[0].freq_hz > f_exhale.voices[0].freq_hz + 60.0);

    // BeamState harmonic voicing transitions + cough transient upward sweep
    let armed_wave = tool.sonification_frame().waveform;
    for _ in 0..180 {
        if tool.step_gating() == BeamState::BeamActive {
            break;
        }
    }
    let active_frame = tool.sonification_frame();
    assert_ne!(active_frame.waveform, armed_wave);

    tool.inject_cough();
    let cough_frame = tool.sonification_frame();
    assert!(cough_frame.voices[0].freq_hz > active_frame.voices[0].freq_hz + 100.0);
    assert!(cough_frame.voices[0].transient_impulse > 0.4);
    for _ in 0..16 {
        if tool.step_gating() == BeamState::BeamHold {
            break;
        }
    }
    let hold_frame = tool.sonification_frame();
    assert_ne!(hold_frame.waveform, active_frame.waveform);

    // Inject NaN fault -> SafetyInterlock -> <= 30 ms mute via notify_solver_diverged
    ensure_controller_muted();
    super::submit_tool_frame("radar_gating", active_frame, false);
    super::toggle_mute_from_user_gesture();
    assert!(!super::snapshot_ui_state().muted);

    tool.inject_nan_fault();
    assert_eq!(tool.step_gating(), BeamState::SafetyInterlock);
    let ctx = egui::Context::default();
    let _ = ctx.run(egui::RawInput::default(), |ctx| tool.show(ctx));
    assert!(super::snapshot_ui_state().muted);
    assert_eq!(tool.sonification_frame().master_gain, 0.0);

    let mut synth = OfflinePcmSynthesizer::new(44_100, 2048);
    let _ = synth.set_muted_from_user_gesture(false);
    synth.apply_frame(&active_frame);
    let _ = synth.render_frames(1024);
    synth.mute_and_suspend(MuteReason::SolverDiverged);
    let tail = synth.render_frames(1350);
    let post_30ms = tail[2640..]
        .iter()
        .copied()
        .fold(0.0_f32, |a, b| a.max(b.abs()));
    assert!(
        post_30ms < 0.005,
        "SafetyInterlock fault must mute below 0.005 within 30 ms, got {post_30ms}"
    );
}

fn click_export_wav_button<F: FnMut(&egui::Context)>(ctx: &egui::Context, mut render: F) {
    let screen = Some(egui::Rect::from_min_size(
        egui::Pos2::ZERO,
        egui::vec2(1440.0, 1080.0),
    ));
    let out = ctx.run(
        egui::RawInput {
            screen_rect: screen,
            ..Default::default()
        },
        &mut render,
    );
    let ak = out.platform_output.accesskit_update.as_ref().unwrap();
    let bounds = ak
        .nodes
        .iter()
        .find(|(_, n)| n.label().is_some_and(|l| l.contains("Export .wav")))
        .and_then(|(_, n)| n.bounds())
        .expect("Export .wav button must be present in Accessibility tree");
    let center = egui::pos2(
        ((bounds.x0 + bounds.x1) * 0.5) as f32,
        ((bounds.y0 + bounds.y1) * 0.5) as f32,
    );
    let _ = ctx.run(
        egui::RawInput {
            screen_rect: screen,
            events: vec![
                egui::Event::PointerMoved(center),
                egui::Event::PointerButton {
                    pos: center,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::NONE,
                },
                egui::Event::PointerButton {
                    pos: center,
                    button: egui::PointerButton::Primary,
                    pressed: false,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
            ..Default::default()
        },
        &mut render,
    );
}

fn assert_valid_non_silent_stereo_wav(wav: &[u8], expected_name: &str, actual_name: &str) {
    assert_eq!(actual_name, expected_name);
    assert!(wav.len() > 44 + 4096);
    assert_eq!(&wav[0..4], b"RIFF");
    assert_eq!(&wav[8..12], b"WAVE");
    assert_eq!(u16::from_le_bytes([wav[20], wav[21]]), 1); // PCM
    assert_eq!(u16::from_le_bytes([wav[22], wav[23]]), 2); // Stereo
    assert_eq!(
        u32::from_le_bytes([wav[24], wav[25], wav[26], wav[27]]),
        44_100
    );
    assert_eq!(u16::from_le_bytes([wav[34], wav[35]]), 16); // 16-bit
    let mut peak_l = 0_i32;
    let mut peak_r = 0_i32;
    for &frame_bytes in wav[44..].as_chunks::<4>().0 {
        let l = i16::from_le_bytes([frame_bytes[0], frame_bytes[1]]) as i32;
        let r = i16::from_le_bytes([frame_bytes[2], frame_bytes[3]]) as i32;
        peak_l = peak_l.max(l.abs());
        peak_r = peak_r.max(r.abs());
    }
    assert!(
        peak_l > 800 && peak_r > 800,
        "{expected_name} must contain non-silent 16-bit stereo samples (L={peak_l}, R={peak_r})"
    );
    let _ = rms_stereo(&[peak_l as f32 / 32_768.0, peak_r as f32 / 32_768.0]);
}

#[test]
fn test_ui_export_wav_buttons_across_mri_kspace_and_radar_tools() {
    ensure_controller_muted();
    let _ = super::take_last_exported_wav();

    // 1. MriTool
    let ctx_mri = egui::Context::default();
    ctx_mri.enable_accesskit();
    let mut mri_tool = MriTool::default();
    mri_tool.step_simulation();
    click_export_wav_button(&ctx_mri, |ctx| mri_tool.show_ui(ctx));
    let (name_mri, wav_mri) = super::take_last_exported_wav().expect("MriTool wav exported");
    assert_valid_non_silent_stereo_wav(&wav_mri, "mri_bloch_sonification.wav", &name_mri);

    // 2. MriKSpaceTool
    let ctx_ks = egui::Context::default();
    ctx_ks.enable_accesskit();
    let mut ks_tool = MriKSpaceTool::default();
    click_export_wav_button(&ctx_ks, |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| ks_tool.show_inside(ui));
    });
    let (name_ks, wav_ks) = super::take_last_exported_wav().expect("MriKSpaceTool wav exported");
    assert_valid_non_silent_stereo_wav(&wav_ks, "mri_kspace_sonification.wav", &name_ks);

    // 3. RadarGatingTool
    let ctx_rad = egui::Context::default();
    ctx_rad.enable_accesskit();
    let mut rad_tool = RadarGatingTool::default();
    let _ = rad_tool.step_gating();
    click_export_wav_button(&ctx_rad, |ctx| rad_tool.show(ctx));
    let (name_rad, wav_rad) =
        super::take_last_exported_wav().expect("RadarGatingTool wav exported");
    assert_valid_non_silent_stereo_wav(&wav_rad, "radar_gating_sonification.wav", &name_rad);
    ensure_controller_muted();
}
