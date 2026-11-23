use num_complex::Complex;
use rustfft::FftPlanner;
use verified_engine::engine::VerifiedEngine;

use super::backend::{MuteReason, SonificationBackend};
use super::frame::{
    SonificationFrame, VoiceSpec, WaveformPreset, FLOOR_DBFS, LIMITER_CEILING_DBFS,
    LIMITER_CEILING_LINEAR,
};
#[cfg(feature = "pure_math")]
use super::frame::{MAX_FREQ_HZ, MIN_FREQ_HZ};
#[cfg(feature = "pure_math")]
use super::mappings::chladni_wave_tank_frame;
use super::mappings::{
    attractor_frame, bifurcation_frame, hodgkin_huxley_frame, just_intonation_ratio,
    neural_ring_frame, prime_chord_frame, quantum_wavepacket_frame, spike_analysis_frame,
};
use super::synth::OfflinePcmSynthesizer;
use super::ui::{format_meter_readout, meter_fill_fraction, render_sonification_panel};
use super::wav::{encode_wav_stereo_pcm16, synthesize_frames_to_wav};

fn compute_mono_fft_magnitudes(interleaved_stereo: &[f32], fft_len: usize) -> Vec<f32> {
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

fn peak_magnitude_around(mags: &[f32], sample_rate: u32, fft_len: usize, target_hz: f32) -> f32 {
    let center = ((target_hz * (fft_len as f32)) / (sample_rate as f32)).round() as usize;
    let lo = center.saturating_sub(2);
    let hi = (center + 2).min(mags.len().saturating_sub(1));
    mags[lo..=hi].iter().copied().fold(0.0_f32, f32::max)
}

#[test]
fn test_zero_alloc_1000_steps_frame_extraction_and_offline_pcm_synthesis() {
    let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut synth = OfflinePcmSynthesizer::new(44_100, 64);
    let _ = synth.set_muted_from_user_gesture(false);
    let factors: [(u64, u32); 3] = [(2, 2), (3, 1), (5, 2)];
    let mut prev_phases = [0.0_f64; 8];
    #[cfg(feature = "pure_math")]
    let detector_line = [0.25_f64; 32];

    let (max_sample, _metrics) = VerifiedEngine::run_verified(|| {
        let mut peak = 0.0_f32;
        for step in 0..1_000usize {
            let phase = (step as f64) * 0.05;
            let frame = match step % 8 {
                0 => hodgkin_huxley_frame(30.0 * phase.sin(), -10.0, 0.8, 0.4, 0.5, 10.0),
                1 => spike_analysis_frame(25.0 * phase.cos(), -15.0, 0.0, 42.0),
                2 => {
                    let v = [-65.0 + 70.0 * phase.sin(), -50.0, 12.0, -70.0, -20.0];
                    let pv = [-70.0, -52.0, -5.0, -70.0, -25.0];
                    neural_ring_frame(&v, &pv, 0.0)
                }
                3 => {
                    let mut modes = [Complex::new(0.0, 0.0); 8];
                    modes[0] = Complex::from_polar(0.7, phase);
                    modes[1] = Complex::from_polar(0.6, phase * 2.3);
                    quantum_wavepacket_frame(&modes, &mut prev_phases, 0.02)
                }
                4 => attractor_frame([12.0 * phase.sin(), 8.0, 24.0], [18.0, -12.0, 9.0], false),
                5 => bifurcation_frame(2.9 + 0.9 * ((step % 10) as f64) * 0.1),
                #[cfg(feature = "pure_math")]
                6 => sample_chladni_step_frame(phase, &detector_line, step % 16 == 0),
                _ => prime_chord_frame(&factors, WaveformPreset::JustHarmonic),
            };
            synth.apply_frame(&frame);
            for &s in synth.render_frames(32) {
                peak = peak.max(s.abs());
            }
        }
        peak
    });
    assert!(max_sample > 0.01);
    assert!(max_sample <= LIMITER_CEILING_LINEAR + 1e-6);
    assert!(synth.meter_reading().peak_dbfs <= LIMITER_CEILING_DBFS);
}

#[test]
fn test_rustfft_fundamental_overtones_and_limiter_ceiling() {
    let sr = 16_384_u32;
    let fft_len = 4_096_usize;
    let mut synth = OfflinePcmSynthesizer::new(sr, fft_len + 1024);
    let _ = synth.set_muted_from_user_gesture(false);
    synth.set_master_volume(1.0);

    let mut frame = SonificationFrame::silent();
    frame.waveform = WaveformPreset::JustHarmonic;
    frame.active_voices = 8;
    frame.master_gain = 1.0;
    frame.voices.fill(VoiceSpec {
        freq_hz: 440.0,
        gain: 1.0,
        pan: 0.0,
        fm_mod_hz: 0.0,
        fm_index: 0.0,
        filter_cutoff_hz: 6000.0,
        transient_impulse: 0.0,
    });
    synth.apply_frame(&frame);
    let _ = synth.render_frames(1024); // settle 15ms smoothing
    let rendered = synth.render_frames(fft_len);
    for &sample in rendered {
        assert!(sample.abs() <= LIMITER_CEILING_LINEAR + 1e-6);
    }

    let mags = compute_mono_fft_magnitudes(rendered, fft_len);
    let f1 = peak_magnitude_around(&mags, sr, fft_len, 440.0);
    let f2 = peak_magnitude_around(&mags, sr, fft_len, 880.0);
    let off = peak_magnitude_around(&mags, sr, fft_len, 610.0);
    assert!(f1 > 80.0, "expected strong 440 Hz fundamental, got {f1}");
    assert!(
        f2 > 15.0,
        "expected Fourier 2nd harmonic at 880 Hz, got {f2}"
    );
    assert!(f1 > off * 10.0);
}

#[test]
fn test_rustfft_quantum_beats_and_prime_chord_just_intonation() {
    assert!((just_intonation_ratio(2) - 2.0).abs() < 1e-6);
    assert!((just_intonation_ratio(3) - 1.5).abs() < 1e-6);
    assert!((just_intonation_ratio(5) - 1.25).abs() < 1e-6);
    assert!((just_intonation_ratio(7) - 1.75).abs() < 1e-6);

    let sr = 16_384_u32;
    let fft_len = 4_096_usize;
    let mut synth = OfflinePcmSynthesizer::new(sr, fft_len + 1024);
    let _ = synth.set_muted_from_user_gesture(false);

    // n = 2^1 * 3^4 * 5^1 -> 220*2 = 440 Hz, 220*1.5 = 330 Hz (dominant), 220*1.25 = 275 Hz
    let chord = prime_chord_frame(&[(2, 1), (3, 4), (5, 1)], WaveformPreset::Sine);
    synth.apply_frame(&chord);
    let _ = synth.render_frames(1024);
    let mags = compute_mono_fft_magnitudes(synth.render_frames(fft_len), fft_len);
    let m_third = peak_magnitude_around(&mags, sr, fft_len, 275.0);
    let m_fifth = peak_magnitude_around(&mags, sr, fft_len, 330.0);
    let m_oct = peak_magnitude_around(&mags, sr, fft_len, 440.0);
    let m_void = peak_magnitude_around(&mags, sr, fft_len, 512.0);
    assert!(m_fifth > m_third && m_fifth > m_oct);
    assert!(m_third > m_void * 8.0 && m_oct > m_void * 8.0);

    // Quantum two-mode superposition produces detuned beat sideband
    let mut prev_phases = [0.0_f64; 8];
    let mut modes = [Complex::new(0.0, 0.0); 8];
    modes[0] = Complex::from_polar(0.75, 0.2);
    modes[1] = Complex::from_polar(0.65, 1.4);
    let _ = quantum_wavepacket_frame(&modes, &mut prev_phases, 0.02);
    modes[0] = Complex::from_polar(0.75, 0.35);
    modes[1] = Complex::from_polar(0.65, 2.85);
    let q_frame = quantum_wavepacket_frame(&modes, &mut prev_phases, 0.02);
    assert!(q_frame.custom_harmonics.is_some());
    synth.apply_frame(&q_frame);
    let _ = synth.render_frames(1024);
    let q_mags = compute_mono_fft_magnitudes(synth.render_frames(fft_len), fft_len);
    let carrier = peak_magnitude_around(&q_mags, sr, fft_len, q_frame.voices[0].freq_hz);
    let beat_voice = peak_magnitude_around(&q_mags, sr, fft_len, q_frame.voices[4].freq_hz);
    assert!(carrier > 15.0 && beat_voice > 5.0);
}

#[test]
fn test_rustfft_period_doubling_subharmonic_and_lorenz_stereo_balance() {
    let sr = 16_384_u32;
    let fft_len = 4_096_usize;
    let mut synth = OfflinePcmSynthesizer::new(sr, fft_len + 1024);
    let _ = synth.set_muted_from_user_gesture(false);

    let p1 = bifurcation_frame(2.75);
    let p2 = bifurcation_frame(3.30);
    let chaos = bifurcation_frame(3.92);
    assert_eq!(p1.voices[1].gain, 0.0);
    assert!(
        p2.voices[1].gain > 0.15,
        "period-2 subharmonic must activate"
    );
    assert!(
        chaos.noise_gain > 0.30,
        "chaotic regime must activate filtered noise"
    );

    synth.apply_frame(&p2);
    let _ = synth.render_frames(1024);
    let mags = compute_mono_fft_magnitudes(synth.render_frames(fft_len), fft_len);
    let sub_mag = peak_magnitude_around(&mags, sr, fft_len, p2.voices[1].freq_hz);
    assert!(
        sub_mag > 10.0,
        "expected audible f0/2 subharmonic peak, got {sub_mag}"
    );

    // Lorenz negative-x vs positive-x stereo panning
    let left_lobe = attractor_frame([-18.0, -12.0, 26.0], [10.0, 15.0, 5.0], false);
    synth.apply_frame(&left_lobe);
    let _ = synth.render_frames(1024);
    let samples_l = synth.render_frames(1024);
    let (rms_l1, rms_r1) = stereo_energy(samples_l);
    assert!(rms_l1 > rms_r1 * 2.0);

    let right_lobe = attractor_frame([18.0, 12.0, 26.0], [10.0, 15.0, 5.0], false);
    synth.apply_frame(&right_lobe);
    let _ = synth.render_frames(1024);
    let samples_r = synth.render_frames(1024);
    let (rms_l2, rms_r2) = stereo_energy(samples_r);
    assert!(rms_r2 > rms_l2 * 2.0);
}

fn stereo_energy(interleaved: &[f32]) -> (f32, f32) {
    let mut el = 0.0_f32;
    let mut er = 0.0_f32;
    for &[l, r] in interleaved.as_chunks::<2>().0 {
        el += l * l;
        er += r * r;
    }
    (el, er)
}

#[test]
fn test_mute_within_30ms_wav_encoding_and_accessible_ui_themes() {
    let mut synth = OfflinePcmSynthesizer::new(44_100, 2048);
    let _ = synth.set_muted_from_user_gesture(false);
    let ring = neural_ring_frame(&[20.0, -10.0, -40.0, 15.0, -60.0], &[-5.0; 5], 0.0);
    synth.apply_frame(&ring);
    let _ = synth.render_frames(1024);

    synth.mute_and_suspend(MuteReason::WebGlContextLost);
    let tail = synth.render_frames(1350); // ~30.6 ms at 44.1 kHz
    let after_30ms_peak = tail[2640..] // samples >= 29.93 ms (1320 stereo frames * 2)
        .iter()
        .copied()
        .fold(0.0_f32, |a, b| a.max(b.abs()));
    assert!(
        after_30ms_peak < 0.005,
        "output must decay to silence within 30 ms, got {after_30ms_peak}"
    );
    assert_eq!(synth.meter_reading().peak_dbfs, FLOOR_DBFS);

    let wav = synthesize_frames_to_wav(&[ring], 44_100, 256, 0.8);
    assert_eq!(&wav[0..4], b"RIFF");
    assert_eq!(&wav[8..12], b"WAVE");
    assert_eq!(&wav[12..16], b"fmt ");
    assert_eq!(&wav[36..40], b"data");
    assert_eq!(encode_wav_stereo_pcm16(44_100, &[0.25, -0.25]).len(), 48);

    assert_eq!(meter_fill_fraction(FLOOR_DBFS, false), 0.0);
    assert!(format_meter_readout(&synth.meter_reading(), true).contains("MUTED"));

    // Render sonification panel across Dark, Light, and HighContrast visuals
    let ctx = eframe::egui::Context::default();
    for mode in [
        crate::app::theme::ThemeMode::Dark,
        crate::app::theme::ThemeMode::Light,
        crate::app::theme::ThemeMode::HighContrast,
    ] {
        mode.apply_to_context(&ctx);
        let _ = ctx.run(eframe::egui::RawInput::default(), |ctx| {
            eframe::egui::CentralPanel::default().show(ctx, |ui| {
                render_sonification_panel(ui, "hodgkin_huxley", &ring, false);
            });
        });
    }

    crate::accessibility::clear_test_announcements();
    super::toggle_mute_from_user_gesture();
    super::on_webgl_context_lost();
    let msgs = crate::accessibility::all_announcements();
    assert!(msgs
        .iter()
        .any(|(text, prio)| prio == "polite" && text.contains("suspended cleanly")));
}

#[cfg(feature = "pure_math")]
fn sample_chladni_step_frame(phase: f64, det: &[f64], strike: bool) -> SonificationFrame {
    use math_explorer::pure_math::analysis::pde::wave_2d::{SpectralMode2D, Wave2DParams};
    let modes = [
        SpectralMode2D {
            kx: 2,
            ky: 1,
            k_radial: 5.0_f64.sqrt(),
            magnitude: 0.04,
            phase,
        },
        SpectralMode2D {
            kx: -3,
            ky: 2,
            k_radial: 13.0_f64.sqrt(),
            magnitude: 0.025,
            phase: -0.5 * phase,
        },
    ];
    let params = Wave2DParams {
        plate_stiffness: 0.0004,
        ..Wave2DParams::default()
    };
    chladni_wave_tank_frame(&modes, &params, 1.0, det, if strike { 0.9 } else { 0.0 })
}

#[cfg(feature = "pure_math")]
fn build_free_and_pinned_plate_solvers(
    n: usize,
    params: &math_explorer::pure_math::analysis::pde::wave_2d::Wave2DParams,
) -> (
    math_explorer::pure_math::analysis::pde::wave_2d::Wave2DSolver,
    math_explorer::pure_math::analysis::pde::wave_2d::Wave2DSolver,
) {
    use math_explorer::pure_math::analysis::pde::wave_2d::{apply_point_clamp, Wave2DSolver};
    let mut free_solver = Wave2DSolver::new_unit_square(n);
    let mut pinned_solver = Wave2DSolver::new_unit_square(n);
    for (cx, cy) in [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)] {
        apply_point_clamp(&mut pinned_solver.wall_mask, n, n, cx, cy, 5.0);
    }
    let init: Vec<f64> = (0..(n * n))
        .map(|idx| {
            let x = (idx % n) as f64 / ((n - 1) as f64);
            let y = (idx / n) as f64 / ((n - 1) as f64);
            (2.0 * std::f64::consts::PI * x).sin() * (2.0 * std::f64::consts::PI * y).sin()
                + 0.72
                    * (4.0 * std::f64::consts::PI * x).sin()
                    * (4.0 * std::f64::consts::PI * y).sin()
        })
        .collect();
    let pinned_init: Vec<f64> = init
        .iter()
        .zip(pinned_solver.wall_mask.iter())
        .map(|(&u, &w)| if w > 0.5 { u } else { 0.0 })
        .collect();
    free_solver.set_stationary_displacement(&init);
    pinned_solver.set_stationary_displacement(&pinned_init);
    for _ in 0..80 {
        free_solver.step(params, 0.004);
        pinned_solver.step(params, 0.004);
    }
    (free_solver, pinned_solver)
}

#[cfg(feature = "pure_math")]
#[test]
fn test_chladni_plate_clamp_pitch_shift_fringes_limiter_and_wav() {
    use math_explorer::pure_math::analysis::pde::wave_2d::{
        BoundaryMode2D, SpectralAnalyzer2D, SpectralMode2D, Wave2DMedium, Wave2DParams,
    };

    let n = 64_usize;
    let params = Wave2DParams {
        medium: Wave2DMedium::BiharmonicPlate,
        boundary: BoundaryMode2D::Dirichlet,
        wave_speed: 0.0,
        plate_stiffness: 0.0016,
        damping: 0.35,
    };
    let (free_solver, pinned_solver) = build_free_and_pinned_plate_solvers(n, &params);

    let mut fft = SpectralAnalyzer2D::new(n, n);
    let mut free_modes = [SpectralMode2D::default(); 6];
    let mut pinned_modes = [SpectralMode2D::default(); 6];
    let _ = fft.compute_spectrum(free_solver.displacement());
    let fc = fft.extract_top_modes(&mut free_modes);
    let _ = fft.compute_spectrum(pinned_solver.displacement());
    let pc = fft.extract_top_modes(&mut pinned_modes);

    let flat_det = [0.0_f64; 64];
    let fringe_det: Vec<f64> = (0..64)
        .map(|y| (std::f64::consts::PI * (y as f64) / 5.0).cos().powi(2))
        .collect();
    let free_frame = chladni_wave_tank_frame(&free_modes[..fc], &params, 1.0, &flat_det, 0.0);
    let pinned_frame =
        chladni_wave_tank_frame(&pinned_modes[..pc], &params, 0.93, &fringe_det, 1.0);
    assert!(pinned_frame.voices[0].freq_hz > free_frame.voices[0].freq_hz);
    assert!(pinned_frame.voices[0].fm_mod_hz > free_frame.voices[0].fm_mod_hz);
    assert!(pinned_frame.voices[0].transient_impulse > 0.9);
    assert!(pinned_frame.custom_harmonics.is_some());

    verify_chladni_dispersion_safety_and_wav(&pinned_modes[..pc], &fringe_det, &pinned_frame);
}

#[cfg(feature = "pure_math")]
fn verify_chladni_dispersion_safety_and_wav(
    modes: &[math_explorer::pure_math::analysis::pde::wave_2d::SpectralMode2D],
    det: &[f64],
    active_frame: &SonificationFrame,
) {
    use math_explorer::pure_math::analysis::pde::wave_2d::{
        BoundaryMode2D, Wave2DMedium, Wave2DParams,
    };
    for stiffness in [0.0, 0.05] {
        let extreme = Wave2DParams {
            medium: Wave2DMedium::BiharmonicPlate,
            boundary: BoundaryMode2D::Neumann,
            wave_speed: 0.0,
            plate_stiffness: stiffness,
            damping: 0.05,
        };
        let f = chladni_wave_tank_frame(modes, &extreme, 1.0, det, 0.0);
        for v in &f.voices[..f.active_voices as usize] {
            assert!((MIN_FREQ_HZ..=MAX_FREQ_HZ).contains(&v.freq_hz));
        }
    }
    let slow_wave = Wave2DParams {
        medium: Wave2DMedium::AcousticWave,
        boundary: BoundaryMode2D::Dirichlet,
        wave_speed: 1.0,
        plate_stiffness: 0.0,
        damping: 0.01,
    };
    let fast_wave = Wave2DParams {
        wave_speed: 2.0,
        ..slow_wave
    };
    let f_slow = chladni_wave_tank_frame(modes, &slow_wave, 1.0, det, 0.0);
    let f_fast = chladni_wave_tank_frame(modes, &fast_wave, 1.0, det, 0.0);
    assert!(f_fast.voices[0].freq_hz > f_slow.voices[0].freq_hz * 1.8);

    assert!(!chladni_wave_tank_frame(&[], &slow_wave, 1.0, det, 0.0).is_audible());
    let nan_params = Wave2DParams {
        wave_speed: f64::NAN,
        ..slow_wave
    };
    assert!(!chladni_wave_tank_frame(modes, &nan_params, 1.0, det, 0.0).is_audible());

    let mut synth = OfflinePcmSynthesizer::new(16_384, 2048);
    let _ = synth.set_muted_from_user_gesture(false);
    synth.set_master_volume(1.0);
    synth.apply_frame(active_frame);
    let pcm = synth.render_frames(2048);
    assert!(pcm.iter().all(|s| s.abs() <= LIMITER_CEILING_LINEAR + 1e-6));
    let wav = synthesize_frames_to_wav(&[*active_frame], 44_100, 512, 0.9);
    assert_eq!(&wav[0..4], b"RIFF");
    assert!(wav.len() > 2_000);
}
