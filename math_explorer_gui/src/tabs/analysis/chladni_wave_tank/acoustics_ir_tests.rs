#![cfg_attr(any(), verified(opt_out = "tests"))]

// [cite:graph_parameters_rust]

use super::*;

fn measure_dominant_stereo_fft_hz(wav: &[u8], sample_rate: u32, frame_count: usize) -> (f64, f64) {
    use num_complex::Complex;
    use rustfft::FftPlanner;

    let ceiling_i16 = (crate::audio::frame::LIMITER_CEILING_LINEAR * 32_767.0).ceil() as i16 + 1;
    let fft_len = 262_144_usize;
    let mut spec = vec![Complex::new(0.0_f64, 0.0); fft_len];
    let mut diff_energy = 0.0_f64;
    for (i, chunk) in wav[44..]
        .as_chunks::<4>()
        .0
        .iter()
        .take(frame_count)
        .enumerate()
    {
        let l = i16::from_le_bytes([chunk[0], chunk[1]]);
        let r = i16::from_le_bytes([chunk[2], chunk[3]]);
        assert!(l.abs() <= ceiling_i16 && r.abs() <= ceiling_i16);
        let lf = f64::from(l) / 32_767.0;
        let rf = f64::from(r) / 32_767.0;
        diff_energy += (lf - rf) * (lf - rf);
        let hann = 0.5 * (1.0 - (std::f64::consts::TAU * (i as f64) / (frame_count as f64)).cos());
        spec[i] = Complex::new(0.5 * (lf + rf) * hann, 0.0);
    }
    let mut planner = FftPlanner::<f64>::new();
    planner.plan_fft_forward(fft_len).process(&mut spec);
    let (peak_bin, _) = spec[8..(fft_len / 2)]
        .iter()
        .enumerate()
        .map(|(k, c)| (k + 8, c.norm()))
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .expect("non-empty FFT spectrum");
    let measured_hz = (peak_bin as f64) * f64::from(sample_rate) / (fft_len as f64);
    (measured_hz, diff_energy)
}

#[test]
fn test_chladni_clamped_square_wav_ir_matches_leissa_omega_within_two_percent() {
    use math_explorer::pure_math::analysis::pde::wave_2d::analytic_clamped_square_omega;

    let mut tool = ChladniWaveTankTool::default();
    tool.state.medium = Wave2DMedium::BiharmonicPlate;
    tool.state.boundary = BoundaryMode2D::Clamped;
    tool.state.clamp_layout = ClampLayout::None;
    tool.state.plate_stiffness = 0.0004;
    tool.state.pickup_left = [0.31, 0.43];
    tool.state.pickup_right = [0.69, 0.57];

    let sample_rate = 44_100_u32;
    let wav = tool
        .export_wav_audio(sample_rate, 1.0)
        .expect("Chladni WAV IR export");
    assert_eq!(&wav[0..4], b"RIFF");
    assert_eq!(&wav[8..12], b"WAVE");
    let frame_count = 44_100_usize;
    assert_eq!(wav.len(), 44 + frame_count * 4);

    let (measured_hz, diff_energy) = measure_dominant_stereo_fft_hz(&wav, sample_rate, frame_count);
    assert!(
        diff_energy > 1.0e-3,
        "Asymmetric L/R pickups must decorrelate stereo channels"
    );
    let measured_omega = std::f64::consts::TAU * measured_hz;
    let expected_omega = analytic_clamped_square_omega(
        1,
        1,
        acoustics::CHLADNI_ACOUSTIC_LENGTH_METERS,
        tool.state.plate_stiffness,
    );
    let rel_err = (measured_omega - expected_omega).abs() / expected_omega;
    assert!(
        rel_err <= 0.02,
        "Chladni IR omega_11 mismatch: measured={measured_omega:.3} rad/s ({measured_hz:.2} Hz), analytic={expected_omega:.3} rad/s, rel_err={rel_err:.4}"
    );

    let midi = tool.export_midi_score(1.5).expect("Chladni MIDI");
    assert_eq!(&midi[0..4], b"MThd");
    let (scl, kbm) = tool.export_scala_tuning().expect("Chladni Scala");
    assert!(scl.contains('.') && !kbm.is_empty());
}

#[test]
fn test_chladni_impulse_render_zero_allocations_under_verified_engine() {
    let tool = ChladniWaveTankTool::default();
    let mut ws = acoustics::ChladniImpulseWorkspace::new(
        acoustics::IR_GRID_SIZE,
        acoustics::CHLADNI_ACOUSTIC_LENGTH_METERS,
    );
    ws.prepare(&tool.state);
    let ir_params = tool.impulse_pde_params();
    let mut stereo = vec![0.0_f32; 1_024 * 2];

    let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let (frames, _metrics) = verified_engine::engine::VerifiedEngine::run_verified(|| {
        acoustics::render_chladni_impulse_into(
            &mut ws,
            &ir_params,
            44_100,
            tool.state.pickup_left,
            tool.state.pickup_right,
            &mut stereo,
        )
    });
    assert_eq!(frames, 1_024);
    assert!(stereo.iter().all(|s| s.is_finite()));
}
