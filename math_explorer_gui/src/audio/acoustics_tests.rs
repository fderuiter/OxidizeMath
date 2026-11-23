#![cfg_attr(any(), verified(opt_out = "tests"))]

use num_complex::Complex;
use rustfft::FftPlanner;
use verified_engine::engine::VerifiedEngine;

use super::backend::SonificationBackend;
use super::frame::{SonificationFrame, LIMITER_CEILING_LINEAR};
use super::synth::OfflinePcmSynthesizer;
use super::wav::synthesize_frames_to_wav;

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

fn chladni_test_rig() -> (
    math_explorer::pure_math::analysis::pde::wave_2d::Wave2DSolver,
    math_explorer::pure_math::analysis::pde::wave_2d::Wave2DParams,
    math_explorer::pure_math::analysis::pde::wave_2d::DriverConfig2D,
) {
    use math_explorer::pure_math::analysis::pde::wave_2d::{
        BoundaryMode2D, DriverConfig2D, Wave2DMedium, Wave2DParams, Wave2DSolver,
    };
    let mut solver = Wave2DSolver::new_unit_square(16);
    solver.inject_gaussian_pulse(0.38, 0.44, 0.12, 0.9);
    let params = Wave2DParams {
        medium: Wave2DMedium::BiharmonicPlate,
        boundary: BoundaryMode2D::Clamped,
        wave_speed: 0.0,
        plate_stiffness: 4.0e-4,
        damping: 0.03,
    };
    let driver = DriverConfig2D {
        enabled: true,
        line_source: false,
        x_norm: 0.35,
        y_norm: 0.45,
        frequency: 1.4,
        amplitude: 35.0,
        radius_cells: 1.8,
    };
    (solver, params, driver)
}

#[test]
fn test_chladni_plate_1000_steps_zero_alloc_rustfft_modes_and_wav() {
    use math_explorer::pure_math::analysis::pde::wave_2d::{SpectralAnalyzer2D, SpectralPeak2D};

    let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let (mut solver, params, driver) = chladni_test_rig();
    let mut fft = SpectralAnalyzer2D::new(16, 16);
    let mut peaks = [SpectralPeak2D::default(); 8];
    let mut synth = OfflinePcmSynthesizer::new(16_384, 4_096 + 1024);
    let _ = synth.set_muted_from_user_gesture(false);

    let ((max_sample, last_frame), _metrics) = VerifiedEngine::run_verified(|| {
        let mut max_s = 0.0_f32;
        let mut latest = SonificationFrame::silent();
        for step in 0..1_000_usize {
            solver.step_driven(&params, &driver, 0.002, None);
            let probe = solver.sample_probe(0.66, 0.52);
            if step % 4 == 0 {
                fft.compute_spectrum(solver.displacement());
                let _ = fft.top_peaks(&mut peaks);
            }
            latest = super::mappings::chladni_plate_frame(
                &peaks,
                probe,
                params.medium,
                driver.frequency,
                params.wave_speed,
                params.plate_stiffness,
                0.66,
            );
            synth.apply_frame(&latest);
            for &s in synth.render_frames(16) {
                max_s = max_s.max(s.abs());
            }
        }
        (max_s, latest)
    });

    assert!(last_frame.is_audible() && last_frame.custom_harmonics.is_some());
    assert!((20.0..=8_000.0).contains(&last_frame.voices[0].freq_hz));
    assert!(max_sample > 0.01 && max_sample <= LIMITER_CEILING_LINEAR + 1e-6);

    let _ = synth.render_frames(1024);
    let mags = compute_mono_fft_magnitudes(synth.render_frames(4_096), 4_096);
    let f0 = peak_magnitude_around(&mags, 16_384, 4_096, last_frame.voices[0].freq_hz);
    assert!(f0 > 12.0, "Expected audible Chladni modal peak, got {f0}");

    let wav = synthesize_frames_to_wav(&[last_frame], 44_100, 512, 0.85);
    assert_eq!(&wav[0..4], b"RIFF");
    assert_eq!(&wav[8..12], b"WAVE");
    assert!(wav.len() > 2_048);
}
