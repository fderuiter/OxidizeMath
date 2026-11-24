//! Verification tests for Chladni biharmonic plate and 2D acoustic wave tank sonification.

#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::backend::SonificationBackend;
use super::frame::{SonificationFrame, LIMITER_CEILING_LINEAR, MAX_FREQ_HZ, MIN_FREQ_HZ};
use super::mappings::chladni_wave_tank_frame;
use super::synth::OfflinePcmSynthesizer;
use super::wav::synthesize_frames_to_wav;

pub(super) fn sample_chladni_step_frame(
    phase: f64,
    det: &[f64],
    strike: bool,
) -> SonificationFrame {
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
