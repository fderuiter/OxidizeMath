// [cite:advanced_linear_algebra]

use std::f64::consts::PI;

use math_commons::math_kernel::types::flatten_2d_index;

use super::masks::{
    apply_biconvex_lens, apply_multi_slit_barrier, apply_sponge_layer, clamped_beam_eigenfunction,
};
use super::particles_and_contours::DEFAULT_SAND_PARTICLES;
use super::*;

#[verified_engine::verified]
fn measure_clamped_mode_omega(mode_m: usize, mode_n: usize, grid_n: usize) -> f64 {
    let stiffness = 1.0_f64;
    let mut solver = Wave2DSolver::new_unit_square(grid_n);
    let mut init = vec![0.0; grid_n * grid_n];
    let denom = (grid_n - 1) as f64;

    for y in 0..grid_n {
        let yn = (y as f64) / denom;
        let py = clamped_beam_eigenfunction(mode_n, yn);
        for x in 0..grid_n {
            let xn = (x as f64) / denom;
            let val = clamped_beam_eigenfunction(mode_m, xn) * py;
            init[flatten_2d_index(x, y, grid_n)] = val;
        }
    }
    solver.set_stationary_displacement(&init);

    let params = Wave2DParams {
        medium: Wave2DMedium::BiharmonicPlate,
        boundary: BoundaryMode2D::Clamped,
        wave_speed: 0.0,
        plate_stiffness: stiffness,
        damping: 0.0,
    };
    let (_sub, dt_sub) = solver.safe_substep_plan(&params, 1.0);
    let dt = dt_sub * 0.8;
    let mut prev_proj: f64 = init.iter().map(|v| v * v).sum();
    let mut t_cross = 0.0_f64;

    for step_idx in 0..12_000 {
        solver.step(&params, dt);
        let curr_proj: f64 = solver
            .displacement()
            .iter()
            .zip(init.iter())
            .map(|(u, phi)| u * phi)
            .sum();
        if prev_proj > 0.0 && curr_proj <= 0.0 {
            let frac = prev_proj / (prev_proj - curr_proj);
            t_cross = ((step_idx as f64) + frac) * dt;
            break;
        }
        prev_proj = curr_proj;
    }
    assert!(
        t_cross > 0.0,
        "Mode failed to reach quarter-period zero crossing"
    );
    0.5 * PI / t_cross
}

#[test]
#[verified_engine::verified]
fn test_clamped_square_plate_eigenfrequencies_within_two_percent() {
    for (m, n) in [(1_usize, 1_usize), (1, 2), (2, 2), (1, 3)] {
        let measured = measure_clamped_mode_omega(m, n, 64);
        let analytic = analytic_clamped_square_omega(m, n, 1.0, 1.0);
        let rel_err = (measured - analytic).abs() / analytic;
        assert!(
            rel_err < 0.02,
            "Clamped mode ({m},{n}) measured={measured:.4}, analytic={analytic:.4}, rel_err={rel_err:.4}"
        );
    }
}

#[test]
#[verified_engine::verified]
fn test_double_slit_young_interference_fringe_spacing_128x128() {
    let n = 128_usize;
    let mut solver = Wave2DSolver::new(n, n, 1.0, 1.0);
    apply_sponge_layer(&mut solver.sponge_mask, n, n, 12, 1.8);

    let barrier_x = 24_usize;
    let detector_x = 96_usize;
    let slit_sep = 40.0_f64;
    apply_multi_slit_barrier(&mut solver.wall_mask, n, n, barrier_x, 1, 2, 2, slit_sep);

    let wave_speed = 1.0_f64;
    let wavelength = 10.0_f64;
    let frequency = wave_speed / wavelength;
    let params = Wave2DParams {
        medium: Wave2DMedium::AcousticWave,
        boundary: BoundaryMode2D::Sponge,
        wave_speed,
        plate_stiffness: 0.0,
        damping: 0.001,
    };
    let driver = DriverConfig2D {
        enabled: true,
        line_source: true,
        x_norm: 18.0 / 127.0,
        y_norm: 0.5,
        frequency,
        amplitude: 3.0,
        radius_cells: 1.5,
    };

    for step_idx in 0..920 {
        let det = if step_idx >= 560 {
            Some(detector_x)
        } else {
            None
        };
        solver.step_driven(&params, &driver, 0.30, det);
    }

    let mut profile = vec![0.0; n];
    solver.read_detector_profile(detector_x, &mut profile);
    let center_y = 64_usize;
    let mut upper_peak_y = center_y + 14;
    let mut upper_peak_val = 0.0_f64;
    for (y, &val) in profile
        .iter()
        .enumerate()
        .take(center_y + 25)
        .skip(center_y + 14)
    {
        if val > upper_peak_val {
            upper_peak_val = val;
            upper_peak_y = y;
        }
    }
    let denom = profile[upper_peak_y - 1] - 2.0 * profile[upper_peak_y] + profile[upper_peak_y + 1];
    let subcell = if denom.abs() > 1.0e-12 {
        0.5 * (profile[upper_peak_y - 1] - profile[upper_peak_y + 1]) / denom
    } else {
        0.0
    };
    let measured_spacing =
        ((upper_peak_y as f64) + subcell.clamp(-0.5, 0.5)) - ((n - 1) as f64 * 0.5);
    let expected_spacing =
        young_double_slit_fringe_spacing(wavelength, slit_sep, (detector_x - barrier_x) as f64);
    assert!(
        (measured_spacing - expected_spacing).abs() <= 1.0,
        "Measured fringe spacing {measured_spacing:.2} vs Young {expected_spacing:.2}"
    );
}

#[test]
#[verified_engine::verified]
fn test_acoustic_lens_focusing_and_spectral_peak() {
    let n = 64_usize;
    let mut solver = Wave2DSolver::new(n, n, 1.0, 1.0);
    apply_sponge_layer(&mut solver.sponge_mask, n, n, 6, 0.35);
    apply_biconvex_lens(&mut solver.wave_speed_mask, n, n, 24, 18.0, 5.0, 1.65);

    let params = Wave2DParams {
        medium: Wave2DMedium::AcousticWave,
        boundary: BoundaryMode2D::Sponge,
        wave_speed: 1.0,
        plate_stiffness: 0.0,
        damping: 0.002,
    };
    let driver = DriverConfig2D {
        enabled: true,
        line_source: true,
        x_norm: 10.0 / 63.0,
        y_norm: 0.5,
        frequency: 1.0 / 6.0,
        amplitude: 2.0,
        radius_cells: 1.4,
    };
    for idx in 0..220 {
        let det = if idx >= 130 { Some(46) } else { None };
        solver.step_driven(&params, &driver, 0.55, det);
    }
    let mut profile = vec![0.0; n];
    solver.read_detector_profile(46, &mut profile);
    let center_intensity = profile[n / 2];
    let flank_intensity = profile[n / 2 - 14];
    assert!(center_intensity > 2.0 * flank_intensity);

    let mut fft = SpectralAnalyzer2D::new(n, n);
    let spec = fft.compute_spectrum(solver.displacement());
    assert_eq!(spec.len(), n * n);
    let peak = fft.dominant_peak();
    assert!(peak.magnitude > 0.0 && peak.k_radial > 2.0);
}

#[test]
#[verified_engine::verified]
fn test_chladni_sand_drift_and_marching_squares_contours() {
    let n = 48_usize;
    let wall = vec![1.0; n * n];
    let mut modal = vec![0.0; n * n];
    let mut energy = vec![0.0; n * n];
    for y in 0..n {
        let yn = (y as f64) / ((n - 1) as f64);
        for x in 0..n {
            let xn = (x as f64) / ((n - 1) as f64);
            let mode =
                (2.0 * PI * xn).cos() * (PI * yn).cos() - (PI * xn).cos() * (2.0 * PI * yn).cos();
            let idx = flatten_2d_index(x, y, n);
            modal[idx] = mode;
            energy[idx] = mode * mode;
        }
    }

    let mut sand = ChladniSandSystem::new(DEFAULT_SAND_PARTICLES, 42, n, n, &wall);
    let initial_mean = sand.mean_normalized_energy(&energy, n);
    for _ in 0..80 {
        sand.step(&energy, &wall, n, n, 18.0, 0.35, 0.15);
    }
    let settled_mean = sand.mean_normalized_energy(&energy, n);
    assert!(
        settled_mean < initial_mean * 0.25,
        "Sand failed to settle onto nodal curves: {initial_mean:.4} -> {settled_mean:.4}"
    );

    let mut ms = MarchingSquares2D::new(4_096);
    let segs = ms.extract_zero_contours(&modal, &wall, n, n, 1.0e-3);
    assert!(
        segs.len() > 20,
        "Expected nodal contour segments, got {}",
        segs.len()
    );
}

#[test]
#[verified_engine::verified]
fn test_bilinear_signed_probe_and_top_peaks_multi_mode_extraction() {
    let n = 64_usize;
    let mut solver = Wave2DSolver::new_unit_square(n);
    let mut init = vec![0.0; n * n];
    let denom = (n - 1) as f64;
    for y in 0..n {
        let yn = (y as f64) / denom;
        for x in 0..n {
            let xn = (x as f64) / denom;
            let m1 = (2.0 * PI * 3.0 * xn).cos() * (2.0 * PI * 2.0 * yn).cos();
            let m2 = 0.55 * (2.0 * PI * 6.0 * xn).cos() * (2.0 * PI * 5.0 * yn).cos();
            init[flatten_2d_index(x, y, n)] = m1 + m2;
        }
    }
    solver.set_stationary_displacement(&init);
    let p0 = solver.sample_probe(0.5, 0.5);
    assert!(p0.displacement < -0.3, "Signed trough must stay negative");
    assert_eq!(p0.velocity, 0.0, "Stationary start velocity must be zero");

    let params = Wave2DParams {
        medium: Wave2DMedium::BiharmonicPlate,
        boundary: BoundaryMode2D::Neumann,
        wave_speed: 0.0,
        plate_stiffness: 1.0e-3,
        damping: 0.01,
    };
    let mut saw_pos_vel = false;
    let mut saw_neg_vel = false;
    for _ in 0..120 {
        solver.step(&params, 0.002);
        let s = solver.sample_probe(0.33, 0.25);
        saw_pos_vel |= s.velocity > 1.0e-3;
        saw_neg_vel |= s.velocity < -1.0e-3;
    }
    assert!(
        saw_pos_vel && saw_neg_vel,
        "Signed pickup velocity must capture both positive and negative half-cycles"
    );

    let mut fft = SpectralAnalyzer2D::new(n, n);
    fft.compute_spectrum(&init);
    let mut peaks = [SpectralPeak2D::default(); 8];
    let count = fft.top_peaks(&mut peaks);
    assert!(
        count >= 4,
        "Expected symmetric (±3,2) and (±6,5) standing-wave modes"
    );
    assert_eq!(peaks[0], fft.dominant_peak());
    assert!(peaks[0].magnitude >= peaks[1].magnitude);
    assert!(peaks[1].magnitude > peaks[2].magnitude);
    assert!((peaks[0].k_radial - (13.0_f64).sqrt()).abs() < 1.2);
    assert!((peaks[1].k_radial - (13.0_f64).sqrt()).abs() < 1.2);
    assert_eq!(peaks[0].kx, -peaks[1].kx);
    assert!((peaks[2].k_radial - (61.0_f64).sqrt()).abs() < 1.5);
    assert!((peaks[3].k_radial - (61.0_f64).sqrt()).abs() < 1.5);
}

#[test]
#[verified_engine::verified]
fn test_biharmonic_soundboard_rib_wave_speed_mask_shifts_plate_frequency() {
    let n = 48_usize;
    let mut base = Wave2DSolver::new_unit_square(n);
    let mut braced = Wave2DSolver::new_unit_square(n);
    for y in 16..32 {
        for x in 8..40 {
            braced.wave_speed_mask[flatten_2d_index(x, y, n)] = 1.55;
        }
    }
    base.inject_gaussian_pulse(0.45, 0.45, 0.08, 1.0);
    braced.inject_gaussian_pulse(0.45, 0.45, 0.08, 1.0);
    let params = Wave2DParams {
        medium: Wave2DMedium::BiharmonicPlate,
        boundary: BoundaryMode2D::Clamped,
        wave_speed: 0.0,
        plate_stiffness: 8.0e-4,
        damping: 0.02,
    };
    for _ in 0..40 {
        base.step(&params, 0.003);
        braced.step(&params, 0.003);
    }
    let pb = base.sample_probe(0.62, 0.50);
    let pr = braced.sample_probe(0.62, 0.50);
    assert!(pb.displacement.is_finite() && pr.displacement.is_finite());
    assert!(
        (pb.displacement - pr.displacement).abs() > 1.0e-4,
        "Soundboard bracing mask must alter Kirchhoff-Love flexural propagation"
    );
}
