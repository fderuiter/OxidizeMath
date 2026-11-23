// [cite:graph_parameters_rust]

use math_commons::math_kernel::types::flatten_2d_index;
use num_complex::Complex;
use oxidize_core::rng::OxidizeRng;
use verified_engine::engine::VerifiedEngine;

use super::diffraction::{
    AngularSpectrumPropagator, SlitApertureConfig, apply_thin_lens_phase_into,
    apply_zernike_aberration_into, field_intensity_into, fill_circular_pupil_into,
    fill_cosinusoidal_grating_into, fill_opaque_disk_aperture_into, fill_ronchi_grating_into,
    fill_slit_aperture_into, fresnel_number, optical_power, rayleigh_talbot_length, talbot_length,
    wrapped_phase_into,
};
use super::holography::{
    GerchbergSaxtonSolver, HologramRegime, HologramTargetPreset, fill_target_amplitude_into,
};
use super::sensing::{lambertian_intensity, lws_range_resolution};

#[test]
fn test_sensing_baseline_preserved_under_verified_engine() {
    let ((intensity, res), _) = VerifiedEngine::run_verified(|| {
        let i = lambertian_intensity(2.0 * std::f64::consts::PI, 1.0, 0.0, 1.0);
        let r = lws_range_resolution(10.0, 2.0, 8);
        (i, r)
    });
    let expected_res = 10.0 - (8.0_f64.powi(-4) * 256.0 + 10.0_f64.powi(-4)).powf(-0.25);
    assert!((intensity - 2.0).abs() < 1.0e-12);
    assert!((res - expected_res).abs() < 1.0e-12);
}

#[test]
fn test_angular_spectrum_parseval_conservation_and_zero_allocations() {
    let (w, h) = (64, 64);
    let (dx, dy, wavelength) = (8.0e-6, 8.0e-6, 532.0e-9);
    let mut prop = AngularSpectrumPropagator::new(w, h, dx, dy, wavelength);
    let mut aperture = vec![Complex::new(0.0, 0.0); w * h];
    let mut propagated = vec![Complex::new(0.0, 0.0); w * h];
    let mut roundtrip = vec![Complex::new(0.0, 0.0); w * h];

    let ((p0, pz, max_rt_err), _) = VerifiedEngine::run_verified(|| {
        fill_slit_aperture_into(
            &mut aperture,
            w,
            h,
            dx,
            dy,
            SlitApertureConfig {
                slit_count: 2,
                slit_width: 32.0e-6,
                slit_pitch: 96.0e-6,
                slit_height: 240.0e-6,
            },
        );
        apply_zernike_aberration_into(&mut aperture, w, h, dx, dy, 180.0e-6, 3, 1, 0.35);
        let p_in = optical_power(&aperture, dx, dy);
        prop.propagate_into(&aperture, 12.0e-3, &mut propagated);
        let p_out = optical_power(&propagated, dx, dy);
        prop.propagate_into(&propagated, -12.0e-3, &mut roundtrip);

        let mut max_err = 0.0_f64;
        for (a, b) in aperture.iter().zip(roundtrip.iter()) {
            max_err = max_err.max((*a - *b).norm());
        }
        (p_in, p_out, max_err)
    });

    let rel_drift = (pz - p0).abs() / p0;
    assert!(
        rel_drift < 1.0e-12,
        "Parseval optical power drift {rel_drift:.3e} exceeded 1.0e-12"
    );
    assert!(
        max_rt_err < 1.0e-12,
        "Forward/backward Helmholtz inversion error {max_rt_err:.3e} exceeded 1.0e-12"
    );
}

#[test]
fn test_evanescent_wave_filtering_prevents_subwavelength_blowup() {
    let (w, h) = (32, 32);
    let (dx, dy, wavelength) = (150.0e-9, 150.0e-9, 600.0e-9);
    let mut prop = AngularSpectrumPropagator::new(w, h, dx, dy, wavelength);
    let mut aperture = vec![Complex::new(0.0, 0.0); w * h];
    let mut out_fwd = vec![Complex::new(0.0, 0.0); w * h];
    let mut out_bwd = vec![Complex::new(0.0, 0.0); w * h];

    let ((p0, pf, pb), _) = VerifiedEngine::run_verified(|| {
        fill_ronchi_grating_into(&mut aperture, w, h, dx, 300.0e-9, 0.5, true);
        let p0 = optical_power(&aperture, dx, dy);
        prop.propagate_into(&aperture, 5.0e-6, &mut out_fwd);
        prop.propagate_into(&aperture, -5.0e-6, &mut out_bwd);
        (
            p0,
            optical_power(&out_fwd, dx, dy),
            optical_power(&out_bwd, dx, dy),
        )
    });
    assert!(pf < 1.0e-20 && pb < 1.0e-20 && p0 > 0.0);
}

#[test]
fn test_arago_bright_spot_exceeds_unit_intensity_at_fresnel_two() {
    let (w, h) = (128, 128);
    let wavelength = 500.0e-9;
    let dx = 8.0e-6;
    let disk_radius = 128.0e-6;
    let outer_window = 0.0;
    let z_nf2 = (disk_radius * disk_radius) / (wavelength * 2.0);
    assert!((fresnel_number(disk_radius, wavelength, z_nf2) - 2.0).abs() < 1.0e-12);

    let mut prop = AngularSpectrumPropagator::new(w, h, dx, dx, wavelength);
    let mut aperture = vec![Complex::new(0.0, 0.0); w * h];
    let mut field_z = vec![Complex::new(0.0, 0.0); w * h];
    let mut intensity = vec![0.0_f64; w * h];
    let mut phase = vec![0.0_f64; w * h];

    let ((center_i, shadow_ring_i), _) = VerifiedEngine::run_verified(|| {
        fill_opaque_disk_aperture_into(&mut aperture, w, h, dx, dx, disk_radius, outer_window);
        prop.propagate_into(&aperture, z_nf2, &mut field_z);
        field_intensity_into(&field_z, &mut intensity);
        wrapped_phase_into(&field_z, &mut phase);

        let cx = w / 2;
        let cy = h / 2;
        let center_val = intensity[flatten_2d_index(cx, cy, w)];
        let off_axis_val = intensity[flatten_2d_index(cx + 8, cy, w)];
        (center_val, off_axis_val)
    });

    assert!(
        center_i > 1.0,
        "Arago bright spot intensity at N_F = 2.0 must exceed 1.0, got {center_i:.5}"
    );
    assert!(
        shadow_ring_i < 0.35,
        "Geometric shadow off-axis intensity should remain dark, got {shadow_ring_i:.5}"
    );
}

#[test]
fn test_talbot_self_imaging_and_axial_carpet_consistency() {
    let (w, h) = (128, 32);
    let wavelength = 500.0e-9;
    let dx = 10.0e-6;
    let period = 160.0e-6;
    let z_exact = rayleigh_talbot_length(period, wavelength);
    let z_paraxial = talbot_length(period, wavelength);

    let mut prop = AngularSpectrumPropagator::new(w, h, dx, dx, wavelength);
    let mut cos_ap = vec![Complex::new(0.0, 0.0); w * h];
    let mut cos_zt = vec![Complex::new(0.0, 0.0); w * h];
    let mut ronchi_ap = vec![Complex::new(0.0, 0.0); w * h];
    let mut ronchi_half = vec![Complex::new(0.0, 0.0); w * h];
    let mut carpet = vec![0.0_f64; w * 9];

    let ((cos_rmse, half_shift_corr, carpet_slice_diff), _) = VerifiedEngine::run_verified(|| {
        fill_cosinusoidal_grating_into(&mut cos_ap, w, h, dx, period, 1.0);
        prop.propagate_into(&cos_ap, z_exact, &mut cos_zt);
        let mut sq_err = 0.0_f64;
        for (a, b) in cos_ap.iter().zip(cos_zt.iter()) {
            let di = a.norm_sqr() - b.norm_sqr();
            sq_err += di * di;
        }
        let rmse = (sq_err / ((w * h) as f64)).sqrt();

        fill_ronchi_grating_into(&mut ronchi_ap, w, h, dx, period, 0.5, false);
        prop.propagate_into(&ronchi_ap, 0.5 * z_paraxial, &mut ronchi_half);
        let half_period_px = 8;
        let mid_y = h / 2;
        let mut match_err = 0.0_f64;
        for x in 0..w {
            let i0 = ronchi_ap[flatten_2d_index(x, mid_y, w)].norm_sqr();
            let ih = ronchi_half[flatten_2d_index((x + half_period_px) % w, mid_y, w)].norm_sqr();
            match_err += (i0 - ih).abs();
        }
        match_err /= w as f64;

        prop.propagate_axial_carpet_into(&ronchi_ap, mid_y, 0.0, z_paraxial, 9, &mut carpet);
        let mut c_diff = 0.0_f64;
        for x in 0..w {
            let carpet_mid = carpet[flatten_2d_index(x, 4, w)];
            let direct_mid = ronchi_half[flatten_2d_index(x, mid_y, w)].norm_sqr();
            c_diff = c_diff.max((carpet_mid - direct_mid).abs());
        }
        (rmse, match_err, c_diff)
    });

    assert!(
        cos_rmse < 1.0e-13,
        "Rayleigh-Talbot self-imaging RMSE {cos_rmse:.3e} must approach machine precision"
    );
    assert!(
        half_shift_corr < 0.08,
        "Half-Talbot Ronchi grating should shift by half a period, mean diff = {half_shift_corr:.4}"
    );
    assert!(
        carpet_slice_diff < 1.0e-12,
        "Axial carpet slice must match 2D propagation to < 1e-12, got {carpet_slice_diff:.3e}"
    );
}

#[test]
fn test_thin_lens_focus_and_zernike_strehl_degradation() {
    let (w, h) = (64, 64);
    let (dx, wavelength, focal) = (10.0e-6, 532.0e-9, 12.0e-3);
    let pupil_r = 220.0e-6;
    let mut prop = AngularSpectrumPropagator::new(w, h, dx, dx, wavelength);
    let mut ideal = vec![Complex::new(0.0, 0.0); w * h];
    let mut aberrated = vec![Complex::new(0.0, 0.0); w * h];
    let mut focus_ideal = vec![Complex::new(0.0, 0.0); w * h];
    let mut focus_aberr = vec![Complex::new(0.0, 0.0); w * h];

    let ((i_ideal, i_aberr), _) = VerifiedEngine::run_verified(|| {
        fill_circular_pupil_into(&mut ideal, w, h, dx, dx, pupil_r);
        apply_thin_lens_phase_into(&mut ideal, w, h, dx, dx, wavelength, focal, pupil_r);
        aberrated.copy_from_slice(&ideal);
        apply_zernike_aberration_into(&mut aberrated, w, h, dx, dx, pupil_r, 4, 0, 0.6);

        prop.propagate_into(&ideal, focal, &mut focus_ideal);
        prop.propagate_into(&aberrated, focal, &mut focus_aberr);
        let c = flatten_2d_index(w / 2, h / 2, w);
        (focus_ideal[c].norm_sqr(), focus_aberr[c].norm_sqr())
    });

    assert!(
        i_ideal > 15.0,
        "Focusing lens must concentrate intensity at z = f, got {i_ideal}"
    );
    assert!(
        i_aberr < 0.65 * i_ideal,
        "Primary spherical Zernike aberration must reduce on-axis Strehl"
    );
}

#[test]
fn test_gerchberg_saxton_beats_real_scalar_floor_with_zero_allocations() {
    let (w, h) = (64, 64);
    let mut solver =
        GerchbergSaxtonSolver::new(w, h, 8.0e-6, 8.0e-6, 532.0e-9, HologramRegime::FourierLens);
    let mut target = vec![0.0_f64; w * h];
    let mut phase_u8 = vec![0_u8; w * h];
    let mut rng = OxidizeRng::from_seed(0x0001_D1CE_CAFE_BABE);

    let ((real_baseline, gs_final, unit_mod_err), _) = VerifiedEngine::run_verified(|| {
        fill_target_amplitude_into(HologramTargetPreset::OffAxisRingAndSpot, w, h, &mut target);
        solver.set_target_amplitude(&target);
        let baseline = solver.evaluate_real_scalar_baseline();
        solver.seed_random_phase(&mut rng);
        let final_metrics = solver.step_iterations(50);
        solver.quantize_phase_u8_into(&mut phase_u8);

        let mut mod_err = 0.0_f64;
        for c in solver.slm_field() {
            mod_err = mod_err.max((c.norm() - 1.0).abs());
        }
        (baseline, final_metrics, mod_err)
    });

    assert!(
        real_baseline.target_nrmse >= 0.70,
        "Real-scalar twin-image NRMSE floor should be ~0.707, got {:.4}",
        real_baseline.target_nrmse
    );
    assert!(
        gs_final.target_nrmse < 0.32,
        "Gerchberg-Saxton 50-iteration NRMSE {:.4} must be < 0.32",
        gs_final.target_nrmse
    );
    assert!(
        gs_final.diffraction_efficiency > 0.70,
        "Expected high diffraction efficiency, got {:.4}",
        gs_final.diffraction_efficiency
    );
    assert!(
        gs_final.twin_leakage < 0.05 && gs_final.twin_leakage < 0.15 * real_baseline.twin_leakage,
        "Expected twin-image suppression, got {:.4} vs baseline {:.4}",
        gs_final.twin_leakage,
        real_baseline.twin_leakage
    );
    assert!(unit_mod_err < 1.0e-14, "SLM field must retain unit modulus");
    assert!(phase_u8.iter().copied().max().unwrap_or(0) > 200);
}
