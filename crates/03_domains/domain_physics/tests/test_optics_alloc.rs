//! Integration and zero-allocation verification tests for `domain_physics::physics::optics`.

use domain_physics::physics::optics::caustics::HamiltonianRayTracer;
use domain_physics::physics::optics::refraction::{
    OpticalMediumConfig, OpticsPreset, snell_refract_vector, snell_refracted_angle,
};
use domain_physics::physics::optics::sensing::{DetectorConfig, sample_detector_line_in_place};
use math_commons::math_kernel::types::flatten_2d_index;
use pure_math::pure_math::analysis::pde::wave::{WaveField2D, WaveFieldConfig2D};
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

#[test]
fn test_optics_solvers_zero_heap_allocation_under_verified_allocator() {
    let mut field = WaveField2D::new(WaveFieldConfig2D {
        width: 80,
        height: 80,
        dx: 1.0 / 80.0,
        dy: 1.0 / 80.0,
        ..WaveFieldConfig2D::default()
    });
    let mut tracer = HamiltonianRayTracer::new(32, 300, 256);
    let mut profile = vec![0.0; 80];
    let det_cfg = DetectorConfig::default();

    for preset in OpticsPreset::ALL {
        let medium = OpticalMediumConfig {
            preset,
            ..OpticalMediumConfig::default()
        };
        let (ok, _) = VerifiedEngine::run_verified(|| {
            field.clear_state();
            medium.apply_to_wave_field(&mut field);
            let src = medium.default_wave_source(field.config());
            let dt = field.cfl_dt();
            for _ in 0..12 {
                field.step(dt, Some(&src));
            }
            tracer.trace_in_place(&medium, 17, 0.004);
            let readout = sample_detector_line_in_place(&field, &det_cfg, &mut profile);
            field.is_finite() && tracer.ray_count() == 17 && readout.axial_range > 0.0
        });
        assert!(
            ok,
            "preset {:?} must run with zero heap allocations",
            preset
        );
    }
}

#[test]
fn test_analytical_snell_angles_and_total_internal_reflection() {
    let theta_in = 0.45_f64;
    let theta_pos = snell_refracted_angle(1.0, 1.5, theta_in).expect("refracted");
    assert!((1.5 * theta_pos.sin() - theta_in.sin()).abs() < 1e-12);

    let theta_neg = snell_refracted_angle(1.0, -1.0, theta_in).expect("negative refraction");
    assert!((theta_neg + theta_in).abs() < 1e-12);

    // Above critical angle arcsin(1 / 1.5) ~ 0.7297 rad -> TIR
    assert!(snell_refracted_angle(1.5, 1.0, 0.90).is_none());
    let inc = [0.90_f64.cos(), 0.90_f64.sin()];
    let refl = snell_refract_vector(inc, [1.0, 0.0], 1.5, 1.0);
    assert!((refl[0] + inc[0]).abs() < 1e-12);
    assert!((refl[1] - inc[1]).abs() < 1e-12);
}

#[test]
fn test_luneburg_and_maxwell_fisheye_ray_focusing() {
    // 1. Luneburg lens: parallel rays focus onto rear surface (cx + R, cy).
    let luneburg = OpticalMediumConfig {
        preset: OpticsPreset::LuneburgLens,
        lens_radius: 0.22,
        ..OpticalMediumConfig::default()
    };
    let [cx, cy] = luneburg.lens_center();
    let focus_x = cx + luneburg.lens_radius;
    let mut tracer = HamiltonianRayTracer::new(24, 360, 256);
    tracer.trace_in_place(&luneburg, 13, 0.003);

    for r in 0..tracer.ray_count() {
        let pts = tracer.ray_points(r);
        let min_dist = pts
            .iter()
            .map(|p| ((p[0] - focus_x).powi(2) + (p[1] - cy).powi(2)).sqrt())
            .fold(f64::INFINITY, f64::min);
        assert!(
            min_dist < 0.008,
            "Luneburg ray {r} must pass through rear rim focus ({focus_x:.3}, {cy:.3}), min_dist={min_dist:.5}"
        );
    }

    // 2. Maxwell fish-eye: fan from (cx - R, cy) reconverges onto (cx + R, cy).
    let fisheye = OpticalMediumConfig {
        preset: OpticsPreset::MaxwellFishEye,
        lens_radius: 0.22,
        ..OpticalMediumConfig::default()
    };
    let [fx_c, fy_c] = fisheye.lens_center();
    let conj_x = fx_c + fisheye.lens_radius;
    tracer.trace_in_place(&fisheye, 11, 0.003);
    for r in 1..(tracer.ray_count() - 1) {
        let pts = tracer.ray_points(r);
        let min_dist = pts
            .iter()
            .map(|p| ((p[0] - conj_x).powi(2) + (p[1] - fy_c).powi(2)).sqrt())
            .fold(f64::INFINITY, f64::min);
        assert!(
            min_dist < 0.012,
            "Maxwell fish-eye ray {r} must converge onto antipodal focus, min_dist={min_dist:.5}"
        );
    }
}

#[test]
fn test_droplet_caustics_and_veselago_ray_foci() {
    let droplet = OpticalMediumConfig {
        preset: OpticsPreset::BiconvexDroplet,
        droplet_index: 1.50,
        ..OpticalMediumConfig::default()
    };
    let mut tracer = HamiltonianRayTracer::new(32, 360, 256);
    tracer.trace_in_place(&droplet, 21, 0.003);
    assert!(
        tracer.caustic_points().len() >= 4,
        "Biconvex droplet must produce spherical-aberration caustic envelope points"
    );

    let veselago = OpticalMediumConfig {
        preset: OpticsPreset::VeselagoLens,
        veselago_index: -1.0,
        frequency: 13.0,
        ..OpticalMediumConfig::default()
    };
    tracer.trace_in_place(&veselago, 15, 0.0025);
    for r in 0..tracer.ray_count() {
        let pts = tracer.ray_points(r);
        let d_internal = pts
            .iter()
            .map(|p| ((p[0] - 0.50).powi(2) + (p[1] - 0.50).powi(2)).sqrt())
            .fold(f64::INFINITY, f64::min);
        let d_external = pts
            .iter()
            .map(|p| ((p[0] - 0.74).powi(2) + (p[1] - 0.50).powi(2)).sqrt())
            .fold(f64::INFINITY, f64::min);
        assert!(
            d_internal < 0.006 && d_external < 0.008,
            "Veselago ray {r} must hit internal (0.50,0.50) and external (0.74,0.50) foci: ({d_internal:.4}, {d_external:.4})"
        );
    }
}

#[test]
fn test_veselago_fdtd_drude_superlens_focusing() {
    let veselago = OpticalMediumConfig {
        preset: OpticsPreset::VeselagoLens,
        veselago_index: -1.0,
        frequency: 13.0,
        ..OpticalMediumConfig::default()
    };
    let n = 100_usize;
    let dx = 1.0 / (n as f64);
    let mut field = WaveField2D::new(WaveFieldConfig2D {
        width: n,
        height: n,
        dx,
        dy: dx,
        sponge_width: 10,
        sponge_max_damping: 24.0,
        irradiance_decay: 0.025,
        ..WaveFieldConfig2D::default()
    });
    veselago.apply_to_wave_field(&mut field);
    let src = veselago.default_wave_source(field.config());
    let dt = field.cfl_dt();
    for _ in 0..420 {
        field.step(dt, Some(&src));
    }
    assert!(field.is_finite());
    let det_x = 74_usize;
    let axial_irr = field.irradiance()[flatten_2d_index(det_x, 50, n)];
    let edge_irr = field.irradiance()[flatten_2d_index(det_x, 22, n)];
    assert!(
        axial_irr > edge_irr && axial_irr > 0.0,
        "Veselago FDTD axial irradiance ({axial_irr}) must exceed off-axis ({edge_irr})"
    );
}
