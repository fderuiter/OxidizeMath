use super::*;
use verified_engine::engine::VerifiedEngine;

#[test]
fn test_kerr_noncircular_radial_infall_clock_dilation_matches_ode_ut() {
    let spec = WorldlineLaunchSpec {
        kind: GeodesicKind::Timelike,
        scale_radius: 2.0,
        spin_chi: 0.70,
        initial_radius: 9.0,
        launch_azimuth: 0.0,
        radial_bias: -0.90,
        step_size: 0.03,
    };
    let mut samples = Vec::with_capacity(MAX_TRAJECTORY_SAMPLES + 8);
    let _ = integrate_kerr_worldline_into(&spec, 80, 1.5, &mut samples);
    assert!(samples.len() > 10);
    let kerr = KerrEquatorialMetric::new(2.0, 0.70).expect("valid Kerr");
    for win in samples.windows(2) {
        let (s0, s1) = (&win[0], &win[1]);
        if s0.radial_coord > 1.2 * kerr.outer_horizon_radius() && s0.radial_vel.abs() > 0.05 {
            let circ = kerr.clock_dilation(s0.radial_coord, s0.dphi_dct).unwrap();
            assert!(s0.clock_rate < circ - 1.0e-3);
            let emp = (s1.proper_tau - s0.proper_tau) / (s1.coord_ct - s0.coord_ct);
            assert!((0.5 * (s0.clock_rate + s1.clock_rate) - emp).abs() < 1.0e-3);
        }
    }
}

#[test]
fn test_worldline_and_photon_fan_integrators_zero_alloc() {
    let spec = WorldlineLaunchSpec {
        kind: GeodesicKind::Null,
        scale_radius: 2.0,
        spin_chi: 0.60,
        initial_radius: 10.0,
        launch_azimuth: 0.0,
        radial_bias: -0.82,
        step_size: 0.04,
    };
    let ast = math_commons::ast::parse("1 - Rs / r").expect("parse lapse");
    let lapse = CompiledExpr::compile_with_vars(&ast, &["r", "Rs"]).expect("compile lapse");
    let mut buf = Vec::with_capacity(MAX_TRAJECTORY_SAMPLES + 8);
    let mut rays: Vec<BundleRay> = (0..4)
        .map(|_| BundleRay::with_capacity(MAX_TRAJECTORY_SAMPLES + 8))
        .collect();
    let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    VerifiedEngine::run_verified(|| {
        let _ = integrate_schwarzschild_worldline_into(&spec, 48, 5.5, &mut buf);
        let _ = integrate_kerr_worldline_into(&spec, 48, 5.5, &mut buf);
        let _ = integrate_wormhole_worldline_into(&spec, 48, 1.2, &mut buf);
        let _ = integrate_custom_lapse_worldline_into(&spec, &lapse, &[], 48, 5.5, &mut buf);
        let _ = integrate_photon_fan_bundle_into(4, 5.2, 2.0, 32, &mut rays, |st, b, out| {
            integrate_kerr_worldline_into(&spec, st, b, out)
        });
    });
    assert!(rays.iter().all(|r| !r.samples.is_empty()));
}
