#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use math_commons::math_kernel::types::flatten_2d_index;

#[test]
fn test_column_major_dmatrix_transposes_to_row_major_accurately() {
    let mut engine = CoupledVortexMorphogenesis::new(VortexMorphogenesisConfig {
        width: 12,
        height: 6,
        advection_coupling: 1.0,
        ..VortexMorphogenesisConfig::default()
    });

    engine.lbm.set_inlet(3, 4, 1, 1, 0.11, -0.04);
    engine.lbm.set_obstacle(7, 2, true);
    engine.transpose_lbm_to_row_major();

    let probe_idx = flatten_2d_index(3, 4, 12);
    assert!((engine.ux_row()[probe_idx] - 0.11).abs() < 1e-12);
    assert!((engine.uy_row()[probe_idx] + 0.04).abs() < 1e-12);

    let obs_idx = flatten_2d_index(7, 2, 12);
    assert!(engine.obstacles_row()[obs_idx]);
    assert_eq!(engine.ux_row()[obs_idx], 0.0);
    assert_eq!(engine.uy_row()[obs_idx], 0.0);
}

#[test]
fn test_velocity_clamp_and_zero_allocation_under_verified_allocator() {
    let mut engine = CoupledVortexMorphogenesis::from_preset(VortexRegimePreset::KarmanVortexSpots);
    // Attempt an extreme inlet velocity > 0.15 and verify safety cap clamps |u| <= 0.15
    engine.config.inlet_ux = 0.45;

    verified_engine::engine::VerifiedEngine::run_verified(|| {
        for _ in 0..5 {
            engine.step();
        }
    });

    assert!(engine.is_finite());
    assert!(engine.metrics().max_speed <= VELOCITY_SAFETY_CAP + 1e-12);
    for idx in 0..engine.ux_row().len() {
        let speed = (engine.ux_row()[idx].powi(2) + engine.uy_row()[idx].powi(2)).sqrt();
        assert!(speed <= VELOCITY_SAFETY_CAP + 1e-12);
        if engine.obstacles_row()[idx] {
            assert_eq!(engine.morphogens.u()[idx], 0.0);
            assert_eq!(engine.morphogens.v()[idx], 0.0);
        }
    }
}

#[test]
fn test_regime_1_karman_vortex_trapped_spots() {
    let mut engine = CoupledVortexMorphogenesis::from_preset(VortexRegimePreset::KarmanVortexSpots);
    for _ in 0..180 {
        engine.step();
    }

    assert!(engine.is_finite());
    let w = engine.config.width;
    let h = engine.config.height;
    let v = engine.morphogens.v();
    let omega = engine.vorticity_row();

    // Verify downstream wake (x in 42..92) contains trapped autocatalytic spots co-located with non-zero vorticity
    let mut wake_max_v = 0.0_f64;
    let mut weighted_y_num = 0.0_f64;
    let mut weighted_vort = 0.0_f64;
    let mut weight_sum = 0.0_f64;
    for y in 4..h - 4 {
        for x in 42..92 {
            let idx = flatten_2d_index(x, y, w);
            if v[idx] > wake_max_v {
                wake_max_v = v[idx];
            }
            if v[idx] > 0.08 {
                weighted_y_num += v[idx] * (y as f64 - (h as f64 * 0.5)).abs();
                weighted_vort += v[idx] * omega[idx].abs();
                weight_sum += v[idx];
            }
        }
    }

    assert!(
        wake_max_v > 0.12 && weight_sum > 0.5,
        "Expected autocatalytic spots transported into Karman wake, got wake_max_v={wake_max_v}"
    );
    assert!(
        weighted_vort / weight_sum > 1e-3 && weighted_y_num / weight_sum > 0.25,
        "Expected spots trapped in off-axis swirling Karman vortices"
    );
}

#[test]
fn test_regime_2_shear_aligned_longitudinal_stripes() {
    let mut engine =
        CoupledVortexMorphogenesis::from_preset(VortexRegimePreset::ShearAlignedStripes);
    for _ in 0..160 {
        engine.step();
    }

    assert!(engine.is_finite());
    let w = engine.config.width;
    let h = engine.config.height;
    let u = engine.morphogens.u();

    // Longitudinal stripes parallel to x have strong transverse gradients |du/dy| >> streamwise |du/dx|
    let mut grad_x_sq = 0.0_f64;
    let mut grad_y_sq = 0.0_f64;
    for y in 3..h - 3 {
        for x in 3..w - 3 {
            let idx = flatten_2d_index(x, y, w);
            let du_dx = 0.5 * (u[idx + 1] - u[idx - 1]);
            let du_dy = 0.5 * (u[idx + w] - u[idx - w]);
            grad_x_sq += du_dx * du_dx;
            grad_y_sq += du_dy * du_dy;
        }
    }

    assert!(
        grad_y_sq > 1.8 * grad_x_sq && grad_y_sq > 1e-3,
        "Expected shear-aligned longitudinal stripes (grad_y_sq={grad_y_sq} vs grad_x_sq={grad_x_sq})"
    );
}

#[test]
fn test_regime_3_stagnation_point_wave_pinning_plumes() {
    let mut engine = CoupledVortexMorphogenesis::from_preset(VortexRegimePreset::StagnationPlumes);
    for _ in 0..160 {
        engine.step();
    }

    assert!(engine.is_finite());
    let w = engine.config.width;
    let v = engine.morphogens.v();

    // Stagnation zone behind bluff obstacle (x in 46..66, y in 18..32) pins a high-concentration plume
    // while outer high-speed freestream (y in 2..10) stays near zero.
    let mut stag_sum = 0.0_f64;
    let mut stag_cells = 0_usize;
    for y in 19..31 {
        for x in 46..66 {
            let idx = flatten_2d_index(x, y, w);
            if !engine.obstacles_row()[idx] {
                stag_sum += v[idx];
                stag_cells += 1;
            }
        }
    }

    let mut freestream_sum = 0.0_f64;
    let mut freestream_cells = 0_usize;
    for y in 2..10 {
        for x in 46..66 {
            let idx = flatten_2d_index(x, y, w);
            freestream_sum += v[idx];
            freestream_cells += 1;
        }
    }

    let stag_mean = stag_sum / (stag_cells.max(1) as f64);
    let freestream_mean = freestream_sum / (freestream_cells.max(1) as f64);
    assert!(
        stag_mean > 0.08 && stag_mean > 5.0 * freestream_mean,
        "Expected pinned stagnation plume: stag_mean={stag_mean}, freestream_mean={freestream_mean}"
    );
}

#[test]
fn test_regime_4_damkohler_washout_bifurcation() {
    // Low-flow control (Da >> 1): identical reaction kinetics sustain spots in the channel
    let mut low_flow =
        CoupledVortexMorphogenesis::from_preset(VortexRegimePreset::DamkohlerWashout);
    low_flow.config.inlet_ux = 0.005;
    low_flow.lbm.set_inlet(0, 1, 100, 48, 0.005, 0.0);

    // High-flow washout (Da < 1): identical kinetics, only high velocity sweeps catalyst to washout
    let mut high_flow =
        CoupledVortexMorphogenesis::from_preset(VortexRegimePreset::DamkohlerWashout);

    for _ in 0..160 {
        low_flow.step();
        high_flow.step();
    }

    let low_max_v = low_flow
        .morphogens
        .v()
        .iter()
        .copied()
        .fold(0.0_f64, f64::max);
    let high_max_v = high_flow
        .morphogens
        .v()
        .iter()
        .copied()
        .fold(0.0_f64, f64::max);

    assert!(
        high_flow.metrics().damkohler < 1.0 && low_flow.metrics().damkohler > 5.0,
        "Expected Damkohler transition across Da=1: low Da={}, high Da={}",
        low_flow.metrics().damkohler,
        high_flow.metrics().damkohler
    );
    assert!(
        low_max_v > 0.18 && high_max_v < 0.02,
        "Expected Damkohler washout bifurcation: low_flow max_v={low_max_v}, high_flow max_v={high_max_v}"
    );
}
