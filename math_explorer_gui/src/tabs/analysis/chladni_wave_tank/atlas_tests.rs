#![cfg_attr(any(), verified(opt_out = "tests"))]

// [cite:advanced_linear_algebra]

use eframe::egui;
use math_explorer::pure_math::analysis::pde::wave_2d::{
    BoundaryMode2D, Wave2DMedium, evaluate_eigenmode_amplitude,
};
use scientific_metadata::theory::TheoryDescribable;

use super::atlas::{
    CHLADNI_ATLAS_AXIS_PAIRS, ChladniModalCartographer, ChladniModalRegime, DRUMHEAD_RADIUS_NORM,
    chladni_parameter_constraints,
};
use super::challenge::CHALLENGE_CHLADNI_RESONANCE;
use super::*;
use crate::async_sim::SimulationRunner;
use crate::async_sim::atlas::{AtlasTeleportEvent, RegimeCartographer};
use crate::framework::InteractiveTool;

fn wcag_contrast_on_black(color: egui::Color32) -> f64 {
    let ch = |c: u8| {
        let s = f64::from(c) / 255.0;
        if s <= 0.04045 {
            s / 12.92
        } else {
            ((s + 0.055) / 1.055).powf(2.4)
        }
    };
    let lum = 0.2126 * ch(color.r()) + 0.7152 * ch(color.g()) + 0.0722 * ch(color.b());
    (lum + 0.05) / 0.05
}

fn build_test_chladni_cartographers() -> (
    ChladniModalCartographer,
    ChladniModalCartographer,
    ChladniModalCartographer,
) {
    let mut tool = ChladniWaveTankTool::default();
    let theory = tool.theory_parameters();
    assert_eq!(theory.get("driver_frequency").unwrap().step, 0.001);
    let len = tool.domain_physical_length();
    let cart_p =
        ChladniModalCartographer::new(&theory, CHLADNI_ATLAS_AXIS_PAIRS[0], &tool.state, len);
    tool.apply_preset(TankPreset::CircularDrumhead);
    let cart_d =
        ChladniModalCartographer::new(&theory, CHLADNI_ATLAS_AXIS_PAIRS[1], &tool.state, len);
    let cart_s =
        ChladniModalCartographer::new(&theory, CHLADNI_ATLAS_AXIS_PAIRS[2], &tool.state, len);
    (cart_p, cart_d, cart_s)
}

#[test]
fn test_chladni_cartographer_zero_allocation_128x128_and_wcag_contrast() {
    let (cart_p, cart_d, cart_s) = build_test_chladni_cartographers();
    let (axes_p, axes_d, axes_s) = (cart_p.axes(), cart_d.axes(), cart_s.axes());
    assert!(cart_p.overlay_contour().is_some() && cart_d.overlay_contour().is_some());
    let shared_req = std::sync::Arc::new(std::sync::RwLock::new(
        crate::async_sim::atlas::AtlasRenderRequest {
            generation: 1,
            width: 128,
            height: 128,
            high_contrast: true,
            cartographer: std::sync::Arc::new(cart_p.clone()),
        },
    ));
    let latest_gen = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(1));
    let mut atlas_runner = crate::async_sim::atlas::AtlasRunner::new(shared_req, latest_gen);

    {
        let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        verified_engine::engine::VerifiedEngine::run_verified(|| {
            while !atlas_runner.is_complete() {
                atlas_runner.step();
            }
            for sy in 0..128 {
                for sx in 0..128 {
                    let (p1, p2) = axes_p.pixel_to_coords(sx, sy, 128, 128);
                    let cell = cart_p.evaluate_cell(p1, p2);
                    assert!(cell.primary.is_finite() && cell.secondary.is_finite());
                    if (sx + sy) % 16 == 0 {
                        assert!(wcag_contrast_on_black(cart_p.color_for_cell(cell, true)) >= 3.0);
                    }
                }
            }
            for sy in 0..24 {
                for sx in 0..24 {
                    let (d1, d2) = axes_d.pixel_to_coords(sx, sy, 24, 24);
                    let cd = cart_d.evaluate_cell(d1, d2);
                    assert!(wcag_contrast_on_black(cart_d.color_for_cell(cd, true)) >= 3.0);
                    let (x1, y1) = axes_s.pixel_to_coords(sx, sy, 24, 24);
                    let cs = cart_s.evaluate_cell(x1, y1);
                    assert!(wcag_contrast_on_black(cart_s.color_for_cell(cs, true)) >= 3.0);
                }
            }
        });
    }
    assert!(atlas_runner.is_complete());
    assert_eq!(
        ChladniModalRegime::from_u8(cart_s.evaluate_cell(0.5, 0.5).regime),
        ChladniModalRegime::NodalDeadZone
    );
}

#[test]
fn test_boundary_accurate_eigenfrequency_readouts_across_plate_and_drumhead() {
    let mut tool = ChladniWaveTankTool::default();
    tool.state.clamp_layout = ClampLayout::None;
    tool.state.medium = Wave2DMedium::BiharmonicPlate;
    tool.state.plate_stiffness = 0.0004;
    tool.state.wave_speed = 0.0;

    tool.state.boundary = BoundaryMode2D::Neumann;
    let (lbl_neu, f_neu) = tool
        .boundary_fundamental_readout()
        .expect("Neumann readout");
    assert!(lbl_neu.starts_with("Neumann f("));

    tool.state.boundary = BoundaryMode2D::Dirichlet;
    let (lbl_dir, f_dir) = tool
        .boundary_fundamental_readout()
        .expect("Dirichlet readout");
    assert!(lbl_dir.starts_with("Dirichlet f(1,1)"));

    tool.state.boundary = BoundaryMode2D::Clamped;
    let (lbl_clp, f_clp) = tool
        .boundary_fundamental_readout()
        .expect("Clamped readout");
    assert!(lbl_clp.starts_with("Clamped f(1,1)"));
    assert!(
        f_neu < f_dir && f_dir < f_clp,
        "Expected free Neumann ({f_neu:.4}) < simply-supported Dirichlet ({f_dir:.4}) < Clamped ({f_clp:.4})"
    );

    tool.apply_preset(TankPreset::CircularDrumhead);
    let (lbl_bes, f_bes) = tool.boundary_fundamental_readout().expect("Bessel readout");
    assert!(lbl_bes.starts_with("Bessel f(0,1)"));
    assert!(
        (f_bes - 0.844).abs() < 0.02,
        "Expected Bessel J01 ~ 0.844 Hz, got {f_bes}"
    );
}

#[test]
fn test_atlas_ridge_click_locks_within_1mhz_snaps_antinode_and_boosts_energy_50x() {
    let mut tool = ChladniWaveTankTool::default();
    tool.apply_challenge(Some(CHALLENGE_CHLADNI_RESONANCE))
        .expect("activate Chladni challenge");
    tool.state.driver_frequency = challenge::CHALLENGE_CHLADNI_BASELINE_FREQ;
    tool.state.driver_x = challenge::CHALLENGE_CHLADNI_BASELINE_X;
    tool.state.driver_y = challenge::CHALLENGE_CHLADNI_BASELINE_Y;
    assert!(!tool.evaluate_challenge().unwrap().passed);

    // Click slightly off the (3, 3) Neumann biharmonic plate ridge (+0.02 Hz)
    let (modes, count) = tool.active_eigenmodes();
    let target = modes[..count]
        .iter()
        .copied()
        .find(|m| m.m == 3 && m.n == 3)
        .expect("surviving (3, 3) Neumann mode");
    tool.apply_atlas_teleport(
        AtlasTeleportEvent {
            x_key: "driver_frequency",
            x_val: target.frequency_hz + 0.02,
            y_key: "plate_stiffness",
            y_val: 0.0004,
            regime_label: "Locked Standing Eigenmode",
        },
        None,
    );
    assert!(
        (tool.state.driver_frequency - target.frequency_hz).abs() <= 0.001,
        "Atlas ridge click must lock driver_frequency within 0.001 Hz (got {}, target {})",
        tool.state.driver_frequency,
        target.frequency_hz
    );
    let snapped_amp = evaluate_eigenmode_amplitude(
        &target,
        tool.state.driver_x,
        tool.state.driver_y,
        DRUMHEAD_RADIUS_NORM,
    )
    .abs();
    assert!(
        snapped_amp >= 0.75 && (tool.state.driver_y - 0.50).abs() > 0.05,
        "Driver must snap off y = 0.50 nodal line onto an antinode (|Φ| = {snapped_amp})"
    );
    assert!(
        tool.evaluate_challenge().unwrap().passed,
        "Challenge must pass after antinodal ridge lock"
    );
    let _ = chladni_parameter_constraints();
}

#[test]
fn test_chladni_antinodal_lock_boosts_steady_state_energy_50x_and_persists_atlas() {
    let mut tool = ChladniWaveTankTool::default();
    tool.apply_preset(TankPreset::SquareChladniPlate);
    tool.state.medium = Wave2DMedium::AcousticWave;
    tool.state.boundary = BoundaryMode2D::Dirichlet;
    tool.state.clamp_layout = ClampLayout::CenterPin;
    tool.state.wave_speed = 2.0;
    tool.state.plate_stiffness = 0.0;
    tool.state.damping = 0.04;
    tool.state.time_step = 0.020;
    tool.state.show_sand_particles = false;
    tool.rebuild_domain_and_warmup(0);
    tool.solver.clear_wave_state();

    // Off-resonant nodal baseline: f = 0.45 Hz on the y = 0.50 nodal line
    let (s_modes, s_count) = tool.active_eigenmodes();
    let sq_mode = s_modes[..s_count]
        .iter()
        .copied()
        .find(|m| m.m == 2 && m.n == 2)
        .expect("surviving Dirichlet (2, 2) mode");
    tool.state.driver_frequency = 0.45;
    tool.state.driver_x = 0.25;
    tool.state.driver_y = 0.50;
    for _ in 0..220 {
        tool.advance_physics_step();
    }
    let off_energy: f64 = tool.solver.energy_field().iter().sum();

    // Teleport onto the (2, 2) dispersion ridge in ("driver_frequency", "wave_speed")
    tool.apply_atlas_teleport(
        AtlasTeleportEvent {
            x_key: "driver_frequency",
            x_val: sq_mode.frequency_hz + 0.03,
            y_key: "wave_speed",
            y_val: 2.0,
            regime_label: "Locked Standing Eigenmode",
        },
        None,
    );
    assert!((tool.state.driver_frequency - sq_mode.frequency_hz).abs() <= 0.001);
    assert!((tool.state.driver_y - 0.50).abs() > 0.10);
    for _ in 0..220 {
        tool.advance_physics_step();
    }
    let locked_energy: f64 = tool.solver.energy_field().iter().sum();
    let ratio = locked_energy / off_energy.max(1.0e-15);
    assert!(
        ratio > 50.0,
        "Expected > 50x steady-state wave energy gain after antinodal lock, got {ratio:.1}x"
    );
}

#[test]
fn test_chladni_fft_bin_lock_and_atlas_json_roundtrip() {
    let mut tool = ChladniWaveTankTool::default();
    let (modes, count) = tool.active_eigenmodes();
    let target = modes[..count].iter().copied().find(|m| m.m >= 2).unwrap();
    let k_bin = target.fft_radial_bin(GRID_SIZE, tool.domain_physical_length());
    tool.state.driver_y = 0.50;
    let locked_fft = tool
        .lock_nearest_fft_eigenmode(k_bin + 0.25, None)
        .expect("FFT bin click must lock nearest eigenmode");
    assert!((tool.state.driver_frequency - locked_fft.frequency_hz).abs() <= 0.001);

    let default_json = tool.save_state().unwrap();
    assert!(!default_json.contains("atlas_visible"));
    tool.atlas.set_visible(true);
    tool.atlas
        .set_axis_keys("driver_x", "driver_y")
        .expect("spatial axes");
    let open_json = tool.save_state().unwrap();
    assert!(open_json.contains("atlas_visible"));
    let mut restored = ChladniWaveTankTool::default();
    restored.try_load_state(&open_json).unwrap();
    assert!(restored.atlas.is_visible());
    assert_eq!(restored.atlas.axis_keys(), ("driver_x", "driver_y"));
    let _ = chladni_parameter_constraints();
}
