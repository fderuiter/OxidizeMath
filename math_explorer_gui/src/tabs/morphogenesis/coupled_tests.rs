#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::presets::{
    PRESET_GRAY_SCOTT_MITOSIS, PRESET_GRAY_SCOTT_SPIRAL_FLOW, PRESET_KARMAN_TURING_WAKE,
};
use super::*;
use crate::app::{contrast_ratio, ThemeMode};
use crate::async_sim::{SimCommand, SimulationRunner};
use crate::framework::recovery::FiniteStateGuard;
use crate::framework::InteractiveTool;
use eframe::egui::Color32;
use math_commons::math_kernel::types::flatten_2d_index;
use std::sync::{Arc, RwLock};

#[test]
fn test_legacy_presets_load_zero_velocity_and_advected_presets_roundtrip() {
    let mut tool = MorphogenesisTool::default();
    tool.import_preset_json(crate::starter_gallery::PRESET_TURING_SPOTS)
        .expect("PRESET_TURING_SPOTS must load");
    let spots_state: MorphogenesisToolState =
        serde_json::from_str(&tool.save_state().unwrap()).unwrap();
    assert_eq!(spots_state.flow_mode, MorphogenesisFlowMode::None);
    assert_eq!(spots_state.inlet_velocity, 0.0);
    assert_eq!(spots_state.circulation, 0.0);
    assert_eq!(
        spots_state.kinetics_mode,
        MorphogenesisKineticsMode::Schnakenberg
    );

    tool.import_preset_json(crate::starter_gallery::PRESET_TURING_STRIPES)
        .expect("PRESET_TURING_STRIPES must load");
    let stripes_state: MorphogenesisToolState =
        serde_json::from_str(&tool.save_state().unwrap()).unwrap();
    assert_eq!(stripes_state.flow_mode, MorphogenesisFlowMode::None);
    assert_eq!(stripes_state.inlet_velocity, 0.0);

    tool.import_preset_json(PRESET_GRAY_SCOTT_MITOSIS)
        .expect("PRESET_GRAY_SCOTT_MITOSIS must load");
    let mitosis_state: MorphogenesisToolState =
        serde_json::from_str(&tool.save_state().unwrap()).unwrap();
    assert_eq!(
        mitosis_state.kinetics_mode,
        MorphogenesisKineticsMode::GrayScott
    );
    assert_eq!(mitosis_state.flow_mode, MorphogenesisFlowMode::None);
    assert!((mitosis_state.f - 0.0367).abs() < 1e-9);
    assert!((mitosis_state.k - 0.0649).abs() < 1e-9);

    tool.import_preset_json(PRESET_KARMAN_TURING_WAKE)
        .expect("PRESET_KARMAN_TURING_WAKE must load");
    let karman_state: MorphogenesisToolState =
        serde_json::from_str(&tool.save_state().unwrap()).unwrap();
    assert_eq!(karman_state.flow_mode, MorphogenesisFlowMode::LbmVortexWake);
    assert_eq!(
        karman_state.overlay_mode,
        MorphogenesisOverlayMode::DualOverlay
    );

    tool.import_preset_json(PRESET_GRAY_SCOTT_SPIRAL_FLOW)
        .expect("PRESET_GRAY_SCOTT_SPIRAL_FLOW must load");
    let gs_state: MorphogenesisToolState =
        serde_json::from_str(&tool.save_state().unwrap()).unwrap();
    assert_eq!(gs_state.kinetics_mode, MorphogenesisKineticsMode::GrayScott);
    assert_eq!(gs_state.flow_mode, MorphogenesisFlowMode::PotentialFlow);
    assert!(tool.theory().theory_citation().contains("Pearson"));
}

#[test]
fn test_brush_sculpting_obstacles_reagents_and_clear_obstacles() {
    let params = Arc::new(RwLock::new(MorphogenesisSharedParams {
        brush_mode: MorphogenesisBrushMode::Obstacle,
        brush_radius: 2,
        ..MorphogenesisSharedParams::default()
    }));
    let mut runner = MorphogenesisTab::create_runner(&params, 30, 30);

    runner.process_command(SimCommand::ApplyBrush {
        cx: 12,
        cy: 14,
        r: 2,
        is_obstacle: true,
    });
    let center_idx = flatten_2d_index(12, 14, 30);
    assert!(runner.system.diffusion.obstacles()[center_idx]);
    #[cfg(feature = "physics")]
    assert!(runner.lbm.is_obstacle(12, 14));

    // Switch to Reagent brush and inject activator pulse away from obstacle
    params.write().unwrap().brush_mode = MorphogenesisBrushMode::Reagent;
    let reagent_idx = flatten_2d_index(22, 14, 30);
    let u_before = runner.system.u()[reagent_idx];
    runner.process_command(SimCommand::ApplyBrush {
        cx: 22,
        cy: 14,
        r: 2,
        is_obstacle: true,
    });
    assert!(runner.system.u()[reagent_idx] > u_before + 0.5);

    // Clear all obstacles
    runner.process_command(SimCommand::ClearObstacles);
    assert!(runner.system.diffusion.obstacles().iter().all(|&b| !b));
    #[cfg(feature = "physics")]
    assert!(!runner.lbm.is_obstacle(12, 14));
}

#[cfg(feature = "physics")]
#[test]
fn test_lbm_coupling_zero_alloc_step_and_live_peclet_damkohler() {
    let params = Arc::new(RwLock::new(MorphogenesisSharedParams {
        a: 0.12,
        b: 0.88,
        d_u: 0.4,
        d_v: 16.0,
        dt: 0.02,
        kinetics_mode: MorphogenesisKineticsMode::Schnakenberg,
        flow_mode: MorphogenesisFlowMode::LbmVortexWake,
        inlet_velocity: 0.09,
        viscosity: 0.018,
        overlay_mode: MorphogenesisOverlayMode::DualOverlay,
        ..MorphogenesisSharedParams::default()
    }));
    let mut runner = MorphogenesisTab::create_runner(&params, 40, 24);
    runner.step();

    {
        let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        verified_engine::engine::VerifiedEngine::run_verified(|| {
            for _ in 0..25 {
                runner.step();
            }
        });
    }
    assert!(FiniteStateGuard::is_finite_state(&runner));
    assert!(runner.peclet > 0.0);
    assert!(runner.damkohler > 0.0);
    assert!(runner.vorticity_buf.iter().any(|&w| w.abs() > 1e-5));

    let pe_low_inlet = runner.peclet;
    let da_low_inlet = runner.damkohler;
    params.write().unwrap().inlet_velocity = 0.18;
    runner.step();
    assert!(runner.peclet > pe_low_inlet);
    assert!(runner.damkohler < da_low_inlet);
}

#[cfg(feature = "physics")]
#[test]
fn test_potential_flow_gray_scott_zero_alloc_and_circulation_readouts() {
    let params = Arc::new(RwLock::new(MorphogenesisSharedParams {
        f: 0.037,
        k: 0.060,
        d_u: 0.20,
        d_v: 0.10,
        dt: 0.40,
        kinetics_mode: MorphogenesisKineticsMode::GrayScott,
        flow_mode: MorphogenesisFlowMode::PotentialFlow,
        inlet_velocity: 0.08,
        circulation: 2.0,
        overlay_mode: MorphogenesisOverlayMode::DualOverlay,
        ..MorphogenesisSharedParams::default()
    }));
    let mut runner = MorphogenesisTab::create_runner(&params, 40, 24);
    runner.step();
    let pe_circ_low = runner.peclet;
    params.write().unwrap().circulation = 10.0;
    runner.step();
    assert!(runner.peclet > pe_circ_low);

    {
        let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        verified_engine::engine::VerifiedEngine::run_verified(|| {
            runner.step();
        });
    }
    assert!(FiniteStateGuard::is_finite_state(&runner));
}

#[test]
fn test_overlays_support_dark_light_and_high_contrast_wcag_bounds() {
    let conc = vec![0.0, 0.25, 0.5, 0.75, 1.0, 0.1, 0.9, 0.4, 0.6];
    let vort = vec![-1.0, -0.5, 0.0, 0.5, 1.0, -0.8, 0.8, -0.2, 0.2];
    let obs = vec![false, false, false, false, true, false, false, false, false];

    for mode in [
        MorphogenesisOverlayMode::Concentration,
        MorphogenesisOverlayMode::Vorticity,
        MorphogenesisOverlayMode::DualOverlay,
    ] {
        let dark = render_morphogenesis_overlay(&conc, &vort, &obs, 3, 3, mode, ThemeMode::Dark);
        let light = render_morphogenesis_overlay(&conc, &vort, &obs, 3, 3, mode, ThemeMode::Light);
        let hc =
            render_morphogenesis_overlay(&conc, &vort, &obs, 3, 3, mode, ThemeMode::HighContrast);
        assert_eq!(dark.pixels.len(), 9);
        assert_eq!(light.pixels.len(), 9);
        assert_eq!(hc.pixels.len(), 9);
        if mode == MorphogenesisOverlayMode::Vorticity {
            assert_ne!(dark.pixels[2], light.pixels[2]);
        }
        for (idx, &px) in hc.pixels.iter().enumerate() {
            let ratio = contrast_ratio(px, Color32::BLACK);
            assert!(
                ratio >= 3.0,
                "HighContrast {mode:?} pixel {idx} ({px:?}) yielded {ratio:.2}:1 (< 3.0:1)"
            );
        }
    }
}
