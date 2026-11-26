#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::brush::{apply_mask_rle_to_kinetics, apply_morpho_cells_to_system, MorphoBrushMode};
use super::minimap::{
    build_atlas_image, build_atlas_image_for_theme, classify_gray_scott, classify_schnakenberg,
    gray_scott_fold_k, gray_scott_hopf_f, GrayScottMinimapRegime, SchnakenbergRegime,
};
use super::presets::{
    PRESET_GRAY_SCOTT_CORAL, PRESET_GRAY_SCOTT_MITOSIS, PRESET_GRAY_SCOTT_SOLITONS,
};
use super::*;
use crate::app::ThemeMode;
use crate::async_sim::{SimCommand, SimWorkerState, SimulationRunner};
use crate::framework::{decode_u8_grid, encode_u8_grid, InteractiveTool, RasterBrushStroke};
use crate::presets::MorphogenesisKineticsMode;
use crate::url_state::{encode_share_fragment, parse_share_location, URL_JSON_BOUNDS};
use math_commons::math_kernel::types::flatten_2d_index;
use std::collections::HashSet;
use std::sync::atomic::Ordering;
use std::sync::Arc;

#[test]
fn test_morphogenesis_chemical_seed_and_hybrid_spot_stripe_mask_evolution() {
    let params = Arc::new(std::sync::RwLock::new(MorphogenesisSharedParams {
        a: 0.1,
        b: 0.9,
        d_u: 1.0,
        d_v: 20.0,
        dt: 0.01,
        high_contrast: false,
        theme_mode: crate::app::ThemeMode::Dark,
        heatmap_mode: MorphogenesisHeatmapMode::ActiveUb,
        mask_rle: None,
    }));
    let mut runner = MorphogenesisTab::create_runner(&params, 24, 24);
    let center = flatten_2d_index(12, 12, 24);
    let initial_u_center = runner.system.u()[center];
    apply_morpho_cells_to_system(
        &mut runner.system,
        24,
        24,
        &[(12, 12)],
        MorphoBrushMode::ActivatorSeed,
    );
    assert!(runner.system.u()[center] > initial_u_center + 0.5);

    // Stamp left half with StripeMask (1) and right half with SpotMask (2)
    let mut mask = vec![0u8; 24 * 24];
    for y in 0..24 {
        for x in 0..24 {
            mask[flatten_2d_index(x, y, 24)] = if x < 12 { 1 } else { 2 };
        }
    }
    let rle = encode_u8_grid(24, 24, &mask, 2).expect("encode hybrid mask");
    apply_mask_rle_to_kinetics(&mut runner.system.kinetics, 24, 24, Some(&rle));
    assert!(runner.system.kinetics.has_spatial_mask());
    assert_eq!(runner.system.kinetics.rates_at(0), brush::STRIPE_MASK_RATES);
    assert_eq!(runner.system.kinetics.rates_at(23), brush::SPOT_MASK_RATES);

    // Verify zero heap allocations during masked step under VerifiedAllocator
    runner.step();
    {
        let _alloc_guard = crate::VERIFIED_ALLOC_TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        verified_engine::engine::VerifiedEngine::run_verified(|| {
            runner.step();
        });
    }

    for _ in 0..25 {
        runner.step();
    }
    let left_mean: f64 = (0..24)
        .flat_map(|y| (0..10).map(move |x| flatten_2d_index(x, y, 24)))
        .map(|i| runner.system.u()[i])
        .sum::<f64>()
        / 240.0;
    let right_mean: f64 = (0..24)
        .flat_map(|y| (14..24).map(move |x| flatten_2d_index(x, y, 24)))
        .map(|i| runner.system.u()[i])
        .sum::<f64>()
        / 240.0;
    assert!(
        (left_mean - right_mean).abs() > 1e-4,
        "Spatially masked stripe vs spot halves should evolve distinct activator concentrations"
    );
}

#[test]
fn test_morphogenesis_mask_rle_persistence_url_share_and_clear() {
    let mut tool = MorphogenesisTool::default();
    let _ = tool.worker.send_command(SimCommand::Pause);
    tool.worker.wait_until_paused();

    tool.brush_mode = MorphoBrushMode::SpotMask;
    tool.apply_brush_stroke(
        RasterBrushStroke {
            cells: vec![(10, 10), (11, 10), (12, 10)],
            secondary: false,
            via_keyboard: true,
            focus_cell: (12, 10),
        },
        None,
    );

    let saved = tool.save_state().expect("save_state");
    assert!(saved.contains("mask_rle"));
    URL_JSON_BOUNDS
        .validate_json(&saved)
        .expect("Morphogenesis masked state fits URL_JSON_BOUNDS");

    let fragment = encode_share_fragment("Morphogenesis", Some("MorphogenesisTool"), Some(&saved))
        .expect("URL fragment encode");
    let decoded_share = parse_share_location(&fragment, "")
        .expect("URL fragment decode")
        .expect("present share params");

    let mut restored = MorphogenesisTool::default();
    restored
        .try_load_state(decoded_share.state_json.as_deref().unwrap())
        .expect("restore masked state");
    let restored_mask = decode_u8_grid(
        restored
            .params
            .read()
            .unwrap()
            .mask_rle
            .as_deref()
            .expect("restored mask_rle"),
        100,
        100,
        2,
    )
    .expect("decode restored mask");
    assert_eq!(restored_mask[flatten_2d_index(10, 10, 100)], 2);
    assert_eq!(restored_mask[flatten_2d_index(11, 10, 100)], 2);
    assert_eq!(restored_mask[flatten_2d_index(12, 10, 100)], 2);

    restored.clear_spatial_mask(None);
    assert!(restored.params.read().unwrap().mask_rle.is_none());
}

#[test]
fn test_minimap_analytical_regimes_fold_hopf_and_themes() {
    assert_eq!(
        classify_schnakenberg(0.1, 0.9, 1.0, 40.0),
        SchnakenbergRegime::TuringUnstable
    );
    assert_eq!(
        classify_schnakenberg(0.05, 0.4, 1.0, 1.0),
        SchnakenbergRegime::HopfOscillatory
    );
    assert_eq!(
        classify_schnakenberg(0.8, 0.8, 1.0, 1.0),
        SchnakenbergRegime::UniformStable
    );
    assert_eq!(
        classify_schnakenberg(f64::NAN, 0.9, 1.0, 40.0),
        SchnakenbergRegime::UniformStable
    );

    let f_sample = 0.04;
    let k_fold = gray_scott_fold_k(f_sample).expect("fold k");
    assert!((4.0 * (f_sample + k_fold).powi(2) - f_sample).abs() < 1e-12);
    let k_sample = 0.05;
    let f_hopf = gray_scott_hopf_f(k_sample).expect("hopf f");
    assert!(((f_hopf + k_sample).powi(2) - f_hopf * k_sample.sqrt()).abs() < 1e-12);

    for &(f, k) in &[(0.0367, 0.0649), (0.030, 0.060)] {
        assert_eq!(
            classify_gray_scott(f, k, 0.16, 0.08),
            GrayScottMinimapRegime::SubcriticalPattern
        );
    }
    for &(f, k) in &[(0.0545, 0.0620), (0.025, 0.052)] {
        assert_eq!(
            classify_gray_scott(f, k, 0.16, 0.08),
            GrayScottMinimapRegime::BifurcationUnstable
        );
    }
    assert_eq!(
        classify_gray_scott(0.08, 0.068, 0.16, 0.08),
        GrayScottMinimapRegime::UniformStable
    );

    for mode in [
        MorphogenesisKineticsMode::Schnakenberg,
        MorphogenesisKineticsMode::GrayScott,
    ] {
        let hc_img = build_atlas_image(mode, 1.0, 40.0, true);
        assert_eq!(hc_img.size, [96, 72]);
        for theme in [ThemeMode::Dark, ThemeMode::Light, ThemeMode::HighContrast] {
            let img = build_atlas_image_for_theme(mode, 0.16, 0.08, theme);
            let distinct: HashSet<eframe::egui::Color32> = img.pixels.iter().copied().collect();
            assert!(distinct.len() >= 2);
        }
    }
}

#[test]
fn test_cfl_substepping_500_steps_zero_alloc_across_four_presets() {
    let presets: [Option<&str>; 4] = [
        None,
        Some(PRESET_GRAY_SCOTT_MITOSIS),
        Some(PRESET_GRAY_SCOTT_CORAL),
        Some(PRESET_GRAY_SCOTT_SOLITONS),
    ];
    for maybe_preset in presets {
        let mut tool = MorphogenesisTool::default();
        if let Some(json) = maybe_preset {
            tool.import_preset_json(json).expect("preset loads");
        }
        tool.width = 24;
        tool.height = 24;
        let mut runner = tool.create_live_runner();
        runner.step();
        {
            let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            verified_engine::engine::VerifiedEngine::run_verified(|| {
                runner.step();
            });
        }
        for _ in 0..498 {
            runner.step();
        }
        assert!(runner.is_finite_state());
        let active = counterfactual::active_morphogen_slice(&runner.system);
        assert!(grid_u_variance(active) > 1e-6);
    }
}

#[test]
fn test_gray_scott_brush_nucleation_erase_minimap_and_auto_dt_roundtrip() {
    let mut tool = MorphogenesisTool::default();
    tool.import_preset_json(PRESET_GRAY_SCOTT_SOLITONS)
        .expect("solitons preset loads");
    assert_eq!(tool.worker.state(), SimWorkerState::Ready);
    assert!(tool.pending_initial_start);

    let cell = (10, 10);
    let idx = flatten_2d_index(cell.0, cell.1, tool.width);
    assert!((tool.local_runner.system.u()[idx] - 1.0).abs() < 1e-12);
    assert_eq!(tool.local_runner.system.v()[idx], 0.0);

    tool.brush_mode = MorphoBrushMode::ActivatorSeed;
    tool.apply_brush_stroke(
        RasterBrushStroke {
            cells: vec![cell],
            secondary: false,
            via_keyboard: false,
            focus_cell: cell,
        },
        None,
    );
    assert!(tool.local_runner.system.u()[idx] <= 0.50 + 1e-12);
    assert!(tool.local_runner.system.v()[idx] >= 0.25 - 1e-12);

    tool.apply_brush_stroke(
        RasterBrushStroke {
            cells: vec![cell],
            secondary: true,
            via_keyboard: false,
            focus_cell: cell,
        },
        None,
    );
    assert!((tool.local_runner.system.u()[idx] - 1.0).abs() < 1e-12);
    assert_eq!(tool.local_runner.system.v()[idx], 0.0);

    tool.apply_minimap_selection(MorphogenesisKineticsMode::GrayScott, 0.0545, 0.0620);
    assert!((tool.kinetics_state.read().unwrap().f - 0.0545).abs() < 1e-12);
    tool.auto_dt.store(false, Ordering::Relaxed);
    let saved = tool.save_state().expect("save_state");
    assert!(saved.contains("\"auto_dt\":false"));
    let mut restored = MorphogenesisTool::default();
    restored
        .try_load_state(&saved)
        .expect("restore auto_dt false");
    assert!(!restored.auto_dt.load(Ordering::Relaxed));
}
