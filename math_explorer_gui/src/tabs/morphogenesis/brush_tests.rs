#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::brush::{apply_mask_rle_to_kinetics, apply_morpho_cells_to_system, MorphoBrushMode};
use super::*;
use crate::async_sim::{SimCommand, SimulationRunner};
use crate::framework::{decode_u8_grid, encode_u8_grid, InteractiveTool, RasterBrushStroke};
use crate::url_state::{encode_share_fragment, parse_share_location, URL_JSON_BOUNDS};
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
    let initial_u_center = runner.system.u()[12 * 24 + 12];
    apply_morpho_cells_to_system(
        &mut runner.system,
        24,
        24,
        &[(12, 12)],
        MorphoBrushMode::ActivatorSeed,
    );
    assert!(runner.system.u()[12 * 24 + 12] > initial_u_center + 0.5);

    // Stamp left half with StripeMask (1) and right half with SpotMask (2)
    let mut mask = vec![0u8; 24 * 24];
    for y in 0..24 {
        for x in 0..24 {
            mask[y * 24 + x] = if x < 12 { 1 } else { 2 };
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
        .flat_map(|y| (0..10).map(move |x| y * 24 + x))
        .map(|i| runner.system.u()[i])
        .sum::<f64>()
        / 240.0;
    let right_mean: f64 = (0..24)
        .flat_map(|y| (14..24).map(move |x| y * 24 + x))
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
    assert_eq!(restored_mask[10 * 100 + 10], 2);
    assert_eq!(restored_mask[10 * 100 + 11], 2);
    assert_eq!(restored_mask[10 * 100 + 12], 2);

    restored.clear_spatial_mask(None);
    assert!(restored.params.read().unwrap().mask_rle.is_none());
}

#[test]
fn test_minimap_curves_and_regime_classification_across_themes() {
    use super::minimap::{
        atlas_ranges, build_atlas_image, classify_gray_scott, classify_schnakenberg,
        gray_scott_fold_k, gray_scott_hopf_f, GrayScottMinimapRegime, SchnakenbergRegime,
    };
    use crate::presets::MorphogenesisKineticsMode;

    assert_eq!(
        classify_schnakenberg(0.1, 0.9, 1.0, 100.0),
        SchnakenbergRegime::TuringUnstable
    );
    assert_eq!(
        classify_schnakenberg(0.02, 0.4, 1.0, 1.0),
        SchnakenbergRegime::HopfOscillatory
    );
    assert_eq!(
        classify_schnakenberg(f64::NAN, 0.9, 1.0, 100.0),
        SchnakenbergRegime::UniformStable
    );
    assert!(gray_scott_fold_k(0.04).is_some_and(|k| (k - 0.06).abs() < 1e-12));
    assert!(gray_scott_fold_k(0.30).is_none());
    assert!(gray_scott_hopf_f(0.05).is_some_and(|f| f > 0.0 && f < 0.0625));
    assert!(gray_scott_hopf_f(0.08).is_none());

    assert_eq!(
        classify_gray_scott(0.0367, 0.0649, 0.16, 0.08),
        GrayScottMinimapRegime::SubcriticalPattern
    );
    assert_eq!(
        classify_gray_scott(0.030, 0.055, 0.16, 0.08),
        GrayScottMinimapRegime::BifurcationUnstable
    );
    assert_eq!(
        classify_gray_scott(0.01, 0.08, 0.16, 0.08),
        GrayScottMinimapRegime::UniformStable
    );

    for mode in [
        MorphogenesisKineticsMode::Schnakenberg,
        MorphogenesisKineticsMode::GrayScott,
    ] {
        let (xr, yr) = atlas_ranges(mode);
        assert!(xr.1 > xr.0 && yr.1 > yr.0);
        let dark = build_atlas_image(mode, 0.16, 0.08, false);
        let hc = build_atlas_image(mode, 0.16, 0.08, true);
        assert_eq!(dark.size, [96, 72]);
        assert_ne!(dark.pixels, hc.pixels);
    }
}

fn assert_500_steps_cfl_zero_alloc(
    mode: crate::presets::MorphogenesisKineticsMode,
    ab: (f64, f64),
    fk: (f64, f64),
    diff: (f64, f64),
    dt: f64,
) {
    let params = Arc::new(std::sync::RwLock::new(MorphogenesisSharedParams {
        a: ab.0,
        b: ab.1,
        d_u: diff.0,
        d_v: diff.1,
        dt,
        high_contrast: false,
        theme_mode: crate::app::ThemeMode::Dark,
        heatmap_mode: MorphogenesisHeatmapMode::ActiveUb,
        mask_rle: None,
    }));
    let ks = Arc::new(std::sync::RwLock::new(
        super::kinetics::MorphogenesisKineticsState {
            mode,
            auto_dt: true,
            f: fk.0,
            k: fk.1,
        },
    ));
    let mask = Arc::new(std::sync::RwLock::new(Vec::new()));
    let mut runner = MorphogenesisTool::create_kinetics_runner(&params, &ks, &mask, 24, 24);
    runner.step();
    {
        let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        verified_engine::engine::VerifiedEngine::run_verified(|| {
            for _ in 0..499 {
                runner.step();
            }
        });
    }
    assert_eq!(runner.step_count, 500);
    assert!(crate::framework::recovery::FiniteStateGuard::is_finite_state(&runner));
}

#[test]
fn test_500_step_cfl_stability_and_gray_scott_presets() {
    use crate::presets::MorphogenesisKineticsMode::{GrayScott, Schnakenberg};
    assert_500_steps_cfl_zero_alloc(
        Schnakenberg,
        (0.1, 0.9),
        (0.0367, 0.0649),
        (1.0, 200.0),
        0.1,
    );
    assert_500_steps_cfl_zero_alloc(GrayScott, (0.1, 0.9), (0.0367, 0.0649), (0.16, 0.08), 1.0);
    assert_500_steps_cfl_zero_alloc(GrayScott, (0.1, 0.9), (0.0545, 0.0620), (0.16, 0.08), 1.0);
    assert_500_steps_cfl_zero_alloc(GrayScott, (0.1, 0.9), (0.030, 0.060), (0.16, 0.08), 1.0);

    let mut tool = MorphogenesisTool::default();
    for preset in [
        presets::PRESET_GRAY_SCOTT_MITOSIS,
        presets::PRESET_GRAY_SCOTT_CORAL,
        presets::PRESET_GRAY_SCOTT_SOLITONS,
    ] {
        tool.import_preset_json(preset)
            .expect("valid Gray-Scott preset");
        assert!(tool.pending_initial_start);
    }
}

#[test]
fn test_gray_scott_brush_nucleation_erase_and_auto_dt_url_roundtrip() {
    use math_commons::math_kernel::types::flatten_2d_index;

    let mut tool = MorphogenesisTool::default();
    tool.apply_gray_scott_preset(0.0367, 0.0649);
    tool.kinetics_state.write().unwrap().auto_dt = false;

    let idx = flatten_2d_index(5, 5, tool.width);
    assert!((tool.local_runner.system.u()[idx] - 1.0).abs() < 1e-12);
    assert_eq!(tool.local_runner.system.v()[idx], 0.0);

    tool.brush_mode = MorphoBrushMode::ActivatorSeed;
    tool.apply_brush_stroke(
        RasterBrushStroke {
            cells: vec![(5, 5)],
            secondary: false,
            via_keyboard: true,
            focus_cell: (5, 5),
        },
        None,
    );
    assert!(tool.local_runner.system.u()[idx] <= 0.50);
    assert!(tool.local_runner.system.v()[idx] >= 0.25);

    tool.apply_brush_stroke(
        RasterBrushStroke {
            cells: vec![(5, 5)],
            secondary: true,
            via_keyboard: true,
            focus_cell: (5, 5),
        },
        None,
    );
    assert!((tool.local_runner.system.u()[idx] - 1.0).abs() < 1e-12);
    assert_eq!(tool.local_runner.system.v()[idx], 0.0);

    let saved = tool.save_state().expect("save_state");
    assert!(saved.contains("\"auto_dt\":false"));
    let fragment = encode_share_fragment("Morphogenesis", Some("MorphogenesisTool"), Some(&saved))
        .expect("encode share fragment");
    let decoded = parse_share_location(&fragment, "")
        .expect("decode share")
        .expect("share params");
    let mut restored = MorphogenesisTool::default();
    restored
        .try_load_state(decoded.state_json.as_deref().unwrap())
        .expect("restore Gray-Scott state");
    assert!(!restored.kinetics_state.read().unwrap().auto_dt);
}

fn click_minimap_point(
    ctx: &egui::Context,
    tab: &mut MorphogenesisTab,
    title: &str,
    rel_xy: (f32, f32),
) {
    let mut frame = eframe::Frame::_new_kittest();
    let raw = |events| egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1400.0, 900.0),
        )),
        events,
        ..Default::default()
    };
    let out = ctx.run(raw(Vec::new()), |ctx| {
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show(ctx, |ui| tab.show_inside(ui, &mut frame));
    });
    let ak = out.platform_output.accesskit_update.as_ref().unwrap();
    let b = ak
        .nodes
        .iter()
        .find(|(_, n)| n.label() == Some(title))
        .and_then(|(_, n)| n.bounds())
        .unwrap_or_else(|| panic!("Minimap '{title}' not found"));
    let pos = egui::pos2(
        (b.x0 + (b.x1 - b.x0) * f64::from(rel_xy.0)) as f32,
        (b.y0 + (b.y1 - b.y0) * f64::from(rel_xy.1)) as f32,
    );
    let btn = |pressed| egui::Event::PointerButton {
        pos,
        button: egui::PointerButton::Primary,
        pressed,
        modifiers: egui::Modifiers::NONE,
    };
    let _ = ctx.run(
        raw(vec![egui::Event::PointerMoved(pos), btn(true), btn(false)]),
        |ctx| {
            egui::CentralPanel::default()
                .frame(egui::Frame::NONE)
                .show(ctx, |ui| tab.show_inside(ui, &mut frame));
        },
    );
}

#[test]
fn test_sidebar_minimap_click_updates_schnakenberg_and_gray_scott_params() {
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let mut tab = MorphogenesisTab::default();

    click_minimap_point(
        &ctx,
        &mut tab,
        "2D Turing Stability Map (a, b)",
        (0.25, 0.30),
    );
    {
        let tool = tab
            .framework
            .active_tool
            .as_ref()
            .and_then(|t| t.as_any()?.downcast_ref::<MorphogenesisTool>())
            .unwrap();
        let p = tool.params.read().unwrap();
        assert!((p.a - 0.25).abs() < 0.03);
        assert!((p.b - 1.40).abs() < 0.06);
    }

    tab.framework
        .active_tool
        .as_mut()
        .and_then(|t| t.as_any_mut()?.downcast_mut::<MorphogenesisTool>())
        .unwrap()
        .apply_gray_scott_preset(0.0367, 0.0649);
    click_minimap_point(
        &ctx,
        &mut tab,
        "2D Gray-Scott Bifurcation Map (F, k)",
        (0.50, 0.50),
    );
    let tool = tab
        .framework
        .active_tool
        .as_ref()
        .and_then(|t| t.as_any()?.downcast_ref::<MorphogenesisTool>())
        .unwrap();
    let ks = tool.kinetics_state.read().unwrap();
    assert!((ks.f - 0.045).abs() < 0.005);
    assert!((ks.k - 0.050).abs() < 0.005);
}
