#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::brush::{apply_mask_rle_to_kinetics, apply_morpho_cells_to_system, MorphoBrushMode};
use super::*;
use crate::async_sim::{SimCommand, SimulationRunner};
use crate::framework::{decode_u8_grid, encode_u8_grid, InteractiveTool, RasterBrushStroke};
use crate::url_state::{encode_share_fragment, parse_share_location, URL_JSON_BOUNDS};

#[test]
fn test_morphogenesis_chemical_seed_and_hybrid_spot_stripe_mask_evolution() {
    let params = Arc::new(std::sync::RwLock::new(MorphogenesisSharedParams {
        a: 0.1,
        b: 0.9,
        d_u: 1.0,
        d_v: 20.0,
        dt: 0.01,
        high_contrast: false,
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

    let fragment = encode_share_fragment(
        "Morphogenesis",
        Some("MorphogenesisTool"),
        Some(&saved),
    )
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
