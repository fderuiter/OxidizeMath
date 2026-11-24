#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::presets::{
    GuiDivergencePreset, GuiLSystemPreset, PRESET_GOLDEN_SUNFLOWER, PRESET_PARAMETRIC_CANOPY,
};
use super::PhyllotaxisLSystemTool;
use crate::framework::InteractiveTool;

#[test]
fn test_phyllotaxis_lsystem_golden_vs_rational_and_presets() {
    let mut tool = PhyllotaxisLSystemTool::default();
    assert!(tool.vogel.analysis().is_fibonacci_pair);
    assert!(!tool.vogel.analysis().has_radial_spokes);
    assert!(!tool.lsystem.segments().is_empty());

    tool.apply_divergence_preset(GuiDivergencePreset::RationalOneThird);
    assert!(tool.vogel.analysis().has_radial_spokes);
    assert_eq!(tool.vogel.analysis().primary_count, 3);

    tool.apply_lsystem_preset(GuiLSystemPreset::ParametricDichotomousTree);
    assert!(tool.lsystem.segments().len() > 20);

    tool.import_preset_json(PRESET_GOLDEN_SUNFLOWER).unwrap();
    assert_eq!(tool.state.primordia_count, 377);
    assert!(tool.vogel.analysis().is_fibonacci_pair);
    assert_eq!(
        (
            tool.vogel.analysis().primary_count,
            tool.vogel.analysis().secondary_count
        ),
        (34, 55)
    );

    tool.import_preset_json(PRESET_PARAMETRIC_CANOPY).unwrap();
    assert_eq!(tool.state.generations, 8);
}

#[test]
fn test_phyllotaxis_lsystem_url_state_bounds_and_exports() {
    let tool = PhyllotaxisLSystemTool::default();
    let json = tool.save_state().expect("must serialize");
    assert!(json.len() < 4096);

    let mut restored = PhyllotaxisLSystemTool::default();
    restored.try_load_state(&json).unwrap();
    assert_eq!(tool.state, restored.state);

    // Reject out-of-bounds primordia count (> 4096) and unknown fields
    assert!(restored
        .try_load_state(r#"{"primordia_count": 9000}"#)
        .is_err());
    assert!(restored.try_load_state(r#"{"unknown_key": 1}"#).is_err());

    let datasets = tool.export_datasets();
    assert!(datasets.len() >= 2);
    let prims = tool.export_plot_primitives();
    assert!(!prims.is_empty());

    let mut hill_tool = PhyllotaxisLSystemTool::default();
    hill_tool
        .import_preset_json(
            crate::tabs::morphogenesis::starter_presets::PRESET_PHYLLOTAXIS_HILL_SHOOT,
        )
        .unwrap();
    assert!(hill_tool.state.nutrient_enabled);
    let vel = hill_tool.nutrient_coupling().reaction_velocity();
    assert!(vel > 0.0 && vel.is_finite());
    let hill_json = hill_tool.save_state().expect("hill state serializes");
    assert!(hill_json.contains("nutrient_enabled"));
    let mut hill_restored = PhyllotaxisLSystemTool::default();
    hill_restored.try_load_state(&hill_json).unwrap();
    assert_eq!(hill_tool.state, hill_restored.state);
}
