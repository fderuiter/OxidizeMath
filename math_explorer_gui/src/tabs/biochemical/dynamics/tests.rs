#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::BiochemicalDynamicsTool;
use super::presets::{PRESET_ALLOSTERIC_HILL, PRESET_AUTOCATALYTIC_WAVE};
use crate::framework::InteractiveTool;

#[test]
fn test_biochemical_dynamics_cooperativity_and_rd_stepping() {
    let mut tool = BiochemicalDynamicsTool::default();
    let (mm_pts, hill_pts) = tool.sample_kinetics_curves();
    assert_eq!(mm_pts.len(), hill_pts.len());

    // At [S] = K_m, both Michaelis-Menten and Hill equal V_max / 2; for [S] > K_m and n > 1, Hill > MM.
    let high_idx = mm_pts.len() - 1;
    assert!(hill_pts[high_idx][1] > mm_pts[high_idx][1]);

    let before_u = tool.rd_system.state.species(0).to_vec();
    tool.step_simulation();
    assert_ne!(tool.rd_system.state.species(0), before_u.as_slice());
    assert!(tool.has_finite_parameters());

    tool.import_preset_json(PRESET_ALLOSTERIC_HILL).unwrap();
    assert!((tool.state.hill_n - 4.0).abs() < 1e-12);

    tool.import_preset_json(PRESET_AUTOCATALYTIC_WAVE).unwrap();
    assert_eq!(tool.state.grid_points, 128);
}

#[test]
fn test_biochemical_dynamics_state_bounds_and_exports() {
    let tool = BiochemicalDynamicsTool::default();
    let saved = tool.save_state().unwrap();
    assert!(saved.len() < 4096);

    let mut restored = BiochemicalDynamicsTool::default();
    restored.try_load_state(&saved).unwrap();
    assert_eq!(tool.state, restored.state);
    assert!(restored.try_load_state(r#"{"hill_n": 99.0}"#).is_err());
    assert!(restored.try_load_state(r#"{"bad_field": 1}"#).is_err());

    let datasets = tool.export_datasets();
    assert_eq!(datasets.len(), 4);
}
