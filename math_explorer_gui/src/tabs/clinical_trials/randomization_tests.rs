#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;

#[test]
fn test_legacy_block_state_roundtrip_and_seeded_stratified_export() {
    let mut tool = RandomizationTool::default();
    let legacy = r#"{"randomization_type":"Block","n_subjects":24,"block_size":6}"#;
    tool.try_load_state(legacy).expect("legacy block state loads");
    assert_eq!(tool.save_state().as_deref(), Some(legacy));
    assert_eq!(tool.schedule.len(), 24);

    let stratified = r#"{"randomization_type":"Stratified","n_subjects":24,"block_size":4,"seed":77,"strata_spec":"SiteA:8, SiteB:12"}"#;
    tool.try_load_state(stratified)
        .expect("stratified state loads");
    assert_eq!(tool.schedule.len(), 20);
    let csv = format_schedule_csv(&tool.schedule, tool.seed);
    assert!(csv.contains("SiteA"));
    assert!(csv.contains("SiteB"));
    assert_eq!(tool.export_datasets().len(), 3);
}

#[test]
fn test_randomization_seed_counter_produces_distinct_allocations() {
    let mut tool = RandomizationTool {
        randomization_type: RandomizationType::Simple,
        n_subjects: 40,
        ..Default::default()
    };
    tool.allocate_subjects();
    let first = tool.assignments.clone();
    assert_eq!(tool.seed_counter, 1);
    assert_eq!(first.len(), 40);

    tool.allocate_subjects();
    let second = tool.assignments.clone();
    assert_eq!(tool.seed_counter, 2);
    assert_ne!(
        first, second,
        "Consecutive Allocate Subjects clicks must advance seed_counter"
    );
}

#[test]
fn test_randomization_legacy_and_stratified_state_roundtrip() {
    let mut tool = RandomizationTool::default();
    let legacy = r#"{"randomization_type":"Block","n_subjects":24,"block_size":6}"#;
    tool.try_load_state(legacy).unwrap();
    assert_eq!(tool.save_state().as_deref(), Some(legacy));

    tool.randomization_type = RandomizationType::Stratified;
    tool.simpson_lab.config.mild_treatment_share = 0.35;
    tool.simpson_lab.apply_allocation_sliders();
    let saved = tool.save_state().unwrap();

    let mut restored = RandomizationTool::default();
    restored.try_load_state(&saved).unwrap();
    assert_eq!(restored.randomization_type, RandomizationType::Stratified);
    assert!((restored.simpson_lab.config.mild_treatment_share - 0.35).abs() < 1e-9);
}

// [cite:clinical_trials_statistics]
