#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::tabs::clinical_trials::randomization_schedule::{
    format_schedule_csv, format_schedule_json,
};

#[test]
fn test_legacy_block_state_roundtrip_and_seeded_stratified_export() {
    let mut tool = RandomizationTool::default();
    let legacy = r#"{"randomization_type":"Block","n_subjects":24,"block_size":6}"#;
    tool.try_load_state(legacy)
        .expect("legacy block state loads");
    assert_eq!(tool.save_state().as_deref(), Some(legacy));
    assert_eq!(tool.schedule.len(), 24);

    let stratified = r#"{"randomization_type":"Stratified","n_subjects":24,"block_size":4,"seed":77,"strata_spec":"SiteA:8, SiteB:12"}"#;
    tool.try_load_state(stratified)
        .expect("stratified state loads");
    assert_eq!(tool.schedule.len(), 20);
    let csv = format_schedule_csv(
        tool.randomization_type.label(),
        tool.effective_seed(),
        &tool.schedule,
    );
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

#[test]
fn test_schedule_retention_and_exports_across_all_three_modes() {
    let mut tool = RandomizationTool::default();
    assert!(tool.export_capabilities().figure && tool.export_capabilities().dataset);
    assert_eq!(tool.schedule.len(), 10);
    assert_eq!(tool.schedule[0].subject_id, "SUBJ-0001");
    assert_eq!(tool.export_datasets().len(), 3);

    // Block mode: exact balance at multiples of block_size
    tool.randomization_type = RandomizationType::Block;
    tool.n_subjects = 16;
    tool.block_size = 4;
    tool.allocate_subjects();
    assert_eq!(tool.schedule.len(), 16);
    assert_eq!(tool.schedule[0].stratum, "Block-1");
    assert_eq!(tool.schedule[4].stratum, "Block-2");
    let csv = format_schedule_csv(
        tool.randomization_type.label(),
        tool.seed_counter,
        &tool.schedule,
    );
    assert!(csv.contains("1,SUBJ-0001,Block-1,"));
    let json = format_schedule_json(
        tool.randomization_type.label(),
        tool.seed_counter,
        &tool.schedule,
    );
    assert!(json.contains("\"treatment_subjects\": 8"));
    assert!(json.contains("\"control_subjects\": 8"));

    // Stratified mode: retains M-i and S-i IDs
    tool.randomization_type = RandomizationType::Stratified;
    tool.simpson_lab.config.stratum_size = 24;
    tool.allocate_subjects();
    let sched = tool.active_schedule();
    assert_eq!(sched.len(), 48);
    assert_eq!(sched[0].subject_id, "M-0");
    assert_eq!(sched[1].subject_id, "S-0");
}

#[cfg(target_arch = "wasm32")]
mod wasm_tests {
    use super::*;
    use wasm_bindgen_test::wasm_bindgen_test;

    #[wasm_bindgen_test]
    fn test_wasm_randomization_modes_ui_and_schedule_json() {
        let mut tool = RandomizationTool::default();
        let ctx = egui::Context::default();
        for mode in [
            RandomizationType::Simple,
            RandomizationType::Block,
            RandomizationType::Stratified,
        ] {
            tool.randomization_type = mode;
            tool.allocate_subjects();
            let raw_input = egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::Vec2::new(1024.0, 768.0),
                )),
                ..Default::default()
            };
            let _ = ctx.run(raw_input, |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| {
                    tool.show_inside(ui);
                });
            });
            let json = format_schedule_json(
                tool.randomization_type.label(),
                tool.seed_counter,
                tool.active_schedule(),
            );
            assert!(json.contains("\"schedule\""));
        }
    }
}

// [cite:clinical_trials_statistics]
