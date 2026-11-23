#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::framework::SimulationFramework;
use crate::tabs::morphogenesis::MorphogenesisTab;
use crate::tabs::ExplorerTab;
use federated_registry::{global_registry, Severity};

fn inject_unknown_config_field(preset_json: &str) -> String {
    let mut val: serde_json::Value = serde_json::from_str(preset_json).expect("valid preset json");
    val["config"]
        .as_object_mut()
        .expect("config must be object")
        .insert(
            "__unknown_contract_field__".to_string(),
            serde_json::json!(999),
        );
    serde_json::to_string(&val).unwrap()
}

fn inject_unknown_metadata_field(preset_json: &str) -> String {
    let mut val: serde_json::Value = serde_json::from_str(preset_json).expect("valid preset json");
    val["metadata"]
        .as_object_mut()
        .expect("metadata must be object")
        .insert(
            "__unknown_meta_field__".to_string(),
            serde_json::json!("bad"),
        );
    serde_json::to_string(&val).unwrap()
}

fn replace_preset_domain(preset_json: &str, bad_domain: &str) -> String {
    let mut val: serde_json::Value = serde_json::from_str(preset_json).expect("valid preset json");
    val["metadata"]["domain"] = serde_json::json!(bad_domain);
    serde_json::to_string(&val).unwrap()
}

fn count_gallery_buttons(output: &egui::FullOutput) -> usize {
    let ak = output
        .platform_output
        .accesskit_update
        .as_ref()
        .expect("AccessKit update expected");
    ak.nodes
        .iter()
        .filter(|(_, node)| {
            node.role() == egui::accesskit::Role::Button
                && node.label().unwrap_or("").contains("Starter Gallery")
        })
        .count()
}

#[test]
fn test_all_57_starter_cards_launch_and_sync_framework_state() {
    let presets: Vec<&'static StarterPresetMetadata> = inventory::iter::<StarterPresetMetadata>
        .into_iter()
        .collect();
    assert_eq!(presets.len(), 85);

    for preset in presets {
        let json = preset.generate_preset();
        let tool_name = preset
            .tool_name
            .unwrap_or_else(|| panic!("Starter card '{}' must specify tool_name", preset.id));
        let mut framework = SimulationFramework::new(preset.domain);
        framework
            .apply_starter_card(preset.title, Some(tool_name), Some(json))
            .unwrap_or_else(|e| {
                panic!(
                    "Starter card '{}' ({}/{}) failed to launch: {}",
                    preset.id, preset.domain, tool_name, e
                )
            });
        assert!(framework.preset_dialog.toast_error.is_none());
        let active = framework.active_tool.as_ref().expect("active tool set");
        let state = active
            .save_state()
            .expect("active tool must serialize state");
        assert_eq!(framework.saved_tool_states.get(active.name()), Some(&state));
        assert_eq!(framework.baseline_tool_state.as_ref(), Some(&state));
        assert_eq!(
            framework
                .last_valid_snapshot
                .as_ref()
                .and_then(|s| s.tool_state.as_ref()),
            Some(&state)
        );
    }
}

#[test]
fn test_all_52_tools_export_and_import_preset_roundtrip() {
    let presets: Vec<&'static StarterPresetMetadata> = inventory::iter::<StarterPresetMetadata>
        .into_iter()
        .collect();

    for meta in inventory::iter::<ToolMetadata> {
        let mut tool = (meta.build)();
        assert_eq!(
            tool.domain(),
            meta.domain,
            "Tool '{}' domain() mismatch",
            meta.id
        );
        let starter = presets
            .iter()
            .find(|p| p.tool_name == Some(meta.id))
            .expect("starter preset exists");
        tool.import_preset_json(starter.generate_preset())
            .unwrap_or_else(|e| panic!("Tool '{}' starter import failed: {}", meta.id, e));

        let exported = tool
            .export_preset_json()
            .unwrap_or_else(|e| panic!("Tool '{}' export_preset_json failed: {}", meta.id, e));
        let mut restored = (meta.build)();
        restored
            .import_preset_json(&exported)
            .unwrap_or_else(|e| panic!("Tool '{}' roundtrip import failed: {}", meta.id, e));
        assert_eq!(
            restored.save_state(),
            tool.save_state(),
            "Tool '{}' state differed after export/import roundtrip",
            meta.id
        );
    }

    let mut morph = MorphogenesisTab::default();
    morph.from_json_str(PRESET_TURING_STRIPES).unwrap();
    let morph_json = morph.to_json_string().unwrap();
    let mut morph_restored = MorphogenesisTab::default();
    morph_restored.from_json_str(&morph_json).unwrap();
    assert_eq!(morph_restored.export_config(), morph.export_config());
}

#[test]
fn test_all_52_tools_reject_unknown_fields_and_wrong_domain() {
    for meta in inventory::iter::<ToolMetadata> {
        let mut tool = (meta.build)();
        let exported = tool
            .export_preset_json()
            .unwrap_or_else(|e| panic!("Tool '{}' export failed: {}", meta.id, e));

        let bad_cfg = inject_unknown_config_field(&exported);
        assert!(
            tool.import_preset_json(&bad_cfg).is_err(),
            "Tool '{}' must reject unknown config field",
            meta.id
        );

        let bad_meta = inject_unknown_metadata_field(&exported);
        assert!(
            tool.import_preset_json(&bad_meta).is_err(),
            "Tool '{}' must reject unknown metadata field",
            meta.id
        );

        let wrong_dom = replace_preset_domain(&exported, "wrong_domain_contract_test");
        assert!(
            tool.import_preset_json(&wrong_dom).is_err(),
            "Tool '{}' must reject wrong metadata.domain",
            meta.id
        );
    }

    let mut morph = MorphogenesisTab::default();
    let morph_exported = morph.to_json_string().unwrap();
    assert!(morph
        .from_json_str(&inject_unknown_config_field(&morph_exported))
        .is_err());
    assert!(morph
        .from_json_str(&replace_preset_domain(&morph_exported, "quantum"))
        .is_err());
}

#[test]
fn test_failed_card_launch_surfaces_toast_and_preset_serialization_telemetry() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let _ = global_registry().try_recv_all();

    let mut framework = SimulationFramework::new("analysis");
    let bad_ode_json = inject_unknown_config_field(PRESET_HARMONIC_OSCILLATOR);
    let res = framework.apply_starter_card(
        "Broken Harmonic",
        Some("OdeSolverTool"),
        Some(&bad_ode_json),
    );
    assert!(res.is_err());
    assert!(framework
        .preset_dialog
        .toast_error
        .as_deref()
        .is_some_and(|m| m.contains("Broken Harmonic")));

    let events = global_registry().try_recv_all();
    assert!(
        events.iter().any(|e| {
            e.source == "PresetSerialization"
                && e.severity == Severity::Warning
                && e.message.contains("Broken Harmonic")
        }),
        "Expected PresetSerialization warning telemetry on failed framework card launch"
    );

    let mut morph_fw = SimulationFramework::new("biology");
    let bad_morph_json = inject_unknown_config_field(PRESET_TURING_SPOTS);
    assert!(morph_fw
        .apply_starter_card(
            "Broken Spots",
            Some("MorphogenesisTool"),
            Some(&bad_morph_json)
        )
        .is_err());
    assert!(morph_fw
        .preset_dialog
        .toast_error
        .as_deref()
        .is_some_and(|m| m.contains("Broken Spots")));
    let morph_events = global_registry().try_recv_all();
    assert!(morph_events.iter().any(|e| {
        e.source == "PresetSerialization"
            && e.severity == Severity::Warning
            && e.message.contains("Broken Spots")
    }));
}

#[test]
fn test_single_starter_gallery_button_per_simulation_view() {
    let screen_rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1400.0, 900.0));
    let mut checks: Vec<(&'static str, &'static str)> = vec![
        ("chaos", "AttractorPlotter"),
        ("fluid_dynamics", "PotentialFlowTool"),
        ("analysis", "OdeSolverTool"),
    ];
    let mut seen_domains = std::collections::HashSet::new();
    for meta in inventory::iter::<ToolMetadata> {
        if seen_domains.insert(meta.domain) {
            checks.push((meta.domain, meta.id));
        }
    }

    for (domain, tool_name) in checks {
        let ctx = egui::Context::default();
        ctx.enable_accesskit();
        let mut framework = SimulationFramework::new(domain);
        assert!(
            framework.select_tool_by_name(tool_name),
            "Failed to select {tool_name} in {domain}"
        );
        let out = ctx.run(
            egui::RawInput {
                screen_rect: Some(screen_rect),
                ..Default::default()
            },
            |ctx| framework.show(ctx, domain),
        );
        assert_eq!(
            count_gallery_buttons(&out),
            1,
            "Expected exactly 1 Starter Gallery button for {domain}/{tool_name}"
        );
    }

    let morph_ctx = egui::Context::default();
    morph_ctx.enable_accesskit();
    let mut morph = MorphogenesisTab::default();
    let mut frame = eframe::Frame::_new_kittest();
    let morph_out = morph_ctx.run(
        egui::RawInput {
            screen_rect: Some(screen_rect),
            ..Default::default()
        },
        |ctx| {
            egui::CentralPanel::default()
                .frame(egui::Frame::NONE)
                .show(ctx, |ui| morph.show_inside(ui, &mut frame));
        },
    );
    assert_eq!(
        count_gallery_buttons(&morph_out),
        1,
        "Expected exactly 1 Starter Gallery button for MorphogenesisTab"
    );
}

#[test]
fn test_all_story_and_challenge_presets_launch_and_compile_ast_cleanly() {
    let stories: Vec<&'static PresetStoryMetadata> =
        inventory::iter::<PresetStoryMetadata>.into_iter().collect();
    let challenges: Vec<&'static GoalChallengeMetadata> = inventory::iter::<GoalChallengeMetadata>
        .into_iter()
        .collect();
    assert!(stories.len() >= 4, "expected flagship guided stories");
    assert!(challenges.len() >= 4, "expected flagship AST challenges");

    for story in stories {
        assert!(!story.steps.is_empty(), "story '{}' has no steps", story.id);
        let mut fw = SimulationFramework::new(story.domain);
        for step_idx in 0..story.steps.len() {
            fw.launch_story(story, step_idx).unwrap_or_else(|e| {
                panic!("Story '{}' step {step_idx} failed to launch: {e}", story.id)
            });
            let bad = inject_unknown_config_field((story.steps[step_idx].preset_generator)());
            assert!(fw
                .active_tool
                .as_mut()
                .unwrap()
                .import_preset_json(&bad)
                .is_err());
        }
    }

    for challenge in challenges {
        crate::framework::CompiledGoalPredicate::compile(challenge.predicate_expr)
            .unwrap_or_else(|e| panic!("Challenge '{}' invalid AST: {e}", challenge.id));
        let mut fw = SimulationFramework::new(challenge.domain);
        fw.launch_challenge(challenge)
            .unwrap_or_else(|e| panic!("Challenge '{}' failed to launch: {e}", challenge.id));
        assert_eq!(
            fw.evaluate_challenge_predicates(None),
            Some(false),
            "Initial preset for challenge '{}' must start unsolved",
            challenge.id
        );
    }
}
