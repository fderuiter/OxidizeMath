#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::framework::SimulationFramework;
use crate::presets::{
    InteractivePreset, MorphogenesisPresetConfig, OdePresetConfig, PresetWrapper,
};
use crate::tabs::morphogenesis::MorphogenesisTab;
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
fn test_starter_cards_presence_and_metadata() {
    let analysis_cards = get_starter_cards_for_domain("analysis", &[]);
    assert!(analysis_cards
        .iter()
        .any(|c| c.title == "Harmonic Oscillator"));
    assert!(analysis_cards.iter().any(|c| c.title == "Logistic Growth"));

    let fluid_cards = get_starter_cards_for_domain("fluid_dynamics", &[]);
    assert!(fluid_cards.iter().any(|c| c.title == "Vector Field"));

    let chaos_cards = get_starter_cards_for_domain("chaos", &[]);
    assert!(chaos_cards.iter().any(|c| c.title == "3D Attractor"));

    let bio_cards = get_starter_cards_for_domain("biology", &[]);
    assert!(bio_cards.iter().any(|c| c.title == "Turing Spots"));
}

#[test]
fn test_preset_json_validity() {
    let ode_env: PresetWrapper<OdePresetConfig> =
        serde_json::from_str(PRESET_HARMONIC_OSCILLATOR).unwrap();
    assert_eq!(ode_env.metadata.domain, "analysis");

    let log_env: PresetWrapper<OdePresetConfig> =
        serde_json::from_str(PRESET_LOGISTIC_GROWTH).unwrap();
    assert_eq!(log_env.metadata.domain, "analysis");

    let morph_env: PresetWrapper<MorphogenesisPresetConfig> =
        serde_json::from_str(PRESET_TURING_SPOTS).unwrap();
    assert_eq!(morph_env.metadata.domain, "biology");
}

#[test]
fn test_all_registered_tools_have_starter_presets() {
    let tools: Vec<&'static ToolMetadata> = inventory::iter::<ToolMetadata>.into_iter().collect();
    assert_eq!(tools.len(), 52, "Expected 52 registered tools");

    let presets: Vec<&'static StarterPresetMetadata> = inventory::iter::<StarterPresetMetadata>
        .into_iter()
        .collect();
    assert_eq!(presets.len(), 57, "Expected 57 registered starter presets");

    for tool in &tools {
        let matching: Vec<_> = presets
            .iter()
            .filter(|p| p.tool_name == Some(tool.name) && p.domain == tool.domain)
            .collect();
        assert!(
            !matching.is_empty(),
            "Tool '{}' in domain '{}' has no registered starter preset",
            tool.name,
            tool.domain
        );
        for preset in matching {
            let parsed: serde_json::Value = serde_json::from_str(preset.generate_preset())
                .unwrap_or_else(|e| {
                    panic!(
                        "Invalid JSON payload for tool '{}', preset '{}': {}",
                        tool.name, preset.title, e
                    )
                });
            assert_eq!(parsed["metadata"]["schema_version"], "1.0");
            assert_eq!(parsed["metadata"]["domain"], tool.domain);
            assert!(parsed["config"].is_object());
        }
    }
}

#[test]
fn test_starter_cards_for_all_registered_domains() {
    let mut domains: std::collections::HashSet<&'static str> = std::collections::HashSet::new();
    for tool in inventory::iter::<ToolMetadata> {
        domains.insert(tool.domain);
    }
    domains.insert("biology");

    for &domain in &domains {
        let cards = get_starter_cards_for_domain(domain, &[]);
        assert!(
            !cards.is_empty(),
            "get_starter_cards_for_domain returned no cards for domain '{}'",
            domain
        );
        for card in &cards {
            assert!(!card.title.is_empty());
            assert!(!card.description.is_empty());
            assert!(card.preset_json.is_some());
        }
    }

    let datasets = vec![(
        "Federated Presets Per Domain".to_string(),
        domains
            .iter()
            .enumerate()
            .map(|(i, &d)| [i as f64, get_starter_cards_for_domain(d, &[]).len() as f64])
            .collect(),
    )];
    let options = egui_plot::export::ExportOptions::default();
    if let Ok(png_data) = egui_plot::export::export_custom_png(&datasets, &options) {
        let _ = std::fs::write("/tmp/federated_preset_inventory_preview.png", png_data);
    }
}

#[test]
fn test_all_57_starter_cards_launch_and_sync_framework_state() {
    let presets: Vec<&'static StarterPresetMetadata> = inventory::iter::<StarterPresetMetadata>
        .into_iter()
        .collect();
    assert_eq!(presets.len(), 57);

    for preset in presets {
        let json = preset.generate_preset();
        if let Some(tool_name) = preset.tool_name {
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
        } else {
            let mut morph = MorphogenesisTab::default();
            morph
                .apply_starter_card(preset.title, Some(json))
                .unwrap_or_else(|e| panic!("Morphogenesis card '{}' failed: {}", preset.id, e));
            assert!(morph.dialog_state.toast_error.is_none());
        }
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
            meta.name
        );
        let starter = presets
            .iter()
            .find(|p| p.tool_name == Some(meta.name))
            .expect("starter preset exists");
        tool.import_preset_json(starter.generate_preset())
            .unwrap_or_else(|e| panic!("Tool '{}' starter import failed: {}", meta.name, e));

        let exported = tool
            .export_preset_json()
            .unwrap_or_else(|e| panic!("Tool '{}' export_preset_json failed: {}", meta.name, e));
        let mut restored = (meta.build)();
        restored
            .import_preset_json(&exported)
            .unwrap_or_else(|e| panic!("Tool '{}' roundtrip import failed: {}", meta.name, e));
        assert_eq!(
            restored.save_state(),
            tool.save_state(),
            "Tool '{}' state differed after export/import roundtrip",
            meta.name
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
            .unwrap_or_else(|e| panic!("Tool '{}' export failed: {}", meta.name, e));

        let bad_cfg = inject_unknown_config_field(&exported);
        assert!(
            tool.import_preset_json(&bad_cfg).is_err(),
            "Tool '{}' must reject unknown config field",
            meta.name
        );

        let bad_meta = inject_unknown_metadata_field(&exported);
        assert!(
            tool.import_preset_json(&bad_meta).is_err(),
            "Tool '{}' must reject unknown metadata field",
            meta.name
        );

        let wrong_dom = replace_preset_domain(&exported, "wrong_domain_contract_test");
        assert!(
            tool.import_preset_json(&wrong_dom).is_err(),
            "Tool '{}' must reject wrong metadata.domain",
            meta.name
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

    let mut morph = MorphogenesisTab::default();
    let bad_morph_json = inject_unknown_config_field(PRESET_TURING_SPOTS);
    assert!(morph
        .apply_starter_card("Broken Spots", Some(&bad_morph_json))
        .is_err());
    assert!(morph
        .dialog_state
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
            checks.push((meta.domain, meta.name));
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
    let morph_out = morph_ctx.run(
        egui::RawInput {
            screen_rect: Some(screen_rect),
            ..Default::default()
        },
        |ctx| {
            morph.draw_controls(ctx);
            morph.draw_canvas(ctx);
        },
    );
    assert_eq!(
        count_gallery_buttons(&morph_out),
        1,
        "Expected exactly 1 Starter Gallery button for MorphogenesisTab"
    );
}
