#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;

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

    let mri_cards = get_starter_cards_for_domain("mri", &[]);
    assert!(mri_cards
        .iter()
        .any(|c| c.title == "Off-Resonance Bloch Precession"));

    let fav_cards = get_starter_cards_for_domain("favoritism", &[]);
    assert!(fav_cards
        .iter()
        .any(|c| c.title == "Crisis & Proximity Showdown"));
}

#[test]
fn test_preset_json_validity() {
    use crate::presets::{deserialize_preset_envelope, MorphogenesisPresetConfig};

    let ode_meta = inventory::iter::<ToolMetadata>
        .into_iter()
        .find(|t| t.id == "OdeSolverTool")
        .expect("OdeSolverTool must be registered");
    let mut ode_tool = (ode_meta.build)();
    ode_tool
        .import_preset_json(PRESET_HARMONIC_OSCILLATOR)
        .unwrap();
    ode_tool.import_preset_json(PRESET_LOGISTIC_GROWTH).unwrap();

    let morph_cfg: MorphogenesisPresetConfig =
        deserialize_preset_envelope(PRESET_TURING_SPOTS, "biology", "Morphogenesis").unwrap();
    assert_eq!(morph_cfg.width, 100);
    assert_eq!(morph_cfg.height, 100);
}

#[test]
fn test_all_registered_tools_have_starter_presets() {
    let tools: Vec<&'static ToolMetadata> = inventory::iter::<ToolMetadata>.into_iter().collect();
    assert_eq!(tools.len(), 67, "Expected 67 registered tools");

    let presets: Vec<&'static StarterPresetMetadata> = inventory::iter::<StarterPresetMetadata>
        .into_iter()
        .collect();
    assert_eq!(presets.len(), 93, "Expected 93 registered starter presets");

    for tool in &tools {
        let matching: Vec<_> = presets
            .iter()
            .filter(|p| p.tool_name == Some(tool.id) && p.domain == tool.domain)
            .collect();
        assert!(
            !matching.is_empty(),
            "Tool '{}' in domain '{}' has no registered starter preset",
            tool.id,
            tool.domain
        );

        for preset in matching {
            let parsed: serde_json::Value = serde_json::from_str(preset.generate_preset())
                .unwrap_or_else(|e| {
                    panic!(
                        "Invalid JSON payload for tool '{}', preset '{}': {}",
                        tool.id, preset.title, e
                    )
                });
            assert!(parsed.is_object(), "Payload must be a JSON object");
            let meta = &parsed["metadata"];
            assert_eq!(meta["schema_version"], "1.0");
            assert_eq!(meta["domain"], tool.domain);
            assert!(parsed["config"].is_object(), "Config must be a JSON object");
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

fn perturb_first_number(val: &mut serde_json::Value) -> bool {
    match val {
        serde_json::Value::Number(num) => {
            if let Some(f) = num.as_f64() {
                if num.is_i64() || num.is_u64() {
                    let next = (num.as_i64().unwrap_or(1) + 1).max(1);
                    *val = serde_json::json!(next);
                } else {
                    let next = if f.abs() > 1e-3 {
                        f * 0.85 + 0.03
                    } else {
                        0.125
                    };
                    *val = serde_json::json!(next);
                }
                return true;
            }
            false
        }
        serde_json::Value::Array(items) => items.iter_mut().any(perturb_first_number),
        serde_json::Value::Object(map) => {
            for (k, v) in map.iter_mut() {
                if k == "next_node_id" || k == "selected_node" || k == "start_node" {
                    continue;
                }
                if perturb_first_number(v) {
                    return true;
                }
            }
            false
        }
        _ => false,
    }
}

fn verify_tool_preset_mutation(tool_meta: &ToolMetadata, preset: &StarterPresetMetadata) {
    let payload = preset.generate_preset();
    let mut target_tool = (tool_meta.build)();
    target_tool.import_preset_json(payload).unwrap_or_else(|e| {
        panic!(
            "import_preset_json failed for preset '{}' on '{}': {}",
            preset.id, tool_meta.id, e
        )
    });
    let expected_str = target_tool
        .save_state()
        .unwrap_or_else(|| panic!("save_state returned None for '{}'", tool_meta.id));
    let expected_val: serde_json::Value = serde_json::from_str(&expected_str).unwrap();

    let mut perturbed_tool = (tool_meta.build)();
    let base_str = perturbed_tool.save_state().unwrap();
    let mut perturbed_val: serde_json::Value = serde_json::from_str(&base_str).unwrap();
    if perturbed_val == expected_val {
        assert!(
            perturb_first_number(&mut perturbed_val),
            "Failed to perturb state for '{}'",
            tool_meta.id
        );
        let perturbed_json = serde_json::to_string(&perturbed_val).unwrap();
        perturbed_tool
            .try_load_state(&perturbed_json)
            .unwrap_or_else(|e| {
                panic!("Perturb try_load_state failed on '{}': {}", tool_meta.id, e)
            });
    }
    let before_val: serde_json::Value =
        serde_json::from_str(&perturbed_tool.save_state().unwrap()).unwrap();
    assert_ne!(
        before_val, expected_val,
        "Perturbed state must differ from preset '{}' on '{}'",
        preset.id, tool_meta.id
    );

    perturbed_tool
        .import_preset_json(payload)
        .unwrap_or_else(|e| panic!("Re-import preset '{}' failed: {}", preset.id, e));
    let after_val: serde_json::Value =
        serde_json::from_str(&perturbed_tool.save_state().unwrap()).unwrap();
    assert_eq!(
        after_val, expected_val,
        "Preset '{}' did not mutate perturbed '{}' to exact target configuration",
        preset.id, tool_meta.id
    );
}

#[test]
fn test_all_57_starter_presets_apply_and_mutate_perturbed_states() {
    let tools: Vec<&'static ToolMetadata> = inventory::iter::<ToolMetadata>.into_iter().collect();
    let presets: Vec<&'static StarterPresetMetadata> = inventory::iter::<StarterPresetMetadata>
        .into_iter()
        .collect();
    assert_eq!(
        presets.len(),
        93,
        "Expected 93 StarterPresetMetadata entries"
    );

    for preset in presets {
        let tool_name = preset
            .tool_name
            .unwrap_or_else(|| panic!("Preset '{}' must specify tool_name", preset.id));
        let tool_meta = tools
            .iter()
            .copied()
            .find(|t| t.id == tool_name && t.domain == preset.domain)
            .unwrap_or_else(|| panic!("Missing ToolMetadata for '{}'", tool_name));
        verify_tool_preset_mutation(tool_meta, preset);
    }
}

#[test]
fn test_all_52_interactive_tools_serialize_and_restore_non_default_state() {
    let tools: Vec<&'static ToolMetadata> = inventory::iter::<ToolMetadata>.into_iter().collect();
    let presets: Vec<&'static StarterPresetMetadata> = inventory::iter::<StarterPresetMetadata>
        .into_iter()
        .collect();
    assert_eq!(tools.len(), 67, "Expected 67 InteractiveTool registrations");

    for tool_meta in tools {
        let default_tool = (tool_meta.build)();
        assert_eq!(default_tool.domain(), tool_meta.domain);
        let default_json = default_tool
            .save_state()
            .unwrap_or_else(|| panic!("save_state returned None for '{}'", tool_meta.id));
        let default_val: serde_json::Value = serde_json::from_str(&default_json).unwrap();

        let preset = presets
            .iter()
            .find(|p| p.tool_name == Some(tool_meta.id) && p.domain == tool_meta.domain)
            .expect("Every registered tool must have a starter preset");
        let mut modified_tool = (tool_meta.build)();
        modified_tool
            .import_preset_json(preset.generate_preset())
            .unwrap();
        let mut non_default_json = modified_tool.save_state().unwrap();
        let mut non_default_val: serde_json::Value =
            serde_json::from_str(&non_default_json).unwrap();

        if non_default_val == default_val {
            assert!(perturb_first_number(&mut non_default_val));
            let candidate = serde_json::to_string(&non_default_val).unwrap();
            modified_tool.try_load_state(&candidate).unwrap();
            non_default_json = modified_tool.save_state().unwrap();
            non_default_val = serde_json::from_str(&non_default_json).unwrap();
        }
        assert_ne!(
            non_default_val, default_val,
            "Non-default state must differ from default for '{}'",
            tool_meta.id
        );

        let mut restored_tool = (tool_meta.build)();
        restored_tool
            .try_load_state(&non_default_json)
            .unwrap_or_else(|e| panic!("try_load_state failed for '{}': {}", tool_meta.id, e));
        let restored_val: serde_json::Value =
            serde_json::from_str(&restored_tool.save_state().unwrap()).unwrap();
        assert_eq!(
            restored_val, non_default_val,
            "State roundtrip mismatch for '{}'",
            tool_meta.id
        );
        assert!(
            restored_tool.try_load_state("{malformed_json").is_err(),
            "Malformed JSON must return Err for '{}'",
            tool_meta.id
        );
    }
}

#[test]
fn test_unified_sim_tool_rejects_unknown_empty_and_non_finite_parameters() {
    let lbm_meta = inventory::iter::<ToolMetadata>
        .into_iter()
        .find(|t| t.id == "LatticeBoltzmannTool")
        .expect("LatticeBoltzmannTool must be registered");
    let mut tool = (lbm_meta.build)();

    assert!(tool.try_load_state("{}").is_err());
    assert!(tool
        .try_load_state(r#"{"reynolds": 100.0, "relaxation_time": 0.6}"#)
        .is_err());
    assert!(tool
        .import_preset_json(
            r#"{"metadata":{"schema_version":"1.0","title":"Bad","domain":"fluid_dynamics","timestamp":"0"},"config":{"unknown_param":1.0}}"#
        )
        .is_err());
    assert!(tool.try_load_state(r#"{"viscosity": 0.04}"#).is_ok());
}

fn assert_overlay_launch_buttons_clamped(cards: &[StarterCard], size: egui::Vec2) {
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, size);
    let mut is_open = true;
    let raw = || egui::RawInput {
        screen_rect: Some(screen),
        ..Default::default()
    };
    let _ = ctx.run(raw(), |ctx| {
        let _ = show_starter_gallery_overlay(ctx, "analysis", cards, &mut is_open);
    });
    let out = ctx.run(raw(), |ctx| {
        let _ = show_starter_gallery_overlay(ctx, "analysis", cards, &mut is_open);
    });
    let ak = out.platform_output.accesskit_update.as_ref().unwrap();
    let launch_bounds: Vec<egui::Rect> = ak
        .nodes
        .iter()
        .filter_map(|(_, n)| {
            n.label()
                .filter(|l| l.contains("▶ Launch "))
                .and_then(|_| n.bounds())
                .map(|b| {
                    egui::Rect::from_min_max(
                        egui::pos2(b.x0 as f32, b.y0 as f32),
                        egui::pos2(b.x1 as f32, b.y1 as f32),
                    )
                })
        })
        .collect();
    assert!(!launch_bounds.is_empty());
    for rect in launch_bounds {
        assert!(
            rect.min.x >= 0.0 && rect.max.x <= size.x,
            "Launch button {rect:?} escaped horizontal viewport 0..{}",
            size.x
        );
    }
}

#[test]
fn test_starter_cards_sorted_and_overlay_buttons_clamped_within_viewport() {
    let cards = get_starter_cards_for_domain("analysis", &[]);
    assert!(cards.len() >= 5);
    let keys: Vec<(&str, &str)> = cards.iter().map(|c| (c.title, c.id)).collect();
    let mut sorted_keys = keys.clone();
    sorted_keys.sort_unstable();
    assert_eq!(
        keys, sorted_keys,
        "Starter cards must be deterministically sorted"
    );

    for size in [egui::vec2(1024.0, 768.0), egui::vec2(360.0, 640.0)] {
        assert_overlay_launch_buttons_clamped(&cards, size);
    }
}
