#![cfg_attr(any(), verified(opt_out = "tests"))]

// [cite:quantum_mechanics]

use eframe::egui::{self, vec2, Context, Event, Key, Modifiers, Pos2, RawInput, Rect};

use super::*;
use crate::app::MathExplorerApp;
use crate::framework::{ExportFormat, SimulationFramework};
use crate::starter_gallery::StarterPresetMetadata;

fn raw_frame() -> RawInput {
    RawInput {
        screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(1280.0, 800.0))),
        ..Default::default()
    }
}

fn raw_key(key: Key) -> RawInput {
    let mut raw = raw_frame();
    raw.events.push(Event::Key {
        key,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers: Modifiers::NONE,
    });
    raw
}

#[test]
fn test_superfluid_state_bounds_validation_and_finite_guard_rollback() {
    let mut tool = SuperfluidVortexTool::default();
    assert!(tool.has_finite_parameters());

    let valid_json = tool.save_state().expect("save_state must succeed");
    assert!(tool.state_bounds().validate_json(&valid_json).is_ok());
    assert!(tool.try_load_state(&valid_json).is_ok());

    let mut unknown: serde_json::Value = serde_json::from_str(&valid_json).unwrap();
    unknown["rogue_field"] = serde_json::json!(123);
    assert!(tool.try_load_state(&unknown.to_string()).is_err());

    let huge = format!("{{\"pad\":\"{}\"}}", "x".repeat(16_384));
    assert!(tool.try_load_state(&huge).is_err());

    let mut bad_g: serde_json::Value = serde_json::from_str(&valid_json).unwrap();
    bad_g["interaction_g"] = serde_json::json!(99_999.0);
    assert!(tool.try_load_state(&bad_g.to_string()).is_err());

    tool.state.interaction_g = f64::NAN;
    tool.state.time_step = 5.0;
    tool.state.stamp_charge = 0;
    assert!(!tool.has_finite_parameters());
    assert!(tool.clamp_parameters());
    assert!(tool.has_finite_parameters());
    assert_eq!(tool.state.stamp_charge, 1);
}

#[test]
fn test_superfluid_strang_norm_conservation_and_vortex_charge_classification() {
    let mut tool = SuperfluidVortexTool::default();
    tool.apply_preset(SuperfluidPreset::VortexDipole);
    assert_eq!(tool.charge_summary.positive_count, 1);
    assert_eq!(tool.charge_summary.negative_count, 1);
    assert_eq!(tool.charge_summary.net_charge, 0);

    for _ in 0..25 {
        tool.step_simulation();
    }
    let norm_err = (tool.current_invariants.norm - tool.initial_invariants.norm).abs();
    assert!(
        norm_err < 1.0e-12,
        "2D Strang GPE norm drift {norm_err:e} exceeded 1e-12"
    );
    assert_eq!(tool.charge_summary.positive_count, 1);
    assert_eq!(tool.charge_summary.negative_count, 1);
    assert_eq!(tool.charge_summary.net_charge, 0);

    tool.apply_preset(SuperfluidPreset::VortexQuadrupole);
    assert_eq!(tool.charge_summary.positive_count, 2);
    assert_eq!(tool.charge_summary.negative_count, 2);
    assert_eq!(tool.charge_summary.net_charge, 0);
}

#[test]
fn test_superfluid_url_hash_hydration_roundtrip() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let mut source_app = MathExplorerApp::default();
    let quantum_idx = source_app
        .tabs
        .iter()
        .position(|t| t.url_slug() == "quantum")
        .expect("QuantumTab must exist");
    source_app.selected_tab = quantum_idx;
    assert!(source_app.tabs[quantum_idx].select_tool_by_name("SuperfluidVortexTool"));

    if let Some(fw) = source_app.tabs[quantum_idx].framework_mut() {
        let tool = fw
            .active_tool
            .as_mut()
            .unwrap()
            .as_any_mut()
            .unwrap()
            .downcast_mut::<SuperfluidVortexTool>()
            .unwrap();
        tool.apply_preset(SuperfluidPreset::CoRotatingPair);
        tool.state.interaction_g = 245.0;
        tool.state_rev = tool.state_rev.wrapping_add(1);
    }
    let expected_state = source_app.tabs[quantum_idx]
        .framework()
        .unwrap()
        .active_tool
        .as_ref()
        .unwrap()
        .save_state();

    let share_url = source_app.encode_current_share_url().unwrap();
    let hash_pos = share_url.find('#').expect("URL must contain hash");
    let hash = &share_url[hash_pos..];

    let mut restored_app = MathExplorerApp::default();
    assert!(restored_app
        .apply_share_location_strings(hash, "", None)
        .unwrap());
    assert_eq!(
        restored_app.tabs[quantum_idx].active_tool_id(),
        Some("SuperfluidVortexTool")
    );
    let actual_state = restored_app.tabs[quantum_idx]
        .framework()
        .unwrap()
        .active_tool
        .as_ref()
        .unwrap()
        .save_state();
    assert_eq!(actual_state, expected_state);
}

#[test]
fn test_superfluid_accesskit_focus_arrow_navigation_enter_stamp_and_tab_exit() {
    let mut tool = SuperfluidVortexTool {
        paused: true,
        ..Default::default()
    };
    let ctx = Context::default();
    ctx.enable_accesskit();

    let _ = ctx.run(raw_frame(), |ctx| tool.show(ctx));
    let out = ctx.run(raw_frame(), |ctx| tool.show(ctx));
    let ak = out.platform_output.accesskit_update.expect("AccessKit");
    let canvas_nodes: Vec<egui::accesskit::NodeId> = ak
        .nodes
        .iter()
        .filter_map(|(id, node)| {
            (node.role() == egui::accesskit::Role::GraphicsDocument
                && node
                    .label()
                    .unwrap_or_default()
                    .contains("2D Superfluid Condensate & Quantized Vortex Canvas")
                && node.supports_action(egui::accesskit::Action::Focus))
            .then_some(*id)
        })
        .collect();
    assert!(!canvas_nodes.is_empty());

    let mut focused = false;
    for _ in 0..80 {
        let step_out = ctx.run(raw_key(Key::Tab), |ctx| tool.show(ctx));
        if step_out
            .platform_output
            .accesskit_update
            .is_some_and(|upd| canvas_nodes.contains(&upd.focus))
        {
            focused = true;
            break;
        }
    }
    assert!(focused, "Superfluid canvas must receive Tab focus");

    let q_before = tool.charge_summary.net_charge;
    let _ = ctx.run(raw_key(Key::ArrowUp), |ctx| tool.show(ctx));
    assert!(tool.state.cursor_y > 0.0);
    let _ = ctx.run(raw_key(Key::Enter), |ctx| tool.show(ctx));
    assert_eq!(tool.charge_summary.net_charge, q_before + 1);

    let next_tab = ctx.run(raw_key(Key::Tab), |ctx| tool.show(ctx));
    if let Some(upd) = next_tab.platform_output.accesskit_update {
        assert!(!canvas_nodes.contains(&upd.focus));
    }
}

#[test]
fn test_superfluid_all_four_starter_presets_export_svg_png_csv_json() {
    let presets: Vec<&'static StarterPresetMetadata> = inventory::iter::<StarterPresetMetadata>
        .into_iter()
        .filter(|p| p.tool_name == Some("SuperfluidVortexTool"))
        .collect();
    assert_eq!(presets.len(), 4);

    let temp_dir =
        std::env::temp_dir().join(format!("oxidize_superfluid_exports_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&temp_dir);
    let ctx = Context::default();

    for preset in presets {
        let mut framework = SimulationFramework::new("quantum");
        framework.export_queue.destination_dir = Some(temp_dir.clone());
        framework
            .apply_starter_card(
                preset.title,
                preset.tool_name,
                Some(preset.generate_preset()),
            )
            .expect("apply_starter_card must succeed");
        let _ = ctx.run(raw_frame(), |ctx| framework.show(ctx, "quantum"));
        for fmt in [
            ExportFormat::Svg,
            ExportFormat::Png,
            ExportFormat::Csv,
            ExportFormat::Json,
        ] {
            framework.queue_export(fmt);
            let _ = ctx.run(raw_frame(), |ctx| framework.show(ctx, "quantum"));
            let filename = framework
                .export_queue
                .last_exported_filename
                .as_ref()
                .expect("Export filename recorded");
            let meta = std::fs::metadata(temp_dir.join(filename)).unwrap();
            assert!(meta.len() > 64);
        }
    }
    let _ = std::fs::remove_dir_all(&temp_dir);
}
