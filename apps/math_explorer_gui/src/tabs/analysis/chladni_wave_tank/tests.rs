#![cfg_attr(any(), verified(opt_out = "tests"))]

// [cite:graph_parameters_rust]

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
fn test_chladni_state_bounds_validation_and_finite_guard_clamping() {
    let mut tool = ChladniWaveTankTool::default();
    assert!(tool.has_finite_parameters());

    let valid_json = tool.save_state().expect("save_state must succeed");
    assert!(tool.state_bounds().validate_json(&valid_json).is_ok());
    assert!(tool.try_load_state(&valid_json).is_ok());

    // Reject unknown fields under #[serde(deny_unknown_fields)]
    let mut val: serde_json::Value = serde_json::from_str(&valid_json).unwrap();
    val["unexpected_field"] = serde_json::json!(42);
    assert!(tool.try_load_state(&val.to_string()).is_err());

    // Reject oversized JSON payloads via ToolStateBounds
    let huge = format!("{{\"pad\":\"{}\"}}", "x".repeat(16_384));
    assert!(tool.try_load_state(&huge).is_err());

    // Reject out-of-range physical parameters
    let mut bad_freq: serde_json::Value = serde_json::from_str(&valid_json).unwrap();
    bad_freq["driver_frequency"] = serde_json::json!(9_999.0);
    assert!(tool.try_load_state(&bad_freq.to_string()).is_err());

    // FiniteStateGuard recovery clamping
    tool.state.wave_speed = f64::NAN;
    tool.state.time_step = 99.0;
    assert!(!tool.has_finite_parameters());
    assert!(tool.clamp_parameters());
    assert!(tool.has_finite_parameters());
    assert!(tool.state.time_step <= DT_MAX);
}

#[test]
fn test_chladni_url_hash_hydration_roundtrip() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let mut source_app = MathExplorerApp::default();
    let analysis_idx = source_app
        .tabs
        .iter()
        .position(|t| t.url_slug() == "analysis")
        .expect("AnalysisTab must exist");
    source_app.selected_tab = analysis_idx;
    assert!(source_app.tabs[analysis_idx].select_tool_by_name("ChladniWaveTankTool"));

    if let Some(fw) = source_app.tabs[analysis_idx].framework_mut() {
        let tool = fw
            .active_tool
            .as_mut()
            .unwrap()
            .as_any_mut()
            .unwrap()
            .downcast_mut::<ChladniWaveTankTool>()
            .unwrap();
        tool.apply_preset(TankPreset::DoubleSlitInterference);
        tool.state.slit_separation = 36.0;
        tool.state_rev = tool.state_rev.wrapping_add(1);
    }
    let expected_state = source_app.tabs[analysis_idx]
        .framework()
        .unwrap()
        .active_tool
        .as_ref()
        .unwrap()
        .save_state();

    let share_url = source_app.encode_current_share_url().unwrap();
    let hash_pos = share_url.find('#').expect("URL must contain hash fragment");
    let hash = &share_url[hash_pos..];

    let mut restored_app = MathExplorerApp::default();
    assert!(restored_app
        .apply_share_location_strings(hash, "", None)
        .unwrap());
    assert_eq!(restored_app.selected_tab, analysis_idx);
    assert_eq!(
        restored_app.tabs[analysis_idx].active_tool_id(),
        Some("ChladniWaveTankTool")
    );
    let actual_state = restored_app.tabs[analysis_idx]
        .framework()
        .unwrap()
        .active_tool
        .as_ref()
        .unwrap()
        .save_state();
    assert_eq!(actual_state, expected_state);
}

fn collect_focusable_canvas_nodes(
    ctx: &Context,
    tool: &mut ChladniWaveTankTool,
) -> Vec<egui::accesskit::NodeId> {
    let _ = ctx.run(raw_frame(), |ctx| tool.show(ctx));
    let out = ctx.run(raw_frame(), |ctx| tool.show(ctx));
    let ak = out
        .platform_output
        .accesskit_update
        .expect("AccessKit update must be emitted");

    ak.nodes
        .iter()
        .filter_map(|(id, node)| {
            (node.role() == egui::accesskit::Role::GraphicsDocument
                && node
                    .label()
                    .unwrap_or_default()
                    .contains("2D Chladni Plate & Wave Tank Visualization")
                && node.supports_action(egui::accesskit::Action::Focus))
            .then_some(*id)
        })
        .collect()
}

#[test]
fn test_chladni_accesskit_focus_arrow_navigation_enter_pulse_and_unlocked_tab() {
    let mut tool = ChladniWaveTankTool {
        paused: true,
        ..Default::default()
    };
    let ctx = Context::default();
    ctx.enable_accesskit();

    let canvas_nodes = collect_focusable_canvas_nodes(&ctx, &mut tool);
    assert!(!canvas_nodes.is_empty(), "Focusable canvas node missing");

    // Step Tab until focus lands on the canvas
    let mut focused_on_canvas = false;
    for _ in 0..80 {
        let step_out = ctx.run(raw_key(Key::Tab), |ctx| tool.show(ctx));
        if step_out
            .platform_output
            .accesskit_update
            .is_some_and(|upd| canvas_nodes.contains(&upd.focus))
        {
            focused_on_canvas = true;
            break;
        }
    }
    assert!(focused_on_canvas, "Canvas must be reachable via Key::Tab");

    // ArrowRight + ArrowUp move harmonic driver coordinates
    let orig_x = tool.state.driver_x;
    let orig_y = tool.state.driver_y;
    let _ = ctx.run(raw_key(Key::ArrowRight), |ctx| tool.show(ctx));
    let _ = ctx.run(raw_key(Key::ArrowUp), |ctx| tool.show(ctx));
    assert!(tool.state.driver_x > orig_x);
    assert!(tool.state.driver_y > orig_y);

    // Enter injects a Gaussian pulse into a quiescent solver
    tool.solver.clear_wave_state();
    assert_eq!(
        tool.solver
            .displacement()
            .iter()
            .fold(0.0_f64, |m, v| m.max(v.abs())),
        0.0
    );
    let _ = ctx.run(raw_key(Key::Enter), |ctx| tool.show(ctx));
    let max_after = tool
        .solver
        .displacement()
        .iter()
        .fold(0.0_f64, |m, v| m.max(v.abs()));
    assert!(max_after > 0.1, "Enter must inject non-zero Gaussian pulse");

    // Pressing Tab once more must exit the canvas (never trapping Tab)
    let next_tab = ctx.run(raw_key(Key::Tab), |ctx| tool.show(ctx));
    if let Some(upd) = next_tab.platform_output.accesskit_update {
        assert!(
            !canvas_nodes.contains(&upd.focus),
            "Key::Tab must not be trapped on focused canvas"
        );
    }
}

fn assert_export_primitives_in_bounds(tool: &dyn crate::framework::InteractiveTool, title: &str) {
    for prim in tool.export_plot_primitives() {
        let pts: &[[f64; 2]] = match &prim {
            egui_plot::ExportPrimitive::Polyline { points, .. } => points,
            egui_plot::ExportPrimitive::Points { points, .. } => points,
            _ => &[],
        };
        for &[x, y] in pts {
            assert!(
                (0.0..=(GRID_SIZE as f64)).contains(&x) && (0.0..=(GRID_SIZE as f64)).contains(&y),
                "Primitive point ({x}, {y}) out of [0, {GRID_SIZE}] raster bounds in {title}"
            );
        }
    }
}

#[test]
fn test_all_four_starter_presets_load_and_export_svg_png_csv_json() {
    let expected_titles = [
        "Square Chladni Plate",
        "Circular Drumhead",
        "Double-Slit Interference",
        "Acoustic Lens",
    ];
    let presets: Vec<&'static StarterPresetMetadata> = inventory::iter::<StarterPresetMetadata>
        .into_iter()
        .filter(|p| p.tool_name == Some("ChladniWaveTankTool"))
        .collect();
    assert_eq!(presets.len(), 4);

    let temp_dir =
        std::env::temp_dir().join(format!("oxidize_chladni_exports_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&temp_dir);
    let ctx = Context::default();

    for expected_title in expected_titles {
        let preset = presets
            .iter()
            .copied()
            .find(|p| p.title == expected_title)
            .unwrap_or_else(|| panic!("Missing starter preset '{expected_title}'"));

        let mut framework = SimulationFramework::new("analysis");
        framework.export_queue.destination_dir = Some(temp_dir.clone());
        framework
            .apply_starter_card(
                preset.title,
                preset.tool_name,
                Some(preset.generate_preset()),
            )
            .expect("apply_starter_card must succeed");

        let _ = ctx.run(raw_frame(), |ctx| framework.show(ctx, "analysis"));
        if let Some(tool) = framework.active_tool.as_deref() {
            assert_export_primitives_in_bounds(tool, expected_title);
        }
        for fmt in [
            ExportFormat::Svg,
            ExportFormat::Png,
            ExportFormat::Csv,
            ExportFormat::Json,
        ] {
            framework.queue_export(fmt);
            let _ = ctx.run(raw_frame(), |ctx| framework.show(ctx, "analysis"));
            let filename = framework
                .export_queue
                .last_exported_filename
                .as_ref()
                .expect("Export filename recorded");
            let meta = std::fs::metadata(temp_dir.join(filename))
                .unwrap_or_else(|e| panic!("Export {fmt:?} missing for {expected_title}: {e}"));
            assert!(meta.len() > 64);
        }
    }
    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_custom_clamps_url_roundtrip_and_bounds_validation() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let default_tool = ChladniWaveTankTool::default();
    let default_json = default_tool.save_state().unwrap();
    assert!(
        !default_json.contains("custom_clamps"),
        "Empty custom_clamps must be omitted via skip_serializing_if"
    );

    let mut source_app = MathExplorerApp::default();
    let analysis_idx = source_app
        .tabs
        .iter()
        .position(|t| t.url_slug() == "analysis")
        .unwrap();
    source_app.selected_tab = analysis_idx;
    assert!(source_app.tabs[analysis_idx].select_tool_by_name("ChladniWaveTankTool"));

    let pins = vec![[0.25, 0.35], [0.72, 0.64], [0.48, 0.22]];
    if let Some(fw) = source_app.tabs[analysis_idx].framework_mut() {
        let tool = fw
            .active_tool
            .as_mut()
            .unwrap()
            .as_any_mut()
            .unwrap()
            .downcast_mut::<ChladniWaveTankTool>()
            .unwrap();
        tool.state.custom_clamps = pins.clone();
        tool.rebuild_masks();
        tool.state_rev = tool.state_rev.wrapping_add(1);
    }

    let share_url = source_app.encode_current_share_url().unwrap();
    let hash = &share_url[share_url.find('#').unwrap()..];
    let mut restored_app = MathExplorerApp::default();
    assert!(restored_app
        .apply_share_location_strings(hash, "", None)
        .unwrap());
    let restored_tool = restored_app.tabs[analysis_idx]
        .framework()
        .unwrap()
        .active_tool
        .as_ref()
        .unwrap()
        .as_any()
        .unwrap()
        .downcast_ref::<ChladniWaveTankTool>()
        .unwrap();
    assert_eq!(restored_tool.state.custom_clamps, pins);
    let pin_cell_x = (0.25 * ((GRID_SIZE - 1) as f64)).round() as usize;
    let pin_cell_y = (0.35 * ((GRID_SIZE - 1) as f64)).round() as usize;
    assert!(restored_tool.solver.wall_mask[pin_cell_y * GRID_SIZE + pin_cell_x] <= 0.5);

    verify_custom_clamp_state_rejections(&default_json);
}

fn verify_custom_clamp_state_rejections(base_json: &str) {
    let mut tool = ChladniWaveTankTool::default();
    let mut sixteen: serde_json::Value = serde_json::from_str(base_json).unwrap();
    sixteen["custom_clamps"] = serde_json::json!(vec![[0.5, 0.5]; 16]);
    assert!(tool.try_load_state(&sixteen.to_string()).is_ok());

    let mut seventeen = sixteen.clone();
    seventeen["custom_clamps"] = serde_json::json!(vec![[0.5, 0.5]; 17]);
    assert!(tool.try_load_state(&seventeen.to_string()).is_err());

    let mut forty = sixteen.clone();
    forty["custom_clamps"] = serde_json::json!(vec![[0.5, 0.5]; 40]);
    assert!(tool.try_load_state(&forty.to_string()).is_err());

    for bad_coord in [[0.005, 0.5], [0.5, 0.995]] {
        let mut out_of_range = sixteen.clone();
        out_of_range["custom_clamps"] = serde_json::json!([bad_coord]);
        assert!(tool.try_load_state(&out_of_range.to_string()).is_err());
    }
}

#[test]
fn test_chladni_sonification_panel_mallet_strike_and_mute_hooks() {
    let mut tool = ChladniWaveTankTool::default();
    tool.step_simulation();
    assert!(tool.audio.latest_frame.is_audible());

    tool.trigger_mallet_pulse(0.45, 0.55, 0.045, 1.0, 1.0);
    assert!(tool.audio.latest_frame.voices[0].transient_impulse > 0.9);

    let ctx = Context::default();
    ctx.enable_accesskit();
    let _ = ctx.run(raw_frame(), |ctx| tool.show(ctx));
    let out = ctx.run(raw_frame(), |ctx| tool.show(ctx));
    let ak = out.platform_output.accesskit_update.unwrap();
    assert!(
        ak.nodes
            .iter()
            .any(|(_, node)| node.role() == egui::accesskit::Role::Meter),
        "Sonification meter must be rendered in ChladniWaveTankTool"
    );
    assert_eq!(tool.audio.pending_impulse, 0.0);

    if crate::audio::snapshot_ui_state().muted {
        crate::audio::toggle_mute_from_user_gesture();
    }
    assert!(!crate::audio::snapshot_ui_state().muted);
    crate::audio::notify_playback_paused();
    assert!(crate::audio::snapshot_ui_state().muted);

    crate::audio::toggle_mute_from_user_gesture();
    crate::audio::notify_playback_reset();
    assert!(crate::audio::snapshot_ui_state().muted);

    crate::audio::toggle_mute_from_user_gesture();
    assert!(tool.clamp_parameters());
    assert!(crate::audio::snapshot_ui_state().muted);
}
