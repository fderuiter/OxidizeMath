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

fn apply_sample_luthiery_strokes(tool: &mut ChladniWaveTankTool) {
    use crate::framework::{BrushMode, BrushStroke, InteractionContext, InteractiveTool};

    let ctx = Context::default();
    let mut dummy_resp = None;
    let _ = ctx.run(raw_frame(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            dummy_resp = Some(ui.label("probe"));
        });
    });
    let resp = dummy_resp.expect("egui dummy response");
    tool.state.pickup_x = 0.72;
    tool.state.pickup_y = 0.34;
    for (mode, start, end, radius) in [
        (BrushMode::Solid, [24.0, 24.0], [28.0, 24.0], 2.0),
        (BrushMode::PotentialBarrier, [40.0, 60.0], [88.0, 60.0], 3.0),
        (BrushMode::VelocityJet, [32.0, 84.0], [72.0, 84.0], 2.0),
        (BrushMode::PotentialWell, [80.0, 40.0], [80.0, 40.0], 5.0),
        (BrushMode::Erase, [24.0, 24.0], [24.0, 24.0], 1.0),
    ] {
        let ic = InteractionContext::from_brush_stroke(
            &resp,
            BrushStroke {
                start,
                end,
                radius,
                mode,
                drag_vector: [end[0] - start[0], end[1] - start[1]],
            },
        );
        tool.on_brush(&ic);
    }
}

fn active_chladni_mut(app: &mut MathExplorerApp, tab_idx: usize) -> &mut ChladniWaveTankTool {
    app.tabs[tab_idx]
        .framework_mut()
        .unwrap()
        .active_tool
        .as_mut()
        .unwrap()
        .as_any_mut()
        .unwrap()
        .downcast_mut::<ChladniWaveTankTool>()
        .unwrap()
}

#[test]
fn test_chladni_sculpt_rle_and_pickup_url_roundtrip_under_4kib() {
    use math_explorer::math_kernel::types::flatten_2d_index;

    use crate::url_state::MAX_DECODED_URL_STATE_BYTES;

    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    assert!(!ChladniWaveTankTool::default()
        .save_state()
        .unwrap()
        .contains("sculpt_rle"));

    let mut source_app = MathExplorerApp::default();
    let analysis_idx = source_app
        .tabs
        .iter()
        .position(|t| t.url_slug() == "analysis")
        .expect("AnalysisTab must exist");
    source_app.selected_tab = analysis_idx;
    assert!(source_app.tabs[analysis_idx].select_tool_by_name("ChladniWaveTankTool"));

    let sculpted_json = {
        let tool = active_chladni_mut(&mut source_app, analysis_idx);
        apply_sample_luthiery_strokes(tool);
        tool.rebuild_masks();
        assert_eq!(
            tool.solver.wall_mask[flatten_2d_index(28, 24, GRID_SIZE)],
            0.0
        );
        assert!(tool.solver.wave_speed_mask[flatten_2d_index(64, 60, GRID_SIZE)] > 1.4);
        assert!(tool.solver.wave_speed_mask[flatten_2d_index(80, 40, GRID_SIZE)] < 0.5);
        tool.save_state().unwrap()
    };
    assert!(sculpted_json.contains("sculpt_rle"));
    assert!(sculpted_json.len() < MAX_DECODED_URL_STATE_BYTES);

    let share_url = source_app.encode_current_share_url().unwrap();
    let hash = &share_url[share_url.find('#').unwrap()..];
    let mut restored_app = MathExplorerApp::default();
    assert!(restored_app
        .apply_share_location_strings(hash, "", None)
        .unwrap());
    let restored = active_chladni_mut(&mut restored_app, analysis_idx);
    assert_eq!(
        restored.save_state().as_deref(),
        Some(sculpted_json.as_str())
    );
    assert!((restored.state.pickup_x - 0.72).abs() < 1.0e-12);
    assert_eq!(
        restored.solver.wall_mask[flatten_2d_index(28, 24, GRID_SIZE)],
        0.0
    );
}

#[test]
fn test_chladni_noisy_sculpt_mask_coarsens_under_4kib() {
    use crate::url_state::MAX_DECODED_URL_STATE_BYTES;

    let mut noisy_tool = ChladniWaveTankTool::default();
    for idx in 0..noisy_tool.sculpt_mask.len() {
        noisy_tool.sculpt_mask[idx] = ((idx * 7 + (idx / GRID_SIZE) * 3) % 8) as u8;
    }
    noisy_tool.state.sculpt_rle =
        super::sculpt::encode_bounded_sculpt_rle(&noisy_tool.sculpt_mask, GRID_SIZE, GRID_SIZE);
    let noisy_json = noisy_tool.save_state().unwrap();
    assert!(noisy_json.len() < MAX_DECODED_URL_STATE_BYTES);
    assert!(noisy_tool.try_load_state(&noisy_json).is_ok());
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
