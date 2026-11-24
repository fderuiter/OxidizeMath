use super::presets::{PRESET_PHYSARUM_MAZE, PRESET_PHYSARUM_STEINER, PRESET_PHYSARUM_TOKYO_RAIL};
use super::*;
use crate::accessibility::{all_announcements, clear_test_announcements};
use crate::framework::recovery::FiniteStateGuard;
use crate::framework::{ExportFormat, ExportStatusSeverity, InteractiveTool, SimulationFramework};
use crate::tabs::morphogenesis::MorphogenesisTab;
use crate::tabs::ExplorerTab;
use crate::url_state::MAX_DECODED_URL_STATE_BYTES;
use math_commons::math_kernel::types::flatten_2d_index;

fn screen_input(events: Vec<egui::Event>) -> egui::RawInput {
    egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1280.0, 800.0),
        )),
        focused: true,
        events,
        ..Default::default()
    }
}

fn key_press(key: egui::Key) -> egui::Event {
    egui::Event::Key {
        key,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers: egui::Modifiers::NONE,
    }
}

fn primary_click_events(target: egui::Pos2) -> Vec<egui::Event> {
    vec![
        egui::Event::PointerMoved(target),
        egui::Event::PointerButton {
            pos: target,
            button: egui::PointerButton::Primary,
            pressed: true,
            modifiers: egui::Modifiers::NONE,
        },
        egui::Event::PointerButton {
            pos: target,
            button: egui::PointerButton::Primary,
            pressed: false,
            modifiers: egui::Modifiers::NONE,
        },
    ]
}

#[test]
fn test_morphogenesis_tab_switches_across_turing_lenia_botanical_and_physarum() {
    let mut tab = MorphogenesisTab::default();
    assert_eq!(tab.active_tool_id(), Some("MorphogenesisTool"));

    assert!(tab.select_tool_by_name("LeniaTool"));
    assert_eq!(tab.active_tool_id(), Some("LeniaTool"));

    assert!(tab.select_tool_by_name("PhyllotaxisLSystemTool"));
    assert_eq!(tab.active_tool_id(), Some("PhyllotaxisLSystemTool"));

    assert!(tab.select_tool_by_name("PhysarumTool"));
    assert_eq!(tab.active_tool_id(), Some("PhysarumTool"));

    for (title, raw) in [
        ("Physarum Steiner Y-Junction", PRESET_PHYSARUM_STEINER),
        ("Physarum Shortest-Path Maze", PRESET_PHYSARUM_MAZE),
        ("Physarum Tokyo-Rail Network", PRESET_PHYSARUM_TOKYO_RAIL),
    ] {
        tab.framework
            .apply_starter_card(title, Some("PhysarumTool"), Some(raw))
            .expect("Physarum starter card must apply cleanly");
        let phys = tab
            .framework
            .active_tool
            .as_mut()
            .and_then(|t| t.as_any_mut())
            .and_then(|a| a.downcast_mut::<PhysarumTool>())
            .expect("active tool should be PhysarumTool");
        phys.step_once();
        assert!(FiniteStateGuard::is_finite_state(phys));
        assert!(phys.engine.telemetry().total_trail_mass > 1.0);
        let obs = phys.observables();
        assert_eq!(obs.len(), 6);
        assert!(obs.iter().all(|(_, v)| v.is_finite()));
        assert_eq!(
            phys.theory().maturity_tier(),
            scientific_metadata::theory::MaturityTier::Validated
        );
    }

    let mut right_pane = MorphogenesisTab::default();
    assert!(right_pane.select_tool_by_name("LeniaTool"));
    assert_eq!(tab.active_tool_id(), Some("PhysarumTool"));
    assert_eq!(right_pane.active_tool_id(), Some("LeniaTool"));
}

fn physarum_canvas_point(out: &egui::FullOutput, frac: f32) -> egui::Pos2 {
    let ak = out
        .platform_output
        .accesskit_update
        .as_ref()
        .expect("AccessKit update");
    let bounds = ak
        .nodes
        .iter()
        .find(|(_, n)| {
            n.label()
                .is_some_and(|l| l.contains("Physarum Network Lab Chemotactic Canvas"))
        })
        .and_then(|(_, n)| n.bounds())
        .expect("Physarum canvas bounds");
    egui::pos2(
        (bounds.x0 + (bounds.x1 - bounds.x0) * f64::from(frac)) as f32,
        (bounds.y0 + (bounds.y1 - bounds.y0) * f64::from(frac)) as f32,
    )
}

#[test]
fn test_physarum_mouse_city_wall_sever_brushes_and_keyboard_accessibility() {
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let mut tool = PhysarumTool {
        brush_mode: PhysarumBrushMode::PlaceCity,
        ..Default::default()
    };
    let initial_cities = tool.engine.terminal_count();

    let out = ctx.run(screen_input(Vec::new()), |ctx| tool.show(ctx));
    let target = physarum_canvas_point(&out, 0.25);

    let _ = ctx.run(screen_input(primary_click_events(target)), |ctx| {
        tool.show(ctx)
    });
    assert_eq!(tool.engine.terminal_count(), initial_cities + 1);
    assert!(tool.focused_canvas_id().is_some());

    clear_test_announcements();
    let key_events = vec![
        key_press(egui::Key::ArrowRight),
        key_press(egui::Key::ArrowDown),
        key_press(egui::Key::W),
    ];
    let _ = ctx.run(screen_input(key_events), |ctx| tool.show(ctx));
    crate::accessibility::sync_context_live_messages(&ctx, false);
    assert!(!tool.obstacle_strokes.is_empty());
    assert!(tool.obstacle_mask_rle.is_some());
    assert!(all_announcements()
        .iter()
        .any(|(msg, _)| msg.contains("Painted Physarum obstacle wall")));
    if let Some(cid) = tool
        .focused_canvas_id()
        .or_else(|| ctx.memory(|m| m.focused()))
    {
        ctx.memory_mut(|m| m.request_focus(cid));
    }

    let mass_before_sever = tool.engine.telemetry().total_trail_mass;
    tool.cursor = [tool.grid_size / 2, tool.grid_size / 2];
    let _ = ctx.run(screen_input(vec![key_press(egui::Key::Minus)]), |ctx| {
        tool.show(ctx)
    });
    let mass_after_sever = tool.engine.telemetry().total_trail_mass;
    assert!(
        mass_after_sever < mass_before_sever,
        "Expected keyboard sever to reduce trail mass: after={mass_after_sever:.2}, before={mass_before_sever:.2}",
    );
}

#[test]
fn test_physarum_bresenham_diagonal_drag_leaves_zero_gaps_and_rle_roundtrips_30_strokes() {
    let mut tool = PhysarumTool {
        brush_radius: 1.0,
        ..Default::default()
    };
    tool.apply_wall_drag_segment((14, 14), (46, 46), true);
    let w = tool.engine.width();
    for k in 14..=46 {
        assert_eq!(
            tool.engine.obstacles()[flatten_2d_index(k, k, w)],
            1,
            "Diagonal Bresenham wall drag must not skip cell ({k}, {k})"
        );
    }

    // Apply > MAX_SAVED_OBSTACLE_STROKES (30 distinct wall & erase strokes) and verify pixel-exact RLE restore
    for step in 0..30usize {
        let gx = 8 + (step * 3) % 108;
        let gy = 84 + (step * 2) % 32;
        tool.apply_wall_stroke(gx, gy, step % 5 != 0);
    }
    let expected_mask = tool.engine.obstacles().to_vec();
    let saved_json = tool.save_state().expect("saved Physarum state");
    assert!(
        saved_json.len() < MAX_DECODED_URL_STATE_BYTES,
        "RLE-backed Physarum state ({}) must remain < 4 KiB",
        saved_json.len()
    );

    let mut restored = PhysarumTool::default();
    restored
        .try_load_state(&saved_json)
        .expect("RLE-backed Physarum state must restore cleanly");
    assert_eq!(restored.engine.obstacles(), expected_mask.as_slice());

    // Reset (rebuild_from_state) must also preserve the full RLE obstacle mask
    restored.rebuild_from_state();
    assert_eq!(restored.engine.obstacles(), expected_mask.as_slice());

    // Erase back to clean preset -> obstacle_mask_rle clears to None
    let mut clean_tool = PhysarumTool::default();
    clean_tool.apply_wall_stroke(12, 12, true);
    assert!(clean_tool.obstacle_mask_rle.is_some());
    clean_tool.apply_wall_stroke(12, 12, false);
    assert!(clean_tool.obstacle_mask_rle.is_none());
}

#[test]
fn test_morphogenesis_tab_propagates_physarum_focused_canvas_id_and_modal_guard() {
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let mut tab = MorphogenesisTab::default();
    assert!(tab.select_tool_by_name("PhysarumTool"));
    assert!(tab.focused_canvas_id().is_none());

    let out = ctx.run(screen_input(Vec::new()), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            tab.framework.show_inside(ui, "biology");
        });
    });
    let target = physarum_canvas_point(&out, 0.5);
    let _ = ctx.run(screen_input(primary_click_events(target)), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            tab.framework.show_inside(ui, "biology");
        });
    });
    let focused = tab.focused_canvas_id().expect("focused Physarum canvas");
    let rev_before_modal = tab.state_revision();
    ctx.memory_mut(|m| m.request_focus(focused));
    ctx.data_mut(|d| d.insert_temp(egui::Id::new("SHELL_MODAL_OR_SASH_FOCUSED"), true));
    let _ = ctx.run(
        screen_input(vec![key_press(egui::Key::C), key_press(egui::Key::W)]),
        |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                tab.framework.show_inside(ui, "biology");
            });
        },
    );
    ctx.data_mut(|d| d.remove::<bool>(egui::Id::new("SHELL_MODAL_OR_SASH_FOCUSED")));
    assert_eq!(tab.state_revision(), rev_before_modal);
}

#[test]
fn test_physarum_finite_state_guard_divergence_and_recovery() {
    let mut tool = PhysarumTool {
        step_size: 9.0,
        sensor_angle: -0.5,
        playing: true,
        ..Default::default()
    };
    tool.step_once();
    assert!(FiniteStateGuard::diverged(&tool));
    assert!(!tool.playing);
    assert!(tool.diagnostic_report().is_some());

    FiniteStateGuard::auto_scale_step_size(&mut tool);
    assert!(FiniteStateGuard::is_finite_state(&tool));
    assert!((0.20..=1.75).contains(&tool.step_size));
    assert!((0.10..=1.50).contains(&tool.sensor_angle));
}

#[test]
fn test_physarum_exports_png_gif_csv_json_and_renders_screenshots() {
    let ctx = egui::Context::default();
    let mut framework = SimulationFramework::new("biology");
    framework.export_options.width = 320;
    framework.export_options.height = 240;
    framework.export_gif_frames = 4;
    assert!(framework.select_tool_by_name("PhysarumTool"));
    let fast_steiner = PRESET_PHYSARUM_STEINER
        .replace("\"grid_size\": 128", "\"grid_size\": 64")
        .replace("\"agent_count\": 6144", "\"agent_count\": 1024");
    framework
        .apply_starter_card(
            "Physarum Steiner Y-Junction",
            Some("PhysarumTool"),
            Some(&fast_steiner),
        )
        .expect("Steiner 64x64 starter");

    for fmt in [
        ExportFormat::Png,
        ExportFormat::Gif,
        ExportFormat::Csv,
        ExportFormat::Json,
    ] {
        let _ = ctx.run(screen_input(Vec::new()), |ctx| {
            framework.show_export_modal = true;
            framework.queue_export(fmt);
            framework.show(ctx, "biology");
        });
        if fmt == ExportFormat::Gif {
            for _ in 0..250 {
                if framework.active_gif_progress().is_none() {
                    break;
                }
                #[cfg(not(target_arch = "wasm32"))]
                std::thread::sleep(std::time::Duration::from_millis(10));
                let _ = ctx.run(screen_input(Vec::new()), |ctx| {
                    framework.show(ctx, "biology")
                });
            }
            let bytes = framework.export_queue.last_exported_bytes.as_deref();
            assert!(bytes.is_some_and(|b| b.starts_with(b"GIF89a")));
        }
        assert_eq!(
            framework.export_status_severity,
            ExportStatusSeverity::Success,
            "Export {fmt:?} failed: {:?}",
            framework.export_status_message
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    if std::env::var_os("WRITE_PHYSARUM_SCREENSHOTS").is_some() {
        screenshot_harness::write_verification_screenshots();
    }
}

#[test]
fn test_physarum_compact_url_share_roundtrip_and_atomic_bounds_rejection() {
    let mut framework = SimulationFramework::new("biology");
    assert!(framework.select_tool_by_name("PhysarumTool"));
    framework
        .apply_starter_card(
            "Physarum Tokyo-Rail Network",
            Some("PhysarumTool"),
            Some(PRESET_PHYSARUM_TOKYO_RAIL),
        )
        .expect("Tokyo-Rail starter");

    let preset_json = framework
        .export_url_preset_json()
        .expect("URL preset export")
        .expect("non-empty URL preset");
    assert!(
        preset_json.len() < MAX_DECODED_URL_STATE_BYTES,
        "Physarum URL JSON {} bytes must stay < 4 KiB",
        preset_json.len()
    );
    let share_url = framework.encode_share_url().expect("encode share URL");
    let hash_idx = share_url.find('#').expect("fragment separator");
    let fragment = &share_url[hash_idx..];

    let mut app = crate::app::MathExplorerApp::default();
    assert!(app
        .apply_share_location_strings(fragment, "", None)
        .unwrap());
    let bio_idx = app
        .tabs
        .iter()
        .position(|t| t.name() == "Morphogenesis")
        .unwrap();
    assert_eq!(app.selected_tab, bio_idx);
    assert_eq!(app.tabs[bio_idx].active_tool_id(), Some("PhysarumTool"));
    assert_eq!(
        app.tabs[bio_idx].export_url_preset_json().unwrap(),
        Some(preset_json)
    );
    assert_physarum_atomic_bounds_and_rle_rejection();
}

fn assert_physarum_atomic_bounds_and_rle_rejection() {
    let mut tool = PhysarumTool::default();
    for idx in 0..=super::presets::MAX_SAVED_OBSTACLE_STROKES {
        tool.apply_wall_stroke(8 + idx * 3, 12 + idx * 2, true);
    }
    let rev_at_cap = tool.url_state_revision();
    tool.apply_wall_stroke(96, 96, false);
    assert_ne!(tool.url_state_revision(), rev_at_cap);

    let snapshot_before_bad = tool.save_state().unwrap();
    let bad_grid = snapshot_before_bad.replace("\"grid_size\":128", "\"grid_size\":512");
    assert!(tool.try_load_state(&bad_grid).is_err());
    assert_eq!(
        tool.save_state().as_deref(),
        Some(snapshot_before_bad.as_str())
    );
    let bad_agents = snapshot_before_bad.replace("\"agent_count\":6144", "\"agent_count\":65536");
    assert!(tool.try_load_state(&bad_agents).is_err());

    // Corrupted / mismatched-dimension RLE mask must be rejected before mutating tool state
    let rle_64 = crate::framework::encode_u8_grid(64, 64, &vec![0u8; 64 * 64], 1).unwrap();
    let mut bad_rle_val: serde_json::Value = serde_json::from_str(&snapshot_before_bad).unwrap();
    bad_rle_val["obstacle_mask_rle"] = serde_json::json!(rle_64);
    assert!(tool.try_load_state(&bad_rle_val.to_string()).is_err());
    bad_rle_val["obstacle_mask_rle"] = serde_json::json!("RM_corrupted_payload");
    assert!(tool.try_load_state(&bad_rle_val.to_string()).is_err());
    assert_eq!(
        tool.save_state().as_deref(),
        Some(snapshot_before_bad.as_str())
    );
}
