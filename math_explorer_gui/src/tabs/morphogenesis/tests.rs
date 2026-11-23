#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::async_sim::{SimCommand, SimWorkerState, SimulationRunner};
use crate::framework::{ExportFormat, ExportStatusSeverity, InteractiveTool};
use math_commons::math_kernel::colormap::heatmap_color;
use std::thread;
use std::time::Duration;

fn wait_for_morphogenesis_snapshot(tool: &mut MorphogenesisTool) {
    for _ in 0..1200 {
        if tool.worker.update().is_some() || tool.worker.latest_snapshot().is_some() {
            return;
        }
        thread::sleep(Duration::from_millis(5));
    }
}

pub(super) fn active_morpho_tool_state(tab: &MorphogenesisTab) -> MorphogenesisToolState {
    let json = tab
        .framework
        .active_tool
        .as_ref()
        .and_then(|t| t.save_state())
        .expect("MorphogenesisTool must serialize state");
    serde_json::from_str(&json).expect("valid MorphogenesisToolState")
}

pub(super) fn render_tool_frame(
    ctx: &egui::Context,
    tool: &mut MorphogenesisTool,
) -> egui::FullOutput {
    ctx.run(egui::RawInput::default(), |ctx| {
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show(ctx, |ui| tool.show_inside(ui));
    })
}

fn get_repaint_delay(output: &egui::FullOutput) -> Duration {
    output
        .viewport_output
        .get(&egui::ViewportId::ROOT)
        .unwrap()
        .repaint_delay
}

#[test]
fn test_morphogenesis_initial_state() {
    let tool = MorphogenesisTool::default();
    assert_eq!(tool.worker.state(), SimWorkerState::Ready);
    assert!(tool.pending_initial_start);
    assert!(tool.dirty);
    assert!(tool.texture.is_none());
    assert!(tool.pixels.is_empty());
}

#[test]
fn test_morphogenesis_render_lifecycle_and_dirty_tracking() {
    let ctx = egui::Context::default();
    let mut tool = MorphogenesisTool::default();
    assert!(tool.dirty);
    assert!(tool.texture.is_none());

    wait_for_morphogenesis_snapshot(&mut tool);
    let _ = render_tool_frame(&ctx, &mut tool);

    assert!(!tool.dirty);
    assert!(tool.texture.is_some());
    assert_eq!(tool.pixels.len(), tool.width * tool.height);

    let initial_pixels_cap = tool.pixels.capacity();
    let initial_tex_id = tool.texture.as_ref().unwrap().id();

    let _ = tool.worker.send_command(SimCommand::Pause);
    tool.worker.wait_until_paused();
    while tool.worker.update().is_some() {}
    tool.dirty = false;

    let _ = render_tool_frame(&ctx, &mut tool);
    assert!(!tool.dirty);
    assert_eq!(tool.pixels.capacity(), initial_pixels_cap);
    assert_eq!(tool.texture.as_ref().unwrap().id(), initial_tex_id);

    tool.dirty = true;
    let _ = render_tool_frame(&ctx, &mut tool);
    assert!(!tool.dirty);
    assert_eq!(tool.pixels.capacity(), initial_pixels_cap);
    assert_eq!(tool.texture.as_ref().unwrap().id(), initial_tex_id);

    let _ = tool.worker.send_command(SimCommand::Reset);
    tool.dirty = true;
    let _ = render_tool_frame(&ctx, &mut tool);
    assert!(!tool.dirty);
    assert_eq!(tool.texture.as_ref().unwrap().id(), initial_tex_id);
}

#[test]
fn test_morphogenesis_grid_resize_and_buffer_safety() {
    let ctx = egui::Context::default();
    let mut tool = MorphogenesisTool::default();

    wait_for_morphogenesis_snapshot(&mut tool);
    let _ = render_tool_frame(&ctx, &mut tool);
    assert_eq!(tool.pixels.len(), 100 * 100);

    let mut state: MorphogenesisToolState =
        serde_json::from_str(&tool.save_state().unwrap()).unwrap();
    state.width = 50;
    state.height = 50;
    assert!(tool
        .try_load_state(&serde_json::to_string(&state).unwrap())
        .is_ok());
    assert!(tool.dirty);

    wait_for_morphogenesis_snapshot(&mut tool);
    let _ = render_tool_frame(&ctx, &mut tool);
    assert!(!tool.dirty);
    assert_eq!(tool.pixels.len(), 50 * 50);
}

#[test]
fn test_plot_concentration_color_mapping() {
    let data = vec![0.0, 0.5, 1.0, 0.25];
    let image = MorphogenesisTab::plot_concentration(&data, 2, 2);
    assert_eq!(image.size, [2, 2]);
    assert_eq!(image.pixels.len(), 4);

    let min_color = heatmap_color(0.0);
    assert_eq!(
        image.pixels[0],
        egui::Color32::from_rgb(min_color[0], min_color[1], min_color[2])
    );

    let max_color = heatmap_color(1.0);
    assert_eq!(
        image.pixels[2],
        egui::Color32::from_rgb(max_color[0], max_color[1], max_color[2])
    );

    let uniform_data = vec![1.0; 4];
    let uniform_image = MorphogenesisTab::plot_concentration(&uniform_data, 2, 2);
    assert_eq!(uniform_image.pixels.len(), 4);
}

#[test]
fn test_morphogenesis_starter_gallery_and_spots_stripes_reset_turing_system() {
    let mut tab = MorphogenesisTab::default();
    assert!(!tab.framework.show_starter_gallery);

    // Apply Turing Stripes via Starter Gallery on SimulationFramework
    assert!(tab
        .framework
        .apply_starter_card(
            "Turing Stripes",
            Some("MorphogenesisTool"),
            Some(crate::starter_gallery::PRESET_TURING_STRIPES),
        )
        .is_ok());
    assert_eq!(tab.export_config().a, 0.14);
    assert_eq!(tab.export_config().b, 0.86);
    assert_eq!(tab.export_config().d_v, 50.0);
    assert_eq!(tab.export_config().dt, 0.02);

    // Verify Spots and Stripes quick presets route through try_load_state, updating dt and marking dirty
    let mut tool = MorphogenesisTool::default();
    let _ = tool.worker.send_command(SimCommand::Pause);
    tool.worker.wait_until_paused();
    tool.dirty = false;

    tool.apply_named_preset(0.14, 0.86, 1.0, 50.0, 0.02);
    assert!(tool.dirty, "Stripes preset must mark texture dirty");
    assert_eq!(tool.params.read().unwrap().dt, 0.02);
    assert_eq!(tool.params.read().unwrap().d_v, 50.0);

    tool.dirty = false;
    tool.apply_named_preset(0.1, 0.9, 1.0, 100.0, 0.05);
    assert!(tool.dirty, "Spots preset must mark texture dirty");
    assert_eq!(tool.params.read().unwrap().dt, 0.05);
    assert_eq!(tool.params.read().unwrap().d_v, 100.0);
}

#[test]
fn test_morphogenesis_fault_halts_repaints_and_recovers() {
    let ctx = egui::Context::default();
    let mut tool = MorphogenesisTool::default();
    wait_for_morphogenesis_snapshot(&mut tool);
    let _ = render_tool_frame(&ctx, &mut tool);

    let _ = tool.worker.send_command(SimCommand::Pause);
    tool.worker.wait_until_paused();
    while tool.worker.update().is_some() {}
    tool.dirty = false;

    let _ = render_tool_frame(&ctx, &mut tool);
    let output_paused = render_tool_frame(&ctx, &mut tool);
    assert_ne!(get_repaint_delay(&output_paused), Duration::ZERO);

    let _ = tool.worker.send_command(SimCommand::Start);
    let output_running = render_tool_frame(&ctx, &mut tool);
    assert_eq!(tool.worker.state(), SimWorkerState::Running);
    assert_eq!(get_repaint_delay(&output_running), Duration::ZERO);

    let _ = tool.worker.send_command(SimCommand::Pause);
    tool.worker.wait_until_paused();
    while tool.worker.update().is_some() {}

    tool.worker.set_state_for_test(SimWorkerState::Faulted);
    let _ = render_tool_frame(&ctx, &mut tool);
    let output_faulted = render_tool_frame(&ctx, &mut tool);
    assert_ne!(get_repaint_delay(&output_faulted), Duration::ZERO);

    tool.retry();
    assert_eq!(tool.worker.state(), SimWorkerState::Ready);
}

#[test]
fn test_morphogenesis_all_controls_update_state_revision_and_legacy_compatibility() {
    let mut tab = MorphogenesisTab::default();
    assert_eq!(
        active_morpho_tool_state(&tab).state,
        Some(SimWorkerState::Ready)
    );
    let mut prev_rev = tab.state_revision();
    assert!(prev_rev > 0);

    for mutate in [
        |p: &mut MorphogenesisSharedParams| p.a = 0.22,
        |p: &mut MorphogenesisSharedParams| p.b = 1.15,
        |p: &mut MorphogenesisSharedParams| p.d_u = 2.4,
        |p: &mut MorphogenesisSharedParams| p.d_v = 85.0,
        |p: &mut MorphogenesisSharedParams| p.dt = 0.03,
    ] {
        mutate(&mut tab.params.write().unwrap());
        let next_rev = tab.state_revision();
        assert_ne!(next_rev, prev_rev);
        prev_rev = next_rev;
    }

    let saved = tab.save_state().expect("save_state should succeed");
    assert!(!saved.contains("\"paused\""));
    let mut restored = MorphogenesisTab::default();
    restored.load_state(&saved);
    assert_eq!(restored.export_config(), tab.export_config());
    assert_eq!(
        active_morpho_tool_state(&restored).state,
        Some(SimWorkerState::Ready)
    );

    // Running state roundtrips cleanly
    let ctx = egui::Context::default();
    let mut frame = eframe::Frame::_new_kittest();
    let _ = ctx.run(egui::RawInput::default(), |ctx| tab.show(ctx, &mut frame));
    assert_eq!(
        active_morpho_tool_state(&tab).state,
        Some(SimWorkerState::Running)
    );
    let saved_running = tab.save_state().expect("save_state should succeed");
    restored.load_state(&saved_running);
    assert_eq!(
        active_morpho_tool_state(&restored).state,
        Some(SimWorkerState::Running)
    );

    // Load legacy flat MorphoState JSON payloads
    restored.load_state(r#"{"dt":0.02,"paused":true,"show_starter_gallery":true}"#);
    assert!(!restored.framework.show_starter_gallery);
    assert_eq!(restored.params.read().unwrap().dt, 0.02);
    assert_eq!(
        active_morpho_tool_state(&restored).state,
        Some(SimWorkerState::Ready)
    );

    restored.load_state(r#"{"dt":0.08,"paused":false,"show_starter_gallery":false}"#);
    assert_eq!(restored.params.read().unwrap().dt, 0.08);
    assert_eq!(
        active_morpho_tool_state(&restored).state,
        Some(SimWorkerState::Running)
    );
}

#[test]
fn test_morphogenesis_poisoned_lock_and_zero_allocation_step_and_png_export() {
    let mut tab = MorphogenesisTab::default();
    let params_clone = Arc::clone(&tab.params);
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut guard = params_clone.write().unwrap();
        guard.a = 0.25;
        guard.dt = 0.04;
        panic!("simulated worker panic");
    }));
    assert!(tab.params.is_poisoned());

    let mut runner = MorphogenesisTab::create_runner(&tab.params, 20, 20);
    assert_eq!(runner.system.kinetics.a, 0.25);

    // Warm up and verify zero heap allocations inside MorphogenesisRunner::step under VerifiedAllocator
    runner.step();
    {
        let _alloc_guard = crate::VERIFIED_ALLOC_TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        verified_engine::engine::VerifiedEngine::run_verified(|| {
            runner.step();
        });
    }
    assert!(FiniteStateGuard::is_finite_state(&runner));

    // Verify recovery actions & PNG raster export survive poisoned lock
    tab.auto_scale_step_size();
    assert!((tab.params.read().unwrap_or_else(|e| e.into_inner()).dt - 0.02).abs() < 1e-12);
    tab.reset_parameters();
    assert_eq!(tab.params.read().unwrap_or_else(|e| e.into_inner()).a, 0.1);

    let ctx = egui::Context::default();
    let mut frame = eframe::Frame::_new_kittest();
    let screen_rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1280.0, 800.0));
    let _ = ctx.run(
        egui::RawInput {
            screen_rect: Some(screen_rect),
            ..Default::default()
        },
        |ctx| {
            tab.framework.show_export_modal = true;
            tab.framework.queue_export(ExportFormat::Png);
            tab.show(ctx, &mut frame);
        },
    );
    assert_eq!(
        tab.framework.export_status_severity,
        ExportStatusSeverity::Success
    );
    let _ = std::fs::write(
        "/tmp/morphogenesis_turing.png",
        tab.framework
            .export_queue
            .last_exported_bytes
            .as_ref()
            .expect("morphogenesis PNG bytes"),
    );
}

fn click_morpho_button(ctx: &egui::Context, tab: &mut MorphogenesisTab, label_substr: &str) {
    let mut frame = eframe::Frame::_new_kittest();
    let screen_rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1400.0, 900.0));
    let out = ctx.run(
        egui::RawInput {
            screen_rect: Some(screen_rect),
            ..Default::default()
        },
        |ctx| tab.show(ctx, &mut frame),
    );
    let ak = out.platform_output.accesskit_update.as_ref().unwrap();
    let bounds = ak
        .nodes
        .iter()
        .find(|(_, n)| n.label().is_some_and(|l| l.contains(label_substr)))
        .and_then(|(_, n)| n.bounds())
        .unwrap_or_else(|| panic!("Button containing '{label_substr}' not found"));
    let center = egui::pos2(
        ((bounds.x0 + bounds.x1) * 0.5) as f32,
        ((bounds.y0 + bounds.y1) * 0.5) as f32,
    );
    let _ = ctx.run(
        egui::RawInput {
            screen_rect: Some(screen_rect),
            events: vec![
                egui::Event::PointerMoved(center),
                egui::Event::PointerButton {
                    pos: center,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::NONE,
                },
                egui::Event::PointerButton {
                    pos: center,
                    button: egui::PointerButton::Primary,
                    pressed: false,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
            ..Default::default()
        },
        |ctx| tab.show(ctx, &mut frame),
    );
}

#[test]
fn test_morphogenesis_undo_mws_reset_live_region_theory_and_preset_io() {
    let ctx = egui::Context::default();
    ctx.enable_accesskit();

    let mut tab = MorphogenesisTab::default();
    click_morpho_button(&ctx, &mut tab, "Stripes");
    assert_eq!(tab.export_config().a, 0.14);
    assert_eq!(tab.export_config().dt, 0.02);

    let preset_json = tab
        .framework
        .active_tool
        .as_ref()
        .unwrap()
        .export_preset_json()
        .expect("morphogenesis preset export");

    click_morpho_button(&ctx, &mut tab, "Reset Parameters");
    assert_eq!(tab.export_config().a, 0.1);
    assert_eq!(tab.export_config().b, 0.9);
    assert_eq!(tab.export_config().d_v, 100.0);
    assert_eq!(tab.export_config().dt, 0.05);
    let live_msg = crate::accessibility::with_live_region(|r| r.slots[r.active_slot].clone());
    assert_eq!(live_msg, "Parameters reset to baseline defaults");

    tab.framework
        .active_tool
        .as_mut()
        .unwrap()
        .import_preset_json(&preset_json)
        .expect("morphogenesis preset import");
    assert_eq!(tab.export_config().a, 0.14);
    assert_eq!(tab.export_config().dt, 0.02);

    assert!(!tab.framework.show_theory_portal);
    click_morpho_button(&ctx, &mut tab, "Theory");
    assert!(tab.framework.show_theory_portal);
    let mut frame = eframe::Frame::_new_kittest();
    let _ = ctx.run(egui::RawInput::default(), |ctx| tab.show(ctx, &mut frame));
    assert!(
        scientific_metadata::citation_registry::CitationRegistry::get("Turing Morphogenesis")
            .is_some_and(|c| !c.is_empty())
    );

    // End-to-end WorkspaceSessionManager Undo + .mws round-trip
    let mut app = crate::app::MathExplorerApp::default();
    let mut mgr = crate::app::WorkspaceSessionManager::default();
    let morpho_idx = app
        .tabs
        .iter()
        .position(|t| t.name() == "Morphogenesis")
        .expect("Morphogenesis tab registered");
    app.selected_tab = morpho_idx;
    mgr.track_frame_changes(&ctx, &app);
    assert!(!mgr.can_undo());

    let baseline_json = app.tabs[morpho_idx].save_state().unwrap();
    app.tabs[morpho_idx].load_state(r#"{"dt":0.025,"paused":true}"#);
    mgr.track_frame_changes(&ctx, &app);
    assert!(mgr.can_undo());

    let mws = mgr.create_envelope(app.get_app_state());
    let parsed = mgr.parse_envelope(&mws).unwrap();
    assert!(parsed.tab_states["Morphogenesis"].contains("0.025"));

    let prev = mgr.pop_undo().expect("undo snapshot");
    assert_eq!(prev.tab_states["Morphogenesis"], baseline_json);
}
