#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::async_sim::{SimCommand, SimWorkerState, SimulationRunner};
use crate::framework::recovery::FiniteStateGuard;
use crate::framework::{ExportFormat, ExportStatusSeverity, InteractiveTool};
use egui::{CentralPanel, Frame};
use math_commons::math_kernel::colormap::heatmap_color;
use std::{sync::Arc, thread, time::Duration};

fn raw_screen_input(w: f32, h: f32, events: Vec<egui::Event>) -> egui::RawInput {
    egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(w, h),
        )),
        events,
        ..Default::default()
    }
}

fn wait_for_morphogenesis_snapshot(tool: &mut MorphogenesisTool) {
    for _ in 0..600 {
        if tool.worker.update().is_some() || tool.worker.latest_snapshot().is_some() {
            return;
        }
        thread::sleep(Duration::from_millis(5));
    }
}

fn morpho_tool(tab: &MorphogenesisTab) -> &MorphogenesisTool {
    tab.framework
        .active_tool
        .as_ref()
        .and_then(|t| t.as_any()?.downcast_ref::<MorphogenesisTool>())
        .expect("active tool must be MorphogenesisTool")
}

fn morpho_tool_mut(tab: &mut MorphogenesisTab) -> &mut MorphogenesisTool {
    tab.framework
        .active_tool
        .as_mut()
        .and_then(|t| t.as_any_mut()?.downcast_mut::<MorphogenesisTool>())
        .expect("active tool must be MorphogenesisTool")
}

fn morpho_state(tab: &MorphogenesisTab) -> MorphogenesisToolState {
    let json = tab.framework.active_tool.as_ref().unwrap().save_state();
    serde_json::from_str(&json.expect("MorphogenesisTool state"))
        .expect("valid Morphogenesis state")
}

fn render_tab_frame(ctx: &egui::Context, tab: &mut MorphogenesisTab) {
    let mut frame = eframe::Frame::_new_kittest();
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        CentralPanel::default()
            .frame(Frame::NONE)
            .show(ctx, |ui| tab.show_inside(ui, &mut frame));
    });
}

fn render_tool_frame(ctx: &egui::Context, tool: &mut MorphogenesisTool) -> egui::FullOutput {
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
    let image = MorphogenesisTool::plot_concentration(&data, 2, 2);
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
    let uniform_image = MorphogenesisTool::plot_concentration(&uniform_data, 2, 2);
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
    let state = morpho_state(&tab);
    assert_eq!(
        (state.a, state.b, state.d_v, state.dt),
        (0.14, 0.86, 50.0, 0.02)
    );
    tab.import_url_preset_json(
        Some("MorphogenesisTool"),
        Some(crate::starter_gallery::PRESET_TURING_SPOTS),
    )
    .expect("URL preset import via default ExplorerTab hook");
    assert_eq!(morpho_tool(&tab).params.read().unwrap().d_v, 100.0);
    let share_params = crate::url_state::ShareUrlParams {
        tab: "morphogenesis".to_string(),
        tool: Some("MorphogenesisTool".to_string()),
        state_json: None,
        preset: Some("turing_stripes".to_string()),
        challenge: None,
        step: None,
    };
    tab.import_url_share_params(&share_params, None)
        .expect("URL share params import via default ExplorerTab hook");
    assert_eq!(morpho_tool(&tab).params.read().unwrap().d_v, 50.0);

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
    assert_eq!(morpho_state(&tab).state, Some(SimWorkerState::Ready));
    let mut prev_rev = tab.state_revision();
    assert!(prev_rev > 0);

    for mutate in [
        |p: &mut MorphogenesisSharedParams| p.a = 0.22,
        |p: &mut MorphogenesisSharedParams| p.b = 1.15,
        |p: &mut MorphogenesisSharedParams| p.d_u = 2.4,
        |p: &mut MorphogenesisSharedParams| p.d_v = 85.0,
        |p: &mut MorphogenesisSharedParams| p.dt = 0.03,
    ] {
        mutate(&mut morpho_tool_mut(&mut tab).params.write().unwrap());
        let next_rev = tab.state_revision();
        assert_ne!(next_rev, prev_rev);
        prev_rev = next_rev;
    }

    let saved = tab.save_state().expect("save_state should succeed");
    assert!(!saved.contains("\"paused\""));
    let mut restored = MorphogenesisTab::default();
    restored.load_state(&saved);
    assert_eq!(morpho_state(&restored), morpho_state(&tab));
    assert_eq!(morpho_state(&restored).state, Some(SimWorkerState::Ready));

    let ctx = egui::Context::default();
    render_tab_frame(&ctx, &mut tab);
    assert_eq!(morpho_state(&tab).state, Some(SimWorkerState::Running));
    let saved_running = tab.save_state().expect("save_state should succeed");
    restored.load_state(&saved_running);
    assert_eq!(morpho_state(&restored).state, Some(SimWorkerState::Running));

    restored.load_state(r#"{"dt":0.02,"paused":true,"show_starter_gallery":true}"#);
    assert!(!restored.framework.show_starter_gallery);
    assert_eq!(morpho_tool(&restored).params.read().unwrap().dt, 0.02);
    assert_eq!(morpho_state(&restored).state, Some(SimWorkerState::Ready));

    restored.load_state(r#"{"dt":0.08,"paused":false,"show_starter_gallery":false}"#);
    assert_eq!(morpho_tool(&restored).params.read().unwrap().dt, 0.08);
    assert_eq!(morpho_state(&restored).state, Some(SimWorkerState::Running));
}

fn assert_tab_png_export(tab: &mut MorphogenesisTab) {
    let ctx = egui::Context::default();
    let mut frame = eframe::Frame::_new_kittest();
    let _ = ctx.run(raw_screen_input(1280.0, 800.0, Vec::new()), |ctx| {
        tab.framework.show_export_modal = true;
        tab.framework.queue_export(ExportFormat::Png);
        CentralPanel::default()
            .frame(Frame::NONE)
            .show(ctx, |ui| tab.show_inside(ui, &mut frame));
    });
    assert_eq!(
        tab.framework.export_status_severity,
        ExportStatusSeverity::Success
    );
    let png = tab.framework.export_queue.last_exported_bytes.as_ref();
    let _ = std::fs::write("/tmp/morphogenesis_turing.png", png.expect("PNG bytes"));
}

#[test]
fn test_morphogenesis_poisoned_lock_and_zero_allocation_step_and_png_export() {
    let mut tab = MorphogenesisTab::default();
    let params_clone = Arc::clone(&morpho_tool(&tab).params);
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut guard = params_clone.write().unwrap();
        guard.a = 0.25;
        guard.dt = 0.04;
        panic!("simulated worker panic");
    }));
    assert!(morpho_tool(&tab).params.is_poisoned());

    let mut runner = MorphogenesisTool::create_runner(&morpho_tool(&tab).params, 20, 20);
    assert_eq!(runner.system.kinetics.a, 0.25);

    runner.step();
    {
        let _alloc_guard = crate::VERIFIED_ALLOC_TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        verified_engine::engine::VerifiedEngine::run_verified(|| runner.step());
    }
    assert!(FiniteStateGuard::is_finite_state(&runner));

    FiniteStateGuard::auto_scale_step_size(morpho_tool_mut(&mut tab));
    let read_dt = morpho_tool(&tab)
        .params
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .dt;
    assert!((read_dt - 0.02).abs() < 1e-12);
    FiniteStateGuard::reset_parameters(morpho_tool_mut(&mut tab));
    let read_a = morpho_tool(&tab)
        .params
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .a;
    assert_eq!(read_a, 0.1);
    wait_for_morphogenesis_snapshot(morpho_tool_mut(&mut tab));
    assert_tab_png_export(&mut tab);
}

fn click_morpho_button(ctx: &egui::Context, tab: &mut MorphogenesisTab, label_substr: &str) {
    let mut frame = eframe::Frame::_new_kittest();
    let out = ctx.run(raw_screen_input(1400.0, 900.0, Vec::new()), |ctx| {
        CentralPanel::default()
            .frame(Frame::NONE)
            .show(ctx, |ui| tab.show_inside(ui, &mut frame));
    });
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
    let btn = |pressed| egui::Event::PointerButton {
        pos: center,
        button: egui::PointerButton::Primary,
        pressed,
        modifiers: egui::Modifiers::NONE,
    };
    let events = vec![egui::Event::PointerMoved(center), btn(true), btn(false)];
    let _ = ctx.run(raw_screen_input(1400.0, 900.0, events), |ctx| {
        CentralPanel::default()
            .frame(Frame::NONE)
            .show(ctx, |ui| tab.show_inside(ui, &mut frame));
    });
}

fn assert_workspace_undo_and_mws_roundtrip(ctx: &egui::Context) {
    let mut app = crate::app::MathExplorerApp::default();
    let mut mgr = crate::app::WorkspaceSessionManager::default();
    let morpho_idx = app
        .tabs
        .iter()
        .position(|t| t.name() == "Morphogenesis")
        .expect("Morphogenesis tab registered");
    app.selected_tab = morpho_idx;
    mgr.track_frame_changes(ctx, &app);
    assert!(!mgr.can_undo());

    let baseline_json = app.tabs[morpho_idx].save_state().unwrap();
    app.tabs[morpho_idx].load_state(r#"{"dt":0.025,"paused":true}"#);
    mgr.track_frame_changes(ctx, &app);
    assert!(mgr.can_undo());

    let mws = mgr.create_envelope(app.get_app_state());
    let parsed = mgr.parse_envelope(&mws).unwrap();
    assert!(parsed.tab_states["Morphogenesis"].contains("0.025"));

    let prev = mgr.pop_undo().expect("undo snapshot");
    assert_eq!(prev.tab_states["Morphogenesis"], baseline_json);
}

#[test]
fn test_morphogenesis_undo_mws_reset_live_region_theory_and_preset_io() {
    let ctx = egui::Context::default();
    ctx.enable_accesskit();

    let mut tab = MorphogenesisTab::default();
    click_morpho_button(&ctx, &mut tab, "Stripes");
    assert_eq!((morpho_state(&tab).a, morpho_state(&tab).dt), (0.14, 0.02));

    let preset_json = morpho_tool(&tab)
        .export_preset_json()
        .expect("morphogenesis preset export");

    click_morpho_button(&ctx, &mut tab, "Reset Parameters");
    let reset = morpho_state(&tab);
    assert_eq!(
        (reset.a, reset.b, reset.d_v, reset.dt),
        (0.1, 0.9, 100.0, 0.05)
    );
    let live_msg = crate::accessibility::with_live_region(|r| r.slots[r.active_slot].clone());
    assert_eq!(live_msg, "Parameters reset to baseline defaults");

    morpho_tool_mut(&mut tab)
        .import_preset_json(&preset_json)
        .expect("morphogenesis preset import");
    assert_eq!((morpho_state(&tab).a, morpho_state(&tab).dt), (0.14, 0.02));

    assert!(!tab.framework.show_theory_portal);
    click_morpho_button(&ctx, &mut tab, "Theory");
    assert!(tab.framework.show_theory_portal);
    render_tab_frame(&ctx, &mut tab);
    assert!(
        scientific_metadata::citation_registry::CitationRegistry::get("Turing Morphogenesis")
            .is_some_and(|c| !c.is_empty())
    );
    assert_workspace_undo_and_mws_roundtrip(&ctx);
}

#[test]
fn test_morphogenesis_defers_start_until_render_and_faults_on_divergence() {
    let ctx = egui::Context::default();
    let mut tab = MorphogenesisTab::default();
    let mut tool = MorphogenesisTool::default();
    assert_eq!(morpho_state(&tab).state, Some(SimWorkerState::Ready));
    assert_eq!(tool.worker.state(), SimWorkerState::Ready);

    thread::sleep(Duration::from_millis(40));
    assert_eq!(morpho_state(&tab).state, Some(SimWorkerState::Ready));
    assert_eq!(tool.worker.state(), SimWorkerState::Ready);

    for _ in 0..400 {
        render_tab_frame(&ctx, &mut tab);
        let _ = render_tool_frame(&ctx, &mut tool);
        if morpho_state(&tab).state == Some(SimWorkerState::Faulted)
            && tool.worker.state() == SimWorkerState::Faulted
        {
            break;
        }
        thread::sleep(Duration::from_millis(10));
    }

    assert_eq!(morpho_state(&tab).state, Some(SimWorkerState::Faulted));
    assert_eq!(tool.worker.state(), SimWorkerState::Faulted);
    assert!(tool.worker.diagnostic_report().is_some());
}
