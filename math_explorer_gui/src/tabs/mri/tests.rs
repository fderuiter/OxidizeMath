#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::runner::MAX_MRI_STEPS_PER_FRAME;
use super::*;
use crate::async_sim::{SimCommand, SimWorkerState, SimulationRunner};
use crate::framework::{ExportFormat, ExportStatusSeverity, InteractiveTool};
use egui::{CentralPanel, Frame};
use egui_plot::PlotPoint;
use math_explorer::physics::mri::bloch::BlochSimulator;
use math_explorer::pure_math::analysis::ode::RungeKutta4;
use nalgebra::Vector3;
use std::collections::VecDeque;
use std::sync::Arc;

fn active_mri_state(tab: &MriTab) -> MriToolState {
    let json = tab
        .framework
        .active_tool
        .as_ref()
        .and_then(|t| t.save_state())
        .expect("MriTool must serialize state");
    serde_json::from_str(&json).expect("valid MriToolState")
}

#[test]
fn test_mri_tool_initial_state() {
    let tool = MriTool::default();
    assert_eq!(tool.worker.state(), SimWorkerState::Ready);
    assert_eq!(tool.time_scale, 1.0);
    assert_eq!(tool.current_time, 0.0);
}

#[test]
fn test_mri_worker_simulation_step_and_zero_allocation() {
    let mut tool = MriTool::default();
    let _ = tool.worker.send_command(SimCommand::Start);
    let mut stepped = false;
    for _ in 0..200 {
        std::thread::sleep(std::time::Duration::from_millis(20));
        if let Some(snap) = tool.worker.update() {
            if snap.custom_data.first().copied().unwrap_or(0.0) > 0.0 {
                stepped = true;
                break;
            }
        }
    }
    assert!(stepped);
    let _ = tool.worker.send_command(SimCommand::Pause);
    tool.worker.wait_until_paused();

    // Verify MriRunner::step performs zero heap allocations under VerifiedAllocator
    let params = Arc::new(std::sync::RwLock::new(MriSharedParams {
        t1: 1.0,
        t2: 0.25,
        b_field: Vector3::new(0.0, 0.0, 1.0e-7),
        time_scale: 5.0,
    }));
    let mut runner = MriRunner::new(Arc::clone(&params), Vector3::new(1.0, 0.0, 0.0), 1.0);
    {
        let _alloc_guard = crate::VERIFIED_ALLOC_TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        verified_engine::engine::VerifiedEngine::run_verified(|| {
            runner.step();
        });
    }

    params.write().unwrap().time_scale = 1e12;
    runner.step();
    assert_eq!(runner.history.len(), 5 + MAX_MRI_STEPS_PER_FRAME + 1);
}

#[test]
fn test_mri_fault_halts_repaints_and_recovers() {
    let ctx = egui::Context::default();
    let mut tool = MriTool::default();

    let _ = ctx.run(egui::RawInput::default(), |ctx| tool.show_ui(ctx));
    let output_ready = ctx.run(egui::RawInput::default(), |ctx| tool.show_ui(ctx));
    assert_eq!(tool.worker.state(), SimWorkerState::Ready);
    let delay_ready = output_ready
        .viewport_output
        .get(&egui::ViewportId::ROOT)
        .unwrap()
        .repaint_delay;
    assert_ne!(delay_ready, std::time::Duration::ZERO);

    let _ = tool.worker.send_command(SimCommand::Start);
    let output_running = ctx.run(egui::RawInput::default(), |ctx| tool.show_ui(ctx));
    assert_eq!(tool.worker.state(), SimWorkerState::Running);
    let delay_running = output_running
        .viewport_output
        .get(&egui::ViewportId::ROOT)
        .unwrap()
        .repaint_delay;
    assert_eq!(delay_running, std::time::Duration::ZERO);

    let _ = tool.worker.send_command(SimCommand::Pause);
    tool.worker.wait_until_paused();
    while tool.worker.update().is_some() {}

    tool.worker.set_state_for_test(SimWorkerState::Faulted);
    let _ = ctx.run(egui::RawInput::default(), |ctx| tool.show_ui(ctx));
    let output_faulted = ctx.run(egui::RawInput::default(), |ctx| tool.show_ui(ctx));
    let delay_faulted = output_faulted
        .viewport_output
        .get(&egui::ViewportId::ROOT)
        .unwrap()
        .repaint_delay;
    assert_ne!(delay_faulted, std::time::Duration::ZERO);

    tool.retry();
    assert_eq!(tool.worker.state(), SimWorkerState::Ready);
}

#[test]
fn test_mri_all_6_controls_persist_hash_and_reset_restores_defaults() {
    let mut tool = MriTool::default();
    let mut prev_rev = tool.state_revision();
    assert!(prev_rev > 0);

    for mutate in [
        |t: &mut MriTool| {
            t.simulator.t1 = 2.4;
            t.params.write().unwrap().t1 = 2.4;
        },
        |t: &mut MriTool| {
            t.simulator.t2 = 0.45;
            t.params.write().unwrap().t2 = 0.45;
        },
        |t: &mut MriTool| {
            t.simulator.b_field.z = 1.75;
            t.params.write().unwrap().b_field.z = 1.75;
        },
        |t: &mut MriTool| {
            t.simulator.b_field.x = 0.03;
            t.params.write().unwrap().b_field.x = 0.03;
        },
        |t: &mut MriTool| {
            t.simulator.b_field.y = -0.02;
            t.params.write().unwrap().b_field.y = -0.02;
        },
        |t: &mut MriTool| {
            t.time_scale = 12.0;
            t.params.write().unwrap().time_scale = 12.0;
        },
    ] {
        mutate(&mut tool);
        let next_rev = tool.state_revision();
        assert_ne!(next_rev, prev_rev);
        prev_rev = next_rev;
    }

    let saved = tool.save_state().expect("save_state must succeed");
    let mut restored = MriTool::default();
    restored.try_load_state(&saved).unwrap();
    assert_eq!(restored.save_state().as_deref(), Some(saved.as_str()));
    assert_eq!(restored.simulator.t1, 2.4);
    assert_eq!(restored.simulator.t2, 0.45);
    assert_eq!(restored.simulator.b_field, Vector3::new(0.03, -0.02, 1.75));
    assert_eq!(restored.time_scale, 12.0);

    // Requirement 6: reset() restores default t1, t2, b_field, and time_scale
    restored.reset();
    assert_eq!(restored.simulator.t1, 1.0);
    assert_eq!(restored.simulator.t2, 0.1);
    assert_eq!(restored.simulator.b_field, Vector3::new(0.0, 0.0, 1.0));
    assert_eq!(restored.time_scale, 1.0);
    let shared = restored.params.read().unwrap();
    assert_eq!(shared.t1, 1.0);
    assert_eq!(shared.t2, 0.1);
    assert_eq!(shared.b_field, Vector3::new(0.0, 0.0, 1.0));
    assert_eq!(shared.time_scale, 1.0);
}

#[test]
fn test_mri_save_load_legacy_compatibility_and_exports() {
    let tab = MriTab::default();
    let saved = tab.save_state().expect("save_state should succeed");
    let mut loaded = MriTab::default();
    loaded.load_state(&saved);
    assert_eq!(active_mri_state(&loaded), active_mri_state(&tab));

    // Legacy paused and running flat payloads
    loaded.load_state(r#"{"paused":true,"time_scale":2.5}"#);
    assert_eq!(active_mri_state(&loaded).state, Some(SimWorkerState::Ready));
    assert_eq!(active_mri_state(&loaded).time_scale, 2.5);

    loaded.load_state(r#"{"paused":false,"time_scale":3.0}"#);
    assert_eq!(
        active_mri_state(&loaded).state,
        Some(SimWorkerState::Running)
    );
    assert_eq!(active_mri_state(&loaded).time_scale, 3.0);

    // Legacy clamping
    loaded.load_state(r#"{"state":"Ready","time_scale":1e12}"#);
    assert_eq!(active_mri_state(&loaded).time_scale, MAX_MRI_TIME_SCALE);
    loaded.load_state(r#"{"state":"Ready","time_scale":-10.0}"#);
    assert_eq!(active_mri_state(&loaded).time_scale, MIN_MRI_TIME_SCALE);

    // Export CSV, JSON, PNG
    let ctx = egui::Context::default();
    let mut frame = eframe::Frame::_new_kittest();
    let screen_rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1280.0, 800.0));
    for fmt in [ExportFormat::Csv, ExportFormat::Json, ExportFormat::Png] {
        let _ = ctx.run(
            egui::RawInput {
                screen_rect: Some(screen_rect),
                ..Default::default()
            },
            |ctx| {
                loaded.framework.queue_export(fmt);
                CentralPanel::default()
                    .frame(Frame::NONE)
                    .show(ctx, |ui| loaded.show_inside(ui, &mut frame));
            },
        );
        assert_eq!(
            loaded.framework.export_status_severity,
            ExportStatusSeverity::Success
        );
    }
}

#[test]
fn test_mri_poisoned_lock_recovery_and_plot_buffers_reuse() {
    let mut tool = MriTool::default();
    let params_clone = Arc::clone(&tool.params);
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _guard = params_clone.write().unwrap();
        panic!("simulated worker thread fault");
    }));
    assert!(tool.params.is_poisoned());

    tool.simulator.t1 = 2.0;
    tool.params.write().unwrap_or_else(|e| e.into_inner()).t1 = tool.simulator.t1;
    tool.simulator.t2 = 0.5;
    tool.params.write().unwrap_or_else(|e| e.into_inner()).t2 = tool.simulator.t2;
    tool.simulator.b_field = Vector3::new(0.05, 0.02, 1.2);
    tool.params
        .write()
        .unwrap_or_else(|e| e.into_inner())
        .b_field = tool.simulator.b_field;
    tool.time_scale = 4.0;
    tool.params
        .write()
        .unwrap_or_else(|e| e.into_inner())
        .time_scale = tool.time_scale;

    tool.retry();
    assert_eq!(tool.worker.state(), SimWorkerState::Ready);
    assert_eq!(tool.simulator.t1, 2.0);

    let mut runner = MriRunner {
        simulator: BlochSimulator::new(Vector3::new(1.0, 0.0, 0.0), 1.0),
        solver: RungeKutta4::new(&Vector3::new(1.0, 0.0, 0.0)),
        params: Arc::clone(&tool.params),
        current_time: 0.0,
        history: VecDeque::with_capacity(100),
        max_history: 100,
    };
    runner.step();
    assert!(runner.current_time > 0.0);

    // Plot buffers reuse capacity across update and reset
    tool.history.push_back((0.0, Vector3::new(1.0, 0.0, 0.0)));
    tool.history.push_back((0.1, Vector3::new(0.8, 0.2, 0.5)));
    tool.update_plot_buffers();
    assert_eq!(tool.mx_points.len(), 2);
    {
        let mx_slice: &[PlotPoint] = &tool.mx_points[..];
        let plot_points_x: egui_plot::PlotPoints<'_> = mx_slice.into();
        assert!(matches!(plot_points_x, egui_plot::PlotPoints::Borrowed(_)));
    }
    let cap_x = tool.mx_points.capacity();

    tool.reset();
    assert!(tool.mx_points.is_empty());
    assert!(tool.history.is_empty());
    assert!(tool.mx_points.capacity() >= cap_x);
}

fn click_mri_button(ctx: &egui::Context, tab: &mut MriTab, label_substr: &str) {
    let mut frame = eframe::Frame::_new_kittest();
    let screen_rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1400.0, 900.0));
    let out = ctx.run(
        egui::RawInput {
            screen_rect: Some(screen_rect),
            ..Default::default()
        },
        |ctx| {
            CentralPanel::default()
                .frame(Frame::NONE)
                .show(ctx, |ui| tab.show_inside(ui, &mut frame));
        },
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
        |ctx| {
            CentralPanel::default()
                .frame(Frame::NONE)
                .show(ctx, |ui| tab.show_inside(ui, &mut frame));
        },
    );
}

#[test]
fn test_mri_reset_live_region_theory_and_preset_io() {
    let ctx = egui::Context::default();
    ctx.enable_accesskit();

    let mut tab = MriTab::default();
    tab.framework
        .active_tool
        .as_mut()
        .unwrap()
        .try_load_state(r#"{"t1":2.5,"t2":0.4,"bx":0.01,"by":-0.01,"bz":1.8,"time_scale":7.0}"#)
        .unwrap();
    click_mri_button(&ctx, &mut tab, "Reset Parameters");
    assert_eq!(active_mri_state(&tab).t1, 1.0);
    assert_eq!(active_mri_state(&tab).t2, 0.1);
    assert_eq!(active_mri_state(&tab).bz, 1.0);
    assert_eq!(active_mri_state(&tab).time_scale, 1.0);
    let live_msg = crate::accessibility::with_live_region(|r| r.slots[r.active_slot].clone());
    assert_eq!(live_msg, "Parameters reset to baseline defaults");

    assert!(tab
        .framework
        .apply_starter_card(
            "Off-Resonance Bloch Precession",
            Some("MriTool"),
            Some(crate::starter_gallery::PRESET_MRI_BLOCH_RELAXATION),
        )
        .is_ok());
    let preset_json = tab
        .framework
        .active_tool
        .as_ref()
        .unwrap()
        .export_preset_json()
        .expect("mri preset export");
    let mut fresh_tab = MriTab::default();
    fresh_tab
        .framework
        .active_tool
        .as_mut()
        .unwrap()
        .import_preset_json(&preset_json)
        .expect("mri preset import");
    assert_eq!(active_mri_state(&fresh_tab).t1, 1.2);

    assert!(!tab.framework.show_theory_portal);
    click_mri_button(&ctx, &mut tab, "Theory");
    assert!(tab.framework.show_theory_portal);
    let mut frame = eframe::Frame::_new_kittest();
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        CentralPanel::default()
            .frame(Frame::NONE)
            .show(ctx, |ui| tab.show_inside(ui, &mut frame));
    });
    assert!(
        scientific_metadata::citation_registry::CitationRegistry::get("MRI Bloch Relaxation")
            .is_some_and(|c| c.contains("Bloch, F. (1946)"))
    );
}

#[test]
fn test_mri_workspace_undo_mws_and_figure_export() {
    let ctx = egui::Context::default();
    let mut tab = MriTab::default();
    assert!(tab
        .framework
        .apply_starter_card(
            "Off-Resonance Bloch Precession",
            Some("MriTool"),
            Some(crate::starter_gallery::PRESET_MRI_BLOCH_RELAXATION),
        )
        .is_ok());
    if let Some(tool) = tab.framework.active_tool.as_mut() {
        for _ in 0..80 {
            tool.step_simulation();
        }
    }
    let mut frame = eframe::Frame::_new_kittest();
    let screen_rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1400.0, 900.0));
    let _ = ctx.run(
        egui::RawInput {
            screen_rect: Some(screen_rect),
            ..Default::default()
        },
        |ctx| {
            tab.framework.queue_export(ExportFormat::Png);
            CentralPanel::default()
                .frame(Frame::NONE)
                .show(ctx, |ui| tab.show_inside(ui, &mut frame));
        },
    );
    let _ = std::fs::write(
        "/tmp/mri_bloch_precession.png",
        tab.framework
            .export_queue
            .last_exported_bytes
            .as_ref()
            .expect("MRI PNG bytes"),
    );

    let mut app = crate::app::MathExplorerApp::default();
    let mut mgr = crate::app::WorkspaceSessionManager::default();
    let mri_idx = app
        .tabs
        .iter()
        .position(|t| t.name() == "MRI Physics")
        .expect("MRI Physics tab registered");
    app.selected_tab = mri_idx;
    mgr.track_frame_changes(&ctx, &app);
    assert!(!mgr.can_undo());

    let baseline_json = app.tabs[mri_idx].save_state().unwrap();
    app.tabs[mri_idx].load_state(r#"{"paused":true,"time_scale":6.5}"#);
    mgr.track_frame_changes(&ctx, &app);
    assert!(mgr.can_undo());

    let mws = mgr.create_envelope(app.get_app_state());
    let parsed = mgr.parse_envelope(&mws).unwrap();
    assert!(parsed.tab_states["MRI Physics"].contains("6.5"));

    let prev = mgr.pop_undo().expect("undo snapshot");
    assert_eq!(prev.tab_states["MRI Physics"], baseline_json);
}
