//! Unit, accessibility, export, URL round-trip, cross-tab handoff, and `VerifiedAllocator`
//! tests for [`super::AdvectiveMorphogenesisTool`].

#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::{AdvectiveKineticsPreset, AdvectiveMorphogenesisTool};
#[cfg(feature = "physics")]
use crate::app::{domain_bar, MathExplorerApp};
use crate::framework::recovery::FiniteStateGuard;
use crate::framework::{
    encode_mask_hex, BrushMode, BrushStroke, ExportFormat, ExportStatusSeverity, FairingStamp,
    InteractionContext, InteractiveTool, SimulationFramework, LBM_CELL_COUNT, LBM_HEIGHT,
    LBM_WIDTH,
};
use crate::url_state::{encode_share_fragment, parse_share_location, URL_JSON_BOUNDS};
use eframe::egui;
use math_commons::math_kernel::types::flatten_2d_index;
use verified_engine::allocator::{lock_allocations, unlock_allocations};
use verified_engine::metrics::{disable_verification, enable_verification};

struct VerificationPause;

impl VerificationPause {
    fn new() -> Self {
        disable_verification();
        Self
    }
}

impl Drop for VerificationPause {
    fn drop(&mut self) {
        enable_verification();
    }
}

fn build_stepped_tool(
    preset: AdvectiveKineticsPreset,
    peclet: f64,
    steps: usize,
) -> AdvectiveMorphogenesisTool {
    let mut tool = AdvectiveMorphogenesisTool::default();
    tool.apply_kinetics_preset(preset);
    tool.cfg.viscosity = 0.020;
    tool.cfg.peclet = peclet;
    tool.engine.reset_fluid_wake(0.020);
    tool.engine.reseed_morphogens(&tool.cfg);
    for _ in 0..steps {
        tool.engine.step_coupled(&tool.cfg);
    }
    tool.refresh_pixels_in_place();
    tool
}

fn run_framework_export(fw: &mut SimulationFramework, fmt: ExportFormat) {
    let ctx = egui::Context::default();
    let raw_input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::Vec2::new(1024.0, 768.0),
        )),
        ..Default::default()
    };
    let _ = ctx.run(raw_input.clone(), |ctx| {
        egui_plot::clear_last_plot_datasets();
        fw.show_export_modal = true;
        fw.queue_export(fmt);
        fw.show(ctx, "biology_advective_export_test");
    });
    for _ in 0..250 {
        if fw.active_gif_progress().is_none() {
            break;
        }
        #[cfg(not(target_arch = "wasm32"))]
        std::thread::sleep(std::time::Duration::from_millis(10));
        let _ = ctx.run(raw_input.clone(), |ctx| {
            fw.show(ctx, "biology_advective_export_test");
        });
    }
    assert_eq!(
        fw.export_status_severity,
        ExportStatusSeverity::Success,
        "Export {fmt:?} failed: {:?}",
        fw.export_status_message
    );
}

fn export_tool_png(tool: AdvectiveMorphogenesisTool, path: &str) {
    let mut fw = SimulationFramework::new("__advective_png__");
    fw.export_options.width = 800;
    fw.export_options.height = 400;
    fw.active_tool = Some(Box::new(tool));
    run_framework_export(&mut fw, ExportFormat::Png);
    if let Some(bytes) = fw.export_queue.last_exported_bytes.as_deref() {
        let _ = std::fs::write(path, bytes);
    }
}

#[test]
fn test_peclet_gain_stretches_schnakenberg_and_gray_scott_plumes_in_vortex_wake() {
    let _pause = VerificationPause::new();
    let low_pe_schnak = build_stepped_tool(AdvectiveKineticsPreset::SchnakenbergSpots, 0.0, 110);
    let high_pe_schnak = build_stepped_tool(AdvectiveKineticsPreset::SchnakenbergSpots, 18.0, 110);

    assert!(high_pe_schnak.engine.metrics.max_vorticity > 0.01);
    assert!(high_pe_schnak.engine.metrics.f_drag > 0.0);
    assert!(
        high_pe_schnak.engine.metrics.plume_elongation
            > low_pe_schnak.engine.metrics.plume_elongation * 1.12,
        "Expected rising Peclet gain to stretch Turing spots into streamwise plumes: Pe=0 {:.3} vs Pe=18 {:.3}",
        low_pe_schnak.engine.metrics.plume_elongation,
        high_pe_schnak.engine.metrics.plume_elongation
    );
    export_tool_png(
        high_pe_schnak,
        "/tmp/advective_schnakenberg_vortex_spots.png",
    );

    let low_pe_gs = build_stepped_tool(AdvectiveKineticsPreset::GrayScottCoral, 0.0, 110);
    let high_pe_gs = build_stepped_tool(AdvectiveKineticsPreset::GrayScottCoral, 6.5, 110);
    assert!(
        high_pe_gs.engine.metrics.plume_elongation
            > low_pe_gs.engine.metrics.plume_elongation * 1.12,
        "Expected rising Peclet gain to shear Gray-Scott coral into wake plumes: Pe=0 {:.3} vs Pe=6.5 {:.3}",
        low_pe_gs.engine.metrics.plume_elongation,
        high_pe_gs.engine.metrics.plume_elongation
    );
    export_tool_png(high_pe_gs, "/tmp/advective_gray_scott_coral_wake.png");
}

fn verify_keyboard_canvas_sculpting(tool: &mut AdvectiveMorphogenesisTool) {
    let ctx = egui::Context::default();
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| tool.show_inside(ui));
    });
    let click_pos = egui::pos2(180.0, 110.0);
    let mut click_input = egui::RawInput::default();
    click_input
        .events
        .push(egui::Event::PointerMoved(click_pos));
    for pressed in [true, false] {
        click_input.events.push(egui::Event::PointerButton {
            pos: click_pos,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::NONE,
        });
    }
    let _ = ctx.run(click_input, |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| tool.show_inside(ui));
    });
    assert!(tool.focused_canvas_id().is_some());

    let cursor_before = tool.cursor;
    let target_x = (cursor_before[0] + 1).min(LBM_WIDTH - 1);
    let was_solid = tool.engine.mask[flatten_2d_index(target_x, cursor_before[1], LBM_WIDTH)];
    let mut key_input = egui::RawInput::default();
    for key in [egui::Key::ArrowRight, egui::Key::Space] {
        key_input.events.push(egui::Event::Key {
            key,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        });
    }
    let _ = ctx.run(key_input, |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| tool.show_inside(ui));
    });
    assert_eq!(tool.cursor[0], target_x);
    assert_eq!(
        tool.engine.mask[flatten_2d_index(target_x, cursor_before[1], LBM_WIDTH)],
        !was_solid
    );
}

#[test]
fn test_mouse_and_keyboard_obstacle_sculpting_and_fairing_drag() {
    let _pause = VerificationPause::new();
    let mut tool = AdvectiveMorphogenesisTool::default();
    tool.cfg.viscosity = 0.020;
    tool.apply_fairing_stamp(FairingStamp::BluffCylinder);
    tool.engine.reset_fluid_wake(0.020);
    for _ in 0..100 {
        tool.engine.step_coupled(&tool.cfg);
    }
    let cd_cyl = tool.engine.metrics.c_d;

    tool.apply_fairing_stamp(FairingStamp::TeardropFairing);
    tool.engine.reset_fluid_wake(0.020);
    for _ in 0..100 {
        tool.engine.step_coupled(&tool.cfg);
    }
    let cd_tear = tool.engine.metrics.c_d;
    assert!(
        cd_cyl > 0.0 && cd_tear > 0.0 && cd_tear < cd_cyl,
        "Expected teardrop C_D ({cd_tear:.4}) < bluff cylinder C_D ({cd_cyl:.4})"
    );

    let stroke = BrushStroke {
        mode: BrushMode::Solid,
        start: [55.5, 25.5],
        end: [58.5, 25.5],
        radius: 1.0,
        drag_vector: [3.0, 0.0],
    };
    let ctx = egui::Context::default();
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            let (_, resp) = ui.allocate_exact_size(egui::vec2(100.0, 50.0), egui::Sense::click());
            let brush_ctx = InteractionContext::from_brush_stroke(&resp, stroke);
            tool.on_brush(&brush_ctx);
        });
    });
    assert!(tool.engine.mask[flatten_2d_index(56, 25, LBM_WIDTH)]);
    assert!(tool.cfg.mask_hex.as_ref().is_some_and(|h| h.len() == 1250));

    verify_keyboard_canvas_sculpting(&mut tool);
}

fn verify_share_url_fragment(tool: &AdvectiveMorphogenesisTool) {
    let json = tool.save_state().expect("save_state must succeed");
    assert!(
        json.len() < 4096,
        "Serialized state {} bytes >= 4 KiB",
        json.len()
    );
    URL_JSON_BOUNDS.validate_json(&json).unwrap();
    assert_eq!(encode_mask_hex(&tool.engine.mask).len(), 1250);

    let fragment = encode_share_fragment(
        "morphogenesis",
        Some("AdvectiveMorphogenesisTool"),
        Some(&json),
    )
    .expect("share fragment must encode under URL caps");
    let parsed = parse_share_location(&fragment, "")
        .expect("share fragment must parse")
        .expect("share params present");
    assert_eq!(parsed.tool.as_deref(), Some("AdvectiveMorphogenesisTool"));

    let mut restored = AdvectiveMorphogenesisTool::default();
    restored
        .try_load_state(parsed.state_json.as_deref().unwrap())
        .unwrap();
    assert_eq!(restored.cfg, tool.cfg);
    assert_eq!(restored.engine.mask, tool.engine.mask);
}

#[test]
fn test_url_round_trip_under_4kib_and_csv_png_exports() {
    let _pause = VerificationPause::new();
    let mut tool = AdvectiveMorphogenesisTool::default();
    tool.apply_kinetics_preset(AdvectiveKineticsPreset::GrayScottMitosis);
    tool.cfg.peclet = 4.5;
    tool.cfg.viscosity = 0.020;
    let mut custom_mask = [false; LBM_CELL_COUNT];
    FairingStamp::SymmetricWedge.write_mask(&mut custom_mask);
    custom_mask[flatten_2d_index(48, 24, LBM_WIDTH)] = true;
    tool.engine.mask = custom_mask;
    tool.sync_custom_mask_mutation();
    tool.step_once();

    verify_share_url_fragment(&tool);
    assert_eq!(tool.export_datasets().len(), 3);
    assert_eq!(
        tool.export_raster_image().unwrap().size,
        [LBM_WIDTH, LBM_HEIGHT]
    );

    let mut fw = SimulationFramework::new("biology");
    assert!(fw.select_tool_by_name("AdvectiveMorphogenesisTool"));
    fw.export_options.width = 200;
    fw.export_options.height = 100;
    fw.export_gif_frames = 2;

    run_framework_export(&mut fw, ExportFormat::Csv);
    let csv = fw.export_queue.last_exported_content.as_deref().unwrap();
    assert!(csv.contains("centerline_morphogen") && csv.contains("shear_layer_vorticity_x10"));

    run_framework_export(&mut fw, ExportFormat::Png);
    assert!(fw
        .export_queue
        .last_exported_bytes
        .as_deref()
        .unwrap()
        .starts_with(b"\x89PNG"));

    run_framework_export(&mut fw, ExportFormat::Gif);
    assert!(fw
        .export_queue
        .last_exported_bytes
        .as_deref()
        .unwrap()
        .starts_with(b"GIF89a"));
}

#[test]
fn test_cfl_clamp_and_finite_state_guard_recovery() {
    let mut tool = AdvectiveMorphogenesisTool::default();
    tool.cfg.dt = 0.95;
    tool.cfg.peclet = 20.0;
    tool.step_once();
    assert!(tool.engine.metrics.cfl_clamped);
    assert!(tool.engine.metrics.effective_dt <= tool.engine.metrics.cfl_limit + 1e-12);
    assert!(FiniteStateGuard::is_finite_state(&tool));

    tool.cfg.peclet = f64::NAN;
    tool.engine.schnak.u_mut()[0] = f64::INFINITY;
    assert!(FiniteStateGuard::diverged(&tool));
    FiniteStateGuard::clamp_parameters(&mut tool);
    assert!(FiniteStateGuard::is_finite_state(&tool));
}

#[cfg(feature = "physics")]
#[test]
fn test_fluid_dynamics_shortcut_launches_advective_morphogenesis_100x50() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let mut app = MathExplorerApp::default();
    let ctx = egui::Context::default();

    let fluid_idx = app
        .tabs
        .iter()
        .position(|t| t.name() == "Fluid Dynamics")
        .expect("Fluid Dynamics tab present");
    app.selected_tab = fluid_idx;

    let fluid_tab = crate::tabs::fluid_dynamics::FluidDynamicsTab::default();
    fluid_tab.launch_advective_morphogenesis_studio(&ctx);
    domain_bar::poll_cross_tab_tool_handoffs(&mut app, &ctx);

    assert_eq!(app.tabs[app.selected_tab].name(), "Morphogenesis");
    assert_eq!(
        app.tabs[app.selected_tab].active_tool_id(),
        Some("AdvectiveMorphogenesisTool")
    );
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            if let Some(fw) = app.tabs[app.selected_tab].framework_mut() {
                fw.show_inside(ui, "biology");
            }
        });
    });
}

#[test]
fn test_verified_allocator_zero_heap_allocations_across_schnakenberg_and_gray_scott_ticks() {
    let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());

    let mut schnak_tool = AdvectiveMorphogenesisTool::default();
    schnak_tool.apply_kinetics_preset(AdvectiveKineticsPreset::SchnakenbergSpots);

    let mut gs_tool = AdvectiveMorphogenesisTool::default();
    gs_tool.apply_kinetics_preset(AdvectiveKineticsPreset::GrayScottCoral);

    enable_verification();
    lock_allocations();
    for _ in 0..5 {
        schnak_tool.engine.step_coupled(&schnak_tool.cfg);
        gs_tool.engine.step_coupled(&gs_tool.cfg);
    }
    unlock_allocations();
    disable_verification();

    assert!(FiniteStateGuard::is_finite_state(&schnak_tool));
    assert!(FiniteStateGuard::is_finite_state(&gs_tool));
}
