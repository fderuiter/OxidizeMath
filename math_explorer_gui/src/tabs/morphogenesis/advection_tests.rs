use super::advection::{
    MorphogenesisAdvectionState, MAX_MORPHO_VISCOSITY, MIN_MORPHO_VISCOSITY,
    PRESET_VORTEX_WAKE_MORPHOGENESIS,
};
use super::presets::PRESET_GRAY_SCOTT_MITOSIS;
use super::runner::{
    MorphogenesisKineticsRunner, MorphogenesisKineticsState, MorphogenesisSharedParams,
};
use super::{MorphogenesisTab, MorphogenesisTool, MorphogenesisToolState};
use crate::async_sim::{SimCommand, SimulationRunner};
use crate::framework::recovery::FiniteStateGuard;
use crate::framework::{
    BrushMode, BrushStroke, ExportFormat, ExportStatusSeverity, InteractiveTool, SpatialBrushKit,
};
use crate::presets::MorphogenesisKineticsMode;
use crate::starter_gallery::{PRESET_TURING_SPOTS, PRESET_TURING_STRIPES};
use crate::tabs::ExplorerTab;
use crate::url_state::{encode_share_fragment, parse_share_location, MAX_DECODED_URL_STATE_BYTES};
use eframe::egui::{self, CentralPanel, Frame, Key, Modifiers};
use math_commons::math_kernel::types::flatten_2d_index;
use math_explorer::biology::diffusion::ReactionCellRole;
use math_explorer::physics::fluid_dynamics::lattice_boltzmann::MAX_LATTICE_JET_SPEED;
use std::{sync::Arc, thread, time::Duration};

fn raw_input(w: f32, h: f32, events: Vec<egui::Event>) -> egui::RawInput {
    egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(w, h),
        )),
        events,
        ..Default::default()
    }
}

fn key_press(key: Key) -> egui::Event {
    egui::Event::Key {
        key,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers: Modifiers::NONE,
    }
}

fn wait_for_snapshot(tool: &mut MorphogenesisTool) {
    for _ in 0..600 {
        if tool.worker.update().is_some() || tool.worker.latest_snapshot().is_some() {
            return;
        }
        thread::sleep(Duration::from_millis(5));
    }
}

fn make_advection_test_runner(
    w: usize,
    h: usize,
    enabled: bool,
    f: f64,
    k: f64,
) -> (
    MorphogenesisKineticsRunner,
    Arc<std::sync::RwLock<MorphogenesisAdvectionState>>,
) {
    let params = Arc::new(std::sync::RwLock::new(MorphogenesisSharedParams {
        a: 0.1,
        b: 0.9,
        d_u: 0.16,
        d_v: 0.08,
        dt: 0.9,
        high_contrast: false,
        periodic_boundary: false,
        theme_mode: crate::app::ThemeMode::Dark,
        heatmap_mode: super::counterfactual::MorphogenesisHeatmapMode::ActiveUb,
        mask_rle: None,
    }));
    let ks = Arc::new(std::sync::RwLock::new(MorphogenesisKineticsState {
        mode: MorphogenesisKineticsMode::GrayScott,
        f,
        k,
    }));
    let adv = Arc::new(std::sync::RwLock::new(MorphogenesisAdvectionState {
        enabled,
        inlet_u: 0.12,
        viscosity: 0.015,
    }));
    let mask = Arc::new(std::sync::RwLock::new(Vec::new()));
    let r =
        MorphogenesisTool::create_kinetics_runner_with_advection(&params, &ks, &adv, &mask, w, h);
    (r, adv)
}

fn build_maze_test_runner(enabled: bool) -> MorphogenesisKineticsRunner {
    let (mut r, adv) = make_advection_test_runner(48, 36, enabled, 0.0, 0.0);
    {
        let mut p = r.params.write().unwrap();
        p.d_u = 0.04;
        p.d_v = 0.02;
        p.dt = 0.85;
    }
    adv.write().unwrap().inlet_u = 0.14;
    r.system.u_mut().fill(0.0);
    r.system.v_mut().fill(0.0);
    r.process_command(SimCommand::Custom("stamp_maze".to_string()));
    for _ in 0..220 {
        r.step();
    }
    r
}

#[test]
fn test_coupled_wind_tunnel_50_steps_zero_heap_allocations() {
    let (mut runner, adv) = make_advection_test_runner(36, 24, true, 0.0367, 0.0649);
    runner.process_command(SimCommand::ApplyBrush {
        cx: 10,
        cy: 12,
        r: 3,
        is_obstacle: true,
    });
    assert_eq!(runner.checkpoint_stride(), None);
    runner.step();
    {
        let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        verified_engine::engine::VerifiedEngine::run_verified(|| {
            for _ in 0..50 {
                runner.step();
            }
        });
    }
    assert!(FiniteStateGuard::is_finite_state(&runner));
    assert!(runner.wind_tunnel.max_abs_ux > 0.05);
    adv.write().unwrap().enabled = false;
    assert_eq!(runner.checkpoint_stride(), Some(16));
}

#[test]
fn test_microfluidic_maze_serpentine_plume_transport_and_wall_sealing() {
    let coupled = build_maze_test_runner(true);
    let uncoupled = build_maze_test_runner(false);
    assert!(FiniteStateGuard::is_finite_state(&coupled));
    let mut downstream_coupled = 0.0;
    let mut downstream_uncoupled = 0.0;
    for y in 0..36 {
        for x in 0..48 {
            let idx = flatten_2d_index(x, y, 48);
            if coupled.system.diffusion.mask.get(x, y) == ReactionCellRole::ZeroFluxWall {
                assert_eq!(coupled.system.u()[idx], 0.0);
                assert_eq!(coupled.system.v()[idx], 0.0);
                assert!(coupled.wind_tunnel.solver.is_obstacle(x, y));
            } else if y > 12 {
                downstream_coupled += coupled.system.u()[idx] + coupled.system.v()[idx];
                downstream_uncoupled += uncoupled.system.u()[idx] + uncoupled.system.v()[idx];
            }
        }
    }
    assert!(downstream_coupled > downstream_uncoupled * 1.35 + 1.0);

    let mut maze_tab = MorphogenesisTab::default();
    {
        let tool = maze_tab
            .framework
            .active_tool
            .as_mut()
            .and_then(|t| t.as_any_mut()?.downcast_mut::<MorphogenesisTool>())
            .unwrap();
        let _ = tool.worker.send_command(SimCommand::Pause);
        tool.worker.wait_until_paused();
        let px = coupled.get_snapshot().pixels.read().unwrap().clone();
        tool.stepped_pixels = Some((48, 36, px));
    }
    assert_rendered_morpho_png(&mut maze_tab);
}

fn assert_pointer_brush_syncs_lbm_obstacles(runner: &mut MorphogenesisKineticsRunner) {
    let wall_stroke = BrushStroke {
        start: [14.5, 21.5],
        end: [18.5, 21.5],
        radius: 1.0,
        mode: BrushMode::ZeroFluxWall,
        drag_vector: [4.0, 0.0],
    };
    runner.process_command(SimCommand::ApplyBrushStroke(wall_stroke));
    let (gx, gy) = SpatialBrushKit::plot_to_grid_y_down([16.5, 21.5], runner.width, runner.height);
    let (ux, uy) = (gx as usize, gy as usize);
    assert_eq!(
        runner.system.diffusion.mask.get(ux, uy),
        ReactionCellRole::ZeroFluxWall
    );
    assert!(runner.wind_tunnel.solver.is_obstacle(ux, uy));

    let erase_stroke = BrushStroke {
        mode: BrushMode::Erase,
        ..wall_stroke
    };
    runner.process_command(SimCommand::ApplyBrushStroke(erase_stroke));
    assert_eq!(
        runner.system.diffusion.mask.get(ux, uy),
        ReactionCellRole::Free
    );
    assert!(!runner.wind_tunnel.solver.is_obstacle(ux, uy));
}

fn run_tool_keys(ctx: &egui::Context, tool: &mut MorphogenesisTool, keys: Vec<egui::Event>) {
    let _ = ctx.run(raw_input(1280.0, 800.0, keys), |ctx| {
        CentralPanel::default()
            .frame(Frame::NONE)
            .show(ctx, |ui| tool.show_inside(ui));
    });
}

#[test]
fn test_pointer_brush_syncs_cell_mask_and_lbm_obstacles() {
    let (mut runner, _) = make_advection_test_runner(40, 40, true, 0.0367, 0.0649);
    assert_pointer_brush_syncs_lbm_obstacles(&mut runner);
}

fn assert_worker_wall_sync(tool: &mut MorphogenesisTool, x: usize, y: usize) {
    let idx = flatten_2d_index(x, y, 100);
    let mut synced = false;
    for _ in 0..400 {
        let _ = tool.worker.update();
        let role = tool.worker.latest_snapshot().and_then(|s| {
            s.structured_data
                .as_ref()?
                .downcast_ref::<super::runner::MorphoSnapshotData>()
                .and_then(|m| m.mask_roles.get(idx).copied())
        });
        if role == Some(ReactionCellRole::ZeroFluxWall as u8) {
            synced = true;
            break;
        }
        thread::sleep(Duration::from_millis(5));
    }
    assert!(synced, "worker must apply keyboard wall stroke");
}

#[test]
fn test_keyboard_reticle_syncs_cell_mask_and_lbm_obstacles() {
    let ctx = egui::Context::default();
    let mut tool = MorphogenesisTool::default();
    tool.params.write().unwrap().dt = 0.002;
    wait_for_snapshot(&mut tool);
    run_tool_keys(&ctx, &mut tool, Vec::new());
    for _ in 0..120 {
        if tool.brush_state.focused_id.is_some() {
            break;
        }
        run_tool_keys(&ctx, &mut tool, vec![key_press(Key::Tab)]);
    }
    let canvas_id = tool.brush_state.focused_id.expect("focused raster canvas");
    let (rx, ry) = tool.brush_state.reticle;
    let (kx, ky) = ((rx + 1).min(99), ry);
    run_tool_keys(
        &ctx,
        &mut tool,
        vec![key_press(Key::ArrowRight), key_press(Key::Space)],
    );
    assert!(tool.local_runner.wind_tunnel.solver.is_obstacle(kx, ky));
    assert_worker_wall_sync(&mut tool, kx, ky);
    let synced = MorphogenesisTool::create_kinetics_runner_with_advection(
        &tool.params,
        &tool.kinetics_state,
        &tool.advection_state,
        &tool.mask_rle,
        100,
        100,
    );
    assert!(synced.wind_tunnel.solver.is_obstacle(kx, ky));
    ctx.memory_mut(|m| m.request_focus(canvas_id));
    run_tool_keys(&ctx, &mut tool, vec![key_press(Key::Backspace)]);
    assert!(!tool.local_runner.wind_tunnel.solver.is_obstacle(kx, ky));
    let erased = MorphogenesisTool::create_kinetics_runner_with_advection(
        &tool.params,
        &tool.kinetics_state,
        &tool.advection_state,
        &tool.mask_rle,
        100,
        100,
    );
    assert!(!erased.wind_tunnel.solver.is_obstacle(kx, ky));
}

#[test]
fn test_baseline_presets_omit_default_advection_fields() {
    let mut tab = MorphogenesisTab::default();
    let cases = [
        ("Turing Spots", PRESET_TURING_SPOTS),
        ("Turing Stripes", PRESET_TURING_STRIPES),
        ("Gray-Scott Mitosis", PRESET_GRAY_SCOTT_MITOSIS),
    ];
    for (title, preset_json) in cases {
        tab.framework
            .apply_starter_card(title, Some("MorphogenesisTool"), Some(preset_json))
            .expect("baseline preset loads");
        let saved = tab
            .framework
            .active_tool
            .as_ref()
            .unwrap()
            .save_state()
            .unwrap();
        assert!(
            !saved.contains("advection_enabled")
                && !saved.contains("inlet_u")
                && !saved.contains("viscosity")
        );
    }
}

#[test]
fn test_vortex_wake_url_roundtrip_and_extreme_cfl_guard() {
    let mut tab = MorphogenesisTab::default();
    tab.framework
        .apply_starter_card(
            "Vortex Wake Morphogenesis",
            Some("MorphogenesisTool"),
            Some(PRESET_VORTEX_WAKE_MORPHOGENESIS),
        )
        .expect("Vortex Wake Morphogenesis starter loads");
    let saved_vortex = tab
        .framework
        .active_tool
        .as_ref()
        .unwrap()
        .save_state()
        .unwrap();
    assert!(saved_vortex.len() < MAX_DECODED_URL_STATE_BYTES);
    let fragment = encode_share_fragment(
        "morphogenesis",
        Some("MorphogenesisTool"),
        Some(&saved_vortex),
    )
    .expect("encode vortex URL");
    let decoded = parse_share_location(&fragment, "").unwrap().unwrap();
    let mut restored = MorphogenesisTool::default();
    restored
        .try_load_state(decoded.state_json.as_deref().unwrap())
        .expect("restore vortex URL");
    let parsed: MorphogenesisToolState =
        serde_json::from_str(&restored.save_state().unwrap()).unwrap();
    assert!(parsed.advection_enabled && (parsed.inlet_u - 0.12).abs() < 1e-12);
    assert!((parsed.viscosity - 0.015).abs() < 1e-12);

    let extreme = MorphogenesisToolState {
        d_u: 1.0,
        d_v: 1.0,
        dt: 2.0,
        inlet_u: MAX_LATTICE_JET_SPEED,
        viscosity: MIN_MORPHO_VISCOSITY,
        ..parsed
    };
    restored
        .try_load_state(&serde_json::to_string(&extreme).unwrap())
        .expect("extreme valid state loads");
    for _ in 0..8 {
        restored.step_simulation();
    }
    assert!(FiniteStateGuard::is_finite_state(&restored));
    const { assert!(MAX_MORPHO_VISCOSITY > MIN_MORPHO_VISCOSITY) };
}

#[test]
fn test_advected_kinetics_mask_rle_roundtrip() {
    let mut grid = vec![0u8; 100 * 100];
    grid[50 * 100 + 50] = 1;
    let rle = crate::framework::encode_u8_grid(100, 100, &grid, 2).expect("encode RLE");
    let tool = MorphogenesisTool::default();
    tool.advection_state.write().unwrap().enabled = true;
    tool.params.write().unwrap().mask_rle = Some(rle.clone());
    let mut restored = MorphogenesisTool::default();
    restored
        .try_load_state(&tool.save_state().unwrap())
        .expect("load state");
    assert!(restored.advection_state.read().unwrap().enabled);
    let actual = restored.params.read().unwrap().mask_rle.clone();
    assert_eq!(actual.as_deref(), Some(rle.as_str()));
}

fn assert_rendered_morpho_png(tab: &mut MorphogenesisTab) {
    let ctx = egui::Context::default();
    let mut frame = eframe::Frame::_new_kittest();
    let _ = ctx.run(raw_input(1360.0, 860.0, Vec::new()), |ctx| {
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
    let bytes = tab
        .framework
        .export_queue
        .last_exported_bytes
        .as_ref()
        .expect("exported PNG bytes");
    assert!(bytes.starts_with(b"\x89PNG"));
    let reader = png::Decoder::new(std::io::Cursor::new(bytes))
        .read_info()
        .expect("decode exported PNG");
    assert!(reader.info().width > 0 && reader.info().height > 0);
    tab.framework.show_export_modal = false;
}

fn setup_vortex_wake_tab_for_cfl_test() -> MorphogenesisTab {
    let mut tab = MorphogenesisTab::default();
    tab.framework
        .apply_starter_card(
            "Vortex Wake Morphogenesis",
            Some("MorphogenesisTool"),
            Some(PRESET_VORTEX_WAKE_MORPHOGENESIS),
        )
        .expect("load vortex wake preset");
    {
        let tool = tab
            .framework
            .active_tool
            .as_mut()
            .and_then(|t| t.as_any_mut()?.downcast_mut::<MorphogenesisTool>())
            .unwrap();
        let _ = tool.worker.send_command(SimCommand::Pause);
        tool.worker.wait_until_paused();
        for _ in 0..55 {
            tool.step_simulation();
        }
        tool.params.write().unwrap().dt = 1.9;
    }
    assert_rendered_morpho_png(&mut tab);
    tab
}

#[test]
fn test_high_contrast_keyboard_cfl_stabilize_live_region_and_png_exports() {
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    crate::accessibility::clear_test_announcements();
    let mut tab = setup_vortex_wake_tab_for_cfl_test();

    let mut frame = eframe::Frame::_new_kittest();
    crate::app::ThemeMode::HighContrast.apply_to_context(&ctx);
    let out = ctx.run(raw_input(1400.0, 900.0, Vec::new()), |ctx| {
        CentralPanel::default()
            .frame(Frame::NONE)
            .show(ctx, |ui| tab.show_inside(ui, &mut frame));
    });
    let target_node_id = out
        .platform_output
        .accesskit_update
        .as_ref()
        .unwrap()
        .nodes
        .iter()
        .find(|(_, n)| {
            n.label()
                .is_some_and(|l| l.contains("Stabilize Step (CFL)"))
        })
        .map(|(nid, _)| nid.0)
        .expect("CFL stabilize button in AccessKit tree");

    for _ in 0..140 {
        if ctx.memory(|m| m.focused().map(|id| id.value())) == Some(target_node_id) {
            break;
        }
        let _ = ctx.run(raw_input(1400.0, 900.0, vec![key_press(Key::Tab)]), |ctx| {
            CentralPanel::default()
                .frame(Frame::NONE)
                .show(ctx, |ui| tab.show_inside(ui, &mut frame));
        });
    }
    assert_eq!(
        ctx.memory(|m| m.focused().map(|id| id.value())),
        Some(target_node_id)
    );
    let _ = ctx.run(
        raw_input(1400.0, 900.0, vec![key_press(Key::Enter)]),
        |ctx| {
            CentralPanel::default()
                .frame(Frame::NONE)
                .show(ctx, |ui| tab.show_inside(ui, &mut frame));
        },
    );
    let live_msg = crate::accessibility::with_live_region(|r| r.slots[r.active_slot].clone());
    assert!(live_msg.starts_with("Time step stabilized to CFL bound dt ="));
    let tool = tab
        .framework
        .active_tool
        .as_ref()
        .and_then(|t| t.as_any()?.downcast_ref::<MorphogenesisTool>())
        .unwrap();
    assert!(tool.advection_diagnostics().3 < 1.0);
    assert_eq!(
        tool.last_theme_mode,
        Some(crate::app::ThemeMode::HighContrast)
    );
}
