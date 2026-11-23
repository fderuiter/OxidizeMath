//! Dedicated unit and integration tests for [`super::PlotBrushController`],
//! Bresenham discrete brush strokes across radii `1..=8`, LBM aerodynamic diagnostics
//! and signed vorticity views, persistent Morphogenesis `Emitter`/`Sink`/`Barrier`
//! masks across Schnakenberg and Gray-Scott kinetics + 3D Gaussian splats,
//! `[-500, 500]` freehand quantum potential sculpting + Web Audio sonification,
//! `[-50, 50]` Potential Flow singularity handles + GLSL shader sync, and
//! `focused_canvas_id()` workspace shortcut preservation.

#![cfg_attr(any(), verified(opt_out = "tests"))]

use crate::app::MathExplorerApp;
use crate::async_sim::unified::UnifiedModel;
use crate::async_sim::{SimCommand, SimulationRunner};
use crate::framework::{
    BrushMode, BrushStroke, InteractionContext, InteractiveTool, PlotBrushController,
    PlotBrushFrame, ShaderDialect, SimulationFramework,
};
use crate::presets::MorphogenesisKineticsMode;
use crate::tabs::fluid_dynamics::lattice_boltzmann::{
    LbmSnapshotMetrics, LbmUnified, LBM_HEIGHT, LBM_WIDTH,
};
use crate::tabs::fluid_dynamics::potential_flow::{
    ElementType, PotentialFlowTool, FLOW_POS_MAX, FLOW_POS_MIN,
};
use crate::tabs::fluid_dynamics::FluidDynamicsTab;
use crate::tabs::morphogenesis::{MorphogenesisTab, MorphogenesisTool};
use crate::tabs::quantum::wave_sim::presets::POTENTIAL_DISPLAY_SCALE;
use crate::tabs::quantum::wave_sim::{WaveData, WaveRunner, WaveSimConfig};
use crate::tabs::quantum::QuantumTab;
use crate::tabs::ExplorerTab;
use crate::url_state::{parse_share_location, MAX_DECODED_URL_STATE_BYTES};
use crate::volumetric_scene::presets::build_scene_from_2d_morphogenesis_slice;
use eframe::egui::{self, Event, Key, Modifiers, Pos2, RawInput, Vec2};
use math_commons::math_kernel::types::flatten_2d_index;
use math_explorer::biology::diffusion::ReactionCellRole;
use scientific_metadata::theory::TheoryDescribable;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use std::time::Duration;

struct DummyTheory;

impl TheoryDescribable for DummyTheory {
    fn theory_description(&self) -> String {
        String::new()
    }
    fn phonetic_description(&self) -> String {
        String::new()
    }
    fn theory_citation(&self) -> String {
        String::new()
    }
    fn available_descriptions(&self) -> HashMap<String, String> {
        HashMap::new()
    }
}

#[derive(Default)]
struct RecordingTool {
    keyboard_hits: usize,
    hover_hits: usize,
    click_hits: usize,
    drag_hits: usize,
    brush_strokes: Vec<BrushStroke>,
    last_pos: Option<Pos2>,
}

impl InteractiveTool for RecordingTool {
    fn name(&self) -> &'static str {
        "RecordingTool"
    }

    fn theory(&self) -> &dyn TheoryDescribable {
        &DummyTheory
    }

    fn on_keyboard(&mut self, ctx: &InteractionContext) {
        self.keyboard_hits += 1;
        self.last_pos = ctx.pointer_pos;
    }

    fn on_hover(&mut self, ctx: &InteractionContext) {
        self.hover_hits += 1;
        self.last_pos = ctx.pointer_pos;
    }

    fn on_click(&mut self, ctx: &InteractionContext) {
        self.click_hits += 1;
        self.last_pos = ctx.pointer_pos;
    }

    fn on_drag(&mut self, ctx: &InteractionContext) {
        self.drag_hits += 1;
        self.last_pos = ctx.pointer_pos;
    }

    fn on_brush(&mut self, ctx: &InteractionContext) {
        if let Some(s) = ctx.brush_stroke {
            self.brush_strokes.push(s);
        }
    }
}

fn stroke(start: [f64; 2], end: [f64; 2], radius: f64, mode: BrushMode) -> BrushStroke {
    BrushStroke {
        start,
        end,
        radius,
        mode,
        drag_vector: [end[0] - start[0], end[1] - start[1]],
    }
}

fn dispatch_synthetic_frame(frame: &PlotBrushFrame, tool: &mut dyn InteractiveTool) {
    let ctx = egui::Context::default();
    let _ = ctx.run(RawInput::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            let (_, resp) =
                ui.allocate_exact_size(egui::vec2(240.0, 120.0), egui::Sense::click_and_drag());
            frame.dispatch(ui, &resp, tool);
        });
    });
}

#[test]
fn test_plot_brush_controller_hooks_and_bresenham_radii_1_to_8() {
    let mut rec = RecordingTool::default();
    let base_frame = PlotBrushFrame {
        strokes: vec![stroke([4.5, 4.5], [18.5, 11.5], 3.0, BrushMode::Solid)],
        plot_pointer_pos: Some([18.5, 11.5]),
        plot_delta: [14.0, 7.0],
        screen_delta: Vec2::new(28.0, -14.0),
        is_dragging: true,
        is_clicked: false,
        is_hovered: true,
        has_focus: true,
        input_mode: crate::framework::InputMode::Mouse,
        multi_touch: None,
        keys_down: HashSet::new(),
        modifiers: Modifiers::NONE,
    };
    dispatch_synthetic_frame(&base_frame, &mut rec);
    assert_eq!(rec.keyboard_hits, 1);
    assert_eq!(rec.drag_hits, 1);
    assert_eq!(rec.brush_strokes.len(), 1);
    assert_eq!(rec.last_pos, Some(Pos2::new(18.5, 11.5)));

    let mut click_frame = base_frame.clone();
    click_frame.strokes.clear();
    click_frame.is_dragging = false;
    click_frame.is_clicked = true;
    dispatch_synthetic_frame(&click_frame, &mut rec);
    assert_eq!(rec.click_hits, 1);

    let mut hover_frame = click_frame;
    hover_frame.is_clicked = false;
    hover_frame.has_focus = false;
    dispatch_synthetic_frame(&hover_frame, &mut rec);
    assert_eq!(rec.hover_hits, 1);

    for r in 1..=8 {
        for mode in [BrushMode::Solid, BrushMode::Erase] {
            let s = stroke([4.5, 4.5], [44.5, 34.5], f64::from(r), mode);
            let cmds = PlotBrushController::bresenham_sim_commands(&s, 100, 50, true);
            assert!(
                cmds.len() >= 41,
                "Bresenham stroke must span all 41 columns"
            );
            let mut prev: Option<(i32, i32)> = None;
            for cmd in cmds {
                let SimCommand::ApplyBrush {
                    cx,
                    cy,
                    r: cr,
                    is_obstacle,
                } = cmd
                else {
                    panic!("Expected ApplyBrush command");
                };
                assert_eq!(cr, r);
                assert_eq!(is_obstacle, mode != BrushMode::Erase);
                if let Some((px, py)) = prev {
                    assert!((cx - px).abs() <= 1 && (cy - py).abs() <= 1);
                }
                prev = Some((cx, cy));
            }
        }
    }
}

fn assert_lbm_fairing_url() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let mut app = MathExplorerApp::default();
    let hash = "#tab=fluid_dynamics&challenge=lbm_low_drag_fairing";
    assert!(app.apply_share_location_strings(hash, "", None).unwrap());
    let url = app.encode_current_share_url().unwrap();
    let parsed = parse_share_location(&url[url.find('#').unwrap()..], "")
        .unwrap()
        .unwrap();
    assert!(parsed.state_json.unwrap().len() <= MAX_DECODED_URL_STATE_BYTES);
}

#[test]
fn test_lbm_fast_drag_radii_forces_signed_vorticity_and_fairing_url() {
    let mut params = HashMap::from([
        ("viscosity".to_string(), 0.02),
        ("naca_profile".to_string(), 2412.0),
        ("angle_of_attack".to_string(), 14.0),
    ]);
    let mut lbm = LbmUnified::new(&params);

    for r in 1..=6 {
        let draw = stroke([40.5, 12.5], [62.5, 22.5], f64::from(r), BrushMode::Solid);
        let draw_cmds =
            PlotBrushController::bresenham_sim_commands(&draw, LBM_WIDTH, LBM_HEIGHT, true);
        let mut centers = Vec::new();
        for cmd in draw_cmds {
            if let SimCommand::ApplyBrush { cx, cy, .. } = cmd {
                centers.push((cx as usize, cy as usize));
            }
            lbm.process_command(cmd, &params);
        }
        assert!(centers.iter().all(|&(x, y)| lbm.solver.is_obstacle(x, y)));
        let erase = stroke([40.5, 12.5], [62.5, 22.5], f64::from(r), BrushMode::Erase);
        for cmd in PlotBrushController::bresenham_sim_commands(&erase, LBM_WIDTH, LBM_HEIGHT, true)
        {
            lbm.process_command(cmd, &params);
        }
        assert!(centers.iter().all(|&(x, y)| !lbm.solver.is_obstacle(x, y)));
    }

    for _ in 0..40 {
        lbm.step(&params);
    }
    let max_pos = lbm.vorticity_buf.iter().copied().fold(0.0, f64::max);
    let min_neg = lbm.vorticity_buf.iter().copied().fold(0.0, f64::min);
    assert!(max_pos > 1e-3 && min_neg < -1e-3);

    lbm.high_contrast = true;
    params.insert("field_view".to_string(), 0.0);
    lbm.step(&params);
    let speed_px = lbm.get_snapshot().pixels.read().unwrap().clone();
    params.insert("field_view".to_string(), 1.0);
    lbm.step(&params);
    let snap_vort = lbm.get_snapshot();
    assert_ne!(speed_px, *snap_vort.pixels.read().unwrap());
    assert!(snap_vort.custom_data[0..3].iter().all(|v| v.is_finite()));
    let metrics = snap_vort
        .structured_data
        .as_ref()
        .and_then(|d| d.downcast_ref::<LbmSnapshotMetrics>())
        .expect("LbmSnapshotMetrics");
    assert!(metrics.drag_coeff.is_finite() && metrics.lift_coeff.is_finite());
    assert_lbm_fairing_url();
}

#[test]
fn test_morphogenesis_emitter_sink_barrier_persistence_and_3d_splats() {
    for mode in [
        MorphogenesisKineticsMode::Schnakenberg,
        MorphogenesisKineticsMode::GrayScott,
    ] {
        let tool = MorphogenesisTool::default();
        tool.kinetics_state.write().unwrap().mode = mode;
        let mut runner = MorphogenesisTool::create_kinetics_runner(
            &tool.params,
            &tool.kinetics_state,
            &tool.mask_rle,
            tool.width,
            tool.height,
        );
        runner.process_command(SimCommand::Reset);
        let base_scene =
            build_scene_from_2d_morphogenesis_slice(runner.system.u(), runner.width, runner.height);

        let strokes = [
            stroke([20.5, 20.5], [22.5, 20.5], 2.0, BrushMode::EmitterU),
            stroke([40.5, 20.5], [42.5, 20.5], 2.0, BrushMode::EmitterV),
            stroke([60.5, 50.5], [62.5, 50.5], 2.0, BrushMode::Sink),
            stroke([30.5, 10.5], [30.5, 80.5], 2.0, BrushMode::ZeroFluxWall),
        ];
        for s in strokes {
            runner.process_command(SimCommand::ApplyBrushStroke(s));
        }
        for _ in 0..12 {
            runner.step();
        }
        assert!(runner.is_finite_state());

        let eu = flatten_2d_index(21, runner.height - 1 - 20, runner.width);
        let ev = flatten_2d_index(41, runner.height - 1 - 20, runner.width);
        let sk = flatten_2d_index(61, runner.height - 1 - 50, runner.width);
        let wl = flatten_2d_index(30, runner.height - 1 - 45, runner.width);
        assert_eq!(
            runner.system.diffusion.mask.roles()[wl],
            ReactionCellRole::ZeroFluxWall as u8
        );
        assert!(
            (runner.system.u()[eu] - runner.system.diffusion.mask.emitter_u_level).abs() < 1e-12
        );
        assert!(
            (runner.system.v()[ev] - runner.system.diffusion.mask.emitter_v_level).abs() < 1e-12
        );
        assert_eq!((runner.system.u()[sk], runner.system.v()[sk]), (0.0, 0.0));
        assert_eq!((runner.system.u()[wl], runner.system.v()[wl]), (0.0, 0.0));

        let sculpted_scene =
            build_scene_from_2d_morphogenesis_slice(runner.system.u(), runner.width, runner.height);
        assert!(!sculpted_scene.gaussians.is_empty());
        assert_ne!(sculpted_scene.gaussians.len(), base_scene.gaussians.len());
    }
}

#[test]
fn test_wave_freehand_500_hills_wells_unitarity_rt_and_sonification() {
    let cfg = Arc::new(RwLock::new(WaveSimConfig::default()));
    let mut runner = WaveRunner::new(cfg);

    let hill = stroke(
        [-0.35, 450.0 * POTENTIAL_DISPLAY_SCALE],
        [0.05, 450.0 * POTENTIAL_DISPLAY_SCALE],
        0.18,
        BrushMode::PotentialBarrier,
    );
    let well = stroke(
        [0.45, -420.0 * POTENTIAL_DISPLAY_SCALE],
        [0.85, -420.0 * POTENTIAL_DISPLAY_SCALE],
        0.18,
        BrushMode::PotentialWell,
    );
    runner.process_command(SimCommand::ApplyBrushStroke(hill));
    runner.process_command(SimCommand::ApplyBrushStroke(well));

    assert!(runner.potential.iter().any(|&v| (v - 450.0).abs() < 1e-6));
    assert!(runner.potential.iter().any(|&v| (v + 420.0).abs() < 1e-6));

    for _ in 0..45 {
        runner.step();
    }
    assert!(runner.is_finite_state());
    let snap = runner.get_snapshot();
    let wdata = snap
        .structured_data
        .as_ref()
        .and_then(|d| d.downcast_ref::<WaveData>())
        .expect("WaveData");
    assert!(
        (wdata.norm_sq - 1.0).abs() < 1e-6,
        "Unitary evolution must preserve ||psi||^2 == 1, got {}",
        wdata.norm_sq
    );
    assert!(wdata.reflection.is_finite() && wdata.transmission.is_finite());
    assert!(wdata.reflection + wdata.transmission <= 1.0 + 1e-6);
    assert!(snap.sonification.active_voices > 0);
    assert!(snap
        .sonification
        .voices
        .iter()
        .all(|v| v.freq_hz.is_finite() && v.gain.is_finite()));
}

#[test]
fn test_potential_flow_handles_clamp_50_sidebar_sync_and_glsl() {
    let mut flow = PotentialFlowTool::default();
    let base_len = flow.elements.len();
    flow.new_element_strength = 4.25;

    let modes = [
        (BrushMode::FlowSource, ElementType::Source, [-2.0, 1.5]),
        (BrushMode::FlowSink, ElementType::Sink, [-1.0, -1.5]),
        (BrushMode::FlowVortex, ElementType::Vortex, [1.2, 1.6]),
        (BrushMode::FlowDoublet, ElementType::Doublet, [2.2, -1.2]),
    ];
    for (mode, expected_type, pt) in modes {
        flow.apply_flow_stroke_with_drag(&stroke(pt, pt, 0.35, mode), false);
        assert_eq!(flow.elements.last().unwrap().element_type, expected_type);
    }
    assert_eq!(flow.elements.len(), base_len + 4);
    let glsl_before = flow.export_glsl_shader(ShaderDialect::Shadertoy).unwrap();

    // Drag the Doublet handle past [-50, 50] and verify clamping, sidebar sync, and GLSL update.
    flow.apply_flow_stroke_with_drag(
        &stroke([2.2, -1.2], [85.0, -92.0], 0.45, BrushMode::FlowSource),
        true,
    );
    assert_eq!(flow.elements.len(), base_len + 4);
    let doublet = flow.elements.last().unwrap();
    assert_eq!(doublet.element_type, ElementType::Doublet);
    assert!((doublet.strength - 4.25).abs() < 1e-9);
    assert_eq!([doublet.x, doublet.y], [FLOW_POS_MAX, FLOW_POS_MIN]);
    assert_eq!(flow.selected_element_type, ElementType::Doublet);
    assert_eq!(
        [flow.new_element_x, flow.new_element_y],
        [FLOW_POS_MAX, FLOW_POS_MIN]
    );
    assert!(!flow.cached_polylines.is_empty());

    let glsl_after = flow.export_glsl_shader(ShaderDialect::Shadertoy).unwrap();
    assert_ne!(glsl_before, glsl_after);
    assert!(glsl_after.contains("vec2(50.0, -50.0)"));

    // Click existing Vortex handle via PlotBrushFrame::dispatch and verify sidebar syncs without adding elements.
    flow.active_drag_element = None;
    let click_frame = PlotBrushFrame {
        strokes: Vec::new(),
        plot_pointer_pos: Some([1.2, 1.6]),
        plot_delta: [0.0, 0.0],
        screen_delta: Vec2::ZERO,
        is_dragging: false,
        is_clicked: true,
        is_hovered: true,
        has_focus: true,
        input_mode: crate::framework::InputMode::Mouse,
        multi_touch: None,
        keys_down: HashSet::new(),
        modifiers: Modifiers::NONE,
    };
    dispatch_synthetic_frame(&click_frame, &mut flow);
    assert_eq!(flow.elements.len(), base_len + 4);
    assert_eq!(flow.selected_element_type, ElementType::Vortex);
}

fn key_press(key: Key) -> Event {
    Event::Key {
        key,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers: Modifiers::NONE,
    }
}

fn verify_canvas_keyboard_sculpt_and_escape(fw: &mut SimulationFramework, id_tag: &'static str) {
    let ctx = egui::Context::default();
    let screen = Some(egui::Rect::from_min_size(
        Pos2::ZERO,
        Vec2::new(1280.0, 800.0),
    ));
    let input = |keys: &[Key]| RawInput {
        screen_rect: screen,
        events: keys.iter().copied().map(key_press).collect(),
        ..Default::default()
    };
    let start = crate::web_io::PlatformInstant::now();
    while fw.focused_canvas_id().is_none() && start.elapsed() < Duration::from_secs(5) {
        let _ = ctx.run(input(&[]), |ctx| fw.show(ctx, id_tag));
        for _ in 0..90 {
            let _ = ctx.run(input(&[Key::Tab]), |ctx| fw.show(ctx, id_tag));
            if fw.focused_canvas_id().is_some() {
                break;
            }
        }
        #[cfg(not(target_arch = "wasm32"))]
        if fw.focused_canvas_id().is_none() {
            std::thread::sleep(Duration::from_millis(10));
        }
    }
    let cid = fw
        .focused_canvas_id()
        .expect("Plot canvas must receive Tab focus");
    let before = fw.active_tool.as_ref().and_then(|t| t.save_state());
    let _ = ctx.run(
        input(&[Key::ArrowRight, Key::CloseBracket, Key::Space]),
        |ctx| fw.show(ctx, id_tag),
    );
    let after = fw.active_tool.as_ref().and_then(|t| t.save_state());
    assert_ne!(before, after, "{id_tag}: keyboard stamp must mutate state");

    let _ = ctx.run(input(&[Key::Tab]), |ctx| fw.show(ctx, id_tag));
    assert_ne!(fw.focused_canvas_id(), Some(cid), "Tab must exit plot");
}

#[test]
fn test_tabs_focused_canvas_id_keyboard_sculpt_and_tab_escape() {
    for (tool_id, tag) in [
        ("LatticeBoltzmannTool", "lbm_kb"),
        ("PotentialFlowTool", "pflow_kb"),
    ] {
        let mut tab = FluidDynamicsTab::default();
        assert!(tab.focused_canvas_id().is_none());
        let fw = tab.framework_mut().unwrap();
        fw.import_url_preset_json(Some(tool_id), None).unwrap();
        verify_canvas_keyboard_sculpt_and_escape(fw, tag);
    }

    let mut morpho = MorphogenesisTab::default();
    assert!(morpho.focused_canvas_id().is_none());
    verify_canvas_keyboard_sculpt_and_escape(morpho.framework_mut().unwrap(), "morpho_kb");

    let mut quantum = QuantumTab::default();
    assert!(quantum.focused_canvas_id().is_none());
    let qfw = quantum.framework_mut().unwrap();
    qfw.import_url_preset_json(Some("WaveSimulator"), None)
        .unwrap();
    verify_canvas_keyboard_sculpt_and_escape(qfw, "quantum_kb");
}
