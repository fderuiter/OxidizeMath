#![cfg_attr(any(), verified(opt_out = "tests"))]
#![cfg(feature = "physics")]

// [cite:graph_parameters_rust]

use eframe::egui::{self, vec2, Context, Event, Key, Modifiers, Pos2, RawInput, Rect};
use math_explorer::pure_math::analysis::pde::wave_2d::{DriverConfig2D, Wave2DParams};
use verified_engine::engine::VerifiedEngine;

use super::*;
use crate::audio::SonificationFrame;
use crate::framework::{BrushInteractionMode, BrushMode};

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
fn test_chladni_workbench_1000_steps_verified_zero_alloc() {
    let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());

    let mut solver = Wave2DSolver::new_unit_square(16);
    solver.inject_gaussian_pulse(0.35, 0.45, 0.10, 0.9);
    let mut fft = SpectralAnalyzer2D::new(16, 16);
    let mut peaks = [SpectralPeak2D::default(); 8];
    let params = Wave2DParams {
        medium: Wave2DMedium::BiharmonicPlate,
        boundary: BoundaryMode2D::Clamped,
        wave_speed: 0.0,
        plate_stiffness: 4.0e-4,
        damping: 0.025,
    };
    let driver = DriverConfig2D {
        enabled: true,
        line_source: false,
        x_norm: 0.37,
        y_norm: 0.43,
        frequency: 1.4,
        amplitude: 30.0,
        radius_cells: 1.8,
    };

    let ((count, frame), _metrics) = VerifiedEngine::run_verified(|| {
        let mut n_peaks = 0_usize;
        let mut audio = SonificationFrame::silent();
        for _ in 0..1_000_usize {
            solver.step_driven(&params, &driver, 0.002, None);
            let probe = solver.sample_probe(0.66, 0.52);
            fft.compute_spectrum(solver.displacement());
            n_peaks = fft.top_peaks(&mut peaks);
            audio = crate::audio::mappings::chladni_plate_frame(
                &peaks[..n_peaks],
                probe,
                params.medium,
                driver.frequency,
                params.wave_speed,
                params.plate_stiffness,
                0.66,
            );
        }
        (n_peaks, audio)
    });

    assert!(count > 0 && frame.is_audible());
    assert!((20.0..=8_000.0).contains(&frame.voices[0].freq_hz));
}

fn apply_sample_luthiery_strokes(tool: &mut ChladniWaveTankTool) {
    use crate::framework::{BrushStroke, InteractionContext, InteractiveTool};

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

fn active_chladni_mut(
    app: &mut crate::app::MathExplorerApp,
    tab_idx: usize,
) -> &mut ChladniWaveTankTool {
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

    use crate::app::MathExplorerApp;
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

fn focus_canvas_and_verify_meter(
    ctx: &Context,
    tool: &mut ChladniWaveTankTool,
) -> Vec<egui::accesskit::NodeId> {
    let _ = ctx.run(raw_frame(), |ctx| tool.show(ctx));
    let ak = ctx
        .run(raw_frame(), |ctx| tool.show(ctx))
        .platform_output
        .accesskit_update
        .expect("AccessKit update expected");
    assert!(
        ak.nodes.iter().any(|(_, n)| {
            n.role() == egui::accesskit::Role::Meter
                && n.label()
                    .is_some_and(|l| l.contains("dBFS") && l.contains("Hz"))
        }),
        "Expected Role::Meter node with dBFS and Hz telemetry"
    );
    let canvas_nodes: Vec<egui::accesskit::NodeId> = ak
        .nodes
        .iter()
        .filter_map(|(id, n)| {
            (n.role() == egui::accesskit::Role::GraphicsDocument
                && n.label()
                    .unwrap_or_default()
                    .contains("2D Chladni Plate & Wave Tank Visualization"))
            .then_some(*id)
        })
        .collect();
    let mut focused = false;
    for _ in 0..80 {
        if ctx
            .run(raw_key(Key::Tab), |ctx| tool.show(ctx))
            .platform_output
            .accesskit_update
            .is_some_and(|upd| canvas_nodes.contains(&upd.focus))
        {
            focused = true;
            break;
        }
    }
    assert!(focused, "Canvas must be reachable via Key::Tab");
    canvas_nodes
}

#[test]
fn test_chladni_accesskit_sculpt_reticle_space_stamp_meter_and_tab_exit() {
    let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut tool = ChladniWaveTankTool::default();
    tool.brush_kit.interaction_mode = BrushInteractionMode::Sculpt;
    tool.brush_kit.mode = BrushMode::PotentialBarrier;
    tool.step_simulation();

    let ctx = Context::default();
    ctx.enable_accesskit();
    if crate::audio::snapshot_ui_state().muted {
        crate::audio::toggle_mute_from_user_gesture();
    }

    let canvas_nodes = focus_canvas_and_verify_meter(&ctx, &mut tool);
    let start_cursor = tool.brush_kit.cursor_pos;
    let start_radius = tool.brush_kit.radius;
    let _ = ctx.run(raw_key(Key::ArrowRight), |ctx| tool.show(ctx));
    let _ = ctx.run(raw_key(Key::ArrowUp), |ctx| tool.show(ctx));
    let _ = ctx.run(raw_key(Key::CloseBracket), |ctx| tool.show(ctx));
    assert!(tool.brush_kit.cursor_pos[0] > start_cursor[0]);
    assert!(tool.brush_kit.cursor_pos[1] > start_cursor[1]);
    assert!(tool.brush_kit.radius > start_radius);

    assert!(tool.state.sculpt_rle.is_empty());
    let _ = ctx.run(raw_key(Key::Space), |ctx| tool.show(ctx));
    assert!(!tool.state.sculpt_rle.is_empty());

    let next_tab = ctx.run(raw_key(Key::Tab), |ctx| tool.show(ctx));
    if let Some(upd) = next_tab.platform_output.accesskit_update {
        assert!(!canvas_nodes.contains(&upd.focus));
    }
    crate::audio::notify_playback_paused();
}

#[cfg(feature = "physics")]
#[test]
fn test_phononic_bravais_crystal_ast_potential_and_dispersion_mashup() {
    use math_explorer::physics::solid_state::phonons::{
        acoustic_dispersion, debye_heat_capacity_low_temp, optical_dispersion,
    };

    use super::metamaterial::BravaisLatticeKind;

    let mut baseline = ChladniWaveTankTool::default();
    baseline.apply_preset(TankPreset::AcousticLens);
    baseline.state.lens_enabled = false;
    baseline.rebuild_domain_and_warmup(240);
    let base_energy: f64 = baseline.detector_buf.iter().sum();
    assert!(base_energy > 0.0);

    let mut crystal = ChladniWaveTankTool::default();
    crystal.apply_preset(TankPreset::AcousticLens);
    crystal.state.lens_enabled = false;
    for kind in [
        BravaisLatticeKind::Simple,
        BravaisLatticeKind::BodyCentered,
        BravaisLatticeKind::FaceCentered,
    ] {
        crystal.metamaterial.lattice_kind = kind;
        crystal.stamp_bravais_crystal();
        assert!(!crystal.state.sculpt_rle.is_empty());
    }
    crystal.rebuild_domain_and_warmup(240);
    let crystal_energy: f64 = crystal.detector_buf.iter().sum();
    assert!((crystal_energy - base_energy).abs() > 1.0e-3 * base_energy);

    let mut ast_tool = ChladniWaveTankTool::default();
    ast_tool.apply_preset(TankPreset::AcousticLens);
    ast_tool.state.lens_enabled = false;
    ast_tool
        .metamaterial
        .set_ast_formula("sqrt(max(0.55, 2.0 - ((x - 0.44)^2 + (y - 0.5)^2) / 0.045))");
    ast_tool.metamaterial.ast_enabled = true;
    ast_tool.rebuild_domain_and_warmup(240);
    let ast_energy: f64 = ast_tool.detector_buf.iter().sum();
    assert!((ast_energy - base_energy).abs() > 1.0e-3 * base_energy);

    ast_tool.stamp_ast_potential_to_sculpt();
    assert!(!ast_tool.state.sculpt_rle.is_empty() && ast_tool.save_state().unwrap().len() < 4_096);

    let w_ac = acoustic_dispersion(0.5, 1.25);
    let w_op = optical_dispersion(0.5, 24.0);
    let cv = debye_heat_capacity_low_temp(25.0, 343.0, 1.0, 1.0);
    assert!(w_op > w_ac && w_ac > 0.0 && cv > 0.0);
    ast_tool.state.secondary_plot = SecondaryPlotMode::PhononDispersion;
    let _ = Context::default().run(raw_frame(), |ctx| ast_tool.show(ctx));
}
