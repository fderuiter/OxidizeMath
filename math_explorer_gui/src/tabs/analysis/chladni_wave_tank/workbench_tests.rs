#![cfg_attr(any(), verified(opt_out = "tests"))]

// [cite:graph_parameters_rust]

use eframe::egui::{self, vec2, Context, Event, Key, Modifiers, Pos2, RawInput, Rect};
use verified_engine::engine::VerifiedEngine;

use super::*;
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
