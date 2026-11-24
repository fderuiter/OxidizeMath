#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::canvas::{apply_emitter_drag, ActiveEmitterDrag, K0_ARROW_SCALE};
use super::presets::{GRID_POINTS, GRID_X_MAX, GRID_X_MIN, SIGMA_MIN};
use super::sculpt::x_to_grid_index;
use super::*;
use crate::async_sim::SimulationRunner;
use crate::framework::recovery::FiniteStateGuard;
use crate::framework::{InteractiveTool, SimulationFramework};
use crate::url_state::{MAX_DECODED_URL_STATE_BYTES, MAX_URL_FRAGMENT_BYTES};
use eframe::egui;
use std::sync::{Arc, RwLock};

#[test]
fn test_sculpt_strokes_coalesce_propagator_rebuild_and_preserve_paused_zero_exp() {
    let rebuilds = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let cfg = Arc::new(RwLock::new(WaveSimConfig {
        preset: WavePotentialPreset::InfiniteWell,
        dt: 0.03,
        ..Default::default()
    }));
    let mut runner = WaveRunner::new_with_counter(Arc::clone(&cfg), Arc::clone(&rebuilds));
    runner.step();
    assert_eq!(rebuilds.load(std::sync::atomic::Ordering::Relaxed), 1);

    runner.process_command(SimCommand::SculptStroke1D {
        i0: 95,
        v0: 6.0,
        i1: 100,
        v1: 16.0,
    });
    runner.process_command(SimCommand::SculptStroke1D {
        i0: 100,
        v0: 16.0,
        i1: 105,
        v1: 6.0,
    });
    runner.process_command(SimCommand::ApplyBrush {
        cx: 130,
        cy: 12,
        r: 2,
        is_obstacle: true,
    });
    assert_eq!(runner.cached_dt, None);
    assert!(runner.hamiltonian_dirty);
    assert_eq!(rebuilds.load(std::sync::atomic::Ordering::Relaxed), 1);

    let wave = runner.build_wave_data();
    assert!((wave.potential[100] - 16.0).abs() < 1e-12);
    assert!((wave.potential[130] - 16.0).abs() < 1e-12);
    assert!((wave.p_transmitted + wave.p_reflected - 1.0).abs() < 1e-14);

    for _ in 0..4 {
        runner.step();
    }
    assert!(!runner.hamiltonian_dirty);
    assert_eq!(rebuilds.load(std::sync::atomic::Ordering::Relaxed), 2);
}

#[test]
fn test_set_wavepacket_and_emitter_drag_keep_sculpted_potential_intact() {
    let cfg = Arc::new(RwLock::new(WaveSimConfig::default()));
    let mut runner = WaveRunner::new(Arc::clone(&cfg));
    runner.process_command(SimCommand::SculptStroke1D {
        i0: 98,
        v0: 22.0,
        i1: 102,
        v1: 22.0,
    });
    runner.process_command(SimCommand::SetWavepacket {
        x0: 1.5,
        k0: -8.0,
        sigma: 0.0,
    });
    assert!((runner.potential[100] - 22.0).abs() < 1e-12);
    assert!((runner.sigma - SIGMA_MIN).abs() < 1e-12);
    assert!((runner.x0 - 1.5).abs() < 1e-12);
    assert!((runner.k0 + 8.0).abs() < 1e-12);
    assert!(FiniteStateGuard::is_finite_state(&runner));
    let wave = runner.build_wave_data();
    assert!(wave.p_transmitted > 0.95);
    assert!((wave.p_transmitted + wave.p_reflected - 1.0).abs() < 1e-14);

    let mut tool = WaveSimulatorTool::new();
    tool.apply_sculpt_stroke(98, 18.0, 102, 18.0, None);
    let expected_sparse = tool.sparse_potential.clone();
    assert!(!expected_sparse.is_empty());
    assert_eq!(
        x_to_grid_index(0.0, GRID_X_MIN, GRID_X_MAX, GRID_POINTS),
        100
    );
    assert!(apply_emitter_drag(
        &mut tool,
        ActiveEmitterDrag::EmitterX0,
        -1.2,
        None,
    ));
    assert!(apply_emitter_drag(
        &mut tool,
        ActiveEmitterDrag::EmitterK0,
        -1.2 + 7.5 * K0_ARROW_SCALE,
        None,
    ));
    assert!(apply_emitter_drag(
        &mut tool,
        ActiveEmitterDrag::EmitterSigma,
        -1.2 + 0.05,
        None,
    ));
    assert!((tool.x0 + 1.2).abs() < 1e-12);
    assert!((tool.k0 - 7.5).abs() < 1e-12);
    assert!((tool.sigma - SIGMA_MIN).abs() < 1e-12);
    assert_eq!(tool.sparse_potential, expected_sparse);
}

#[test]
fn test_sculpted_sparse_potential_url_roundtrip_and_bounds_rejection() {
    let mut tool = WaveSimulatorTool::new();
    let default_json = tool.save_state().expect("default save_state");
    assert!(
        !default_json.contains("sparse_potential"),
        "Unsculpted state must omit sparse_potential: {default_json}"
    );

    tool.apply_sculpt_stroke(60, -18.5, 140, 24.5, None);
    assert!(!tool.sparse_potential.is_empty());
    let saved = tool.save_state().expect("sculpted sparse save_state");

    let mut fw = SimulationFramework::new("quantum");
    assert!(fw.select_tool_by_name("WaveSimulator"));
    fw.active_tool
        .as_mut()
        .unwrap()
        .try_load_state(&saved)
        .expect("load sculpted state");
    let envelope = fw
        .export_url_preset_json()
        .unwrap()
        .expect("preset envelope");
    assert!(envelope.len() <= MAX_DECODED_URL_STATE_BYTES);
    let url = fw.encode_share_url().expect("share URL");
    let hash = &url[url.find('#').expect("fragment expected")..];
    assert!(hash.len() <= MAX_URL_FRAGMENT_BYTES);

    let mut restored = WaveSimulatorTool::new();
    restored.try_load_state(&saved).expect("round-trip load");
    assert_eq!(restored.sparse_potential, tool.sparse_potential);
    assert!(restored
        .try_load_state(r#"{"preset":"InfiniteWell","sparse_potential":[[999,12.0]]}"#)
        .is_err());
}

fn raw_key_input(screen: Option<egui::Rect>, keys: &[egui::Key]) -> egui::RawInput {
    let events = keys
        .iter()
        .map(|&key| egui::Event::Key {
            key,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        })
        .collect();
    egui::RawInput {
        screen_rect: screen,
        events,
        ..Default::default()
    }
}

#[test]
fn test_wcag_reticle_keyboard_sculpting_focus_ring_and_live_announcement() {
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let screen = Some(egui::Rect::from_min_size(
        egui::Pos2::ZERO,
        egui::Vec2::new(1280.0, 800.0),
    ));
    let mut tool = WaveSimulatorTool::new();
    tool.step_simulation();

    let out1 = ctx.run(raw_key_input(screen, &[]), |ctx| tool.show(ctx));
    assert!(out1
        .platform_output
        .accesskit_update
        .as_ref()
        .is_some_and(|ak| ak.nodes.iter().any(|(_, n)| {
            n.role() == egui::accesskit::Role::GraphicsDocument
                && n.label() == Some("Quantum Potential Sculpting Canvas")
        })));
    let plot_id = ctx
        .data(|d| d.get_temp::<egui::Id>(egui::Id::new("quantum_plot_response_id")))
        .expect("quantum_plot_response_id must be registered");
    ctx.memory_mut(|m| m.request_focus(plot_id));
    let _ = ctx.run(raw_key_input(screen, &[]), |ctx| tool.show(ctx));

    let nav = raw_key_input(screen, &[egui::Key::ArrowRight, egui::Key::ArrowUp]);
    let _ = ctx.run(nav, |ctx| tool.show(ctx));

    let live_msg = ctx.data(|d| d.get_temp::<String>(egui::Id::new("aria_live_message")));
    assert!(
        live_msg
            .as_deref()
            .is_some_and(|m| m.contains("Reticle x =") && m.contains("P_T =")),
        "Expected live reticle announcement, got: {live_msg:?}"
    );

    let _ = ctx.run(raw_key_input(screen, &[egui::Key::Space]), |ctx| {
        tool.show(ctx);
    });
    assert!(!tool.sparse_potential.is_empty());
}

#[test]
fn test_starter_presets_fine_tip_stroke_and_resonant_well_png_artifact() {
    for preset_id in [
        "wave_sim_tunneling_barrier",
        "wave_sim_resonant_double_well",
    ] {
        let mut fw = SimulationFramework::new("quantum");
        let cards =
            crate::starter_gallery::get_starter_cards_for_domain("quantum", &fw.available_tools);
        let card = cards
            .iter()
            .find(|c| c.id == preset_id)
            .unwrap_or_else(|| panic!("missing quantum starter card {preset_id}"));
        fw.launch_starter_card(card).expect("launch starter card");
        for _ in 0..16 {
            fw.active_tool.as_mut().unwrap().step_simulation();
        }
        if preset_id == "wave_sim_resonant_double_well" {
            let stroke = crate::framework::BrushStroke {
                mode: crate::framework::BrushMode::PotentialBarrier,
                start: [1.3, 0.05],
                end: [2.1, 0.35],
                radius: 0.10,
                drag_vector: [0.8, 0.30],
            };
            let wave = fw
                .active_tool
                .as_mut()
                .unwrap()
                .as_any_mut()
                .unwrap()
                .downcast_mut::<WaveSimulatorTool>()
                .expect("WaveSimulatorTool");
            wave.apply_ui_brush_stroke(stroke, None);
            assert!(!wave.sparse_potential.is_empty());
            wave.step_simulation();
            let datasets = wave.export_datasets();
            let opts = egui_plot::export::ExportOptions {
                width: 1100,
                height: 680,
                ..Default::default()
            };
            let png = egui_plot::export::export_custom_png(&datasets, &opts).expect("export PNG");
            let _ = std::fs::write("/tmp/quantum_resonant_double_well_v2.png", png);
        }
    }
}
