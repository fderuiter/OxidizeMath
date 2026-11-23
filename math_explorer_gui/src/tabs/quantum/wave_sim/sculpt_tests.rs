#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::super::canvas::{apply_emitter_drag, x_to_grid_index, ActiveCanvasDrag, K0_ARROW_SCALE};
use super::super::presets::GRID_POINTS;
use super::super::*;
use crate::async_sim::SimulationRunner;
use crate::framework::recovery::FiniteStateGuard;
use crate::framework::{InteractiveTool, SimulationFramework};
use crate::url_state::{MAX_DECODED_URL_STATE_BYTES, MAX_URL_FRAGMENT_BYTES};
use eframe::egui;
use std::sync::{Arc, RwLock};

#[test]
fn test_sculpt_strokes_coalesce_propagator_rebuild_and_preserve_paused_zero_exp() {
    let cfg = Arc::new(RwLock::new(WaveSimConfig {
        preset: WavePotentialPreset::InfiniteWell,
        dt: 0.03,
        ..Default::default()
    }));
    let mut runner = WaveRunner::new(Arc::clone(&cfg));
    runner.step();
    assert_eq!(runner.propagator_rebuilds, 1);

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
    assert_eq!(runner.propagator_rebuilds, 1);

    let wave = runner.build_wave_data();
    assert!((wave.potential[100] - 16.0).abs() < 1e-12);
    assert!((wave.potential[130] - 12.0).abs() < 1e-12);
    assert!((wave.p_transmitted + wave.p_reflected - 1.0).abs() < 1e-14);

    for _ in 0..4 {
        runner.step();
    }
    assert_eq!(runner.propagator_rebuilds, 2);
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
    assert!((runner.sigma - 0.2).abs() < 1e-12);
    assert!((runner.x0 - 1.5).abs() < 1e-12);
    assert!((runner.k0 + 8.0).abs() < 1e-12);
    assert!(FiniteStateGuard::is_finite_state(&runner));
    let wave = runner.build_wave_data();
    assert!(wave.p_transmitted > 0.95);
    assert!((wave.p_transmitted + wave.p_reflected - 1.0).abs() < 1e-14);

    let mut tool = WaveSimulatorTool::new();
    tool.custom_potential = Some(vec![5.0; GRID_POINTS]);
    assert_eq!(x_to_grid_index(0.0, GRID_POINTS), 100);
    assert!(apply_emitter_drag(
        &mut tool,
        ActiveCanvasDrag::EmitterX0,
        -1.2,
        None,
    ));
    assert!(apply_emitter_drag(
        &mut tool,
        ActiveCanvasDrag::EmitterK0,
        -1.2 + 7.5 * K0_ARROW_SCALE,
        None,
    ));
    assert!(apply_emitter_drag(
        &mut tool,
        ActiveCanvasDrag::EmitterSigma,
        -1.2 + 0.05,
        None,
    ));
    assert!((tool.x0 + 1.2).abs() < 1e-12);
    assert!((tool.k0 - 7.5).abs() < 1e-12);
    assert!((tool.sigma - 0.2).abs() < 1e-12);
    assert_eq!(
        tool.custom_potential.as_ref().map(Vec::len),
        Some(GRID_POINTS)
    );
}

#[test]
fn test_custom_potential_url_roundtrip_flat_default_and_bounds_rejection() {
    let mut tool = WaveSimulatorTool::new();
    let default_json = tool.save_state().expect("default save_state");
    assert!(
        !default_json.contains("custom_potential"),
        "Unsculpted state must omit custom_potential: {default_json}"
    );

    tool.custom_potential = Some(
        (0..200)
            .map(|i| -19.99 + (i as f64 / 199.0) * 59.98)
            .collect(),
    );
    tool.sync_config_and_reset(None);
    let saved = tool.save_state().expect("quantized custom save_state");

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
    assert_eq!(restored.save_state().as_deref(), Some(saved.as_str()));
    assert!(restored
        .try_load_state(r#"{"preset":"InfiniteWell","custom_potential":[1.0,2.0,3.0]}"#)
        .is_err());
}

#[test]
fn test_wcag_reticle_keyboard_sculpting_focus_ring_and_live_announcement() {
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let mut tool = WaveSimulatorTool::new();
    for _ in 0..50 {
        if tool.worker.update().is_some() || tool.worker.latest_snapshot().is_some() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(2));
    }

    let out1 = ctx.run(egui::RawInput::default(), |ctx| tool.show(ctx));
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
    let _ = ctx.run(egui::RawInput::default(), |ctx| tool.show(ctx));

    let mut input = egui::RawInput::default();
    for key in [egui::Key::ArrowRight, egui::Key::ArrowUp] {
        input.events.push(egui::Event::Key {
            key,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        });
    }
    let _ = ctx.run(input, |ctx| tool.show(ctx));

    let live_msg = ctx.data(|d| d.get_temp::<String>(egui::Id::new("aria_live_message")));
    assert!(
        live_msg
            .as_deref()
            .is_some_and(|m| m.contains("Reticle x =") && m.contains("V(x) = 2.0")),
        "Expected live reticle announcement, got: {live_msg:?}"
    );
    let sculpted = tool
        .custom_potential
        .as_ref()
        .expect("sculpted custom_potential expected");
    assert!((sculpted[101] - 2.0).abs() < 1e-9);
}
