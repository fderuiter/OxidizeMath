#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::canvas::{apply_emitter_drag, x_to_grid_index, ActiveCanvasDrag, K0_ARROW_SCALE};
use super::*;
use crate::async_sim::unified::{CompiledFormulaPayload, UnifiedCanvasContext, UnifiedSimTool};
use crate::framework::recovery::FiniteStateGuard;
use crate::framework::InteractiveTool;
use crate::url_state::{MAX_DECODED_URL_STATE_BYTES, MAX_URL_FRAGMENT_BYTES};
use math_commons::ast::{parse, CompiledExpr};
use math_explorer::physics::quantum::evolve_state;
use std::collections::BTreeMap;

#[test]
fn test_wave_unified_dt_caching_and_trajectory_parity() {
    let mut params = HashMap::from([
        ("potential_type".to_string(), 0.0),
        ("dt".to_string(), 0.05),
    ]);
    let mut sim = WaveUnified::new(&params);
    sim.n_points = 20;
    sim.init_system(0.0);
    assert_eq!(sim.cached_dt, None);
    assert_eq!(sim.scratch.len(), 20);

    let mut expected_psi = sim.psi.clone();
    for _ in 0..3 {
        sim.step(&params);
        expected_psi = evolve_state(&expected_psi, &sim.hamiltonian, 0.05, 1.0).normalize();
    }
    assert_eq!(sim.cached_dt, Some(0.05));
    let ptr_before = sim.propagator.matrix.as_ptr();
    sim.step(&params);
    expected_psi = evolve_state(&expected_psi, &sim.hamiltonian, 0.05, 1.0).normalize();
    assert_eq!(sim.propagator.matrix.as_ptr(), ptr_before);
    assert!((&sim.psi.vector - &expected_psi.vector).norm() < 1e-12);

    params.insert("dt".to_string(), 0.02);
    sim.step(&params);
    expected_psi = evolve_state(&expected_psi, &sim.hamiltonian, 0.02, 1.0).normalize();
    assert_eq!(sim.cached_dt, Some(0.02));
    assert!((&sim.psi.vector - &expected_psi.vector).norm() < 1e-12);
}

#[test]
fn test_wave_unified_reset_invalidates_propagator_on_potential_change() {
    let mut params = HashMap::from([
        ("potential_type".to_string(), 0.0),
        ("dt".to_string(), 0.04),
    ]);
    let mut sim = WaveUnified::new(&params);
    sim.n_points = 20;
    sim.init_system(0.0);
    sim.step(&params);
    assert_eq!(sim.cached_dt, Some(0.04));
    let well_propagator = sim.propagator.matrix.clone();

    params.insert("potential_type".to_string(), 1.0);
    sim.process_command(SimCommand::Reset, &params);
    assert_eq!(sim.cached_dt, None);
    assert_eq!(sim.step_count, 0);
    assert_eq!(sim.time, 0.0);

    let init_psi = sim.psi.clone();
    sim.step(&params);
    assert_eq!(sim.cached_dt, Some(0.04));
    assert_ne!(sim.propagator.matrix, well_propagator);
    let expected = evolve_state(&init_psi, &sim.hamiltonian, 0.04, 1.0).normalize();
    assert!((&sim.psi.vector - &expected.vector).norm() < 1e-12);
}

#[test]
fn test_sculpt_strokes_coalesce_propagator_rebuild_and_preserve_paused_zero_exp() {
    let params = HashMap::from([
        ("potential_type".to_string(), 0.0),
        ("dt".to_string(), 0.03),
    ]);
    let mut sim = WaveUnified::new(&params);
    sim.n_points = 40;
    sim.init_system(0.0);
    sim.step(&params);
    assert_eq!(sim.propagator_rebuilds, 1);

    let s1 = SimCommand::SculptStroke1D {
        i0: 15,
        v0: 6.0,
        i1: 20,
        v1: 16.0,
    };
    let s2 = SimCommand::SculptStroke1D {
        i0: 20,
        v0: 16.0,
        i1: 25,
        v1: 6.0,
    };
    let brush = SimCommand::ApplyBrush {
        cx: 30,
        cy: 12,
        r: 2,
        is_obstacle: true,
    };
    sim.process_command(s1, &params);
    sim.process_command(s2, &params);
    sim.process_command(brush, &params);
    assert_eq!(sim.cached_dt, None);
    assert!(sim.hamiltonian_dirty);
    assert_eq!(sim.propagator_rebuilds, 1);

    let snap = sim.get_snapshot();
    let wave = snap
        .structured_data
        .as_ref()
        .and_then(|d| d.downcast_ref::<WaveData>())
        .expect("WaveData snapshot expected");
    assert!((wave.potential[20] - 16.0).abs() < 1e-12);
    assert!((wave.potential[30] - 12.0).abs() < 1e-12);
    assert!((wave.p_transmitted + wave.p_reflected - 1.0).abs() < 1e-14);

    for _ in 0..4 {
        sim.step(&params);
    }
    assert_eq!(sim.propagator_rebuilds, 2);
}

#[test]
fn test_set_wavepacket_clamps_sigma_and_keeps_sculpted_potential_intact() {
    let params = HashMap::from([
        ("potential_type".to_string(), 0.0),
        ("dt".to_string(), 0.02),
    ]);
    let mut sim = WaveUnified::new(&params);
    sim.process_command(
        SimCommand::SculptStroke1D {
            i0: 98,
            v0: 22.0,
            i1: 102,
            v1: 22.0,
        },
        &params,
    );
    sim.process_command(
        SimCommand::SetWavepacket {
            x0: 1.5,
            k0: -8.0,
            sigma: 0.0,
        },
        &params,
    );
    assert!((sim.potential[100] - 22.0).abs() < 1e-12);
    assert!((sim.sigma - 0.2).abs() < 1e-12);
    assert!((sim.x0 - 1.5).abs() < 1e-12);
    assert!((sim.k0 + 8.0).abs() < 1e-12);
    assert!(FiniteStateGuard::is_finite_state(&sim));

    let snap = sim.get_snapshot();
    let wave = snap
        .structured_data
        .unwrap()
        .downcast::<WaveData>()
        .unwrap();
    assert!(wave.p_transmitted > 0.95);
    assert!((wave.p_transmitted + wave.p_reflected - 1.0).abs() < 1e-14);
}

#[test]
fn test_emitter_handle_dragging_updates_wavepacket_without_clearing_potential() {
    let mut tool = UnifiedSimTool::<WaveUnified>::new();
    tool.custom_series
        .write()
        .unwrap()
        .insert(CUSTOM_POTENTIAL_KEY.to_string(), vec![5.0; GRID_POINTS]);
    assert_eq!(x_to_grid_index(0.0, GRID_POINTS), 100);

    {
        let mut ctx = UnifiedCanvasContext {
            stepped_snapshot: None,
            controller: &mut tool.controller,
            params: &tool.params,
            custom_series: &tool.custom_series,
        };
        assert!(apply_emitter_drag(
            &mut ctx,
            ActiveCanvasDrag::EmitterX0,
            -1.2,
            (-2.0, 5.0, 0.5),
        ));
        assert!(apply_emitter_drag(
            &mut ctx,
            ActiveCanvasDrag::EmitterK0,
            -1.2 + 7.5 * K0_ARROW_SCALE,
            (-1.2, 5.0, 0.5),
        ));
        assert!(apply_emitter_drag(
            &mut ctx,
            ActiveCanvasDrag::EmitterSigma,
            -1.2 + 0.05,
            (-1.2, 7.5, 0.5),
        ));
    }

    let guard = tool.params.read().unwrap();
    assert!((guard["x0"] + 1.2).abs() < 1e-12);
    assert!((guard["k0"] - 7.5).abs() < 1e-12);
    assert!((guard["sigma"] - 0.2).abs() < 1e-12);
    assert_eq!(
        tool.custom_series.read().unwrap()[CUSTOM_POTENTIAL_KEY].len(),
        GRID_POINTS
    );
}

#[test]
fn test_custom_potential_url_roundtrip_flat_default_and_bounds_rejection() {
    let tool = UnifiedSimTool::<WaveUnified>::new();
    let default_json = tool.save_state().expect("default save_state");
    assert!(
        !default_json.contains("custom_potential"),
        "Empty custom_series must remain a flat scalar JSON map: {default_json}"
    );

    let custom: Vec<f64> = (0..200)
        .map(|i| -19.99 + (i as f64 / 199.0) * 59.98)
        .collect();
    tool.custom_series
        .write()
        .unwrap()
        .insert(CUSTOM_POTENTIAL_KEY.to_string(), custom);

    let saved = tool.save_state().expect("quantized custom save_state");
    let mut fw = crate::framework::SimulationFramework::new("quantum");
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

    let mut restored = UnifiedSimTool::<WaveUnified>::new();
    restored.try_load_state(&saved).expect("round-trip load");
    assert_eq!(restored.save_state().as_deref(), Some(saved.as_str()));
    assert!(restored
        .try_load_state(r#"{"dt":0.05,"custom_potential":[1.0,2.0,3.0]}"#)
        .is_err());
}

#[test]
fn test_wcag_reticle_keyboard_sculpting_focus_ring_and_live_announcement() {
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let mut tool = UnifiedSimTool::<WaveUnified>::new();
    for _ in 0..50 {
        if tool.controller.update().is_some() || tool.controller.latest_snapshot().is_some() {
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
    let series = tool.custom_series.read().unwrap();
    assert!((series[CUSTOM_POTENTIAL_KEY][101] - 2.0).abs() < 1e-9);
}

fn assert_custom_vx_tool_roundtrip() {
    let mut tool = UnifiedSimTool::<WaveUnified>::new();
    tool.custom_formula_enabled = true;
    {
        let mut p = tool.params.write().unwrap();
        p.insert("potential_type".to_string(), 1.0);
        p.insert("dt".to_string(), 0.025);
    }
    let bar = tool.formula_bar.as_mut().expect("formula bar configured");
    bar.set_expression("b * cos(x)^2");
    let mut pmap = BTreeMap::new();
    pmap.insert("b".to_string(), 3.5);
    bar.set_params_map(&pmap);
    tool.publish_formula_update();

    let saved = tool
        .save_state()
        .expect("wave tool saves custom V(x) state");
    assert!(saved.len() < MAX_DECODED_URL_STATE_BYTES);

    let mut restored = UnifiedSimTool::<WaveUnified>::new();
    restored
        .try_load_state(&saved)
        .expect("wave tool loads custom V(x) state");
    assert!(restored.custom_formula_enabled);
    assert_eq!(restored.params.read().unwrap()["potential_type"], 1.0);
    let restored_bar = restored.formula_bar.as_ref().unwrap();
    assert_eq!(restored_bar.raw_text, "b * cos(x)^2");
    assert_eq!(restored_bar.params_map(), pmap);
}

#[test]
fn test_wave_unified_custom_vx_hamiltonian_and_tool_roundtrip() {
    let params = HashMap::from([
        ("potential_type".to_string(), 0.0),
        ("dt".to_string(), 0.03),
    ]);
    let mut sim = WaveUnified::new(&params);
    sim.n_points = 24;
    sim.init_system(0.0);

    let expr = parse("0.1 * x^4 - a * x^2 + 3.0").unwrap();
    let compiled = Arc::new(CompiledExpr::compile_with_vars(&expr, &["x"]).unwrap());
    let mut param_buf = [0.0; 6];
    param_buf[0] = 1.2;
    let payload = CompiledFormulaPayload {
        revision: 2,
        compiled,
        params: param_buf,
        param_len: 1,
    };

    sim.apply_compiled_formula(Some(&payload), &params);
    assert_eq!(sim.cached_dt, None);

    let dx = (sim.x_max - sim.x_min) / (sim.n_points as f64 - 1.0);
    let expected_v = DVector::from_iterator(
        sim.n_points,
        sim.x_axis
            .iter()
            .map(|&x| 0.1 * x.powi(4) - 1.2 * x * x + 3.0),
    );
    let expected_h = construct_1d_hamiltonian(&expected_v, dx, 1.0, 1.0);
    assert!((&sim.hamiltonian.matrix - &expected_h.matrix).norm() < 1e-12);

    sim.step(&params);
    assert!(FiniteStateGuard::is_finite_state(&sim));
    assert!((sim.psi.vector.norm() - 1.0).abs() < 1e-10);
    assert_custom_vx_tool_roundtrip();
}
