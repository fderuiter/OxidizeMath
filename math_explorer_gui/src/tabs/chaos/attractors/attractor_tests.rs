#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::async_sim::SimulationRunner;
use crate::framework::{Camera3D, InteractiveTool};
use eframe::egui;

#[test]
fn test_attractor_divergence_detection() {
    let mut plotter = AttractorPlotter::default();
    assert!(!plotter.diverged());
    assert!(!plotter.is_paused());
    assert_eq!(plotter.execution_state, SolverExecutionState::Ready);

    // Introduce a non-finite state
    plotter.system.state.vec.x = f64::NAN;

    let ctx = egui::Context::default();
    let _ = ctx.run(egui::RawInput::default(), |ctx| plotter.show(ctx));

    assert!(plotter.diverged() && plotter.is_paused());
    match plotter.execution_state {
        SolverExecutionState::Diverged {
            invalid_vec,
            diverged_at_step,
        } => {
            assert!(invalid_vec.x.is_nan() && diverged_at_step == 0);
        }
        _ => panic!("Expected Diverged execution state"),
    }

    // Verify history contains no non-finite values
    assert!(plotter
        .history
        .iter()
        .all(|v| v.x.is_finite() && v.y.is_finite() && v.z.is_finite()));

    // Reset clears diverged flag cleanly and restores Ready
    plotter.reset();
    assert!(!plotter.diverged() && !plotter.is_paused());
    assert_eq!(plotter.execution_state, SolverExecutionState::Ready);
    assert_eq!(plotter.step_count, 0);
}

#[test]
fn test_attractor_reset_parameters() {
    let mut plotter = AttractorPlotter::default();
    plotter.system.sigma = f64::NAN;
    plotter.execution_state = SolverExecutionState::Diverged {
        invalid_vec: Vector3::new(f64::NAN, 1.0, 1.0),
        diverged_at_step: 10,
    };

    plotter.reset_parameters();
    assert!(!plotter.diverged());
    assert_eq!(plotter.execution_state, SolverExecutionState::Ready);
    assert_eq!(plotter.step_count, 0);
    assert_eq!(plotter.system.sigma, 10.0);
}

#[test]
fn test_attractor_transition_validation_non_finite_rejection() {
    let mut plotter = AttractorPlotter::default();
    plotter.system.state.vec.x = f64::NAN;

    let ctx = egui::Context::default();
    let _ = ctx.run(egui::RawInput::default(), |ctx| plotter.show(ctx));

    assert!(plotter.diverged() && plotter.is_paused());

    // Resume attempt must fail while coordinates are non-finite
    assert!(plotter.try_resume().is_err() && plotter.diverged());

    // Slider modifications must remain inert while state is non-finite
    plotter.system.sigma = 15.0;
    plotter.on_slider_changed();
    assert!(plotter.diverged());

    plotter.dt = 0.02;
    plotter.on_slider_changed();
    assert!(plotter.diverged());

    // Explicit reset restores Ready state and finite coordinates
    plotter.reset();
    assert_eq!(plotter.execution_state, SolverExecutionState::Ready);
    assert!(!plotter.diverged() && !plotter.is_paused() && plotter.is_finite_state());

    // Resume now succeeds
    assert!(plotter.try_resume().is_ok());
    assert_eq!(plotter.execution_state, SolverExecutionState::Running);
}

#[test]
fn test_attractor_parameter_sanitization_and_rejection() {
    let mut plotter = AttractorPlotter::default();
    plotter.system.sigma = f64::NAN;
    plotter.dt = f64::NAN;

    // try_resume must reject non-finite parameters
    assert!(plotter.try_resume().is_err());

    // Reset sanitizes non-finite parameters to default values
    plotter.reset();
    assert_eq!(plotter.system.sigma, 10.0);
    assert_eq!(plotter.dt, 0.01);
    assert_eq!(plotter.execution_state, SolverExecutionState::Ready);
}

#[test]
fn test_attractor_step_counting_and_divergence_metadata() {
    let mut plotter = AttractorPlotter::default();
    assert_eq!(plotter.step_count, 0);

    plotter.step_simulation();
    assert_eq!(plotter.execution_state, SolverExecutionState::Running);
    assert_eq!(plotter.step_count, plotter.simulation_speed);

    let datasets = plotter.export_datasets();
    let options = egui_plot::export::ExportOptions::default();
    if let Ok(png_data) = egui_plot::export::export_custom_png(&datasets, &options) {
        let _ = std::fs::write("/tmp/attractor_plotter_trajectory.png", png_data);
    }

    // Force divergence during stepping
    plotter.system.state.vec.y = f64::INFINITY;
    plotter.step_simulation();

    assert!(plotter.diverged());
    match plotter.execution_state {
        SolverExecutionState::Diverged {
            invalid_vec,
            diverged_at_step,
        } => {
            assert!(invalid_vec.y.is_infinite());
            assert_eq!(diverged_at_step, plotter.simulation_speed);
        }
        _ => panic!("Expected Diverged execution state"),
    }
}

#[test]
fn test_attractor_annotations_toggle_and_fixed_points() {
    let mut plotter = AttractorPlotter::default();
    assert!(plotter.show_annotations);

    // Toggle off
    plotter.show_annotations = false;
    assert!(!plotter.show_annotations);

    // Toggle on
    plotter.show_annotations = true;
    assert!(plotter.show_annotations);

    // Verify fixed point coordinates for standard Lorenz parameters (rho=28, beta=8/3)
    let rho = plotter.system.rho;
    let beta = plotter.system.beta;
    assert!(rho > 1.0 && beta > 0.0);
    let x = (beta * (rho - 1.0)).sqrt();
    let z = rho - 1.0;
    // For standard parameters: x ~ 8.485, z = 27.0
    assert!((x - (8.0 / 3.0 * 27.0_f64).sqrt()).abs() < 1e-10);
    assert_eq!(z, 27.0);

    // Verify 3D projection produces finite 2D canvas coordinates
    let c_pos = plotter.project(Vector3::new(x, x, z));
    let c_neg = plotter.project(Vector3::new(-x, -x, z));
    assert!(c_pos[0].is_finite() && c_pos[1].is_finite());
    assert!(c_neg[0].is_finite() && c_neg[1].is_finite());

    // Test UI rendering runs cleanly with annotations enabled and disabled
    let ctx = egui::Context::default();
    let _ = ctx.run(egui::RawInput::default(), |ctx| plotter.show(ctx));

    plotter.show_annotations = false;
    let _ = ctx.run(egui::RawInput::default(), |ctx| plotter.show(ctx));
}

#[test]
fn test_attractor_state_serialization_roundtrip_with_annotations() {
    let plotter = AttractorPlotter {
        show_annotations: false,
        ..Default::default()
    };

    let saved = plotter.save_state().expect("save_state should succeed");
    let mut restored = AttractorPlotter::default();
    restored.load_state(&saved);

    assert!(!restored.show_annotations);
}

#[test]
fn test_attractor_execution_state_roundtrip_all_variants() {
    // 1. Ready
    let plotter = AttractorPlotter {
        execution_state: SolverExecutionState::Ready,
        ..Default::default()
    };
    let saved = plotter.save_state().expect("save_state should succeed");
    let mut restored = AttractorPlotter::default();
    restored.load_state(&saved);
    assert_eq!(restored.execution_state, SolverExecutionState::Ready);

    // 2. Running
    let plotter = AttractorPlotter {
        execution_state: SolverExecutionState::Running,
        ..Default::default()
    };
    let saved = plotter.save_state().expect("save_state should succeed");
    let mut restored = AttractorPlotter::default();
    restored.load_state(&saved);
    assert_eq!(restored.execution_state, SolverExecutionState::Running);

    // 3. Paused
    let plotter = AttractorPlotter {
        execution_state: SolverExecutionState::Paused,
        ..Default::default()
    };
    let saved = plotter.save_state().expect("save_state should succeed");
    let mut restored = AttractorPlotter::default();
    restored.load_state(&saved);
    assert_eq!(restored.execution_state, SolverExecutionState::Paused);

    // 4. Diverged with non-finite coordinates (NaN, Infinity, -Infinity)
    let invalid_vec = Vector3::new(f64::NAN, f64::INFINITY, f64::NEG_INFINITY);
    let plotter = AttractorPlotter {
        execution_state: SolverExecutionState::Diverged {
            invalid_vec,
            diverged_at_step: 123,
        },
        step_count: 123,
        ..Default::default()
    };
    let saved = plotter.save_state().expect("save_state should succeed");
    let mut restored = AttractorPlotter::default();
    restored.load_state(&saved);

    assert!(restored.diverged());
    assert!(restored.is_paused());
    assert_eq!(restored.step_count, 123);
    match restored.execution_state {
        SolverExecutionState::Diverged {
            invalid_vec: restored_vec,
            diverged_at_step,
        } => {
            assert!(restored_vec.x.is_nan());
            assert_eq!(restored_vec.y, f64::INFINITY);
            assert_eq!(restored_vec.z, f64::NEG_INFINITY);
            assert_eq!(diverged_at_step, 123);
        }
        _ => panic!("Expected Diverged state"),
    }
    // Verify non-finite state coordinates were preserved on system.state.vec
    assert!(restored.system.state.vec.x.is_nan());
    assert_eq!(restored.system.state.vec.y, f64::INFINITY);
    assert_eq!(restored.system.state.vec.z, f64::NEG_INFINITY);

    // Illegal resumption must be rejected
    assert!(restored.try_resume().is_err());

    // Verify divergence warning banner renders cleanly without crashing
    let ctx = egui::Context::default();
    let _ = ctx.run(egui::RawInput::default(), |ctx| restored.show(ctx));
}

#[test]
fn test_attractor_legacy_session_backward_compatibility() {
    // Legacy file with paused = true
    let legacy_paused_json = r#"{
        "sigma": 15.5,
        "rho": 35.0,
        "beta": 3.25,
        "dt": 0.005,
        "simulation_speed": 8,
        "paused": true,
        "show_annotations": false
    }"#;
    let mut plotter = AttractorPlotter::default();
    plotter.load_state(legacy_paused_json);

    assert_eq!(plotter.execution_state, SolverExecutionState::Paused);
    assert_eq!(plotter.system.sigma, 15.5);
    assert_eq!(plotter.system.rho, 35.0);
    assert_eq!(plotter.system.beta, 3.25);
    assert_eq!(plotter.dt, 0.005);
    assert_eq!(plotter.simulation_speed, 8);
    assert!(!plotter.show_annotations);

    // Legacy file with paused = false
    let legacy_running_json = r#"{
        "sigma": 10.0,
        "rho": 28.0,
        "beta": 2.6666666666666665,
        "dt": 0.01,
        "simulation_speed": 5,
        "paused": false,
        "show_annotations": true
    }"#;
    let mut plotter2 = AttractorPlotter::default();
    plotter2.load_state(legacy_running_json);

    assert_eq!(plotter2.execution_state, SolverExecutionState::Ready);
    assert_eq!(plotter2.system.sigma, 10.0);
    assert!(plotter2.show_annotations);
}

#[test]
fn test_attractor_poisoned_lock_recovery() {
    let mut plotter = AttractorPlotter::default();

    // Poison shared parameter RwLock
    let params_clone = Arc::clone(&plotter.params);
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut guard = params_clone.write().unwrap();
        guard.sigma = 14.0;
        guard.simulation_speed = 4;
        panic!("simulated worker panic");
    }));
    assert!(plotter.params.is_poisoned());

    // Runner reset, speed update, and step survive poisoned lock
    let mut runner = AttractorRunner::from_params(Arc::clone(&plotter.params));
    runner.process_command(SimCommand::Reset);
    assert_eq!(runner.system.sigma, 14.0);

    runner.process_command(SimCommand::SetSpeed(6));
    runner.step();
    assert_eq!(runner.step_count, 6);
    assert_eq!(runner.history.len(), 6);
    assert_eq!(runner.execution_state, SolverExecutionState::Running);

    // Slider handler and plotter reset propagate updates through poisoned lock
    plotter.system.sigma = 12.5;
    plotter.system.rho = 30.0;
    plotter.simulation_speed = 3;
    plotter.on_slider_changed();
    {
        let p = plotter.params.read().unwrap_or_else(|e| e.into_inner());
        assert_eq!(p.sigma, 12.5);
        assert_eq!(p.rho, 30.0);
        assert_eq!(p.simulation_speed, 3);
    }

    runner.step();
    assert_eq!(runner.system.sigma, 12.5);
    assert_eq!(runner.system.rho, 30.0);
    assert_eq!(runner.step_count, 9);

    plotter.reset();
    assert_eq!(plotter.execution_state, SolverExecutionState::Ready);

    // Step trajectory under poisoned lock and export verification plot
    for _ in 0..100 {
        runner.step();
    }
    plotter.history = runner.history.clone();
    let datasets = plotter.export_datasets();
    let options = egui_plot::export::ExportOptions::default();
    if let Ok(png_data) = egui_plot::export::export_custom_png(&datasets, &options) {
        let _ = std::fs::write("/tmp/poisoned_lock_recovery_attractor.png", png_data);
    }

    let ctx = egui::Context::default();
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        plotter.show(ctx);
    });
}

#[test]
fn test_attractor_camera_preset_roundtrip_and_validation() {
    let mut plotter = AttractorPlotter::default();
    let base_rev = plotter.state_revision();
    plotter.camera = Camera3D::new(0.45, -0.75, 1.35);
    assert_ne!(plotter.state_revision(), base_rev);

    let saved = plotter.save_state().expect("camera state should serialize");
    let mut restored = AttractorPlotter::default();
    restored
        .try_load_state(&saved)
        .expect("camera state should deserialize");
    assert!((restored.camera.pitch - 0.45).abs() < 1e-6);
    assert!((restored.camera.yaw + 0.75).abs() < 1e-6);
    assert!((restored.camera.zoom - 1.35).abs() < 1e-6);

    restored.camera.reset();
    assert_eq!(
        (
            restored.camera.pitch,
            restored.camera.yaw,
            restored.camera.zoom
        ),
        (0.0, 0.0, 1.0)
    );

    let legacy = r#"{"sigma":10.0,"rho":28.0,"beta":2.6667,"dt":0.01,"simulation_speed":5,"pitch":0.3,"yaw":-0.4,"zoom":1.2}"#;
    restored
        .try_load_state(legacy)
        .expect("legacy pitch/yaw/zoom fields should deserialize");
    assert!((restored.camera.zoom - 1.2).abs() < 1e-6);

    let bad_zoom = r#"{"sigma":10.0,"rho":28.0,"beta":2.6667,"dt":0.01,"simulation_speed":5,"camera":{"pitch":0.0,"yaw":0.0,"zoom":-1.0}}"#;
    assert!(restored.try_load_state(bad_zoom).is_err());
}

fn find_callout_center(shapes: &[egui::epaint::ClippedShape], needle: &str) -> egui::Pos2 {
    for clipped in shapes {
        if let egui::epaint::Shape::Text(text) = &clipped.shape {
            if text.galley.text().contains(needle) {
                return text.visual_bounding_rect().center();
            }
        }
    }
    panic!("Callout containing '{needle}' not found in output shapes");
}

#[test]
fn test_attractor_fixed_callout_drag_leaves_camera_unchanged() {
    let ctx = egui::Context::default();
    let mut plotter = AttractorPlotter::default();
    let screen = Some(egui::Rect::from_min_size(
        egui::Pos2::ZERO,
        egui::vec2(1000.0, 700.0),
    ));
    let _ = ctx.run(
        egui::RawInput {
            screen_rect: screen,
            ..Default::default()
        },
        |ctx| plotter.show(ctx),
    );
    let out = ctx.run(
        egui::RawInput {
            screen_rect: screen,
            ..Default::default()
        },
        |ctx| plotter.show(ctx),
    );

    let start = find_callout_center(&out.shapes, "Fixed Point C+");
    let end = start + egui::vec2(30.0, -20.0);
    let initial_offset = plotter.fixed_callout_offsets[0];
    let initial_pose = (plotter.camera.pitch, plotter.camera.yaw);

    let press = egui::RawInput {
        screen_rect: screen,
        events: vec![
            egui::Event::PointerMoved(start),
            egui::Event::PointerButton {
                pos: start,
                button: egui::PointerButton::Primary,
                pressed: true,
                modifiers: egui::Modifiers::NONE,
            },
        ],
        ..Default::default()
    };
    let _ = ctx.run(press, |ctx| plotter.show(ctx));
    let drag = egui::RawInput {
        screen_rect: screen,
        events: vec![egui::Event::PointerMoved(end)],
        ..Default::default()
    };
    let _ = ctx.run(drag, |ctx| plotter.show(ctx));
    let release = egui::RawInput {
        screen_rect: screen,
        events: vec![egui::Event::PointerButton {
            pos: end,
            button: egui::PointerButton::Primary,
            pressed: false,
            modifiers: egui::Modifiers::NONE,
        }],
        ..Default::default()
    };
    let _ = ctx.run(release, |ctx| plotter.show(ctx));

    assert_ne!(plotter.fixed_callout_offsets[0], initial_offset);
    assert_eq!((plotter.camera.pitch, plotter.camera.yaw), initial_pose);
}
