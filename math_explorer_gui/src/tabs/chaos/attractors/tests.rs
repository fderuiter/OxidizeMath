#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;

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
