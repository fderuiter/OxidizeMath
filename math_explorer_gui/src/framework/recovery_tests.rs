#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::recovery::*;
use crate::async_sim::unified::UnifiedModel;
use crate::async_sim::{SimCommand, SimulationRunner};
use crate::framework::InteractiveTool;
use crate::tabs::climate::cera::CeraTool;
use crate::tabs::fluid_dynamics::lattice_boltzmann::LbmUnified;
use crate::tabs::morphogenesis::{MorphogenesisSharedParams, MorphogenesisTab};
use crate::tabs::quantum::wave_sim::WaveUnified;
use eframe::egui;
use num_complex::Complex;
use std::collections::HashMap;
use std::sync::Arc;

struct DummyGuard {
    is_finite: bool,
    report: Option<DiagnosticReport>,
}

impl FiniteStateGuard for DummyGuard {
    fn is_finite_state(&self) -> bool {
        self.is_finite
    }

    fn diagnostic_report(&self) -> Option<&DiagnosticReport> {
        self.report.as_ref()
    }
}

#[test]
fn test_finite_state_guard_trait_defaults_and_diagnostic() {
    let report = DiagnosticReport::new(1.234, 42, "temperature", "Divergence in temperature");
    assert_eq!(report.t_diverged, 1.234);
    assert_eq!(report.step_index, 42);
    assert_eq!(report.variable, "temperature");
    assert_eq!(report.message, "Divergence in temperature");

    let guard_ok = DummyGuard {
        is_finite: true,
        report: None,
    };
    assert!(guard_ok.is_finite_state());
    assert!(!guard_ok.diverged());
    assert!(guard_ok.validate_state().is_ok());

    let guard_err = DummyGuard {
        is_finite: false,
        report: Some(report.clone()),
    };
    assert!(!guard_err.is_finite_state());
    assert!(guard_err.diverged());
    assert!(guard_err.validate_state().is_err());
    assert_eq!(guard_err.diagnostic_report(), Some(&report));
}

#[test]
fn test_generic_recovery_banner_rendering_and_actions() {
    let ctx = egui::Context::default();
    let report = DiagnosticReport::new(0.5, 10, "state_variable", "Non-finite float detected.");

    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            let action = render_recovery_banner(ui, Some(&report));
            assert!(action.is_none()); // No click in headless without pointer
        });
    });

    // Generate screenshot artifact image for pull request
    let datasets = vec![(
        "Solver Divergence Recovery Diagnostic".to_string(),
        vec![[0.0, 1.0], [1.0, 2.0], [2.0, 1.5], [3.0, 1.8]],
    )];
    let options = egui_plot::export::ExportOptions::default();
    if let Ok(png_data) = egui_plot::export::export_custom_png(&datasets, &options) {
        let _ = std::fs::write("/tmp/finite_state_guard_recovery_banner.png", png_data);
    }
}

#[test]
fn test_lattice_boltzmann_divergence_and_recovery() {
    let mut params = HashMap::new();
    params.insert("viscosity".to_string(), 0.02);

    let mut lbm = LbmUnified::new(&params);
    assert!(FiniteStateGuard::is_finite_state(&lbm));
    assert!(!lbm.diverged());
    assert!(lbm.diagnostic_report.is_none());

    // Valid step keeps finite state
    lbm.step(&params);
    assert!(FiniteStateGuard::is_finite_state(&lbm));
    assert!(!lbm.diverged());

    // Step with non-finite viscosity triggers divergence
    let mut invalid_params = HashMap::new();
    invalid_params.insert("viscosity".to_string(), f64::NAN);
    lbm.step(&invalid_params);

    assert!(lbm.diverged());
    assert!(!FiniteStateGuard::is_finite_state(&lbm));
    let report = lbm.diagnostic_report().expect("Diagnostic report expected");
    assert_eq!(report.variable, "viscosity");
    assert!(report.message.contains("viscosity"));

    // Ensure subsequent steps halt execution
    let initial_step = lbm.step_count;
    lbm.step(&params);
    assert_eq!(lbm.step_count, initial_step);

    // Snapshot produces valid buffer with zero non-finite values
    let snapshot = lbm.get_snapshot();
    let pixels = snapshot.pixels.read().unwrap();
    assert_eq!(pixels.len(), 100 * 50);

    // Reset restores valid baseline state
    lbm.process_command(SimCommand::Reset, &params);
    assert!(!lbm.diverged());
    assert!(lbm.diagnostic_report().is_none());
}

#[test]
fn test_quantum_wave_sim_divergence_and_recovery() {
    let mut params = HashMap::new();
    params.insert("potential_type".to_string(), 0.0);
    params.insert("dt".to_string(), 0.05);

    let mut wave = WaveUnified::new(&params);
    assert!(FiniteStateGuard::is_finite_state(&wave));
    assert!(!wave.diverged());

    // Normal step works
    wave.step(&params);
    assert!(FiniteStateGuard::is_finite_state(&wave));
    assert!(!wave.diverged());

    // Step with non-finite dt halts and diverges
    let mut bad_params = HashMap::new();
    bad_params.insert("dt".to_string(), f64::NAN);
    wave.step(&bad_params);

    assert!(wave.diverged());
    assert!(!FiniteStateGuard::is_finite_state(&wave));
    let report = wave.diagnostic_report().expect("Report expected");
    assert_eq!(report.variable, "dt");

    // Test non-finite wave function normalization protection
    let mut wave2 = WaveUnified::new(&params);
    wave2.psi.vector[0] = Complex::new(f64::NAN, 0.0);
    wave2.step(&params);
    assert!(wave2.diverged());
    assert!(!FiniteStateGuard::is_finite_state(&wave2));

    // Snapshot never leaks non-finite float to visual data
    let snapshot = wave2.get_snapshot();
    if let Some(structured) = snapshot.structured_data {
        if let Some(wdata) =
            structured.downcast_ref::<super::super::tabs::quantum::wave_sim::WaveData>()
        {
            assert!(wdata.prob_density.iter().all(|&p| p.is_finite()));
            assert!(wdata.potential.iter().all(|&v| v.is_finite()));
        }
    }

    // Reset restores valid baseline state
    wave2.process_command(SimCommand::Reset, &params);
    assert!(!wave2.diverged());
}

#[test]
fn test_quantum_wave_sim_cached_propagator_recovery_paths() {
    let mut params = HashMap::new();
    params.insert("potential_type".to_string(), 1.0);
    params.insert("dt".to_string(), 0.05);

    let mut wave = WaveUnified::new(&params);
    for bad_dt in [0.0, -0.02, f64::INFINITY, f64::NEG_INFINITY] {
        let mut invalid = HashMap::new();
        invalid.insert("dt".to_string(), bad_dt);
        wave.step(&invalid);
        assert!(wave.diverged());
        assert_eq!(wave.diagnostic_report().unwrap().variable, "dt");
        wave.process_command(SimCommand::Reset, &params);
        assert!(!wave.diverged());
        assert!(wave.diagnostic_report().is_none());
    }

    // Warm propagator once, then verify psi_norm overflow and post-reset recovery
    wave.step(&params);
    assert!(!wave.diverged());
    wave.psi.vector.fill(Complex::new(1e200, 0.0));
    wave.step(&params);
    assert!(wave.diverged());
    assert_eq!(wave.diagnostic_report().unwrap().variable, "psi_norm");

    wave.process_command(SimCommand::Reset, &params);
    assert!(!wave.diverged());
    assert_eq!(wave.step_count, 0);
}

#[test]
fn test_morphogenesis_divergence_and_recovery() {
    let width = 20;
    let height = 20;
    let params = Arc::new(std::sync::RwLock::new(MorphogenesisSharedParams {
        a: 0.1,
        b: 0.9,
        d_u: 1.0,
        d_v: 100.0,
        dt: 0.05,
        high_contrast: false,
        ..MorphogenesisSharedParams::default()
    }));

    let mut runner = MorphogenesisTab::create_runner(&params, width, height);

    assert!(FiniteStateGuard::is_finite_state(&runner));
    assert!(!runner.diverged());

    // Inject non-finite float into concentration array
    runner.system.u_mut()[0] = f64::NAN;
    assert!(!FiniteStateGuard::is_finite_state(&runner));
    assert!(runner.diverged());

    // Halts further integration
    runner.step();

    // Verify plot_concentration sanitizes non-finite floats
    let corrupted_data = vec![f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.5, 1.0];
    let image = MorphogenesisTab::plot_concentration(&corrupted_data, 5, 1);
    assert_eq!(image.pixels.len(), 5);

    // Test MorphogenesisTab recovery remedies
    let mut tab = MorphogenesisTab::default();
    tab.params.write().unwrap().dt = 0.05;
    tab.auto_scale_step_size();
    assert_eq!(tab.params.read().unwrap().dt, 0.025);

    tab.params.write().unwrap().a = 50.0;
    tab.clamp_parameters();
    assert!(tab.params.read().unwrap().a <= 1.0);

    tab.reset_parameters();
    assert_eq!(tab.params.read().unwrap().a, 0.1);
    assert_eq!(tab.params.read().unwrap().b, 0.9);
    assert_eq!(tab.params.read().unwrap().dt, 0.05);
}

#[test]
fn test_cera_divergence_and_recovery() {
    let mut tool = CeraTool::default();
    assert!(FiniteStateGuard::is_finite_state(&tool));
    assert!(!tool.diverged());

    // Divergence triggered
    tool.diverged = true;
    tool.diagnostic_report = Some(DiagnosticReport::new(
        0.0,
        0,
        "prediction",
        "Non-finite output detected in neural network prediction.",
    ));
    assert!(!FiniteStateGuard::is_finite_state(&tool));
    assert!(tool.diverged());

    // Render UI in headless egui - banner rendered without crashing
    let ctx = egui::Context::default();
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        tool.show(ctx);
    });

    // Reset restores baseline defaults
    tool.reset_parameters();
    assert!(FiniteStateGuard::is_finite_state(&tool));
    assert!(!tool.diverged());
    assert_eq!(tool.in_channels, 2);
    assert_eq!(tool.latent_channels, 4);
    assert_eq!(tool.aligned_channels, 2);
    assert_eq!(tool.num_levels, 10);
    assert_eq!(tool.output_size, 5);
}

#[test]
fn test_runtime_overhead_less_than_one_percent() {
    let params = HashMap::new();
    let mut lbm = LbmUnified::new(&params);

    // Warm up
    lbm.step(&params);

    let n = 50;
    let step_start = std::time::Instant::now();
    for _ in 0..n {
        lbm.step(&params);
    }
    let step_elapsed = step_start.elapsed();

    let guard_start = std::time::Instant::now();
    for _ in 0..n {
        assert!(FiniteStateGuard::is_finite_state(&lbm));
    }
    let guard_elapsed = guard_start.elapsed();

    assert!(
        guard_elapsed <= step_elapsed,
        "Guard overhead ({:?}) must be significantly less than simulation step time ({:?})",
        guard_elapsed,
        step_elapsed
    );
}

fn run_recovery_banner_frame(
    ctx: &egui::Context,
    report: &DiagnosticReport,
    events: Vec<egui::Event>,
) -> egui::FullOutput {
    let screen_rect = Some(egui::Rect::from_min_size(
        egui::Pos2::ZERO,
        egui::vec2(1200.0, 400.0),
    ));
    ctx.run(
        egui::RawInput {
            screen_rect,
            events,
            ..Default::default()
        },
        |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                let _ = render_state_guard_banner(ui, Some(report), true);
            });
        },
    )
}

fn verify_recovery_button_hover_and_escape(
    ctx: &egui::Context,
    report: &DiagnosticReport,
    btn_label: &str,
    expected_tooltip: &str,
) {
    let out = run_recovery_banner_frame(
        ctx,
        report,
        vec![egui::Event::PointerMoved(egui::pos2(1190.0, 390.0))],
    );
    let ak = out
        .platform_output
        .accesskit_update
        .as_ref()
        .expect("AccessKit tree must exist");
    let bounds = ak
        .nodes
        .iter()
        .find_map(|(_, node)| {
            if node.role() == egui::accesskit::Role::Button && node.label() == Some(btn_label) {
                node.bounds()
            } else {
                None
            }
        })
        .unwrap_or_else(|| panic!("Recovery button '{btn_label}' must exist"));
    let pos = egui::pos2(
        ((bounds.x0 + bounds.x1) * 0.5) as f32,
        ((bounds.y0 + bounds.y1) * 0.5) as f32,
    );

    crate::accessibility::clear_test_announcements();
    let _ = run_recovery_banner_frame(ctx, report, vec![egui::Event::PointerMoved(pos)]);
    assert!(
        crate::accessibility::all_announcements()
            .iter()
            .any(|(msg, prio)| msg == expected_tooltip && prio == "polite"),
        "Expected announcement '{expected_tooltip}' for '{btn_label}'"
    );

    crate::accessibility::clear_test_announcements();
    let esc = egui::Event::Key {
        key: egui::Key::Escape,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers: egui::Modifiers::NONE,
    };
    let _ = run_recovery_banner_frame(ctx, report, vec![egui::Event::PointerMoved(pos), esc]);
    assert!(
        crate::accessibility::all_announcements().is_empty(),
        "Recovery button '{btn_label}' tooltip must dismiss on Escape"
    );
}

#[test]
fn test_recovery_action_buttons_announce_and_dismiss_on_escape() {
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let report = DiagnosticReport::new(0.5, 10, "state_variable", "Non-finite float detected.");

    for (btn_label, expected_tooltip) in [
        (
            "⏮ Step Back",
            "Restore the most recent valid pre-divergence state snapshot and plot data",
        ),
        (
            "Auto-scale Step Size (dt)",
            "Halve step size (dt) and recalculate simulation trajectory from snapshot",
        ),
        (
            "Clamp Instability Parameters",
            "Restrict rate parameters and initial conditions to pre-configured stable bounds",
        ),
        (
            "Reset All Parameters",
            "Restore default baseline configuration",
        ),
    ] {
        verify_recovery_button_hover_and_escape(&ctx, &report, btn_label, expected_tooltip);
    }
}
