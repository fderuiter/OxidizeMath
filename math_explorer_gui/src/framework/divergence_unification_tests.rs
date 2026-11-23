#![cfg_attr(any(), verified(opt_out = "tests"))]

use crate::async_sim::unified::UnifiedSimTool;
use crate::async_sim::SimCommand;
use crate::framework::recovery::DiagnosticReport;
use crate::framework::SimulationFramework;
use crate::tabs::analysis::ode::OdeSolverTool;
use crate::tabs::chaos::attractors::{AttractorPlotter, SolverExecutionState};
use crate::tabs::climate::cera::CeraTool;
use crate::tabs::fluid_dynamics::lattice_boltzmann::LbmUnified;
use crate::tabs::mri::MriTool;
use crate::tabs::neuroscience::hodgkin_huxley::HodgkinHuxleyUnified;
use crate::tabs::quantum::wave_sim::WaveSimulatorTool;
use eframe::egui;
use nalgebra::Vector3;

fn select_matching_tool(fw: &mut SimulationFramework, needle: &str) {
    let idx = fw
        .available_tools
        .iter()
        .position(|t| t.id.contains(needle) || t.display_name.contains(needle))
        .unwrap_or_else(|| panic!("Tool matching '{needle}' not found in {}", fw.domain));
    fw.select_tool(idx);
}

fn render_frame(ctx: &egui::Context, fw: &mut SimulationFramework, id: &str) -> egui::FullOutput {
    let rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1400.0, 900.0));
    let was_diverged = fw.is_diverged();
    let out = ctx.run(
        egui::RawInput {
            screen_rect: Some(rect),
            ..Default::default()
        },
        |ctx| fw.show(ctx, id),
    );
    if was_diverged && !fw.is_diverged() {
        return ctx.run(
            egui::RawInput {
                screen_rect: Some(rect),
                ..Default::default()
            },
            |ctx| fw.show(ctx, id),
        );
    }
    out
}

fn count_text_occurrences(output: &egui::FullOutput, needle: &str) -> usize {
    let mut count = 0;
    let mut stack: Vec<&egui::epaint::Shape> = output.shapes.iter().map(|c| &c.shape).collect();
    while let Some(shape) = stack.pop() {
        match shape {
            egui::epaint::Shape::Text(t) if t.galley.job.text.contains(needle) => {
                count += 1;
            }
            egui::epaint::Shape::Vec(v) => stack.extend(v.iter()),
            _ => {}
        }
    }
    count
}

fn assert_single_banner_and_solver_report(
    output: &egui::FullOutput,
    fw: &SimulationFramework,
    expected_t: f64,
    expected_step: usize,
    expected_var: &str,
) {
    assert!(fw.is_diverged());
    let report = fw
        .diagnostic_report()
        .expect("Diverged framework must expose solver DiagnosticReport");
    assert!(
        (report.t_diverged - expected_t).abs() < 1e-9,
        "Expected solver t_diverged={expected_t}, got {} (session elapsed={})",
        report.t_diverged,
        fw.start_time.elapsed().as_secs_f64()
    );
    assert_eq!(report.step_index, expected_step);
    assert_eq!(report.variable, expected_var);
    assert_eq!(
        count_text_occurrences(output, "Solver Divergence Warning"),
        1,
        "Must render only the single shared top recovery banner"
    );
    assert_eq!(
        count_text_occurrences(output, "Simulation Worker Faulted"),
        0,
        "Must not render duplicate worker fault banner"
    );
}

fn verify_ode_divergence(ctx: &egui::Context) {
    let mut ode_fw = SimulationFramework::new("analysis");
    select_matching_tool(&mut ode_fw, "OdeSolverTool");
    std::thread::sleep(std::time::Duration::from_millis(25));
    {
        let ode = ode_fw
            .active_tool
            .as_mut()
            .and_then(|t| t.as_any_mut())
            .and_then(|a| a.downcast_mut::<OdeSolverTool>())
            .unwrap();
        ode.diverged = true;
        ode.diagnostic_report = Some(DiagnosticReport::new(
            1.25,
            25,
            "y(t)",
            "ODE diverged at t=1.2500s",
        ));
    }
    let out = render_frame(ctx, &mut ode_fw, "ode_div_test");
    assert_single_banner_and_solver_report(&out, &ode_fw, 1.25, 25, "y(t)");

    ode_fw
        .active_tool
        .as_mut()
        .and_then(|t| t.as_any_mut())
        .and_then(|a| a.downcast_mut::<OdeSolverTool>())
        .unwrap()
        .recalculate();
    let out_recovered = render_frame(ctx, &mut ode_fw, "ode_div_test");
    assert!(!ode_fw.is_diverged() && ode_fw.diagnostic_report().is_none());
    assert_eq!(
        count_text_occurrences(&out_recovered, "Solver Divergence Warning"),
        0
    );
}

fn verify_cera_divergence(ctx: &egui::Context) {
    let mut cera_fw = SimulationFramework::new("climate");
    select_matching_tool(&mut cera_fw, "CeraTool");
    std::thread::sleep(std::time::Duration::from_millis(25));
    {
        let cera = cera_fw
            .active_tool
            .as_mut()
            .and_then(|t| t.as_any_mut())
            .and_then(|a| a.downcast_mut::<CeraTool>())
            .unwrap();
        cera.diverged = true;
        cera.diagnostic_report = Some(DiagnosticReport::new(
            0.0,
            6,
            "prediction",
            "CERA output diverged",
        ));
    }
    let cera_out = render_frame(ctx, &mut cera_fw, "cera_div_test");
    assert_single_banner_and_solver_report(&cera_out, &cera_fw, 0.0, 6, "prediction");
    assert!(cera_fw.rollback_auto_scale_step_size());
    let cera_recovered = render_frame(ctx, &mut cera_fw, "cera_div_test");
    assert!(!cera_fw.is_diverged());
    assert_eq!(
        count_text_occurrences(&cera_recovered, "Solver Divergence Warning"),
        0
    );
}

#[test]
fn test_ode_and_cera_single_banner_solver_diagnostics_and_param_edit_auto_dismiss() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    egui_plot::clear_last_plot_datasets();
    let ctx = egui::Context::default();
    verify_ode_divergence(&ctx);
    verify_cera_divergence(&ctx);
}

fn verify_hh_divergence(ctx: &egui::Context) {
    let mut hh_fw = SimulationFramework::new("neuroscience");
    select_matching_tool(&mut hh_fw, "Hodgkin");
    std::thread::sleep(std::time::Duration::from_millis(25));
    {
        let hh = hh_fw
            .active_tool
            .as_mut()
            .and_then(|t| t.as_any_mut())
            .and_then(|a| a.downcast_mut::<UnifiedSimTool<HodgkinHuxleyUnified>>())
            .unwrap();
        hh.params
            .write()
            .unwrap()
            .insert("dt".to_string(), f64::NAN);
    }
    let hh_out = render_frame(ctx, &mut hh_fw, "hh_div_test");
    assert_single_banner_and_solver_report(&hh_out, &hh_fw, 0.0, 0, "dt");
    hh_fw.clamp_parameters();
    let hh_rec = render_frame(ctx, &mut hh_fw, "hh_div_test");
    assert!(!hh_fw.is_diverged());
    assert_eq!(
        count_text_occurrences(&hh_rec, "Solver Divergence Warning"),
        0
    );
}

fn verify_lbm_divergence(ctx: &egui::Context) {
    let mut lbm_fw = SimulationFramework::new("fluid_dynamics");
    select_matching_tool(&mut lbm_fw, "Boltzmann");
    std::thread::sleep(std::time::Duration::from_millis(25));
    {
        let lbm = lbm_fw
            .active_tool
            .as_mut()
            .and_then(|t| t.as_any_mut())
            .and_then(|a| a.downcast_mut::<UnifiedSimTool<LbmUnified>>())
            .unwrap();
        lbm.params
            .write()
            .unwrap()
            .insert("viscosity".to_string(), f64::NAN);
        let _ = lbm.controller.send_command(SimCommand::Step(1));
        for _ in 0..100 {
            let _ = lbm.controller.update();
            if lbm.state().is_faulted() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(lbm.state().is_faulted());
    }
    let lbm_out = render_frame(ctx, &mut lbm_fw, "lbm_div_test");
    assert_single_banner_and_solver_report(&lbm_out, &lbm_fw, 0.0, 0, "viscosity");
    assert!(lbm_fw.rollback_auto_scale_step_size());
    let lbm_rec = render_frame(ctx, &mut lbm_fw, "lbm_div_test");
    assert!(!lbm_fw.is_diverged());
    assert_eq!(
        count_text_occurrences(&lbm_rec, "Solver Divergence Warning"),
        0
    );
}

fn verify_wave_divergence(ctx: &egui::Context) {
    let mut wave_fw = SimulationFramework::new("quantum");
    select_matching_tool(&mut wave_fw, "WaveSimulator");
    std::thread::sleep(std::time::Duration::from_millis(25));
    {
        let wave = wave_fw
            .active_tool
            .as_mut()
            .and_then(|t| t.as_any_mut())
            .and_then(|a| a.downcast_mut::<WaveSimulatorTool>())
            .unwrap();
        wave.shared_config.write().unwrap().dt = f64::NAN;
        let _ = wave.worker.send_command(SimCommand::Step(1));
        for _ in 0..100 {
            let _ = wave.worker.update();
            if wave.state().is_faulted() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(wave.state().is_faulted());
    }
    let wave_out = render_frame(ctx, &mut wave_fw, "wave_div_test");
    assert_single_banner_and_solver_report(&wave_out, &wave_fw, 0.0, 0, "dt");
    {
        let wave = wave_fw
            .active_tool
            .as_mut()
            .and_then(|t| t.as_any_mut())
            .and_then(|a| a.downcast_mut::<WaveSimulatorTool>())
            .unwrap();
        wave.dt = 0.01;
        wave.sync_config_and_reset(None);
    }
    let wave_rec = render_frame(ctx, &mut wave_fw, "wave_div_test");
    assert!(!wave_fw.is_diverged());
    assert_eq!(
        count_text_occurrences(&wave_rec, "Solver Divergence Warning"),
        0
    );
}

#[test]
fn test_unified_hh_lbm_and_wave_single_banner_and_recovery_auto_dismiss() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    egui_plot::clear_last_plot_datasets();
    let ctx = egui::Context::default();
    verify_hh_divergence(&ctx);
    verify_lbm_divergence(&ctx);
    verify_wave_divergence(&ctx);
}

fn verify_attractor_divergence(ctx: &egui::Context) {
    let mut attr_fw = SimulationFramework::new("chaos");
    select_matching_tool(&mut attr_fw, "AttractorPlotter");
    std::thread::sleep(std::time::Duration::from_millis(25));
    {
        let attr = attr_fw
            .active_tool
            .as_mut()
            .and_then(|t| t.as_any_mut())
            .and_then(|a| a.downcast_mut::<AttractorPlotter>())
            .unwrap();
        attr.dt = 0.02;
        attr.execution_state = SolverExecutionState::Diverged {
            invalid_vec: Vector3::new(1.0, f64::INFINITY, 2.0),
            diverged_at_step: 15,
        };
    }
    let attr_out = render_frame(ctx, &mut attr_fw, "attr_div_test");
    assert_single_banner_and_solver_report(&attr_out, &attr_fw, 0.30, 15, "y");
    attr_fw
        .active_tool
        .as_mut()
        .and_then(|t| t.as_any_mut())
        .and_then(|a| a.downcast_mut::<AttractorPlotter>())
        .unwrap()
        .reset();
    let attr_rec = render_frame(ctx, &mut attr_fw, "attr_div_test");
    assert!(!attr_fw.is_diverged() && attr_fw.diagnostic_report().is_none());
    assert_eq!(
        count_text_occurrences(&attr_rec, "Solver Divergence Warning"),
        0
    );
}

fn verify_mri_divergence(ctx: &egui::Context) {
    let mut mri_fw = SimulationFramework::new("mri");
    select_matching_tool(&mut mri_fw, "MriTool");
    std::thread::sleep(std::time::Duration::from_millis(25));
    {
        let mri = mri_fw
            .active_tool
            .as_mut()
            .and_then(|t| t.as_any_mut())
            .and_then(|a| a.downcast_mut::<MriTool>())
            .unwrap();
        let _ = mri.worker.send_command(SimCommand::Pause);
        mri.worker.wait_until_paused();
        let _ = mri.worker.update();
        mri.current_time = 0.42;
        mri.history.clear();
        mri.history.push_back((0.0, Vector3::new(1.0, 0.0, 0.0)));
        mri.history.push_back((0.21, Vector3::new(0.8, 0.1, 0.2)));
        mri.simulator.magnetization.x = f64::NAN;
    }
    let mri_out = render_frame(ctx, &mut mri_fw, "mri_div_test");
    assert_single_banner_and_solver_report(&mri_out, &mri_fw, 0.42, 2, "magnetization");
    mri_fw
        .active_tool
        .as_mut()
        .and_then(|t| t.as_any_mut())
        .and_then(|a| a.downcast_mut::<MriTool>())
        .unwrap()
        .reset();
    let mri_rec = render_frame(ctx, &mut mri_fw, "mri_div_test");
    assert!(!mri_fw.is_diverged() && mri_fw.diagnostic_report().is_none());
    assert_eq!(
        count_text_occurrences(&mri_rec, "Solver Divergence Warning"),
        0
    );
}

#[test]
fn test_attractor_and_mri_single_banner_and_playback_reset_auto_dismiss() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    egui_plot::clear_last_plot_datasets();
    let ctx = egui::Context::default();
    verify_attractor_divergence(&ctx);
    verify_mri_divergence(&ctx);
}

#[test]
fn test_import_url_preset_json_preserves_factory_and_preset_baselines() {
    let mut fw = SimulationFramework::new("analysis");
    assert!(fw.available_tools.len() >= 2);
    let ode_idx = fw
        .available_tools
        .iter()
        .position(|t| t.id == "OdeSolverTool")
        .unwrap();
    let other_idx = if ode_idx == 0 { 1 } else { 0 };
    fw.select_tool(other_idx);

    let ode_factory = (fw.available_tools[ode_idx].build)().save_state().unwrap();
    let mut cfg: serde_json::Value = serde_json::from_str(&ode_factory).unwrap();
    cfg.as_object_mut()
        .unwrap()
        .insert("dt".to_string(), serde_json::json!(0.09));
    let url_envelope = crate::presets::serialize_preset_envelope("ODE Solvers", "analysis", cfg)
        .expect("envelope");

    fw.import_url_preset_json(Some("ODE Solvers"), Some(&url_envelope))
        .expect("URL preset import should switch tool and apply preset");

    let imported_state = fw.active_tool.as_ref().unwrap().save_state().unwrap();
    assert_ne!(imported_state, ode_factory);
    assert_eq!(fw.factory_tool_state.as_deref(), Some(ode_factory.as_str()));
    assert_eq!(
        fw.preset_tool_state.as_deref(),
        Some(imported_state.as_str())
    );
    fw.reset_to_factory_defaults();
    assert_eq!(
        fw.active_tool.as_ref().unwrap().save_state().as_deref(),
        Some(ode_factory.as_str())
    );
    assert!(fw.reset_to_preset());
    assert_eq!(
        fw.active_tool.as_ref().unwrap().save_state().as_deref(),
        Some(imported_state.as_str())
    );
}
