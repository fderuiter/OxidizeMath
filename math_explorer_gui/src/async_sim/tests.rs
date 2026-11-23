use super::*;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
#[cfg(not(target_arch = "wasm32"))]
use std::thread;
use std::time::Duration;

#[cfg(not(target_arch = "wasm32"))]
struct MockRunner {
    step_count: Arc<AtomicUsize>,
    steps_per_frame: usize,
    max_steps: usize,
}

#[cfg(not(target_arch = "wasm32"))]
impl SimulationRunner for MockRunner {
    fn process_command(&mut self, cmd: SimCommand) {
        if let SimCommand::SetSpeed(speed) = cmd {
            self.steps_per_frame = speed;
        }
    }

    fn step(&mut self) {
        self.step_count.fetch_add(1, Ordering::SeqCst);
    }

    fn get_snapshot(&self) -> StateSnapshot {
        StateSnapshot {
            width: 1,
            height: 1,
            pixels: Arc::new(std::sync::RwLock::new(vec![eframe::egui::Color32::WHITE])),
            custom_data: vec![self.step_count.load(Ordering::SeqCst) as f64],
            structured_data: None,
            sonification: crate::audio::SonificationFrame::silent(),
        }
    }

    fn get_steps_per_frame(&self) -> usize {
        self.steps_per_frame
    }

    fn is_complete(&self) -> bool {
        self.step_count.load(Ordering::SeqCst) >= self.max_steps
    }
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_simulation_controller_start_pause() {
    let steps = Arc::new(AtomicUsize::new(0));
    let runner = MockRunner {
        step_count: Arc::clone(&steps),
        steps_per_frame: 2,
        max_steps: 1000,
    };

    let mut controller = SimulationController::new(runner);
    assert!(!controller.is_running());
    assert_eq!(steps.load(Ordering::SeqCst), 0);

    // Start simulation and poll until the worker produces a running snapshot
    let _ = controller.send_command(SimCommand::Start);
    let mut snap_is_some = false;
    for _ in 0..400 {
        if controller.update().is_some() && steps.load(Ordering::SeqCst) > 0 {
            snap_is_some = true;
            break;
        }
        thread::sleep(Duration::from_millis(5));
    }
    assert!(controller.is_running());
    assert!(snap_is_some);
    let current_steps = steps.load(Ordering::SeqCst);
    assert!(current_steps > 0);

    // Pause simulation and wait for worker acknowledgment
    let _ = controller.send_command(SimCommand::Pause);
    controller.wait_until_paused();
    let _ = controller.update();
    assert!(!controller.is_running());
    let paused_steps = steps.load(Ordering::SeqCst);

    // Ensure zero background processing while paused
    thread::sleep(Duration::from_millis(50));
    assert_eq!(steps.load(Ordering::SeqCst), paused_steps);
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_runner_auto_completion() {
    let steps = Arc::new(AtomicUsize::new(0));
    let runner = MockRunner {
        step_count: Arc::clone(&steps),
        steps_per_frame: 5,
        max_steps: 15,
    };

    let mut controller = SimulationController::new(runner);
    let _ = controller.send_command(SimCommand::Start);

    // Wait for worker to finish max_steps
    for _ in 0..400 {
        thread::sleep(Duration::from_millis(15));
        controller.update();
        if !controller.is_running() && steps.load(Ordering::SeqCst) >= 15 {
            break;
        }
    }

    assert_eq!(steps.load(Ordering::SeqCst), 15);
    assert!(!controller.is_running());
}

struct TestRunner {
    step_count: usize,
    steps_per_frame: usize,
    panic_flag: Arc<AtomicBool>,
}

impl TestRunner {
    fn new(panic_flag: Arc<AtomicBool>) -> Self {
        Self {
            step_count: 0,
            steps_per_frame: 1,
            panic_flag,
        }
    }
}

impl SimulationRunner for TestRunner {
    fn process_command(&mut self, cmd: SimCommand) {
        if let SimCommand::SetSpeed(speed) = cmd {
            self.steps_per_frame = speed;
        }
    }

    fn step(&mut self) {
        if self.panic_flag.load(Ordering::SeqCst) {
            panic!("Simulated worker panic");
        }
        self.step_count += 1;
    }

    fn get_snapshot(&self) -> StateSnapshot {
        StateSnapshot {
            width: 1,
            height: 1,
            pixels: Arc::new(std::sync::RwLock::new(vec![eframe::egui::Color32::BLACK])),
            custom_data: vec![self.step_count as f64],
            structured_data: None,
            sonification: crate::audio::SonificationFrame::silent(),
        }
    }

    fn get_steps_per_frame(&self) -> usize {
        self.steps_per_frame
    }
}

#[test]
fn test_initial_state_is_ready() {
    let panic_flag = Arc::new(AtomicBool::new(false));
    let runner = TestRunner::new(panic_flag);
    let controller = SimulationController::new(runner);

    assert_eq!(controller.state(), SimWorkerState::Ready);
    assert!(controller.state().is_ready());
    assert!(!controller.state().is_running());
    assert!(!controller.state().is_faulted());
    assert!(!controller.state().is_terminated());
}

#[test]
fn test_valid_start_and_pause_transitions() {
    let panic_flag = Arc::new(AtomicBool::new(false));
    let runner = TestRunner::new(panic_flag);
    let mut controller = SimulationController::new(runner);

    // Start running
    assert!(controller.send_command(SimCommand::Start).is_ok());
    assert_eq!(controller.state(), SimWorkerState::Running);
    assert!(controller.state().is_running());

    // Pause
    assert!(controller.send_command(SimCommand::Pause).is_ok());
    assert_eq!(controller.state(), SimWorkerState::Ready);
    assert!(controller.state().is_ready());
}

#[test]
fn test_invalid_command_transitions_rejected() {
    let panic_flag = Arc::new(AtomicBool::new(false));
    let runner = TestRunner::new(panic_flag);
    let mut controller = SimulationController::new(runner);

    // Initial state is Ready, so Pause should be rejected
    let res = controller.send_command(SimCommand::Pause);
    assert!(matches!(
        res,
        Err(SimCommandError::InvalidTransition {
            current_state: SimWorkerState::Ready,
            command: "Pause"
        })
    ));

    // Transition to Running
    assert!(controller.send_command(SimCommand::Start).is_ok());

    // Running state, so Start should be rejected
    let res = controller.send_command(SimCommand::Start);
    assert!(matches!(
        res,
        Err(SimCommandError::InvalidTransition {
            current_state: SimWorkerState::Running,
            command: "Start"
        })
    ));
}

#[test]
fn test_reset_command_from_running_pauses_to_ready() {
    let panic_flag = Arc::new(AtomicBool::new(false));
    let runner = TestRunner::new(panic_flag);
    let mut controller = SimulationController::new(runner);

    assert!(controller.send_command(SimCommand::Start).is_ok());
    assert_eq!(controller.state(), SimWorkerState::Running);

    assert!(controller.send_command(SimCommand::Reset).is_ok());
    assert_eq!(controller.state(), SimWorkerState::Ready);
}

#[test]
fn test_worker_panic_transitions_to_faulted_within_one_frame() {
    let panic_flag = Arc::new(AtomicBool::new(false));
    let runner = TestRunner::new(panic_flag.clone());
    let mut controller = SimulationController::new(runner);
    assert!(controller.update().is_some());

    // Trigger panic in background worker on its next step
    panic_flag.store(true, Ordering::SeqCst);
    assert!(controller.send_command(SimCommand::Start).is_ok());

    // Wait for background thread to hit the step panic and unwind its runner handle
    for _ in 0..400 {
        if Arc::strong_count(&panic_flag) == 1 {
            break;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    std::thread::sleep(Duration::from_millis(15));

    // On the next frame update, disconnection must be detected and state set to Faulted
    assert!(controller.update().is_none());
    assert_eq!(controller.state(), SimWorkerState::Faulted);
    assert!(controller.state().is_faulted());
    assert!(!controller.state().is_running());

    // Non-reset commands in Faulted state must be rejected
    assert!(matches!(
        controller.send_command(SimCommand::Start),
        Err(SimCommandError::InvalidTransition { .. })
    ));

    // Direct Reset on a panicked worker detects channel disconnection
    assert!(matches!(
        controller.send_command(SimCommand::Reset),
        Err(SimCommandError::ChannelDisconnected)
    ));

    // Respawn helper recovers panicked worker to Ready with a seeded snapshot
    panic_flag.store(false, Ordering::SeqCst);
    assert!(controller.reset_or_respawn(|| TestRunner::new(panic_flag.clone())));
    assert_eq!(controller.state(), SimWorkerState::Ready);
    assert!(controller.latest_snapshot().is_some());
}

#[test]
fn test_update_returns_snapshot_only_on_new_snapshot() {
    let panic_flag = Arc::new(AtomicBool::new(false));
    let runner = TestRunner::new(panic_flag);
    let mut controller = SimulationController::new(runner);

    // Seed snapshot is populated immediately and delivered once via update()
    assert!(controller.latest_snapshot().is_some());
    assert!(controller.update().is_some());
    assert!(controller.update().is_none());

    // Start simulation and poll until a step snapshot arrives
    assert!(controller.send_command(SimCommand::Start).is_ok());
    let mut got_step_snapshot = false;
    for _ in 0..400 {
        if controller.update().is_some() {
            got_step_snapshot = true;
            break;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    assert!(got_step_snapshot);

    // Pause simulation and wait for worker Ready acknowledgment
    assert!(controller.send_command(SimCommand::Pause).is_ok());
    controller.wait_until_paused();
    // Drain any final pre-pause snapshot
    let _ = controller.update();

    // Subsequent updates while paused should return None (no new snapshots)
    assert!(controller.update().is_none());
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_pause_pending_clears_upon_worker_ready_acknowledgment() {
    let steps = Arc::new(AtomicUsize::new(0));
    let runner = MockRunner {
        step_count: Arc::clone(&steps),
        steps_per_frame: 4,
        max_steps: 10_000,
    };
    let mut controller = SimulationController::new(runner);
    assert!(!controller.is_pause_pending());

    assert!(controller.send_command(SimCommand::Start).is_ok());
    assert!(!controller.is_pause_pending());

    assert!(controller.send_command(SimCommand::Pause).is_ok());
    assert!(controller.is_pause_pending());

    controller.wait_until_paused();
    assert!(!controller.is_pause_pending());
    assert_eq!(controller.state(), SimWorkerState::Ready);
}

#[test]
fn test_state_display_and_helpers() {
    assert_eq!(
        format!("{}", SimWorkerState::Uninitialized),
        "Uninitialized"
    );
    assert_eq!(format!("{}", SimWorkerState::Ready), "Ready");
    assert_eq!(format!("{}", SimWorkerState::Running), "Running");
    assert_eq!(format!("{}", SimWorkerState::Stepping), "Stepping");
    assert_eq!(format!("{}", SimWorkerState::Faulted), "Faulted");
    assert_eq!(format!("{}", SimWorkerState::Terminated), "Terminated");

    assert!(SimWorkerState::Ready.is_ready());
    assert!(SimWorkerState::Running.is_running());
    assert!(SimWorkerState::Stepping.is_stepping());
    assert!(SimWorkerState::Faulted.is_faulted());
    assert!(SimWorkerState::Terminated.is_terminated());
}

#[test]
fn test_domain_simulation_tools_initialization() {
    use crate::async_sim::unified::UnifiedSimTool;
    use crate::tabs::fluid_dynamics::lattice_boltzmann::LbmUnified;
    use crate::tabs::neuroscience::hodgkin_huxley::HodgkinHuxleyUnified;
    use crate::tabs::quantum::wave_sim::WaveSimulatorTool;

    let lbm_tool = UnifiedSimTool::<LbmUnified>::new();
    assert_eq!(lbm_tool.state(), SimWorkerState::Ready);

    let hh_tool = UnifiedSimTool::<HodgkinHuxleyUnified>::new();
    assert_eq!(hh_tool.state(), SimWorkerState::Ready);

    let wave_tool = WaveSimulatorTool::new();
    assert_eq!(wave_tool.state(), SimWorkerState::Ready);
}

#[test]
fn test_unified_model_observable_hook_and_voltage_derivatives() {
    use crate::async_sim::unified::UnifiedModel;
    use crate::sensitivity::SensitivityEngine;
    use crate::tabs::fluid_dynamics::lattice_boltzmann::LbmUnified;
    use crate::tabs::neuroscience::hodgkin_huxley::HodgkinHuxleyUnified;
    use std::collections::HashMap;

    let lbm_params = HashMap::from([("viscosity".to_string(), 0.02)]);
    assert!(LbmUnified::evaluate_observable(&lbm_params).is_none());

    // Verify negative conductance perturbations near zero clamp cleanly instead of failing builder
    let zero_perturbation = HashMap::from([
        ("g_na".to_string(), -0.01),
        ("g_k".to_string(), -0.01),
        ("g_l".to_string(), -0.01),
        ("i_ext".to_string(), 10.0),
    ]);
    let clamped_obs = HodgkinHuxleyUnified::evaluate_observable(&zero_perturbation);
    assert!(clamped_obs.is_some());

    let baseline_params = vec![
        ("g_k".to_string(), 36.0),
        ("g_l".to_string(), 0.3),
        ("g_na".to_string(), 120.0),
        ("i_ext".to_string(), 10.0),
    ];
    let mut engine = SensitivityEngine::new();
    let eval_hh = |p_map: &HashMap<String, f64>| {
        HodgkinHuxleyUnified::evaluate_observable(p_map)
            .map(|(_, v)| v)
            .unwrap_or(0.0)
    };
    engine.compute(&baseline_params, eval_hh);
    engine.compute(&baseline_params, eval_hh);

    assert!(engine.baseline_output > 20.0);
    let g_k_deriv = engine.metrics[0].partial_derivative;
    let g_na_deriv = engine.metrics[2].partial_derivative;
    assert!(
        g_k_deriv < 0.0,
        "Potassium conductance should oppose peak depolarization"
    );
    assert!(
        g_na_deriv > 0.0,
        "Sodium conductance should raise peak depolarization"
    );
    assert!(
        (g_na_deriv - 1.0).abs() > 1e-2,
        "Must not equal synthetic slider-sum derivative"
    );
}

#[test]
fn test_unified_sim_tool_panel_gating() {
    use crate::async_sim::unified::UnifiedSimTool;
    use crate::framework::InteractiveTool;
    use crate::tabs::fluid_dynamics::lattice_boltzmann::LbmUnified;
    use crate::tabs::neuroscience::hodgkin_huxley::HodgkinHuxleyUnified;
    use crate::tabs::quantum::wave_sim::WaveSimulatorTool;

    let render_texts = |tool: &mut dyn InteractiveTool| -> Vec<String> {
        let ctx = eframe::egui::Context::default();
        let out = ctx.run(eframe::egui::RawInput::default(), |ctx| tool.show(ctx));
        let shapes: Vec<_> = out.shapes.into_iter().map(|cs| cs.shape).collect();
        let mut texts = Vec::new();
        crate::sensitivity::tests::collect_shape_texts(&shapes, &mut texts);
        texts
    };

    let lbm_texts = render_texts(&mut UnifiedSimTool::<LbmUnified>::new());
    assert!(!lbm_texts
        .iter()
        .any(|t| t.contains("Sensitivity & Perturbation Analysis")));

    let wave_texts = render_texts(&mut WaveSimulatorTool::new());
    assert!(!wave_texts
        .iter()
        .any(|t| t.contains("Sensitivity & Perturbation Analysis")));

    let hh_texts = render_texts(&mut UnifiedSimTool::<HodgkinHuxleyUnified>::new());
    assert!(hh_texts
        .iter()
        .any(|t| t.contains("Sensitivity & Perturbation Analysis")));
    assert!(hh_texts
        .iter()
        .any(|t| t.contains("Baseline Peak Membrane Potential (mV):")));

    let ctx = eframe::egui::Context::default();
    let input = eframe::egui::RawInput {
        screen_rect: Some(eframe::egui::Rect::from_min_size(
            eframe::egui::Pos2::ZERO,
            eframe::egui::vec2(960.0, 640.0),
        )),
        ..Default::default()
    };
    let mut hh_tool = UnifiedSimTool::<HodgkinHuxleyUnified>::new();
    let out1 = ctx.run(input.clone(), |ctx| hh_tool.show(ctx));
    assert!(!hh_tool.sensitivity_engine.is_converged());
    assert!(out1.viewport_output[&eframe::egui::ViewportId::ROOT]
        .repaint_delay
        .is_zero());
    let _ = ctx.run(input.clone(), |ctx| hh_tool.show(ctx));
    assert!(hh_tool.sensitivity_engine.is_converged());
    let out_idle = ctx.run(input, |ctx| hh_tool.show(ctx));
    assert!(!out_idle.viewport_output[&eframe::egui::ViewportId::ROOT]
        .repaint_delay
        .is_zero());
}
