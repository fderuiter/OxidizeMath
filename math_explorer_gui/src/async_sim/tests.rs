use super::*;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

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

    assert!(controller.send_command(SimCommand::Start).is_ok());

    // Trigger panic in background worker
    panic_flag.store(true, Ordering::SeqCst);

    // Allow background thread to hit the step panic
    std::thread::sleep(Duration::from_millis(50));

    // On the next frame update, disconnection must be detected and state set to Faulted
    assert!(controller.update().is_none());
    assert_eq!(controller.state(), SimWorkerState::Faulted);
    assert!(controller.state().is_faulted());
    assert!(!controller.state().is_running());

    // Commands in Faulted state should be rejected
    assert!(matches!(
        controller.send_command(SimCommand::Start),
        Err(SimCommandError::InvalidTransition { .. })
    ));
}

#[test]
fn test_update_returns_snapshot_only_on_new_snapshot() {
    let panic_flag = Arc::new(AtomicBool::new(false));
    let runner = TestRunner::new(panic_flag);
    let mut controller = SimulationController::new(runner);

    // Initial check when paused: no snapshots emitted yet
    assert!(controller.update().is_none());

    // Start simulation and wait briefly for a step snapshot
    assert!(controller.send_command(SimCommand::Start).is_ok());
    std::thread::sleep(Duration::from_millis(30));

    let first_update = controller.update();
    assert!(first_update.is_some());

    // Pause simulation
    assert!(controller.send_command(SimCommand::Pause).is_ok());
    // Drain any remaining messages
    let _ = controller.update();

    // Subsequent updates while paused should return None (no new snapshots)
    assert!(controller.update().is_none());
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
    assert!(SimWorkerState::Faulted.is_faulted());
    assert!(SimWorkerState::Terminated.is_terminated());
}

#[test]
fn test_domain_simulation_tools_initialization() {
    use crate::async_sim::unified::UnifiedSimTool;
    use crate::tabs::fluid_dynamics::lattice_boltzmann::LbmUnified;
    use crate::tabs::neuroscience::hodgkin_huxley::HodgkinHuxleyUnified;
    use crate::tabs::quantum::wave_sim::WaveUnified;

    let lbm_tool = UnifiedSimTool::<LbmUnified>::new();
    assert_eq!(lbm_tool.state(), SimWorkerState::Ready);

    let hh_tool = UnifiedSimTool::<HodgkinHuxleyUnified>::new();
    assert_eq!(hh_tool.state(), SimWorkerState::Ready);

    let wave_tool = UnifiedSimTool::<WaveUnified>::new();
    assert_eq!(wave_tool.state(), SimWorkerState::Ready);
}
