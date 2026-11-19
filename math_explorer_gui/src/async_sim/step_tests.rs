use super::*;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
#[cfg(not(target_arch = "wasm32"))]
use std::thread;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Duration;

#[cfg(not(target_arch = "wasm32"))]
struct MockStepRunner {
    step_count: Arc<AtomicUsize>,
    steps_per_frame: usize,
}

#[cfg(not(target_arch = "wasm32"))]
impl SimulationRunner for MockStepRunner {
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
        }
    }

    fn get_steps_per_frame(&self) -> usize {
        self.steps_per_frame
    }
}

struct IdleValidationRunner {
    paused: AtomicBool,
}

impl SimulationRunner for IdleValidationRunner {
    fn process_command(&mut self, cmd: SimCommand) {
        if matches!(cmd, SimCommand::Pause) {
            self.paused.store(true, Ordering::Relaxed);
        }
    }

    fn step(&mut self) {}

    fn get_snapshot(&self) -> StateSnapshot {
        StateSnapshot {
            width: 1,
            height: 1,
            pixels: Arc::new(std::sync::RwLock::new(vec![eframe::egui::Color32::BLACK])),
            custom_data: Vec::new(),
            structured_data: None,
        }
    }

    fn get_steps_per_frame(&self) -> usize {
        1
    }
}

#[cfg(not(target_arch = "wasm32"))]
struct DivergingStepRunner {
    value: f64,
}

#[cfg(not(target_arch = "wasm32"))]
impl SimulationRunner for DivergingStepRunner {
    fn process_command(&mut self, cmd: SimCommand) {
        if matches!(cmd, SimCommand::Reset) {
            self.value = 0.0;
        }
    }

    fn step(&mut self) {
        self.value = f64::NAN;
    }

    fn get_snapshot(&self) -> StateSnapshot {
        StateSnapshot {
            width: 1,
            height: 1,
            pixels: Arc::new(std::sync::RwLock::new(vec![eframe::egui::Color32::BLACK])),
            custom_data: vec![self.value],
            structured_data: None,
        }
    }

    fn get_steps_per_frame(&self) -> usize {
        1
    }

    fn is_finite_state(&self) -> bool {
        self.value.is_finite()
    }
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_step_command_advances_once_and_returns_to_ready() {
    crate::accessibility::clear_test_announcements();

    let steps = Arc::new(AtomicUsize::new(0));
    let runner = MockStepRunner {
        step_count: Arc::clone(&steps),
        steps_per_frame: 5,
    };
    let mut controller = SimulationController::new(runner);
    assert_eq!(controller.state(), SimWorkerState::Ready);

    assert!(controller.send_command(SimCommand::Step(1)).is_ok());
    for _ in 0..40 {
        thread::sleep(Duration::from_millis(10));
        if controller.update().is_some() && controller.state() == SimWorkerState::Ready {
            break;
        }
    }

    assert_eq!(steps.load(Ordering::SeqCst), 1);
    assert_eq!(controller.state(), SimWorkerState::Ready);
    assert_eq!(controller.latest_snapshot().unwrap().custom_data, vec![1.0]);
    assert!(crate::accessibility::all_announcements()
        .iter()
        .any(|(msg, prio)| msg == "Simulation step completed" && prio == "polite"));

    assert!(controller.send_command(SimCommand::Step(3)).is_ok());
    for _ in 0..40 {
        thread::sleep(Duration::from_millis(10));
        if controller.update().is_some() && controller.state() == SimWorkerState::Ready {
            break;
        }
    }
    assert_eq!(steps.load(Ordering::SeqCst), 4);
    assert_eq!(controller.state(), SimWorkerState::Ready);
}

#[test]
fn test_step_command_rejected_in_non_ready_states() {
    let mut controller = SimulationController::new(IdleValidationRunner {
        paused: AtomicBool::new(false),
    });

    for invalid_state in [
        SimWorkerState::Running,
        SimWorkerState::Stepping,
        SimWorkerState::Faulted,
        SimWorkerState::Terminated,
        SimWorkerState::Uninitialized,
    ] {
        controller.set_state_for_test(invalid_state);
        let err = controller.send_command(SimCommand::Step(1));
        assert_eq!(
            err,
            Err(SimCommandError::InvalidTransition {
                current_state: invalid_state,
                command: "Step",
            })
        );
    }
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_step_divergence_transitions_to_faulted_and_emits_diagnostic() {
    crate::accessibility::clear_test_announcements();

    let mut controller = SimulationController::new(DivergingStepRunner { value: 0.0 });
    assert!(controller.send_command(SimCommand::Step(1)).is_ok());

    for _ in 0..40 {
        thread::sleep(Duration::from_millis(10));
        let _ = controller.update();
        if controller.state() == SimWorkerState::Faulted {
            break;
        }
    }

    assert_eq!(controller.state(), SimWorkerState::Faulted);
    assert!(controller.diagnostic_report().is_some());
    assert!(crate::accessibility::all_announcements()
        .iter()
        .any(|(msg, prio)| msg.contains("faulted") && prio == "polite"));
}
