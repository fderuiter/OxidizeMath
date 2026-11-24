#![cfg_attr(any(), verified(opt_out = "tests"))]
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
            sonification: crate::audio::SonificationFrame::silent(),
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
            sonification: crate::audio::SonificationFrame::silent(),
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
            sonification: crate::audio::SonificationFrame::silent(),
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
    for _ in 0..400 {
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
    for _ in 0..400 {
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

    for _ in 0..400 {
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

#[cfg(not(target_arch = "wasm32"))]
struct MockRingRunner {
    ring: crate::framework::TimelineRingBuffer<usize>,
    next_val: usize,
}

#[cfg(not(target_arch = "wasm32"))]
impl SimulationRunner for MockRingRunner {
    fn process_command(&mut self, cmd: SimCommand) {
        if matches!(cmd, SimCommand::Reset) {
            self.ring.clear();
            self.next_val = 0;
            self.ring.push(0);
        }
    }

    fn step(&mut self) {
        self.next_val += 1;
        self.ring.push(self.next_val);
    }

    fn get_snapshot(&self) -> StateSnapshot {
        StateSnapshot {
            width: 1,
            height: 1,
            pixels: Arc::new(std::sync::RwLock::new(vec![eframe::egui::Color32::WHITE])),
            custom_data: vec![self.next_val as f64, self.ring.len() as f64],
            structured_data: None,
            sonification: crate::audio::SonificationFrame::silent(),
        }
    }

    fn get_steps_per_frame(&self) -> usize {
        1
    }

    fn checkpoint_stride(&self) -> Option<usize> {
        Some(1)
    }

    fn current_step(&self) -> usize {
        self.ring.cursor()
    }

    fn capture_active_params(&self) -> Option<CheckpointParams> {
        Some(CheckpointParams::scalars(&[1.0]))
    }

    fn capture_checkpoint(&self) -> Option<TimeTravelCheckpoint> {
        Some(TimeTravelCheckpoint {
            step: self.ring.cursor(),
            rng: oxidize_core::rng::OxidizeRng::default(),
            params: CheckpointParams::scalars(&[1.0]),
            state: CheckpointPayload::scalars(&[self.next_val as f64]),
        })
    }

    fn restore_checkpoint(&mut self, checkpoint: &TimeTravelCheckpoint) {
        if let Some(s) = checkpoint.state.as_scalars() {
            if let Some(&v) = s.first() {
                self.next_val = v as usize;
            }
        }
        let _ = self.ring.seek(checkpoint.step);
    }
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_step_back_and_seek_frame_pause_restore_and_truncate_on_resume() {
    let mut ring = crate::framework::TimelineRingBuffer::with_capacity(32, 0);
    ring.push(0);
    let mut controller = SimulationController::new(MockRingRunner { ring, next_val: 0 });

    assert!(controller.send_command(SimCommand::Step(5)).is_ok());
    for _ in 0..400 {
        thread::sleep(Duration::from_millis(5));
        if controller.update().is_some() && controller.state() == SimWorkerState::Ready {
            break;
        }
    }
    assert_eq!(
        controller.latest_snapshot().unwrap().custom_data,
        vec![5.0, 6.0]
    );

    assert!(controller.send_command(SimCommand::StepBack(2)).is_ok());
    for _ in 0..400 {
        thread::sleep(Duration::from_millis(5));
        if controller.update().is_some() && controller.state() == SimWorkerState::Ready {
            break;
        }
    }
    assert_eq!(
        controller.latest_snapshot().unwrap().custom_data,
        vec![3.0, 6.0]
    );

    assert!(controller.send_command(SimCommand::SeekStep(1)).is_ok());
    for _ in 0..400 {
        thread::sleep(Duration::from_millis(5));
        if controller.update().is_some() && controller.state() == SimWorkerState::Ready {
            break;
        }
    }
    assert_eq!(
        controller.latest_snapshot().unwrap().custom_data,
        vec![1.0, 6.0]
    );

    // Stepping forward truncates frames 2..=5 and appends frame 2 -> len becomes 3.
    assert!(controller.send_command(SimCommand::Step(1)).is_ok());
    for _ in 0..400 {
        thread::sleep(Duration::from_millis(5));
        if controller.update().is_some() && controller.state() == SimWorkerState::Ready {
            break;
        }
    }
    assert_eq!(
        controller.latest_snapshot().unwrap().custom_data,
        vec![2.0, 3.0]
    );
}

static COUNTING_INIT_CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

struct CountingModel {
    step: usize,
}

impl crate::framework::recovery::FiniteStateGuard for CountingModel {}

impl scientific_metadata::theory::TheoryDescribable for CountingModel {
    fn theory_description(&self) -> String {
        "Counting test model".to_string()
    }
    fn phonetic_description(&self) -> String {
        "Counting test model".to_string()
    }
    fn theory_citation(&self) -> String {
        "Test (2026)".to_string()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
}

impl crate::async_sim::unified::UnifiedModel for CountingModel {
    fn new(_params: &std::collections::HashMap<String, f64>) -> Self {
        COUNTING_INIT_CALLS.fetch_add(1, Ordering::SeqCst);
        Self { step: 0 }
    }
    fn step(&mut self, _params: &std::collections::HashMap<String, f64>) {
        self.step += 1;
    }
    fn get_snapshot(&self) -> StateSnapshot {
        StateSnapshot {
            width: 0,
            height: 0,
            pixels: Arc::new(std::sync::RwLock::new(Vec::new())),
            custom_data: vec![0.0, 0.0, self.step as f64, (self.step * 2) as f64],
            structured_data: None,
            sonification: crate::audio::SonificationFrame::silent(),
        }
    }
    fn parameters(
    ) -> std::collections::HashMap<String, scientific_metadata::theory::ParameterConstraint> {
        std::collections::HashMap::from([(
            "rate".to_string(),
            scientific_metadata::theory::ParameterConstraint {
                min: 0.1,
                max: 2.0,
                step: 0.1,
            },
        )])
    }
    fn name() -> &'static str {
        "Counting Unified Model"
    }
}

#[test]
fn test_unified_sim_tool_persists_step_model_across_gif_frames_and_clears_on_reset() {
    use crate::async_sim::unified::UnifiedSimTool;
    use crate::framework::InteractiveTool;

    COUNTING_INIT_CALLS.store(0, Ordering::SeqCst);
    let mut tool = UnifiedSimTool::<CountingModel>::new();
    let baseline_inits = COUNTING_INIT_CALLS.load(Ordering::SeqCst);

    let frames = tool.sample_animation_frames(4);
    assert_eq!(frames.len(), 4);
    // Only one lazy step_model construction across all 4 GIF frames
    assert_eq!(
        COUNTING_INIT_CALLS.load(Ordering::SeqCst),
        baseline_inits + 1
    );
    assert_eq!(frames[0][0].1.last().copied(), Some([5.0, 10.0]));
    assert_eq!(frames[3][0].1.last().copied(), Some([20.0, 40.0]));
    assert_eq!(tool.export_datasets(), frames[3]);

    // Reset via try_load_state clears step_model so subsequent sampling starts fresh
    tool.try_load_state(r#"{"rate": 1.0}"#)
        .expect("restore valid state");
    assert!(tool.step_model.is_none());
    let post_reset = tool.sample_animation_frames(2);
    assert_eq!(post_reset[0][0].1.last().copied(), Some([5.0, 10.0]));
    assert_eq!(post_reset[1][0].1.last().copied(), Some([10.0, 20.0]));
}

#[cfg(feature = "physics")]
#[test]
fn test_unified_snapshot_datasets_extracts_structured_wave_data() {
    use crate::async_sim::unified::UnifiedModel;
    use crate::tabs::quantum::wave_sim::{WaveRunner, WaveSimConfig};

    let runner = WaveRunner::new(Arc::new(std::sync::RwLock::new(WaveSimConfig::default())));
    let snap = runner.get_snapshot();
    let ds = CountingModel::snapshot_datasets(&snap);
    assert_eq!(ds.len(), 3);
    assert!(ds[0].0.starts_with("Probability |ψ(x, t="));
}
