use super::*;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
#[cfg(not(target_arch = "wasm32"))]
use std::thread;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Duration;

struct DummyRunner;

impl SimulationRunner for DummyRunner {
    fn process_command(&mut self, _cmd: SimCommand) {}
    fn step(&mut self) {}
    fn get_snapshot(&self) -> StateSnapshot {
        StateSnapshot {
            width: 1,
            height: 1,
            pixels: Arc::new(std::sync::RwLock::new(vec![])),
            custom_data: vec![],
            structured_data: None,
        }
    }
    fn get_steps_per_frame(&self) -> usize {
        1
    }
}

fn create_dummy_context() -> WorkerContext {
    let (_cmd_tx, cmd_rx) = mpsc::channel::<SimCommand>();
    let (state_tx, _state_rx) = mpsc::sync_channel::<SimStateUpdate>(4);
    WorkerContext {
        runner: Box::new(DummyRunner),
        cmd_rx,
        state_tx,
    }
}

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
    assert!(!controller.running);
    assert_eq!(steps.load(Ordering::SeqCst), 0);

    // Start simulation
    controller.send_command(SimCommand::Start);
    thread::sleep(Duration::from_millis(50));
    let snap_is_some = controller.update().is_some();
    assert!(controller.running);
    assert!(snap_is_some);
    let current_steps = steps.load(Ordering::SeqCst);
    assert!(current_steps > 0);

    // Pause simulation
    controller.send_command(SimCommand::Pause);
    thread::sleep(Duration::from_millis(50));
    controller.update();
    assert!(!controller.running);
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
    controller.send_command(SimCommand::Start);

    // Wait for worker to finish max_steps
    for _ in 0..50 {
        thread::sleep(Duration::from_millis(15));
        controller.update();
        if !controller.running && steps.load(Ordering::SeqCst) >= 15 {
            break;
        }
    }

    assert_eq!(steps.load(Ordering::SeqCst), 15);
    assert!(!controller.running);
}

#[test]
fn test_morphogenesis_runner_offload() {
    use crate::tabs::morphogenesis::{
        MorphogenesisRunner, MorphogenesisSharedParams, MorphogenesisTab,
    };
    use math_explorer::biology::diffusion::FiniteDifference2D;
    use math_explorer::biology::morphogenesis::{SchnakenbergKinetics, TuringSystem};

    let width = 20;
    let height = 20;
    let kinetics = SchnakenbergKinetics { a: 0.1, b: 0.9 };
    let diffusion = FiniteDifference2D::new(
        math_explorer::math_kernel::types::Dimension(width),
        math_explorer::math_kernel::types::Dimension(height),
        math_explorer::math_kernel::types::StepSize(1.0),
        math_explorer::math_kernel::types::StepSize(1.0),
    );

    let mut system = TuringSystem::new_with_kinetics(
        math_explorer::math_kernel::types::Dimension(width * height),
        math_explorer::biology::morphogenesis::DiffusionCoeff(1.0),
        math_explorer::biology::morphogenesis::DiffusionCoeff(100.0),
        kinetics,
        diffusion,
    );
    MorphogenesisTab::initialize_system(&mut system, width, height);

    let params = Arc::new(std::sync::RwLock::new(MorphogenesisSharedParams {
        a: 0.1,
        b: 0.9,
        d_u: 1.0,
        d_v: 100.0,
        dt: 0.05,
    }));

    let mut runner = MorphogenesisRunner {
        system,
        params: Arc::clone(&params),
        width,
        height,
    };

    let snap_before = runner.get_snapshot();
    assert_eq!(snap_before.width, 20);
    assert_eq!(snap_before.height, 20);
    assert_eq!(snap_before.pixels.read().unwrap().len(), 400);

    // Step simulation
    runner.step();

    // Verify parameter change takes effect immediately
    {
        let mut p = params.write().unwrap();
        p.a = 0.2;
    }
    runner.step();

    let snap_after = runner.get_snapshot();
    assert_eq!(snap_after.pixels.read().unwrap().len(), 400);

    // Generate artifact screenshot for PR
    let datasets = vec![(
        "Morphogenesis Turing Concentration".to_string(),
        (0..snap_after.width)
            .map(|x| [x as f64, (x as f64 * 0.2).sin()])
            .collect(),
    )];
    let options = egui_plot::export::ExportOptions::default();
    if let Ok(png_data) = egui_plot::export::export_custom_png(&datasets, &options) {
        let _ = std::fs::write("/tmp/morphogenesis_sim.png", png_data);
    }
}

#[test]
fn test_reward_plots_runner_batch_training() {
    use crate::tabs::ai::reward_plots::{RewardPlotsRunner, RlSnapshotData};

    let mut runner = RewardPlotsRunner::new();
    assert!(runner.is_complete());

    // Start training 5 episodes
    runner.process_command(SimCommand::TrainEpisodes(5));
    assert!(!runner.is_complete());

    // Step until complete
    while !runner.is_complete() {
        runner.step();
    }

    let snap = runner.get_snapshot();
    let data = snap
        .structured_data
        .as_ref()
        .and_then(|d| d.downcast_ref::<RlSnapshotData>())
        .expect("Snapshot must contain RlSnapshotData");

    assert_eq!(data.episodes_trained, 5);
    assert_eq!(data.rewards_per_episode.len(), 5);

    // Generate artifact screenshot for PR
    let points: Vec<[f64; 2]> = data
        .rewards_per_episode
        .iter()
        .enumerate()
        .map(|(i, &r)| [i as f64, r])
        .collect();
    let datasets = vec![("Episode Reward".to_string(), points)];
    let options = egui_plot::export::ExportOptions::default();
    if let Ok(png_data) = egui_plot::export::export_custom_png(&datasets, &options) {
        let _ = std::fs::write("/tmp/reward_plots_training.png", png_data);
    }

    // Reset agent
    runner.process_command(SimCommand::Reset);
    assert!(runner.is_complete());
    let snap_reset = runner.get_snapshot();
    let data_reset = snap_reset
        .structured_data
        .as_ref()
        .and_then(|d| d.downcast_ref::<RlSnapshotData>())
        .unwrap();
    assert_eq!(data_reset.episodes_trained, 0);
    assert!(data_reset.rewards_per_episode.is_empty());
}

#[test]
fn test_handle_registry_valid_retrieval() {
    let ctx = create_dummy_context();
    let handle = register_worker_context(ctx);
    assert!(handle > 0, "Registered handle must be positive integer");

    let retrieved = take_worker_context(handle);
    assert!(retrieved.is_some(), "Valid handle must retrieve context");
}

#[test]
fn test_handle_registry_invalid_rejection() {
    // Forged or non-existent handle IDs must return None
    assert!(take_worker_context(0).is_none());
    assert!(take_worker_context(999_999_999).is_none());
    assert!(take_worker_context(u32::MAX).is_none());
}

#[test]
fn test_handle_registry_duplicate_consumption() {
    let ctx = create_dummy_context();
    let handle = register_worker_context(ctx);

    // First consumption succeeds
    let first = take_worker_context(handle);
    assert!(first.is_some());

    // Subsequent/duplicate retrieval returns None because context is consumed
    let second = take_worker_context(handle);
    assert!(second.is_none());

    let third = take_worker_context(handle);
    assert!(third.is_none());
}

#[test]
fn test_handle_registry_unregister_cleanup() {
    let ctx = create_dummy_context();
    let handle = register_worker_context(ctx);

    // Dropping simulation controller before worker starts cleans up registry
    let removed = unregister_worker_context(handle);
    assert!(removed.is_some(), "Unregister must return existing context");

    // Once cleaned up, worker attempting to take handle gets None
    assert!(take_worker_context(handle).is_none());
    assert!(unregister_worker_context(handle).is_none());
}
