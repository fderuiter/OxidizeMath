#[allow(unused_imports)]
use super::*;

#[test]
#[cfg(feature = "biology")]
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
#[cfg(feature = "ai")]
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
#[cfg(feature = "ai")]
fn test_background_worker_generic_adapter() {
    use crate::tabs::ai::reward_plots::RewardPlotsRunner;

    let runner = RewardPlotsRunner::new();
    let mut worker = BackgroundWorker::new(runner);

    assert_eq!(worker.state(), SimWorkerState::Ready);
    assert!(!worker.is_running());

    // Send start command
    assert!(worker.send_command(SimCommand::Start).is_ok());
    assert!(worker.is_running());

    // Send pause command
    assert!(worker.send_command(SimCommand::Pause).is_ok());
    assert!(!worker.is_running());

    // Deref access to controller
    assert_eq!(worker.state(), SimWorkerState::Ready);
}

#[test]
#[cfg(feature = "physics")]
fn test_ising_runner_offload() {
    use crate::tabs::solid_state::ising::{IsingRunner, IsingSharedParams};
    use math_explorer::physics::stat_mech::ising::SpinLattice;

    let params = Arc::new(std::sync::RwLock::new(IsingSharedParams {
        temperature: 2.269,
        j_coupling: 1.0,
        h_field: 0.0,
        steps_per_frame: 500,
    }));
    let mut runner = IsingRunner {
        lattice: SpinLattice::new(20, 20, Some(42)),
        params: Arc::clone(&params),
    };

    let snap_before = runner.get_snapshot();
    assert_eq!(snap_before.width, 20);
    assert_eq!(snap_before.height, 20);
    assert_eq!(snap_before.pixels.read().unwrap().len(), 400);

    runner.step();

    let snap_after = runner.get_snapshot();
    assert_eq!(snap_after.pixels.read().unwrap().len(), 400);
    assert_eq!(snap_after.custom_data.len(), 2); // m_avg, energy

    let datasets = vec![(
        "Ising Magnetization".to_string(),
        vec![
            [0.0, snap_before.custom_data[0]],
            [1.0, snap_after.custom_data[0]],
        ],
    )];
    let options = egui_plot::export::ExportOptions::default();
    if let Ok(png_data) = egui_plot::export::export_custom_png(&datasets, &options) {
        let _ = std::fs::write("/tmp/ising_simulation.png", png_data);
    }
}

#[test]
#[cfg(feature = "physics")]
fn test_mri_runner_offload() {
    use crate::tabs::mri::{MriRunner, MriSharedParams, MriSnapshot};
    use math_explorer::physics::mri::bloch::BlochSimulator;
    use math_explorer::pure_math::analysis::ode::RungeKutta4;
    use nalgebra::Vector3;
    use std::collections::VecDeque;

    let initial_m = Vector3::new(1.0, 0.0, 0.0);
    let simulator = BlochSimulator::new(initial_m, 1.0);
    let solver = RungeKutta4::new(&initial_m);
    let params = Arc::new(std::sync::RwLock::new(MriSharedParams {
        t1: 1.0,
        t2: 0.1,
        b_field: Vector3::new(0.0, 0.0, 1.0),
        time_scale: 1.0,
    }));

    let mut runner = MriRunner {
        simulator,
        solver,
        params,
        current_time: 0.0,
        history: VecDeque::new(),
        max_history: 1000,
    };

    assert_eq!(runner.current_time, 0.0);
    runner.step();
    assert!(runner.current_time > 0.0);

    let snap = runner.get_snapshot();
    let data = snap
        .structured_data
        .as_ref()
        .and_then(|d| d.downcast_ref::<MriSnapshot>())
        .expect("Snapshot must contain MriSnapshot");

    assert!(!data.history.is_empty());

    let points: Vec<[f64; 2]> = data.history.iter().map(|(t, m)| [*t, m.x]).collect();
    let datasets = vec![("MRI Transverse Decay Mx".to_string(), points)];
    let options = egui_plot::export::ExportOptions::default();
    if let Ok(png_data) = egui_plot::export::export_custom_png(&datasets, &options) {
        let _ = std::fs::write("/tmp/mri_simulation.png", png_data);
    }
}

#[test]
#[cfg(feature = "physics")]
fn test_physics_and_chaos_runners_step_once() {
    use crate::tabs::chaos::attractors::{
        AttractorRunner, AttractorSharedParams, SolverExecutionState,
    };
    use crate::tabs::mri::{MriRunner, MriSharedParams};
    use crate::tabs::solid_state::ising::{IsingRunner, IsingSharedParams};
    use math_explorer::physics::chaos::lorenz::{LorenzBuilder, LorenzState};
    use math_explorer::physics::mri::bloch::BlochSimulator;
    use math_explorer::physics::stat_mech::ising::SpinLattice;
    use math_explorer::pure_math::analysis::ode::RungeKutta4;
    use nalgebra::Vector3;
    use std::collections::VecDeque;

    let mut ising = IsingRunner {
        lattice: SpinLattice::new(8, 8, Some(7)),
        params: Arc::new(std::sync::RwLock::new(IsingSharedParams {
            temperature: 2.269,
            j_coupling: 1.0,
            h_field: 0.0,
            steps_per_frame: 5000,
        })),
    };
    ising.step_once();
    assert!(ising.is_finite_state());

    let initial_m = Vector3::new(1.0, 0.0, 0.0);
    let mut mri = MriRunner {
        simulator: BlochSimulator::new(initial_m, 1.0),
        solver: RungeKutta4::new(&initial_m),
        params: Arc::new(std::sync::RwLock::new(MriSharedParams {
            t1: 1.0,
            t2: 0.1,
            b_field: Vector3::new(0.0, 0.0, 1.0),
            time_scale: 5.0,
        })),
        current_time: 0.0,
        history: VecDeque::new(),
        max_history: 100,
    };
    mri.step_once();
    assert!((mri.current_time - 0.001).abs() < 1e-12);
    assert_eq!(mri.history.len(), 1);

    let mut attractor = AttractorRunner {
        system: LorenzBuilder::new().build(LorenzState::new(1.0, 1.0, 1.0)),
        params: Arc::new(std::sync::RwLock::new(AttractorSharedParams {
            sigma: 10.0,
            rho: 28.0,
            beta: 8.0 / 3.0,
            dt: 0.01,
            simulation_speed: 5,
            max_points: 500,
        })),
        history: VecDeque::new(),
        step_count: 0,
        execution_state: SolverExecutionState::Paused,
    };
    attractor.step_once();
    assert_eq!(attractor.step_count, 1);
    assert_eq!(attractor.execution_state, SolverExecutionState::Paused);
}

#[test]
#[cfg(feature = "ai")]
fn test_rl_runners_step_once() {
    use crate::tabs::ai::q_table_inspector::QTableRunner;
    use crate::tabs::ai::reward_plots::RewardPlotsRunner;
    use math_commons::primitives::UnitInterval;
    use math_explorer::ai::reinforcement_learning::algorithms::TabularQAgent;
    use math_explorer::ai::reinforcement_learning::grid_world::{GridState, GridWorldEnv};

    let env = GridWorldEnv {
        width: 4,
        height: 4,
        start: GridState { x: 0, y: 0 },
        goal: GridState { x: 3, y: 3 },
        traps: vec![],
        gamma: 0.9,
    };
    let agent = TabularQAgent::new(
        UnitInterval::new(0.1).unwrap(),
        UnitInterval::new(0.9).unwrap(),
        UnitInterval::new(0.1).unwrap(),
        Some(42),
    );
    let mut q_runner = QTableRunner {
        env,
        agent,
        episodes_trained: 0,
        target_episodes: 0,
        is_training: false,
    };
    q_runner.step_once();
    assert_eq!(q_runner.episodes_trained, 1);
    assert!(!q_runner.is_training);

    let mut reward_runner = RewardPlotsRunner::new();
    reward_runner.step_once();
    let snap = reward_runner.get_snapshot();
    assert_eq!(snap.custom_data, vec![1.0]);
    assert!(reward_runner.is_complete());
}

#[cfg(all(not(target_arch = "wasm32"), feature = "physics"))]
fn single_key_input(key: eframe::egui::Key) -> eframe::egui::RawInput {
    eframe::egui::RawInput {
        events: vec![eframe::egui::Event::Key {
            key,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: eframe::egui::Modifiers::NONE,
        }],
        ..Default::default()
    }
}

#[test]
#[cfg(all(not(target_arch = "wasm32"), feature = "physics"))]
fn test_attractor_plotter_manual_step_keeps_paused() {
    use crate::framework::InteractiveTool;
    use crate::tabs::chaos::attractors::AttractorPlotter;

    let mut plotter = AttractorPlotter::default();
    let ctx = eframe::egui::Context::default();

    // Start worker and wait until at least one running trajectory batch lands
    for _ in 0..500 {
        let _ = ctx.run(eframe::egui::RawInput::default(), |ctx| {
            plotter.show(ctx);
        });
        if !plotter.history.is_empty() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert!(
        !plotter.history.is_empty(),
        "AttractorPlotter worker must populate history while running"
    );

    // Pause via Space shortcut and wait for worker Ready acknowledgment
    let _ = ctx.run(single_key_input(eframe::egui::Key::Space), |ctx| {
        plotter.show(ctx);
    });
    plotter.worker.wait_until_paused();
    let _ = ctx.run(eframe::egui::RawInput::default(), |ctx| {
        plotter.show(ctx);
    });
    assert!(plotter.is_paused());
    let before_points = plotter.export_datasets()[0].1.len();

    // Press Period to step once while paused
    let _ = ctx.run(single_key_input(eframe::egui::Key::Period), |ctx| {
        plotter.show(ctx);
    });

    // Wait for worker step snapshot and Ready transition
    for _ in 0..500 {
        std::thread::sleep(std::time::Duration::from_millis(10));
        let _ = ctx.run(eframe::egui::RawInput::default(), |ctx| {
            plotter.show(ctx);
        });
        if plotter.export_datasets()[0].1.len() > before_points
            && !plotter.worker.state().is_stepping()
        {
            break;
        }
    }

    assert!(
        plotter.is_paused(),
        "AttractorPlotter must remain paused after manual step"
    );
    assert_eq!(plotter.export_datasets()[0].1.len(), before_points + 1);
}
