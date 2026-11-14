use super::*;

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
