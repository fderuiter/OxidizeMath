#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;

#[test]
fn test_mri_tab_initial_state() {
    let tab = MriTab::default();
    assert_eq!(tab.worker.state(), SimWorkerState::Ready);
    assert_eq!(tab.time_scale, 1.0);
    assert_eq!(tab.current_time, 0.0);
}

#[test]
fn test_mri_worker_simulation_step() {
    let mut tab = MriTab::default();
    let _ = tab.worker.send_command(SimCommand::Start);
    let mut stepped = false;
    for _ in 0..50 {
        std::thread::sleep(std::time::Duration::from_millis(20));
        if let Some(snap) = tab.worker.update() {
            if snap.custom_data.first().copied().unwrap_or(0.0) > 0.0 {
                stepped = true;
                break;
            }
        }
    }
    assert!(stepped);
}

#[test]
fn test_mri_fault_halts_repaints_and_recovers() {
    let ctx = egui::Context::default();
    let mut tab = MriTab::default();
    let mut frame = eframe::Frame::_new_kittest();

    // Ready state: repaints halted
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        tab.show(ctx, &mut frame);
    });
    let output_ready = ctx.run(egui::RawInput::default(), |ctx| {
        tab.show(ctx, &mut frame);
    });
    assert_eq!(tab.worker.state(), SimWorkerState::Ready);
    let delay_ready = output_ready
        .viewport_output
        .get(&egui::ViewportId::ROOT)
        .unwrap()
        .repaint_delay;
    assert_ne!(delay_ready, std::time::Duration::ZERO);

    // Start running: repaints requested
    let _ = tab.worker.send_command(SimCommand::Start);
    let output_running = ctx.run(egui::RawInput::default(), |ctx| {
        tab.show(ctx, &mut frame);
    });
    assert_eq!(tab.worker.state(), SimWorkerState::Running);
    let delay_running = output_running
        .viewport_output
        .get(&egui::ViewportId::ROOT)
        .unwrap()
        .repaint_delay;
    assert_eq!(delay_running, std::time::Duration::ZERO);

    // Pause running worker
    let _ = tab.worker.send_command(SimCommand::Pause);
    while tab.worker.update().is_some() {}

    // Simulate fault
    tab.worker.set_state_for_test(SimWorkerState::Faulted);
    assert_eq!(tab.worker.state(), SimWorkerState::Faulted);

    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        tab.show(ctx, &mut frame);
    });
    let output_faulted = ctx.run(egui::RawInput::default(), |ctx| {
        tab.show(ctx, &mut frame);
    });
    let delay_faulted = output_faulted
        .viewport_output
        .get(&egui::ViewportId::ROOT)
        .unwrap()
        .repaint_delay;
    assert_ne!(delay_faulted, std::time::Duration::ZERO);

    // Retry restores Ready
    tab.retry();
    assert_eq!(tab.worker.state(), SimWorkerState::Ready);
}

#[test]
fn test_mri_save_and_load_state_compatibility() {
    let mut tab = MriTab::default();
    let _ = tab.worker.send_command(SimCommand::Start);

    let saved = tab.save_state().expect("save_state should succeed");
    assert!(saved.contains("\"state\":\"Running\""));
    assert!(!saved.contains("\"paused\""));

    let mut loaded = MriTab::default();
    loaded.load_state(&saved);
    assert_eq!(loaded.worker.state(), SimWorkerState::Running);

    // Legacy paused
    let legacy_paused = r#"{"paused":true,"time_scale":2.5}"#;
    loaded.load_state(legacy_paused);
    assert_eq!(loaded.worker.state(), SimWorkerState::Ready);
    assert_eq!(loaded.time_scale, 2.5);

    // Legacy running
    let legacy_running = r#"{"paused":false,"time_scale":3.0}"#;
    loaded.load_state(legacy_running);
    assert_eq!(loaded.worker.state(), SimWorkerState::Running);
    assert_eq!(loaded.time_scale, 3.0);
}

#[test]
fn test_mri_failure_state_screenshot() {
    let mut tab = MriTab::default();
    tab.worker.set_state_for_test(SimWorkerState::Faulted);
    let datasets = vec![
        ("Mx (Transverse)".to_string(), vec![[0.0, 1.0], [1.0, 0.0]]),
        ("Worker Fault Recovery".to_string(), vec![[0.0, 0.0], [1.0, 1.0]]),
    ];
    let options = egui_plot::export::ExportOptions::default();
    if let Ok(png_data) = egui_plot::export::export_custom_png(&datasets, &options) {
        let _ = std::fs::write("/tmp/simulation_worker_fault_recovery.png", png_data);
    }
}

#[test]
fn test_mri_poisoned_lock_recovery() {
    let mut tab = MriTab::default();

    // Intentionally poison tab.params RwLock
    let params_clone = Arc::clone(&tab.params);
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _guard = params_clone.write().unwrap();
        panic!("simulated worker thread fault");
    }));
    assert!(tab.params.is_poisoned());

    // Verify parameter write calls survive poisoned lock
    tab.simulator.t1 = 2.0;
    tab.params.write().unwrap_or_else(|e| e.into_inner()).t1 = tab.simulator.t1;
    tab.simulator.t2 = 0.5;
    tab.params.write().unwrap_or_else(|e| e.into_inner()).t2 = tab.simulator.t2;
    tab.simulator.b_field = Vector3::new(0.05, 0.02, 1.2);
    tab.params.write().unwrap_or_else(|e| e.into_inner()).b_field = tab.simulator.b_field;
    tab.time_scale = 4.0;
    tab.params.write().unwrap_or_else(|e| e.into_inner()).time_scale = tab.time_scale;

    {
        let read_params = tab.params.read().unwrap_or_else(|e| e.into_inner());
        assert_eq!(read_params.t1, 2.0);
        assert_eq!(read_params.t2, 0.5);
        assert_eq!(read_params.b_field, Vector3::new(0.05, 0.02, 1.2));
        assert_eq!(read_params.time_scale, 4.0);
    }

    // Verify tab retry / restart_worker recovers latest params from poisoned lock
    tab.retry();
    assert_eq!(tab.worker.state(), SimWorkerState::Ready);
    assert_eq!(tab.simulator.t1, 2.0);
    assert_eq!(tab.simulator.t2, 0.5);
    assert_eq!(tab.simulator.b_field, Vector3::new(0.05, 0.02, 1.2));

    // Verify MriRunner::step executes without panicking when lock is poisoned
    let runner_simulator = BlochSimulator::new(Vector3::new(1.0, 0.0, 0.0), 1.0);
    let mut runner = MriRunner {
        simulator: runner_simulator,
        solver: RungeKutta4::new(&Vector3::new(1.0, 0.0, 0.0)),
        params: Arc::clone(&tab.params),
        current_time: 0.0,
        history: VecDeque::new(),
        max_history: 100,
    };
    runner.step();
    assert!(runner.current_time > 0.0);
    assert!(!runner.history.is_empty());

    // Verify show() layout and rendering does not crash with poisoned lock
    let ctx = egui::Context::default();
    let mut frame = eframe::Frame::_new_kittest();
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        tab.show(ctx, &mut frame);
    });

    // Verify load_state recovers poisoned lock
    let saved = r#"{"state":"Running","time_scale":3.5}"#;
    tab.load_state(saved);
    assert_eq!(tab.time_scale, 3.5);
    let read_params = tab.params.read().unwrap_or_else(|e| e.into_inner());
    assert_eq!(read_params.time_scale, 3.5);
}

#[test]
fn test_mri_plot_buffers_reuse_and_borrowed_slice() {
    let mut tab = MriTab::default();

    // Populate history and update buffers
    tab.history.push_back((0.0, Vector3::new(1.0, 0.0, 0.0)));
    tab.history.push_back((0.1, Vector3::new(0.8, 0.2, 0.5)));
    tab.update_plot_buffers();

    assert_eq!(tab.mx_points.len(), 2);
    assert_eq!(tab.my_points.len(), 2);
    assert_eq!(tab.mz_points.len(), 2);

    // Verify borrowed slice conversions for egui_plot
    {
        let mx_slice: &[PlotPoint] = &tab.mx_points[..];
        let plot_points_x: egui_plot::PlotPoints<'_> = mx_slice.into();
        assert!(matches!(plot_points_x, egui_plot::PlotPoints::Borrowed(_)));
        assert_eq!(plot_points_x.points().len(), 2);
        assert_eq!(plot_points_x.points()[0], PlotPoint::new(0.0, 1.0));
        assert_eq!(plot_points_x.points()[1], PlotPoint::new(0.1, 0.8));

        let my_slice: &[PlotPoint] = &tab.my_points[..];
        let plot_points_y: egui_plot::PlotPoints<'_> = my_slice.into();
        assert!(matches!(plot_points_y, egui_plot::PlotPoints::Borrowed(_)));
        assert_eq!(plot_points_y.points()[1], PlotPoint::new(0.1, 0.2));

        let mz_slice: &[PlotPoint] = &tab.mz_points[..];
        let plot_points_z: egui_plot::PlotPoints<'_> = mz_slice.into();
        assert!(matches!(plot_points_z, egui_plot::PlotPoints::Borrowed(_)));
        assert_eq!(plot_points_z.points()[1], PlotPoint::new(0.1, 0.5));
    }

    // Verify capacity reuse when repopulating
    let cap_x = tab.mx_points.capacity();
    let cap_y = tab.my_points.capacity();
    let cap_z = tab.mz_points.capacity();

    tab.history.pop_back();
    tab.update_plot_buffers();
    assert_eq!(tab.mx_points.len(), 1);
    assert!(tab.mx_points.capacity() >= cap_x);
    assert!(tab.my_points.capacity() >= cap_y);
    assert!(tab.mz_points.capacity() >= cap_z);

    // Verify reset clears buffers while retaining allocated capacity
    tab.reset();
    assert!(tab.mx_points.is_empty());
    assert!(tab.my_points.is_empty());
    assert!(tab.mz_points.is_empty());
    assert!(tab.history.is_empty());
    assert!(tab.mx_points.capacity() >= cap_x);
}

#[test]
fn test_mri_trajectory_plot_screenshot() {
    let datasets = vec![
        ("Mx (Transverse)".to_string(), vec![[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]]),
        ("My (Transverse)".to_string(), vec![[0.0, 0.0], [0.5, 0.7], [1.0, 0.0]]),
        ("Mz (Longitudinal)".to_string(), vec![[0.0, 0.0], [0.5, 0.3], [1.0, 0.8]]),
    ];
    let options = egui_plot::export::ExportOptions::default();
    if let Ok(png_data) = egui_plot::export::export_custom_png(&datasets, &options) {
        let _ = std::fs::write("/tmp/mri_trajectory_plot.png", png_data);
    }
}
