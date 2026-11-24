#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::presets::OdePreset;
use crate::tabs::analysis::ode::OdeSolverTool;
use crate::tabs::epidemiology::sir::SirTool;
use crate::web_io::PlatformInstant;
use eframe::egui;
use egui_plot::PlotPoint;

fn frame_input() -> egui::RawInput {
    egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1200.0, 800.0),
        )),
        ..Default::default()
    }
}

fn maybe_write_perf_screenshot(output_path: &str) {
    if std::env::var_os("OXIDIZE_GENERATE_PERF_SCREENSHOTS").is_some() {
        let opts = egui_plot::export::ExportOptions {
            width: 1000,
            height: 600,
            ..Default::default()
        };
        let prms = egui_plot::get_last_plot_primitives();
        let img = egui_plot::render_primitives_image(&prms, &opts).expect("render image");
        let png = egui_plot::color32_image_to_png(&img).expect("encode PNG");
        std::fs::write(output_path, png).expect("write verification PNG");
    }
}

fn build_400k_borrowed_sir_tool() -> SirTool {
    let mut tool = SirTool::default();
    tool.s_points = (0..400_002)
        .map(|i| {
            let t = i as f64 * 0.00025;
            PlotPoint::new(t, 1000.0 * (-0.03 * t).exp())
        })
        .collect();
    tool.sync_series_cache();
    tool
}

fn assert_valid_400k_snapshot(framework: &mut SimulationFramework) {
    let snap = framework
        .last_valid_snapshot()
        .expect("snapshot must exist after warmup");
    assert!(
        snap.memory_size() <= MAX_SNAPSHOT_BYTES,
        "Snapshot {} bytes exceeded 2 MiB budget",
        snap.memory_size()
    );
    assert!(
        framework.current_valid_plot_datasets[0].1.len()
            <= crate::framework::MAX_DISPLAY_POLYLINE_POINTS
    );
    assert_eq!(
        framework
            .active_tool
            .as_ref()
            .expect("active SirTool")
            .export_datasets()[0]
            .1
            .len(),
        400_002
    );
    let ((), snap_metrics) =
        verified_engine::engine::VerifiedEngine::run_verified(|| framework.cache_valid_snapshot());
    assert_eq!(snap_metrics.allocations, 0);
    assert_eq!(snap_metrics.allocated_bytes, 0);
}

#[test]
fn test_idle_frames_zero_series_clones_400k_borrowed_points() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    egui_plot::clear_last_plot_datasets();
    egui_plot::clear_last_plot_primitives();

    let mut framework = SimulationFramework::new("epidemiology");
    assert!(framework.select_tool_by_name("SirTool"));
    framework.active_tool = Some(Box::new(build_400k_borrowed_sir_tool()));

    let ctx = egui::Context::default();
    let input = frame_input();
    for _ in 0..4 {
        let _ = ctx.run(input.clone(), |c| framework.show(c, "sir_400k_bench"));
    }
    assert_valid_400k_snapshot(&mut framework);

    let (avg_ms, alloc_count, alloc_bytes) = crate::test_alloc::measure_coord_series_allocs(|| {
        let start = PlatformInstant::now();
        for _ in 0..5 {
            let _ = ctx.run(input.clone(), |c| framework.show(c, "sir_400k_bench"));
        }
        start.elapsed().as_secs_f64() * 1000.0 / 5.0
    });

    let single_series_clone_bytes = 400_002 * std::mem::size_of::<[f64; 2]>();
    println!(
        "[shared_points_perf] 400,002 borrowed points: avg_idle_frame={avg_ms:.3} ms, total_thread_allocs={alloc_count}, total_thread_bytes={alloc_bytes}"
    );
    assert!(
        alloc_bytes < single_series_clone_bytes,
        "Expected 5 idle 400,002-point frames to allocate less than a single series clone ({single_series_clone_bytes} bytes), got {alloc_bytes} bytes across {alloc_count} unfiltered allocations"
    );
    if !cfg!(debug_assertions) {
        assert!(
            avg_ms < 16.67,
            "Idle 400,002-point frame took {avg_ms:.3} ms (>= 16.67 ms)"
        );
    }
    maybe_write_perf_screenshot("/tmp/sir_shared_series_verification.png");
}

fn build_three_200k_ode_comparison_tool() -> OdeSolverTool {
    let mut tool = OdeSolverTool {
        preset: OdePreset::HarmonicOscillator,
        dt: 0.001,
        total_time: 100.0,
        ..Default::default()
    };
    tool.set_comparison_mode(true);
    let exact_len = 200_002;
    tool.comparison_manager.clear();
    for (idx, solver_name) in ["Runge-Kutta 4 (RK4)", "Euler", "Heun (RK2)"]
        .into_iter()
        .enumerate()
    {
        let phase = idx as f64 * 0.01;
        let pts: Vec<[f64; 2]> = (0..exact_len)
            .map(|i| {
                let t = i as f64 * 0.0005;
                [t, (t + phase).cos()]
            })
            .collect();
        tool.comparison_manager
            .add_snapshot(solver_name, "", vec![TraceSeries::new("y(t)", pts)]);
    }
    tool
}

fn assert_ode_3x200k_snapshot(framework: &SimulationFramework) {
    let snap = framework
        .last_valid_snapshot()
        .expect("ODE snapshot must exist");
    assert!(snap.memory_size() <= MAX_SNAPSHOT_BYTES);
    assert_eq!(framework.current_valid_plot_datasets.len(), 3);
    assert!(framework
        .current_valid_plot_datasets
        .iter()
        .all(|(_, pts)| pts.len() <= crate::framework::MAX_DISPLAY_POLYLINE_POINTS));
    let exported = framework
        .active_tool
        .as_ref()
        .expect("active OdeSolverTool")
        .export_datasets();
    assert_eq!(
        exported
            .iter()
            .filter(|(_, pts)| pts.len() == 200_002)
            .count(),
        3
    );
}

#[test]
fn test_idle_frames_zero_series_clones_three_200k_ode_comparison_series() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    egui_plot::clear_last_plot_datasets();
    egui_plot::clear_last_plot_primitives();

    let mut framework = SimulationFramework::new("analysis");
    assert!(framework.select_tool_by_name("OdeSolverTool"));
    let ode = build_three_200k_ode_comparison_tool();
    assert_eq!(ode.comparison_manager().snapshots().len(), 3);
    assert_eq!(
        ode.comparison_manager().snapshots()[0].series[0]
            .points
            .len(),
        200_002
    );
    framework.active_tool = Some(Box::new(ode));

    let ctx = egui::Context::default();
    let input = frame_input();
    for _ in 0..5 {
        let _ = ctx.run(input.clone(), |c| framework.show(c, "ode_3x200k_bench"));
    }

    assert_ode_3x200k_snapshot(&framework);

    let ((), snap_metrics) =
        verified_engine::engine::VerifiedEngine::run_verified(|| framework.cache_valid_snapshot());
    assert_eq!(snap_metrics.allocations, 0);
    assert_eq!(snap_metrics.allocated_bytes, 0);

    let (avg_ms, alloc_count, alloc_bytes) = crate::test_alloc::measure_coord_series_allocs(|| {
        let start = PlatformInstant::now();
        for _ in 0..5 {
            let _ = ctx.run(input.clone(), |c| framework.show(c, "ode_3x200k_bench"));
        }
        start.elapsed().as_secs_f64() * 1000.0 / 5.0
    });

    let unshared_five_frame_series_bytes = 5 * 3 * 200_002 * std::mem::size_of::<[f64; 2]>();
    println!(
        "[shared_points_perf] 3x200,002 ODE comparison: avg_idle_frame={avg_ms:.3} ms, total_thread_allocs={alloc_count}, total_thread_bytes={alloc_bytes}"
    );
    assert!(
        alloc_bytes < unshared_five_frame_series_bytes / 3,
        "Expected 5 idle 3x200,002 ODE frames to stay well below unshared series clone budget ({} bytes), got {alloc_bytes} bytes across {alloc_count} unfiltered allocations",
        unshared_five_frame_series_bytes / 3
    );
    if !cfg!(debug_assertions) {
        assert!(
            avg_ms < 16.67,
            "Idle 3x200,002 ODE frame took {avg_ms:.3} ms (>= 16.67 ms)"
        );
    }
    maybe_write_perf_screenshot("/tmp/ode_shared_comparison_verification.png");
}

#[test]
fn test_shared_datasets_preserve_csv_json_and_divergence_rollback() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    egui_plot::clear_last_plot_datasets();
    egui_plot::clear_last_plot_primitives();

    let mut framework = SimulationFramework::new("analysis");
    assert!(framework.select_tool_by_name("OdeSolverTool"));
    let ctx = egui::Context::default();
    let input = frame_input();
    let _ = ctx.run(input.clone(), |c| framework.show(c, "ode_parity"));

    let ds_frame1 = egui_plot::get_last_plot_datasets();
    let prms_frame1 = egui_plot::get_last_plot_primitives();
    let csv_before = egui_plot::export::format_csv(&ds_frame1);
    let json_before = egui_plot::export::format_json(&ds_frame1);
    let opts = egui_plot::export::ExportOptions::default();
    let svg_before = egui_plot::export::export_custom_svg(&ds_frame1, &opts).expect("svg export");
    let png_before = egui_plot::export::export_custom_png(&ds_frame1, &opts).expect("png export");
    let gif_before =
        egui_plot::export::export_custom_gif(std::slice::from_ref(&ds_frame1), &opts, 10)
            .expect("gif export");
    let prm_svg_before =
        egui_plot::export_primitives::export_primitives_svg(&prms_frame1, &opts).expect("prm svg");

    let _ = ctx.run(input, |c| framework.show(c, "ode_parity"));
    let ds_frame2 = egui_plot::get_last_plot_datasets();
    let prms_frame2 = egui_plot::get_last_plot_primitives();
    assert_eq!(csv_before, egui_plot::export::format_csv(&ds_frame2));
    assert_eq!(json_before, egui_plot::export::format_json(&ds_frame2));
    assert_eq!(
        svg_before,
        egui_plot::export::export_custom_svg(&ds_frame2, &opts).expect("svg export")
    );
    assert_eq!(
        png_before,
        egui_plot::export::export_custom_png(&ds_frame2, &opts).expect("png export")
    );
    assert_eq!(
        gif_before,
        egui_plot::export::export_custom_gif(std::slice::from_ref(&ds_frame2), &opts, 10)
            .expect("gif export")
    );
    assert_eq!(
        prm_svg_before,
        egui_plot::export_primitives::export_primitives_svg(&prms_frame2, &opts).expect("prm svg")
    );

    // Inject NaN dataset and verify immediate detection + Step Back restoration
    egui_plot::clear_last_plot_datasets();
    egui_plot::append_last_plot_datasets(&[("y(t)".to_string(), vec![[0.0, f64::NAN]])]);
    assert!(framework.inspect_runtime_state().is_err());
    assert!(framework.is_diverged());
    assert!(framework.rollback_step_back());
    assert!(!framework.is_diverged());
    assert_eq!(egui_plot::get_last_plot_datasets(), ds_frame2);
}
