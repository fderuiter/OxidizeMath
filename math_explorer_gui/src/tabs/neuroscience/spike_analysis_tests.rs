#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;

#[test]
fn test_spike_analysis_simulation_and_borrowed_slices() {
    let mut tool = SpikeAnalysisTool::default();
    assert!(tool.history_points.is_empty());
    assert!(tool.spike_plot_points.is_empty());

    tool.step_once();
    assert!(!tool.is_running);
    assert_eq!(tool.history_points.len(), 1);
    assert!((tool.time - 0.01).abs() < 1e-9);

    tool.simulation_speed = 50;
    tool.input_current = 20.0;
    for _ in 0..100 {
        tool.step_simulation();
    }

    assert!(!tool.history_points.is_empty());

    {
        let history_slice: &[PlotPoint] = &tool.history_points[..];
        let history_points: egui_plot::PlotPoints<'_> = history_slice.into();
        assert!(matches!(history_points, egui_plot::PlotPoints::Borrowed(_)));
        assert_eq!(history_points.points().len(), tool.history_points.len());

        let spike_slice: &[PlotPoint] = &tool.spike_plot_points[..];
        let spike_plot_points: egui_plot::PlotPoints<'_> = spike_slice.into();
        assert!(matches!(
            spike_plot_points,
            egui_plot::PlotPoints::Borrowed(_)
        ));
        assert_eq!(
            spike_plot_points.points().len(),
            tool.spike_plot_points.len()
        );
    }

    let old_threshold = tool.spike_threshold;
    tool.record_spike(12.34);
    assert!(!tool.spike_plot_points.is_empty());
    assert_eq!(tool.spike_plot_points.last().unwrap().y, old_threshold);

    let new_threshold = 15.0;
    tool.spike_threshold = new_threshold;
    for pt in &mut tool.spike_plot_points {
        pt.y = tool.spike_threshold;
    }
    assert_eq!(tool.spike_plot_points.last().unwrap().y, new_threshold);

    tool.reset();
    assert!(tool.history_points.is_empty());
    assert!(tool.spike_plot_points.is_empty());
    assert!(tool.spike_times.is_empty());
    assert_eq!(tool.time, 0.0);
}
