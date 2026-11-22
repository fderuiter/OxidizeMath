//! Unit and WASM contract tests for [`super::ParrondoParadoxTool`].

use super::{KellyEdgeBadge, ParrondoParadoxTool, ParrondoViewMode};
use crate::framework::InteractiveTool;
use egui_plot::PlotPoints;

#[test]
fn test_canonical_parrondo_tool_kelly_badges_and_gains() {
    let tool = ParrondoParadoxTool::default();
    assert_eq!(tool.name(), "Parrondo's Paradox");
    assert_eq!(tool.domain(), "financial_math");
    assert!(tool.error_msg.is_none());

    let report = tool.report.as_ref().expect("canonical report must solve");
    assert!(report.is_paradox());
    assert!((report.gain_a - (-0.0100)).abs() < 1e-4);
    assert!((report.gain_b - (-0.0087)).abs() < 1e-4);
    assert!((report.gain_mix - 0.0157).abs() < 1e-4);
    assert!((report.gain_schedule - 0.0574).abs() < 1e-4);

    assert!(tool.kelly_a.is_no_edge());
    assert!(tool.kelly_b.is_no_edge());
    assert!(matches!(
        tool.kelly_mix,
        KellyEdgeBadge::PositiveEdge {
            optimal_fraction,
            ..
        } if optimal_fraction > 0.015
    ));
    assert!(matches!(
        tool.kelly_schedule,
        KellyEdgeBadge::PositiveEdge {
            optimal_fraction,
            ..
        } if optimal_fraction > 0.057
    ));
}

#[test]
fn test_borrowed_plot_slices_and_deterministic_seed() {
    let mut tool_a = ParrondoParadoxTool::default();
    let mut tool_b = ParrondoParadoxTool::default();
    {
        let slice: &[egui_plot::PlotPoint] = &tool_a.traj_schedule[..];
        let plot_pts: PlotPoints<'_> = slice.into();
        assert!(matches!(plot_pts, PlotPoints::Borrowed(_)));
    }
    assert_eq!(tool_a.traj_schedule, tool_b.traj_schedule);

    tool_b.seed = 999;
    tool_b.recompute_ensembles_only();
    assert_ne!(tool_a.traj_schedule, tool_b.traj_schedule);

    tool_a.seed = 999;
    tool_a.recompute_ensembles_only();
    assert_eq!(tool_a.traj_schedule, tool_b.traj_schedule);
}

#[test]
fn test_schedule_validation_and_state_roundtrip_bounds() {
    let mut tool = ParrondoParadoxTool::default();
    assert!(tool.apply_schedule_text("aabb"));
    assert_eq!(tool.active_schedule.as_string(), "AABB");
    assert!(!tool.apply_schedule_text("ABX"));
    assert!(tool.schedule_error.is_some());
    assert!(!tool.apply_schedule_text(""));

    let saved = tool.save_state().expect("save_state");
    let mut restored = ParrondoParadoxTool::default();
    restored
        .try_load_state(&saved)
        .expect("valid roundtrip state");
    assert_eq!(restored.active_schedule.as_string(), "AABB");

    // Reject even modulus and out-of-bounds epsilon
    let bad_even = r#"{"p_a_base":0.5,"p_b0_base":0.1,"p_b1_base":0.75,"epsilon":0.005,"modulus":4,"schedule":"ABB","num_walkers":128,"num_steps":100,"seed":7,"view_mode":"SplitOverview"}"#;
    assert!(restored.try_load_state(bad_even).is_err());

    let bad_sched = r#"{"p_a_base":0.5,"p_b0_base":0.1,"p_b1_base":0.75,"epsilon":0.005,"modulus":3,"schedule":"ABZ","num_walkers":128,"num_steps":100,"seed":7,"view_mode":"SplitOverview"}"#;
    assert!(restored.try_load_state(bad_sched).is_err());
}

#[test]
fn test_exports_across_all_view_modes_and_accesskit_render() {
    let mut tool = ParrondoParadoxTool::default();
    let opts = egui_plot::export::ExportOptions::default();

    for mode in ParrondoViewMode::ALL {
        tool.view_mode = mode;
        let datasets = tool.export_datasets();
        assert!(!datasets.is_empty(), "datasets empty in {mode:?}");
        let prims = tool.export_plot_primitives();
        assert!(!prims.is_empty(), "primitives empty in {mode:?}");
        let img = egui_plot::export_primitives::render_primitives_image(&prims, &opts)
            .expect("offscreen raster render");
        let png = egui_plot::export::color32_image_to_png(&img).expect("png encode");
        assert!(png.starts_with(b"\x89PNG\r\n\x1a\n"));
        let svg =
            egui_plot::export_primitives::export_primitives_svg(&prims, &opts).expect("svg encode");
        assert!(svg.contains("<svg"));
    }

    let ctx = eframe::egui::Context::default();
    ctx.enable_accesskit();
    let out = ctx.run(
        eframe::egui::RawInput {
            screen_rect: Some(eframe::egui::Rect::from_min_size(
                eframe::egui::Pos2::ZERO,
                eframe::egui::vec2(1280.0, 800.0),
            )),
            ..Default::default()
        },
        |ctx| tool.show(ctx),
    );
    let ak = out
        .platform_output
        .accesskit_update
        .expect("accesskit update");
    assert!(
        ak.nodes.iter().any(|(_, n)| {
            n.label()
                .is_some_and(|l| l.contains("Parrondo") || l.contains("Trap Occupancy"))
        }),
        "Expected AccessKit canvas labels for Parrondo plots"
    );
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen_test::wasm_bindgen_test]
fn test_wasm32_parrondo_determinism_and_export() {
    let mut tool = ParrondoParadoxTool::default();
    let baseline = tool.traj_schedule.clone();
    tool.recompute_all();
    assert_eq!(tool.traj_schedule, baseline);
    assert!(!tool.export_plot_primitives().is_empty());
}

// [cite:clinical_trials_statistics]
