//! Unit, zero-allocation, export, and AccessKit tests for [`super::MonteCarloWorkbenchTool`].

#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::{
    BootstrapStatKind, MonteCarloWorkbenchTool, WorkbenchBenchmark, WorkbenchViewMode, MAX_SAMPLES,
};
use crate::framework::InteractiveTool;

#[test]
fn test_workbench_defaults_qmc_beat_iid_and_ess_readout() {
    let mut tool = MonteCarloWorkbenchTool::default();
    assert_eq!(tool.name(), "Monte Carlo & Bootstrap Workbench");
    assert_eq!(tool.domain(), "financial_math");

    tool.benchmark = WorkbenchBenchmark::SmoothSeparable3D;
    tool.num_samples = 1024;
    tool.importance_tilt = 0.90;
    tool.recompute();

    let iid = tool.est_iid.expect("IID estimate");
    let sobol = tool.est_sobol.expect("Sobol RQMC estimate");
    assert!(
        sobol.standard_error < iid.standard_error,
        "Sobol RQMC SE ({}) should beat IID SE ({}) on SmoothSeparable3D",
        sobol.standard_error,
        iid.standard_error
    );

    let snis = tool.est_snis.expect("SNIS estimate");
    assert!(snis.effective_sample_size > 200.0);
    assert!(snis.effective_sample_size <= 1024.0 + 1e-9);
}

#[test]
fn test_workbench_zero_allocation_hot_recompute() {
    let mut tool = MonteCarloWorkbenchTool {
        num_samples: 256,
        bootstrap_replicates: 200,
        bootstrap_statistic: BootstrapStatKind::Median,
        ..Default::default()
    };
    tool.recompute_in_place();

    {
        let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        verified_engine::engine::VerifiedEngine::run_verified(|| {
            tool.seed = 77;
            tool.recompute_in_place();
        });
    }

    let comp = tool.boot_comparison.expect("bootstrap comparison");
    assert!(comp.bca.ci_lower <= comp.bca.ci_upper);
    assert!(comp.percentile.ci_lower <= comp.percentile.ci_upper);
}

#[test]
fn test_workbench_state_roundtrip_and_max_iterations_bounds() {
    let mut tool = MonteCarloWorkbenchTool {
        benchmark: WorkbenchBenchmark::SkewedWealthReturn,
        view_mode: WorkbenchViewMode::BootstrapHistogram,
        num_samples: 400,
        bootstrap_replicates: 900,
        seed: 2026,
        ..Default::default()
    };
    tool.recompute();

    let saved = tool.save_state().expect("save_state");
    let mut restored = MonteCarloWorkbenchTool::default();
    restored.try_load_state(&saved).expect("valid state loads");
    assert_eq!(restored.benchmark, WorkbenchBenchmark::SkewedWealthReturn);
    assert_eq!(restored.view_mode, WorkbenchViewMode::BootstrapHistogram);
    assert_eq!(restored.num_samples, 400);
    assert_eq!(restored.bootstrap_replicates, 900);
    assert_eq!(restored.seed, 2026);

    let over_cap = format!(
        r#"{{"benchmark":"SmoothSeparable3D","view_mode":"SplitOverview","num_samples":{},"bootstrap_replicates":500,"confidence_level":0.95,"importance_tilt":0.5,"bootstrap_statistic":"Mean","seed":1}}"#,
        MAX_SAMPLES + 1
    );
    assert!(restored.try_load_state(&over_cap).is_err());
}

#[test]
fn test_workbench_exports_and_accesskit_canvas_labels() {
    let mut tool = MonteCarloWorkbenchTool::default();
    let opts = egui_plot::export::ExportOptions::default();

    for mode in WorkbenchViewMode::ALL {
        tool.view_mode = mode;
        assert!(!tool.export_datasets().is_empty());
        let prims = tool.export_plot_primitives();
        assert!(!prims.is_empty());
        let img = egui_plot::export_primitives::render_primitives_image(&prims, &opts)
            .expect("render primitives");
        let png = egui_plot::export::color32_image_to_png(&img).expect("encode png");
        assert!(png.starts_with(b"\x89PNG\r\n\x1a\n"));
        match mode {
            WorkbenchViewMode::ErrorDecayLogPlot => {
                let _ = std::fs::write("/tmp/monte_carlo_workbench_convergence.png", &png);
            }
            WorkbenchViewMode::BootstrapHistogram => {
                let _ = std::fs::write("/tmp/monte_carlo_workbench_bootstrap.png", &png);
            }
            WorkbenchViewMode::SplitOverview => {}
        }
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
    let ak = out.platform_output.accesskit_update.expect("AccessKit");
    assert!(ak.nodes.iter().any(|(_, n)| {
        n.label()
            .is_some_and(|l| l.contains("Error Decay") || l.contains("Bootstrap"))
    }));
}

// [cite:clinical_trials_statistics]
