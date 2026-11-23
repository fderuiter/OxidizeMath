#![cfg_attr(any(), verified(opt_out = "tests"))]

use eframe::egui;
use verified_engine::engine::VerifiedEngine;

use super::*;
use crate::app::MathExplorerApp;
use crate::framework::recovery::FiniteStateGuard;
use crate::framework::{ExportFormat, ExportStatusSeverity, InteractiveTool};
use crate::tabs::ExplorerTab;

fn screen_input() -> egui::RawInput {
    egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1280.0, 800.0),
        )),
        ..Default::default()
    }
}

#[test]
fn stochastic_pde_tab_is_auto_discovered_and_hydrates_share_urls() {
    let mut app = MathExplorerApp::default();
    let idx = app
        .tabs
        .iter()
        .position(|t| t.url_slug() == "stochastic_pde")
        .expect("StochasticPdeTab must be auto-discovered by build.rs");
    assert_eq!(app.tabs[idx].academic_domain(), "Physics");

    let hydrated = app
        .apply_share_location_strings(
            "#tab=stochastic_pde&tool=SpinodalDecompositionTool",
            "",
            None,
        )
        .expect("stochastic_pde URL must parse");
    assert!(hydrated);
    assert_eq!(app.selected_tab, idx);
    assert_eq!(
        app.tabs[idx].active_tool_id(),
        Some("SpinodalDecompositionTool")
    );
}

#[test]
fn kpz_and_spinodal_tools_register_accesskit_canvas_presets_rollback_and_exports() {
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let mut frame = eframe::Frame::_new_kittest();
    let mut tab = StochasticPdeTab::default();

    for tool_id in ["KpzInterfaceTool", "SpinodalDecompositionTool"] {
        assert!(tab.select_tool_by_name(tool_id));
        let out = ctx.run(screen_input(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| tab.show_inside(ui, &mut frame));
        });
        let ak = out
            .platform_output
            .accesskit_update
            .as_ref()
            .expect("AccessKit enabled");
        assert!(
            ak.nodes
                .iter()
                .any(|(_, n)| n.role() == egui::accesskit::Role::GraphicsDocument),
            "{tool_id} must expose a Role::GraphicsDocument canvas node"
        );

        let fw = tab.framework_mut().expect("framework");
        fw.export_options.width = 320;
        fw.export_options.height = 240;
        for fmt in [ExportFormat::Png, ExportFormat::Svg, ExportFormat::Csv] {
            let _ = ctx.run(screen_input(), |ctx| {
                egui_plot::clear_last_plot_datasets();
                fw.show_export_modal = true;
                fw.queue_export(fmt);
                fw.show(ctx, "spde_export_test");
            });
            assert_eq!(fw.export_status_severity, ExportStatusSeverity::Success);
        }
    }

    // Verify FiniteStateGuard rollback/clamping on both tools
    let mut kpz = KpzInterfaceTool {
        dt: f64::NAN,
        lambda: 999.0,
        ..Default::default()
    };
    assert!(FiniteStateGuard::diverged(&kpz));
    assert!(InteractiveTool::clamp_parameters(&mut kpz));
    assert!(!FiniteStateGuard::diverged(&kpz));

    let mut spinodal = SpinodalDecompositionTool {
        temperature: f64::INFINITY,
        dt: -5.0,
        ..Default::default()
    };
    assert!(FiniteStateGuard::diverged(&spinodal));
    FiniteStateGuard::auto_scale_step_size(&mut spinodal);
    assert!(!FiniteStateGuard::diverged(&spinodal));
}

#[test]
fn kpz_2d_and_spinodal_model_b_conserve_and_step_zero_alloc_under_mutex() {
    let mut kpz = KpzInterfaceTool::default();
    kpz.import_preset_json(presets::PRESET_EW_THERMAL_1D)
        .expect("EW preset valid");
    kpz.dimension = KpzDimensionMode::Dim2D;
    kpz.system_size = 32;
    kpz.rebuild_and_seed();
    assert!(kpz.export_raster_image().is_some());

    let mut spinodal = SpinodalDecompositionTool::default();
    spinodal
        .import_preset_json(presets::PRESET_CAHN_HILLIARD_MAZE)
        .expect("CH preset valid");
    let m0 = spinodal.solver.mean_concentration();
    spinodal.step_once();
    assert!((spinodal.solver.mean_concentration() - m0).abs() < 1e-12);

    let _alloc_guard = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    VerifiedEngine::run_verified(|| {
        kpz.solver_1d.step(&mut kpz.rng);
        kpz.solver_2d.step(&mut kpz.rng);
        spinodal.solver.step(&mut spinodal.rng);
        let _ = spinodal.solver.refresh_telemetry();
    });
}
