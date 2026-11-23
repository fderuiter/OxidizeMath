#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::bifurcation_state::{
    CannibalismRegime, CannibalismToolState, PRESET_DIEKMANN_HOPF_WAVES,
    PRESET_TRIBOLIUM_TWO_CYCLES,
};
use super::{CannibalismTool, PopulationStructureTab};
use crate::framework::theory_links::resolve_portal_paper_links;
use crate::framework::{ExportFormat, ExportStatusSeverity, InteractiveTool};
use crate::tabs::ExplorerTab;
use eframe::egui::{self, CentralPanel, Frame};

#[test]
fn test_cannibalism_all_controls_update_state_revision_and_roundtrip() {
    let mut tool = CannibalismTool::default();
    let mut prev_rev = tool.state_revision();
    assert!(prev_rev > 0);

    let mutations: [fn(&mut CannibalismTool); 9] = [
        |t| t.state.regime = CannibalismRegime::TriboliumChaos,
        |t| t.state.beta0 = 45.0,
        |t| t.state.c_cannibalism = 0.85,
        |t| t.state.c_pa = 0.7,
        |t| t.state.mu0 = 0.25,
        |t| t.state.grid_size = 128,
        |t| t.state.steps_per_frame = 6,
        |t| t.state.sweep_points = 40,
        |t| t.state.is_playing = true,
    ];

    for (idx, mutate) in mutations.into_iter().enumerate() {
        mutate(&mut tool);
        let next_rev = tool.state_revision();
        assert_ne!(
            next_rev, prev_rev,
            "Cannibalism mutation #{idx} did not change state_revision()"
        );
        prev_rev = next_rev;
    }

    // Transient PDE stepping must NOT alter persisted parameter state_revision()
    let rev_before_step = tool.state_revision();
    tool.advance_pde_steps(5);
    assert_eq!(tool.state_revision(), rev_before_step);

    let saved = tool.save_state().expect("save_state must succeed");
    let mut restored = CannibalismTool::default();
    restored
        .try_load_state(&saved)
        .expect("try_load_state must succeed");
    assert_eq!(restored.save_state().as_deref(), Some(saved.as_str()));
    assert_eq!(restored.state_revision(), tool.state_revision());
}

#[test]
fn test_cannibalism_state_rejects_unknown_fields_and_out_of_bounds_values() {
    let mut tool = CannibalismTool::default();
    let mut val: serde_json::Value =
        serde_json::from_str(&tool.save_state().unwrap()).expect("valid state JSON");
    val.as_object_mut()
        .unwrap()
        .insert("rogue_field".to_string(), serde_json::json!(123));
    assert!(
        tool.try_load_state(&val.to_string()).is_err(),
        "deny_unknown_fields must reject extra JSON keys"
    );

    let bad = CannibalismToolState {
        beta0: 500.0,
        ..Default::default()
    };
    let bad_json = serde_json::to_string(&bad).unwrap();
    assert!(tool.try_load_state(&bad_json).is_err());

    let oversized = format!(
        "{{\"regime\":\"DiekmannHopf\",\"pad\":\"{}\"}}",
        "x".repeat(70_000)
    );
    assert!(tool.try_load_state(&oversized).is_err());
}

#[test]
fn test_cannibalism_starter_presets_and_theory_paper_links() {
    let links = resolve_portal_paper_links("CannibalismTool", "cannibalism");
    assert!(
        links.iter().any(|l| l.paper_name == "cannibalism.tex"),
        "CannibalismTool must link to cannibalism.tex in Theory Context Portal, got {links:?}"
    );

    let mut hopf_tool = CannibalismTool::default();
    hopf_tool
        .import_preset_json(PRESET_DIEKMANN_HOPF_WAVES)
        .expect("Diekmann Hopf preset must load");
    assert_eq!(hopf_tool.state.regime, CannibalismRegime::DiekmannHopf);
    assert!((hopf_tool.state.beta0 - 45.0).abs() < 1e-12);
    hopf_tool.complete_bifurcation_sweep();
    assert!(!hopf_tool.bifurcation_points.is_empty());

    let mut tri_tool = CannibalismTool::default();
    tri_tool
        .import_preset_json(PRESET_TRIBOLIUM_TWO_CYCLES)
        .expect("Tribolium 2-cycles preset must load");
    assert_eq!(tri_tool.state.regime, CannibalismRegime::TriboliumChaos);
    assert!((tri_tool.state.c_pa - 0.7).abs() < 1e-12);
    tri_tool.complete_bifurcation_sweep();
    assert!(!tri_tool.bifurcation_points.is_empty());
}

#[test]
fn test_cannibalism_deferred_sweep_and_high_contrast_lexis() {
    let ctx = egui::Context::default();
    let mut tool = CannibalismTool::default();
    tool.rebuild_simulation(false);
    assert_eq!(tool.sweep_cursor, 0);
    assert!(tool.bifurcation_points.is_empty());

    let screen_rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1400.0, 900.0));
    let _ = ctx.run(
        egui::RawInput {
            screen_rect: Some(screen_rect),
            ..Default::default()
        },
        |ctx| {
            CentralPanel::default()
                .frame(Frame::NONE)
                .show(ctx, |ui| tool.show_inside(ui));
        },
    );
    assert!(
        tool.sweep_cursor > 0 && tool.sweep_cursor <= tool.state.sweep_points,
        "Idle frame should advance deferred bifurcation cursor"
    );

    let hc_img = tool.lexis.build_color_image(true);
    assert!(hc_img.pixels.iter().all(|px| px != &egui::Color32::BLACK));
}

fn run_tab_frame(
    ctx: &egui::Context,
    tab: &mut PopulationStructureTab,
    export: Option<ExportFormat>,
) -> egui::FullOutput {
    let mut frame = eframe::Frame::_new_kittest();
    let screen_rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1400.0, 900.0));
    ctx.run(
        egui::RawInput {
            screen_rect: Some(screen_rect),
            ..Default::default()
        },
        |ctx| {
            if let Some(fmt) = export {
                tab.framework.queue_export(fmt);
            }
            CentralPanel::default()
                .frame(Frame::NONE)
                .show(ctx, |ui| tab.show_inside(ui, &mut frame));
        },
    )
}

#[test]
fn test_cannibalism_multi_tool_switching_and_exports() {
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let mut tab = PopulationStructureTab::default();
    assert_eq!(tab.academic_domain(), "Applied Math");
    assert_eq!(tab.url_slug(), "cannibalism");
    assert_eq!(tab.active_tool_id(), Some("PopulationStructureTool"));
    assert_eq!(tab.framework.available_tools.len(), 2);

    tab.framework
        .apply_starter_card(
            "Diekmann Hopf Cohort Waves",
            Some("CannibalismTool"),
            Some(PRESET_DIEKMANN_HOPF_WAVES),
        )
        .unwrap();
    assert_eq!(tab.active_tool_id(), Some("CannibalismTool"));
    let _ = run_tab_frame(&ctx, &mut tab, None);

    for fmt in [
        ExportFormat::Csv,
        ExportFormat::Json,
        ExportFormat::Svg,
        ExportFormat::Png,
    ] {
        let _ = run_tab_frame(&ctx, &mut tab, Some(fmt));
        assert_eq!(
            tab.framework.export_status_severity,
            ExportStatusSeverity::Success
        );
    }
    let bytes = tab
        .framework
        .export_queue
        .last_exported_bytes
        .as_deref()
        .expect("cannibalism PNG bytes");
    assert!(bytes.starts_with(b"\x89PNG\r\n\x1a\n"));
}

#[test]
fn test_export_tribolium_and_lexis_image_generation() {
    let ctx = egui::Context::default();
    let mut tab = PopulationStructureTab::default();
    tab.framework
        .apply_starter_card(
            "Tribolium 2-Cycle Bifurcation",
            Some("CannibalismTool"),
            Some(PRESET_TRIBOLIUM_TWO_CYCLES),
        )
        .unwrap();
    let _ = run_tab_frame(&ctx, &mut tab, None);
    let _ = run_tab_frame(&ctx, &mut tab, Some(ExportFormat::Png));
    let bytes = tab
        .framework
        .export_queue
        .last_exported_bytes
        .as_deref()
        .expect("tribolium PNG bytes");
    assert!(bytes.starts_with(b"\x89PNG\r\n\x1a\n"));

    let mut tool = CannibalismTool::default();
    tool.import_preset_json(PRESET_DIEKMANN_HOPF_WAVES).unwrap();
    tool.advance_pde_steps(600);
    for _ in 0..96 {
        tool.advance_pde_steps(3);
    }
    let color_img = tool.lexis.build_color_image(false);
    assert_eq!(color_img.size, [96, 100]);
}
