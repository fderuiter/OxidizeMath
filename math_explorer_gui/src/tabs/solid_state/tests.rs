#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::band_structure::{BandRegime, BandStructureTool, QuasiperiodicPlotView};
use super::bcs_superconductivity::{BcsSubView, BcsSuperconductivityTool};
use super::quasicrystal_diffraction::{DiffractionStructure, QuasicrystalDiffractionTool};
use super::SolidStateTab;
use crate::framework::InteractiveTool;
use crate::tabs::ExplorerTab;
use eframe::egui;

fn render_tool_frame(tool: &mut dyn InteractiveTool) {
    let ctx = egui::Context::default();
    let raw = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::Vec2::new(1024.0, 768.0),
        )),
        ..Default::default()
    };
    let _ = ctx.run(raw, |ctx| {
        egui_plot::clear_last_plot_datasets();
        tool.show(ctx);
    });
}

#[test]
fn solid_state_tab_selects_all_six_tools() {
    let mut tab = SolidStateTab::default();
    for id in [
        "BandStructureTool",
        "CrystalViewer",
        "IsingModelTool",
        "QuasicrystalDiffractionTool",
        "BcsSuperconductivityTool",
        "AbelianSandpileStudio",
    ] {
        assert!(
            tab.select_tool_by_name(id),
            "SolidStateTab must select {id}"
        );
    }
}

#[test]
fn quasicrystal_diffraction_renders_all_symmetries_and_validates_bounds() {
    let mut tool = QuasicrystalDiffractionTool::default();
    for structure in [
        DiffractionStructure::Fibonacci1D,
        DiffractionStructure::Penrose5Fold,
        DiffractionStructure::AmmannBeenker8Fold,
        DiffractionStructure::Penrose10Fold,
        DiffractionStructure::Stampfli12Fold,
        DiffractionStructure::Square2D,
        DiffractionStructure::Bcc100Slice,
        DiffractionStructure::Fcc100Slice,
    ] {
        tool.state.structure = structure;
        tool.state.show_perp_window = false;
        tool.recompute_physics();
        render_tool_frame(&mut tool);
        assert!(
            !egui_plot::get_last_plot_datasets().is_empty(),
            "Real-space plot datasets must be non-empty for {structure:?}"
        );
        let img = tool.export_raster_image().expect("Bragg raster image");
        assert_eq!(img.size, [tool.state.grid_res, tool.state.grid_res]);
    }

    tool.state.structure = DiffractionStructure::Penrose10Fold;
    tool.state.show_perp_window = true;
    tool.recompute_physics();
    render_tool_frame(&mut tool);
    assert!(!egui_plot::get_last_plot_datasets().is_empty());

    assert!(tool.try_load_state(r#"{"unknown":1}"#).is_err());
    assert!(tool
        .try_load_state(r#"{"structure":"Penrose5Fold","patch_radius":99.0,"window_scale":1.0,"phason_x":0.0,"phason_y":0.0,"q_max":10.0,"grid_res":64,"gamma":0.5,"show_tile_edges":true,"show_bz_overlay":true,"show_perp_window":false}"#)
        .is_err());
}

#[test]
fn band_structure_renders_quasiperiodic_views_and_rejects_malformed_json() {
    let mut tool = BandStructureTool::default();
    for regime in [BandRegime::FibonacciChain, BandRegime::AubryAndre] {
        for view in [
            QuasiperiodicPlotView::SpectrumAndGaps,
            QuasiperiodicPlotView::IdosStaircase,
            QuasiperiodicPlotView::EigenstateIpr,
        ] {
            tool.regime = regime;
            tool.plot_view = view;
            render_tool_frame(&mut tool);
            assert!(
                !egui_plot::get_last_plot_datasets().is_empty(),
                "Plot datasets must be non-empty for {regime:?} / {view:?}"
            );
        }
    }
    assert!(tool
        .try_load_state(r#"{"e0":0.0,"t":1.0,"a":1.0,"hbar":1.0,"m":1.0,"show_free_electron":false,"extra":true}"#)
        .is_err());
    assert!(tool
        .try_load_state(r#"{"e0":0.0,"t":1.0,"a":1.0,"hbar":1.0,"m":1.0,"show_free_electron":false,"regime":"FibonacciChain","chain_sites":999}"#)
        .is_err());
}

#[test]
fn bcs_superconductivity_renders_subviews_and_handles_weak_coupling() {
    let mut tool = BcsSuperconductivityTool::default();
    for view in [
        BcsSubView::GapAndCoherence,
        BcsSubView::PhononsAndDebye,
        BcsSubView::ThomasFermiScreening,
    ] {
        tool.state.view = view;
        tool.solve_bcs_gap();
        render_tool_frame(&mut tool);
        assert!(
            !egui_plot::get_last_plot_datasets().is_empty(),
            "Plot datasets must be non-empty for {view:?}"
        );
    }

    // Weak pairing potential stays finite without panicking
    tool.state.pairing_potential = 0.02;
    tool.solve_bcs_gap();
    assert!(tool.converged_gap_ev.is_finite());

    assert!(tool.try_load_state(r#"{"unknown_field":42}"#).is_err());
    assert!(tool
        .try_load_state(r#"{"view":"GapAndCoherence","pairing_potential":-1.0,"debye_cutoff":2.0,"half_bandwidth":4.0,"mixing_param":0.5,"sound_velocity":2.0,"einstein_freq":4.0,"debye_temp":300.0,"frohlich_alpha":1.0,"k_tf":1.0}"#)
        .is_err());
}

#[test]
fn solid_state_tools_state_roundtrip() {
    for (mut tool, sample_json) in [
        (
            Box::new(BandStructureTool::default()) as Box<dyn InteractiveTool>,
            r#"{"e0":0.0,"t":1.0,"a":1.0,"hbar":1.0,"m":1.0,"show_free_electron":false,"regime":"FibonacciChain","plot_view":"IdosStaircase","chain_sites":55,"modulation":0.45,"aubry_lambda":2.0,"phason_phase":0.0}"#,
        ),
        (
            Box::new(QuasicrystalDiffractionTool::default()) as Box<dyn InteractiveTool>,
            r#"{"structure":"AmmannBeenker8Fold","patch_radius":3.6,"window_scale":1.0,"phason_x":0.25,"phason_y":-0.15,"q_max":15.0,"grid_res":64,"gamma":0.5,"show_tile_edges":true,"show_bz_overlay":true,"show_perp_window":false}"#,
        ),
        (
            Box::new(BcsSuperconductivityTool::default()) as Box<dyn InteractiveTool>,
            r#"{"view":"GapAndCoherence","pairing_potential":0.5,"debye_cutoff":3.0,"half_bandwidth":4.5,"mixing_param":0.5,"sound_velocity":2.4,"einstein_freq":4.8,"debye_temp":360.0,"frohlich_alpha":1.1,"k_tf":1.5}"#,
        ),
    ] {
        tool.try_load_state(sample_json)
            .expect("valid solid-state JSON state");
        let saved = tool.save_state().expect("saved solid-state JSON");
        assert_eq!(saved, sample_json);
    }
}

fn key_raw(key: egui::Key) -> egui::RawInput {
    let mut raw = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::Vec2::new(1280.0, 800.0),
        )),
        ..Default::default()
    };
    raw.events.push(egui::Event::Key {
        key,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers: egui::Modifiers::NONE,
    });
    raw
}

#[test]
fn quasicrystal_bragg_canvas_focus_and_arrow_steer() {
    let mut qc = QuasicrystalDiffractionTool::default();
    let ctx = egui::Context::default();
    let base_raw = || egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::Vec2::new(1280.0, 800.0),
        )),
        ..Default::default()
    };
    let _ = ctx.run(base_raw(), |ctx| qc.show(ctx));
    assert!(qc.focused_canvas_id().is_none());
    let mut focused_id = None;
    for _ in 0..80 {
        let _ = ctx.run(key_raw(egui::Key::Tab), |ctx| qc.show(ctx));
        let _ = ctx.run(base_raw(), |ctx| qc.show(ctx));
        if let Some(id) = qc.focused_canvas_id() {
            focused_id = Some(id);
            break;
        }
    }
    let cid = focused_id.expect("Tab traversal must reach Bragg diffraction canvas");

    let _ = ctx.run(key_raw(egui::Key::ArrowRight), |ctx| qc.show(ctx));
    assert!(
        qc.state.phason_x > 0.0,
        "ArrowRight must steer phason_x when focused"
    );

    let _ = ctx.run(key_raw(egui::Key::Tab), |ctx| qc.show(ctx));
    let _ = ctx.run(base_raw(), |ctx| qc.show(ctx));
    assert_ne!(
        qc.focused_canvas_id(),
        Some(cid),
        "Tab must exit Bragg canvas"
    );
}

#[test]
fn solid_state_multi_pane_workspace_url_roundtrip_preserves_split_tools() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let mut app = crate::app::MathExplorerApp::default();
    let idx = app
        .tabs
        .iter()
        .position(|t| t.url_slug() == "solid_state")
        .unwrap();
    app.set_layout(crate::app::ViewportLayout::SplitHorizontal);
    app.set_pane_tab(0, idx);
    app.set_pane_tab(1, idx);
    app.focused_pane = 1;

    let qc = r#"{"structure":"Stampfli12Fold","patch_radius":3.4,"window_scale":1.0,"phason_x":0.3,"phason_y":-0.2,"q_max":16.0,"grid_res":64,"gamma":0.5,"show_tile_edges":true,"show_bz_overlay":true,"show_perp_window":false}"#;
    assert!(app.tabs[idx].select_tool_by_name("QuasicrystalDiffractionTool"));
    app.tabs[idx]
        .framework_mut()
        .unwrap()
        .active_tool
        .as_mut()
        .unwrap()
        .try_load_state(qc)
        .unwrap();

    let bcs = r#"{"view":"PhononsAndDebye","pairing_potential":0.45,"debye_cutoff":2.5,"half_bandwidth":4.0,"mixing_param":0.5,"sound_velocity":2.6,"einstein_freq":5.0,"debye_temp":340.0,"frohlich_alpha":1.2,"k_tf":1.4}"#;
    let p1 = app.pane_tab_mut(1).expect("pane 1 active");
    assert!(p1.select_tool_by_name("BcsSuperconductivityTool"));
    p1.framework_mut()
        .unwrap()
        .active_tool
        .as_mut()
        .unwrap()
        .try_load_state(bcs)
        .unwrap();

    let frag = app.encode_current_share_fragment().unwrap();
    assert!(frag.starts_with("#ws="));
    let mut restored = crate::app::MathExplorerApp::default();
    assert!(restored
        .apply_share_location_strings(&frag, "", None)
        .unwrap());
    assert_eq!(
        restored.viewport_layout,
        crate::app::ViewportLayout::SplitHorizontal
    );
    assert_eq!(restored.focused_pane, 1);
    let p0_saved = restored.tabs[restored.selected_tab]
        .framework()
        .unwrap()
        .active_tool
        .as_ref()
        .unwrap()
        .save_state();
    let p1_saved = restored
        .pane_tab(1)
        .unwrap()
        .framework()
        .unwrap()
        .active_tool
        .as_ref()
        .unwrap()
        .save_state();
    assert_eq!(p0_saved.as_deref(), Some(qc));
    assert_eq!(p1_saved.as_deref(), Some(bcs));
}
