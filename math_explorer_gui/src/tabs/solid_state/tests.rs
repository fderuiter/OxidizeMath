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
fn solid_state_tab_selects_all_five_tools() {
    let mut tab = SolidStateTab::default();
    for id in [
        "BandStructureTool",
        "CrystalViewer",
        "IsingModelTool",
        "QuasicrystalDiffractionTool",
        "BcsSuperconductivityTool",
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
