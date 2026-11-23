//! Sidebar controls, superconducting presets, and plot lenses for `BcsSuperconductivityTool`.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{
    export, BcsPlotView, BcsSuperconductivityTool, MAX_COUPLING_NV, MAX_DEBYE_MEV, MAX_DOS_PER_EV,
    MAX_FROHLICH_ALPHA, MAX_GAMMA_MEV, MAX_TEMP_K, MIN_COUPLING_NV, MIN_DEBYE_MEV, MIN_DOS_PER_EV,
    MIN_FROHLICH_ALPHA, MIN_GAMMA_MEV, MIN_TEMP_K,
};
use crate::accessibility::{
    paint_canvas_focus_ring, register_canvas_accessibility, AccessibleHoverText,
};
use crate::app::SemanticPalette;
use crate::framework::{scoped_panel_id, PaneSidePanel};
use crate::reflective_ui::render_copyable_metric;
use eframe::egui::{self, accesskit::Role};
use egui_plot::{Corner, HLine, Legend, Line, LineStyle, Plot, PlotPoints, PlotUi, Points, VLine};

/// Renders the sidebar controls and central spectroscopy plot for `BcsSuperconductivityTool`.
pub(crate) fn show_bcs_inside(tool: &mut BcsSuperconductivityTool, parent_ui: &mut egui::Ui) {
    tool.datasheet_export.poll();
    let panel_id = scoped_panel_id(parent_ui, "bcs_superconductivity_sidebar");
    PaneSidePanel::left(panel_id)
        .default_width(345.0)
        .show_inside(parent_ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                render_sidebar(tool, ui);
            });
        });

    egui::CentralPanel::default().show_inside(parent_ui, |ui| {
        render_central_panel(tool, ui);
    });
}

fn render_sidebar(tool: &mut BcsSuperconductivityTool, ui: &mut egui::Ui) {
    ui.heading("BCS Superconductivity Lab");
    let prev_summary = tool.last_param_summary.clone();
    let prev_series = export::build_active_trace_series(tool);

    let mut changed = render_material_presets(tool, ui);
    ui.add_space(4.0);
    changed |= render_bcs_sliders(tool, ui);
    if changed {
        if !prev_series.is_empty() {
            tool.snapshot_manager
                .push_history_run(prev_summary, prev_series);
        }
        tool.recalculate();
        tool.revision = tool.revision.wrapping_add(1).max(1);
    }

    ui.separator();
    render_metrics(tool, ui);
    ui.add_space(4.0);
    render_overlays_and_datasheets(tool, ui);
}

fn render_material_presets(tool: &mut BcsSuperconductivityTool, ui: &mut egui::Ui) -> bool {
    let mut changed = false;
    ui.group(|ui| {
        ui.label(egui::RichText::new("Elemental Superconductor Presets").strong());
        ui.horizontal_wrapped(|ui| {
            if ui
                .button("Al (T_c ≈ 1.2 K)")
                .accessible_hover_text("Load weak-coupling Aluminum parameters (Debye 36.9 meV)")
                .clicked()
            {
                (
                    tool.temperature_k,
                    tool.debye_energy_mev,
                    tool.coupling_nv,
                    tool.dos_per_ev,
                ) = (0.35, 36.9, 0.175, 0.72);
                changed = true;
            }
            if ui
                .button("Nb (T_c ≈ 9.2 K)")
                .accessible_hover_text("Load Niobium cryogenic parameters (Debye 23.7 meV)")
                .clicked()
            {
                (
                    tool.temperature_k,
                    tool.debye_energy_mev,
                    tool.coupling_nv,
                    tool.dos_per_ev,
                ) = (4.2, 23.7, 0.32, 1.45);
                changed = true;
            }
            if ui
                .button("Pb (T_c ≈ 7.2 K)")
                .accessible_hover_text("Load Lead parameters (Debye 9.0 meV, N(0)V = 0.39)")
                .clicked()
            {
                (
                    tool.temperature_k,
                    tool.debye_energy_mev,
                    tool.coupling_nv,
                    tool.dos_per_ev,
                ) = (2.0, 9.0, 0.39, 1.15);
                changed = true;
            }
        });
    });
    changed
}

fn render_bcs_sliders(tool: &mut BcsSuperconductivityTool, ui: &mut egui::Ui) -> bool {
    let mut changed = false;
    ui.group(|ui| {
        ui.label(egui::RichText::new("Pairing & Tunneling Parameters").strong());
        changed |= ui
            .add(
                egui::Slider::new(&mut tool.temperature_k, MIN_TEMP_K..=MAX_TEMP_K)
                    .text("Temperature T (K)"),
            )
            .accessible_hover_text("Bath temperature T in Kelvin (normal state above T_c)")
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut tool.debye_energy_mev, MIN_DEBYE_MEV..=MAX_DEBYE_MEV)
                    .text("Debye Cutoff ħω_D (meV)"),
            )
            .accessible_hover_text("Phonon Debye cutoff energy in millielectronvolts")
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut tool.coupling_nv, MIN_COUPLING_NV..=MAX_COUPLING_NV)
                    .text("Pairing N(0)V"),
            )
            .accessible_hover_text("Dimensionless BCS electron-phonon pairing strength N(0)V")
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut tool.dos_per_ev, MIN_DOS_PER_EV..=MAX_DOS_PER_EV)
                    .text("Fermi DOS N(0) (1/eV)"),
            )
            .accessible_hover_text("Single-spin normal electronic density of states at E_F")
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut tool.dynes_gamma_mev, MIN_GAMMA_MEV..=MAX_GAMMA_MEV)
                    .text("Dynes Broadening Γ (meV)"),
            )
            .accessible_hover_text("Inelastic quasiparticle lifetime broadening parameter Gamma")
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(
                    &mut tool.frohlich_alpha,
                    MIN_FROHLICH_ALPHA..=MAX_FROHLICH_ALPHA,
                )
                .text("Fröhlich Vertex α"),
            )
            .accessible_hover_text("Dimensionless Frohlich electron-phonon vertex coupling")
            .changed();
    });
    changed
}

fn render_metrics(tool: &BcsSuperconductivityTool, ui: &mut egui::Ui) {
    let m = &tool.metrics;
    let palette = SemanticPalette::from_visuals(ui.visuals());
    if let Some(ref err) = tool.solver_error {
        ui.colored_label(palette.error, format!("⚠ {err}"));
    }
    let phase = if m.delta_t_mev > 1e-4 {
        "Superconducting Phase (T < T_c)"
    } else {
        "Normal Fermi-Liquid State (T ≥ T_c)"
    };
    ui.colored_label(
        if m.delta_t_mev > 1e-4 {
            palette.success
        } else {
            palette.warning
        },
        phase,
    );
    render_copyable_metric(ui, "Gap Δ(T)", &format!("{:.4} meV", m.delta_t_mev));
    render_copyable_metric(
        ui,
        "Zero-T Gap Δ(0)",
        &format!("{:.4} meV", m.delta_zero_mev),
    );
    render_copyable_metric(ui, "Critical Temp T_c", &format!("{:.3} K", m.tc_kelvin));
    render_copyable_metric(
        ui,
        "Universal 2Δ(0)/(k_B T_c)",
        &format!("{:.3}", m.universal_ratio),
    );
    render_copyable_metric(
        ui,
        "Zero-Bias NIS G_NS(0)/G_NN",
        &format!("{:.4}", m.zero_bias_conductance),
    );
    render_copyable_metric(
        ui,
        "Fröhlich |M(q_D)|²",
        &format!("{:.3e} eV²", m.frohlich_mq_sq_ev2),
    );
    render_copyable_metric(
        ui,
        "Fermion CAR {c, c†}",
        &format!("{:.3}", m.cooper_car_norm),
    );
}

fn render_overlays_and_datasheets(tool: &mut BcsSuperconductivityTool, ui: &mut egui::Ui) {
    let summary = tool.param_summary();
    let next_idx = tool.snapshot_manager.len() + 1;
    let active_series = export::build_active_trace_series(tool);
    if tool.snapshot_manager.render_ui(ui, move || {
        (!active_series.is_empty()).then(|| (format!("BCS Run {next_idx}"), summary, active_series))
    }) {
        tool.revision = tool.revision.wrapping_add(1).max(1);
    }
    ui.add_space(4.0);
    let md = export::format_markdown_card(tool);
    let csv = export::format_csv_card(tool);
    tool.datasheet_export.render_controls(
        ui,
        "Superconducting Datasheets",
        "bcs_superconductivity_datasheet.md",
        "bcs_superconductivity_datasheet.csv",
        move || md.clone(),
        move || csv.clone(),
    );
}

fn render_central_panel(tool: &mut BcsSuperconductivityTool, ui: &mut egui::Ui) {
    ui.horizontal_wrapped(|ui| {
        for view in [
            BcsPlotView::GapVsTemperature,
            BcsPlotView::QuasiparticleAndCoherence,
            BcsPlotView::NisTunnelingAndCoupling,
        ] {
            if ui
                .selectable_value(&mut tool.plot_view, view, view.label())
                .accessible_hover_text("Switch active BCS spectroscopy plot view")
                .changed()
            {
                tool.revision = tool.revision.wrapping_add(1).max(1);
            }
        }
    });
    ui.separator();

    let palette = SemanticPalette::from_visuals(ui.visuals());
    let (x_lbl, y_lbl) = match tool.plot_view {
        BcsPlotView::GapVsTemperature => ("Temperature T (K)", "Energy Gap Δ(T) (meV)"),
        BcsPlotView::QuasiparticleAndCoherence => (
            "Normal Band Energy ξ_k (meV)",
            "E_k (meV) / Coherence Weight u_k², v_k²",
        ),
        BcsPlotView::NisTunnelingAndCoupling => (
            "Bias Voltage eV (meV) / Wavevector q (Å⁻¹)",
            "Normalized Conductance dI/dV & DOS",
        ),
    };
    let plot_id = scoped_panel_id(ui, "bcs_superconductivity_plot");
    let resp = Plot::new(plot_id)
        .x_axis_label(x_lbl)
        .y_axis_label(y_lbl)
        .include_y(0.0)
        .legend(Legend::default().position(Corner::LeftTop))
        .show(ui, |plot_ui| {
            draw_active_bcs_view(tool, plot_ui, palette);
            tool.snapshot_manager.render_on_plot(plot_ui);
            tool.callout_manager.render_on_plot(plot_ui);
        });

    let summary = format!(
        "{}: T = {:.2} K, Delta(T) = {:.3} meV, T_c = {:.2} K, 2Delta(0)/(k_B T_c) = {:.3}",
        tool.plot_view.label(),
        tool.temperature_k,
        tool.metrics.delta_t_mev,
        tool.metrics.tc_kelvin,
        tool.metrics.universal_ratio
    );
    register_canvas_accessibility(
        ui,
        &resp.response,
        Role::GraphicsDocument,
        "BCS Superconductivity Spectroscopy Plot",
        Some(&summary),
    );
    paint_canvas_focus_ring(ui, &resp.response);
    tool.focused_canvas = resp.response.has_focus().then_some(resp.response.id);
}

fn draw_active_bcs_view(
    tool: &BcsSuperconductivityTool,
    plot_ui: &mut PlotUi<'_>,
    palette: SemanticPalette,
) {
    match tool.plot_view {
        BcsPlotView::GapVsTemperature => draw_gap_curve(tool, plot_ui, palette),
        BcsPlotView::QuasiparticleAndCoherence => draw_quasiparticle_curves(tool, plot_ui, palette),
        BcsPlotView::NisTunnelingAndCoupling => draw_nis_curves(tool, plot_ui, palette),
    }
}

fn draw_gap_curve(
    tool: &BcsSuperconductivityTool,
    plot_ui: &mut PlotUi<'_>,
    palette: SemanticPalette,
) {
    plot_ui.line(
        Line::new(
            "BCS Gap Δ(T)",
            PlotPoints::new(tool.gap_vs_temp_points.clone()),
        )
        .color(palette.series_color(0))
        .width(2.4_f32),
    );
    plot_ui.vline(
        VLine::new("Critical Temperature T_c", tool.metrics.tc_kelvin)
            .color(palette.warning)
            .style(LineStyle::Dashed { length: 7.0 }),
    );
    plot_ui.points(
        Points::new(
            "Operating Point (T, Δ(T))",
            vec![[tool.temperature_k, tool.metrics.delta_t_mev]],
        )
        .color(palette.accent)
        .radius(5.0_f32),
    );
}

fn draw_quasiparticle_curves(
    tool: &BcsSuperconductivityTool,
    plot_ui: &mut PlotUi<'_>,
    palette: SemanticPalette,
) {
    plot_ui.line(
        Line::new(
            "Bogoliubov E_k (meV)",
            PlotPoints::new(tool.dispersion_points.clone()),
        )
        .color(palette.series_color(0))
        .width(2.2_f32),
    );
    plot_ui.line(
        Line::new(
            "Normal |ξ_k| (meV)",
            PlotPoints::new(tool.normal_disp_points.clone()),
        )
        .color(palette.series_color(3))
        .style(LineStyle::Dotted { spacing: 5.0 }),
    );
    plot_ui.line(
        Line::new(
            "Electron-like u_k²",
            PlotPoints::new(tool.u_sq_points.clone()),
        )
        .color(palette.series_color(1)),
    );
    plot_ui.line(
        Line::new("Hole-like v_k²", PlotPoints::new(tool.v_sq_points.clone()))
            .color(palette.series_color(2))
            .style(LineStyle::Dashed { length: 6.0 }),
    );
}

fn draw_nis_curves(
    tool: &BcsSuperconductivityTool,
    plot_ui: &mut PlotUi<'_>,
    palette: SemanticPalette,
) {
    plot_ui.line(
        Line::new(
            "NIS dI/dV (G_NS/G_NN)",
            PlotPoints::new(tool.nis_didv_points.clone()),
        )
        .color(palette.series_color(0))
        .width(2.4_f32),
    );
    plot_ui.line(
        Line::new(
            "Dynes DOS N_S(E)/N(0)",
            PlotPoints::new(tool.dynes_dos_points.clone()),
        )
        .color(palette.series_color(1))
        .style(LineStyle::Dashed { length: 6.0 }),
    );
    let frohlich_scaled: Vec<[f64; 2]> = tool
        .frohlich_points
        .iter()
        .map(|&[q, mq]| [q, mq * 1e4])
        .collect();
    plot_ui.line(
        Line::new(
            "Fröhlich |M_q|² (×10⁴ eV²)",
            PlotPoints::new(frohlich_scaled),
        )
        .color(palette.series_color(2))
        .style(LineStyle::Dotted { spacing: 5.0 }),
    );
    plot_ui.hline(
        HLine::new("Normal-State Conductance G_NN", 1.0)
            .color(palette.warning)
            .style(LineStyle::Dashed { length: 8.0 }),
    );
}

// [cite:mmwave_radiotherapy_setup]
