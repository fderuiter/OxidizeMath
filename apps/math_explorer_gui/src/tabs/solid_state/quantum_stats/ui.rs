//! Sidebar controls, `StatMechError` guidance, and plot lenses for `QuantumStatsPhononTool`.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{
    export, QuantumPhononView, QuantumStatsMode, QuantumStatsPhononTool, MAX_EMIN_EV, MAX_ESPAN_EV,
    MAX_EXCHANGE_MEV, MAX_K_TF, MAX_MU_EV, MAX_OPTICAL_MEV, MAX_SOUND_V, MAX_TEMP_K, MAX_THETA_D_K,
    MIN_EMIN_EV, MIN_ESPAN_EV, MIN_EXCHANGE_MEV, MIN_K_TF, MIN_MU_EV, MIN_OPTICAL_MEV, MIN_SOUND_V,
    MIN_TEMP_K, MIN_THETA_D_K,
};
use crate::accessibility::{
    paint_canvas_focus_ring, register_canvas_accessibility, AccessibleHoverText,
};
use crate::app::SemanticPalette;
use crate::framework::{scoped_panel_id, InteractiveTool, PaneSidePanel};
use crate::reflective_ui::render_copyable_metric;
use eframe::egui::{self, accesskit::Role};
use egui_plot::{Corner, HLine, Legend, Line, LineStyle, Plot, PlotPoints, PlotUi, VLine};

/// Renders the sidebar controls and central physics plot for `QuantumStatsPhononTool`.
pub(crate) fn show_quantum_stats_inside(
    tool: &mut QuantumStatsPhononTool,
    parent_ui: &mut egui::Ui,
) {
    tool.datasheet_export.poll();
    let panel_id = scoped_panel_id(parent_ui, "quantum_stats_phonon_sidebar");
    PaneSidePanel::left(panel_id)
        .default_width(355.0)
        .show_inside(parent_ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                render_sidebar(tool, ui);
            });
        });

    egui::CentralPanel::default().show_inside(parent_ui, |ui| {
        render_central_panel(tool, ui);
    });
}

fn render_sidebar(tool: &mut QuantumStatsPhononTool, ui: &mut egui::Ui) {
    ui.heading("Quantum Stats & Lattice Lab");
    let prev_summary = tool.last_param_summary.clone();
    let prev_series = export::build_active_trace_series(tool);

    let mut changed = render_ensemble_controls(tool, ui);
    ui.add_space(4.0);
    changed |= render_lattice_and_screening_controls(tool, ui);
    if changed {
        if !prev_series.is_empty() && tool.stat_mech_error.is_none() {
            tool.snapshot_manager
                .push_history_run(prev_summary, prev_series);
        }
        tool.recalculate();
        tool.revision = tool.revision.wrapping_add(1).max(1);
    }

    render_stat_mech_error_banner(tool, ui);
    ui.separator();
    render_metrics(tool, ui);
    ui.add_space(4.0);
    render_overlays_and_datasheets(tool, ui);
}

fn render_ensemble_controls(tool: &mut QuantumStatsPhononTool, ui: &mut egui::Ui) -> bool {
    let mut changed = false;
    ui.group(|ui| {
        ui.label(egui::RichText::new("Quantum Ensemble & Chemical Potential").strong());
        for mode in [
            QuantumStatsMode::FermiDiracCompare,
            QuantumStatsMode::BoseEinstein,
            QuantumStatsMode::MaxwellBoltzmann,
        ] {
            changed |= ui
                .radio_value(&mut tool.stats_mode, mode, mode.label())
                .accessible_hover_text("Select active quantum statistical distribution family")
                .changed();
        }
        changed |= ui
            .add(
                egui::Slider::new(&mut tool.temperature_k, MIN_TEMP_K..=MAX_TEMP_K)
                    .text("Temperature T (K)"),
            )
            .accessible_hover_text("Thermodynamic bath temperature T in Kelvin")
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut tool.chemical_potential_ev, MIN_MU_EV..=MAX_MU_EV)
                    .text("Chemical Potential μ (eV)"),
            )
            .accessible_hover_text("Fermi level or chemical potential mu in electronvolts")
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut tool.energy_min_ev, MIN_EMIN_EV..=MAX_EMIN_EV)
                    .text("Band Minimum E_min (eV)"),
            )
            .accessible_hover_text("Single-particle ground-state energy E_min in eV")
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut tool.energy_span_ev, MIN_ESPAN_EV..=MAX_ESPAN_EV)
                    .text("Energy Window ΔE (eV)"),
            )
            .accessible_hover_text("Sampled single-particle energy span in eV")
            .changed();
    });
    changed
}

fn render_lattice_and_screening_controls(
    tool: &mut QuantumStatsPhononTool,
    ui: &mut egui::Ui,
) -> bool {
    let mut changed = false;
    ui.group(|ui| {
        ui.label(egui::RichText::new("Phonons, Magnons & Screening").strong());
        changed |= ui
            .add(
                egui::Slider::new(&mut tool.sound_velocity_eva, MIN_SOUND_V..=MAX_SOUND_V)
                    .text("Acoustic Slope ħv_s (meV·Å)"),
            )
            .accessible_hover_text("Long-wavelength acoustic phonon group velocity slope")
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(
                    &mut tool.optical_omega_mev,
                    MIN_OPTICAL_MEV..=MAX_OPTICAL_MEV,
                )
                .text("Optical Branch ħω_E (meV)"),
            )
            .accessible_hover_text("Einstein zone-center optical phonon energy in meV")
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut tool.debye_theta_k, MIN_THETA_D_K..=MAX_THETA_D_K)
                    .text("Debye Temp Θ_D (K)"),
            )
            .accessible_hover_text("Debye cutoff temperature Theta_D for T^3 lattice heat capacity")
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut tool.k_tf_inv_a, MIN_K_TF..=MAX_K_TF)
                    .text("Thomas-Fermi k_TF (Å⁻¹)"),
            )
            .accessible_hover_text("Inverse Thomas-Fermi screening wavevector in inverse Angstroms")
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(
                    &mut tool.exchange_j_mev,
                    MIN_EXCHANGE_MEV..=MAX_EXCHANGE_MEV,
                )
                .text("Heisenberg J (meV)"),
            )
            .accessible_hover_text("Ferromagnetic spin-wave exchange constant J in meV")
            .changed();
    });
    changed
}

fn render_stat_mech_error_banner(tool: &mut QuantumStatsPhononTool, ui: &mut egui::Ui) {
    let Some(err_msg) = tool.stat_mech_error.clone() else {
        return;
    };
    let palette = SemanticPalette::from_visuals(ui.visuals());
    ui.add_space(4.0);
    ui.group(|ui| {
        ui.colored_label(
            palette.error,
            egui::RichText::new("⚠ Bose-Einstein Condensate Domain Guard (StatMechError)").strong(),
        );
        ui.colored_label(palette.error, &err_msg);
        ui.small("Last valid finite occupancy curves remain preserved on the plot.");
        if ui
            .button("Clamp μ < E_min (Safe Boson Domain)")
            .accessible_hover_text(
                "Clamp chemical potential strictly below E_min to clear StatMechError",
            )
            .clicked()
        {
            tool.clamp_parameters();
            tool.revision = tool.revision.wrapping_add(1).max(1);
        }
    });
}

fn render_metrics(tool: &QuantumStatsPhononTool, ui: &mut egui::Ui) {
    let m = &tool.metrics;
    ui.label(egui::RichText::new("Canonical & Lattice Observables").strong());
    render_copyable_metric(
        ui,
        "Thermal k_B T",
        &format!("{:.3} meV", m.thermal_energy_mev),
    );
    render_copyable_metric(ui, "Partition Function Z", &format!("{:.4}", m.partition_z));
    render_copyable_metric(
        ui,
        "Helmholtz Free F",
        &format!("{:.4} eV", m.free_energy_ev),
    );
    render_copyable_metric(
        ui,
        "Internal Energy U",
        &format!("{:.4} eV", m.internal_energy_ev),
    );
    render_copyable_metric(
        ui,
        "Entropy S",
        &format!("{:.4} meV/K", m.entropy_mev_per_k),
    );
    render_copyable_metric(
        ui,
        "Debye C_V / (N k_B)",
        &format!("{:.4}", m.debye_cv_per_nk),
    );
    render_copyable_metric(
        ui,
        "Screening Length λ_TF",
        &format!("{:.3} Å", m.screening_length_a),
    );
    render_copyable_metric(
        ui,
        "Heisenberg Dimer E_H",
        &format!("{:.3} meV", m.heisenberg_dimer_mev),
    );
    render_copyable_metric(ui, "Boson CCR [b, b†]", &format!("{:.3}", m.boson_ccr_norm));
    render_copyable_metric(
        ui,
        "1D Walk Diffusion D",
        &format!("{:.3}", m.lattice_diffusion_d),
    );
}

fn render_overlays_and_datasheets(tool: &mut QuantumStatsPhononTool, ui: &mut egui::Ui) {
    let summary = tool.param_summary();
    let next_idx = tool.snapshot_manager.len() + 1;
    let active_series = export::build_active_trace_series(tool);
    if tool.snapshot_manager.render_ui(ui, move || {
        (!active_series.is_empty())
            .then(|| (format!("Ensemble {next_idx}"), summary, active_series))
    }) {
        tool.revision = tool.revision.wrapping_add(1).max(1);
    }
    ui.add_space(4.0);
    let md = export::format_markdown_card(tool);
    let csv = export::format_csv_card(tool);
    tool.datasheet_export.render_controls(
        ui,
        "Thermodynamic & Lattice Datasheets",
        "quantum_stats_phonon_datasheet.md",
        "quantum_stats_phonon_datasheet.csv",
        move || md.clone(),
        move || csv.clone(),
    );
}

fn render_central_panel(tool: &mut QuantumStatsPhononTool, ui: &mut egui::Ui) {
    render_view_selector(tool, ui);
    ui.separator();

    let palette = SemanticPalette::from_visuals(ui.visuals());
    if let Some(ref err) = tool.stat_mech_error {
        ui.colored_label(palette.error, format!("⚠ {err}"));
    }
    let (x_lbl, y_lbl) = quantum_axis_labels(tool.plot_view);
    let plot_id = scoped_panel_id(ui, "quantum_stats_phonon_plot");
    let resp = Plot::new(plot_id)
        .x_axis_label(x_lbl)
        .y_axis_label(y_lbl)
        .include_y(0.0)
        .legend(Legend::default().position(Corner::LeftTop))
        .show(ui, |plot_ui| {
            draw_active_quantum_view(tool, plot_ui, palette);
            tool.snapshot_manager.render_on_plot(plot_ui);
            tool.callout_manager.render_on_plot(plot_ui);
        });

    let summary = format!(
        "{}: T = {:.1} K (k_B T = {:.2} meV), mu = {:.3} eV, Z = {:.3}, F = {:.3} eV",
        tool.plot_view.label(),
        tool.temperature_k,
        tool.metrics.thermal_energy_mev,
        tool.chemical_potential_ev,
        tool.metrics.partition_z,
        tool.metrics.free_energy_ev
    );
    register_canvas_accessibility(
        ui,
        &resp.response,
        Role::GraphicsDocument,
        "Quantum Statistics, Phonon Dispersion, and Screening Plot",
        Some(&summary),
    );
    paint_canvas_focus_ring(ui, &resp.response);
}

fn render_view_selector(tool: &mut QuantumStatsPhononTool, ui: &mut egui::Ui) {
    ui.horizontal_wrapped(|ui| {
        for view in [
            QuantumPhononView::QuantumOccupancy,
            QuantumPhononView::PhononAndDebye,
            QuantumPhononView::ScreeningAndYukawa,
        ] {
            if ui
                .selectable_value(&mut tool.plot_view, view, view.label())
                .accessible_hover_text("Switch active quantum statistics / phonon plot view")
                .changed()
            {
                tool.revision = tool.revision.wrapping_add(1).max(1);
            }
        }
    });
}

const fn quantum_axis_labels(view: QuantumPhononView) -> (&'static str, &'static str) {
    match view {
        QuantumPhononView::QuantumOccupancy => {
            ("Single-Particle Energy E (eV)", "Mean Occupancy f(E)")
        }
        QuantumPhononView::PhononAndDebye => (
            "Wavevector k (Å⁻¹) / Low-T Sweep (K)",
            "Dispersion ħω (meV) & Debye C_V / Nk_B",
        ),
        QuantumPhononView::ScreeningAndYukawa => (
            "Wavevector q (Å⁻¹) / Radius r (Å)",
            "Dielectric ε(q) & Screened Yukawa V(r)",
        ),
    }
}

fn draw_active_quantum_view(
    tool: &QuantumStatsPhononTool,
    plot_ui: &mut PlotUi<'_>,
    palette: SemanticPalette,
) {
    match tool.plot_view {
        QuantumPhononView::QuantumOccupancy => draw_occupancy_curves(tool, plot_ui, palette),
        QuantumPhononView::PhononAndDebye => draw_phonon_magnon_curves(tool, plot_ui, palette),
        QuantumPhononView::ScreeningAndYukawa => draw_screening_curves(tool, plot_ui, palette),
    }
}

fn draw_occupancy_curves(
    tool: &QuantumStatsPhononTool,
    plot_ui: &mut PlotUi<'_>,
    palette: SemanticPalette,
) {
    if tool.stats_mode != QuantumStatsMode::BoseEinstein {
        plot_ui.line(
            Line::new(
                "Fermi-Dirac f_FD(E)",
                PlotPoints::new(tool.fd_points.clone()),
            )
            .color(palette.series_color(0))
            .width(2.4_f32),
        );
    }
    if !tool.be_points.is_empty() && tool.stats_mode != QuantumStatsMode::MaxwellBoltzmann {
        plot_ui.line(
            Line::new(
                "Bose-Einstein f_BE(E)",
                PlotPoints::new(tool.be_points.clone()),
            )
            .color(palette.series_color(1))
            .width(2.2_f32),
        );
    }
    plot_ui.line(
        Line::new(
            "Maxwell-Boltzmann f_MB(E)",
            PlotPoints::new(tool.mb_points.clone()),
        )
        .color(palette.series_color(2))
        .style(LineStyle::Dashed { length: 6.0 }),
    );
    plot_ui.vline(
        VLine::new("Chemical Potential μ", tool.chemical_potential_ev)
            .color(palette.warning)
            .style(LineStyle::Dashed { length: 7.0 }),
    );
    plot_ui.hline(
        HLine::new("Half-Filling f = 0.5", 0.5)
            .color(palette.series_color(3))
            .style(LineStyle::Dotted { spacing: 5.0 }),
    );
}

fn draw_phonon_magnon_curves(
    tool: &QuantumStatsPhononTool,
    plot_ui: &mut PlotUi<'_>,
    palette: SemanticPalette,
) {
    plot_ui.line(
        Line::new(
            "Acoustic Phonon ħv_s k (meV)",
            PlotPoints::new(tool.acoustic_points.clone()),
        )
        .color(palette.series_color(0))
        .width(2.2_f32),
    );
    plot_ui.line(
        Line::new(
            "Optical Phonon ħω_E (meV)",
            PlotPoints::new(tool.optical_points.clone()),
        )
        .color(palette.series_color(1))
        .style(LineStyle::Dashed { length: 6.0 }),
    );
    plot_ui.line(
        Line::new(
            "Ferromagnetic Magnon E(k) (meV)",
            PlotPoints::new(tool.magnon_points.clone()),
        )
        .color(palette.series_color(2)),
    );
    let debye_scaled: Vec<[f64; 2]> = tool
        .debye_cv_points
        .iter()
        .map(|&[t_k, cv]| [t_k * 0.02, cv * 40.0])
        .collect();
    plot_ui.line(
        Line::new("Debye T³ C_V (scaled)", PlotPoints::new(debye_scaled))
            .color(palette.series_color(3))
            .style(LineStyle::Dotted { spacing: 5.0 }),
    );
}

fn draw_screening_curves(
    tool: &QuantumStatsPhononTool,
    plot_ui: &mut PlotUi<'_>,
    palette: SemanticPalette,
) {
    plot_ui.line(
        Line::new(
            "Thomas-Fermi ε(q)",
            PlotPoints::new(tool.dielectric_points.clone()),
        )
        .color(palette.series_color(0))
        .width(2.4_f32),
    );
    plot_ui.line(
        Line::new(
            "Yukawa Screened V(r)",
            PlotPoints::new(tool.yukawa_points.clone()),
        )
        .color(palette.series_color(1))
        .style(LineStyle::Dashed { length: 6.0 }),
    );
    plot_ui.hline(
        HLine::new("Vacuum Dielectric ε = 1", 1.0)
            .color(palette.warning)
            .style(LineStyle::Dotted { spacing: 5.0 }),
    );
}

// [cite:mmwave_radiotherapy_setup]
