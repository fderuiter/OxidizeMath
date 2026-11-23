#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Side-panel controls, Weizsäcker breakdown readouts, and plots for [`NuclearStabilityTool`].

use super::nuclear_chart::{element_symbol, render_nuclide_chart_canvas};
use super::nuclear_tool::{NuclearPlotMode, NuclearStabilityTool, NuclideObservables};
use crate::accessibility::AccessibleTheoryHover;
use crate::app::SemanticPalette;
use crate::framework::{scoped_panel_id, PaneSidePanel};
use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints, Points};

const ISOTOPE_PRESETS: [(&str, u32, u32); 7] = [
    ("C-12", 6, 6),
    ("O-16", 8, 8),
    ("Fe-56", 26, 30),
    ("Ni-62", 28, 34),
    ("Sn-120", 50, 70),
    ("Pb-208", 82, 126),
    ("U-238", 92, 146),
];

impl NuclearStabilityTool {
    pub(super) fn render_nuclear_tool(&mut self, parent_ui: &mut egui::Ui) {
        self.draw_side_controls(parent_ui);
        self.draw_central_workspace(parent_ui);
    }

    fn draw_side_controls(&mut self, parent_ui: &mut egui::Ui) {
        let panel_id = scoped_panel_id(parent_ui, "subatomic_nuclear_side");
        PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("Nuclear Stability & Decay");
                ui.separator();
                let mut changed = self.draw_nuclide_controls(ui);
                ui.separator();
                changed |= self.draw_shell_and_tunneling_controls(ui);
                ui.separator();
                changed |= self.draw_decay_and_resonance_controls(ui);
                if changed {
                    self.clamp_in_place();
                    self.revision = self.revision.wrapping_add(1);
                }
            });
        });
    }

    fn draw_nuclide_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.label("Benchmark Isotopes:");
        ui.horizontal_wrapped(|ui| {
            for &(label, z, n) in &ISOTOPE_PRESETS {
                let active = self.state.selected_z == z && self.state.selected_n == n;
                if ui.selectable_label(active, label).clicked() {
                    self.state.selected_z = z;
                    self.state.selected_n = n;
                    changed = true;
                }
            }
        });
        changed |= ui
            .add(egui::Slider::new(&mut self.state.selected_z, 1..=96).text("Protons Z"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.state.selected_n, 0..=150).text("Neutrons N"))
            .changed();
        changed
    }

    fn draw_shell_and_tunneling_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.strong("Shell Spin-Orbit & Alpha Tunneling");
        changed |= ui
            .add(egui::Slider::new(&mut self.state.orbital_l, 0..=6).text("Orbital l"))
            .changed();
        changed |= ui
            .checkbox(
                &mut self.state.spin_aligned,
                "Spin-aligned j = l + 1/2 (else l - 1/2)",
            )
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.state.alpha_beta, 0.01..=0.25).text("Alpha v/c"))
            .changed();
        changed
    }

    fn draw_decay_and_resonance_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        egui::ComboBox::from_label("Plot View")
            .selected_text(self.state.plot_mode.label())
            .show_ui(ui, |ui| {
                for mode in [
                    NuclearPlotMode::BatemanDecayChain,
                    NuclearPlotMode::BreitWignerResonance,
                    NuclearPlotMode::IsobarParabola,
                ] {
                    changed |= ui
                        .selectable_value(&mut self.state.plot_mode, mode, mode.label())
                        .changed();
                }
            });

        ui.strong("Bateman 4-Step Chain (X1 -> X2 -> X3 -> X4)");
        changed |= ui
            .add(
                egui::Slider::new(&mut self.state.half_life_1, 0.5..=150.0).text("T1/2 Parent (s)"),
            )
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut self.state.half_life_2, 0.5..=150.0)
                    .text("T1/2 Daughter (s)"),
            )
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut self.state.half_life_3, 0.5..=150.0)
                    .text("T1/2 Granddaughter (s)"),
            )
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.state.max_decay_time, 5.0..=300.0).text("Horizon (s)"))
            .changed();

        ui.strong("Breit-Wigner Resonance");
        changed |= ui
            .add(
                egui::Slider::new(&mut self.state.resonance_energy_mev, 0.5..=30.0)
                    .text("E_res (MeV)"),
            )
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut self.state.resonance_gamma_mev, 0.1..=10.0)
                    .text("Width Gamma (MeV)"),
            )
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.state.probe_energy_mev, 0.1..=35.0).text("Probe E"))
            .changed();
        changed
    }

    fn draw_central_workspace(&mut self, parent_ui: &mut egui::Ui) {
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            let palette = SemanticPalette::from_visuals(ui.visuals());
            render_nuclide_chart_canvas(self, ui);
            if let Some(obs) = self.evaluate_observables() {
                self.draw_weizsacker_summary_strip(ui, &obs, &palette);
            }
            ui.separator();
            self.draw_selected_plot(ui, &palette);
        });
    }

    fn draw_weizsacker_summary_strip(
        &self,
        ui: &mut egui::Ui,
        obs: &NuclideObservables,
        palette: &SemanticPalette,
    ) {
        let sym = element_symbol(self.state.selected_z);
        let bd = &obs.breakdown;
        ui.horizontal_wrapped(|ui| {
            ui.strong(format!(
                "{sym}-{} (Z={}, N={})",
                obs.mass_a, self.state.selected_z, self.state.selected_n
            ));
            ui.colored_label(
                palette.success,
                format!("B = {:.2} MeV ({:.3} MeV/A)", bd.total, bd.per_nucleon),
            );
            ui.colored_label(
                palette.info,
                format!(
                    "R = {:.2} fm | rho = {:.4} fm^-3",
                    obs.radius_fm, obs.density_fm3
                ),
            );
        });
        ui.horizontal_wrapped(|ui| {
            ui.colored_label(palette.axis_y, format!("Vol {:+.1}", bd.volume));
            ui.colored_label(palette.warning, format!("Surf {:+.1}", bd.surface));
            ui.colored_label(palette.axis_x, format!("Coul {:+.1}", bd.coulomb));
            ui.colored_label(palette.axis_z, format!("Asym {:+.1}", bd.asymmetry));
            ui.colored_label(palette.accent, format!("Pair {:+.2} MeV", bd.pairing));
            ui.separator();
            ui.colored_label(
                palette.muted_text,
                format!(
                    "<L.S>(j={:.1}) = {:.3e} | Gamow 2pi*eta = {:.2} | Q_alpha = {:+.2} MeV | N1(t/2) = {:.1}% | sigma(E) = {:.3}",
                    obs.spin_orbit_j,
                    obs.spin_orbit_shift,
                    obs.gamow_eta,
                    obs.alpha_q_mev,
                    obs.remaining_parent_pct,
                    obs.resonance_sigma,
                ),
            );
        });
    }

    fn draw_selected_plot(&self, ui: &mut egui::Ui, palette: &SemanticPalette) {
        ui.strong(self.state.plot_mode.label());
        let datasets = self.compute_datasets();
        let plot_id = scoped_panel_id(ui, "subatomic_nuclear_plot");
        let response = Plot::new(plot_id)
            .allow_boxed_zoom(false)
            .show(ui, |plot_ui| {
                for (idx, (name, pts)) in datasets.into_iter().enumerate() {
                    plot_ui.line(
                        Line::new(name, PlotPoints::new(pts))
                            .color(palette.series_color(idx))
                            .width(2.1_f32),
                    );
                }
                if let Some(obs) = self.evaluate_observables() {
                    if self.state.plot_mode == NuclearPlotMode::BreitWignerResonance {
                        plot_ui.points(
                            Points::new(
                                "Probe Energy",
                                vec![[self.state.probe_energy_mev, obs.resonance_sigma]],
                            )
                            .radius(5.0_f32)
                            .color(palette.accent),
                        );
                    } else if self.state.plot_mode == NuclearPlotMode::IsobarParabola {
                        plot_ui.points(
                            Points::new(
                                "Selected Nuclide",
                                vec![[f64::from(self.state.selected_z), obs.breakdown.per_nucleon]],
                            )
                            .radius(5.0_f32)
                            .color(palette.accent),
                        );
                    }
                }
            });
        response.response.accessible_theory_hover(self);
    }
}
