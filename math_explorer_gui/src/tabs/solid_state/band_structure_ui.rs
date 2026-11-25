#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

// [cite:solid_state]

use super::{BandRegime, BandStructureTool, QuasiperiodicPlotView};
use crate::accessibility::AccessibleHoverText;
use crate::app::SemanticPalette;
use eframe::egui;
use egui_plot::{Legend, Line, LineStyle, MarkerShape, Plot, PlotPoints, Points};
use math_explorer::physics::solid_state::band_theory::{free_electron_1d, tight_binding_1d};
use math_explorer::physics::solid_state::reciprocal::brillouin_zone_1d;
use std::f64::consts::{PI, TAU};

impl BandStructureTool {
    pub(super) fn draw_controls_panel(&mut self, parent_ui: &mut egui::Ui) {
        let panel_id = crate::framework::scoped_panel_id(parent_ui, "band_structure_controls");
        crate::framework::PaneSidePanel::right(panel_id)
            .resizable(false)
            .min_width(265.0)
            .show_inside(parent_ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    ui.heading("Band & Chain Parameters");
                    ui.separator();

                    ui.label("Hamiltonian Regime:");
                    egui::ComboBox::from_id_salt("band_regime_combo")
                        .selected_text(self.regime.label())
                        .show_ui(ui, |ui| {
                            for variant in [
                                BandRegime::PeriodicBloch,
                                BandRegime::FibonacciChain,
                                BandRegime::AubryAndre,
                            ] {
                                ui.selectable_value(&mut self.regime, variant, variant.label());
                            }
                        });

                    ui.separator();
                    self.draw_common_tight_binding_sliders(ui);

                    if self.regime.is_periodic() {
                        self.draw_periodic_controls(ui);
                    } else {
                        self.draw_quasiperiodic_controls(ui);
                    }
                });
            });
    }

    fn draw_common_tight_binding_sliders(&mut self, ui: &mut egui::Ui) {
        ui.add(
            egui::Slider::new(&mut self.e0, -5.0..=5.0)
                .text("E₀ (On-site Energy)")
                .step_by(0.1),
        );
        ui.add(
            egui::Slider::new(&mut self.t, 0.0..=5.0)
                .text("t (Hopping Parameter)")
                .step_by(0.05),
        );
    }

    fn draw_periodic_controls(&mut self, ui: &mut egui::Ui) {
        ui.add(
            egui::Slider::new(&mut self.a, 0.1..=5.0)
                .text("a (Lattice Constant)")
                .step_by(0.1),
        );
        let [k_min, k_max] = brillouin_zone_1d(self.a).unwrap_or([-PI, PI]);
        ui.small(format!("1st Brillouin Zone: [{k_min:.2}, {k_max:.2}] Å⁻¹"));
        ui.separator();
        ui.checkbox(&mut self.show_free_electron, "Show Free Electron Model");
        if self.show_free_electron {
            ui.add(
                egui::Slider::new(&mut self.hbar, 0.1..=5.0)
                    .text("ħ (Reduced Planck)")
                    .step_by(0.1),
            );
            ui.add(
                egui::Slider::new(&mut self.m, 0.1..=5.0)
                    .text("m (Effective Mass)")
                    .step_by(0.1),
            );
        }
    }

    fn draw_quasiperiodic_controls(&mut self, ui: &mut egui::Ui) {
        ui.separator();
        ui.label("Observable View:");
        egui::ComboBox::from_id_salt("band_qp_view_combo")
            .selected_text(self.plot_view.label())
            .show_ui(ui, |ui| {
                for view in [
                    QuasiperiodicPlotView::SpectrumAndGaps,
                    QuasiperiodicPlotView::IdosStaircase,
                    QuasiperiodicPlotView::EigenstateIpr,
                ] {
                    ui.selectable_value(&mut self.plot_view, view, view.label());
                }
            });

        ui.add(egui::Slider::new(&mut self.chain_sites, 8..=233).text("N (Chain Sites)"));
        ui.horizontal_wrapped(|ui| {
            ui.small("Fibonacci F_k:");
            for fk in [13_usize, 21, 34, 55, 89, 144] {
                if ui
                    .small_button(format!("{fk}"))
                    .accessible_hover_text(format!("Set chain length to Fibonacci number {fk}"))
                    .clicked()
                {
                    self.chain_sites = fk;
                }
            }
        });

        match self.regime {
            BandRegime::FibonacciChain => {
                ui.add(
                    egui::Slider::new(&mut self.modulation, 0.0..=0.9).text("δ (Bond Modulation)"),
                );
            }
            BandRegime::AubryAndre => {
                ui.add(
                    egui::Slider::new(&mut self.aubry_lambda, 0.0..=5.0)
                        .text("λ (Quasiperiodic Potential)"),
                );
                ui.add(
                    egui::Slider::new(&mut self.phason_phase, -TAU..=TAU).text("φ (Phason Phase)"),
                );
                let crit = 2.0 * self.t;
                let phase_lbl = if (self.aubry_lambda - crit).abs() < 0.05 {
                    "Critical multifractal (λ ≈ 2t)"
                } else if self.aubry_lambda < crit {
                    "Extended metallic (λ < 2t)"
                } else {
                    "Exponentially localized (λ > 2t)"
                };
                ui.small(phase_lbl);
            }
            BandRegime::PeriodicBloch => {}
        }

        ui.separator();
        let gaps = self.workspace.count_prominent_gaps(1.5);
        ui.label(format!("Prominent Cantor Gaps: {gaps}"));
        ui.label(format!(
            "Mean IPR: {:.4} (1/N = {:.4})",
            self.workspace.mean_ipr(),
            1.0 / (self.chain_sites as f64)
        ));
    }

    pub(super) fn draw_central_plot(&self, parent_ui: &mut egui::Ui) {
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            let palette = SemanticPalette::from_visuals(ui.visuals());
            if let Some(err) = &self.last_error {
                ui.colored_label(palette.error, err);
            }
            if self.regime.is_periodic() {
                self.draw_periodic_bloch_plot(ui, palette);
            } else {
                self.draw_quasiperiodic_plot(ui, palette);
            }
        });
    }

    fn draw_periodic_bloch_plot(&self, ui: &mut egui::Ui, palette: SemanticPalette) {
        ui.heading("E-k Dispersion in First Brillouin Zone");
        let [k_min, k_max] = brillouin_zone_1d(self.a).unwrap_or([-PI, PI]);
        let num_points = 200;
        let k_step = (k_max - k_min) / (f64::from(num_points) - 1.0);
        let tb_points: PlotPoints = (0..num_points)
            .map(|i| {
                let k = k_min + f64::from(i) * k_step;
                [k, tight_binding_1d(k, self.e0, self.t, self.a)]
            })
            .collect();

        Plot::new("band_structure_plot")
            .x_axis_label("Wavevector k (Å⁻¹)")
            .y_axis_label("Energy E (eV)")
            .legend(Legend::default())
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new("Tight Binding", tb_points)
                        .color(palette.series_color(0))
                        .width(2.2_f32),
                );
                if self.show_free_electron {
                    let fe_points: PlotPoints = (0..num_points)
                        .map(|i| {
                            let k = k_min + f64::from(i) * k_step;
                            [k, free_electron_1d(k, self.hbar, self.m)]
                        })
                        .collect();
                    plot_ui.line(
                        Line::new("Free Electron", fe_points)
                            .color(palette.series_color(1))
                            .style(LineStyle::Dashed { length: 7.0_f32 })
                            .width(2.0_f32),
                    );
                }
            });
    }

    fn draw_quasiperiodic_plot(&self, ui: &mut egui::Ui, palette: SemanticPalette) {
        ui.heading(format!(
            "{} — {}",
            self.regime.label(),
            self.plot_view.label()
        ));
        if self.workspace.eigenvalues().is_empty() {
            return;
        }
        match self.plot_view {
            QuasiperiodicPlotView::SpectrumAndGaps => self.plot_qp_spectrum(ui, palette),
            QuasiperiodicPlotView::IdosStaircase => self.plot_qp_idos(ui, palette),
            QuasiperiodicPlotView::EigenstateIpr => self.plot_qp_ipr(ui, palette),
        }
    }

    fn plot_qp_spectrum(&self, ui: &mut egui::Ui, palette: SemanticPalette) {
        let evals = self.workspace.eigenvalues();
        let idos = self.workspace.idos();
        let qp_pts: Vec<[f64; 2]> = idos
            .iter()
            .copied()
            .zip(evals.iter().copied())
            .map(|(u, e)| [u, e])
            .collect();
        let bloch_pts: PlotPoints = (0..160)
            .map(|i| {
                let u = f64::from(i) / 159.0;
                [u, self.e0 - 2.0 * self.t * (PI * u).cos()]
            })
            .collect();
        Plot::new("band_qp_spectrum_plot")
            .x_axis_label("Normalized State Index n / N")
            .y_axis_label("Eigenenergy E_n")
            .legend(Legend::default())
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new("1D Periodic Bloch Reference", bloch_pts)
                        .color(palette.muted_text)
                        .style(LineStyle::Dashed { length: 6.0_f32 }),
                );
                plot_ui.points(
                    Points::new("Quasiperiodic Eigenvalues", qp_pts)
                        .color(palette.series_color(0))
                        .shape(MarkerShape::Circle)
                        .radius(3.2_f32),
                );
            });
    }

    fn plot_qp_idos(&self, ui: &mut egui::Ui, palette: SemanticPalette) {
        let evals = self.workspace.eigenvalues();
        let idos = self.workspace.idos();
        let mut stair = Vec::with_capacity(evals.len() * 2);
        let mut prev_idos = 0.0_f64;
        for (&e, &u) in evals.iter().zip(idos.iter()) {
            stair.push([e, prev_idos]);
            stair.push([e, u]);
            prev_idos = u;
        }
        Plot::new("band_qp_idos_plot")
            .x_axis_label("Energy E")
            .y_axis_label("Integrated Density of States IDOS(E)")
            .legend(Legend::default())
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new("IDOS Devil's Staircase", stair)
                        .color(palette.series_color(2))
                        .width(2.2_f32),
                );
            });
    }

    fn plot_qp_ipr(&self, ui: &mut egui::Ui, palette: SemanticPalette) {
        let evals = self.workspace.eigenvalues();
        let ipr = self.workspace.ipr();
        let ipr_pts: Vec<[f64; 2]> = evals
            .iter()
            .copied()
            .zip(ipr.iter().copied())
            .map(|(e, p)| [e, p])
            .collect();
        let inv_n = 1.0 / (evals.len() as f64);
        let (e_min, e_max) = (evals[0], evals[evals.len() - 1]);
        Plot::new("band_qp_ipr_plot")
            .x_axis_label("Eigenenergy E_n")
            .y_axis_label("Inverse Participation Ratio IPR")
            .include_y(0.0)
            .legend(Legend::default())
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new(
                        "Extended Bloch Limit 1/N",
                        vec![[e_min, inv_n], [e_max, inv_n]],
                    )
                    .color(palette.muted_text)
                    .style(LineStyle::Dashed { length: 6.0_f32 }),
                );
                plot_ui.points(
                    Points::new("Eigenstate IPR(E_n)", ipr_pts)
                        .color(palette.series_color(3))
                        .shape(MarkerShape::Diamond)
                        .radius(3.8_f32),
                );
            });
    }
}
