#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

// [cite:solid_state]

use super::{BcsSubView, BcsSuperconductivityTool};
use crate::app::SemanticPalette;
use eframe::egui;
use egui_plot::{Legend, Line, LineStyle, Plot, PlotPoints};
use math_explorer::physics::solid_state::bcs::coherence_factors_strong;
use math_explorer::physics::solid_state::interactions::FrohlichVertex;
use math_explorer::physics::solid_state::phonons::{
    acoustic_dispersion, debye_heat_capacity_low_temp, optical_dispersion,
};
use math_explorer::physics::solid_state::screening::{thomas_fermi_dielectric, yukawa_potential};
use math_explorer::physics::solid_state::types::ElectronVolts;

impl BcsSuperconductivityTool {
    pub(super) fn draw_controls_panel(&mut self, parent_ui: &mut egui::Ui) {
        let panel_id = crate::framework::scoped_panel_id(parent_ui, "bcs_controls");
        crate::framework::PaneSidePanel::left(panel_id)
            .resizable(false)
            .min_width(265.0)
            .show_inside(parent_ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    ui.heading("BCS, Phonons & Screening");
                    ui.separator();

                    ui.label("Physics Module:");
                    egui::ComboBox::from_id_salt("bcs_subview_combo")
                        .selected_text(self.state.view.label())
                        .show_ui(ui, |ui| {
                            for v in [
                                BcsSubView::GapAndCoherence,
                                BcsSubView::PhononsAndDebye,
                                BcsSubView::ThomasFermiScreening,
                            ] {
                                ui.selectable_value(&mut self.state.view, v, v.label());
                            }
                        });

                    ui.separator();
                    let changed = match self.state.view {
                        BcsSubView::GapAndCoherence => self.draw_bcs_sliders(ui),
                        BcsSubView::PhononsAndDebye => self.draw_phonon_sliders(ui),
                        BcsSubView::ThomasFermiScreening => self.draw_screening_sliders(ui),
                    };
                    if changed {
                        self.mark_changed();
                    }
                });
            });
    }

    fn draw_bcs_sliders(&mut self, ui: &mut egui::Ui) -> bool {
        let mut c = false;
        c |= ui
            .add(
                egui::Slider::new(&mut self.state.pairing_potential, 0.05..=1.5)
                    .text("V (Pairing Potential eV)"),
            )
            .changed();
        c |= ui
            .add(
                egui::Slider::new(&mut self.state.debye_cutoff, 0.4..=6.0)
                    .text("ħω_D (Debye Cutoff eV)"),
            )
            .changed();
        c |= ui
            .add(
                egui::Slider::new(&mut self.state.half_bandwidth, 1.0..=8.0)
                    .text("W/2 (Half-Bandwidth eV)"),
            )
            .changed();
        c |= ui
            .add(
                egui::Slider::new(&mut self.state.mixing_param, 0.1..=0.95)
                    .text("α (Solver Mixing)"),
            )
            .changed();

        ui.separator();
        let tc_est = self.converged_gap_ev / 1.764;
        ui.label(format!(
            "Self-Consistent Gap Δ = {:.4} eV",
            self.converged_gap_ev
        ));
        ui.small(format!(
            "Weak-Coupling k_B T_c ≈ Δ / 1.764 = {tc_est:.4} eV"
        ));
        if let Some(warn) = &self.convergence_warning {
            let palette = SemanticPalette::from_visuals(ui.visuals());
            ui.colored_label(palette.warning, warn);
        }
        c
    }

    fn draw_phonon_sliders(&mut self, ui: &mut egui::Ui) -> bool {
        let mut c = false;
        c |= ui
            .add(
                egui::Slider::new(&mut self.state.sound_velocity, 0.4..=6.0)
                    .text("v_s (Sound Velocity)"),
            )
            .changed();
        c |= ui
            .add(
                egui::Slider::new(&mut self.state.einstein_freq, 0.5..=9.0)
                    .text("ω_E (Einstein Optical Freq)"),
            )
            .changed();
        c |= ui
            .add(
                egui::Slider::new(&mut self.state.debye_temp, 40.0..=800.0)
                    .text("Θ_D (Debye Temperature K)"),
            )
            .changed();
        c |= ui
            .add(
                egui::Slider::new(&mut self.state.frohlich_alpha, 0.1..=5.0)
                    .text("α_F (Fröhlich Coupling)"),
            )
            .changed();
        let cv_20k = debye_heat_capacity_low_temp(20.0, self.state.debye_temp, 1.0, 1.0);
        ui.separator();
        ui.small(format!("C_V(20 K) / (N k_B) = {cv_20k:.5}"));
        c
    }

    fn draw_screening_sliders(&mut self, ui: &mut egui::Ui) -> bool {
        let changed = ui
            .add(
                egui::Slider::new(&mut self.state.k_tf, 0.1..=4.0)
                    .text("k_TF (Thomas-Fermi Wavevector)"),
            )
            .changed();
        ui.separator();
        ui.small(format!(
            "Screening Length λ_TF = 1 / k_TF = {:.3} Å",
            1.0 / self.state.k_tf.max(1e-6)
        ));
        ui.small(format!(
            "ε(q = 1 Å⁻¹) = {:.3}",
            thomas_fermi_dielectric(1.0, self.state.k_tf)
        ));
        changed
    }

    pub(super) fn draw_central_plots(&self, parent_ui: &mut egui::Ui) {
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            let palette = SemanticPalette::from_visuals(ui.visuals());
            match self.state.view {
                BcsSubView::GapAndCoherence => self.draw_bcs_gap_and_coherence_plots(ui, palette),
                BcsSubView::PhononsAndDebye => self.draw_phonons_and_debye_plots(ui, palette),
                BcsSubView::ThomasFermiScreening => self.draw_screening_plots(ui, palette),
            }
        });
    }

    fn draw_bcs_gap_and_coherence_plots(&self, ui: &mut egui::Ui, palette: SemanticPalette) {
        let half_h = (ui.available_height() * 0.46).max(160.0);
        let bw = self.state.half_bandwidth;
        let delta = self.converged_gap_ev.max(1e-5);
        let n = 180;
        let upper: PlotPoints = (0..n)
            .map(|i| {
                let xi = -bw + 2.0 * bw * (f64::from(i) / f64::from(n - 1));
                [xi, xi.hypot(delta)]
            })
            .collect();
        let lower: PlotPoints = (0..n)
            .map(|i| {
                let xi = -bw + 2.0 * bw * (f64::from(i) / f64::from(n - 1));
                [xi, -xi.hypot(delta)]
            })
            .collect();
        let normal: PlotPoints = (0..n)
            .map(|i| {
                let xi = -bw + 2.0 * bw * (f64::from(i) / f64::from(n - 1));
                [xi, xi.abs()]
            })
            .collect();

        ui.strong("Bogoliubov Quasiparticle Dispersion ±E_k vs Normal |ξ_k|");
        Plot::new("bcs_quasiparticle_plot")
            .height(half_h)
            .x_axis_label("Single-Particle Energy ξ_k = ε_k - μ (eV)")
            .y_axis_label("Excitation Energy (eV)")
            .legend(Legend::default())
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new("+E_k = +√(ξ_k² + Δ²)", upper)
                        .color(palette.series_color(0))
                        .width(2.2_f32),
                );
                plot_ui.line(
                    Line::new("-E_k = -√(ξ_k² + Δ²)", lower)
                        .color(palette.series_color(1))
                        .width(2.2_f32),
                );
                plot_ui.line(
                    Line::new("Normal |ξ_k| (Δ = 0)", normal)
                        .color(palette.muted_text)
                        .style(LineStyle::Dashed { length: 6.0_f32 }),
                );
            });

        self.draw_coherence_factors_plot(ui, palette, half_h, bw, delta);
    }

    fn draw_coherence_factors_plot(
        &self,
        ui: &mut egui::Ui,
        palette: SemanticPalette,
        height: f32,
        bw: f64,
        delta: f64,
    ) {
        let mut u_sq_pts = Vec::with_capacity(180);
        let mut v_sq_pts = Vec::with_capacity(180);
        let delta_ev = ElectronVolts::new(delta);
        for i in 0..180 {
            let xi = -bw + 2.0 * bw * (f64::from(i) / 179.0);
            if let Ok((u, v)) = coherence_factors_strong(ElectronVolts::new(xi), delta_ev) {
                u_sq_pts.push([xi, u * u]);
                v_sq_pts.push([xi, v * v]);
            }
        }
        ui.strong("Bogoliubov Coherence Factors u_k² (Electron-Like) & v_k² (Hole-Like)");
        Plot::new("bcs_coherence_plot")
            .height(height)
            .x_axis_label("Single-Particle Energy ξ_k (eV)")
            .y_axis_label("Occupation Weight")
            .include_y(0.0)
            .include_y(1.0)
            .legend(Legend::default())
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new("v_k² (Pair Occupation)", v_sq_pts)
                        .color(palette.series_color(2))
                        .width(2.2_f32),
                );
                plot_ui.line(
                    Line::new("u_k² = 1 - v_k²", u_sq_pts)
                        .color(palette.series_color(3))
                        .width(2.2_f32),
                );
            });
    }

    fn draw_phonons_and_debye_plots(&self, ui: &mut egui::Ui, palette: SemanticPalette) {
        let half_h = (ui.available_height() * 0.46).max(160.0);
        let vertex = FrohlichVertex::new(self.state.frohlich_alpha, self.state.einstein_freq);
        let ac_pts: PlotPoints = (1..=160)
            .map(|i| {
                let k = f64::from(i) * 0.02;
                [k, acoustic_dispersion(k, self.state.sound_velocity)]
            })
            .collect();
        let op_pts: PlotPoints = (1..=160)
            .map(|i| {
                let k = f64::from(i) * 0.02;
                [k, optical_dispersion(k, self.state.einstein_freq)]
            })
            .collect();
        let fr_pts: PlotPoints = (4..=160)
            .map(|i| {
                let q = f64::from(i) * 0.02;
                [q, vertex.amplitude(q).norm_sqr().min(12.0)]
            })
            .collect();

        ui.strong("Acoustic / Optical Phonon Branches & Fröhlich Vertex |M_q|²");
        Plot::new("phonon_dispersion_plot")
            .height(half_h)
            .x_axis_label("Wavevector k / q (Å⁻¹)")
            .y_axis_label("ω(k) / |M_q|²")
            .legend(Legend::default())
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new("Acoustic ω = v_s k", ac_pts)
                        .color(palette.series_color(0))
                        .width(2.2_f32),
                );
                plot_ui.line(
                    Line::new("Optical ω = ω_E", op_pts)
                        .color(palette.series_color(1))
                        .width(2.2_f32),
                );
                plot_ui.line(
                    Line::new("Fröhlich |M_q|² ~ α/q²", fr_pts)
                        .color(palette.series_color(4))
                        .style(LineStyle::Dashed { length: 6.0_f32 }),
                );
            });

        self.draw_debye_cv_plot(ui, palette, half_h);
    }

    fn draw_debye_cv_plot(&self, ui: &mut egui::Ui, palette: SemanticPalette, height: f32) {
        let t_max = (0.35 * self.state.debye_temp).max(10.0);
        let cv_pts: PlotPoints = (0..160)
            .map(|i| {
                let t = t_max * (f64::from(i) / 159.0);
                [
                    t,
                    debye_heat_capacity_low_temp(t, self.state.debye_temp, 1.0, 1.0),
                ]
            })
            .collect();
        ui.strong("Low-Temperature Debye Heat Capacity C_V(T) / (N k_B) ∝ (T / Θ_D)³");
        Plot::new("debye_cv_plot")
            .height(height)
            .x_axis_label("Temperature T (K)")
            .y_axis_label("C_V / (N k_B)")
            .legend(Legend::default())
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new("Debye T³ Law", cv_pts)
                        .color(palette.series_color(2))
                        .width(2.2_f32),
                );
            });
    }

    fn draw_screening_plots(&self, ui: &mut egui::Ui, palette: SemanticPalette) {
        let half_h = (ui.available_height() * 0.46).max(160.0);
        let eps_pts: PlotPoints = (5..=180)
            .map(|i| {
                let q = f64::from(i) * 0.03;
                [q, thomas_fermi_dielectric(q, self.state.k_tf)]
            })
            .collect();
        ui.strong("Static Thomas-Fermi Dielectric Function ε(q) = 1 + k_TF² / q²");
        Plot::new("tf_dielectric_plot")
            .height(half_h)
            .x_axis_label("Wavevector q (Å⁻¹)")
            .y_axis_label("ε(q)")
            .legend(Legend::default())
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new("Thomas-Fermi ε(q)", eps_pts)
                        .color(palette.series_color(0))
                        .width(2.2_f32),
                );
            });

        let yuk_pts: PlotPoints = (5..=180)
            .map(|i| {
                let r = f64::from(i) * 0.03;
                [r, yukawa_potential(r, self.state.k_tf)]
            })
            .collect();
        let coul_pts: PlotPoints = (5..=180)
            .map(|i| {
                let r = f64::from(i) * 0.03;
                [r, 1.0 / r]
            })
            .collect();
        ui.strong("Screened Yukawa Potential V(r) = exp(-k_TF r)/r vs Bare Coulomb 1/r");
        Plot::new("yukawa_screening_plot")
            .height(half_h)
            .x_axis_label("Distance r (Å)")
            .y_axis_label("Potential V(r)")
            .legend(Legend::default())
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new("Screened Yukawa V(r)", yuk_pts)
                        .color(palette.series_color(2))
                        .width(2.2_f32),
                );
                plot_ui.line(
                    Line::new("Bare Coulomb 1/r", coul_pts)
                        .color(palette.muted_text)
                        .style(LineStyle::Dashed { length: 6.0_f32 }),
                );
            });
    }
}
