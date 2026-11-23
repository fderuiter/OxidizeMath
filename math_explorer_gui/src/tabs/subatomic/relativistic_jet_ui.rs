#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! UI controls, 4-momentum shell badges, and beaming plots for [`RelativisticJetTool`].

use super::relativistic_jet_tool::{JetObservables, JetPlotView, RelativisticJetTool};
use crate::accessibility::AccessibleTheoryHover;
use crate::app::SemanticPalette;
use crate::framework::{scoped_panel_id, PaneSidePanel};
use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints, Points};
use math_explorer::physics::high_energy::{
    apparent_transverse_beta, calculate_lorentz_factor, radiation::synchrotron_power,
    relativistic_doppler_factor, C,
};

impl RelativisticJetTool {
    pub(super) fn render_jet_tool(&mut self, parent_ui: &mut egui::Ui) {
        self.draw_jet_side_panel(parent_ui);
        let obs = self.evaluate_observables();
        self.draw_jet_central_panel(parent_ui, &obs);
    }

    fn draw_jet_side_panel(&mut self, parent_ui: &mut egui::Ui) {
        let panel_id = scoped_panel_id(parent_ui, "subatomic_jet_side");
        PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("Relativistic Jet & Observer");
                ui.separator();
                let mut changed = self.draw_kinematic_controls(ui);
                ui.separator();
                changed |= self.draw_radiation_and_fluid_controls(ui);
                ui.separator();
                changed |= self.draw_counting_statistics_controls(ui);
                if changed {
                    self.clamp_in_place();
                    self.revision = self.revision.wrapping_add(1);
                }
            });
        });
    }

    fn draw_kinematic_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        egui::ComboBox::from_label("Plot View")
            .selected_text(self.state.plot_view.label())
            .show_ui(ui, |ui| {
                for v in [
                    JetPlotView::AngularBeamingAndSuperluminal,
                    JetPlotView::LorentzAndSynchrotronVsBeta,
                ] {
                    changed |= ui
                        .selectable_value(&mut self.state.plot_view, v, v.label())
                        .changed();
                }
            });

        ui.strong("Lorentz Boost & Viewing Geometry");
        changed |= ui
            .add(egui::Slider::new(&mut self.state.beta, 0.0..=0.995).text("Jet Speed v/c (beta)"))
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut self.state.viewing_angle_deg, 0.0..=180.0)
                    .text("Viewing Angle theta (deg)"),
            )
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.state.rest_mass_kg, 0.0..=10.0).text("Mass m (kg)"))
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut self.state.comoving_px_factor, -3.0..=3.0)
                    .text("Comoving p_x / (mc)"),
            )
            .changed();
        changed |= ui
            .checkbox(
                &mut self.state.continuous_jet,
                "Steady Continuous Jet (k=2; unchecked = Plasmoid k=3)",
            )
            .changed();
        changed
    }

    fn draw_radiation_and_fluid_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.strong("Synchrotron, IC & Shock Fluid");
        changed |= ui
            .add(
                egui::Slider::new(&mut self.state.magnetic_energy_density_ub, 0.01..=50.0)
                    .logarithmic(true)
                    .text("Mag Energy U_B (J/m^3)"),
            )
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut self.state.electron_power_law_p, 1.1..=4.5)
                    .text("Electron Index p"),
            )
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut self.state.adiabatic_index, 1.1..=2.0)
                    .text("Adiabatic Index gamma_ad"),
            )
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.state.fluid_density, 0.1..=20.0).text("Density rho"))
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut self.state.pressure_over_rho_c2, 0.0..=2.0)
                    .text("Pressure P/(rho c^2)"),
            )
            .changed();
        changed
    }

    fn draw_counting_statistics_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.strong("Li & Ma Gamma-Ray Detection");
        changed |= ui
            .add(egui::Slider::new(&mut self.state.photon_n_on, 0..=2000).text("On Counts N_on"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.state.photon_n_off, 0..=2000).text("Off Counts N_off"))
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut self.state.exposure_alpha, 0.05..=2.0)
                    .text("Exposure alpha"),
            )
            .changed();
        changed
    }

    fn draw_jet_central_panel(&self, parent_ui: &mut egui::Ui, obs: &JetObservables) {
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            let palette = SemanticPalette::from_visuals(ui.visuals());
            self.draw_jet_readout_banner(ui, obs, &palette);
            ui.separator();
            ui.strong(self.state.plot_view.label());
            let datasets = self.compute_jet_datasets();
            let plot_id = scoped_panel_id(ui, "subatomic_jet_plot");
            let response = Plot::new(plot_id)
                .allow_boxed_zoom(false)
                .show(ui, |plot_ui| {
                    for (idx, (name, pts)) in datasets.into_iter().enumerate() {
                        plot_ui.line(
                            Line::new(name, PlotPoints::new(pts))
                                .color(palette.series_color(idx))
                                .width(2.2_f32),
                        );
                    }
                    match self.state.plot_view {
                        JetPlotView::AngularBeamingAndSuperluminal => {
                            plot_ui.points(
                                Points::new(
                                    "Observer Angle",
                                    vec![[self.state.viewing_angle_deg, obs.doppler_delta]],
                                )
                                .radius(5.0_f32)
                                .color(palette.accent),
                            );
                        }
                        JetPlotView::LorentzAndSynchrotronVsBeta => {
                            plot_ui.points(
                                Points::new("Current Jet beta", vec![[self.state.beta, obs.gamma]])
                                    .radius(5.0_f32)
                                    .color(palette.accent),
                            );
                        }
                    }
                });
            response.response.accessible_theory_hover(self);
        });
    }

    fn draw_jet_readout_banner(
        &self,
        ui: &mut egui::Ui,
        obs: &JetObservables,
        palette: &SemanticPalette,
    ) {
        ui.horizontal_wrapped(|ui| {
            let shell_color = if obs.momentum_shell_valid {
                palette.success
            } else {
                palette.error
            };
            ui.colored_label(
                shell_color,
                format!(
                    "4-Momentum p^mu p_mu = -(mc)^2: {} (rest mc={:.3e} -> boosted E/c={:.3e}, p_x={:.3e})",
                    if obs.momentum_shell_valid {
                        "VERIFIED"
                    } else {
                        "VIOLATED"
                    },
                    obs.comoving_p.t,
                    obs.boosted_p.t,
                    obs.boosted_p.x,
                ),
            );
        });
        ui.horizontal_wrapped(|ui| {
            ui.colored_label(
                palette.info,
                format!(
                    "gamma = {:.3} | Doppler delta = {:.3} | beta_app = {:.3}c | Flux Beaming = {:.2e}x",
                    obs.gamma, obs.doppler_delta, obs.beta_apparent, obs.beaming_amplification
                ),
            );
        });
        ui.horizontal_wrapped(|ui| {
            let sigma_color = if obs.li_ma_sigma >= 5.0 {
                palette.success
            } else if obs.li_ma_sigma >= 3.0 {
                palette.warning
            } else {
                palette.muted_text
            };
            ui.colored_label(
                palette.accent,
                format!(
                    "P_sync = {:.3e} W | IC alpha = {:.2} | Shock r = {:.2} | h/c^2 = {:.3}",
                    obs.sync_power_watts,
                    obs.ic_spectral_index,
                    obs.shock_ratio,
                    obs.enthalpy_over_c2,
                ),
            );
            ui.separator();
            ui.colored_label(
                sigma_color,
                format!("Li & Ma Significance S = {:.2} sigma", obs.li_ma_sigma),
            );
        });
    }

    pub(super) fn compute_jet_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        match self.state.plot_view {
            JetPlotView::AngularBeamingAndSuperluminal => self.angular_beaming_datasets(),
            JetPlotView::LorentzAndSynchrotronVsBeta => self.beta_sweep_datasets(),
        }
    }

    fn angular_beaming_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let beta = self.state.beta;
        let samples = 121usize;
        let mut doppler_pts = Vec::with_capacity(samples);
        let mut beta_app_pts = Vec::with_capacity(samples);
        for i in 0..samples {
            let deg = 180.0 * (i as f64) / ((samples - 1) as f64);
            let rad = deg.to_radians();
            if let Ok(d) = relativistic_doppler_factor(beta, rad) {
                doppler_pts.push([deg, d]);
            }
            if let Ok(ba) = apparent_transverse_beta(beta, rad) {
                beta_app_pts.push([deg, ba]);
            }
        }
        vec![
            ("Doppler Factor delta(theta)".into(), doppler_pts),
            ("Apparent Speed beta_app(theta) [c]".into(), beta_app_pts),
            (
                "Superluminal Threshold (beta_app = 1)".into(),
                vec![[0.0, 1.0], [180.0, 1.0]],
            ),
        ]
    }

    fn beta_sweep_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let ub = self.state.magnetic_energy_density_ub;
        let ref_p = synchrotron_power(ub, 0.5, calculate_lorentz_factor(0.5 * C).unwrap_or(1.15))
            .unwrap_or(1.0)
            .max(1e-30);
        let samples = 100usize;
        let mut gamma_pts = Vec::with_capacity(samples);
        let mut sync_pts = Vec::with_capacity(samples);
        for i in 0..samples {
            let b = 0.992 * (i as f64) / ((samples - 1) as f64);
            if let Ok(g) = calculate_lorentz_factor(b * C) {
                gamma_pts.push([b, g]);
                if let Ok(ps) = synchrotron_power(ub, b, g) {
                    sync_pts.push([b, (ps / ref_p).min(25.0)]);
                }
            }
        }
        vec![
            ("Lorentz Factor gamma(beta)".into(), gamma_pts),
            (
                "Normalized Synchrotron Power P(beta)/P(0.5)".into(),
                sync_pts,
            ),
        ]
    }
}
