#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! UI controls, electroweak/Landau-pole banners, and plot curves for [`StandardModelTool`].

use super::standard_model_tool::{GuiNeutrinoFlavor, SmObservables, SmPlotView, StandardModelTool};
use crate::accessibility::AccessibleTheoryHover;
use crate::app::SemanticPalette;
use crate::framework::{scoped_panel_id, PaneSidePanel};
use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints, Points};
use math_explorer::error::StandardModelError;
use math_explorer::physics::standard_model::{
    neutrinos::{msw_oscillation_probabilities, pmns_oscillation_probabilities},
    qcd::running_coupling,
};

const MZ_REF_GEV: f64 = 91.1876;

impl StandardModelTool {
    pub(super) fn render_standard_model_tool(&mut self, parent_ui: &mut egui::Ui) {
        self.draw_sm_side_panel(parent_ui);
        let obs = self.evaluate_observables();
        self.draw_sm_central_panel(parent_ui, &obs);
    }

    fn draw_sm_side_panel(&mut self, parent_ui: &mut egui::Ui) {
        let panel_id = scoped_panel_id(parent_ui, "subatomic_sm_side");
        PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("Standard Model & Neutrinos");
                ui.separator();
                let mut changed = self.draw_view_and_neutrino_controls(ui);
                ui.separator();
                changed |= self.draw_electroweak_and_higgs_controls(ui);
                ui.separator();
                changed |= self.draw_ckm_and_qcd_controls(ui);
                if changed {
                    self.clamp_in_place();
                    self.revision = self.revision.wrapping_add(1);
                }
            });
        });
    }

    fn draw_view_and_neutrino_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = self.draw_view_and_flavor_selectors(ui);
        changed |= self.draw_pmns_and_msw_sliders(ui);
        changed
    }

    fn draw_view_and_flavor_selectors(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        egui::ComboBox::from_label("Plot View")
            .selected_text(self.state.plot_view.label())
            .show_ui(ui, |ui| {
                for v in [
                    SmPlotView::MswOscillations,
                    SmPlotView::QcdRunning,
                    SmPlotView::CkmUnitarityTriangle,
                ] {
                    changed |= ui
                        .selectable_value(&mut self.state.plot_view, v, v.label())
                        .changed();
                }
            });

        ui.strong("3-Flavor PMNS + MSW Matter");
        egui::ComboBox::from_label("Source Flavor")
            .selected_text(self.state.initial_flavor.label())
            .show_ui(ui, |ui| {
                for f in [
                    GuiNeutrinoFlavor::Electron,
                    GuiNeutrinoFlavor::Muon,
                    GuiNeutrinoFlavor::Tau,
                ] {
                    changed |= ui
                        .selectable_value(&mut self.state.initial_flavor, f, f.label())
                        .changed();
                }
            });
        changed |= ui
            .checkbox(&mut self.state.inverted_ordering, "Inverted Mass Ordering")
            .changed();
        changed |= ui
            .checkbox(&mut self.state.antineutrino, "Antineutrino Mode (bar_nu)")
            .changed();
        changed
    }

    fn draw_pmns_and_msw_sliders(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        changed |= ui
            .add(
                egui::Slider::new(&mut self.state.pmns_theta12_deg, 0.0..=90.0)
                    .text("PMNS th12 (deg)"),
            )
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut self.state.pmns_theta23_deg, 0.0..=90.0)
                    .text("PMNS th23 (deg)"),
            )
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut self.state.pmns_theta13_deg, 0.0..=45.0)
                    .text("PMNS th13 (deg)"),
            )
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut self.state.pmns_delta_cp_deg, 0.0..=360.0)
                    .text("PMNS delta_CP (deg)"),
            )
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.state.neutrino_energy_gev, 0.2..=30.0).text("E (GeV)"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.state.baseline_km, 10.0..=12_800.0).text("L (km)"))
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut self.state.matter_density_g_cm3, 0.0..=14.0)
                    .text("Matter rho (g/cm^3)"),
            )
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.state.electron_fraction, 0.3..=0.7).text("Y_e"))
            .changed();
        changed
    }

    fn draw_electroweak_and_higgs_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.strong("Electroweak Gauge & Higgs Potential");
        changed |= ui
            .add(egui::Slider::new(&mut self.state.g1, 0.1..=1.2).text("U(1)_Y g1"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.state.g2, 0.1..=1.5).text("SU(2)_L g2"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.state.higgs_mu2, -25_000.0..=-100.0).text("mu^2"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.state.higgs_lambda, 0.02..=1.0).text("lambda"))
            .changed();
        changed
    }

    fn draw_ckm_and_qcd_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.strong("CKM Quark Mixing & 1-Loop QCD");
        changed |= ui
            .add(egui::Slider::new(&mut self.state.ckm_theta12_deg, 0.5..=30.0).text("CKM th12"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.state.ckm_delta_deg, 0.0..=360.0).text("CKM delta"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.state.alpha_s_mz, 0.07..=0.20).text("alpha_s(M_Z)"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.state.active_flavors, 3.0..=6.0).text("Active n_f"))
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut self.state.probe_scale_gev, 0.08..=1000.0)
                    .logarithmic(true)
                    .text("Scale Q (GeV)"),
            )
            .changed();
        changed
    }

    fn draw_sm_central_panel(&self, parent_ui: &mut egui::Ui, obs: &SmObservables) {
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            let palette = SemanticPalette::from_visuals(ui.visuals());
            self.draw_sm_summary_cards(ui, obs, &palette);
            ui.separator();
            ui.strong(self.state.plot_view.label());
            let datasets = self.compute_plot_datasets();
            let plot_id = scoped_panel_id(ui, "subatomic_sm_plot");
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
                    self.draw_sm_plot_markers(plot_ui, obs, &palette);
                });
            response.response.accessible_theory_hover(self);
        });
    }

    fn draw_sm_summary_cards(
        &self,
        ui: &mut egui::Ui,
        obs: &SmObservables,
        palette: &SemanticPalette,
    ) {
        ui.horizontal_wrapped(|ui| {
            ui.colored_label(
                palette.info,
                format!(
                    "sin^2(theta_W) = {:.4} | cos(theta_W) = {:.4}",
                    obs.sin2_theta_w, obs.cos_theta_w
                ),
            );
            ui.separator();
            ui.colored_label(
                palette.success,
                format!(
                    "Higgs VEV v = {:.1} GeV | M_W = {:.2} GeV | M_Z = {:.2} GeV",
                    obs.vev_gev, obs.m_w_gev, obs.m_z_gev
                ),
            );
        });
        ui.horizontal_wrapped(|ui| {
            ui.colored_label(
                palette.accent,
                format!(
                    "CKM J = {:.3e} | Apex (rho_bar, eta_bar) = ({:.3}, {:.3})",
                    obs.ckm_jarlskog, obs.ckm_apex.0, obs.ckm_apex.1
                ),
            );
            ui.separator();
            match &obs.qcd_alpha_probe {
                Ok(alpha) => ui.colored_label(
                    palette.axis_y,
                    format!(
                        "QCD alpha_s({:.2} GeV) = {:.4} (Lambda_QCD pole ~ {:.3} GeV)",
                        self.state.probe_scale_gev, alpha, obs.landau_pole_gev
                    ),
                ),
                Err(StandardModelError::LandauPole { scale }) => ui.colored_label(
                    palette.error,
                    format!(
                        "Infrared LandauPole reached at Q = {scale:.3} GeV <= Lambda_QCD ({:.3} GeV)!",
                        obs.landau_pole_gev
                    ),
                ),
                Err(err) => ui.colored_label(palette.warning, format!("QCD status: {err}")),
            };
        });
        ui.horizontal_wrapped(|ui| {
            ui.colored_label(
                palette.muted_text,
                format!(
                    "MSW @ L/E={:.0} km/GeV (A_CC={:+.2e} eV^2): P_e={:.3} (vac {:.3}), P_mu={:.3}, P_tau={:.3}",
                    self.state.baseline_km / self.state.neutrino_energy_gev,
                    obs.matter_acc_ev2,
                    obs.msw_probs[0],
                    obs.vac_probs[0],
                    obs.msw_probs[1],
                    obs.msw_probs[2],
                ),
            );
        });
    }

    fn draw_sm_plot_markers(
        &self,
        plot_ui: &mut egui_plot::PlotUi<'_>,
        obs: &SmObservables,
        palette: &SemanticPalette,
    ) {
        match self.state.plot_view {
            SmPlotView::MswOscillations => {
                let le = self.state.baseline_km / self.state.neutrino_energy_gev;
                plot_ui.points(
                    Points::new("Current L/E (P_e)", vec![[le, obs.msw_probs[0]]])
                        .radius(5.0_f32)
                        .color(palette.accent),
                );
            }
            SmPlotView::QcdRunning => {
                if let Ok(alpha) = obs.qcd_alpha_probe {
                    plot_ui.points(
                        Points::new(
                            "Probe Scale log10(Q)",
                            vec![[self.state.probe_scale_gev.log10(), alpha.min(4.0)]],
                        )
                        .radius(5.0_f32)
                        .color(palette.accent),
                    );
                }
            }
            SmPlotView::CkmUnitarityTriangle => {
                plot_ui.points(
                    Points::new(
                        "Apex (rho_bar, eta_bar)",
                        vec![[obs.ckm_apex.0, obs.ckm_apex.1]],
                    )
                    .radius(5.5_f32)
                    .color(palette.accent),
                );
            }
        }
    }

    pub(super) fn compute_plot_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        match self.state.plot_view {
            SmPlotView::MswOscillations => self.msw_datasets(),
            SmPlotView::QcdRunning => self.qcd_datasets(),
            SmPlotView::CkmUnitarityTriangle => self.ckm_triangle_datasets(),
        }
    }

    fn msw_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let s = &self.state;
        let pmns = self.pmns_params();
        let flv = s.initial_flavor.to_kernel();
        let samples = 110usize;
        let max_le = (1.6 * s.baseline_km / s.neutrino_energy_gev).clamp(400.0, 2500.0);
        let (mut pe, mut pmu, mut ptau, mut pe_vac) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for i in 0..samples {
            let le = max_le * (i as f64) / ((samples - 1) as f64);
            let l_km = le * s.neutrino_energy_gev;
            if let Ok(p) = msw_oscillation_probabilities(
                &pmns,
                flv,
                l_km,
                s.neutrino_energy_gev,
                s.matter_density_g_cm3,
                s.electron_fraction,
                s.antineutrino,
            ) {
                pe.push([le, p[0]]);
                pmu.push([le, p[1]]);
                ptau.push([le, p[2]]);
            }
            if let Ok(pv) = pmns_oscillation_probabilities(
                &pmns,
                flv,
                l_km,
                s.neutrino_energy_gev,
                s.antineutrino,
            ) {
                pe_vac.push([le, pv[0]]);
            }
        }
        vec![
            ("MSW P(-> nu_e)".into(), pe),
            ("MSW P(-> nu_mu)".into(), pmu),
            ("MSW P(-> nu_tau)".into(), ptau),
            ("Vacuum P(-> nu_e)".into(), pe_vac),
        ]
    }

    fn qcd_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let s = &self.state;
        let pole = self.landau_pole_scale_gev().max(0.05);
        let q_min = (1.06 * pole).max(0.1);
        let log_min = q_min.log10();
        let log_max = 3.0_f64; // 1000 GeV
        let samples = 100usize;
        let mut curve = Vec::with_capacity(samples);
        for i in 0..samples {
            let log_q = log_min + (log_max - log_min) * (i as f64) / ((samples - 1) as f64);
            let q = 10.0_f64.powf(log_q);
            if let Ok(alpha) = running_coupling(MZ_REF_GEV, s.alpha_s_mz, q, s.active_flavors) {
                curve.push([log_q, alpha.min(3.5)]);
            }
        }
        let pole_log = pole.log10();
        vec![
            ("1-Loop alpha_s(log10 Q)".into(), curve),
            (
                "Infrared Landau Pole Boundary".into(),
                vec![[pole_log, 0.0], [pole_log, 3.5]],
            ),
        ]
    }

    fn ckm_triangle_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let obs = self.evaluate_observables();
        let (rho, eta) = obs.ckm_apex;
        vec![
            (
                "Unitarity Triangle".into(),
                vec![[0.0, 0.0], [1.0, 0.0], [rho, eta], [0.0, 0.0]],
            ),
            ("Real Baseline [0, 1]".into(), vec![[0.0, 0.0], [1.0, 0.0]]),
        ]
    }
}
