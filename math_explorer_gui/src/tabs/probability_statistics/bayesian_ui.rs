#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Side-panel controls, posterior + MCMC density overlays, and trace plots for [`BayesianInferenceTool`].

use super::bayesian_tool::{BayesianInferenceTool, ConjugateFamily, MAX_MCMC_STEPS_PER_FRAME};
use crate::accessibility::AccessibleHoverText;
use crate::framework::{PlaybackAction, PlaybackTransport};
use eframe::egui;
use egui_plot::{Bar, BarChart, ExportPrimitive, Legend, Line, Plot, PlotPoints, VLine};

const MCMC_HIST_BINS: usize = 22;

impl BayesianInferenceTool {
    fn select_conjugate_family(&mut self, family: ConjugateFamily) {
        if self.config.family == family {
            return;
        }
        self.config.family = family;
        match family {
            ConjugateFamily::BetaBinomial => {
                self.config.prior_a = 2.0;
                self.config.prior_b = 2.0;
                self.config.obs_stat = 14.0;
                self.config.obs_count = 20;
                self.config.proposal_scale = 0.14;
            }
            ConjugateFamily::GammaPoisson => {
                self.config.prior_a = 3.0;
                self.config.prior_b = 1.5;
                self.config.obs_stat = 12.0;
                self.config.obs_count = 15;
                self.config.proposal_scale = 0.35;
            }
            ConjugateFamily::NormalNormal => {
                self.config.prior_a = 0.0;
                self.config.prior_b = 1.5;
                self.config.obs_stat = 1.4;
                self.config.obs_count = 12;
                self.config.proposal_scale = 0.32;
            }
            ConjugateFamily::NormalInverseGamma => {
                self.config.prior_a = 0.0;
                self.config.prior_b = 2.5;
                self.config.obs_stat = 1.8;
                self.config.obs_count = 16;
                self.config.proposal_scale = 0.38;
            }
        }
        self.rebuild_chain();
    }

    fn show_mcmc_transport(&mut self, ui: &mut egui::Ui) {
        ui.label(egui::RichText::new("Zero-Allocation MCMC Transport").strong());
        if let Some(action) = PlaybackTransport::in_thread(
            "bayesian_mcmc_transport",
            "Bayesian Inference Lab",
            self.is_playing,
        )
        .show(ui)
        {
            match action {
                PlaybackAction::Play => {
                    self.is_playing = true;
                    self.bump_revision();
                }
                PlaybackAction::Pause => {
                    self.is_playing = false;
                    self.bump_revision();
                }
                PlaybackAction::Step => {
                    self.is_playing = false;
                    self.step_mcmc_draws(0, 16);
                    self.bump_revision();
                }
                PlaybackAction::Reset => {
                    self.is_playing = false;
                    self.rebuild_chain();
                }
            }
        }
    }

    fn show_bayesian_prior_sliders(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        let is_positive_prior = matches!(
            self.config.family,
            ConjugateFamily::BetaBinomial | ConjugateFamily::GammaPoisson
        );
        let a_range = if is_positive_prior { 0.3..=12.0 } else { -4.0..=6.0 };
        let b_range = if self.config.family == ConjugateFamily::NormalInverseGamma {
            1.2..=10.0
        } else {
            0.3..=10.0
        };
        let mut pa = self.config.prior_a.clamp(*a_range.start(), *a_range.end());
        if ui
            .add(egui::Slider::new(&mut pa, a_range).text("Prior Param A"))
            .accessible_hover_text("Primary conjugate prior hyperparameter")
            .changed()
        {
            self.config.prior_a = pa;
            changed = true;
        }
        let mut pb = self.config.prior_b.clamp(*b_range.start(), *b_range.end());
        if ui
            .add(egui::Slider::new(&mut pb, b_range).text("Prior Param B"))
            .accessible_hover_text("Secondary conjugate prior hyperparameter")
            .changed()
        {
            self.config.prior_b = pb;
            changed = true;
        }
        let mut cnt = self.config.obs_count;
        if ui
            .add(egui::Slider::new(&mut cnt, 2..=80).text("Sample Count n"))
            .accessible_hover_text("Observed trial or sample count n")
            .changed()
        {
            self.config.obs_count = cnt;
            changed = true;
        }
        changed
    }

    fn show_bayesian_sliders(&mut self, ui: &mut egui::Ui) {
        let mut changed = self.show_bayesian_prior_sliders(ui);
        let stat_range = if self.config.family == ConjugateFamily::BetaBinomial {
            0.0..=(self.config.obs_count as f64)
        } else {
            -3.0..=20.0
        };
        let mut obs = self.config.obs_stat.clamp(*stat_range.start(), *stat_range.end());
        if ui
            .add(egui::Slider::new(&mut obs, stat_range).text("Observed Stat"))
            .accessible_hover_text("Observed successes k, count scale, or sample mean")
            .changed()
        {
            self.config.obs_stat = obs;
            changed = true;
        }
        let mut cred = self.config.credible_level;
        if ui
            .add(egui::Slider::new(&mut cred, 0.50..=0.99).text("Credible Level"))
            .accessible_hover_text("Posterior credible interval mass (e.g. 0.95)")
            .changed()
        {
            self.config.credible_level = cred;
            changed = true;
        }
        let mut prop = self.config.proposal_scale;
        if ui
            .add(egui::Slider::new(&mut prop, 0.03..=1.50).text("MH Proposal SD"))
            .accessible_hover_text("Random-walk Metropolis-Hastings Gaussian proposal scale")
            .changed()
        {
            self.config.proposal_scale = prop;
            changed = true;
        }
        let mut spf = self.config.steps_per_frame;
        if ui
            .add(egui::Slider::new(&mut spf, 1..=MAX_MCMC_STEPS_PER_FRAME).text("Draws / Frame"))
            .accessible_hover_text("MCMC transitions executed per live frame")
            .changed()
        {
            self.config.steps_per_frame = spf;
            self.bump_revision();
        }
        if changed {
            self.rebuild_chain();
        }
    }

    fn show_bayesian_readout(&self, ui: &mut egui::Ui) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        let a = &self.analytical;
        ui.group(|ui| {
            ui.label(egui::RichText::new("Conjugate vs MCMC Summary").strong());
            ui.label(format!(
                "Exact Mean: {:.4} (SD {:.4})",
                a.posterior_mean, a.posterior_std
            ));
            ui.label(format!(
                "MCMC Mean:  {:.4} (SD {:.4})",
                self.mcmc_summary.mean,
                self.mcmc_summary.variance.sqrt()
            ));
            ui.colored_label(
                palette.info,
                format!(
                    "Equal-Tailed {:.0}%: [{:.4}, {:.4}]",
                    self.config.credible_level * 100.0,
                    a.equal_tailed.lower,
                    a.equal_tailed.upper
                ),
            );
            ui.colored_label(
                palette.success,
                format!(
                    "Shortest HPD {:.0}%: [{:.4}, {:.4}]",
                    self.config.credible_level * 100.0,
                    a.hpd.lower,
                    a.hpd.upper
                ),
            );
            ui.label(format!(
                "Acceptance: {:.1}% | ESS: {:.0} ({} draws)",
                self.mcmc_summary.acceptance_rate * 100.0,
                self.mcmc_summary.effective_sample_size,
                self.chain_len
            ));
        });
        ui.add_space(4.0);
        ui.group(|ui| {
            ui.label(egui::RichText::new("Jeffreys Model Selection").strong());
            ui.colored_label(
                palette.accent,
                format!(
                    "ln BF10 = {:.3} (BF10 = {:.3})",
                    a.bayes_factor.ln_bf10,
                    a.bayes_factor.bf10.min(99_999.0)
                ),
            );
            ui.label(format!("Evidence: {}", self.jeffreys_label()));
        });
    }

    pub(crate) fn render_workbench(&mut self, parent_ui: &mut egui::Ui) {
        let side_id = crate::framework::scoped_panel_id(parent_ui, "bayesian_lab_controls");
        crate::framework::PaneSidePanel::left(side_id).show_inside(parent_ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("Bayesian Inference Lab");
                ui.separator();
                ui.label(egui::RichText::new("Conjugate Model").strong());
                ui.horizontal_wrapped(|ui| {
                    for fam in ConjugateFamily::ALL {
                        if ui
                            .selectable_label(self.config.family == fam, fam.label())
                            .accessible_hover_text(format!("Select {} model", fam.label()))
                            .clicked()
                        {
                            self.select_conjugate_family(fam);
                        }
                    }
                });
                ui.separator();
                self.show_mcmc_transport(ui);
                ui.separator();
                self.show_bayesian_sliders(ui);
                ui.separator();
                self.show_bayesian_readout(ui);
            });
        });

        let bottom_id = crate::framework::scoped_panel_id(parent_ui, "bayesian_trace_panel");
        egui::TopBottomPanel::bottom(bottom_id)
            .resizable(true)
            .default_height(195.0)
            .min_height(130.0)
            .show_inside(parent_ui, |ui| {
                self.render_trace_plot(ui);
            });

        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            self.render_posterior_plot(ui);
        });
    }

    fn mcmc_histogram_bars(&self) -> Vec<[f64; 3]> {
        let (lo, hi) = self.analytical.plot_bounds;
        let active = &self.chain_buffer[..self.chain_len];
        if active.is_empty() || hi <= lo {
            return Vec::new();
        }
        let width = (hi - lo) / (MCMC_HIST_BINS as f64);
        let mut counts = [0_usize; MCMC_HIST_BINS];
        for &theta in active {
            if theta >= lo && theta <= hi {
                let idx = (((theta - lo) / width).floor() as usize).min(MCMC_HIST_BINS - 1);
                counts[idx] += 1;
            }
        }
        let denom = (active.len() as f64) * width;
        (0..MCMC_HIST_BINS)
            .map(|b| [lo + (b as f64 + 0.5) * width, (counts[b] as f64) / denom, width * 0.88])
            .collect()
    }

    fn render_posterior_plot(&self, ui: &mut egui::Ui) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        ui.label(
            egui::RichText::new(format!(
                "{} — Prior, Exact Conjugate Posterior, HPD Interval & MCMC Density",
                self.config.family.label()
            ))
            .strong(),
        );
        let plot_id = crate::framework::scoped_panel_id(ui, "bayesian_posterior_plot");
        Plot::new(plot_id)
            .legend(Legend::default())
            .x_axis_label("Parameter theta")
            .y_axis_label("Density")
            .include_y(0.0)
            .show(ui, |plot_ui| {
                let bars: Vec<Bar> = self
                    .mcmc_histogram_bars()
                    .into_iter()
                    .map(|[cx, h, w]| Bar::new(cx, h).width(w).fill(palette.info.gamma_multiply(0.42)))
                    .collect();
                if !bars.is_empty() {
                    plot_ui.bar_chart(BarChart::new("MCMC Empirical Density", bars));
                }
                plot_ui.line(
                    Line::new("Prior p(theta)", PlotPoints::from(self.analytical.prior_curve.clone()))
                        .color(palette.warning)
                        .width(1.8_f32),
                );
                plot_ui.line(
                    Line::new(
                        "Exact Posterior p(theta|D)",
                        PlotPoints::from(self.analytical.posterior_curve.clone()),
                    )
                    .color(palette.accent)
                    .width(2.6_f32),
                );
                plot_ui.vline(VLine::new("HPD Lower", self.analytical.hpd.lower).color(palette.success));
                plot_ui.vline(VLine::new("HPD Upper", self.analytical.hpd.upper).color(palette.success));
            });
    }

    fn trace_points(&self) -> Vec<[f64; 2]> {
        self.chain_buffer[..self.chain_len]
            .iter()
            .enumerate()
            .map(|(idx, &v)| [idx as f64, v])
            .collect()
    }

    fn render_trace_plot(&self, ui: &mut egui::Ui) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        ui.label(egui::RichText::new("Metropolis-Hastings Chain Trace (Pre-Allocated Buffer)").strong());
        let plot_id = crate::framework::scoped_panel_id(ui, "bayesian_mcmc_trace_plot");
        Plot::new(plot_id)
            .legend(Legend::default())
            .x_axis_label("Retained Sample Index t")
            .y_axis_label("Parameter theta_t")
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new("MCMC Chain", PlotPoints::from(self.trace_points())).color(palette.info),
                );
            });
    }

    pub(crate) fn build_export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        vec![
            ("Prior Density".to_string(), self.analytical.prior_curve.clone()),
            (
                "Exact Posterior Density".to_string(),
                self.analytical.posterior_curve.clone(),
            ),
            ("MCMC Chain Trace".to_string(), self.trace_points()),
        ]
    }

    pub(crate) fn build_export_primitives(&self) -> Vec<ExportPrimitive> {
        let palette = crate::app::SemanticPalette::from_visuals(&egui::Visuals::dark());
        vec![
            ExportPrimitive::Polyline {
                name: "Prior Density".to_string(),
                points: self.analytical.prior_curve.clone(),
                color: palette.warning,
                stroke_width: 1.8,
                closed: false,
                fill: None,
            },
            ExportPrimitive::Polyline {
                name: "Exact Posterior Density".to_string(),
                points: self.analytical.posterior_curve.clone(),
                color: palette.accent,
                stroke_width: 2.4,
                closed: false,
                fill: None,
            },
        ]
    }
}

// [cite:clinical_trials_statistics]
