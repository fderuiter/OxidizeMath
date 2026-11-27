//! `egui_plot` visualizations and posterior heatmaps for [`super::MarkovSequenceWorkbenchTool`].
// [cite:clinical_trials_statistics]

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{CtmcPlotView, DtmcPlotView, HmmPlotView, MarkovSequenceWorkbenchTool};
use crate::app::{SemanticPalette, ThemeMode};
use crate::presets::MarkovWorkbenchMode;
use eframe::egui;
use egui_plot::{Bar, BarChart, HLine, Legend, Line, LineStyle, Plot, Points, Polygon, VLine};
use math_commons::math_kernel::colormap::{heatmap_color, heatmap_color_high_contrast};

fn posterior_cell_color(prob: f64, high_contrast: bool) -> egui::Color32 {
    let rgb = if high_contrast {
        heatmap_color_high_contrast(prob)
    } else {
        heatmap_color(prob)
    };
    egui::Color32::from_rgba_unmultiplied(rgb[0], rgb[1], rgb[2], 185)
}

impl MarkovSequenceWorkbenchTool {
    fn show_plot_lens_selector(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| match self.config.mode {
            MarkovWorkbenchMode::Hmm => {
                for (view, label) in [
                    (
                        HmmPlotView::PosteriorHeatmap,
                        "Posterior Heatmap + Viterbi Path",
                    ),
                    (HmmPlotView::PosteriorLines, "γ_t(i) Posterior Trajectories"),
                    (HmmPlotView::LogLikelihoodCurve, "Baum-Welch Log-Likelihood"),
                ] {
                    ui.selectable_value(&mut self.hmm_plot_view, view, label);
                }
            }
            MarkovWorkbenchMode::Ctmc => {
                for (view, label) in [
                    (CtmcPlotView::TransitionSweep, "P(t) = exp(Gt) Sweep"),
                    (
                        CtmcPlotView::MttfAndSteadyState,
                        "MTTF & Steady-State Metrics",
                    ),
                ] {
                    ui.selectable_value(&mut self.ctmc_plot_view, view, label);
                }
            }
            MarkovWorkbenchMode::AbsorbingDtmc => {
                for (view, label) in [
                    (
                        DtmcPlotView::EpvAndAbsorption,
                        "EPV & Absorption Probabilities",
                    ),
                    (
                        DtmcPlotView::FundamentalVisits,
                        "Fundamental Matrix N = (I - Q)⁻¹",
                    ),
                ] {
                    ui.selectable_value(&mut self.dtmc_plot_view, view, label);
                }
            }
        });
    }

    pub(super) fn show_central_plot(&mut self, ui: &mut egui::Ui) {
        self.show_plot_lens_selector(ui);
        let palette = SemanticPalette::from_visuals(ui.visuals());
        let high_contrast = ThemeMode::from_visuals(ui.visuals()) == ThemeMode::HighContrast;
        Plot::new("markov_workbench_primary_plot")
            .legend(Legend::default())
            .allow_drag(true)
            .allow_zoom(true)
            .show(ui, |plot_ui| match self.config.mode {
                MarkovWorkbenchMode::Hmm => match self.hmm_plot_view {
                    HmmPlotView::PosteriorHeatmap => {
                        self.plot_hmm_heatmap(plot_ui, palette, high_contrast)
                    }
                    HmmPlotView::PosteriorLines => self.plot_hmm_posterior_lines(plot_ui, palette),
                    HmmPlotView::LogLikelihoodCurve => self.plot_bw_history(plot_ui, palette),
                },
                MarkovWorkbenchMode::Ctmc => match self.ctmc_plot_view {
                    CtmcPlotView::TransitionSweep => self.plot_ctmc_sweep(plot_ui, palette),
                    CtmcPlotView::MttfAndSteadyState => self.plot_ctmc_bars(plot_ui, palette),
                },
                MarkovWorkbenchMode::AbsorbingDtmc => match self.dtmc_plot_view {
                    DtmcPlotView::EpvAndAbsorption => self.plot_dtmc_epv(plot_ui, palette),
                    DtmcPlotView::FundamentalVisits => self.plot_dtmc_fundamental(plot_ui, palette),
                },
            });
    }

    fn plot_hmm_heatmap(
        &self,
        plot_ui: &mut egui_plot::PlotUi<'_>,
        palette: SemanticPalette,
        high_contrast: bool,
    ) {
        let t_len = self.config.hmm.observations.len().max(1);
        let n_states = self.config.hmm.state_names.len();
        let stride = (t_len / 200).max(1);

        for s in 0..n_states {
            let Some(row) = self.hmm_out.posteriors.get(s) else {
                continue;
            };
            let mut t = 0;
            while t < t_len {
                let end_t = (t + stride).min(t_len);
                let avg_p = row[t..end_t].iter().sum::<f64>() / (end_t - t) as f64;
                let (x0, x1) = (t as f64 - 0.5, end_t as f64 - 0.5);
                let (y0, y1) = (s as f64, (s + 1) as f64);
                let cell = Polygon::new("", vec![[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
                    .fill_color(posterior_cell_color(avg_p, high_contrast))
                    .stroke(egui::Stroke::NONE);
                plot_ui.polygon(cell);
                t = end_t;
            }
        }

        let vit_pts: Vec<[f64; 2]> = self
            .hmm_out
            .viterbi_path
            .iter()
            .enumerate()
            .step_by(stride)
            .map(|(t, &st)| [t as f64, st as f64 + 0.5])
            .collect();
        plot_ui.line(
            Line::new("Viterbi Decoded State (s + 0.5)", vit_pts.clone())
                .color(palette.accent)
                .width(2.6_f32),
        );
        if t_len <= 120 {
            plot_ui.points(
                Points::new("Viterbi Steps", vit_pts)
                    .color(palette.node_stroke)
                    .radius(3.8_f32),
            );
        }
    }

    fn plot_hmm_posterior_lines(
        &self,
        plot_ui: &mut egui_plot::PlotUi<'_>,
        palette: SemanticPalette,
    ) {
        let t_len = self.config.hmm.observations.len().max(1);
        let stride = (t_len / 500).max(1);
        for (s, state_name) in self.config.hmm.state_names.iter().enumerate() {
            if let Some(row) = self.hmm_out.posteriors.get(s) {
                let pts: Vec<[f64; 2]> = row
                    .iter()
                    .enumerate()
                    .step_by(stride)
                    .map(|(t, &p)| [t as f64, p])
                    .collect();
                plot_ui.line(
                    Line::new(format!("P({state_name} | Y)"), pts)
                        .color(palette.series_color(s))
                        .width(2.1_f32),
                );
            }
        }
        let denom = (self.config.hmm.state_names.len().saturating_sub(1)).max(1) as f64;
        let vit_scaled: Vec<[f64; 2]> = self
            .hmm_out
            .viterbi_path
            .iter()
            .enumerate()
            .step_by(stride)
            .map(|(t, &st)| [t as f64, st as f64 / denom])
            .collect();
        plot_ui.line(
            Line::new("Normalized Viterbi Path", vit_scaled)
                .color(palette.accent)
                .style(LineStyle::dashed_dense())
                .width(1.8_f32),
        );
    }

    fn plot_bw_history(&self, plot_ui: &mut egui_plot::PlotUi<'_>, palette: SemanticPalette) {
        let mut pts = Vec::with_capacity(self.hmm_out.bw_history.len() + 1);
        if let Some(rep) = self.hmm_out.last_bw_report {
            pts.push([0.0, rep.initial_log_likelihood]);
            for (idx, &ll) in self.hmm_out.bw_history.iter().enumerate() {
                pts.push([(idx + 1) as f64, ll]);
            }
        } else {
            pts.push([0.0, self.hmm_out.log_likelihood]);
            pts.push([1.0, self.hmm_out.log_likelihood]);
        }
        plot_ui.line(
            Line::new("ln P(Y | θ)", pts.clone())
                .color(palette.series_color(0))
                .width(2.4_f32),
        );
        plot_ui.points(
            Points::new("EM Iteration", pts)
                .color(palette.accent)
                .radius(4.5_f32),
        );
    }

    fn plot_ctmc_sweep(&self, plot_ui: &mut egui_plot::PlotUi<'_>, palette: SemanticPalette) {
        for (j, name) in self.config.ctmc.state_names.iter().enumerate() {
            if let Some(curve) = self.ctmc_out.sweep_probs.get(j) {
                let pts: Vec<[f64; 2]> = self
                    .ctmc_out
                    .sweep_times
                    .iter()
                    .zip(curve.iter())
                    .map(|(&t, &p)| [t, p])
                    .collect();
                plot_ui.line(
                    Line::new(format!("P(X(t) = {name})"), pts)
                        .color(palette.series_color(j))
                        .width(2.2_f32),
                );
            }
            if let Some(pi_val) = self.ctmc_out.steady_state.as_ref().and_then(|v| v.get(j)) {
                plot_ui.hline(
                    HLine::new(format!("π({name})"), *pi_val)
                        .color(palette.series_color(j))
                        .style(LineStyle::dashed_loose()),
                );
            }
        }
        plot_ui.vline(
            VLine::new("t_eval", self.config.ctmc.eval_time)
                .color(palette.accent)
                .style(LineStyle::dotted_dense()),
        );
    }

    fn plot_ctmc_bars(&self, plot_ui: &mut egui_plot::PlotUi<'_>, palette: SemanticPalette) {
        if let Some(mttf) = &self.ctmc_out.mttf {
            let bars: Vec<Bar> = mttf
                .iter()
                .enumerate()
                .map(|(idx, &val)| {
                    let st = self.config.ctmc.transient_states[idx];
                    Bar::new(idx as f64, val)
                        .name(format!("MTTF ({})", self.config.ctmc.state_names[st]))
                        .fill(palette.series_color(idx))
                })
                .collect();
            plot_ui.bar_chart(BarChart::new("Mean Time To Failure (MTTF)", bars).width(0.55));
        } else if let Some(pi) = &self.ctmc_out.steady_state {
            let bars: Vec<Bar> = pi
                .iter()
                .enumerate()
                .map(|(i, &val)| {
                    Bar::new(i as f64, val)
                        .name(format!("π({})", self.config.ctmc.state_names[i]))
                        .fill(palette.series_color(i))
                })
                .collect();
            plot_ui.bar_chart(BarChart::new("Steady-State π", bars).width(0.55));
        } else {
            let src = self.config.ctmc.source_state;
            let row = self
                .ctmc_out
                .eval_matrix
                .get(src)
                .cloned()
                .unwrap_or_default();
            let pts: Vec<[f64; 2]> = row
                .into_iter()
                .enumerate()
                .map(|(j, p)| [j as f64, p])
                .collect();
            plot_ui.line(Line::new("P(t_eval)", pts).color(palette.series_color(0)));
        }
    }

    fn plot_dtmc_epv(&self, plot_ui: &mut egui_plot::PlotUi<'_>, palette: SemanticPalette) {
        let epv_bars: Vec<Bar> = self
            .dtmc_out
            .epv
            .iter()
            .enumerate()
            .map(|(r, &val)| {
                let st = self.dtmc_out.transient_indices[r];
                Bar::new(r as f64, val)
                    .name(format!("EPV ({})", self.config.dtmc.state_names[st]))
                    .fill(palette.series_color(0))
            })
            .collect();
        plot_ui.bar_chart(BarChart::new("Expected Possession Value (EPV)", epv_bars).width(0.45));

        for (k, &abs_i) in self.dtmc_out.absorbing_indices.iter().enumerate() {
            let pts: Vec<[f64; 2]> = self
                .dtmc_out
                .absorption_probs
                .iter()
                .enumerate()
                .map(|(r, row)| [r as f64, row.get(k).copied().unwrap_or(0.0)])
                .collect();
            plot_ui.line(
                Line::new(
                    format!("P(Absorb → {})", self.config.dtmc.state_names[abs_i]),
                    pts.clone(),
                )
                .color(palette.series_color(k + 1))
                .width(2.1_f32),
            );
            plot_ui.points(
                Points::new(
                    format!("B[:, {}]", self.config.dtmc.state_names[abs_i]),
                    pts,
                )
                .color(palette.series_color(k + 1))
                .radius(4.5_f32),
            );
        }
    }

    fn plot_dtmc_fundamental(&self, plot_ui: &mut egui_plot::PlotUi<'_>, palette: SemanticPalette) {
        for (r, &st_i) in self.dtmc_out.transient_indices.iter().enumerate() {
            if let Some(row) = self.dtmc_out.fundamental.get(r) {
                let pts: Vec<[f64; 2]> = row
                    .iter()
                    .enumerate()
                    .map(|(c, &visits)| [c as f64, visits])
                    .collect();
                plot_ui.line(
                    Line::new(
                        format!("Visits from {}", self.config.dtmc.state_names[st_i]),
                        pts.clone(),
                    )
                    .color(palette.series_color(r))
                    .width(2.1_f32),
                );
                plot_ui.points(
                    Points::new(format!("N[{}, :]", self.config.dtmc.state_names[st_i]), pts)
                        .color(palette.series_color(r))
                        .radius(4.2_f32),
                );
            }
        }
    }
}
