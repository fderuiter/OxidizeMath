#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

pub use super::dist_types::{DistFamily, DistMetrics, DistributionsState};
use super::render_tool_header;
use crate::accessibility::AccessibleHoverText;
use crate::app::SemanticPalette;
use crate::framework::{ExportCapabilities, InteractiveTool};
use eframe::egui;
use egui_plot::{Bar, BarChart, Legend, Line, LineStyle, Plot, PlotPoints, VLine};
use math_explorer::engine_prob::{
    wasserstein_1d, Bernoulli, Beta, Binomial, ChiSquared, ContinuousDist, DiscreteDist,
    Exponential, Gamma, Normal, Poisson, StudentsT, Uniform,
};
use math_explorer::rng::OxidizeRng;

/// Interactive explorer for continuous & discrete distributions in `engine_prob`.
pub struct DistributionsExplorerTool {
    state: DistributionsState,
    metrics: DistMetrics,
    revision: u64,
    density_curve: Vec<[f64; 2]>,
    cdf_curve: Vec<[f64; 2]>,
    hist_bars: Vec<[f64; 3]>,
}

impl Default for DistributionsExplorerTool {
    fn default() -> Self {
        let mut tool = Self {
            state: DistributionsState::default(),
            metrics: DistMetrics::default(),
            revision: 1,
            density_curve: Vec::new(),
            cdf_curve: Vec::new(),
            hist_bars: Vec::new(),
        };
        tool.recompute_all();
        tool
    }
}

impl DistributionsExplorerTool {
    #[cfg(test)]
    pub(super) fn metrics(&self) -> DistMetrics {
        self.metrics
    }

    #[cfg(test)]
    pub(super) fn set_family_for_test(&mut self, family: DistFamily) {
        self.state.family = family;
        self.recompute_all();
    }

    fn recompute_all(&mut self) {
        self.density_curve.clear();
        self.cdf_curve.clear();
        self.hist_bars.clear();
        if self.state.family.is_discrete() {
            self.recompute_discrete();
        } else {
            self.recompute_continuous();
        }
        self.revision = self.revision.wrapping_add(1);
    }

    fn recompute_continuous(&mut self) {
        let mut rng = OxidizeRng::new(self.state.seed);
        let n = self.state.sample_count.clamp(64, 4096);
        let q_prob = self.state.quantile_prob.clamp(0.001, 0.999);
        match self.state.family {
            DistFamily::Normal => {
                let mu = self.state.param_a.clamp(-10.0, 10.0);
                let sig = self.state.param_b.clamp(0.1, 8.0);
                let d = Normal::new(mu, sig).unwrap();
                self.populate_cont_curves(&d, mu - 4.0 * sig, mu + 4.0 * sig, q_prob, n, &mut rng);
            }
            DistFamily::Gamma => {
                let k = self.state.param_a.clamp(0.3, 15.0);
                let th = self.state.param_b.clamp(0.1, 5.0);
                let d = Gamma::new(k, th).unwrap();
                let hi = d.quantile(0.995).unwrap_or(k * th * 3.0).max(0.5);
                self.populate_cont_curves(&d, 1.0e-3, hi, q_prob, n, &mut rng);
            }
            DistFamily::Beta => {
                let a = self.state.param_a.clamp(0.3, 15.0);
                let b = self.state.param_b.clamp(0.3, 15.0);
                let d = Beta::new(a, b).unwrap();
                self.populate_cont_curves(&d, 0.002, 0.998, q_prob, n, &mut rng);
            }
            DistFamily::StudentsT => {
                let nu = self.state.param_a.clamp(2.1, 30.0);
                let d = StudentsT::new(nu).unwrap();
                self.populate_cont_curves(&d, -5.0, 5.0, q_prob, n, &mut rng);
            }
            DistFamily::ChiSquared => {
                let nu = self.state.param_a.clamp(1.0, 25.0);
                let d = ChiSquared::new(nu).unwrap();
                let hi = d.quantile(0.995).unwrap_or(nu * 3.0).max(2.0);
                self.populate_cont_curves(&d, 0.02, hi, q_prob, n, &mut rng);
            }
            DistFamily::Exponential => {
                let rate = self.state.param_a.clamp(0.1, 10.0);
                let d = Exponential::new(rate).unwrap();
                let hi = d.quantile(0.995).unwrap_or(5.0 / rate);
                self.populate_cont_curves(&d, 0.0, hi, q_prob, n, &mut rng);
            }
            _ => {
                let lo = self.state.param_a.clamp(-10.0, 10.0);
                let w = self.state.param_b.clamp(0.2, 12.0);
                let d = Uniform::new(lo, lo + w).unwrap();
                self.populate_cont_curves(&d, lo - 0.15 * w, lo + 1.15 * w, q_prob, n, &mut rng);
            }
        }
    }

    fn populate_cont_curves<D: ContinuousDist>(
        &mut self,
        dist: &D,
        x_min: f64,
        x_max: f64,
        q_prob: f64,
        n: usize,
        rng: &mut OxidizeRng,
    ) {
        let pts = 160_usize;
        for i in 0..pts {
            let x = x_min + (x_max - x_min) * (i as f64) / ((pts - 1) as f64);
            self.density_curve.push([x, dist.pdf(x)]);
            self.cdf_curve.push([x, dist.cdf(x)]);
        }
        let samples: Vec<f64> = (0..n).map(|_| dist.sample(rng)).collect();
        let bins = 24_usize;
        let width = ((x_max - x_min) / (bins as f64)).max(1.0e-6);
        let mut counts = vec![0_usize; bins];
        for &s in &samples {
            if s >= x_min && s <= x_max {
                let idx = (((s - x_min) / width) as usize).min(bins - 1);
                counts[idx] += 1;
            }
        }
        for (idx, &cnt) in counts.iter().enumerate() {
            let cx = x_min + ((idx as f64) + 0.5) * width;
            let dens = (cnt as f64) / ((n as f64) * width);
            self.hist_bars.push([cx, dens, width * 0.88]);
        }
        let half = n / 2;
        let split_w1 = wasserstein_1d(&samples[..half], &samples[half..]).unwrap_or(0.0);
        self.metrics = DistMetrics {
            mean: dist.mean(),
            variance: dist.variance(),
            quantile_x: dist.quantile(q_prob).unwrap_or(0.0),
            sample_mean: samples.iter().sum::<f64>() / (n as f64),
            split_w1,
        };
    }

    fn recompute_discrete(&mut self) {
        let mut rng = OxidizeRng::new(self.state.seed);
        let n_samp = self.state.sample_count.clamp(64, 4096);
        let q_prob = self.state.quantile_prob.clamp(0.001, 0.999);
        match self.state.family {
            DistFamily::Poisson => {
                let lam = self.state.param_a.clamp(0.2, 25.0);
                let d = Poisson::new(lam).unwrap();
                let k_max = (lam + 4.0 * lam.sqrt() + 6.0) as u64;
                self.populate_disc_curves(&d, k_max, q_prob, n_samp, &mut rng);
            }
            DistFamily::Binomial => {
                let n_trials = (self.state.param_a.round().clamp(1.0, 40.0)) as u64;
                let p = self.state.param_b.clamp(0.02, 0.98);
                let d = Binomial::new(n_trials, p).unwrap();
                self.populate_disc_curves(&d, n_trials, q_prob, n_samp, &mut rng);
            }
            _ => {
                let p = self.state.param_a.clamp(0.02, 0.98);
                let d = Bernoulli::new(p).unwrap();
                self.populate_disc_curves(&d, 1, q_prob, n_samp, &mut rng);
            }
        }
    }

    fn populate_disc_curves<D: DiscreteDist>(
        &mut self,
        dist: &D,
        k_max: u64,
        q_prob: f64,
        n_samp: usize,
        rng: &mut OxidizeRng,
    ) {
        let mut counts = vec![0_usize; (k_max as usize) + 1];
        let mut samples = Vec::with_capacity(n_samp);
        for _ in 0..n_samp {
            let k = dist.sample(rng);
            samples.push(k as f64);
            if let Some(slot) = counts.get_mut(k as usize) {
                *slot += 1;
            }
        }
        for k in 0..=k_max {
            let x = k as f64;
            self.density_curve.push([x, dist.pmf(k)]);
            self.cdf_curve.push([x, dist.cdf(k)]);
            let emp = (counts[k as usize] as f64) / (n_samp as f64);
            self.hist_bars.push([x, emp, 0.38]);
        }
        let half = n_samp / 2;
        let split_w1 = wasserstein_1d(&samples[..half], &samples[half..]).unwrap_or(0.0);
        self.metrics = DistMetrics {
            mean: dist.mean(),
            variance: dist.variance(),
            quantile_x: dist.quantile(q_prob).unwrap_or(0) as f64,
            sample_mean: samples.iter().sum::<f64>() / (n_samp as f64),
            split_w1,
        };
    }

    fn render_controls(&mut self, ui: &mut egui::Ui) {
        let mut changed = false;
        egui::ComboBox::from_label("Family")
            .selected_text(self.state.family.label())
            .show_ui(ui, |ui| {
                for fam in DistFamily::ALL {
                    if ui
                        .selectable_value(&mut self.state.family, fam, fam.label())
                        .changed()
                    {
                        changed = true;
                    }
                }
            });
        ui.separator();
        changed |= self.render_param_sliders(ui);
        changed |= ui
            .add(egui::Slider::new(&mut self.state.quantile_prob, 0.001..=0.999).text("Quantile q"))
            .accessible_hover_text("Target cumulative probability q for inverse CDF F^-1(q)")
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.state.sample_count, 64..=4096).text("Samples N"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.state.seed, 1..=9999).text("RNG Seed"))
            .changed();
        changed |= ui
            .checkbox(&mut self.state.show_cdf, "Overlay CDF F(x)")
            .changed();
        if changed {
            self.recompute_all();
        }
    }

    fn render_param_sliders(&mut self, ui: &mut egui::Ui) -> bool {
        let ((label_a, range_a), second) = self.state.family.slider_specs();
        let mut c = ui
            .add(egui::Slider::new(&mut self.state.param_a, range_a).text(label_a))
            .changed();
        if let Some((label_b, range_b)) = second {
            c |= ui
                .add(egui::Slider::new(&mut self.state.param_b, range_b).text(label_b))
                .changed();
        }
        c
    }

    fn render_plot(&self, ui: &mut egui::Ui) {
        let palette = SemanticPalette::from_visuals(ui.visuals());
        let m = self.metrics;
        ui.horizontal_wrapped(|ui| {
            ui.colored_label(palette.series[0], format!("E[X]: {:.4}", m.mean));
            ui.separator();
            ui.colored_label(palette.series[1], format!("Var(X): {:.4}", m.variance));
            ui.separator();
            ui.colored_label(
                palette.accent,
                format!("F⁻¹({:.3}) = {:.5}", self.state.quantile_prob, m.quantile_x),
            );
            ui.separator();
            ui.colored_label(palette.success, format!("MC Mean: {:.4}", m.sample_mean));
            ui.separator();
            ui.colored_label(palette.muted_text, format!("Split W₁: {:.4}", m.split_w1));
        });
        ui.add_space(4.0);

        Plot::new(ui.id().with("prob_distributions_plot"))
            .legend(Legend::default())
            .x_axis_label("Support x")
            .y_axis_label("Density / Mass / Cumulative Probability")
            .show(ui, |plot_ui| {
                let bars: Vec<Bar> = self
                    .hist_bars
                    .iter()
                    .map(|&[x, h, w]| Bar::new(x, h).width(w))
                    .collect();
                plot_ui.bar_chart(
                    BarChart::new("OxidizeRng Histogram", bars)
                        .color(palette.series[1].gamma_multiply(0.45)),
                );
                plot_ui.line(
                    Line::new(
                        if self.state.family.is_discrete() {
                            "Exact PMF P(X = k)"
                        } else {
                            "Exact PDF f(x)"
                        },
                        PlotPoints::from(self.density_curve.clone()),
                    )
                    .color(palette.series[0])
                    .width(2.5_f32),
                );
                if self.state.show_cdf {
                    plot_ui.line(
                        Line::new("Exact CDF F(x)", PlotPoints::from(self.cdf_curve.clone()))
                            .color(palette.success)
                            .style(LineStyle::Dashed { length: 7.0_f32 })
                            .width(1.9_f32),
                    );
                }
                plot_ui.vline(
                    VLine::new("Exact Quantile F⁻¹(q)", m.quantile_x)
                        .color(palette.accent)
                        .width(2.0_f32),
                );
            });
    }
}

impl InteractiveTool for DistributionsExplorerTool {
    fn name(&self) -> &'static str {
        "Distributions Explorer"
    }
    fn domain(&self) -> &'static str {
        "probability"
    }
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }
    fn show_inside(&mut self, parent_ui: &mut egui::Ui) {
        let panel_id = crate::framework::scoped_panel_id(parent_ui, "prob_dist_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            render_tool_header(ui, "Distributions Explorer");
            self.render_controls(ui);
        });
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            self.render_plot(ui);
        });
    }
    fn state_revision(&self) -> u64 {
        self.revision
    }
    fn has_finite_parameters(&self) -> bool {
        self.state.validate().is_ok()
    }
    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&self.state).ok()
    }
    fn try_load_state(&mut self, json: &str) -> Result<(), String> {
        self.state_bounds().validate_json(json)?;
        let parsed: DistributionsState =
            serde_json::from_str(json).map_err(|e| format!("Invalid DistributionsState: {e}"))?;
        parsed.validate()?;
        self.state = parsed;
        self.recompute_all();
        Ok(())
    }
    fn clamp_parameters(&mut self) -> bool {
        self.state.param_a = self.state.param_a.clamp(-10.0, 40.0);
        self.state.param_b = self.state.param_b.clamp(0.05, 20.0);
        self.state.quantile_prob = self.state.quantile_prob.clamp(0.001, 0.999);
        self.state.sample_count = self.state.sample_count.clamp(64, 4096);
        self.recompute_all();
        true
    }
    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }
    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        vec![
            ("Exact Density / PMF".into(), self.density_curve.clone()),
            ("Exact CDF".into(), self.cdf_curve.clone()),
        ]
    }
    fn step_simulation(&mut self) {
        self.state.seed = (self.state.seed % 9999) + 1;
        self.recompute_all();
    }
}

// [cite:probability]
