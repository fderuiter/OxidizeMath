#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::render_tool_header;
use crate::accessibility::AccessibleHoverText;
use crate::app::SemanticPalette;
use crate::framework::{ExportCapabilities, InteractiveTool};
use eframe::egui;
use egui_plot::{Legend, Line, LineStyle, Plot, PlotPoints, VLine};
use math_explorer::engine_prob::{
    BetaBinomial, ContinuousDist, GammaPoisson, NormalInverseGamma, NormalNormal,
};
use serde::{Deserialize, Serialize};

/// Supported zero-allocation conjugate Bayesian families.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConjugateFamily {
    /// $\text{Beta}(\alpha_0, \beta_0)$ prior with Binomial likelihood $(k, n)$.
    BetaBinomial,
    /// $\text{Gamma}(\alpha_0, \beta_0)$ rate prior with Poisson likelihood $(\sum x, E)$.
    GammaPoisson,
    /// $\mathcal{N}(\mu_0, \tau_0^2)$ mean prior with known $\sigma = 1.0$ Gaussian observations.
    NormalNormal,
    /// $\text{NIG}(\mu_0, \kappa_0, \alpha_0, \beta_0)$ joint $(\mu, \sigma^2)$ prior with Student-$t$ marginal.
    NormalInverseGamma,
}

impl ConjugateFamily {
    const ALL: [Self; 4] = [
        Self::BetaBinomial,
        Self::GammaPoisson,
        Self::NormalNormal,
        Self::NormalInverseGamma,
    ];

    fn label(self) -> &'static str {
        match self {
            Self::BetaBinomial => "Beta-Binomial (Proportion θ)",
            Self::GammaPoisson => "Gamma-Poisson (Rate λ)",
            Self::NormalNormal => "Normal-Normal (Mean μ, σ=1)",
            Self::NormalInverseGamma => "Normal-Inverse-Gamma (Student-t μ)",
        }
    }
}

/// Bounded serializable parameters for `ConjugateBayesianTool`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BayesianState {
    /// Active conjugate model family.
    pub family: ConjugateFamily,
    /// First prior hyperparameter ($\alpha_0$ or $\mu_0$).
    pub prior_a: f64,
    /// Second prior hyperparameter ($\beta_0$ or $\tau_0$).
    pub prior_b: f64,
    /// Sample count / exposure $n \in [1.0, 200.0]$.
    pub sample_size: f64,
    /// Sufficient statistic ($k / n$ proportion, event rate, or sample mean $\bar{x}$).
    pub observed_stat: f64,
    /// Equal-tailed credible level $1 - \alpha \in [0.50, 0.99]$.
    pub credible_level: f64,
}

impl Default for BayesianState {
    fn default() -> Self {
        Self {
            family: ConjugateFamily::BetaBinomial,
            prior_a: 2.0,
            prior_b: 2.0,
            sample_size: 24.0,
            observed_stat: 0.75,
            credible_level: 0.95,
        }
    }
}

impl BayesianState {
    fn validate(&self) -> Result<(), String> {
        let ok = self.prior_a.is_finite()
            && (-5.0..=25.0).contains(&self.prior_a)
            && self.prior_b.is_finite()
            && (0.2..=25.0).contains(&self.prior_b)
            && self.sample_size.is_finite()
            && (1.0..=200.0).contains(&self.sample_size)
            && self.observed_stat.is_finite()
            && (-5.0..=15.0).contains(&self.observed_stat)
            && self.credible_level.is_finite()
            && (0.50..=0.99).contains(&self.credible_level);
        if !ok {
            return Err("BayesianState parameter out of bounds".into());
        }
        Ok(())
    }
}

/// Exact analytical posterior and marginal-likelihood metrics.
#[derive(Clone, Copy, Debug, Default)]
pub struct BayesianMetrics {
    /// Prior mean.
    pub prior_mean: f64,
    /// Posterior mean.
    pub posterior_mean: f64,
    /// Posterior variance.
    pub posterior_variance: f64,
    /// Lower equal-tailed credible bound.
    pub ci_low: f64,
    /// Upper equal-tailed credible bound.
    pub ci_high: f64,
    /// Log marginal likelihood $\ln p(D \mid M_1)$.
    pub log_evidence: f64,
    /// Exact Bayes factor $BF_{10}$ against the canonical neutral baseline $M_0$.
    pub bayes_factor_10: f64,
}

/// Interactive zero-allocation conjugate Bayesian updater powered by `engine_prob`.
pub struct ConjugateBayesianTool {
    state: BayesianState,
    metrics: BayesianMetrics,
    revision: u64,
    prior_curve: Vec<[f64; 2]>,
    posterior_curve: Vec<[f64; 2]>,
}

impl Default for ConjugateBayesianTool {
    fn default() -> Self {
        let mut tool = Self {
            state: BayesianState::default(),
            metrics: BayesianMetrics::default(),
            revision: 1,
            prior_curve: Vec::new(),
            posterior_curve: Vec::new(),
        };
        tool.recompute_all();
        tool
    }
}

impl ConjugateBayesianTool {
    #[cfg(test)]
    pub(super) fn metrics(&self) -> BayesianMetrics {
        self.metrics
    }

    #[cfg(test)]
    pub(super) fn set_family_for_test(&mut self, family: ConjugateFamily) {
        self.state.family = family;
        self.recompute_all();
    }

    fn recompute_all(&mut self) {
        self.prior_curve.clear();
        self.posterior_curve.clear();
        match self.state.family {
            ConjugateFamily::BetaBinomial => self.recompute_beta_binomial(),
            ConjugateFamily::GammaPoisson => self.recompute_gamma_poisson(),
            ConjugateFamily::NormalNormal => self.recompute_normal_normal(),
            ConjugateFamily::NormalInverseGamma => self.recompute_nig(),
        }
        self.revision = self.revision.wrapping_add(1);
    }

    fn synth_gaussian_samples(n: usize, target_mean: f64, target_var: f64) -> Vec<f64> {
        if n <= 1 {
            return vec![target_mean; n.max(1)];
        }
        let mid = 0.5 * ((n - 1) as f64);
        let raw_ss: f64 = (0..n).map(|i| ((i as f64) - mid).powi(2)).sum();
        let scale = (((n - 1) as f64) * target_var / raw_ss.max(1.0e-12)).sqrt();
        (0..n)
            .map(|i| target_mean + ((i as f64) - mid) * scale)
            .collect()
    }

    fn recompute_beta_binomial(&mut self) {
        let a0 = self.state.prior_a.clamp(0.3, 25.0);
        let b0 = self.state.prior_b.clamp(0.3, 25.0);
        let n = (self.state.sample_size.round().clamp(1.0, 200.0)) as u64;
        let k = ((self.state.observed_stat.clamp(0.0, 1.0) * (n as f64)).round() as u64).min(n);
        let prior = BetaBinomial::new(a0, b0).unwrap();
        let post = prior.update(k, n - k);
        let null_prior = BetaBinomial::new(1.0, 1.0).unwrap();
        let (lo, hi) = post
            .credible_interval(self.state.credible_level)
            .unwrap_or((0.0, 1.0));
        let prior_dist = prior.posterior().unwrap();
        let post_dist = post.posterior().unwrap();
        for i in 0..160 {
            let x = 0.003 + 0.994 * (i as f64) / 159.0;
            self.prior_curve.push([x, prior_dist.pdf(x)]);
            self.posterior_curve.push([x, post_dist.pdf(x)]);
        }
        self.metrics = BayesianMetrics {
            prior_mean: prior.posterior_mean(),
            posterior_mean: post.posterior_mean(),
            posterior_variance: post.posterior_variance(),
            ci_low: lo,
            ci_high: hi,
            log_evidence: prior.log_marginal_likelihood(k, n).unwrap_or(0.0),
            bayes_factor_10: prior.bayes_factor(&null_prior, k, n).unwrap_or(1.0),
        };
    }

    fn recompute_gamma_poisson(&mut self) {
        let a0 = self.state.prior_a.clamp(0.5, 25.0);
        let b0 = self.state.prior_b.clamp(0.2, 25.0);
        let n_exp = (self.state.sample_size.round().clamp(1.0, 200.0)) as usize;
        let exposure = n_exp as f64;
        let rate_hat = self.state.observed_stat.clamp(0.1, 15.0);
        let total_k = (rate_hat * exposure).round().max(0.0) as u64;
        let base_c = total_k / (n_exp as u64);
        let rem_c = (total_k % (n_exp as u64)) as usize;
        let counts: Vec<u64> = (0..n_exp).map(|i| base_c + u64::from(i < rem_c)).collect();
        let prior = GammaPoisson::new(a0, b0).unwrap();
        let post = prior.update(total_k, exposure).unwrap();
        let null_prior = GammaPoisson::new(1.0, 1.0).unwrap();
        let (lo, hi) = post
            .credible_interval(self.state.credible_level)
            .unwrap_or((0.0, 10.0));
        let prior_dist = prior.posterior().unwrap();
        let post_dist = post.posterior().unwrap();
        let x_max = hi
            .max(prior_dist.quantile(0.99).unwrap_or(6.0))
            .mul_add(1.35, 0.5);
        for i in 0..160 {
            let x = 0.01 + x_max * (i as f64) / 159.0;
            self.prior_curve.push([x, prior_dist.pdf(x)]);
            self.posterior_curve.push([x, post_dist.pdf(x)]);
        }
        self.metrics = BayesianMetrics {
            prior_mean: prior.posterior_mean(),
            posterior_mean: post.posterior_mean(),
            posterior_variance: post.posterior_variance(),
            ci_low: lo,
            ci_high: hi,
            log_evidence: prior.log_marginal_likelihood(&counts).unwrap_or(0.0),
            bayes_factor_10: prior.bayes_factor(&null_prior, &counts).unwrap_or(1.0),
        };
    }

    fn recompute_normal_normal(&mut self) {
        let mu0 = self.state.prior_a.clamp(-5.0, 10.0);
        let tau0 = self.state.prior_b.clamp(0.2, 6.0);
        let n = (self.state.sample_size.round().clamp(1.0, 200.0)) as usize;
        let x_bar = self.state.observed_stat.clamp(-5.0, 10.0);
        let xs = Self::synth_gaussian_samples(n, x_bar, 0.85);
        let prior = NormalNormal::new(mu0, tau0, 1.0).unwrap();
        let post = prior.update_summary(n, x_bar).unwrap();
        let null_prior = NormalNormal::new(0.0, 2.0, 1.0).unwrap();
        let (lo, hi) = post
            .credible_interval(self.state.credible_level)
            .unwrap_or((-2.0, 2.0));
        let prior_d = prior.posterior().unwrap();
        let post_d = post.posterior().unwrap();
        let x_min = (mu0 - 3.5 * tau0).min(lo - 1.5);
        let x_max = (mu0 + 3.5 * tau0).max(hi + 1.5);
        for i in 0..160 {
            let x = x_min + (x_max - x_min) * (i as f64) / 159.0;
            self.prior_curve.push([x, prior_d.pdf(x)]);
            self.posterior_curve.push([x, post_d.pdf(x)]);
        }
        self.metrics = BayesianMetrics {
            prior_mean: prior.posterior_mean(),
            posterior_mean: post.posterior_mean(),
            posterior_variance: post.posterior_variance(),
            ci_low: lo,
            ci_high: hi,
            log_evidence: prior.log_marginal_likelihood(&xs).unwrap_or(0.0),
            bayes_factor_10: prior.bayes_factor(&null_prior, &xs).unwrap_or(1.0),
        };
    }

    fn recompute_nig(&mut self) {
        let mu0 = self.state.prior_a.clamp(-5.0, 10.0);
        let alpha0 = self.state.prior_b.clamp(2.2, 15.0);
        let n = (self.state.sample_size.round().clamp(2.0, 200.0)) as usize;
        let x_bar = self.state.observed_stat.clamp(-5.0, 10.0);
        let xs = Self::synth_gaussian_samples(n, x_bar, 1.15);
        let prior = NormalInverseGamma::new(mu0, 1.5, alpha0, alpha0 - 1.0).unwrap();
        let post = prior.update_slice(&xs).unwrap();
        let null_prior = NormalInverseGamma::new(0.0, 0.5, 2.5, 1.5).unwrap();
        let (lo, hi) = post
            .mean_credible_interval(self.state.credible_level)
            .unwrap_or((-2.0, 2.0));
        let (t_prior, scale_0) = (
            math_explorer::engine_prob::StudentsT::new(2.0 * prior.alpha()).unwrap(),
            (prior.beta() / (prior.alpha() * prior.kappa())).sqrt(),
        );
        let (t_post, scale_n) = (
            math_explorer::engine_prob::StudentsT::new(2.0 * post.alpha()).unwrap(),
            (post.beta() / (post.alpha() * post.kappa())).sqrt(),
        );
        let x_min = (mu0 - 3.5 * scale_0).min(lo - 1.5);
        let x_max = (mu0 + 3.5 * scale_0).max(hi + 1.5);
        for i in 0..160 {
            let x = x_min + (x_max - x_min) * (i as f64) / 159.0;
            self.prior_curve
                .push([x, t_prior.pdf((x - prior.mu()) / scale_0) / scale_0]);
            self.posterior_curve
                .push([x, t_post.pdf((x - post.mu()) / scale_n) / scale_n]);
        }
        let post_var = if post.alpha() > 1.0 {
            post.beta() / ((post.alpha() - 1.0) * post.kappa())
        } else {
            scale_n * scale_n
        };
        self.metrics = BayesianMetrics {
            prior_mean: prior.posterior_mean(),
            posterior_mean: post.posterior_mean(),
            posterior_variance: post_var,
            ci_low: lo,
            ci_high: hi,
            log_evidence: prior.log_marginal_likelihood(&xs).unwrap_or(0.0),
            bayes_factor_10: prior.bayes_factor(&null_prior, &xs).unwrap_or(1.0),
        };
    }

    fn render_controls(&mut self, ui: &mut egui::Ui) {
        let mut changed = false;
        egui::ComboBox::from_label("Conjugate Model")
            .selected_text(self.state.family.label())
            .show_ui(ui, |ui| {
                for fam in ConjugateFamily::ALL {
                    if ui
                        .selectable_value(&mut self.state.family, fam, fam.label())
                        .changed()
                    {
                        changed = true;
                    }
                }
            });
        ui.separator();
        changed |= ui
            .add(egui::Slider::new(&mut self.state.prior_a, -5.0..=25.0).text("Prior α₀ / μ₀"))
            .accessible_hover_text("Primary prior hyperparameter (shape alpha_0 or location mu_0)")
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.state.prior_b, 0.2..=25.0).text("Prior β₀ / τ₀"))
            .accessible_hover_text("Secondary prior hyperparameter (rate beta_0 or spread tau_0)")
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.state.sample_size, 1.0..=200.0).text("Sample Size n"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.state.observed_stat, -5.0..=15.0).text("Obs Stat"))
            .accessible_hover_text("Observed proportion k/n, event rate, or sample mean x_bar")
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut self.state.credible_level, 0.50..=0.99).text("Credible 1−α"),
            )
            .changed();
        if changed {
            self.recompute_all();
        }
    }

    fn render_metrics_bar(&self, ui: &mut egui::Ui, palette: &SemanticPalette) {
        let m = self.metrics;
        ui.horizontal_wrapped(|ui| {
            ui.colored_label(
                palette.muted_text,
                format!("Prior Mean: {:.4}", m.prior_mean),
            );
            ui.separator();
            ui.colored_label(
                palette.series[0],
                format!(
                    "Post Mean: {:.4} (Var {:.4})",
                    m.posterior_mean, m.posterior_variance
                ),
            );
            ui.separator();
            ui.colored_label(
                palette.accent,
                format!(
                    "{:.0}% CI: [{:.4}, {:.4}]",
                    self.state.credible_level * 100.0,
                    m.ci_low,
                    m.ci_high
                ),
            );
            ui.separator();
            ui.colored_label(
                palette.success,
                format!(
                    "BF₁₀: {:.3} (ln p(D) = {:.2})",
                    m.bayes_factor_10, m.log_evidence
                ),
            );
        });
    }

    fn render_plot(&self, ui: &mut egui::Ui) {
        let palette = SemanticPalette::from_visuals(ui.visuals());
        let m = self.metrics;
        self.render_metrics_bar(ui, &palette);
        ui.add_space(4.0);

        Plot::new(ui.id().with("prob_conjugate_bayes_plot"))
            .legend(Legend::default())
            .x_axis_label("Parameter θ / λ / μ")
            .y_axis_label("Prior & Posterior Density")
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new("Prior π(θ)", PlotPoints::from(self.prior_curve.clone()))
                        .color(palette.muted_text)
                        .style(LineStyle::Dashed { length: 7.0_f32 })
                        .width(2.0_f32),
                );
                plot_ui.line(
                    Line::new(
                        "Posterior π(θ | D)",
                        PlotPoints::from(self.posterior_curve.clone()),
                    )
                    .color(palette.series[0])
                    .width(2.6_f32),
                );
                plot_ui.vline(
                    VLine::new("Posterior Mean", m.posterior_mean)
                        .color(palette.success)
                        .width(2.0_f32),
                );
                plot_ui.vline(
                    VLine::new("Credible Lower", m.ci_low)
                        .color(palette.accent)
                        .style(LineStyle::Dashed { length: 5.0_f32 }),
                );
                plot_ui.vline(
                    VLine::new("Credible Upper", m.ci_high)
                        .color(palette.accent)
                        .style(LineStyle::Dashed { length: 5.0_f32 }),
                );
            });
    }
}

impl InteractiveTool for ConjugateBayesianTool {
    fn name(&self) -> &'static str {
        "Conjugate Bayesian Updater"
    }
    fn domain(&self) -> &'static str {
        "probability"
    }
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }
    fn show_inside(&mut self, parent_ui: &mut egui::Ui) {
        let panel_id = crate::framework::scoped_panel_id(parent_ui, "prob_bayes_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            render_tool_header(ui, "Conjugate Bayesian Updater");
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
        let parsed: BayesianState =
            serde_json::from_str(json).map_err(|e| format!("Invalid BayesianState: {e}"))?;
        parsed.validate()?;
        self.state = parsed;
        self.recompute_all();
        Ok(())
    }
    fn clamp_parameters(&mut self) -> bool {
        self.state.prior_a = self.state.prior_a.clamp(-5.0, 25.0);
        self.state.prior_b = self.state.prior_b.clamp(0.2, 25.0);
        self.state.sample_size = self.state.sample_size.clamp(1.0, 200.0);
        self.state.observed_stat = self.state.observed_stat.clamp(-5.0, 15.0);
        self.state.credible_level = self.state.credible_level.clamp(0.50, 0.99);
        self.recompute_all();
        true
    }
    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }
    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        vec![
            ("Prior Density".into(), self.prior_curve.clone()),
            ("Posterior Density".into(), self.posterior_curve.clone()),
        ]
    }
    fn step_simulation(&mut self) {
        self.state.sample_size = (self.state.sample_size + 4.0).min(200.0);
        self.recompute_all();
    }
}

// [cite:probability]
