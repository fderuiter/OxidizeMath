//! Analytical `BivariateWeibullModel` simulator, probability win-ratio curves,
//! and plot renderers for [`super::win_ratio::WinRatioWorkbenchTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::framework::bounded_drag_value;
use eframe::egui;
use egui_plot::{Bar, BarChart, Legend, Line, Plot, PlotPoints};
use math_explorer::applied::win_ratio::probability_win_ratio::{
    calculate_probability_win_ratio, ProbabilityWinRatioContext,
};
use math_explorer::applied::win_ratio::simulation::{win_ratio_parameter, SimulationParams};

use super::win_ratio_io::{CohortEvaluation, TierSpec};

/// Parameters driving the analytical `BivariateWeibullModel` simulator view.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WeibullSimConfig {
    /// Fatal hazard scale $\lambda_1$.
    pub lambda1: f64,
    /// Non-fatal hazard scale $\lambda_2$.
    pub lambda2: f64,
    /// Shared association shape $\alpha$.
    pub alpha: f64,
    /// Treatment hazard multiplier $\theta$.
    pub theta: f64,
    /// Administrative follow-up horizon $c$.
    pub horizon_c: f64,
}

impl Default for WeibullSimConfig {
    fn default() -> Self {
        Self {
            lambda1: 0.08,
            lambda2: 0.18,
            alpha: 1.25,
            theta: 0.65,
            horizon_c: 8.0,
        }
    }
}

impl WeibullSimConfig {
    /// Checks parameter finiteness and domain bounds.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        [
            self.lambda1,
            self.lambda2,
            self.alpha,
            self.theta,
            self.horizon_c,
        ]
        .iter()
        .all(|v| v.is_finite())
            && (0.005..=5.0).contains(&self.lambda1)
            && (0.005..=5.0).contains(&self.lambda2)
            && (0.2..=5.0).contains(&self.alpha)
            && (0.1..=3.0).contains(&self.theta)
            && (0.2..=30.0).contains(&self.horizon_c)
    }
}

/// Sampled survival and probability win-ratio curves from `BivariateWeibullModel`.
#[derive(Debug, Clone)]
pub struct WeibullEvaluation {
    /// Uncensored analytical win-ratio parameter $\theta^{-\alpha}$.
    pub uncensored_wr: f64,
    /// Integrated win probability $W(c)$ at `horizon_c`.
    pub win_prob: f64,
    /// Integrated loss probability $L(c)$ at `horizon_c`.
    pub loss_prob: f64,
    /// Integrated probability win ratio $PR(c) = W(c) / L(c)$.
    pub prob_wr: f64,
    /// Control marginal survival curve $S_0(t)$ on $[0, c]$.
    pub s0_curve: Vec<[f64; 2]>,
    /// Treatment marginal survival curve $S_1(t)$ on $[0, c]$.
    pub s1_curve: Vec<[f64; 2]>,
    /// Horizon probability win-ratio trajectory $PR(t)$ on $(0, c]$.
    pub pr_curve: Vec<[f64; 2]>,
}

fn probability_wr_at_horizon(params: &SimulationParams, c: f64) -> (f64, f64, f64) {
    let ctrl = params.control_model();
    let trt = params.treatment_model();
    let s0_c = ctrl.marginal_survival_t(c);
    let s1_c = trt.marginal_survival_t(c);
    let ctx = ProbabilityWinRatioContext::new(s0_c, s1_c, c, 1e-5);
    let w = ctx.calculate_win_probability(
        |t| trt.marginal_survival_t(t),
        |t| ctrl.pdf_t(t),
        |x| trt.conditional_survival_x_given_t(x, c),
        |x| ctrl.pdf_x_given_t(x, c),
    );
    let l = ctx.calculate_loss_probability(
        |t| ctrl.marginal_survival_t(t),
        |t| trt.pdf_t(t),
        |x| ctrl.conditional_survival_x_given_t(x, c),
        |x| trt.pdf_x_given_t(x, c),
    );
    let w_pos = w.max(0.0);
    let l_pos = l.max(0.0);
    (w_pos, l_pos, calculate_probability_win_ratio(w_pos, l_pos))
}

/// Evaluates `BivariateWeibullModel` marginal survival curves and `ProbabilityWinRatioContext` trajectories.
pub fn evaluate_weibull_simulator(cfg: &WeibullSimConfig) -> Result<WeibullEvaluation, String> {
    if !cfg.is_valid() {
        return Err("Weibull simulator parameters are out of valid finite bounds".into());
    }
    let params = SimulationParams::new(cfg.lambda1, cfg.lambda2, cfg.alpha, cfg.theta);
    let ctrl = params.control_model();
    let trt = params.treatment_model();
    let uncensored_wr = win_ratio_parameter(&params);
    let (win_prob, loss_prob, prob_wr) = probability_wr_at_horizon(&params, cfg.horizon_c);

    let steps = 32usize;
    let mut s0_curve = Vec::with_capacity(steps + 1);
    let mut s1_curve = Vec::with_capacity(steps + 1);
    let mut pr_curve = Vec::with_capacity(steps);

    for step in 0..=steps {
        let t = cfg.horizon_c * (step as f64 / steps as f64);
        s0_curve.push([t, ctrl.marginal_survival_t(t)]);
        s1_curve.push([t, trt.marginal_survival_t(t)]);
        if step > 0 {
            let (_, _, pr_t) = probability_wr_at_horizon(&params, t);
            let finite_pr = if pr_t.is_finite() {
                pr_t.clamp(0.0, 50.0)
            } else {
                uncensored_wr.clamp(0.0, 50.0)
            };
            pr_curve.push([t, finite_pr]);
        }
    }

    Ok(WeibullEvaluation {
        uncensored_wr,
        win_prob,
        loss_prob,
        prob_wr,
        s0_curve,
        s1_curve,
        pr_curve,
    })
}

/// Renders the `WeibullSimConfig` parameter grid, returning `true` when any parameter changes.
pub fn render_weibull_controls_grid(ui: &mut egui::Ui, cfg: &mut WeibullSimConfig) -> bool {
    ui.label("Bivariate Weibull Joint Survival Model:");
    let mut changed = false;
    egui::Grid::new("weibull_sim_grid")
        .num_columns(2)
        .show(ui, |ui| {
            ui.label("Fatal Scale (λ₁):");
            changed |= ui
                .add(bounded_drag_value(&mut cfg.lambda1, 0.005..=5.0, 0.01))
                .changed();
            ui.end_row();
            ui.label("Non-Fatal Scale (λ₂):");
            changed |= ui
                .add(bounded_drag_value(&mut cfg.lambda2, 0.005..=5.0, 0.01))
                .changed();
            ui.end_row();
            ui.label("Association (α):");
            changed |= ui
                .add(bounded_drag_value(&mut cfg.alpha, 0.2..=5.0, 0.05))
                .changed();
            ui.end_row();
            ui.label("Hazard Ratio (θ):");
            changed |= ui
                .add(bounded_drag_value(&mut cfg.theta, 0.1..=3.0, 0.02))
                .changed();
            ui.end_row();
            ui.label("Follow-up Horizon (c):");
            changed |= ui
                .add(bounded_drag_value(&mut cfg.horizon_c, 0.2..=30.0, 0.2))
                .changed();
            ui.end_row();
        });
    changed
}

/// Renders the hierarchical Win/Loss/Tie waterfall `BarChart`.
pub fn render_waterfall_bar_plot(
    ui: &mut egui::Ui,
    eval: Option<&CohortEvaluation>,
    tier_specs: &[TierSpec],
    palette: crate::app::SemanticPalette,
) {
    ui.heading("Hierarchical Composite Endpoint Waterfall");
    let Some(eval) = eval else {
        Plot::new("win_ratio_waterfall_empty").show(ui, |plot_ui| {
            plot_ui.line(Line::new("Empty", vec![[0.0, 0.0]]).color(palette.muted_text));
        });
        return;
    };
    let mut win_bars = Vec::with_capacity(eval.tiers.len());
    let mut loss_bars = Vec::with_capacity(eval.tiers.len());
    let mut tie_bars = Vec::with_capacity(eval.tiers.len());
    for (idx, counts) in eval.tiers.iter().enumerate() {
        let center = (idx + 1) as f64;
        let label = tier_specs.get(idx).map_or("Tier", |s| s.label.as_str());
        win_bars.push(
            Bar::new(center - 0.25, counts.wins as f64)
                .width(0.22)
                .name(format!("{label} Wins")),
        );
        loss_bars.push(
            Bar::new(center, counts.losses as f64)
                .width(0.22)
                .name(format!("{label} Losses")),
        );
        tie_bars.push(
            Bar::new(center + 0.25, counts.ties as f64)
                .width(0.22)
                .name(format!("{label} Remaining Ties")),
        );
    }
    Plot::new("win_ratio_waterfall_plot")
        .legend(Legend::default())
        .view_aspect(1.9)
        .x_axis_label("Endpoint Hierarchy Tier")
        .y_axis_label("Pair Count")
        .show(ui, |plot_ui| {
            plot_ui.bar_chart(BarChart::new("Treatment Wins", win_bars).color(palette.success));
            plot_ui.bar_chart(BarChart::new("Treatment Losses", loss_bars).color(palette.error));
            plot_ui
                .bar_chart(BarChart::new("Unresolved Ties", tie_bars).color(palette.series_color(0)));
        });
}

/// Renders the `BivariateWeibullModel` survival and probability win-ratio curves.
pub fn render_weibull_curves_plot(
    ui: &mut egui::Ui,
    sim: Option<&WeibullEvaluation>,
    palette: crate::app::SemanticPalette,
) {
    ui.heading("Bivariate Weibull Survival & Probability Win Ratio");
    let Some(sim) = sim else {
        return;
    };
    Plot::new("win_ratio_weibull_plot")
        .legend(Legend::default())
        .view_aspect(1.9)
        .x_axis_label("Follow-up Time t")
        .y_axis_label("Survival S(t) / Probability WR(t)")
        .show(ui, |plot_ui| {
            plot_ui.line(
                Line::new("Control Fatal Survival S0(t)", PlotPoints::new(sim.s0_curve.clone()))
                    .color(palette.series_color(1)),
            );
            plot_ui.line(
                Line::new(
                    "Treatment Fatal Survival S1(t)",
                    PlotPoints::new(sim.s1_curve.clone()),
                )
                .color(palette.success),
            );
            plot_ui.line(
                Line::new(
                    "Probability Win Ratio PR(t)",
                    PlotPoints::new(sim.pr_curve.clone()),
                )
                .color(palette.series_color(0)),
            );
        });
}

// [cite:win_ratio]
