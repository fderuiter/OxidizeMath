//! Interactive Volatility Drag, Push Settlement, and Drawdown-Constrained Ensemble Laboratory.
//!
//! Demonstrates the divergence between ensemble-average expected wealth `E[W_t]`,
//! typical median wealth `W_median(t) = W_0 * exp(t * g(f))` under overbetting (`f > 2 * f*`),
//! and Busseti-Ryu-Boyd drawdown-constrained trajectories using preallocated buffers.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::accessibility::AccessibleHoverText;
use crate::framework::CalloutManager;
use eframe::egui;
use egui_plot::{Line, Plot, PlotPoint};
use math_explorer::pure_math::statistics::kelly::{
    drawdown_constrained_push_kelly, push_expected_growth_rate, push_expected_value,
    push_kelly_fraction, BankrollFraction, DrawdownConstraint, Odds, PushOutcome, PushWager,
    UnitInterval,
};
use oxidize_core::rng::{OxidizeRng, BASE_SEED};
use rand::Rng;

/// Maximum number of sequential wagers preallocated for zero-allocation evaluation.
pub const MAX_ENSEMBLE_BETS: usize = 1000;
/// Number of Monte Carlo ensemble trajectories simulated per rerun.
pub const ENSEMBLE_PATH_COUNT: usize = 64;
/// Upper bound on custom wager fraction `f` so `ln(1 - f)` remains finite.
pub const MAX_CUSTOM_FRACTION: f64 = 0.95;
/// Default custom wager fraction demonstrating overbetting volatility drag.
pub const DEFAULT_CUSTOM_FRACTION: f64 = 0.30;

/// Active view mode inside [`super::bankroll_growth::BankrollGrowthTool`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum BankrollLabMode {
    /// Classic Full / Half / Quarter / Drawdown-Constrained Kelly compounding trajectories.
    #[default]
    KellyStrategies,
    /// Interactive non-ergodic volatility-drag and overbetting ruin laboratory.
    VolatilityDrag,
}

/// User hypothesis choice made before inspecting the non-ergodic trajectory split.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum VolatilityHypothesis {
    /// User predicts typical wealth compounds upward alongside expected wealth.
    TypicalGrows,
    /// User predicts volatility drag drives typical wealth down while `E[W_t]` grows.
    TypicalShrinksDiverges,
}

/// Preallocated simulator and UI controller for Kelly volatility-drag and drawdown experiments.
pub struct BankrollEnsembleLab {
    /// Custom wager fraction `f` clamped to `0.0..=0.95`.
    pub custom_fraction: f64,
    /// User's predict-before-run hypothesis selection.
    pub hypothesis: Option<VolatilityHypothesis>,
    /// Plot callout overlay manager for growth rate and ruin annotations.
    pub callout_manager: CalloutManager,
    /// Analytical expected wealth trajectory `E[W_t] = W_0 * (1 + f * EV)^t`.
    pub expected_points: Vec<PlotPoint>,
    /// Analytical asymptotic median trajectory $W_{\text{median}}(t) = W_0 e^{t g(f)}$.
    pub median_points: Vec<PlotPoint>,
    /// Empirical median trajectory across `ENSEMBLE_PATH_COUNT` stochastic paths.
    pub empirical_median_points: Vec<PlotPoint>,
    /// Empirical median trajectory under the Busseti-Ryu-Boyd drawdown-constrained fraction.
    pub drawdown_median_points: Vec<PlotPoint>,
    path_wealths: Vec<f64>,
    drawdown_path_wealths: Vec<f64>,
    scratch_sort: Vec<f64>,
    /// Latest logarithmic growth rate $g(f)$.
    pub last_growth_rate: f64,
    /// Latest optimal Kelly fraction $f^*$ when positive edge exists.
    pub last_f_star: Option<f64>,
    /// Latest Busseti-Ryu-Boyd risk-constrained Kelly fraction when active.
    pub last_drawdown_f: Option<f64>,
    /// Latest terminal expected wealth `E[W_T]`.
    pub last_expected_terminal: f64,
    /// Latest terminal median wealth $W_{\text{median}}(T)$.
    pub last_median_terminal: f64,
    /// Latest terminal empirical median wealth for the drawdown-constrained strategy.
    pub last_drawdown_terminal: f64,
    /// Whether `f > 2 * f*` (or `g(f) < 0` despite positive edge), flagging ruin.
    pub is_ruin_zone: bool,
}

impl Default for BankrollEnsembleLab {
    fn default() -> Self {
        Self::new()
    }
}

impl BankrollEnsembleLab {
    /// Constructs a new ensemble lab with preallocated trajectory and sorting buffers.
    pub fn new() -> Self {
        Self {
            custom_fraction: DEFAULT_CUSTOM_FRACTION,
            hypothesis: None,
            callout_manager: CalloutManager::new(),
            expected_points: Vec::with_capacity(MAX_ENSEMBLE_BETS + 1),
            median_points: Vec::with_capacity(MAX_ENSEMBLE_BETS + 1),
            empirical_median_points: Vec::with_capacity(MAX_ENSEMBLE_BETS + 1),
            drawdown_median_points: Vec::with_capacity(MAX_ENSEMBLE_BETS + 1),
            path_wealths: vec![0.0; ENSEMBLE_PATH_COUNT],
            drawdown_path_wealths: vec![0.0; ENSEMBLE_PATH_COUNT],
            scratch_sort: Vec::with_capacity(ENSEMBLE_PATH_COUNT),
            last_growth_rate: 0.0,
            last_f_star: None,
            last_drawdown_f: None,
            last_expected_terminal: 0.0,
            last_median_terminal: 0.0,
            last_drawdown_terminal: 0.0,
            is_ruin_zone: false,
        }
    }

    /// Resets custom fraction, hypothesis state, and callouts to defaults.
    pub fn reset_defaults(&mut self) {
        self.custom_fraction = DEFAULT_CUSTOM_FRACTION;
        self.hypothesis = None;
        self.callout_manager.clear_auto();
    }

    /// Clamps `custom_fraction` into the safe finite interval `0.0..=0.95`.
    pub fn clamp_fraction(&mut self) {
        if !self.custom_fraction.is_finite() {
            self.custom_fraction = DEFAULT_CUSTOM_FRACTION;
        } else {
            self.custom_fraction = self.custom_fraction.clamp(0.0, MAX_CUSTOM_FRACTION);
        }
    }

    /// Evaluates binary analytical curves and ensemble paths with zero heap allocations.
    pub fn evaluate_in_place(
        &mut self,
        initial_bankroll: f64,
        p: &UnitInterval,
        o: &Odds,
        num_bets: usize,
        seed_counter: u64,
    ) {
        let Ok(wager) = PushWager::from_win_and_push(p.value(), 0.0, *o) else {
            return;
        };
        let dc = DrawdownConstraint::new(0.70, 0.10).ok();
        self.evaluate_push_and_drawdown_in_place(
            initial_bankroll,
            &wager,
            dc.as_ref(),
            num_bets,
            seed_counter,
        );
    }

    /// Evaluates three-state push settlements and drawdown-constrained ensemble paths with zero allocations.
    pub fn evaluate_push_and_drawdown_in_place(
        &mut self,
        initial_bankroll: f64,
        wager: &PushWager,
        constraint: Option<&DrawdownConstraint>,
        num_bets: usize,
        seed_counter: u64,
    ) {
        self.clamp_fraction();
        let steps = num_bets.min(MAX_ENSEMBLE_BETS);
        let w0 = initial_bankroll.max(0.0);
        let f_clamped = self.custom_fraction;
        let Ok(frac) = BankrollFraction::new(f_clamped) else {
            return;
        };

        let ev = push_expected_value(wager);
        let g = push_expected_growth_rate(wager, &frac);
        let f_star = push_kelly_fraction(wager).ok().map(|fk| fk.value());
        let f_rc = constraint
            .and_then(|dc| drawdown_constrained_push_kelly(wager, dc).ok())
            .map(|fr| fr.value());
        self.last_growth_rate = g;
        self.last_f_star = f_star;
        self.last_drawdown_f = f_rc;
        self.is_ruin_zone = f_star.is_some_and(|fs| f_clamped > 2.0 * fs);

        self.reset_trajectory_buffers(w0);
        let mut rng = OxidizeRng::from_seed(BASE_SEED ^ seed_counter.wrapping_mul(0x9E37_79B9));
        let step_ev_factor = (1.0 + f_clamped * ev).max(0.0);
        let mut ew = w0;
        let rc_frac = f_rc.unwrap_or(0.0);

        for t in 1..=steps {
            ew = (ew * step_ev_factor).max(0.0);
            let w_med = if g.is_finite() {
                (w0 * ((t as f64) * g).exp()).max(0.0)
            } else {
                0.0
            };
            let (emp_med, rc_med) = self.step_ensemble_medians(&mut rng, wager, f_clamped, rc_frac);
            let x = t as f64;
            self.expected_points.push(PlotPoint::new(x, ew));
            self.median_points.push(PlotPoint::new(x, w_med));
            self.empirical_median_points
                .push(PlotPoint::new(x, emp_med));
            self.drawdown_median_points.push(PlotPoint::new(x, rc_med));
        }

        self.last_expected_terminal = self.expected_points.last().map_or(w0, |pt| pt.y);
        self.last_median_terminal = self.median_points.last().map_or(w0, |pt| pt.y);
        self.last_drawdown_terminal = self.drawdown_median_points.last().map_or(w0, |pt| pt.y);
    }

    fn reset_trajectory_buffers(&mut self, w0: f64) {
        self.expected_points.clear();
        self.median_points.clear();
        self.empirical_median_points.clear();
        self.drawdown_median_points.clear();
        self.path_wealths.fill(w0);
        self.drawdown_path_wealths.fill(w0);

        self.expected_points.push(PlotPoint::new(0.0, w0));
        self.median_points.push(PlotPoint::new(0.0, w0));
        self.empirical_median_points.push(PlotPoint::new(0.0, w0));
        self.drawdown_median_points.push(PlotPoint::new(0.0, w0));
    }

    fn step_ensemble_medians(
        &mut self,
        rng: &mut OxidizeRng,
        wager: &PushWager,
        f_custom: f64,
        f_rc: f64,
    ) -> (f64, f64) {
        let net_b = wager.odds().net_profit_multiplier();
        for (w, w_rc) in self
            .path_wealths
            .iter_mut()
            .zip(self.drawdown_path_wealths.iter_mut())
        {
            let outcome = wager.sample_outcome(rng.r#gen::<f64>());
            *w = settle_raw_push(*w, f_custom, net_b, outcome);
            *w_rc = settle_raw_push(*w_rc, f_rc, net_b, outcome);
        }
        let emp_med = median_from_slice(&self.path_wealths, &mut self.scratch_sort);
        let rc_med = median_from_slice(&self.drawdown_path_wealths, &mut self.scratch_sort);
        (emp_med, rc_med)
    }

    /// Synchronizes automated `CalloutManager` overlays for `expected_growth_rate` and ruin.
    pub fn sync_callouts(&mut self, num_bets: usize) {
        let steps = num_bets.min(MAX_ENSEMBLE_BETS);
        if self.median_points.is_empty() {
            self.callout_manager.clear_auto();
            return;
        }
        let mid_idx = (steps / 2).min(self.median_points.len() - 1);
        let mid_pt = self.median_points[mid_idx];
        self.callout_manager.set_auto_callout(
            "growth_rate",
            [mid_pt.x, mid_pt.y],
            format!(
                "g(f={:.2}) = {:+.4}/bet",
                self.custom_fraction, self.last_growth_rate
            ),
            [24.0, -26.0],
        );

        if let Some(fs) = self
            .last_f_star
            .filter(|&fs| self.custom_fraction > 2.0 * fs)
        {
            let end_x = steps as f64;
            self.callout_manager.set_auto_callout(
                "ruin_warning",
                [end_x, self.last_median_terminal],
                format!(
                    "Ruin Trap (f={:.2} > 2f*={:.2}): W_med={:.4} vs E[W]={:.2}",
                    self.custom_fraction,
                    2.0 * fs,
                    self.last_median_terminal,
                    self.last_expected_terminal
                ),
                [-90.0, -34.0],
            );
        } else {
            self.remove_auto_tag("ruin_warning");
        }
    }

    fn remove_auto_tag(&mut self, tag: &str) {
        if let Some(id) = self
            .callout_manager
            .callouts()
            .iter()
            .find(|c| c.tag.as_deref() == Some(tag))
            .map(|c| c.id)
        {
            self.callout_manager.remove_callout(id);
        }
    }

    /// Renders the hypothesis prompt, bet-fraction slider, metrics, and volatility-drag plot.
    pub fn show_lab_ui(
        &mut self,
        ui: &mut egui::Ui,
        initial_bankroll: &mut f64,
        probability: &mut f64,
        odds: &mut f64,
        num_bets: &mut usize,
    ) -> bool {
        let mut changed = self.render_hypothesis_card(ui);
        ui.add_space(4.0);
        changed |= self.render_fraction_and_preset_controls(
            ui,
            initial_bankroll,
            probability,
            odds,
            num_bets,
        );
        self.render_metrics_banner(ui, *num_bets);
        self.render_ensemble_plot(ui);
        let _ = self.callout_manager.render_ui(ui);
        changed
    }

    fn render_hypothesis_card(&mut self, ui: &mut egui::Ui) -> bool {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        ui.group(|ui| {
            ui.label(
                egui::RichText::new(
                    "Predict Before Inspecting: At custom fraction f, how does typical wealth W_median(T) behave relative to E[W_T]?",
                )
                .strong(),
            );
            ui.horizontal_wrapped(|ui| {
                ui.radio_value(
                    &mut self.hypothesis,
                    Some(VolatilityHypothesis::TypicalGrows),
                    "Typical Wealth Compounds Upward",
                );
                ui.radio_value(
                    &mut self.hypothesis,
                    Some(VolatilityHypothesis::TypicalShrinksDiverges),
                    "Diverges: E[W_T] Grows, Typical Shrinks (Volatility Drag)",
                );
            });
            if let Some(choice) = self.hypothesis {
                let shrinks = self.last_growth_rate < 0.0;
                let correct = (choice == VolatilityHypothesis::TypicalShrinksDiverges) == shrinks;
                let (prefix, color) = if correct {
                    ("Hypothesis Confirmed:", palette.success)
                } else {
                    ("Counterintuitive Outcome:", palette.warning)
                };
                let detail = if shrinks {
                    "Overbetting past 2f* makes g(f) < 0: rare jackpots inflate E[W_T] while typical bettors decay toward ruin."
                } else {
                    "When 0 < f < 2f*, g(f) > 0 so typical wealth compounds upward alongside E[W_T]."
                };
                ui.colored_label(color, format!("{prefix} {detail}"));
            }
        });
        false
    }

    fn render_fraction_and_preset_controls(
        &mut self,
        ui: &mut egui::Ui,
        initial_bankroll: &mut f64,
        probability: &mut f64,
        odds: &mut f64,
        num_bets: &mut usize,
    ) -> bool {
        let mut changed = ui
            .add(
                egui::Slider::new(&mut self.custom_fraction, 0.0..=MAX_CUSTOM_FRACTION)
                    .text("Custom Bet Fraction (f)"),
            )
            .accessible_hover_text("Fraction of bankroll wagered each round, clamped to 0.0..=0.95")
            .changed();

        if ui
            .button("⚡ Load Volatility-Drag Paradox (W₀=100, p=0.55, b=1.0, T=200, f=0.30)")
            .accessible_hover_text(
                "Configure canonical non-ergodic overbetting divergence where E[W_200]=36,935.58 yet W_median(200)=3.9137",
            )
            .clicked()
        {
            *initial_bankroll = 100.0;
            *probability = 0.55;
            *odds = 1.0;
            *num_bets = 200;
            self.custom_fraction = 0.30;
            changed = true;
        }
        changed
    }

    fn render_metrics_banner(&self, ui: &mut egui::Ui, num_bets: usize) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        let f_star_text = self.last_f_star.map_or_else(
            || "No positive edge".to_string(),
            |fs| format!("{fs:.3} (2f* = {:.3})", 2.0 * fs),
        );
        ui.horizontal_wrapped(|ui| {
            ui.label(format!("Optimal f*: {f_star_text}"));
            ui.separator();
            ui.label(format!(
                "Growth Rate g(f): {:+.5}/bet",
                self.last_growth_rate
            ));
            ui.separator();
            ui.label(format!(
                "E[W_{num_bets}]: {:.2}",
                self.last_expected_terminal
            ));
            ui.separator();
            let med_color = if self.is_ruin_zone {
                palette.error
            } else {
                palette.success
            };
            ui.colored_label(
                med_color,
                format!("W_median({num_bets}): {:.4}", self.last_median_terminal),
            );
            if let Some(frc) = self.last_drawdown_f {
                ui.separator();
                ui.label(format!(
                    "Risk-Constrained (f={:.2}%): {:.2}",
                    frc * 100.0,
                    self.last_drawdown_terminal
                ));
            }
        });
    }

    fn render_ensemble_plot(&mut self, ui: &mut egui::Ui) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        let plot = Plot::new("bankroll_volatility_drag_plot")
            .view_aspect(2.1)
            .legend(egui_plot::Legend::default());
        plot.show(ui, |plot_ui| {
            plot_ui.line(
                Line::new("Expected Wealth E[W_t]", &self.expected_points[..])
                    .color(palette.series_color(1)),
            );
            plot_ui.line(
                Line::new("Analytical Median W_median(t)", &self.median_points[..])
                    .color(palette.series_color(3)),
            );
            plot_ui.line(
                Line::new(
                    "Ensemble Median (64 Seeded Paths)",
                    &self.empirical_median_points[..],
                )
                .color(palette.series_color(5)),
            );
            plot_ui.line(
                Line::new(
                    "Drawdown-Constrained Median",
                    &self.drawdown_median_points[..],
                )
                .color(palette.series_color(2)),
            );
            self.callout_manager.render_on_plot(plot_ui);
        });
    }
}

fn settle_raw_push(wealth: f64, frac: f64, net_b: f64, outcome: PushOutcome) -> f64 {
    let stake = wealth * frac;
    match outcome {
        PushOutcome::Win => (wealth + stake * net_b).max(0.0),
        PushOutcome::Push => wealth.max(0.0),
        PushOutcome::Loss => (wealth - stake).max(0.0),
    }
}

fn median_from_slice(values: &[f64], scratch: &mut Vec<f64>) -> f64 {
    scratch.clear();
    scratch.extend_from_slice(values);
    scratch.sort_by(f64::total_cmp);
    let mid = scratch.len() / 2;
    0.5 * (scratch[mid - 1] + scratch[mid])
}

// [cite:clinical_trials_statistics]
