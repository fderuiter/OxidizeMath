//! Interactive Volatility Drag and Non-Ergodic Bankroll Ensemble Laboratory.
//!
//! Demonstrates the divergence between ensemble-average expected wealth `E[W_t]`
//! and time-average / typical median wealth `W_median(t) = W_0 * exp(t * g(f))`
//! under overbetting (`f > 2 * f*`) using preallocated zero-allocation buffers.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::accessibility::AccessibleHoverText;
use crate::framework::CalloutManager;
use eframe::egui;
use egui_plot::{Line, Plot, PlotPoint};
use math_explorer::pure_math::statistics::kelly::{
    expected_growth_rate, expected_value, kelly_fraction, BankrollFraction, Odds, UnitInterval,
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
    /// Classic Full / Half / Quarter Kelly compounding trajectories.
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

/// Preallocated simulator and UI controller for Kelly volatility-drag experiments.
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
    path_wealths: Vec<f64>,
    scratch_sort: Vec<f64>,
    /// Latest logarithmic growth rate $g(f)$ from `expected_growth_rate`.
    pub last_growth_rate: f64,
    /// Latest optimal Kelly fraction $f^*$ when positive edge exists.
    pub last_f_star: Option<f64>,
    /// Latest terminal expected wealth `E[W_T]`.
    pub last_expected_terminal: f64,
    /// Latest terminal median wealth $W_{\text{median}}(T)$.
    pub last_median_terminal: f64,
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
            path_wealths: vec![0.0; ENSEMBLE_PATH_COUNT],
            scratch_sort: Vec::with_capacity(ENSEMBLE_PATH_COUNT),
            last_growth_rate: 0.0,
            last_f_star: None,
            last_expected_terminal: 0.0,
            last_median_terminal: 0.0,
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

    /// Evaluates analytical curves and stochastic ensemble paths with zero heap allocations.
    pub fn evaluate_in_place(
        &mut self,
        initial_bankroll: f64,
        p: &UnitInterval,
        o: &Odds,
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

        let ev = expected_value(p, o);
        let g = expected_growth_rate(p, o, &frac);
        let f_star = kelly_fraction(p, o).ok().map(|fk| fk.value());
        self.last_growth_rate = g;
        self.last_f_star = f_star;
        self.is_ruin_zone = f_star.is_some_and(|fs| f_clamped > 2.0 * fs);

        self.expected_points.clear();
        self.median_points.clear();
        self.empirical_median_points.clear();
        self.path_wealths.fill(w0);

        self.expected_points.push(PlotPoint::new(0.0, w0));
        self.median_points.push(PlotPoint::new(0.0, w0));
        self.empirical_median_points.push(PlotPoint::new(0.0, w0));

        let mut rng = OxidizeRng::from_seed(BASE_SEED ^ seed_counter.wrapping_mul(0x9E37_79B9));
        let net_b = o.net_profit_multiplier();
        let step_ev_factor = (1.0 + f_clamped * ev).max(0.0);
        let mut ew = w0;

        for t in 1..=steps {
            ew = (ew * step_ev_factor).max(0.0);
            let w_med = if g.is_finite() {
                (w0 * ((t as f64) * g).exp()).max(0.0)
            } else {
                0.0
            };
            let emp_med = self.step_ensemble_median(&mut rng, p.value(), f_clamped, net_b);
            let x = t as f64;
            self.expected_points.push(PlotPoint::new(x, ew));
            self.median_points.push(PlotPoint::new(x, w_med));
            self.empirical_median_points.push(PlotPoint::new(x, emp_med));
        }

        self.last_expected_terminal = self.expected_points.last().map_or(w0, |pt| pt.y);
        self.last_median_terminal = self.median_points.last().map_or(w0, |pt| pt.y);
    }

    fn step_ensemble_median(
        &mut self,
        rng: &mut OxidizeRng,
        win_prob: f64,
        f_clamped: f64,
        net_b: f64,
    ) -> f64 {
        for w in &mut self.path_wealths {
            let bet = *w * f_clamped;
            if rng.r#gen::<f64>() < win_prob {
                *w = (*w + bet * net_b).max(0.0);
            } else {
                *w = (*w - bet).max(0.0);
            }
        }
        self.scratch_sort.clear();
        self.scratch_sort.extend_from_slice(&self.path_wealths);
        self.scratch_sort.sort_by(f64::total_cmp);
        let mid = self.scratch_sort.len() / 2;
        0.5 * (self.scratch_sort[mid - 1] + self.scratch_sort[mid])
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

        if let Some(fs) = self.last_f_star.filter(|&fs| self.custom_fraction > 2.0 * fs) {
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
        let f_star_text = self
            .last_f_star
            .map_or_else(|| "No positive edge".to_string(), |fs| format!("{fs:.3} (2f* = {:.3})", 2.0 * fs));
        ui.horizontal_wrapped(|ui| {
            ui.label(format!("Optimal f*: {f_star_text}"));
            ui.separator();
            ui.label(format!("Growth Rate g(f): {:+.5}/bet", self.last_growth_rate));
            ui.separator();
            ui.label(format!("E[W_{num_bets}]: {:.2}", self.last_expected_terminal));
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
            self.callout_manager.render_on_plot(plot_ui);
        });
    }
}

// [cite:clinical_trials_statistics]
