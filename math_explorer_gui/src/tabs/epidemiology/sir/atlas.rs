#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::SirTool;
use crate::async_sim::atlas::{
    atlas_scalar_color, AtlasAxisPair, AtlasAxisSpec, AtlasCell, AtlasTeleportEvent,
    RegimeCartographer,
};
use crate::framework::TraceSeries;
use crate::reflective_ui::render_copyable_metric;
use eframe::egui;
use egui_plot::PlotPoint;
use math_explorer::epidemiology::analytics::calculate_final_size;
use scientific_metadata::theory::{ParameterConstraint, TheoryDescribable};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// Supported `(x, y)` parameter axis pairs for [`SirCartographer`].
pub const SIR_ATLAS_AXIS_PAIRS: &[(&str, &str)] = &[("beta", "gamma"), ("beta", "I0")];

/// Default serialized atlas axis keys for [`super::SirTool`].
#[must_use]
pub fn default_sir_atlas_axes() -> [String; 2] {
    ["beta".to_string(), "gamma".to_string()]
}

/// Predicate omitting default `atlas_axes` during JSON serialization for backward compatibility.
#[must_use]
pub fn is_default_sir_atlas_axes(axes: &[String; 2]) -> bool {
    axes[0] == "beta" && axes[1] == "gamma"
}

/// Discrete Kermack-McKendrick SIR epidemic regimes across the $R_0 = 1$ threshold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum SirRegime {
    /// Sub-threshold decay ($R_0 < 1$): infections burn out without exponential takeoff.
    SubthresholdDecay = 0,
    /// Critical threshold ($|R_0 - 1| \le 0.02$): epidemic tipping boundary.
    CriticalThreshold = 1,
    /// Supercritical epidemic outbreak ($R_0 > 1$): macroscopic population wave.
    EpidemicOutbreak = 2,
}

impl SirRegime {
    /// Classifies $R_0$ into a discrete [`SirRegime`].
    #[inline]
    #[must_use]
    pub fn from_r0(r0: f64) -> Self {
        if !r0.is_finite() || r0 > 1.02 {
            Self::EpidemicOutbreak
        } else if r0 < 0.98 {
            Self::SubthresholdDecay
        } else {
            Self::CriticalThreshold
        }
    }

    /// Converts a raw `u8` regime code into [`Self`].
    #[inline]
    #[must_use]
    pub const fn from_u8(code: u8) -> Self {
        match code {
            0 => Self::SubthresholdDecay,
            1 => Self::CriticalThreshold,
            _ => Self::EpidemicOutbreak,
        }
    }

    /// Human-readable epidemic regime label.
    #[inline]
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::SubthresholdDecay => "Sub-Threshold Decay (R₀ < 1)",
            Self::CriticalThreshold => "Critical Threshold (R₀ ≈ 1)",
            Self::EpidemicOutbreak => "Epidemic Outbreak (R₀ > 1)",
        }
    }
}

#[inline]
fn fixed_point_final_susceptible(r0: f64, s0: f64, n: f64) -> f64 {
    let mut x = if r0 > 1.0 {
        (s0 * (-r0).exp()).clamp(0.0, s0)
    } else {
        s0
    };
    for _ in 0..16 {
        x = (s0 * (-r0 * (1.0 - x / n)).exp()).clamp(0.0, s0);
    }
    x
}

/// Evaluates `(r0, final_infected_fraction)` via [`calculate_final_size`] with zero heap allocations.
///
/// Guards boundary inputs (`r0 <= 0`, `gamma <= 0`, extreme `r0 > 15`, and near-zero Newton
/// slope) on the stack so [`calculate_final_size`] never allocates an `Err(String)`.
#[must_use]
pub fn eval_sir_final_outbreak(beta: f64, gamma: f64, i0: f64, n: f64) -> (f64, f64) {
    let safe_n = if n.is_finite() { n.max(1.0) } else { 1000.0 };
    let safe_i0 = if i0.is_finite() {
        i0.clamp(0.0, safe_n)
    } else {
        0.0
    };
    let s0 = (safe_n - safe_i0).max(0.0);
    let base_frac = (safe_i0 / safe_n).clamp(0.0, 1.0);

    if !beta.is_finite() || beta <= 1e-9 {
        return (0.0, base_frac);
    }
    if safe_i0 <= 0.0 {
        let r0 = if gamma.is_finite() && gamma > 1e-9 {
            beta / gamma
        } else {
            f64::INFINITY
        };
        return (r0, 0.0);
    }
    if !gamma.is_finite() || gamma <= 1e-9 {
        return (f64::INFINITY, 1.0);
    }

    let r0 = beta / gamma;
    if s0 <= 1e-3 {
        return (r0, 1.0);
    }
    let init_x = if r0 > 1.0 { s0 * (-r0).exp() } else { s0 };
    let init_slope = (1.0 / init_x.max(1e-5) - r0 / safe_n).abs();
    let s_inf = if r0 <= 15.0 && init_x >= 5e-5 && init_slope >= 1e-4 {
        calculate_final_size(r0, s0, safe_n)
            .unwrap_or_else(|_| fixed_point_final_susceptible(r0, s0, safe_n))
    } else {
        fixed_point_final_susceptible(r0, s0, safe_n)
    };
    let frac = ((safe_n - s_inf.clamp(0.0, s0)) / safe_n).clamp(base_frac, 1.0);
    (r0, frac)
}

fn sir_axis_spec(
    theory_params: &HashMap<String, ParameterConstraint>,
    key: &'static str,
    n: f64,
) -> AtlasAxisSpec {
    match key {
        "gamma" => AtlasAxisSpec::from_theory(
            theory_params,
            "gamma",
            "Recovery Rate (γ)",
            ParameterConstraint {
                min: 0.0,
                max: 1.0,
                step: 0.01,
                default: Some(0.1),
            },
        ),
        "I0" => {
            let max_i0 = (n * 0.5).max(2.0);
            AtlasAxisSpec {
                key: "I0",
                label: "Initial Infected (I₀)",
                constraint: ParameterConstraint {
                    min: 1.0,
                    max: max_i0,
                    step: 1.0,
                    default: Some(10.0_f64.min(max_i0)),
                },
            }
        }
        _ => AtlasAxisSpec::from_theory(
            theory_params,
            "beta",
            "Transmission Rate (β)",
            ParameterConstraint {
                min: 0.0,
                max: 5.0,
                step: 0.02,
                default: Some(0.5),
            },
        ),
    }
}

/// Zero-allocation 2D outbreak cartographer for [`super::SirTool`].
#[derive(Clone, Debug)]
pub struct SirCartographer {
    axes: AtlasAxisPair,
    base_n: f64,
    base_i0: f64,
    base_beta: f64,
    base_gamma: f64,
}

impl SirCartographer {
    /// Builds a [`SirCartographer`] from [`super::SirTool`] parameters and theory constraints.
    #[must_use]
    pub fn new(
        theory_params: &HashMap<String, ParameterConstraint>,
        axis_keys: (&'static str, &'static str),
        n: f64,
        i0: f64,
        beta: f64,
        gamma: f64,
    ) -> Self {
        Self {
            axes: AtlasAxisPair {
                x: sir_axis_spec(theory_params, axis_keys.0, n),
                y: sir_axis_spec(theory_params, axis_keys.1, n),
            },
            base_n: n,
            base_i0: i0,
            base_beta: beta,
            base_gamma: gamma,
        }
    }

    #[inline]
    fn resolve_params(&self, p1: f64, p2: f64) -> (f64, f64, f64) {
        let mut beta = self.base_beta;
        let mut gamma = self.base_gamma;
        let mut i0 = self.base_i0;
        for (key, val) in [(self.axes.x.key, p1), (self.axes.y.key, p2)] {
            match key {
                "beta" => beta = val,
                "gamma" => gamma = val,
                "I0" => i0 = val,
                _ => {}
            }
        }
        (beta, gamma, i0)
    }
}

impl RegimeCartographer for SirCartographer {
    fn axes(&self) -> AtlasAxisPair {
        self.axes.clone()
    }

    fn evaluate_cell(&self, p1: f64, p2: f64) -> AtlasCell {
        let (beta, gamma, i0) = self.resolve_params(p1, p2);
        let (r0, frac) = eval_sir_final_outbreak(beta, gamma, i0, self.base_n);
        AtlasCell {
            regime: SirRegime::from_r0(r0) as u8,
            primary: frac,
            secondary: r0,
        }
    }

    fn color_for_cell(&self, cell: AtlasCell, high_contrast: bool) -> egui::Color32 {
        let frac = cell.primary.clamp(0.0, 1.0);
        let mapped = match SirRegime::from_u8(cell.regime) {
            SirRegime::SubthresholdDecay => (frac * 4.0).clamp(0.0, 0.24),
            SirRegime::CriticalThreshold => 0.28 + 0.12 * frac,
            SirRegime::EpidemicOutbreak => 0.32 + 0.68 * frac,
        };
        atlas_scalar_color(mapped, high_contrast)
    }

    fn regime_label(&self, regime: u8) -> &'static str {
        SirRegime::from_u8(regime).label()
    }

    fn primary_metric_label(&self) -> &'static str {
        "Final Infected (%)"
    }

    fn format_hover_tooltip(&self, p1: f64, p2: f64, cell: AtlasCell) -> String {
        let r0_txt = if cell.secondary.is_finite() {
            format!("{:.3}", cell.secondary)
        } else {
            "∞".to_string()
        };
        format!(
            "{} = {:.3}, {} = {:.3}\nRegime: {}\nR₀ = {}  |  Final Infected: {:.2}%",
            self.axes.x.key,
            p1,
            self.axes.y.key,
            p2,
            self.regime_label(cell.regime),
            r0_txt,
            cell.primary * 100.0
        )
    }

    fn overlay_contour(&self) -> Option<(&'static str, Vec<[f64; 2]>)> {
        if self.axes.x.key == "beta" && self.axes.y.key == "gamma" {
            let limit = self.axes.x.constraint.max.min(self.axes.y.constraint.max);
            Some(("R₀ = 1 Threshold", vec![[0.0, 0.0], [limit, limit]]))
        } else if self.axes.x.key == "beta" && self.axes.y.key == "I0" {
            let b = self.axes.x.clamp(self.base_gamma);
            Some((
                "R₀ = 1 Threshold",
                vec![
                    [b, self.axes.y.constraint.min],
                    [b, self.axes.y.constraint.max],
                ],
            ))
        } else {
            None
        }
    }

    fn legend_entries(&self, high_contrast: bool) -> Vec<(&'static str, egui::Color32)> {
        vec![
            (
                SirRegime::SubthresholdDecay.label(),
                atlas_scalar_color(0.08, high_contrast),
            ),
            (
                SirRegime::CriticalThreshold.label(),
                atlas_scalar_color(0.30, high_contrast),
            ),
            (
                SirRegime::EpidemicOutbreak.label(),
                atlas_scalar_color(0.85, high_contrast),
            ),
        ]
    }
}

impl SirTool {
    /// Evaluates `(r0, final_infected_fraction)` for the active parameters via `calculate_final_size`.
    #[must_use]
    pub fn final_outbreak_metrics(&self) -> (f64, f64) {
        eval_sir_final_outbreak(self.beta, self.gamma, self.i0, self.n)
    }

    /// Applies an [`AtlasTeleportEvent`] and recalculates $S(t), I(t), R(t)$ immediately.
    pub fn apply_atlas_teleport(&mut self, ev: AtlasTeleportEvent) {
        let prev_summary = self.last_param_summary.clone();
        for (key, val) in [(ev.x_key, ev.x_val), (ev.y_key, ev.y_val)] {
            match key {
                "beta" => self.beta = val.clamp(0.0, 5.0),
                "gamma" => self.gamma = val.clamp(0.0, 1.0),
                "I0" => self.i0 = val.clamp(1.0, (self.n * 0.5).max(1.0)),
                _ => {}
            }
        }
        let to_pairs = |pts: &[PlotPoint]| pts.iter().map(|p| [p.x, p.y]).collect::<Vec<_>>();
        let series = vec![
            TraceSeries::new("Susceptible", to_pairs(&self.s_points)),
            TraceSeries::new("Infected", to_pairs(&self.i_points)),
            TraceSeries::new("Recovered", to_pairs(&self.r_points)),
        ];
        self.snapshot_manager
            .push_history_run(&prev_summary, series);
        self.recalculate();
    }

    pub(super) fn render_sir_readouts_and_atlas(&mut self, ui: &mut egui::Ui) {
        let (_r0, final_frac) = self.final_outbreak_metrics();
        render_copyable_metric(
            ui,
            "Final Outbreak Size (S_∞ Solver)",
            &format!(
                "{:.2}% ({:.0} / {:.0})",
                final_frac * 100.0,
                final_frac * self.n,
                self.n
            ),
        );
        ui.horizontal(|ui| {
            let _ = self.atlas.show_toggle_button(ui);
        });
        self.render_sir_atlas_panel(ui);
    }

    fn render_sir_atlas_panel(&mut self, ui: &mut egui::Ui) {
        if !self.atlas.is_visible() {
            return;
        }
        let (xk, yk) = self.atlas.axis_keys();
        let get_val = |k: &str| match k {
            "beta" => self.beta,
            "gamma" => self.gamma,
            "I0" => self.i0,
            _ => 0.0,
        };
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.n.to_bits().hash(&mut h);
        if xk != "I0" && yk != "I0" {
            self.i0.to_bits().hash(&mut h);
        }
        if xk != "gamma" && yk != "gamma" {
            self.gamma.to_bits().hash(&mut h);
        }
        let theory_params = self.theory_parameters();
        let (n, i0, beta, gamma) = (self.n, self.i0, self.beta, self.gamma);
        if let Some(ev) = self.atlas.show_panel(
            ui,
            "sir",
            get_val(xk),
            get_val(yk),
            h.finish(),
            move |keys| {
                Arc::new(SirCartographer::new(
                    &theory_params,
                    keys,
                    n,
                    i0,
                    beta,
                    gamma,
                ))
            },
        ) {
            self.apply_atlas_teleport(ev);
        }
    }
}
