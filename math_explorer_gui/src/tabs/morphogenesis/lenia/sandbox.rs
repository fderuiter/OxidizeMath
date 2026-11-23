//! Custom `CompiledExpr` rule sandboxes (`G(u)` and `K(r)`) and live 32-bin neighborhood
//! potential $U = A * K$ histogram calculations for [`super::LeniaTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::LeniaTool;
use crate::framework::MathExpressionInputBar;
use eframe::egui;
use math_explorer::biology::lenia::{CompiledGrowthRule, CompiledKernelRule};
use std::collections::BTreeMap;

/// Canonical Gaussian growth expression matching `GrowthParams::evaluate`.
pub const DEFAULT_GAUSSIAN_GROWTH_EXPR: &str = "2 * exp(-((u - mu)^2) / (2 * sigma^2)) - 1";
/// Chan (2019) quartic polynomial growth niche expression.
pub const CHAN_POLYNOMIAL_GROWTH_EXPR: &str = "2 * max(0, 1 - ((u - mu) / (3 * sigma))^2)^4 - 1";
/// Smooth-step continuous Conway growth window expression.
pub const SMOOTH_STEP_CONWAY_GROWTH_EXPR: &str =
    "2 * (1 / (1 + exp(-(u - mu) / sigma))) * (1 / (1 + exp((u - mu - 3 * sigma) / sigma))) - 1";

/// Chan (2019) polynomial radial shell expression on $r \in [0, 1]$.
pub const CHAN_POLYNOMIAL_KERNEL_EXPR: &str = "max(0, 1 - ((r - b) / w)^2)^4";
/// Smooth-step annulus radial kernel expression on $r \in [0, 1]$.
pub const SMOOTH_STEP_KERNEL_EXPR: &str =
    "1 / ((1 + exp(-(r - 0.22) / s)) * (1 + exp((r - 0.78) / s)))";
/// Gaussian ring radial kernel expression on $r \in [0, 1]$.
pub const GAUSSIAN_RING_KERNEL_EXPR: &str = "exp(-((r - b)^2) / (2 * w^2))";

/// Number of histogram bins used to display active-cell neighborhood potential $U = A * K$.
pub const POTENTIAL_HIST_BINS: usize = 32;
const QUANTILE_BINS: usize = 256;

/// Summary of the live active-cell neighborhood potential $U = A * K$ distribution.
#[derive(Clone, Copy, Debug)]
pub struct PotentialDistribution {
    /// `[bin_center_u, normalized_bar_height]` across $u \in [0, 1]$.
    pub bins: [[f64; 2]; POTENTIAL_HIST_BINS],
    /// Number of active or non-vacuum cells contributing to the histogram.
    pub active_count: usize,
    /// Approximate 5th-percentile neighborhood potential among active cells.
    pub u_p05: f64,
    /// Exact maximum finite neighborhood potential $\max_{x, y} U(x, y)$.
    pub u_max: f64,
}

/// Builds the `G(u)` [`MathExpressionInputBar`] sandbox (`axis 0 = u`) with Chan polynomial and smooth-step presets.
#[must_use]
pub fn build_growth_sandbox(mu: f64, sigma: f64) -> MathExpressionInputBar {
    let mut bar = MathExpressionInputBar::new(
        "G(u)",
        DEFAULT_GAUSSIAN_GROWTH_EXPR,
        vec![
            ("Gaussian Niche", DEFAULT_GAUSSIAN_GROWTH_EXPR),
            ("Chan Polynomial", CHAN_POLYNOMIAL_GROWTH_EXPR),
            ("Smooth-Step Conway", SMOOTH_STEP_CONWAY_GROWTH_EXPR),
        ],
    )
    .with_axes(&["u"])
    .with_param_range(0.01..=2.0, 0.5)
    .with_param_spec("mu", "Growth μ", 0.01..=0.60, mu)
    .with_param_spec("sigma", "Width σ", 0.002..=0.15, sigma)
    .with_disallowed_vars(&[
        ("x", "Growth G(u) depends on neighborhood potential 'u', not spatial 'x'."),
        ("y", "Growth G(u) depends on neighborhood potential 'u', not spatial 'y'."),
        ("r", "Use 'u' in G(u); normalized distance 'r' belongs in Kernel K(r)."),
    ]);
    bar.set_param("mu", mu);
    bar.set_param("sigma", sigma);
    bar
}

/// Builds the `K(r)` [`MathExpressionInputBar`] sandbox (`axis 0 = r`) with polynomial and smooth-step shell presets.
#[must_use]
pub fn build_kernel_sandbox() -> MathExpressionInputBar {
    let mut bar = MathExpressionInputBar::new(
        "K(r)",
        CHAN_POLYNOMIAL_KERNEL_EXPR,
        vec![
            ("Chan Polynomial Shell", CHAN_POLYNOMIAL_KERNEL_EXPR),
            ("Smooth-Step Annulus", SMOOTH_STEP_KERNEL_EXPR),
            ("Gaussian Ring", GAUSSIAN_RING_KERNEL_EXPR),
        ],
    )
    .with_axes(&["r"])
    .with_param_range(0.02..=2.0, 0.5)
    .with_param_spec("b", "Shell Peak b", 0.10..=0.90, 0.50)
    .with_param_spec("w", "Shell Half-Width w", 0.05..=0.80, 0.50)
    .with_param_spec("s", "Step Softness s", 0.01..=0.20, 0.04)
    .with_disallowed_vars(&[
        ("x", "Radial kernel K(r) depends on normalized distance 'r' in [0, 1]."),
        ("y", "Radial kernel K(r) depends on normalized distance 'r' in [0, 1]."),
        ("u", "Use 'r' in K(r); neighborhood potential 'u' belongs in Growth G(u)."),
    ]);
    bar.set_param("b", 0.50);
    bar.set_param("w", 0.50);
    bar.set_param("s", 0.04);
    bar
}

/// Computes the 32-bin active-cell potential histogram and `[u_p05, u_max]` quantiles in $O(W H)$ stack space.
#[must_use]
pub fn compute_potential_distribution(state: &[f64], potential: &[f64]) -> PotentialDistribution {
    let mut counts = [0u32; POTENTIAL_HIST_BINS];
    let mut cdf = [0u32; QUANTILE_BINS];
    let mut active_count = 0usize;
    let mut u_max = 0.0_f64;

    for (&a, &u) in state.iter().zip(potential.iter()) {
        if !u.is_finite() {
            continue;
        }
        if u > u_max {
            u_max = u;
        }
        if a > 0.01 || u > 0.01 {
            let clamped = u.clamp(0.0, 0.999_999);
            counts[(clamped * (POTENTIAL_HIST_BINS as f64)) as usize] += 1;
            cdf[(clamped * (QUANTILE_BINS as f64)) as usize] += 1;
            active_count += 1;
        }
    }

    let max_bin = counts.iter().copied().max().unwrap_or(0).max(1) as f64;
    let mut bins = [[0.0; 2]; POTENTIAL_HIST_BINS];
    for (idx, &c) in counts.iter().enumerate() {
        let u_center = ((idx as f64) + 0.5) / (POTENTIAL_HIST_BINS as f64);
        let norm_height = if active_count > 0 {
            0.90 * (c as f64) / max_bin
        } else {
            0.0
        };
        bins[idx] = [u_center, norm_height];
    }

    let u_p05 = if active_count == 0 {
        0.0
    } else {
        quantile_from_cdf(&cdf, active_count, 0.05).min(u_max)
    };

    PotentialDistribution {
        bins,
        active_count,
        u_p05,
        u_max,
    }
}

fn quantile_from_cdf(cdf: &[u32; QUANTILE_BINS], total: usize, q: f64) -> f64 {
    let target = (q * (total as f64)).max(0.0);
    let mut acc = 0u32;
    for (idx, &count) in cdf.iter().enumerate() {
        let next = acc + count;
        if (next as f64) >= target && count > 0 {
            let frac = ((target - (acc as f64)) / (count as f64)).clamp(0.0, 1.0);
            return ((idx as f64) + frac) / (QUANTILE_BINS as f64);
        }
        acc = next;
    }
    1.0
}

pub(crate) struct SerializedSandboxes {
    pub growth_expr: Option<String>,
    pub growth_params: BTreeMap<String, f64>,
    pub kernel_expr: Option<String>,
    pub kernel_params: BTreeMap<String, f64>,
}

impl LeniaTool {
    /// Returns `true` when `growth_bar` defines a non-default custom growth expression.
    #[must_use]
    pub fn has_custom_growth_formula(&self) -> bool {
        let trimmed = self.growth_bar.raw_text.trim();
        !trimmed.is_empty() && trimmed != DEFAULT_GAUSSIAN_GROWTH_EXPR
    }

    /// Sets the `G(u)` formula text, synthesizes free-parameter sliders, and synchronizes `LeniaEngine`.
    #[cfg(test)]
    pub fn set_growth_expression(&mut self, expr: &str) {
        self.growth_bar.set_param("mu", self.mu);
        self.growth_bar.set_param("sigma", self.sigma);
        self.growth_bar.set_expression(expr);
        self.pull_mu_sigma_from_growth_bar();
        self.sync_engine_parameters();
        self.user_mutation_nonce = self.user_mutation_nonce.wrapping_add(1);
    }

    /// Sets the `K(r)` formula text, enables custom kernel mode (unless locked by challenge), and syncs `LeniaEngine`.
    #[cfg(test)]
    pub fn set_kernel_expression(&mut self, expr: &str) {
        if self.is_kernel_locked_by_challenge() {
            return;
        }
        self.use_custom_kernel = true;
        self.kernel_bar.set_expression(expr);
        self.sync_engine_parameters();
        self.user_mutation_nonce = self.user_mutation_nonce.wrapping_add(1);
    }

    pub(crate) fn pull_mu_sigma_from_growth_bar(&mut self) {
        if let Some(m) = self.growth_bar.active_param_value("mu") {
            self.mu = m.clamp(0.01, 0.60);
        }
        if let Some(s) = self.growth_bar.active_param_value("sigma") {
            self.sigma = s.abs().clamp(0.002, 0.15);
        }
    }

    pub(crate) fn push_mu_sigma_to_growth_bar(&mut self) {
        self.growth_bar.set_param("mu", self.mu);
        self.growth_bar.set_param("sigma", self.sigma);
    }

    pub(crate) fn sync_sandboxes_to_engine(&mut self) {
        if self.has_custom_growth_formula()
            || (self.engine.custom_growth().is_some() && !self.growth_bar.is_valid())
        {
            if let Some(compiled) = self.growth_bar.active_compiled() {
                let rule = CompiledGrowthRule::new(compiled.clone(), self.growth_bar.param_slice());
                self.engine.set_custom_growth(Some(rule));
            }
        } else {
            self.engine.set_custom_growth(None);
        }

        if self.use_custom_kernel && !self.is_kernel_locked_by_challenge() {
            if let Some(compiled) = self.kernel_bar.active_compiled() {
                let rule = CompiledKernelRule::new(compiled.clone(), self.kernel_bar.param_slice());
                self.engine.set_custom_kernel(Some(rule));
            }
        } else {
            self.engine.set_custom_kernel(None);
        }
    }

    pub(crate) fn serialize_sandbox_fields(&self) -> SerializedSandboxes {
        let (growth_expr, growth_params) =
            if self.has_custom_growth_formula() && self.growth_bar.is_valid() {
                (
                    Some(self.growth_bar.raw_text.trim().to_string()),
                    self.growth_bar.params_map(),
                )
            } else {
                (None, BTreeMap::new())
            };
        let (kernel_expr, kernel_params) = if self.use_custom_kernel
            && !self.is_kernel_locked_by_challenge()
            && self.kernel_bar.is_valid()
        {
            (
                Some(self.kernel_bar.raw_text.trim().to_string()),
                self.kernel_bar.params_map(),
            )
        } else {
            (None, BTreeMap::new())
        };
        SerializedSandboxes {
            growth_expr,
            growth_params,
            kernel_expr,
            kernel_params,
        }
    }

    pub(crate) fn restore_sandbox_fields(
        &mut self,
        growth_expr: Option<&str>,
        growth_params: &BTreeMap<String, f64>,
        kernel_expr: Option<&str>,
        kernel_params: &BTreeMap<String, f64>,
    ) {
        self.push_mu_sigma_to_growth_bar();
        if let Some(expr) = growth_expr {
            self.growth_bar.set_params_map(growth_params);
            self.growth_bar.set_expression(expr);
            self.pull_mu_sigma_from_growth_bar();
        } else {
            self.growth_bar.set_expression(DEFAULT_GAUSSIAN_GROWTH_EXPR);
            self.push_mu_sigma_to_growth_bar();
        }

        if let Some(expr) = kernel_expr.filter(|_| !self.is_kernel_locked_by_challenge()) {
            self.use_custom_kernel = true;
            self.kernel_bar.set_params_map(kernel_params);
            self.kernel_bar.set_expression(expr);
        } else {
            self.use_custom_kernel = false;
        }
    }

    pub(crate) fn draw_rule_sandboxes(&mut self, ui: &mut egui::Ui) {
        egui::CollapsingHeader::new("Formula Sandboxes G(u) & K(r)")
            .id_salt("lenia_formula_sandboxes")
            .default_open(true)
            .show(ui, |ui| {
                ui.small("Growth Mapping G(u) ∈ [-1, 1] (axis: u)");
                if self.growth_bar.ui(ui) {
                    self.pull_mu_sigma_from_growth_bar();
                    self.sync_engine_parameters();
                    self.user_mutation_nonce = self.user_mutation_nonce.wrapping_add(1);
                }

                ui.add_space(4.0);
                let kernel_locked = self.is_kernel_locked_by_challenge();
                ui.add_enabled_ui(!kernel_locked, |ui| {
                    if ui
                        .checkbox(&mut self.use_custom_kernel, "Custom Radial Kernel K(r)")
                        .changed()
                    {
                        self.sync_engine_parameters();
                        self.user_mutation_nonce = self.user_mutation_nonce.wrapping_add(1);
                    }
                    if self.use_custom_kernel && self.kernel_bar.ui(ui) {
                        self.sync_engine_parameters();
                        self.user_mutation_nonce = self.user_mutation_nonce.wrapping_add(1);
                    }
                });
            });
    }
}
