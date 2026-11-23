//! Shared `BranchDivergenceHud` widget, transport state descriptors, and ARIA live announcements.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::BranchDivergenceMetrics;
use crate::accessibility::{announce_live_status, AccessibleHoverText};
use crate::app::palette::SemanticPalette;
use eframe::egui;

/// Current timeline scrubber and branch status projected onto [`crate::framework::PlaybackTransport`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimelineTransportState {
    /// Active simulation step index $k$.
    pub current_step: usize,
    /// Earliest retained checkpoint step in the ring buffer.
    pub min_step: usize,
    /// Latest reachable or buffered simulation step.
    pub max_step: usize,
    /// Simulation time $t_k$ corresponding to `current_step`.
    pub current_time: f64,
    /// Display unit suffix for `current_time` (e.g. `"d"`, `"s"`, `"ms"`, `"steps"`).
    pub time_unit: &'static str,
    /// Whether an unperturbed shadow-twin branch is currently active.
    pub fork_active: bool,
    /// Step index $k$ where the active shadow twin was forked.
    pub fork_step: Option<usize>,
    /// Simulation time $t_k$ where the active shadow twin was forked.
    pub fork_time: Option<f64>,
}

impl TimelineTransportState {
    /// Returns `true` when stepping backward is valid (`current_step > min_step`).
    #[inline]
    #[must_use]
    pub const fn can_step_back(&self) -> bool {
        self.current_step > self.min_step
    }

    /// Returns `true` when scrubbing across multiple buffered steps is available.
    #[inline]
    #[must_use]
    pub const fn can_scrub(&self) -> bool {
        self.max_step > self.min_step
    }
}

/// User timeline or shadow-branch action emitted by [`crate::framework::PlaybackTransport`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimelineBranchAction {
    /// Step backward to the preceding checkpoint without resetting to $t = 0$.
    StepBack,
    /// Scrub directly to target step $k$ in the ring buffer.
    ScrubTo(usize),
    /// Fork an unperturbed shadow twin at the current step $t_k$, or clear if already forked.
    ToggleForkBranch,
    /// Explicitly clear the active shadow-twin branch overlay.
    ClearBranch,
}

/// Structured branch provenance from fork instant $t_k$ included in dataset and figure exports.
#[derive(Debug, Clone, PartialEq)]
pub struct BranchMetadata {
    /// Fork step index $k$.
    pub fork_step: usize,
    /// Fork timestamp $t_k$.
    pub fork_time: f64,
    /// Display unit for $t_k$.
    pub time_unit: &'static str,
    /// Frozen baseline shadow-twin parameter summary at $t_k$.
    pub baseline_summary: String,
    /// Active counterfactual branch parameter summary.
    pub branch_summary: String,
    /// Latest divergence metrics accumulated since $t_k$.
    pub divergence: BranchDivergenceMetrics,
}

impl BranchMetadata {
    /// Formats a compact single-line provenance descriptor for CSV, JSON, SVG, and PNG exports.
    #[must_use]
    pub fn summary_tag(&self) -> String {
        format!(
            "fork_step={}, t_k={:.4}{}, baseline=[{}], counterfactual=[{}], l2={:.6e}, peak_abs_delta={:.6e}, area_delta={:.6e}, log_slope={:.6e}",
            self.fork_step,
            self.fork_time,
            self.time_unit,
            self.baseline_summary,
            self.branch_summary,
            self.divergence.instantaneous_l2,
            self.divergence.peak_abs_delta,
            self.divergence.cumulative_area_delta,
            self.divergence.log_divergence_slope,
        )
    }
}

/// Dispatches a polite ARIA live-region announcement when a shadow-twin branch is forked at $t_k$.
pub fn announce_branch_forked(
    ctx: &egui::Context,
    tool_name: &str,
    fork_step: usize,
    fork_time: f64,
    time_unit: &str,
) {
    let unit_suffix = if time_unit.is_empty() {
        String::new()
    } else {
        format!(" {time_unit}")
    };
    announce_live_status(
        ctx,
        format!(
            "Forked shadow twin branch in {tool_name} at step {fork_step} (t_k = {fork_time:.3}{unit_suffix})"
        ),
    );
}

/// Dispatches a polite ARIA live-region announcement when a shadow-twin branch is cleared.
pub fn announce_branch_cleared(ctx: &egui::Context, tool_name: &str) {
    announce_live_status(
        ctx,
        format!("Cleared shadow twin branch overlay in {tool_name}"),
    );
}

/// Shared accessible HUD displaying shadow-twin branch status, legend swatches, and divergence readouts.
pub struct BranchDivergenceHud<'a> {
    id_salt: &'a str,
    tool_name: &'a str,
    fork_step: usize,
    fork_time: f64,
    time_unit: &'a str,
    metrics: BranchDivergenceMetrics,
    extra_summary: Option<&'a str>,
}

impl<'a> BranchDivergenceHud<'a> {
    /// Creates a divergence HUD for an active shadow-twin branch forked at `(fork_step, fork_time)`.
    #[must_use]
    pub const fn new(
        id_salt: &'a str,
        tool_name: &'a str,
        fork_step: usize,
        fork_time: f64,
        time_unit: &'a str,
        metrics: BranchDivergenceMetrics,
    ) -> Self {
        Self {
            id_salt,
            tool_name,
            fork_step,
            fork_time,
            time_unit,
            metrics,
            extra_summary: None,
        }
    }

    /// Attaches an optional domain-specific summary line (e.g. SIR peak-infection reduction).
    #[must_use]
    pub const fn with_extra_summary(mut self, summary: &'a str) -> Self {
        self.extra_summary = Some(summary);
        self
    }

    /// Renders the HUD card using [`SemanticPalette`] tokens and returns `true` if `Clear Branch` was clicked.
    pub fn show(self, ui: &mut egui::Ui) -> bool {
        let palette = SemanticPalette::from_visuals(ui.visuals());
        let mut clear_clicked = false;
        let frame = egui::Frame::group(ui.style())
            .fill(palette.surface_fill)
            .stroke(egui::Stroke::new(1.0_f32, palette.accent));

        let resp = frame.show(ui, |ui| {
            self.render_header_row(ui, palette, &mut clear_clicked);
            self.render_legend_row(ui, palette);
            ui.add_space(2.0);
            self.render_metrics_grid(ui, palette);
            if let Some(extra) = self.extra_summary {
                ui.add_space(2.0);
                ui.label(egui::RichText::new(extra).small().color(palette.info));
            }
        });

        let a11y_summary = format!(
            "{} shadow branch from step {}, L2 {:.3e}, peak delta {:.3e}, area delta {:.3e}, log slope {:.3}",
            self.tool_name,
            self.fork_step,
            self.metrics.instantaneous_l2,
            self.metrics.peak_abs_delta,
            self.metrics.cumulative_area_delta,
            self.metrics.log_divergence_slope
        );
        resp.response.widget_info(move || {
            egui::WidgetInfo::labeled(egui::WidgetType::Label, true, a11y_summary.clone())
        });

        if clear_clicked {
            announce_branch_cleared(ui.ctx(), self.tool_name);
        }
        clear_clicked
    }

    fn render_header_row(
        &self,
        ui: &mut egui::Ui,
        palette: SemanticPalette,
        clear_clicked: &mut bool,
    ) {
        ui.horizontal_wrapped(|ui| {
            let unit = if self.time_unit.is_empty() {
                ""
            } else {
                self.time_unit
            };
            ui.label(
                egui::RichText::new(format!(
                    "⑂ Shadow Twin Active @ t_k = {:.3}{} (step {})",
                    self.fork_time, unit, self.fork_step
                ))
                .strong()
                .color(palette.accent),
            );
            let clear_a11y = format!("Clear {} shadow twin branch", self.tool_name);
            let btn = ui.small_button(egui::RichText::new("✕ Clear Branch").color(palette.warning));
            btn.widget_info({
                let label = clear_a11y.clone();
                move || egui::WidgetInfo::labeled(egui::WidgetType::Button, true, label.clone())
            });
            if btn.accessible_hover_text(&clear_a11y).clicked() {
                *clear_clicked = true;
            }
        });
    }

    fn render_legend_row(&self, ui: &mut egui::Ui, palette: SemanticPalette) {
        ui.horizontal_wrapped(|ui| {
            ui.label(
                egui::RichText::new("━━ Counterfactual Branch")
                    .small()
                    .color(palette.series_color(0)),
            );
            ui.separator();
            ui.label(
                egui::RichText::new("╌╌ Shadow Baseline (Unperturbed)")
                    .small()
                    .color(palette.series_color(1)),
            );
        });
    }

    fn render_metrics_grid(&self, ui: &mut egui::Ui, palette: SemanticPalette) {
        egui::Grid::new((self.id_salt, "branch_div_metrics"))
            .num_columns(2)
            .spacing([10.0, 2.0])
            .show(ui, |ui| {
                render_metric_pair(
                    ui,
                    palette,
                    "Instantaneous L2:",
                    format!("{:.4e}", self.metrics.instantaneous_l2),
                    "Euclidean L2 distance ||x_branch(t) - x_shadow(t)||_2",
                );
                render_metric_pair(
                    ui,
                    palette,
                    "Peak |Δ|:",
                    format!("{:.4e}", self.metrics.peak_abs_delta),
                    "Peak element-wise absolute difference since fork instant t_k",
                );
                render_metric_pair(
                    ui,
                    palette,
                    "Area ∫||Δ||dt:",
                    format!("{:.4e}", self.metrics.cumulative_area_delta),
                    "Cumulative trapezoidal integral of L2 divergence since t_k",
                );
                render_metric_pair(
                    ui,
                    palette,
                    "Log Slope λ:",
                    format!("{:+.4}", self.metrics.log_divergence_slope),
                    "Empirical exponential divergence slope d(ln ||Δx||)/dt since t_k",
                );
            });
    }
}

fn render_metric_pair(
    ui: &mut egui::Ui,
    palette: SemanticPalette,
    label: &str,
    value: String,
    tooltip: &str,
) {
    ui.label(egui::RichText::new(label).small().color(palette.muted_text))
        .accessible_hover_text(tooltip);
    let tone = if value.starts_with("0.0000e0") || value == "+0.0000" {
        palette.success
    } else {
        palette.info
    };
    ui.label(egui::RichText::new(value).monospace().small().color(tone))
        .accessible_hover_text(tooltip);
    ui.end_row();
}
