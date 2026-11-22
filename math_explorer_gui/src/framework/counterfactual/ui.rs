//! Reusable UI transport controls, Divergence Strip, and plot styling for counterfactual twins.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{
    sanitize_perturbation_delta, BranchControlAction, CounterfactualBranchEngine,
    DivergenceSnapshot, MAX_PERTURBATION_DELTA, MIN_PERTURBATION_DELTA,
};
use crate::accessibility::{announce_status, AccessibleHoverText};
use crate::app::palette::SemanticPalette;
use crate::framework::InputMode;
use eframe::egui;

/// Renders `"⑂ Fork Branch Here"`, `"⇄ Swap"`, `"✕ Clear Fork"`, and `δx` controls beside `PlaybackTransport`.
pub struct CounterfactualTransport<'a> {
    engine: &'a mut CounterfactualBranchEngine,
    can_fork: bool,
    is_forked: bool,
    show_delta_control: bool,
    fork_tooltip: &'a str,
}

impl<'a> CounterfactualTransport<'a> {
    /// Creates a new counterfactual transport control strip bound to `engine`.
    #[must_use]
    pub fn new(
        engine: &'a mut CounterfactualBranchEngine,
        can_fork: bool,
        is_forked: bool,
    ) -> Self {
        Self {
            engine,
            can_fork,
            is_forked,
            show_delta_control: true,
            fork_tooltip: "Clone current state (t_k, x(t_k), rng) into baseline Branch A and evolve active Branch B in lockstep",
        }
    }

    /// Overrides the hover tooltip on `"⑂ Fork Branch Here"`.
    #[must_use]
    pub fn fork_tooltip(mut self, tooltip: &'a str) -> Self {
        self.fork_tooltip = tooltip;
        self
    }

    /// Controls whether the inline `δx` state perturbation selector is shown.
    #[must_use]
    pub fn show_delta_control(mut self, show: bool) -> Self {
        self.show_delta_control = show;
        self
    }

    /// Renders the control strip and returns any user-triggered [`BranchControlAction`].
    pub fn show(self, ui: &mut egui::Ui) -> Option<BranchControlAction> {
        let touch = ui
            .ctx()
            .data(|d| d.get_temp::<InputMode>(egui::Id::new("INPUT_MODE")))
            == Some(InputMode::Touch);
        let btn_min = if touch {
            egui::vec2(44.0, 44.0)
        } else {
            egui::vec2(0.0, 26.0)
        };
        let mut action = None;
        ui.horizontal_wrapped(|ui| {
            let fork_resp = ui
                .add_enabled(
                    self.can_fork,
                    egui::Button::new("⑂ Fork Branch Here").min_size(btn_min),
                )
                .accessible_hover_text(self.fork_tooltip)
                .on_disabled_hover_text(
                    "Advance the simulation past t = 0 (Step or Play) to fork a counterfactual twin",
                );
            if fork_resp.clicked() {
                announce_status("Forked counterfactual shadow twin at current step");
                action = Some(BranchControlAction::ForkHere);
            }
            if render_swap_and_clear_buttons(ui, self.is_forked, btn_min, &mut action) {
                return;
            }
            if self.show_delta_control {
                if let Some(delta_action) = render_delta_perturbation_widget(ui, self.engine) {
                    action = Some(delta_action);
                }
            }
        });
        action
    }
}

fn render_swap_and_clear_buttons(
    ui: &mut egui::Ui,
    is_forked: bool,
    btn_min: egui::Vec2,
    action: &mut Option<BranchControlAction>,
) -> bool {
    let swap_resp = ui
        .add_enabled(is_forked, egui::Button::new("⇄ Swap").min_size(btn_min))
        .accessible_hover_text("Swap baseline Branch A and active perturbed Branch B")
        .on_disabled_hover_text("Fork a counterfactual branch first to swap Branch A and Branch B");
    if swap_resp.clicked() {
        announce_status("Swapped Branch A baseline and Branch B active trajectories");
        *action = Some(BranchControlAction::SwapBranches);
        return true;
    }
    let clear_resp = ui
        .add_enabled(
            is_forked,
            egui::Button::new("✕ Clear Fork").min_size(btn_min),
        )
        .accessible_hover_text("Discard shadow twin Branch A and keep active Branch B")
        .on_disabled_hover_text("No active counterfactual fork to clear");
    if clear_resp.clicked() {
        announce_status("Cleared counterfactual shadow twin fork");
        *action = Some(BranchControlAction::ClearFork);
        return true;
    }
    false
}

fn render_delta_perturbation_widget(
    ui: &mut egui::Ui,
    engine: &mut CounterfactualBranchEngine,
) -> Option<BranchControlAction> {
    let mut enabled = engine.perturbation_enabled;
    let mut mag = sanitize_perturbation_delta(engine.slider_delta).max(MIN_PERTURBATION_DELTA);
    let mut changed = false;

    let toggle_resp = ui
        .checkbox(&mut enabled, "δx")
        .accessible_hover_text(
            "Enable state perturbation δx on Branch B at fork time (uncheck for δx = 0 parameter-only or parity fork)",
        );
    if toggle_resp.changed() {
        engine.perturbation_enabled = enabled;
        changed = true;
    }
    if enabled {
        let slider = egui::Slider::new(&mut mag, MIN_PERTURBATION_DELTA..=MAX_PERTURBATION_DELTA)
            .logarithmic(true)
            .custom_formatter(|n, _| format!("{n:.1e}"));
        let slider_resp = ui.add(slider).accessible_hover_text(
            "State perturbation magnitude δx in [1e-8, 1.0] applied on fork",
        );
        if slider_resp.changed() {
            engine.slider_delta = sanitize_perturbation_delta(mag);
            changed = true;
        }
    } else {
        let palette = SemanticPalette::from_visuals(ui.visuals());
        ui.colored_label(palette.muted_text, "0 (exact)")
            .accessible_hover_text("Zero state perturbation (δx = 0.0)");
    }
    changed.then(|| BranchControlAction::SetPerturbation(engine.effective_delta()))
}

/// Renders the theme-accessible, copyable Divergence Strip alongside or above the shared canvas.
pub fn render_divergence_strip(ui: &mut egui::Ui, tool_name: &str, snapshot: &DivergenceSnapshot) {
    let palette = SemanticPalette::from_visuals(ui.visuals());
    let frame = egui::Frame::group(ui.style())
        .fill(palette.surface_fill)
        .stroke(egui::Stroke::new(1.0_f32, palette.graph_edge))
        .inner_margin(egui::Margin::symmetric(8, 6));

    frame.show(ui, |ui| {
        ui.horizontal_wrapped(|ui| {
            ui.colored_label(
                palette.accent,
                egui::RichText::new(format!("⑂ Fork @ t_k = {:.2}", snapshot.fork_time)).strong(),
            )
            .accessible_hover_text(format!(
                "Counterfactual twin anchored at step {} (t_k = {:.4})",
                snapshot.fork_step, snapshot.fork_time
            ));
            ui.separator();
            render_metric_chip(
                ui,
                &palette,
                "||x_B - x_A||_2",
                &format!("{:.3e}", snapshot.l2_separation),
                "Instantaneous Euclidean separation between active Branch B and baseline Branch A",
            );
            render_metric_chip(
                ui,
                &palette,
                "ln||δx||/Δt",
                &format!("{:+.3}", snapshot.log_separation_slope),
                "Finite-time logarithmic separation rate ln(||δx(t)|| / ||δx_ref||) / Δt",
            );
            render_metric_chip(
                ui,
                &palette,
                "Peak ||δx||",
                &format!("{:.3e}", snapshot.peak_separation),
                "Peak Euclidean separation observed since fork time t_k",
            );
            if let Some(ref d1) = snapshot.primary_delta {
                render_metric_chip(ui, &palette, d1.label, &d1.formatted, d1.tooltip);
            }
            if let Some(ref d2) = snapshot.secondary_delta {
                render_metric_chip(ui, &palette, d2.label, &d2.formatted, d2.tooltip);
            }
            ui.separator();
            ui.colored_label(palette.muted_text, "── B (Active)  ╌╌ A (Ghost)");
            let copy_resp = ui
                .small_button("📋 Copy")
                .accessible_hover_text("Copy all live divergence metrics to clipboard");
            if copy_resp.clicked() {
                ui.ctx()
                    .copy_text(snapshot.summary_clipboard_text(tool_name));
                announce_status("Copied counterfactual divergence strip metrics to clipboard");
            }
        });
    });
}

fn render_metric_chip(
    ui: &mut egui::Ui,
    palette: &SemanticPalette,
    label: &str,
    value: &str,
    tooltip: &str,
) {
    let text = format!("{label}: {value}");
    let resp = ui
        .add(egui::Button::new(egui::RichText::new(&text).monospace()).frame(false))
        .accessible_hover_text(format!("{tooltip} (click to copy)"));
    if resp.clicked() {
        ui.ctx().copy_text(text);
        announce_status(&format!("Copied {label} metric to clipboard"));
    }
    let _ = palette;
}

/// Builds a solid `egui_plot::Line` for active `Branch B` with accessible stroke width.
#[must_use]
pub fn branch_b_solid_line(
    name: impl Into<String>,
    points: Vec<[f64; 2]>,
    color: egui::Color32,
) -> egui_plot::Line<'static> {
    egui_plot::Line::new(name.into(), egui_plot::PlotPoints::from(points))
        .color(color)
        .style(egui_plot::LineStyle::Solid)
        .width(2.4_f32)
}

/// Builds a dashed ghost `egui_plot::Line` for unperturbed baseline `Branch A`.
#[must_use]
pub fn branch_a_ghost_line(
    name: impl Into<String>,
    points: Vec<[f64; 2]>,
    color: egui::Color32,
) -> egui_plot::Line<'static> {
    egui_plot::Line::new(name.into(), egui_plot::PlotPoints::from(points))
        .color(color)
        .style(egui_plot::LineStyle::Dashed { length: 6.0_f32 })
        .width(1.7_f32)
}

/// Builds a vertical intervention marker line at `t_k` using `SemanticPalette::accent`.
#[must_use]
pub fn fork_marker_vline(palette: &SemanticPalette, fork_time: f64) -> egui_plot::VLine {
    egui_plot::VLine::new(format!("Fork @ t_k = {fork_time:.2}"), fork_time)
        .color(palette.accent)
        .style(egui_plot::LineStyle::Dashed { length: 4.0_f32 })
        .width(1.6_f32)
}

/// Builds a pinned `"Fork @ t_k"` callout annotation on 2D or projected 3D plots.
#[must_use]
pub fn fork_marker_callout(
    point: [f64; 2],
    fork_time: f64,
    palette: &SemanticPalette,
) -> egui_plot::Callout {
    egui_plot::Callout::new(point, format!("Fork @ t_k = {fork_time:.2}"))
        .text_color(palette.accent)
        .offset(egui::vec2(14.0_f32, -22.0_f32))
}
