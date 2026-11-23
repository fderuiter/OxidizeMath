#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! UI controls and accessible `egui_plot` charts for [`super::CtmcQueueWorkbenchTool`].

use super::state::{ContinuousModelMode, PoissonVariant};
use super::{CtmcQueueWorkbenchTool, MAX_CAPACITY, MAX_SERVERS, MIN_CAPACITY, MIN_SERVERS};
use crate::accessibility::{
    paint_canvas_focus_ring, register_canvas_accessibility, AccessibleHoverText,
};
use crate::app::SemanticPalette;
use crate::framework::{scoped_panel_id, PaneSidePanel};
use eframe::egui;
use eframe::egui::accesskit::Role;
use egui_plot::{Bar, BarChart, Legend, Line, Plot};

impl CtmcQueueWorkbenchTool {
    pub(super) fn render_workspace(&mut self, ui: &mut egui::Ui) {
        let panel_id = scoped_panel_id(ui, "ctmc_queue_workbench_controls");
        PaneSidePanel::left(panel_id)
            .default_width(310.0)
            .show_inside(ui, |panel_ui| {
                self.render_controls_panel(panel_ui);
            });

        egui::CentralPanel::default().show_inside(ui, |central_ui| {
            self.render_plots_area(central_ui);
        });
    }

    pub(super) fn render_controls_panel(&mut self, ui: &mut egui::Ui) {
        let palette = SemanticPalette::from_visuals(ui.visuals());
        ui.heading("CTMC, Poisson & Queues");

        ui.label("Continuous-Time Regime");
        ui.horizontal_wrapped(|ui| {
            for m in ContinuousModelMode::ALL {
                if ui
                    .selectable_label(self.mode == m, m.label())
                    .accessible_hover_text("Select continuous-time Markov, M/M/c queue, or Poisson model.")
                    .clicked()
                {
                    self.set_mode(m);
                }
            }
        });
        ui.separator();

        let mut changed = match self.mode {
            ContinuousModelMode::SlowRateCtmc => self.render_ctmc_sliders(ui),
            ContinuousModelMode::MmcBirthDeathQueue => self.render_queue_sliders(ui),
            ContinuousModelMode::InhomogeneousPoisson => self.render_poisson_sliders(ui),
        };
        ui.horizontal(|ui| {
            ui.label("Seed:");
            changed |= ui
                .add(egui::DragValue::new(&mut self.seed).range(0..=1_000_000))
                .changed();
        });
        if changed {
            self.recompute_all();
        }

        ui.separator();
        self.render_diagnostics(ui, &palette);
    }

    fn render_ctmc_sliders(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        changed |= ui
            .add(egui::Slider::new(&mut self.rate_log10, -7.0..=0.0).text("log10(scale)"))
            .accessible_hover_text("Log10 generator rate scale down to 1e-7.")
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.forward_weight, 0.2..=5.0).text("Forward w_f"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.backward_weight, 0.2..=5.0).text("Backward w_b"))
            .changed();
        changed
    }

    fn render_queue_sliders(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        changed |= ui
            .add(egui::Slider::new(&mut self.servers, MIN_SERVERS..=MAX_SERVERS).text("Servers c"))
            .accessible_hover_text("Number of parallel exponential servers in M/M/c.")
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.service_rate, 0.5..=10.0).text("Service μ"))
            .changed();
        let max_lambda = (self.servers as f64 * self.service_rate * 0.95).clamp(0.25, 25.0);
        self.arrival_rate = self.arrival_rate.min(max_lambda);
        changed |= ui
            .add(egui::Slider::new(&mut self.arrival_rate, 0.2..=max_lambda).text("Arrival λ"))
            .changed();
        let min_cap = self.servers.max(MIN_CAPACITY);
        self.capacity = self.capacity.max(min_cap);
        changed |= ui
            .add(egui::Slider::new(&mut self.capacity, min_cap..=MAX_CAPACITY).text("CTMC Cap K"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.horizon, 5.0..=100.0).text("Horizon T"))
            .changed();
        changed
    }

    fn render_poisson_sliders(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.horizontal(|ui| {
            ui.label("Intensity:");
            for v in [PoissonVariant::Sinusoidal, PoissonVariant::StepBurst] {
                changed |= ui
                    .selectable_value(&mut self.poisson_variant, v, v.label())
                    .changed();
            }
        });
        changed |= ui
            .add(egui::Slider::new(&mut self.arrival_rate, 0.2..=25.0).text("Base Rate λ0"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.horizon, 5.0..=100.0).text("Horizon T"))
            .changed();
        changed
    }

    fn render_diagnostics(&self, ui: &mut egui::Ui, palette: &SemanticPalette) {
        ui.heading("Diagnostics");
        if let Some(ref err) = self.error_msg {
            ui.colored_label(palette.error, err);
            return;
        }
        match self.mode {
            ContinuousModelMode::SlowRateCtmc => {
                ui.label(format!("Rate scale: 10^({:.1})", self.rate_log10));
                if let Some(pi) = self.ctmc_steady {
                    ui.colored_label(
                        palette.success,
                        format!("QR π = [{:.4}, {:.4}, {:.4}]", pi[0], pi[1], pi[2]),
                    );
                }
                ui.label(format!(
                    "Gillespie = [{:.4}, {:.4}, {:.4}]",
                    self.ctmc_empirical[0], self.ctmc_empirical[1], self.ctmc_empirical[2]
                ));
            }
            ContinuousModelMode::MmcBirthDeathQueue => {
                if let Some(m) = self.queue_metrics {
                    ui.colored_label(
                        palette.success,
                        format!(
                            "ρ = {:.3} | p0 = {:.3} | Erlang-C = {:.3}",
                            m.utilization, m.idle_probability, m.prob_wait
                        ),
                    );
                    ui.label(format!(
                        "L = {:.3}, Lq = {:.3} | W = {:.3}, Wq = {:.3}",
                        m.mean_in_system, m.mean_in_queue, m.mean_sojourn_time, m.mean_waiting_time
                    ));
                }
            }
            ContinuousModelMode::InhomogeneousPoisson => {
                ui.colored_label(
                    palette.success,
                    format!("Expected Λ(T) = {:.2} arrivals", self.expected_arrivals),
                );
                ui.label(format!("Realized N(T) = {} thinned arrivals", self.realized_arrivals));
            }
        }
        ui.small(format!("Solved in {:.2} ms", self.last_compute_ms));
    }

    fn render_plots_area(&self, ui: &mut egui::Ui) {
        let palette = SemanticPalette::from_visuals(ui.visuals());
        let avail_h = ui.available_height().max(260.0);
        match self.mode {
            ContinuousModelMode::SlowRateCtmc => {
                let top_h = (avail_h * 0.50).max(150.0);
                let bot_h = (avail_h - top_h - 18.0).max(130.0);
                self.render_ctmc_bars(ui, &palette, top_h);
                ui.add_space(6.0);
                self.render_trajectory_plot(
                    ui,
                    &palette,
                    bot_h,
                    "ctmc_gillespie_plot",
                    "Scaled Time s·t",
                    "State X(t)",
                    "Gillespie Sample Trajectory",
                );
            }
            ContinuousModelMode::MmcBirthDeathQueue => {
                let top_h = (avail_h * 0.50).max(150.0);
                let bot_h = (avail_h - top_h - 18.0).max(130.0);
                self.render_queue_pmf_bars(ui, &palette, top_h);
                ui.add_space(6.0);
                self.render_trajectory_plot(
                    ui,
                    &palette,
                    bot_h,
                    "mmc_queue_traj_plot",
                    "Time t",
                    "Customers N(t)",
                    "Birth-Death Queue Trajectory vs Little's Law L",
                );
            }
            ContinuousModelMode::InhomogeneousPoisson => {
                self.render_poisson_plot(ui, &palette, (avail_h - 24.0).max(220.0));
            }
        }
    }

    fn render_ctmc_bars(&self, ui: &mut egui::Ui, palette: &SemanticPalette, height: f32) {
        ui.strong("Rate-Preconditioned QR Steady State (G^T π = 0) vs Gillespie Sojourn");
        let resp = Plot::new("ctmc_steady_vs_sojourn_plot")
            .height(height)
            .x_axis_label("State i")
            .y_axis_label("Steady-State Mass")
            .include_y(0.0)
            .include_y(1.0)
            .legend(Legend::default())
            .show(ui, |plot_ui| {
                if let Some(pi) = self.ctmc_steady {
                    let bars: Vec<Bar> = (0..3)
                        .map(|s| Bar::new(s as f64 - 0.14, pi[s]).width(0.24))
                        .collect();
                    plot_ui.bar_chart(
                        BarChart::new("Exact QR π", bars).color(palette.series_color(0)),
                    );
                }
                let emp: Vec<Bar> = (0..3)
                    .map(|s| Bar::new(s as f64 + 0.14, self.ctmc_empirical[s]).width(0.24))
                    .collect();
                plot_ui.bar_chart(
                    BarChart::new("Gillespie Sojourn", emp).color(palette.series_color(1)),
                );
            });
        let summary = format!("Slow-rate CTMC scale 10^{:.1}, steady {:?}", self.rate_log10, self.ctmc_steady);
        register_canvas_accessibility(
            ui,
            &resp.response,
            Role::GraphicsDocument,
            "CTMC Exact QR Steady-State vs Gillespie Sojourn Bar Chart",
            Some(&summary),
        );
        paint_canvas_focus_ring(ui, &resp.response);
    }

    fn render_queue_pmf_bars(&self, ui: &mut egui::Ui, palette: &SemanticPalette, height: f32) {
        ui.strong("M/M/c Analytic Erlang-C p_k vs Truncated CTMC steady_state()");
        let resp = Plot::new("mmc_pmf_comparison_plot")
            .height(height)
            .x_axis_label("Customers in System k")
            .y_axis_label("Stationary Probability p_k")
            .include_y(0.0)
            .legend(Legend::default())
            .show(ui, |plot_ui| {
                let b_ana: Vec<Bar> = self
                    .queue_analytic_pk
                    .iter()
                    .enumerate()
                    .map(|(k, &p)| Bar::new(k as f64 - 0.16, p).width(0.28))
                    .collect();
                let b_ctmc: Vec<Bar> = self
                    .queue_ctmc_pk
                    .iter()
                    .enumerate()
                    .map(|(k, &p)| Bar::new(k as f64 + 0.16, p).width(0.28))
                    .collect();
                plot_ui.bar_chart(
                    BarChart::new("Analytic M/M/c p_k", b_ana).color(palette.series_color(0)),
                );
                plot_ui.bar_chart(
                    BarChart::new("Truncated CTMC π_k", b_ctmc).color(palette.series_color(2)),
                );
            });
        let summary = format!(
            "M/M/{} queue arrival {:.2} service {:.2} stationary distribution",
            self.servers, self.arrival_rate, self.service_rate
        );
        register_canvas_accessibility(
            ui,
            &resp.response,
            Role::GraphicsDocument,
            "M/M/c Queue Stationary Probability Bar Chart",
            Some(&summary),
        );
        paint_canvas_focus_ring(ui, &resp.response);
    }

    #[allow(clippy::too_many_arguments)]
    fn render_trajectory_plot(
        &self,
        ui: &mut egui::Ui,
        palette: &SemanticPalette,
        height: f32,
        plot_id: &'static str,
        x_label: &'static str,
        y_label: &'static str,
        heading: &'static str,
    ) {
        ui.strong(heading);
        let resp = Plot::new(plot_id)
            .height(height)
            .x_axis_label(x_label)
            .y_axis_label(y_label)
            .include_y(0.0)
            .legend(Legend::default())
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new("Sample Path", &self.primary_curve[..])
                        .color(palette.series_color(1))
                        .width(1.8_f32),
                );
                if !self.secondary_curve.is_empty() {
                    plot_ui.line(
                        Line::new("Mean L", &self.secondary_curve[..])
                            .color(palette.warning)
                            .width(2.0_f32),
                    );
                }
            });
        register_canvas_accessibility(
            ui,
            &resp.response,
            Role::GraphicsDocument,
            heading,
            Some("Continuous-time jump trajectory simulated with OxidizeRng."),
        );
        paint_canvas_focus_ring(ui, &resp.response);
    }

    fn render_poisson_plot(&self, ui: &mut egui::Ui, palette: &SemanticPalette, height: f32) {
        ui.strong("Lewis-Shedler Thinned Counting Process N(t) vs Cumulative Intensity Λ(t)");
        let resp = Plot::new("inhomogeneous_poisson_counting_plot")
            .height(height)
            .x_axis_label("Time t")
            .y_axis_label("Cumulative Arrivals")
            .include_y(0.0)
            .legend(Legend::default())
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new("Thinned N(t)", &self.primary_curve[..])
                        .color(palette.series_color(0))
                        .width(2.0_f32),
                );
                plot_ui.line(
                    Line::new("Integrated Mean Λ(t)", &self.secondary_curve[..])
                        .color(palette.series_color(1))
                        .width(2.2_f32),
                );
            });
        let summary = format!(
            "Inhomogeneous Poisson ({}) expected {:.2}, realized {}",
            self.poisson_variant.label(),
            self.expected_arrivals,
            self.realized_arrivals
        );
        register_canvas_accessibility(
            ui,
            &resp.response,
            Role::GraphicsDocument,
            "Inhomogeneous Poisson Counting Process Plot",
            Some(&summary),
        );
        paint_canvas_focus_ring(ui, &resp.response);
    }
}

// [cite:clinical_trials_statistics]
