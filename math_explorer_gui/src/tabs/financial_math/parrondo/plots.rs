//! Plot rendering for [`super::ParrondoParadoxTool`].

use super::{ParrondoParadoxTool, ParrondoViewMode};
use crate::accessibility::{paint_canvas_focus_ring, register_canvas_accessibility};
use crate::app::SemanticPalette;
use eframe::egui;
use eframe::egui::accesskit::Role;
use egui_plot::{Bar, BarChart, Legend, Line, LineStyle, Plot, PlotPoint};
use math_explorer::pure_math::statistics::markov::parrondo::{
    ParrondoPhaseGridConfig, ParrondoPhasePoint, ParrondoReport,
};

pub(super) fn extract_phase_boundaries(
    cfg: &ParrondoPhaseGridConfig,
    pts: &[ParrondoPhasePoint],
    boundary_a: &mut Vec<PlotPoint>,
    boundary_b: &mut Vec<PlotPoint>,
    boundary_sched: &mut Vec<PlotPoint>,
) {
    boundary_a.clear();
    boundary_b.clear();
    boundary_sched.clear();
    if cfg.p_a_steps < 2 || cfg.epsilon_steps < 2 || pts.len() < cfg.p_a_steps * cfg.epsilon_steps {
        return;
    }
    boundary_a.push(PlotPoint::new(0.5, cfg.epsilon_min));
    boundary_a.push(PlotPoint::new(0.5, cfg.epsilon_max));

    for iy in 0..(cfg.epsilon_steps - 1) {
        let row0 = &pts[iy * cfg.p_a_steps];
        let row1 = &pts[(iy + 1) * cfg.p_a_steps];
        if (row0.gain_b >= 0.0 && row1.gain_b <= 0.0) || (row0.gain_b <= 0.0 && row1.gain_b >= 0.0)
        {
            let denom = row1.gain_b - row0.gain_b;
            let frac = if denom.abs() > 1e-12 {
                (-row0.gain_b / denom).clamp(0.0, 1.0)
            } else {
                0.5
            };
            let eps_zero = row0.epsilon + frac * (row1.epsilon - row0.epsilon);
            boundary_b.push(PlotPoint::new(cfg.p_a_min, eps_zero));
            boundary_b.push(PlotPoint::new(cfg.p_a_max, eps_zero));
            break;
        }
    }

    for iy in 0..cfg.epsilon_steps {
        let row = &pts[iy * cfg.p_a_steps..(iy + 1) * cfg.p_a_steps];
        for win in row.windows(2) {
            let (p0, p1) = (win[0], win[1]);
            if (p0.gain_schedule <= 0.0 && p1.gain_schedule >= 0.0)
                || (p0.gain_schedule >= 0.0 && p1.gain_schedule <= 0.0)
            {
                let denom = p1.gain_schedule - p0.gain_schedule;
                let frac = if denom.abs() > 1e-12 {
                    (-p0.gain_schedule / denom).clamp(0.0, 1.0)
                } else {
                    0.5
                };
                let pa_zero = p0.p_a + frac * (p1.p_a - p0.p_a);
                boundary_sched.push(PlotPoint::new(pa_zero, p0.epsilon));
                break;
            }
        }
    }
}

impl ParrondoParadoxTool {
    pub(super) fn render_central_views(&mut self, ui: &mut egui::Ui) {
        let palette = SemanticPalette::from_visuals(ui.visuals());
        if let Some(ref err) = self.error_msg {
            ui.colored_label(palette.error, err);
            return;
        }
        let Some(report) = self.report.clone() else {
            return;
        };

        let avail_h = ui.available_height().max(260.0);
        match self.view_mode {
            ParrondoViewMode::SplitOverview => {
                let top_h = (avail_h * 0.46).max(150.0);
                let bot_h = (avail_h - top_h - 16.0).max(160.0);
                self.render_trajectories_card(ui, &report, &palette, top_h);
                ui.add_space(6.0);
                if self.atlas.is_visible() {
                    ui.columns(2, |cols| {
                        self.render_trap_occupancy_card(&mut cols[0], &report, &palette, bot_h);
                        self.render_phase_atlas_card(&mut cols[1], bot_h);
                    });
                } else {
                    self.render_trap_occupancy_card(ui, &report, &palette, bot_h);
                }
            }
            ParrondoViewMode::WealthTrajectories => {
                self.render_trajectories_card(ui, &report, &palette, avail_h - 28.0);
                if self.atlas.is_visible() {
                    self.render_phase_atlas_card(ui, 220.0);
                }
            }
            ParrondoViewMode::TrapOccupancy => {
                self.render_trap_occupancy_card(ui, &report, &palette, avail_h - 28.0);
                if self.atlas.is_visible() {
                    self.render_phase_atlas_card(ui, 220.0);
                }
            }
            ParrondoViewMode::PhaseWedge => {
                self.atlas.set_visible(true);
                self.render_phase_atlas_card(ui, avail_h - 28.0);
            }
        }
    }

    fn render_kelly_badge_strip(&self, ui: &mut egui::Ui, sched: &str, palette: &SemanticPalette) {
        ui.horizontal_wrapped(|ui| {
            ui.strong("Wealth Trajectories & Markov Drift");
            for (lbl, badge) in [
                ("A", self.kelly_a),
                ("B", self.kelly_b),
                (sched, self.kelly_schedule),
            ] {
                let col = if badge.is_no_edge() {
                    palette.error
                } else {
                    palette.success
                };
                ui.colored_label(col, format!("{lbl}: {}", badge.badge_text()));
            }
        });
    }

    fn render_trajectories_card(
        &mut self,
        ui: &mut egui::Ui,
        report: &ParrondoReport,
        palette: &SemanticPalette,
        height: f32,
    ) {
        let sched = self.active_schedule.as_string();
        self.render_kelly_badge_strip(ui, &sched, palette);

        let summary = format!(
            "Step {} mean wealth: Game A {:.2}, Game B {:.2}, Mixture {:.2}, Schedule {} {:.2}; exact schedule drift {:+.4}",
            self.num_steps,
            self.traj_a.last().map_or(0.0, |p| p.y),
            self.traj_b.last().map_or(0.0, |p| p.y),
            self.traj_mix.last().map_or(0.0, |p| p.y),
            sched,
            self.traj_schedule.last().map_or(0.0, |p| p.y),
            report.gain_schedule
        );
        let Self {
            traj_a,
            traj_b,
            traj_mix,
            traj_schedule,
            drift_a,
            drift_b,
            drift_mix,
            drift_schedule,
            callouts,
            ..
        } = self;
        let plot_resp = Plot::new("parrondo_trajectories_plot")
            .height(height.max(120.0))
            .x_axis_label("Step t")
            .y_axis_label("Expected / Mean Wealth W(t)")
            .legend(Legend::default())
            .show(ui, |plot_ui| {
                draw_trajectory_lines(
                    plot_ui,
                    palette,
                    &sched,
                    [traj_a, traj_b, traj_mix, traj_schedule],
                    [drift_a, drift_b, drift_mix, drift_schedule],
                );
                callouts.render_on_plot(plot_ui);
            });
        register_canvas_accessibility(
            ui,
            &plot_resp.response,
            Role::GraphicsDocument,
            "Parrondo Ensemble Wealth Trajectories and Exact Markov Drift",
            Some(&summary),
        );
        paint_canvas_focus_ring(ui, &plot_resp.response);
    }

    fn render_trap_occupancy_card(
        &self,
        ui: &mut egui::Ui,
        report: &ParrondoReport,
        palette: &SemanticPalette,
        height: f32,
    ) {
        ui.strong(format!(
            "Modulo-M Trap Occupancy (M = {}, Trap W ≡ 0)",
            self.modulus
        ));
        let plot_resp = Plot::new("parrondo_trap_occupancy_plot")
            .height(height.max(120.0))
            .x_axis_label("Capital Residue r = W mod M (r = 0 is Losing Trap)")
            .y_axis_label("Stationary / Empirical Occupancy π(r)")
            .include_y(0.0)
            .legend(Legend::default())
            .show(ui, |plot_ui| {
                self.draw_occupancy_bars(plot_ui, report, palette);
            });
        let summary = format!(
            "Trap residue r=0 occupancy: Game B {:.4} vs Schedule {:.4} (empirical {:.4})",
            report.pi_b[0], report.pi_schedule_mean[0], self.emp_occ_schedule[0]
        );
        register_canvas_accessibility(
            ui,
            &plot_resp.response,
            Role::GraphicsDocument,
            "Modulo-M Stationary and Empirical Trap Occupancy Bar Chart",
            Some(&summary),
        );
        paint_canvas_focus_ring(ui, &plot_resp.response);
    }

    fn draw_occupancy_bars(
        &self,
        plot_ui: &mut egui_plot::PlotUi<'_>,
        report: &ParrondoReport,
        palette: &SemanticPalette,
    ) {
        let m = self.modulus.min(report.pi_b.len());
        let width = 0.18;
        let bars_b: Vec<Bar> = (0..m)
            .map(|r| Bar::new(r as f64 - 0.27, report.pi_b[r]).width(width))
            .collect();
        let bars_mix: Vec<Bar> = (0..m)
            .map(|r| Bar::new(r as f64 - 0.09, report.pi_mix[r]).width(width))
            .collect();
        let bars_sched: Vec<Bar> = (0..m)
            .map(|r| Bar::new(r as f64 + 0.09, report.pi_schedule_mean[r]).width(width))
            .collect();
        let bars_emp: Vec<Bar> = (0..m)
            .map(|r| Bar::new(r as f64 + 0.27, self.emp_occ_schedule[r]).width(width))
            .collect();

        plot_ui.bar_chart(BarChart::new("π_B (Pure B)", bars_b).color(palette.series_color(1)));
        plot_ui.bar_chart(BarChart::new("π_mix (50/50)", bars_mix).color(palette.series_color(2)));
        plot_ui
            .bar_chart(BarChart::new("π_sched (Exact)", bars_sched).color(palette.series_color(3)));
        plot_ui
            .bar_chart(BarChart::new("Empirical Sched", bars_emp).color(palette.series_color(4)));
    }
}

fn draw_trajectory_lines<'a>(
    plot_ui: &mut egui_plot::PlotUi<'a>,
    palette: &SemanticPalette,
    sched: &str,
    trajs: [&'a [PlotPoint]; 4],
    drifts: [&'a [PlotPoint]; 4],
) {
    let dashed = LineStyle::Dashed { length: 8.0 };
    let cols = [
        palette.series_color(0),
        palette.series_color(1),
        palette.series_color(2),
        palette.series_color(3),
    ];
    plot_ui.line(Line::new("Game A (Ensemble)", trajs[0]).color(cols[0]));
    plot_ui.line(Line::new("Game B (Ensemble)", trajs[1]).color(cols[1]));
    plot_ui.line(Line::new("50/50 Mix (Ensemble)", trajs[2]).color(cols[2]));
    plot_ui.line(
        Line::new(format!("Schedule {sched} (Ensemble)"), trajs[3])
            .color(cols[3])
            .width(2.2_f32),
    );
    plot_ui.line(
        Line::new("Game A Markov Drift", drifts[0])
            .color(cols[0])
            .style(dashed),
    );
    plot_ui.line(
        Line::new("Game B Markov Drift", drifts[1])
            .color(cols[1])
            .style(dashed),
    );
    plot_ui.line(
        Line::new("50/50 Mix Markov Drift", drifts[2])
            .color(cols[2])
            .style(dashed),
    );
    plot_ui.line(
        Line::new(format!("Schedule {sched} Markov Drift"), drifts[3])
            .color(cols[3])
            .style(dashed),
    );
}

// [cite:clinical_trials_statistics]
