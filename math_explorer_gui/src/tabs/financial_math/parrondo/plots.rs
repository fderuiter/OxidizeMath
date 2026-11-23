//! Plot rendering and offscreen export builders for [`super::ParrondoParadoxTool`].

use super::export::to_pairs;
use super::{ParrondoParadoxTool, ParrondoViewMode};
use crate::accessibility::{paint_canvas_focus_ring, register_canvas_accessibility};
use crate::app::SemanticPalette;
use eframe::egui;
use eframe::egui::accesskit::Role;
use egui_plot::{Bar, BarChart, Legend, Line, LineStyle, MarkerShape, Plot, PlotPoint, Points};
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
    pub(super) fn render_central_views(&self, ui: &mut egui::Ui) {
        let palette = SemanticPalette::from_visuals(ui.visuals());
        if let Some(ref err) = self.error_msg {
            ui.colored_label(palette.error, err);
            return;
        }
        let Some(ref report) = self.report else {
            return;
        };

        let avail_h = ui.available_height().max(260.0);
        match self.view_mode {
            ParrondoViewMode::SplitOverview => {
                let top_h = (avail_h * 0.48).max(160.0);
                let bot_h = (avail_h - top_h - 16.0).max(150.0);
                self.render_trajectories_card(ui, report, &palette, top_h);
                ui.add_space(6.0);
                ui.columns(2, |cols| {
                    self.render_trap_occupancy_card(&mut cols[0], report, &palette, bot_h);
                    self.render_phase_wedge_card(&mut cols[1], report, &palette, bot_h);
                });
            }
            ParrondoViewMode::WealthTrajectories => {
                self.render_trajectories_card(ui, report, &palette, avail_h - 28.0);
            }
            ParrondoViewMode::TrapOccupancy => {
                self.render_trap_occupancy_card(ui, report, &palette, avail_h - 28.0);
            }
            ParrondoViewMode::PhaseWedge => {
                self.render_phase_wedge_card(ui, report, &palette, avail_h - 28.0);
            }
        }
    }

    fn render_trajectories_card(
        &self,
        ui: &mut egui::Ui,
        report: &ParrondoReport,
        palette: &SemanticPalette,
        height: f32,
    ) {
        let sched = self.active_schedule.as_string();
        ui.horizontal_wrapped(|ui| {
            ui.strong("Wealth Trajectories & Markov Drift");
            ui.colored_label(
                if self.kelly_a.is_no_edge() {
                    palette.error
                } else {
                    palette.success
                },
                format!("A: {}", self.kelly_a.badge_text()),
            );
            ui.colored_label(
                if self.kelly_b.is_no_edge() {
                    palette.error
                } else {
                    palette.success
                },
                format!("B: {}", self.kelly_b.badge_text()),
            );
            ui.colored_label(
                if self.kelly_schedule.is_no_edge() {
                    palette.error
                } else {
                    palette.success
                },
                format!("{sched}: {}", self.kelly_schedule.badge_text()),
            );
        });

        let plot_resp = Plot::new("parrondo_trajectories_plot")
            .height(height.max(120.0))
            .x_axis_label("Step t")
            .y_axis_label("Expected / Mean Wealth W(t)")
            .legend(Legend::default())
            .show(ui, |plot_ui| {
                self.draw_trajectory_series(plot_ui, palette, &sched);
            });
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
        register_canvas_accessibility(
            ui,
            &plot_resp.response,
            Role::GraphicsDocument,
            "Parrondo Ensemble Wealth Trajectories and Exact Markov Drift",
            Some(&summary),
        );
        paint_canvas_focus_ring(ui, &plot_resp.response);
    }

    fn draw_trajectory_series<'a>(
        &'a self,
        plot_ui: &mut egui_plot::PlotUi<'a>,
        palette: &SemanticPalette,
        sched: &str,
    ) {
        let dashed = LineStyle::Dashed { length: 8.0 };
        let (col_a, col_b, col_mix, col_sched) = (
            palette.series_color(0),
            palette.series_color(1),
            palette.series_color(2),
            palette.series_color(3),
        );
        plot_ui.line(Line::new("Game A (Ensemble)", &self.traj_a[..]).color(col_a));
        plot_ui.line(Line::new("Game B (Ensemble)", &self.traj_b[..]).color(col_b));
        plot_ui.line(Line::new("50/50 Mix (Ensemble)", &self.traj_mix[..]).color(col_mix));
        plot_ui.line(
            Line::new(
                format!("Schedule {sched} (Ensemble)"),
                &self.traj_schedule[..],
            )
            .color(col_sched)
            .width(2.2_f32),
        );
        plot_ui.line(
            Line::new("Game A Markov Drift", &self.drift_a[..])
                .color(col_a)
                .style(dashed),
        );
        plot_ui.line(
            Line::new("Game B Markov Drift", &self.drift_b[..])
                .color(col_b)
                .style(dashed),
        );
        plot_ui.line(
            Line::new("50/50 Mix Markov Drift", &self.drift_mix[..])
                .color(col_mix)
                .style(dashed),
        );
        plot_ui.line(
            Line::new(
                format!("Schedule {sched} Markov Drift"),
                &self.drift_schedule[..],
            )
            .color(col_sched)
            .style(dashed),
        );
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

    fn render_phase_wedge_card(
        &self,
        ui: &mut egui::Ui,
        report: &ParrondoReport,
        palette: &SemanticPalette,
        height: f32,
    ) {
        ui.strong(format!(
            "(p_A, ε) Paradox Phase Wedge ({})",
            self.active_schedule.as_string()
        ));
        let active_pa = self.p_a_base - self.epsilon;
        let active_pt = [PlotPoint::new(active_pa, self.epsilon)];
        let plot_resp = Plot::new("parrondo_phase_wedge_plot")
            .height(height.max(120.0))
            .x_axis_label("Effective Game A Win Probability p_A = p_A,0 - ε")
            .y_axis_label("Bias Perturbation ε")
            .legend(Legend::default())
            .show(ui, |plot_ui| {
                self.draw_phase_wedge_series(plot_ui, palette, &active_pt);
            });
        let summary = format!(
            "Active point p_A = {:.4}, epsilon = {:.4}, paradox = {}, wedge samples = {}",
            active_pa,
            self.epsilon,
            report.is_paradox(),
            self.paradox_scatter.len()
        );
        register_canvas_accessibility(
            ui,
            &plot_resp.response,
            Role::GraphicsDocument,
            "Parrondo Paradox p_A versus epsilon Phase Wedge",
            Some(&summary),
        );
        paint_canvas_focus_ring(ui, &plot_resp.response);
    }

    fn draw_phase_wedge_series<'a>(
        &'a self,
        plot_ui: &mut egui_plot::PlotUi<'a>,
        palette: &SemanticPalette,
        active_pt: &'a [PlotPoint; 1],
    ) {
        plot_ui.points(
            Points::new("Non-Paradox Grid", &self.non_paradox_scatter[..])
                .color(palette.series_color(5))
                .radius(2.2_f32),
        );
        plot_ui.points(
            Points::new(
                "Paradox Wedge (A<0, B<0, Sched>0)",
                &self.paradox_scatter[..],
            )
            .color(palette.success)
            .radius(4.0_f32),
        );
        for (name, pts, col, width) in [
            (
                "E[ΔW_A] = 0",
                &self.boundary_game_a[..],
                palette.series_color(0),
                1.6_f32,
            ),
            (
                "E[ΔW_B] = 0",
                &self.boundary_game_b[..],
                palette.series_color(1),
                1.6_f32,
            ),
            (
                "E[ΔW_sched] = 0",
                &self.boundary_schedule[..],
                palette.series_color(3),
                2.0_f32,
            ),
        ] {
            if !pts.is_empty() {
                plot_ui.line(Line::new(name, pts).color(col).width(width));
            }
        }
        plot_ui.points(
            Points::new("Active (p_A, ε)", &active_pt[..])
                .color(palette.accent)
                .shape(MarkerShape::Diamond)
                .radius(6.5_f32),
        );
    }

    pub(super) fn build_export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        match self.view_mode {
            ParrondoViewMode::SplitOverview | ParrondoViewMode::PhaseWedge => {
                let mut out = vec![
                    ("Paradox Wedge".into(), to_pairs(&self.paradox_scatter)),
                    (
                        "Non-Paradox Grid".into(),
                        to_pairs(&self.non_paradox_scatter),
                    ),
                    (
                        "Active Operating Point".into(),
                        vec![[self.p_a_base - self.epsilon, self.epsilon]],
                    ),
                ];
                if !self.boundary_schedule.is_empty() {
                    out.push(("E[dW_sched] = 0".into(), to_pairs(&self.boundary_schedule)));
                }
                out
            }
            ParrondoViewMode::WealthTrajectories => vec![
                ("Game A Ensemble".into(), to_pairs(&self.traj_a)),
                ("Game B Ensemble".into(), to_pairs(&self.traj_b)),
                ("50/50 Mixture Ensemble".into(), to_pairs(&self.traj_mix)),
                ("Schedule Ensemble".into(), to_pairs(&self.traj_schedule)),
                (
                    "Schedule Exact Drift".into(),
                    to_pairs(&self.drift_schedule),
                ),
            ],
            ParrondoViewMode::TrapOccupancy => {
                let Some(ref r) = self.report else {
                    return Vec::new();
                };
                let m = self.modulus.min(r.pi_b.len());
                vec![
                    (
                        "pi_B".into(),
                        (0..m).map(|s| [s as f64, r.pi_b[s]]).collect(),
                    ),
                    (
                        "pi_mix".into(),
                        (0..m).map(|s| [s as f64, r.pi_mix[s]]).collect(),
                    ),
                    (
                        "pi_schedule".into(),
                        (0..m).map(|s| [s as f64, r.pi_schedule_mean[s]]).collect(),
                    ),
                ]
            }
        }
    }
}

// [cite:clinical_trials_statistics]
