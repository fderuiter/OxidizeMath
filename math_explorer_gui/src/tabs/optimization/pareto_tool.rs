#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Interactive Multi-Objective Pareto Frontier & L1 FISTA Sparsity Workbench.

use crate::app::SemanticPalette;
use crate::framework::{InteractiveTool, PaneSidePanel};
use eframe::egui;
use egui_plot::{Line, MarkerShape, Plot, Points};
use math_explorer::nalgebra::{DMatrix, DVector};
use math_explorer::pure_math::analysis::optimization::{
    epsilon_constraint_sweep, extract_pareto_frontier, weighted_sum_sweep, BoxBounds,
    L1RegularizedLeastSquares, ParetoPoint, ProximalConfig, ProximalWorkspace,
};

const PROBLEM_NAMES: [&str; 3] = [
    "Non-Convex Dent (ε-Constraint vs Weighted-Sum)",
    "Convex Bi-Objective Quadratic",
    "L1 Sparse LASSO Trade-off (FISTA)",
];

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ParetoState {
    concavity_amp: f64,
    problem_mode: usize,
    num_samples: usize,
    selected_idx: usize,
}

impl Default for ParetoState {
    fn default() -> Self {
        Self {
            concavity_amp: 0.36,
            problem_mode: 0,
            num_samples: 19,
            selected_idx: 8,
        }
    }
}

/// Interactive Pareto multi-objective workbench (`weighted_sum_sweep`, `epsilon_constraint_sweep`, `FISTA`).
pub struct ParetoFrontierTool {
    state: ParetoState,
    fista_ws: ProximalWorkspace,
    eps_frontier: Vec<ParetoPoint>,
    ws_frontier: Vec<ParetoPoint>,
    dominated_cloud: Vec<[f64; 2]>,
}

impl Default for ParetoFrontierTool {
    fn default() -> Self {
        let mut tool = Self {
            state: ParetoState::default(),
            fista_ws: ProximalWorkspace::new(5, 4),
            eps_frontier: Vec::new(),
            ws_frontier: Vec::new(),
            dominated_cloud: Vec::new(),
        };
        tool.recompute();
        tool
    }
}

impl ParetoFrontierTool {
    /// Returns the number of extracted non-dominated Pareto points.
    #[cfg(test)]
    #[must_use]
    pub fn frontier_len(&self) -> usize {
        self.eps_frontier.len()
    }

    /// Switches problem mode (`0 = Non-Convex`, `1 = Convex`, `2 = L1 FISTA`) and recomputes.
    #[cfg(test)]
    pub fn set_problem_mode(&mut self, mode: usize) {
        self.state.problem_mode = mode.min(2);
        self.recompute();
    }

    fn eval_biobjective_fg(
        amp: f64,
        nonconvex: bool,
        x: &DVector<f64>,
        g1: &mut DVector<f64>,
        g2: &mut DVector<f64>,
    ) -> (f64, f64) {
        let (t, u) = (x[0], x[1]);
        let f1 = t + 0.4 * u * u;
        g1[0] = 1.0;
        g1[1] = 0.8 * u;

        let pi = std::f64::consts::PI;
        let (dent, dent_dt) = if nonconvex {
            let s = (pi * t).sin();
            let c = (pi * t).cos();
            (amp * s * s, 2.0 * amp * pi * s * c)
        } else {
            (0.0, 0.0)
        };
        let f2 = (1.0 - t).powi(2) + dent + 0.4 * (u - 0.15).powi(2);
        g2[0] = -2.0 * (1.0 - t) + dent_dt;
        g2[1] = 0.8 * (u - 0.15);
        (f1, f2)
    }

    fn recompute(&mut self) {
        self.dominated_cloud.clear();
        if self.state.problem_mode == 2 {
            self.recompute_lasso_pareto();
        } else {
            self.recompute_biobjective_sweeps();
        }
        if !self.eps_frontier.is_empty() && self.state.selected_idx >= self.eps_frontier.len() {
            self.state.selected_idx = self.eps_frontier.len() / 2;
        }
    }

    fn recompute_biobjective_sweeps(&mut self) {
        let amp = self.state.concavity_amp;
        let nonconvex = self.state.problem_mode == 0;
        let mut g1 = DVector::zeros(2);
        let mut g2 = DVector::zeros(2);
        for it in 0..14 {
            let t = it as f64 / 13.0;
            for iu in 1..6 {
                let u = 0.15 + 0.16 * iu as f64;
                let xv = DVector::from_row_slice(&[t, u]);
                let (f1, f2) = Self::eval_biobjective_fg(amp, nonconvex, &xv, &mut g1, &mut g2);
                self.dominated_cloud.push([f1, f2]);
            }
        }
        let Ok(bounds) = BoxBounds::uniform(2, 0.0, 1.0) else {
            return;
        };
        let x0 = DVector::from_row_slice(&[0.5, 0.1]);
        let n = self.state.num_samples.clamp(5, 35);

        if let Ok((_, ws_front)) = weighted_sum_sweep(
            |xv, ga, gb| Self::eval_biobjective_fg(amp, nonconvex, xv, ga, gb),
            &bounds,
            &x0,
            n,
        ) {
            self.ws_frontier = ws_front;
        }
        if let Ok((_, eps_front)) = epsilon_constraint_sweep(
            |xv, ga, gb| Self::eval_biobjective_fg(amp, nonconvex, xv, ga, gb),
            &bounds,
            &x0,
            (0.01, 1.05),
            n,
        ) {
            self.eps_frontier = eps_front;
        }
    }

    fn recompute_lasso_pareto(&mut self) {
        let a = DMatrix::from_row_slice(
            5,
            4,
            &[
                1.0, 0.2, 0.0, 0.1, 0.1, 1.1, 0.2, 0.0, 0.0, 0.3, 1.0, 0.2, 0.2, 0.0, 0.1, 1.2,
                0.5, -0.4, 0.6, -0.3,
            ],
        );
        let b = DVector::from_row_slice(&[1.35, -0.82, 0.64, 0.12, 0.95]);
        let n = self.state.num_samples.clamp(5, 35);
        let mut candidates = Vec::with_capacity(n);
        let cfg = ProximalConfig {
            max_iterations: 300,
            tolerance: 1e-8,
            step_size: None,
        };
        for k in 0..n {
            let frac = k as f64 / (n - 1) as f64;
            let lam = 1e-3 + 1.2 * frac * frac;
            let prob = L1RegularizedLeastSquares::new(lam);
            let mut x = DVector::zeros(4);
            let _ = prob.solve_fista_in_place(&a, &b, &mut x, &cfg, &mut self.fista_ws);
            let r = &a * &x - &b;
            let mse = 0.5 * r.dot(&r);
            let l1: f64 = x.iter().map(|v| v.abs()).sum();
            candidates.push(ParetoPoint {
                decision: x,
                objectives: vec![l1, mse],
                parameter: lam,
            });
        }
        self.eps_frontier = extract_pareto_frontier(&candidates, 1e-6);
        self.ws_frontier.clone_from(&self.eps_frontier);
    }

    fn draw_sidebar(&mut self, parent_ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        let panel_id = crate::framework::scoped_panel_id(parent_ui, "opt_pareto_controls");
        PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            super::render_tool_header(ui, self.name());
            let prev = self.state.problem_mode;
            egui::ComboBox::from_label("Problem")
                .selected_text(PROBLEM_NAMES[self.state.problem_mode.min(2)])
                .show_ui(ui, |ui| {
                    for (i, &lbl) in PROBLEM_NAMES.iter().enumerate() {
                        ui.selectable_value(&mut self.state.problem_mode, i, lbl);
                    }
                });
            changed |= self.state.problem_mode != prev;
            if self.state.problem_mode == 0 {
                changed |= ui
                    .add(
                        egui::Slider::new(&mut self.state.concavity_amp, 0.0..=0.75)
                            .text("Non-Convex Dent"),
                    )
                    .changed();
            }
            changed |= ui
                .add(egui::Slider::new(&mut self.state.num_samples, 6..=31).text("Sweep Samples"))
                .changed();
            ui.separator();
            self.draw_selection_readout(ui);
        });
        changed
    }

    fn draw_selection_readout(&mut self, ui: &mut egui::Ui) {
        let p = SemanticPalette::from_visuals(ui.visuals());
        ui.label(format!(
            "ε-Constraint / Exact Frontier: {} pts",
            self.eps_frontier.len()
        ));
        if self.state.problem_mode != 2 {
            ui.colored_label(
                p.warning,
                format!("Weighted-Sum Frontier: {} pts", self.ws_frontier.len()),
            );
        }
        if !self.eps_frontier.is_empty() {
            let max_i = self.eps_frontier.len() - 1;
            self.state.selected_idx = self.state.selected_idx.min(max_i);
            ui.add(
                egui::Slider::new(&mut self.state.selected_idx, 0..=max_i)
                    .text("Operating Point #"),
            );
            let sel = &self.eps_frontier[self.state.selected_idx];
            ui.colored_label(
                p.success,
                format!(
                    "f₁* = {:.4}, f₂* = {:.4}",
                    sel.objectives[0], sel.objectives[1]
                ),
            );
            let x_vals: Vec<String> = sel.decision.iter().map(|v| format!("{v:.3}")).collect();
            ui.small(format!("Decision x* = [{}]", x_vals.join(", ")));
        }
    }

    fn draw_plot(&self, ui: &mut egui::Ui) {
        let palette = SemanticPalette::from_visuals(ui.visuals());
        let (x_lbl, y_lbl) = if self.state.problem_mode == 2 {
            ("‖x‖₁ Sparsity Penalty", "½‖Ax − b‖₂² Data Misfit")
        } else {
            ("Objective f₁(x)", "Objective f₂(x)")
        };
        let eps_pts: Vec<[f64; 2]> = self
            .eps_frontier
            .iter()
            .map(|p| [p.objectives[0], p.objectives[1]])
            .collect();
        let ws_pts: Vec<[f64; 2]> = self
            .ws_frontier
            .iter()
            .map(|p| [p.objectives[0], p.objectives[1]])
            .collect();

        Plot::new("pareto_workbench_plot")
            .x_axis_label(x_lbl)
            .y_axis_label(y_lbl)
            .show(ui, |plot_ui| {
                if !self.dominated_cloud.is_empty() {
                    plot_ui.points(
                        Points::new("Dominated Decisions", self.dominated_cloud.clone())
                            .color(palette.muted_text)
                            .radius(2.8_f32),
                    );
                }
                plot_ui.line(
                    Line::new("Pareto Frontier (ε-Constraint / FISTA)", eps_pts.clone())
                        .color(palette.series_color(0))
                        .width(2.4_f32),
                );
                plot_ui.points(
                    Points::new("Non-Dominated Points", eps_pts)
                        .color(palette.series_color(0))
                        .radius(4.5_f32),
                );
                if self.state.problem_mode != 2 {
                    plot_ui.points(
                        Points::new("Weighted-Sum Sweep", ws_pts)
                            .color(palette.warning)
                            .radius(5.5_f32)
                            .shape(MarkerShape::Square),
                    );
                }
                if let Some(sel) = self.eps_frontier.get(self.state.selected_idx) {
                    plot_ui.points(
                        Points::new(
                            "Selected Operating Point",
                            vec![[sel.objectives[0], sel.objectives[1]]],
                        )
                        .color(palette.success)
                        .radius(8.0_f32)
                        .shape(MarkerShape::Diamond),
                    );
                }
            });
    }
}

impl InteractiveTool for ParetoFrontierTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }

    fn name(&self) -> &'static str {
        "Pareto Multi-Objective Workbench"
    }

    fn domain(&self) -> &'static str {
        "optimization"
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&self.state).ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let s: ParetoState =
            serde_json::from_str(state).map_err(|e| format!("Invalid Pareto state: {e}"))?;
        let valid = s.concavity_amp.is_finite()
            && (0.0..=2.0).contains(&s.concavity_amp)
            && s.problem_mode <= 2
            && (5..=60).contains(&s.num_samples)
            && s.selected_idx <= 60;
        if !valid {
            return Err("Pareto workbench parameters out of valid bounds".into());
        }
        self.state = s;
        self.recompute();
        Ok(())
    }

    fn show_inside(&mut self, parent_ui: &mut egui::Ui) {
        if self.draw_sidebar(parent_ui) {
            self.recompute();
        }
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            self.draw_plot(ui);
        });
    }
}

// [cite:lorahub]
