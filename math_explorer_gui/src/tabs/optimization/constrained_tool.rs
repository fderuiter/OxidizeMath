#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Interactive 2D Nonlinear Constrained Contour workbench (`AugmentedLagrangianSolver`,
//! `LbfgsBSolver`, `ProjectedGradientSolver`, and derivative-free `NelderMead`).

use super::geometry::{blend_palette_color, sample_iso_contours};
use crate::app::SemanticPalette;
use crate::framework::{InteractiveTool, PaneSidePanel};
use eframe::egui;
use egui_plot::{Line, LineStyle, MarkerShape, Plot, Points};
use math_explorer::nalgebra::{DMatrix, DVector};
use math_explorer::pure_math::analysis::optimization::{
    AugmentedLagrangianConfig, AugmentedLagrangianSolver, AugmentedLagrangianWorkspace, BoxBounds,
    LbfgsBConfig, LbfgsBSolver, LbfgsBWorkspace, NelderMead, NelderMeadConfig, NelderMeadWorkspace,
    ProjectedGradientConfig, ProjectedGradientSolver, ProjectedGradientWorkspace,
};

const SOLVER_LABELS: [&str; 4] = [
    "Augmented Lagrangian (Disk + Box)",
    "L-BFGS-B (Box Bounds)",
    "Projected Gradient (Box Bounds)",
    "Nelder-Mead (Penalty Simplex)",
];

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ConstrainedState {
    start: [f64; 2],
    box_bounds: [f64; 4],
    circle_radius: f64,
    circle_equality: bool,
    rosenbrock: bool,
    solver_idx: usize,
}

impl Default for ConstrainedState {
    fn default() -> Self {
        Self {
            start: [-1.1, 1.15],
            box_bounds: [-1.4, 0.85, -0.7, 1.4],
            circle_radius: 1.0,
            circle_equality: false,
            rosenbrock: true,
            solver_idx: 0,
        }
    }
}

/// Interactive nonlinear constrained contour workbench comparing 4 preallocated solvers.
pub struct ConstrainedContourTool {
    state: ConstrainedState,
    al_eq_ws: AugmentedLagrangianWorkspace,
    al_ineq_ws: AugmentedLagrangianWorkspace,
    lbfgs_ws: LbfgsBWorkspace,
    pg_ws: ProjectedGradientWorkspace,
    nm_ws: NelderMeadWorkspace,
    trajectory: Vec<[f64; 2]>,
    opt_point: [f64; 2],
    opt_value: f64,
    violation: f64,
    multiplier: f64,
    iterations: usize,
}

impl Default for ConstrainedContourTool {
    fn default() -> Self {
        let mut tool = Self {
            state: ConstrainedState::default(),
            al_eq_ws: AugmentedLagrangianWorkspace::new(2, 1, 0, 120),
            al_ineq_ws: AugmentedLagrangianWorkspace::new(2, 0, 1, 120),
            lbfgs_ws: LbfgsBWorkspace::new(2, 8, 250),
            pg_ws: ProjectedGradientWorkspace::new(2, 300),
            nm_ws: NelderMeadWorkspace::new(2, 400),
            trajectory: Vec::new(),
            opt_point: [0.0, 0.0],
            opt_value: 0.0,
            violation: 0.0,
            multiplier: 0.0,
            iterations: 0,
        };
        tool.recompute();
        tool
    }
}

impl ConstrainedContourTool {
    /// Selects one of the 4 nonlinear solvers (`0..4`) and recomputes the iterate path.
    #[cfg(test)]
    pub fn set_solver_index(&mut self, idx: usize) {
        self.state.solver_idx = idx.min(3);
        self.recompute();
    }

    /// Returns the latest converged point `(x1*, x2*)`.
    #[cfg(test)]
    #[must_use]
    pub const fn optimal_point(&self) -> [f64; 2] {
        self.opt_point
    }

    fn eval_obj(rosenbrock: bool, x1: f64, x2: f64) -> (f64, [f64; 2]) {
        if rosenbrock {
            let d1 = x1 - 1.0;
            let d2 = x2 - x1 * x1;
            let val = d1 * d1 + 12.0 * d2 * d2;
            let g1 = 2.0 * d1 - 48.0 * x1 * d2;
            let g2 = 24.0 * d2;
            (val, [g1, g2])
        } else {
            let d1 = x1 - 1.1;
            let d2 = x2 - 0.9;
            let val = 1.5 * d1 * d1 + 0.5 * d1 * d2 + 2.2 * d2 * d2;
            (val, [3.0 * d1 + 0.5 * d2, 0.5 * d1 + 4.4 * d2])
        }
    }

    fn recompute(&mut self) {
        let b = self.state.box_bounds;
        let Ok(bounds) = BoxBounds::new(
            DVector::from_row_slice(&[b[0], b[2]]),
            DVector::from_row_slice(&[b[1], b[3]]),
        ) else {
            return;
        };
        match self.state.solver_idx {
            0 => self.run_augmented_lagrangian(&bounds),
            1 => self.run_lbfgsb(&bounds),
            2 => self.run_projected_gradient(&bounds),
            _ => self.run_nelder_mead(&bounds),
        }
    }

    fn run_augmented_lagrangian(&mut self, bounds: &BoxBounds) {
        let mut x = DVector::from_row_slice(&self.state.start);
        let r2 = self.state.circle_radius * self.state.circle_radius;
        let eq_mode = self.state.circle_equality;
        let rosen = self.state.rosenbrock;
        let solver = AugmentedLagrangianSolver::new(AugmentedLagrangianConfig::default());
        let ws = if eq_mode {
            &mut self.al_eq_ws
        } else {
            &mut self.al_ineq_ws
        };
        if let Ok(outer_iters) = solver.solve_in_place(
            |xv, grad, h_v, j_h, g_v, j_g: &mut DMatrix<f64>| {
                let (val, g) = Self::eval_obj(rosen, xv[0], xv[1]);
                grad[0] = g[0];
                grad[1] = g[1];
                let c_val = xv[0] * xv[0] + xv[1] * xv[1] - r2;
                if eq_mode {
                    h_v[0] = c_val;
                    j_h[(0, 0)] = 2.0 * xv[0];
                    j_h[(0, 1)] = 2.0 * xv[1];
                } else {
                    g_v[0] = c_val;
                    j_g[(0, 0)] = 2.0 * xv[0];
                    j_g[(0, 1)] = 2.0 * xv[1];
                }
                val
            },
            bounds,
            &mut x,
            ws,
        ) {
            self.opt_point = [x[0], x[1]];
            let (obj, viol) = ws.summary();
            self.opt_value = obj;
            self.violation = viol;
            self.iterations = outer_iters;
            self.multiplier = if eq_mode {
                ws.eq_multipliers()[0]
            } else {
                ws.ineq_multipliers()[0]
            };
            self.trajectory = ws.iterate_path_slice().to_vec();
        }
    }

    fn run_lbfgsb(&mut self, bounds: &BoxBounds) {
        let mut x = DVector::from_row_slice(&self.state.start);
        let rosen = self.state.rosenbrock;
        let solver = LbfgsBSolver::new(LbfgsBConfig::default());
        if let Ok((fx, _, iters)) = solver.solve_in_place(
            |xv, g| {
                let (val, gr) = Self::eval_obj(rosen, xv[0], xv[1]);
                g[0] = gr[0];
                g[1] = gr[1];
                val
            },
            bounds,
            &mut x,
            &mut self.lbfgs_ws,
        ) {
            self.opt_point = [x[0], x[1]];
            self.opt_value = fx;
            self.violation = 0.0;
            self.multiplier = 0.0;
            self.iterations = iters;
            self.trajectory = self.lbfgs_ws.iterate_path_slice().to_vec();
        }
    }

    fn run_projected_gradient(&mut self, bounds: &BoxBounds) {
        let mut x = DVector::from_row_slice(&self.state.start);
        let rosen = self.state.rosenbrock;
        let solver = ProjectedGradientSolver::new(ProjectedGradientConfig::default());
        if let Ok((fx, _, iters)) = solver.solve_in_place(
            |xv, g| {
                let (val, gr) = Self::eval_obj(rosen, xv[0], xv[1]);
                g[0] = gr[0];
                g[1] = gr[1];
                val
            },
            bounds,
            &mut x,
            &mut self.pg_ws,
        ) {
            self.opt_point = [x[0], x[1]];
            self.opt_value = fx;
            self.violation = 0.0;
            self.multiplier = 0.0;
            self.iterations = iters;
            self.trajectory = self.pg_ws.iterate_path_slice().to_vec();
        }
    }

    fn run_nelder_mead(&mut self, bounds: &BoxBounds) {
        let mut x = DVector::from_row_slice(&self.state.start);
        let rosen = self.state.rosenbrock;
        let r2 = self.state.circle_radius * self.state.circle_radius;
        let eq_mode = self.state.circle_equality;
        let solver = NelderMead::new(NelderMeadConfig::default());
        if let Ok((_, iters)) = solver.minimize_in_place(
            |xv| {
                let (val, _) = Self::eval_obj(rosen, xv[0], xv[1]);
                let mut pen = 0.0;
                for i in 0..2 {
                    pen += (bounds.lower[i] - xv[i]).max(0.0).powi(2);
                    pen += (xv[i] - bounds.upper[i]).max(0.0).powi(2);
                }
                let c = xv[0] * xv[0] + xv[1] * xv[1] - r2;
                let c_pen = if eq_mode { c * c } else { c.max(0.0).powi(2) };
                val + 120.0 * (pen + c_pen)
            },
            &mut x,
            &mut self.nm_ws,
        ) {
            self.opt_point = [x[0], x[1]];
            self.opt_value = Self::eval_obj(rosen, x[0], x[1]).0;
            let c = x[0] * x[0] + x[1] * x[1] - r2;
            self.violation = if eq_mode { c.abs() } else { c.max(0.0) };
            self.multiplier = 0.0;
            self.iterations = iters;
            self.trajectory = self.nm_ws.iterate_path_slice().to_vec();
        }
    }

    fn draw_sidebar(&mut self, parent_ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        let panel_id = crate::framework::scoped_panel_id(parent_ui, "opt_constrained_controls");
        PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            super::render_tool_header(ui, self.name());
            egui::ScrollArea::vertical()
                .id_salt("constrained_sidebar_scroll")
                .show(ui, |ui| {
                    changed |= self.draw_solver_and_surface_controls(ui);
                    ui.separator();
                    changed |= self.draw_barrier_controls(ui);
                    ui.separator();
                    let p = SemanticPalette::from_visuals(ui.visuals());
                    ui.colored_label(
                        p.success,
                        format!(
                            "Converged x* = ({:.3}, {:.3})",
                            self.opt_point[0], self.opt_point[1]
                        ),
                    );
                    ui.label(format!("Objective f(x*) = {:.4}", self.opt_value));
                    ui.label(format!(
                        "Iterations: {} | Viol: {:.2e}",
                        self.iterations, self.violation
                    ));
                    if self.state.solver_idx == 0 {
                        ui.label(format!("Lagrange Multiplier = {:.4}", self.multiplier));
                    }
                });
        });
        changed
    }

    fn draw_solver_and_surface_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        let prev = self.state.solver_idx;
        egui::ComboBox::from_label("Solver")
            .selected_text(SOLVER_LABELS[self.state.solver_idx.min(3)])
            .show_ui(ui, |ui| {
                for (i, &lbl) in SOLVER_LABELS.iter().enumerate() {
                    ui.selectable_value(&mut self.state.solver_idx, i, lbl);
                }
            });
        changed |= self.state.solver_idx != prev;
        ui.horizontal(|ui| {
            changed |= ui
                .selectable_value(&mut self.state.rosenbrock, true, "Rosenbrock Valley")
                .changed();
            changed |= ui
                .selectable_value(&mut self.state.rosenbrock, false, "Quadratic Bowl")
                .changed();
        });
        changed |= ui
            .add(egui::Slider::new(&mut self.state.start[0], -1.8..=1.8).text("Start x₁"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.state.start[1], -1.5..=1.8).text("Start x₂"))
            .changed();
        changed
    }

    fn draw_barrier_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        let b = &mut self.state.box_bounds;
        let hi_x = (b[1] - 0.2).max(-1.9);
        changed |= ui
            .add(egui::Slider::new(&mut b[0], -2.0..=hi_x).text("Box x₁_min"))
            .changed();
        let lo_x = (b[0] + 0.2).min(1.9);
        changed |= ui
            .add(egui::Slider::new(&mut b[1], lo_x..=2.0).text("Box x₁_max"))
            .changed();
        let hi_y = (b[3] - 0.2).max(-1.9);
        changed |= ui
            .add(egui::Slider::new(&mut b[2], -2.0..=hi_y).text("Box x₂_min"))
            .changed();
        let lo_y = (b[2] + 0.2).min(1.9);
        changed |= ui
            .add(egui::Slider::new(&mut b[3], lo_y..=2.0).text("Box x₂_max"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.state.circle_radius, 0.3..=1.8).text("Disk Radius r"))
            .changed();
        changed |= ui
            .checkbox(
                &mut self.state.circle_equality,
                "Ring Equality x₁² + x₂² = r²",
            )
            .changed();
        changed
    }

    fn draw_plot(&self, ui: &mut egui::Ui) {
        let palette = SemanticPalette::from_visuals(ui.visuals());
        let rosen = self.state.rosenbrock;
        let contours = sample_iso_contours(
            |x, y| Self::eval_obj(rosen, x, y).0.ln_1p(),
            [-2.0, 2.0, -1.6, 2.0],
            40,
            10,
        );
        Plot::new("constrained_contour_plot")
            .x_axis_label("x₁")
            .y_axis_label("x₂")
            .data_aspect(1.0)
            .show(ui, |plot_ui| {
                for (t, segs) in contours {
                    let c = blend_palette_color(palette.axis_z, palette.axis_x, t);
                    for s in segs {
                        plot_ui.line(Line::new("Contour", s.to_vec()).color(c).width(1.1_f32));
                    }
                }
                self.draw_constraint_and_path_overlays(plot_ui, &palette);
            });
    }

    fn draw_constraint_and_path_overlays(
        &self,
        plot_ui: &mut egui_plot::PlotUi<'_>,
        palette: &SemanticPalette,
    ) {
        let b = self.state.box_bounds;
        let box_loop = vec![
            [b[0], b[2]],
            [b[1], b[2]],
            [b[1], b[3]],
            [b[0], b[3]],
            [b[0], b[2]],
        ];
        plot_ui.line(
            Line::new("Box Bounds", box_loop)
                .color(palette.error)
                .style(LineStyle::dashed_dense())
                .width(2.0_f32),
        );
        if self.state.solver_idx == 0 || self.state.solver_idx == 3 {
            let r = self.state.circle_radius;
            let ring: Vec<[f64; 2]> = (0..=64)
                .map(|k| {
                    let a = std::f64::consts::TAU * (k as f64) / 64.0;
                    [r * a.cos(), r * a.sin()]
                })
                .collect();
            plot_ui.line(
                Line::new("Disk / Ring Constraint", ring)
                    .color(palette.warning)
                    .width(2.0_f32),
            );
        }
        plot_ui.line(
            Line::new("Iterate Trajectory", self.trajectory.clone())
                .color(palette.series_color(0))
                .width(2.4_f32),
        );
        plot_ui.points(
            Points::new("Start x₀", vec![self.state.start])
                .color(palette.warning)
                .radius(5.5_f32),
        );
        plot_ui.points(
            Points::new("Solution x*", vec![self.opt_point])
                .color(palette.success)
                .radius(7.5_f32)
                .shape(MarkerShape::Diamond),
        );
    }
}

impl InteractiveTool for ConstrainedContourTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }

    fn name(&self) -> &'static str {
        "Nonlinear Constrained Contours"
    }

    fn domain(&self) -> &'static str {
        "optimization"
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&self.state).ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let s: ConstrainedState =
            serde_json::from_str(state).map_err(|e| format!("Invalid constrained state: {e}"))?;
        let valid = s.start.iter().all(|v| v.is_finite() && v.abs() <= 20.0)
            && s.box_bounds
                .iter()
                .all(|v| v.is_finite() && v.abs() <= 20.0)
            && s.box_bounds[0] + 0.05 <= s.box_bounds[1]
            && s.box_bounds[2] + 0.05 <= s.box_bounds[3]
            && s.circle_radius.is_finite()
            && (0.1..=10.0).contains(&s.circle_radius)
            && s.solver_idx < 4;
        if !valid {
            return Err("Constrained contour bounds or solver index invalid".into());
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
