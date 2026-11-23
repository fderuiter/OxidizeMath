#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Interactive 2D Active-Set Convex Quadratic Programming workbench.

use super::geometry::{
    blend_palette_color, clip_halfspace_polygon, constraint_boundary_segment, sample_iso_contours,
};
use crate::app::SemanticPalette;
use crate::framework::{InteractiveTool, PaneSidePanel};
use eframe::egui;
use egui_plot::{Line, LineStyle, MarkerShape, Plot, Points, Polygon};
use math_explorer::nalgebra::{DMatrix, DVector};
use math_explorer::pure_math::analysis::optimization::{
    ActiveSetQpSolver, ConstraintSense, QpSolution, QpStatus, QuadraticProgram,
};

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct QpState {
    target: [f64; 2],
    q_diag: [f64; 2],
    q_offdiag: f64,
    ineq: [[f64; 3]; 2],
    eq_enabled: bool,
    eq_line: [f64; 3],
}

impl Default for QpState {
    fn default() -> Self {
        Self {
            target: [3.2, 2.8],
            q_diag: [2.0, 1.4],
            q_offdiag: 0.35,
            ineq: [[1.0, 1.0, 4.0], [1.0, -0.5, 2.2]],
            eq_enabled: false,
            eq_line: [1.0, 2.0, 5.0],
        }
    }
}

/// Interactive Active-Set Convex QP workbench visualizing quadratic contours and KKT multipliers.
pub struct QuadraticProgrammingTool {
    state: QpState,
    solution: QpSolution,
}

impl Default for QuadraticProgrammingTool {
    fn default() -> Self {
        let mut tool = Self {
            state: QpState::default(),
            solution: QpSolution {
                status: QpStatus::Optimal,
                x: DVector::zeros(2),
                objective: 0.0,
                eq_multipliers: DVector::zeros(0),
                ineq_multipliers: DVector::zeros(4),
                active_inequalities: Vec::new(),
                iterations: 0,
                iterate_path: Vec::new(),
            },
        };
        tool.recompute();
        tool
    }
}

impl QuadraticProgrammingTool {
    /// Returns whether the latest active-set QP solve certified KKT optimality.
    #[cfg(test)]
    #[must_use]
    pub fn is_optimal(&self) -> bool {
        self.solution.status == QpStatus::Optimal
    }

    /// Returns the primal optimum `(x1*, x2*)`.
    #[cfg(test)]
    #[must_use]
    pub fn optimal_point(&self) -> [f64; 2] {
        [self.solution.x[0], self.solution.x[1]]
    }

    fn safe_offdiag(&self) -> f64 {
        let cap = (self.state.q_diag[0] * self.state.q_diag[1]).sqrt() * 0.85;
        self.state.q_offdiag.clamp(-cap, cap)
    }

    fn eval_quadratic(&self, x1: f64, x2: f64) -> f64 {
        let q11 = self.state.q_diag[0];
        let q22 = self.state.q_diag[1];
        let q12 = self.safe_offdiag();
        let dx1 = x1 - self.state.target[0];
        let dx2 = x2 - self.state.target[1];
        0.5 * (q11 * dx1 * dx1 + 2.0 * q12 * dx1 * dx2 + q22 * dx2 * dx2)
    }

    fn recompute(&mut self) {
        let q11 = self.state.q_diag[0];
        let q22 = self.state.q_diag[1];
        let q12 = self.safe_offdiag();
        let q = DMatrix::from_row_slice(2, 2, &[q11, q12, q12, q22]);
        let c = DVector::from_row_slice(&[
            -(q11 * self.state.target[0] + q12 * self.state.target[1]),
            -(q12 * self.state.target[0] + q22 * self.state.target[1]),
        ]);
        let (a_eq, b_eq) = if self.state.eq_enabled {
            (
                DMatrix::from_row_slice(1, 2, &[self.state.eq_line[0], self.state.eq_line[1]]),
                DVector::from_row_slice(&[self.state.eq_line[2]]),
            )
        } else {
            (DMatrix::zeros(0, 2), DVector::zeros(0))
        };
        let a_ineq = DMatrix::from_row_slice(
            4,
            2,
            &[
                self.state.ineq[0][0],
                self.state.ineq[0][1],
                self.state.ineq[1][0],
                self.state.ineq[1][1],
                -1.0,
                0.0,
                0.0,
                -1.0,
            ],
        );
        let b_ineq =
            DVector::from_row_slice(&[self.state.ineq[0][2], self.state.ineq[1][2], 0.0, 0.0]);
        if let Ok(qp) = QuadraticProgram::new(q, c, a_eq, b_eq, a_ineq, b_ineq) {
            if let Ok(sol) = ActiveSetQpSolver::default().solve(&qp, None) {
                self.solution = sol;
            }
        }
    }

    fn draw_sidebar(&mut self, parent_ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        let panel_id = crate::framework::scoped_panel_id(parent_ui, "opt_qp_controls");
        PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            super::render_tool_header(ui, self.name());
            egui::ScrollArea::vertical()
                .id_salt("qp_sidebar_scroll")
                .show(ui, |ui| {
                    changed |= self.draw_quadratic_controls(ui);
                    ui.separator();
                    changed |= self.draw_constraint_controls(ui);
                    ui.separator();
                    self.draw_kkt_readout(ui);
                });
        });
        changed
    }

    fn draw_quadratic_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.strong("Quadratic Objective ½(x−t)ᵀQ(x−t)");
        changed |= ui
            .add(egui::Slider::new(&mut self.state.target[0], -1.0..=6.0).text("Target t₁"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.state.target[1], -1.0..=6.0).text("Target t₂"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.state.q_diag[0], 0.2..=6.0).text("Curvature Q₁₁"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.state.q_diag[1], 0.2..=6.0).text("Curvature Q₂₂"))
            .changed();
        let cap = (self.state.q_diag[0] * self.state.q_diag[1]).sqrt() * 0.85;
        self.state.q_offdiag = self.state.q_offdiag.clamp(-cap, cap);
        changed |= ui
            .add(egui::Slider::new(&mut self.state.q_offdiag, -cap..=cap).text("Coupling Q₁₂"))
            .changed();
        changed
    }

    fn draw_constraint_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        for (i, r) in self.state.ineq.iter_mut().enumerate() {
            ui.group(|ui| {
                ui.strong(format!("Inequality g_{}(x) ≤ b_{}", i + 1, i + 1));
                changed |= ui
                    .add(egui::Slider::new(&mut r[0], -4.0..=4.0).text("a₁"))
                    .changed();
                changed |= ui
                    .add(egui::Slider::new(&mut r[1], -4.0..=4.0).text("a₂"))
                    .changed();
                changed |= ui
                    .add(egui::Slider::new(&mut r[2], -2.0..=10.0).text("b"))
                    .changed();
            });
        }
        changed |= ui
            .checkbox(
                &mut self.state.eq_enabled,
                "Enable Equality e₁x₁ + e₂x₂ = d",
            )
            .changed();
        if self.state.eq_enabled {
            let eq = &mut self.state.eq_line;
            changed |= ui
                .add(egui::Slider::new(&mut eq[0], -4.0..=4.0).text("e₁"))
                .changed();
            changed |= ui
                .add(egui::Slider::new(&mut eq[1], -4.0..=4.0).text("e₂"))
                .changed();
            changed |= ui
                .add(egui::Slider::new(&mut eq[2], 0.0..=10.0).text("d"))
                .changed();
        }
        changed
    }

    fn draw_kkt_readout(&self, ui: &mut egui::Ui) {
        let p = SemanticPalette::from_visuals(ui.visuals());
        let (lbl, col) = match self.solution.status {
            QpStatus::Optimal => ("Status: KKT OPTIMAL", p.success),
            QpStatus::Infeasible => ("Status: INFEASIBLE", p.error),
        };
        ui.colored_label(col, lbl);
        ui.label(format!("Active-Set Steps: {}", self.solution.iterations));
        if self.solution.status == QpStatus::Optimal {
            let x = &self.solution.x;
            let mu = &self.solution.ineq_multipliers;
            ui.label(format!("Primal x* = ({:.3}, {:.3})", x[0], x[1]));
            ui.label(format!(
                "Inequality μ* = [{:.3}, {:.3}, {:.3}, {:.3}]",
                mu[0], mu[1], mu[2], mu[3]
            ));
            if self.solution.eq_multipliers.len() == 1 {
                ui.label(format!(
                    "Equality λ* = {:.3}",
                    self.solution.eq_multipliers[0]
                ));
            }
        }
    }

    fn draw_plot(&self, ui: &mut egui::Ui) {
        let palette = SemanticPalette::from_visuals(ui.visuals());
        let bounds = [-0.5, 6.5, -0.5, 6.5];
        let contours = sample_iso_contours(|x, y| self.eval_quadratic(x, y), bounds, 36, 9);
        let mut poly = vec![[0.0, 0.0], [6.5, 0.0], [6.5, 6.5], [0.0, 6.5]];
        for r in &self.state.ineq {
            poly = clip_halfspace_polygon(&poly, r[0], r[1], ConstraintSense::LessEqual, r[2]);
        }

        Plot::new("qp_active_set_plot")
            .x_axis_label("x₁")
            .y_axis_label("x₂")
            .data_aspect(1.0)
            .show(ui, |plot_ui| {
                for (t, segs) in contours {
                    let c = blend_palette_color(palette.axis_z, palette.axis_x, t);
                    for s in segs {
                        plot_ui.line(
                            Line::new("QP Iso-Contour", s.to_vec())
                                .color(c)
                                .width(1.1_f32),
                        );
                    }
                }
                if poly.len() >= 3 {
                    plot_ui.polygon(
                        Polygon::new("Feasible Set", poly)
                            .fill_color(palette.series_color(0).gamma_multiply(0.16))
                            .stroke(egui::Stroke::new(1.6_f32, palette.series_color(0))),
                    );
                }
                self.draw_qp_overlays(plot_ui, &palette);
            });
    }

    fn draw_qp_overlays(&self, plot_ui: &mut egui_plot::PlotUi<'_>, palette: &SemanticPalette) {
        if self.state.eq_enabled {
            let eq = self.state.eq_line;
            if let Some(seg) =
                constraint_boundary_segment(eq[0], eq[1], eq[2], [-0.5, 6.5], [-0.5, 6.5])
            {
                plot_ui.line(
                    Line::new("Equality eᵀx = d", seg.to_vec())
                        .color(palette.error)
                        .style(LineStyle::dashed_dense())
                        .width(2.0_f32),
                );
            }
        }
        plot_ui.points(
            Points::new("Unconstrained Minimum", vec![self.state.target])
                .color(palette.warning)
                .radius(6.0_f32)
                .shape(MarkerShape::Cross),
        );
        if !self.solution.iterate_path.is_empty() {
            plot_ui.line(
                Line::new("Active-Set Path", self.solution.iterate_path.clone())
                    .color(palette.success)
                    .width(2.5_f32),
            );
        }
        if self.solution.status == QpStatus::Optimal {
            let opt = [self.solution.x[0], self.solution.x[1]];
            plot_ui.points(
                Points::new("Constrained Optimum x*", vec![opt])
                    .color(palette.success)
                    .radius(7.0_f32)
                    .shape(MarkerShape::Diamond),
            );
            for i in 0..2 {
                let mu = self.solution.ineq_multipliers[i];
                if mu > 1e-4 {
                    let g = self.state.ineq[i];
                    let tip = [opt[0] + 0.35 * mu * g[0], opt[1] + 0.35 * mu * g[1]];
                    plot_ui.line(
                        Line::new(format!("KKT Force μ_{}∇g_{}", i + 1, i + 1), vec![opt, tip])
                            .color(palette.warning)
                            .width(2.4_f32),
                    );
                }
            }
        }
    }
}

impl InteractiveTool for QuadraticProgrammingTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }

    fn name(&self) -> &'static str {
        "Quadratic Programming (Active-Set)"
    }

    fn domain(&self) -> &'static str {
        "optimization"
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&self.state).ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let s: QpState =
            serde_json::from_str(state).map_err(|e| format!("Invalid QP state: {e}"))?;
        let valid = s.target.iter().all(|v| v.is_finite() && v.abs() <= 50.0)
            && s.q_diag
                .iter()
                .all(|&v| v.is_finite() && (0.05..=50.0).contains(&v))
            && s.q_offdiag.is_finite()
            && s.q_offdiag.abs() < (s.q_diag[0] * s.q_diag[1]).sqrt()
            && s.ineq
                .iter()
                .flatten()
                .all(|v| v.is_finite() && v.abs() <= 100.0)
            && s.eq_line.iter().all(|v| v.is_finite() && v.abs() <= 100.0);
        if !valid {
            return Err("QP Hessian not positive definite or bounds out of range".into());
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
// [cite:advanced_linear_algebra]
