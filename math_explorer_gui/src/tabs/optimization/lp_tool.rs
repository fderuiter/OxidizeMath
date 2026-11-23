#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Interactive 2D Linear Programming (Two-Phase Simplex with Bland's Rule) workbench.

use super::geometry::{clip_halfspace_polygon, constraint_boundary_segment};
use crate::app::SemanticPalette;
use crate::framework::{InteractiveTool, PaneSidePanel};
use eframe::egui;
use egui_plot::{Line, LineStyle, MarkerShape, Plot, Points, Polygon};
use math_explorer::nalgebra::{DMatrix, DVector};
use math_explorer::pure_math::analysis::optimization::{
    ConstraintSense, LinearProgram, LpSense, LpSolution, LpStatus, SimplexSolver,
};

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct LpState {
    c: [f64; 2],
    maximize: bool,
    constraints: [[f64; 3]; 3],
    senses: [u8; 3],
}

impl Default for LpState {
    fn default() -> Self {
        Self {
            c: [3.0, 2.0],
            maximize: true,
            constraints: [[2.0, 1.0, 18.0], [2.0, 3.0, 42.0], [3.0, 1.0, 24.0]],
            senses: [0, 0, 0],
        }
    }
}

/// Interactive Two-Phase Simplex LP workbench visualizing polytopes, pivot paths, and dual prices.
pub struct LinearProgrammingTool {
    state: LpState,
    solution: LpSolution,
}

impl Default for LinearProgrammingTool {
    fn default() -> Self {
        let mut tool = Self {
            state: LpState::default(),
            solution: LpSolution {
                status: LpStatus::Optimal,
                x: DVector::zeros(2),
                objective: 0.0,
                shadow_prices: DVector::zeros(3),
                iterations: 0,
                pivot_path: Vec::new(),
            },
        };
        tool.recompute();
        tool
    }
}

fn code_to_sense(code: u8) -> ConstraintSense {
    match code {
        1 => ConstraintSense::GreaterEqual,
        2 => ConstraintSense::Equal,
        _ => ConstraintSense::LessEqual,
    }
}

impl LinearProgrammingTool {
    /// Returns the status of the latest Simplex solve (`Optimal`, `Infeasible`, or `Unbounded`).
    #[cfg(test)]
    #[must_use]
    pub const fn last_status(&self) -> LpStatus {
        self.solution.status
    }

    /// Returns the optimal objective value of the latest Simplex solve.
    #[cfg(test)]
    #[must_use]
    pub const fn optimal_objective(&self) -> f64 {
        self.solution.objective
    }

    /// Configures an infeasible LP conflict (`x1 + x2 <= 4` and `x1 + x2 >= 8`).
    pub fn apply_infeasible_preset(&mut self) {
        self.state.c = [2.0, 3.0];
        self.state.maximize = true;
        self.state.constraints = [[1.0, 1.0, 4.0], [1.0, 1.0, 8.0], [1.0, 0.0, 10.0]];
        self.state.senses = [0, 1, 0];
        self.recompute();
    }

    /// Configures an unbounded LP cone (`-x1 + x2 <= 4`, `x1 - 2 x2 <= 6`, `x1 + x2 >= 2`).
    pub fn apply_unbounded_preset(&mut self) {
        self.state.c = [2.0, 1.5];
        self.state.maximize = true;
        self.state.constraints = [[-1.0, 1.0, 4.0], [1.0, -2.0, 6.0], [1.0, 1.0, 2.0]];
        self.state.senses = [0, 0, 1];
        self.recompute();
    }

    fn recompute(&mut self) {
        let mut a = DMatrix::zeros(3, 2);
        let mut b = DVector::zeros(3);
        let mut senses = Vec::with_capacity(3);
        for i in 0..3 {
            a[(i, 0)] = self.state.constraints[i][0];
            a[(i, 1)] = self.state.constraints[i][1];
            b[i] = self.state.constraints[i][2];
            senses.push(code_to_sense(self.state.senses[i]));
        }
        let sense = if self.state.maximize {
            LpSense::Maximize
        } else {
            LpSense::Minimize
        };
        let c = DVector::from_row_slice(&self.state.c);
        if let Ok(lp) = LinearProgram::new(sense, c, a, b, senses) {
            if let Ok(sol) = SimplexSolver::default().solve(&lp) {
                self.solution = sol;
            }
        }
    }

    fn feasible_polygon(&self, box_max: f64) -> Vec<[f64; 2]> {
        let mut poly = vec![
            [0.0, 0.0],
            [box_max, 0.0],
            [box_max, box_max],
            [0.0, box_max],
        ];
        for i in 0..3 {
            let row = self.state.constraints[i];
            poly = clip_halfspace_polygon(
                &poly,
                row[0],
                row[1],
                code_to_sense(self.state.senses[i]),
                row[2],
            );
        }
        poly
    }

    fn draw_sidebar(&mut self, parent_ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        let panel_id = crate::framework::scoped_panel_id(parent_ui, "opt_lp_controls");
        PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            super::render_tool_header(ui, self.name());
            egui::ScrollArea::vertical()
                .id_salt("lp_sidebar_scroll")
                .show(ui, |ui| {
                    changed |= self.draw_objective_and_presets(ui);
                    ui.separator();
                    changed |= self.draw_constraint_editors(ui);
                    ui.separator();
                    self.draw_solution_readout(ui);
                });
        });
        changed
    }

    fn draw_objective_and_presets(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.horizontal(|ui| {
            changed |= ui
                .selectable_value(&mut self.state.maximize, true, "Maximize")
                .changed();
            changed |= ui
                .selectable_value(&mut self.state.maximize, false, "Minimize")
                .changed();
        });
        changed |= ui
            .add(egui::Slider::new(&mut self.state.c[0], -10.0..=10.0).text("Objective c₁"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.state.c[1], -10.0..=10.0).text("Objective c₂"))
            .changed();
        ui.horizontal_wrapped(|ui| {
            if ui.small_button("3-Vertex Optimal").clicked() {
                self.state = LpState::default();
                changed = true;
            }
            if ui.small_button("Infeasible").clicked() {
                self.apply_infeasible_preset();
            }
            if ui.small_button("Unbounded").clicked() {
                self.apply_unbounded_preset();
            }
        });
        changed
    }

    fn draw_constraint_editors(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        for i in 0..3 {
            ui.group(|ui| {
                ui.horizontal(|ui| {
                    ui.strong(format!("Constraint #{}", i + 1));
                    for (code, sym) in [(0_u8, "≤"), (1_u8, "≥"), (2_u8, "=")] {
                        changed |= ui
                            .selectable_value(&mut self.state.senses[i], code, sym)
                            .changed();
                    }
                });
                let row = &mut self.state.constraints[i];
                changed |= ui
                    .add(egui::Slider::new(&mut row[0], -6.0..=6.0).text("a₁"))
                    .changed();
                changed |= ui
                    .add(egui::Slider::new(&mut row[1], -6.0..=6.0).text("a₂"))
                    .changed();
                changed |= ui
                    .add(egui::Slider::new(&mut row[2], 0.0..=60.0).text("b"))
                    .changed();
            });
        }
        changed
    }

    fn draw_solution_readout(&self, ui: &mut egui::Ui) {
        let p = SemanticPalette::from_visuals(ui.visuals());
        let (badge, col) = match self.solution.status {
            LpStatus::Optimal => ("Status: OPTIMAL (Bland's Simplex)", p.success),
            LpStatus::Infeasible => ("Status: INFEASIBLE (Phase-I > 0)", p.error),
            LpStatus::Unbounded => ("Status: UNBOUNDED RAY", p.warning),
        };
        ui.colored_label(col, badge);
        ui.label(format!("Simplex Pivots: {}", self.solution.iterations));
        if self.solution.status == LpStatus::Optimal {
            ui.label(format!(
                "Primal x* = ({:.3}, {:.3})",
                self.solution.x[0], self.solution.x[1]
            ));
            ui.label(format!("Objective z* = {:.4}", self.solution.objective));
            let sp = &self.solution.shadow_prices;
            ui.label(format!(
                "Shadow Prices y* = [{:.3}, {:.3}, {:.3}]",
                sp[0], sp[1], sp[2]
            ));
        }
    }

    fn draw_plot(&self, ui: &mut egui::Ui) {
        let palette = SemanticPalette::from_visuals(ui.visuals());
        let box_max = 22.0;
        let poly = self.feasible_polygon(box_max);
        Plot::new("lp_simplex_plot")
            .x_axis_label("x₁ (≥ 0)")
            .y_axis_label("x₂ (≥ 0)")
            .data_aspect(1.0)
            .show(ui, |plot_ui| {
                if poly.len() >= 3 {
                    let fill = palette.series_color(0).gamma_multiply(0.18);
                    plot_ui.polygon(
                        Polygon::new("Feasible Polytope", poly)
                            .fill_color(fill)
                            .stroke(egui::Stroke::new(1.5_f32, palette.series_color(0))),
                    );
                }
                for (i, &r) in self.state.constraints.iter().enumerate() {
                    if let Some(seg) = constraint_boundary_segment(
                        r[0],
                        r[1],
                        r[2],
                        [0.0, box_max],
                        [0.0, box_max],
                    ) {
                        plot_ui.line(
                            Line::new(format!("Constraint #{}", i + 1), seg.to_vec())
                                .color(palette.series_color(i + 1))
                                .width(1.6_f32),
                        );
                    }
                }
                self.draw_simplex_trajectory(plot_ui, &palette, box_max);
            });
    }

    fn draw_simplex_trajectory(
        &self,
        plot_ui: &mut egui_plot::PlotUi<'_>,
        palette: &SemanticPalette,
        box_max: f64,
    ) {
        if !self.solution.pivot_path.is_empty() {
            plot_ui.line(
                Line::new("Simplex Pivot Path", self.solution.pivot_path.clone())
                    .color(palette.success)
                    .width(2.8_f32),
            );
            plot_ui.points(
                Points::new("Visited BFS Vertices", self.solution.pivot_path.clone())
                    .color(palette.success)
                    .radius(5.0_f32),
            );
        }
        if self.solution.status == LpStatus::Optimal {
            let opt = [self.solution.x[0], self.solution.x[1]];
            if let Some(iso) = constraint_boundary_segment(
                self.state.c[0],
                self.state.c[1],
                self.solution.objective,
                [0.0, box_max],
                [0.0, box_max],
            ) {
                plot_ui.line(
                    Line::new("Iso-Objective cᵀx = z*", iso.to_vec())
                        .color(palette.warning)
                        .style(LineStyle::dashed_dense())
                        .width(2.0_f32),
                );
            }
            plot_ui.points(
                Points::new("Optimal Vertex x*", vec![opt])
                    .color(palette.warning)
                    .radius(7.5_f32)
                    .shape(MarkerShape::Diamond),
            );
        }
    }
}

impl InteractiveTool for LinearProgrammingTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }

    fn name(&self) -> &'static str {
        "Linear Programming (Simplex)"
    }

    fn domain(&self) -> &'static str {
        "optimization"
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&self.state).ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let s: LpState =
            serde_json::from_str(state).map_err(|e| format!("Invalid LP state: {e}"))?;
        let valid = s.c.iter().all(|v| v.is_finite() && v.abs() <= 1e3)
            && s.constraints
                .iter()
                .flatten()
                .all(|v| v.is_finite() && v.abs() <= 1e4)
            && s.constraints.iter().all(|r| r[2] >= 0.0)
            && s.senses.iter().all(|&code| code <= 2);
        if !valid {
            return Err("LP coefficients or RHS bounds out of range".into());
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
