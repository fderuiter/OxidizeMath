#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{
    RootDimension, RootFinderTool, ScalarRootMethod, MAX_ROOT_TOL, MIN_ROOT_TOL, ROOT_DOMAIN_MAX,
    ROOT_DOMAIN_MIN, ROOT_MIN_SPAN,
};
use crate::accessibility::announce_live_status;
use crate::framework::{bounded_drag_value, enforce_ordered_drag_range};
use crate::reflective_ui::render_copyable_metric;
use eframe::egui;
use egui_plot::{Line, LineStyle, MarkerShape, Plot, PlotPoints, Points, Polygon};

impl RootFinderTool {
    pub(super) fn draw_controls(&mut self, parent_ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        let panel_id = crate::framework::scoped_panel_id(parent_ui, "root_finder_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            ui.heading("Root Finder Controls");
            ui.separator();
            changed |= self.draw_mode_and_method_selector(ui);
            ui.add_space(6.0);
            changed |= self.draw_expression_inputs(ui);
            ui.add_space(8.0);
            changed |= self.draw_domain_and_solver_sliders(ui);
            ui.separator();
            self.draw_step_scrubber_and_metrics(ui);
        });
        changed
    }

    fn draw_mode_and_method_selector(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.horizontal(|ui| {
            ui.label("System:");
            if ui
                .selectable_value(
                    &mut self.dimension,
                    RootDimension::Scalar1D,
                    "1D Scalar f(x)=0",
                )
                .changed()
            {
                announce_live_status(ui.ctx(), "Switched Root Finder to 1D scalar mode");
                changed = true;
            }
            if ui
                .selectable_value(
                    &mut self.dimension,
                    RootDimension::System2D,
                    "2D Coupled F(x,y)=0",
                )
                .changed()
            {
                announce_live_status(ui.ctx(), "Switched Root Finder to 2D coupled Newton mode");
                changed = true;
            }
        });

        if self.dimension == RootDimension::Scalar1D {
            ui.add_space(4.0);
            ui.label("1D Solver Strategy:");
            egui::ComboBox::from_id_salt("root_finder_scalar_method")
                .selected_text(self.scalar_method.name())
                .show_ui(ui, |ui| {
                    for method in ScalarRootMethod::ALL {
                        if ui
                            .selectable_value(&mut self.scalar_method, method, method.name())
                            .changed()
                        {
                            announce_live_status(
                                ui.ctx(),
                                format!("Selected {} root solver", method.name()),
                            );
                            changed = true;
                        }
                    }
                });
        } else {
            ui.small("Solver: Damped Multivariate Newton-Raphson");
        }
        changed
    }

    fn draw_expression_inputs(&mut self, ui: &mut egui::Ui) -> bool {
        if self.dimension == RootDimension::Scalar1D {
            self.scalar_input.ui(ui)
        } else {
            let mut ch = self.coupled_bank.ui_inputs(ui, "root_finder_2d_bank");
            ch |= self.coupled_bank.sliders_ui(ui, "root_finder_2d_sliders");
            ch
        }
    }

    fn draw_domain_and_solver_sliders(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = if self.dimension == RootDimension::Scalar1D {
            self.draw_1d_bracket_controls(ui)
        } else {
            self.draw_2d_domain_controls(ui)
        };
        changed |= ui
            .add(
                egui::Slider::new(&mut self.tolerance, MIN_ROOT_TOL..=MAX_ROOT_TOL)
                    .logarithmic(true)
                    .text("Tolerance"),
            )
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.max_iterations, 4..=200).text("Max Iterations"))
            .changed();
        changed
    }

    fn draw_1d_bracket_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let label = if self.scalar_method.is_bracketing() {
            "Bracket [a, b]"
        } else {
            "Seed Pair / Window [x0, x1]"
        };
        ui.label(label);
        let (mut min_ch, mut max_ch) = (false, false);
        ui.horizontal(|ui| {
            min_ch = ui
                .add(bounded_drag_value(
                    &mut self.bracket_min,
                    ROOT_DOMAIN_MIN..=ROOT_DOMAIN_MAX,
                    0.05,
                ))
                .changed();
            ui.label("to");
            max_ch = ui
                .add(bounded_drag_value(
                    &mut self.bracket_max,
                    ROOT_DOMAIN_MIN..=ROOT_DOMAIN_MAX,
                    0.05,
                ))
                .changed();
        });
        enforce_ordered_drag_range(
            &mut self.bracket_min,
            &mut self.bracket_max,
            ROOT_DOMAIN_MIN,
            ROOT_DOMAIN_MAX,
            ROOT_MIN_SPAN,
            min_ch,
            max_ch,
        );
        min_ch || max_ch
    }

    fn draw_2d_domain_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.label("Initial Guess (x0, y0):");
        ui.horizontal(|ui| {
            changed |= ui
                .add(bounded_drag_value(
                    &mut self.init_2d[0],
                    ROOT_DOMAIN_MIN..=ROOT_DOMAIN_MAX,
                    0.05,
                ))
                .changed();
            changed |= ui
                .add(bounded_drag_value(
                    &mut self.init_2d[1],
                    ROOT_DOMAIN_MIN..=ROOT_DOMAIN_MAX,
                    0.05,
                ))
                .changed();
        });
        changed |= ui
            .add(egui::Slider::new(&mut self.domain_span_2d, 0.5..=25.0).text("Contour Radius"))
            .changed();
        changed
    }

    fn draw_step_scrubber_and_metrics(&mut self, ui: &mut egui::Ui) {
        let max_idx = self.max_step_index();
        if max_idx > 0
            && ui
                .add(egui::Slider::new(&mut self.selected_step, 0..=max_idx).text("Inspect Step"))
                .changed()
        {
            announce_live_status(
                ui.ctx(),
                format!("Inspecting iteration step {}", self.selected_step),
            );
        }

        ui.heading("Convergence & Residuals");
        if let Some(err) = &self.status_error {
            ui.colored_label(ui.visuals().warn_fg_color, err);
        }
        match self.dimension {
            RootDimension::Scalar1D => {
                if let Some(r) = self.converged_root_1d {
                    render_copyable_metric(ui, "Root x*", &format!("{r:.10}"));
                }
                render_copyable_metric(
                    ui,
                    "Final |f(x*)|",
                    &format!("{:.3e}", self.final_residual),
                );
                render_copyable_metric(ui, "Iterations", &format!("{}", self.scalar_steps.len()));
                if let Some(step) = self.scalar_steps.get(self.selected_step) {
                    render_copyable_metric(ui, "Step x_k", &format!("{:.8}", step.x));
                    render_copyable_metric(ui, "Step |f(x_k)|", &format!("{:.3e}", step.fx.abs()));
                }
            }
            RootDimension::System2D => {
                if let Some([rx, ry]) = self.converged_root_2d {
                    render_copyable_metric(ui, "Root (x*, y*)", &format!("({rx:.7}, {ry:.7})"));
                }
                render_copyable_metric(
                    ui,
                    "Final ||F||_inf",
                    &format!("{:.3e}", self.final_residual),
                );
                render_copyable_metric(
                    ui,
                    "Newton Steps",
                    &format!("{}", self.system_steps.len().saturating_sub(1)),
                );
                if let Some(step) = self.system_steps.get(self.selected_step) {
                    render_copyable_metric(
                        ui,
                        "Step ||F(x_k)||_inf",
                        &format!("{:.3e}", step.inf_norm),
                    );
                }
            }
        }
    }

    pub(super) fn draw_plot(&self, parent_ui: &mut egui::Ui) {
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            ui.heading("Root Geometry & Contour View");
            let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
            let (x_lbl, y_lbl, aspect) = match self.dimension {
                RootDimension::Scalar1D => ("x", "f(x)", 2.0),
                RootDimension::System2D => ("x", "y", 1.0),
            };
            Plot::new("root_finder_plot")
                .title("Root Finder Visualization")
                .view_aspect(aspect)
                .x_axis_label(x_lbl)
                .y_axis_label(y_lbl)
                .legend(egui_plot::Legend::default())
                .show(ui, |plot_ui| match self.dimension {
                    RootDimension::Scalar1D => self.render_1d_plot_items(plot_ui, &palette),
                    RootDimension::System2D => self.render_2d_plot_items(plot_ui, &palette),
                });
        });
    }

    fn render_1d_plot_items(
        &self,
        plot_ui: &mut egui_plot::PlotUi<'_>,
        palette: &crate::app::SemanticPalette,
    ) {
        let span = (self.bracket_max - self.bracket_min).abs().max(0.1);
        let (pmin, pmax) = (self.bracket_min - 0.15 * span, self.bracket_max + 0.15 * span);
        plot_ui.line(
            Line::new("y = 0", PlotPoints::new(vec![[pmin, 0.0], [pmax, 0.0]]))
                .color(palette.graph_edge),
        );
        plot_ui.line(
            Line::new("f(x)", PlotPoints::new(self.cached_1d_curve.clone()))
                .color(palette.series_color(0)),
        );

        if let Some(step) = self.scalar_steps.get(self.selected_step) {
            if let Some([a, b]) = step.bracket {
                let y_hi = self
                    .cached_1d_curve
                    .iter()
                    .fold(1.0_f64, |acc, p| acc.max(p[1].abs()));
                let fill = egui::Color32::from_rgba_unmultiplied(
                    palette.series_color(1).r(),
                    palette.series_color(1).g(),
                    palette.series_color(1).b(),
                    45,
                );
                plot_ui.polygon(
                    Polygon::new(
                        "Active Bracket",
                        PlotPoints::new(vec![[a, -y_hi], [b, -y_hi], [b, y_hi], [a, y_hi]]),
                    )
                    .fill_color(fill),
                );
            }
            if let Some([p0, p1]) = step.construction_segment {
                let geom_label = if self.scalar_method == ScalarRootMethod::NewtonRaphson {
                    "Newton Tangent"
                } else {
                    "Secant / Bracket Chord"
                };
                plot_ui.line(
                    Line::new(geom_label, PlotPoints::new(vec![p0, p1]))
                        .style(LineStyle::Dashed { length: 8.0 })
                        .color(palette.series_color(2)),
                );
            }
            plot_ui.points(
                Points::new("Step Iterate", PlotPoints::new(vec![[step.x, step.fx]]))
                    .radius(5.0_f32)
                    .shape(MarkerShape::Diamond)
                    .color(palette.series_color(2)),
            );
        }
        if let Some(root) = self.converged_root_1d {
            plot_ui.points(
                Points::new("Converged Root", PlotPoints::new(vec![[root, 0.0]]))
                    .radius(6.0_f32)
                    .shape(MarkerShape::Circle)
                    .color(palette.accent),
            );
        }
    }

    fn render_2d_plot_items(
        &self,
        plot_ui: &mut egui_plot::PlotUi<'_>,
        palette: &crate::app::SemanticPalette,
    ) {
        for (idx, seg) in self.contours_f1.iter().enumerate() {
            let name = if idx == 0 { "F1(x, y) = 0" } else { "" };
            plot_ui.line(
                Line::new(name, PlotPoints::new(seg.clone())).color(palette.series_color(0)),
            );
        }
        for (idx, seg) in self.contours_f2.iter().enumerate() {
            let name = if idx == 0 { "F2(x, y) = 0" } else { "" };
            plot_ui.line(
                Line::new(name, PlotPoints::new(seg.clone())).color(palette.series_color(1)),
            );
        }
        let path: Vec<[f64; 2]> = self
            .system_steps
            .iter()
            .take(self.selected_step + 1)
            .map(|s| s.point)
            .collect();
        if !path.is_empty() {
            plot_ui.line(
                Line::new("Newton Path", PlotPoints::new(path.clone()))
                    .style(LineStyle::Dashed { length: 7.0 })
                    .color(palette.series_color(2)),
            );
            plot_ui.points(
                Points::new("Newton Iterates", PlotPoints::new(path))
                    .radius(4.5_f32)
                    .color(palette.series_color(2)),
            );
        }
        if let Some([rx, ry]) = self.converged_root_2d {
            plot_ui.points(
                Points::new("Converged 2D Root", PlotPoints::new(vec![[rx, ry]]))
                    .radius(6.5_f32)
                    .shape(MarkerShape::Cross)
                    .color(palette.accent),
            );
        }
    }
}
