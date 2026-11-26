#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::*;
use crate::framework::{bounded_drag_value, enforce_ordered_drag_range};
use crate::reflective_ui::render_formula_with_export;
use eframe::egui;

impl ActivationFunctionsTool {
    pub(super) fn draw_controls(&mut self, parent_ui: &mut egui::Ui) {
        let panel_id =
            crate::framework::scoped_panel_id(parent_ui, "activation_functions_controls");
        crate::framework::PaneSidePanel::right(panel_id).show_inside(parent_ui, |ui| {
            ui.heading("Controls");
            ui.separator();
            ui.label("Select Activation Function:");
            for (fn_kind, label) in [
                (ActivationFunction::ReLU, "ReLU"),
                (ActivationFunction::Sigmoid, "Sigmoid"),
                (ActivationFunction::Tanh, "Tanh"),
                (ActivationFunction::Gelu, "GELU"),
                (ActivationFunction::Custom, "Custom Formula"),
            ] {
                ui.radio_value(&mut self.selected_function, fn_kind, label);
            }
            if self.selected_function == ActivationFunction::Custom {
                ui.add_space(6.0);
                self.math_input.ui(ui);
            }
            self.draw_range_controls(ui);
            self.draw_math_info(ui);
        });
    }

    fn draw_range_controls(&mut self, ui: &mut egui::Ui) {
        ui.separator();
        ui.label("Plot Range");
        let mut min_changed = false;
        let mut max_changed = false;
        ui.horizontal(|ui| {
            ui.label("X Min:");
            min_changed = ui
                .add(bounded_drag_value(
                    &mut self.x_min,
                    ACTIVATION_X_MIN_BOUND..=ACTIVATION_X_MAX_BOUND,
                    0.1,
                ))
                .changed();
        });
        ui.horizontal(|ui| {
            ui.label("X Max:");
            max_changed = ui
                .add(bounded_drag_value(
                    &mut self.x_max,
                    ACTIVATION_X_MIN_BOUND..=ACTIVATION_X_MAX_BOUND,
                    0.1,
                ))
                .changed();
        });
        enforce_ordered_drag_range(
            &mut self.x_min,
            &mut self.x_max,
            ACTIVATION_X_MIN_BOUND,
            ACTIVATION_X_MAX_BOUND,
            ACTIVATION_MIN_SPAN,
            min_changed,
            max_changed,
        );
    }

    fn draw_math_info(&self, ui: &mut egui::Ui) {
        ui.separator();
        ui.label("Math Info");
        match self.selected_function {
            ActivationFunction::ReLU => {
                render_formula_with_export(ui, r"f(x) = \max(0, x)", Some("f(x) = max(0, x)"));
                render_formula_with_export(
                    ui,
                    r"f'(x) = 1 \text{ if } x > 0 \text{ else } 0",
                    Some("f'(x) = 1 if x > 0 else 0"),
                );
            }
            ActivationFunction::Sigmoid => {
                render_formula_with_export(
                    ui,
                    r"f(x) = \frac{1}{1 + e^{-x}}",
                    Some("f(x) = 1 / (1 + e^(-x))"),
                );
                render_formula_with_export(
                    ui,
                    r"f'(x) = f(x)(1 - f(x))",
                    Some("f'(x) = f(x) * (1 - f(x))"),
                );
            }
            ActivationFunction::Tanh => {
                render_formula_with_export(ui, r"f(x) = \tanh(x)", Some("f(x) = tanh(x)"));
                render_formula_with_export(
                    ui,
                    r"f'(x) = 1 - \tanh^2(x)",
                    Some("f'(x) = 1 - tanh^2(x)"),
                );
            }
            ActivationFunction::Gelu => {
                render_formula_with_export(
                    ui,
                    r"f(x) \approx 0.5x(1 + \tanh(\sqrt{2/\pi}(x + 0.044715x^3)))",
                    Some("f(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))"),
                );
            }
            ActivationFunction::Custom => {
                render_formula_with_export(
                    ui,
                    r"f'(x) \approx \frac{f(x + h) - f(x - h)}{2h}",
                    Some("f'(x) = (f(x + h) - f(x - h)) / (2 * h)"),
                );
            }
        }
    }
}
