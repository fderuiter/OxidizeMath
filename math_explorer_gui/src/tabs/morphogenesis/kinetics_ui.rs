//! Sidebar controls, velocity plot, theory metadata, and starter preset for [`super::EnzymeKineticsTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{
    EnzymeKineticsTool, EnzymeKineticsVariant, MAX_HILL_N, MAX_S_MAX, MIN_HILL_N, MIN_KM,
    MIN_S_MAX, MIN_VMAX,
};
use crate::framework::empirical_csv::EmpiricalPanelOptions;
use crate::reflective_ui::render_copyable_metric;
use eframe::egui;
use egui_plot::{HLine, Line, MarkerShape, Plot, PlotPoints, PlotUi, Points, VLine};
use math_explorer::biology::kinetics::{HillKinetics, KineticsModel, MichaelisMenten};

impl EnzymeKineticsTool {
    pub(super) fn render_sidebar(&mut self, ui: &mut egui::Ui) {
        ui.heading("Kinetic Rate Law");
        egui::ComboBox::from_id_salt("enzyme_kinetics_variant")
            .selected_text(self.variant.label())
            .show_ui(ui, |ui| {
                for v in [
                    EnzymeKineticsVariant::MichaelisMenten,
                    EnzymeKineticsVariant::Hill,
                ] {
                    if ui.selectable_value(&mut self.variant, v, v.label()).changed() {
                        self.revision = self.revision.wrapping_add(1).max(1);
                    }
                }
            });
        ui.add_space(6.0);
        if ui
            .add(egui::Slider::new(&mut self.v_max, MIN_VMAX..=1_000.0).text("Vmax (Max Velocity)"))
            .changed()
            || ui
                .add(egui::Slider::new(&mut self.k_m, MIN_KM..=500.0).text("Km (Half-Saturation)"))
                .changed()
        {
            self.revision = self.revision.wrapping_add(1).max(1);
        }
        if self.variant == EnzymeKineticsVariant::Hill
            && ui
                .add(
                    egui::Slider::new(&mut self.hill_n, MIN_HILL_N..=MAX_HILL_N)
                        .text("Hill Coefficient (n)"),
                )
                .changed()
        {
            self.revision = self.revision.wrapping_add(1).max(1);
        }
        if ui
            .add(
                egui::Slider::new(&mut self.substrate_max, 5.0..=1_000.0)
                    .text("Max Substrate [S]max"),
            )
            .changed()
        {
            self.revision = self.revision.wrapping_add(1).max(1);
        }
        ui.separator();
        self.render_sensitivity(ui);
        ui.separator();
        let opts = EmpiricalPanelOptions {
            id_salt: "enzyme_kinetics_csv",
            x_label: "Substrate [S] Column (X)",
            y_label: "Velocity v Column (Y)",
            aux_label: None,
            residual_filename: "enzyme_kinetics_residuals.csv",
            fit_button_label: "Fit Parameters",
        };
        if self.empirical_csv.show_panel(ui, &opts) {
            let ctx = ui.ctx().clone();
            let _ = self.calibrate_from_loaded_csv(Some(&ctx));
        }
    }

    fn render_sensitivity(&mut self, ui: &mut egui::Ui) {
        let variant = self.variant;
        let s_probe = self.k_m.max(0.1);
        let ctx_key = (variant as u8, crate::sensitivity::canonical_f64_bits(s_probe));
        let params: &[(&str, f64)] = match variant {
            EnzymeKineticsVariant::MichaelisMenten => &[("Vmax", self.v_max), ("Km", self.k_m)],
            EnzymeKineticsVariant::Hill => {
                &[("Vmax", self.v_max), ("Km", self.k_m), ("n", self.hill_n)]
            }
        };
        self.sensitivity_engine
            .compute_with_context(params, &ctx_key, |p_map| {
                let vm = p_map.get("Vmax").copied().unwrap_or(100.0).max(1e-6);
                let km = p_map.get("Km").copied().unwrap_or(25.0).max(1e-6);
                let n = p_map.get("n").copied().unwrap_or(1.0).max(0.1);
                match variant {
                    EnzymeKineticsVariant::MichaelisMenten => MichaelisMenten::new(vm, km)
                        .and_then(|m| m.reaction_velocity(s_probe))
                        .unwrap_or(0.0),
                    EnzymeKineticsVariant::Hill => HillKinetics::new(vm, km, n)
                        .and_then(|m| m.reaction_velocity(s_probe))
                        .unwrap_or(0.0),
                }
            });
        crate::sensitivity::render_sensitivity_panel_ui(
            ui,
            &self.sensitivity_engine,
            "enzyme_kinetics_tool",
        );
    }

    pub(super) fn sample_curve(&self) -> Vec<[f64; 2]> {
        let s_max = self.substrate_max.clamp(MIN_S_MAX, MAX_S_MAX);
        (0..=160)
            .map(|i| {
                let s = s_max * (i as f64) / 160.0;
                [s, self.evaluate_velocity(s)]
            })
            .collect()
    }

    pub(super) fn render_plot(&self, ui: &mut egui::Ui) {
        ui.heading("Enzyme Reaction Velocity Curve");
        let pts = self.sample_curve();
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        Plot::new("enzyme_kinetics_plot")
            .view_aspect(2.0)
            .x_axis_label("Substrate Concentration [S]")
            .y_axis_label("Reaction Velocity v")
            .include_y(0.0)
            .include_y(self.v_max * 1.05)
            .legend(egui_plot::Legend::default())
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new(self.variant.label(), PlotPoints::new(pts))
                        .color(palette.series_color(0)),
                );
                plot_ui.hline(
                    HLine::new("Vmax", self.v_max)
                        .color(palette.series_color(1))
                        .style(egui_plot::LineStyle::Dashed { length: 8.0 }),
                );
                plot_ui.hline(
                    HLine::new("Vmax / 2", 0.5 * self.v_max)
                        .color(palette.warning)
                        .style(egui_plot::LineStyle::Dotted { spacing: 6.0 }),
                );
                plot_ui.vline(
                    VLine::new("Km", self.k_m)
                        .color(palette.warning)
                        .style(egui_plot::LineStyle::Dotted { spacing: 6.0 }),
                );
                self.render_empirical_points(plot_ui, palette.series_color(2));
            });

        ui.add_space(6.0);
        render_copyable_metric(
            ui,
            "Half-Maximal Velocity (Vmax / 2)",
            &format!("{:.3}", 0.5 * self.v_max),
        );
        render_copyable_metric(
            ui,
            "Catalytic Efficiency Proxy (Vmax / Km)",
            &format!("{:.4}", self.v_max / self.k_m.max(1e-12)),
        );
    }

    fn render_empirical_points(&self, plot_ui: &mut PlotUi<'_>, color: egui::Color32) {
        let Some(ref table) = self.empirical_csv.table else {
            return;
        };
        let pts = table.scatter_points(self.empirical_csv.x_col, self.empirical_csv.y_col);
        if !pts.is_empty() {
            plot_ui.points(
                Points::new(format!("Observed Rates ({})", table.name), pts)
                    .radius(4.0_f32)
                    .shape(MarkerShape::Circle)
                    .color(color),
            );
        }
    }
}

impl scientific_metadata::theory::TheoryDescribable for EnzymeKineticsTool {
    fn theory_description(&self) -> String {
        match self.variant {
            EnzymeKineticsVariant::MichaelisMenten => format!(
                "Michaelis-Menten enzyme saturation kinetics: $v = \\frac{{V_{{\\max}}[S]}}{{K_m + [S]}}$ with $V_{{\\max}} = {:.2}$ and $K_m = {:.2}$.",
                self.v_max, self.k_m
            ),
            EnzymeKineticsVariant::Hill => format!(
                "Cooperative Hill enzyme kinetics: $v = \\frac{{V_{{\\max}}[S]^n}}{{K_m^n + [S]^n}}$ with $V_{{\\max}} = {:.2}$, $K_m = {:.2}$, and Hill exponent $n = {:.2}$.",
                self.v_max, self.k_m, self.hill_n
            ),
        }
    }
    fn phonetic_description(&self) -> String {
        "Michaelis-Menten and Hill cooperative enzyme kinetics theory".into()
    }
    fn theory_citation(&self) -> String {
        "Die Kinetik der Invertinwirkung (Michaelis & Menten, 1913); Hill (1910)".into()
    }
    fn theory_latex_equation(&self) -> Option<String> {
        Some(match self.variant {
            EnzymeKineticsVariant::MichaelisMenten => r"v = \frac{V_{\max}[S]}{K_m + [S]}".into(),
            EnzymeKineticsVariant::Hill => r"v = \frac{V_{\max}[S]^n}{K_m^n + [S]^n}".into(),
        })
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "Michaelis-Menten".into(),
            "Non-cooperative single-site saturation kinetics (n = 1)".into(),
        );
        map.insert(
            "Hill Kinetics".into(),
            "Allosteric cooperative binding kinetics with Hill exponent n".into(),
        );
        map
    }
}

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "EnzymeKineticsTool",
        display_name: "Enzyme Kinetics",
        domain: "biology",
        tags: &["enzyme", "kinetics", "michaelis-menten", "hill", "calibration"],
        build: || Box::new(EnzymeKineticsTool::default()),
    }
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Cooperative Hill Enzyme Kinetics",
    "domain": "biology",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "variant": "Hill",
    "v_max": 120.0,
    "k_m": 30.0,
    "hill_n": 2.2,
    "substrate_max": 200.0
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "enzyme_kinetics_starter",
        title: "Cooperative Hill Enzyme Kinetics",
        description: "v = Vmax·[S]^n / (Km^n + [S]^n): Michaelis-Menten and cooperative Hill saturation curves with CSV fitting.",
        category: "Biochemistry",
        icon: "🧪",
        domain: "biology",
        tool_name: Some("EnzymeKineticsTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}
