use crate::accessibility::AccessibleHoverText;
use crate::framework::InteractiveTool;
use eframe::egui;
use egui_plot::{Bar, BarChart, Plot};
use math_explorer::pure_math::number_theory::partitions;

#[derive(PartialEq, Clone, Copy, serde::Serialize, serde::Deserialize)]
enum PartitionFunctionType {
    PStar,
    M,
    TStar,
    A,
    B,
    K,
    L,
    FK,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct PartitionsWidgetState {
    precision: usize,
    selected_function: PartitionFunctionType,
    f_k_val: usize,
}

pub struct PartitionsWidget {
    precision: usize,
    selected_function: PartitionFunctionType,
    f_k_val: usize,
    coeffs: Vec<i64>,
}

impl Default for PartitionsWidget {
    fn default() -> Self {
        Self {
            precision: 20,
            selected_function: PartitionFunctionType::PStar,
            f_k_val: 1,
            coeffs: Vec::new(),
        }
    }
}

impl PartitionsWidget {
    fn calculate(&mut self) {
        let q_series = match self.selected_function {
            PartitionFunctionType::PStar => partitions::gen_p_star(self.precision),
            PartitionFunctionType::M => partitions::gen_m(self.precision),
            PartitionFunctionType::TStar => partitions::gen_t_star(self.precision),
            PartitionFunctionType::A => partitions::gen_a(self.precision),
            PartitionFunctionType::B => partitions::gen_b(self.precision),
            PartitionFunctionType::K => partitions::gen_k(self.precision),
            PartitionFunctionType::L => partitions::gen_l(self.precision),
            PartitionFunctionType::FK => partitions::f_k(self.f_k_val, self.precision),
        };

        self.coeffs = q_series.coeffs;
    }
}

impl InteractiveTool for PartitionsWidget {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable { self }
    fn name(&self) -> &'static str {
        "Partition Function"
    }
    fn domain(&self) -> &'static str {
        "number_theory"
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&PartitionsWidgetState {
            precision: self.precision,
            selected_function: self.selected_function,
            f_k_val: self.f_k_val,
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let s: PartitionsWidgetState = serde_json::from_str(state)
            .map_err(|e| format!("Invalid PartitionsWidget state: {e}"))?;
        if s.precision == 0 || s.f_k_val == 0 {
            return Err("precision and f_k_val must be positive in PartitionsWidget".to_string());
        }
        self.precision = s.precision;
        self.selected_function = s.selected_function;
        self.f_k_val = s.f_k_val;
        self.calculate();
        Ok(())
    }

    #[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
    fn show(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Partition Function & Q-Series");
            ui.label("Calculate and visualize restricted partition functions.");

            ui.horizontal(|ui| {
                ui.vertical(|ui| {
                    ui.set_min_width(200.0);
                    ui.heading("Configuration");

                    ui.add(
                        egui::Slider::new(&mut self.precision, 1..=200)
                            .text("Precision (max q power)"),
                    );

                    ui.separator();
                    ui.label("Select Partition Function:");

                    ui.radio_value(
                        &mut self.selected_function,
                        PartitionFunctionType::PStar,
                        "P*(n)",
                    );
                    ui.radio_value(
                        &mut self.selected_function,
                        PartitionFunctionType::M,
                        "M(n)",
                    );
                    ui.radio_value(
                        &mut self.selected_function,
                        PartitionFunctionType::TStar,
                        "T*(n)",
                    );
                    ui.radio_value(
                        &mut self.selected_function,
                        PartitionFunctionType::A,
                        "A(n)",
                    );
                    ui.radio_value(
                        &mut self.selected_function,
                        PartitionFunctionType::B,
                        "B(n)",
                    );
                    ui.radio_value(
                        &mut self.selected_function,
                        PartitionFunctionType::K,
                        "K(n)",
                    );
                    ui.radio_value(
                        &mut self.selected_function,
                        PartitionFunctionType::L,
                        "L(n)",
                    );
                    ui.horizontal(|ui| {
                        ui.radio_value(
                            &mut self.selected_function,
                            PartitionFunctionType::FK,
                            "f_k(n)",
                        );
                        if self.selected_function == PartitionFunctionType::FK {
                            ui.add(egui::DragValue::new(&mut self.f_k_val).range(1..=100));
                            ui.label("k");
                        }
                    });

                    ui.separator();

                    if ui
                        .button("▶ Calculate")
                        .accessible_hover_text(
                            "Calculate the Q-series coefficients up to the given precision",
                        )
                        .clicked()
                    {
                        self.calculate();
                    }

                    if !self.coeffs.is_empty() {
                        ui.separator();
                        ui.label("Calculated Coefficients:");
                        egui::ScrollArea::vertical()
                            .max_height(300.0)
                            .show(ui, |ui| {
                                for (n, coeff) in self.coeffs.iter().enumerate() {
                                    ui.label(format!("n = {}: {}", n, coeff));
                                }
                            });
                    }
                });

                ui.separator();

                ui.vertical(|ui| {
                    ui.heading("Partition Function Plot");
                    ui.label("Bar chart of coefficients for the selected generating function.");

                    if self.coeffs.is_empty() {
                        ui.label(
                            "Click 'Calculate' to generate the Q-series and display the plot.",
                        );
                        return;
                    }

                    let bars: Vec<Bar> = self
                        .coeffs
                        .iter()
                        .enumerate()
                        .map(|(n, &coeff)| Bar::new(n as f64, coeff as f64).width(0.8))
                        .collect();

                    let chart = BarChart::new("Coefficients", bars).name("Coefficients");

                    Plot::new("partition_plot")
                        .allow_zoom(true)
                        .allow_drag(true)
                        .show(ui, |plot_ui| {
                            plot_ui.bar_chart(chart);
                        });
                });
            });
        });
    }
}

// [cite:partitions_implementation]


inventory::submit! {
    crate::framework::ToolMetadata {
        id: "PartitionsWidget",
        display_name: "Partition Function",
        domain: "number_theory",
        tags: &[],
        build: || Box::new(PartitionsWidget::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for PartitionsWidget {
    fn theory_description(&self) -> String {
        format!(
            "Number theoretic partition functions, colored partitions, and Eisenstein series q-series expansions calculated to precision order $O(q^{{{}}})$: Euler generating function $P(q) = \\prod_{{n=1}}^\\infty \\frac{{1}}{{1 - q^n}}$.",
            self.precision
        )
    }
    fn phonetic_description(&self) -> String { "Integer partition functions and Eisenstein series theory".into() }
    fn theory_citation(&self) -> String { "On Eisenstein series, color partition and divisor function (Pushpa & Vasuki, 2022); The Theory of Partitions (Andrews, 1976)".into() }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert("PStar".into(), "Unrestricted partition counting function generating series".into());
        map.insert("Eisenstein".into(), "Modular forms and divisor sum Eisenstein series components".into());
        map
    }
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Integer Partitions (Hardy-Ramanujan)",
    "domain": "number_theory",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "precision": 25,
    "selected_function": "PStar",
    "f_k_val": 1
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "partitions_starter",
        title: "Integer Partitions (Hardy-Ramanujan)",
        description: "p(n) integer partition function asymptotic counting and Young Ferrers diagram displays.",
        category: "Combinatorics",
        icon: "🧱",
        domain: "number_theory",
        tool_name: Some("PartitionsWidget"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}
