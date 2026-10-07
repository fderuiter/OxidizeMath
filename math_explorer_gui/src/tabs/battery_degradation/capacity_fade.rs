use crate::framework::InteractiveTool;
use crate::reflective_ui::render_copyable_metric;
use crate::widgets::{GridDataSource, NumericalDataGrid};
use eframe::egui;
use egui_plot::{HLine, Line, Plot, PlotPoints};
use math_explorer::applied::battery_degradation::{Cycles, DepthOfDischarge, PowerLawModel};
use std::collections::HashMap;

#[derive(Clone, Copy, PartialEq)]
enum CapacityFadeViewMode {
    Plot,
    DataGrid,
}

pub struct CapacityFadeTool {
    dod: f64,
    temperature: f64,
    cycles_to_simulate: f64,
    view_mode: CapacityFadeViewMode,
    grid_widget: NumericalDataGrid,
    overrides: HashMap<usize, f64>,
}

impl Default for CapacityFadeTool {
    fn default() -> Self {
        Self {
            dod: 80.0,
            temperature: 25.0,
            cycles_to_simulate: 2000.0,
            view_mode: CapacityFadeViewMode::Plot,
            grid_widget: NumericalDataGrid::new(),
            overrides: HashMap::new(),
        }
    }
}

struct CapacityGridAdapter<'a> {
    tool: &'a mut CapacityFadeTool,
}

impl<'a> GridDataSource for CapacityGridAdapter<'a> {
    fn num_rows(&self) -> usize {
        101
    }

    fn num_cols(&self) -> usize {
        4
    }

    fn header(&self, col: usize) -> String {
        match col {
            0 => "Step".to_string(),
            1 => "Cycle".to_string(),
            2 => "State of Health (SoH)".to_string(),
            3 => "Degradation (%)".to_string(),
            _ => String::new(),
        }
    }

    fn cell_value(&self, row: usize, col: usize) -> f64 {
        let cycle = (self.tool.cycles_to_simulate / 100.0) * (row as f64);
        match col {
            0 => row as f64,
            1 => cycle,
            2 => {
                if let Some(&ov) = self.tool.overrides.get(&row) {
                    ov
                } else {
                    let model = PowerLawModel::standard();
                    if let (Ok(c), Ok(d)) = (Cycles::new(cycle), DepthOfDischarge::new(self.tool.dod)) {
                        model.capacity(c, d).as_f64()
                    } else {
                        0.0
                    }
                }
            }
            3 => {
                let soh = self.cell_value(row, 2);
                (1.0 - soh) * 100.0
            }
            _ => 0.0,
        }
    }

    fn set_cell_value(&mut self, row: usize, col: usize, val: f64) {
        if col == 2 {
            self.tool.overrides.insert(row, val.clamp(0.0, 1.0));
        }
    }

    fn is_editable(&self, _row: usize, col: usize) -> bool {
        col == 2
    }
}

impl InteractiveTool for CapacityFadeTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable { self }
    fn name(&self) -> &'static str {
        "Capacity Fade"
    }

    #[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
    fn show(&mut self, ctx: &egui::Context) {
        // Enforce valid ranges for internal state before using it
        self.dod = self.dod.clamp(0.0, 100.0);
        self.cycles_to_simulate = self.cycles_to_simulate.max(100.0);

        egui::SidePanel::left("battery_controls").show(ctx, |ui| {
            ui.heading("Parameters");
            ui.add_space(5.0);

            ui.add(egui::Slider::new(&mut self.dod, 0.0..=100.0).text("Depth of Discharge (DoD) - %"));
            ui.small("Percentage of battery capacity used per cycle.");

            ui.add_space(10.0);

            ui.add_enabled_ui(false, |ui| {
                ui.add(egui::Slider::new(&mut self.temperature, -20.0..=60.0).text("Temperature (°C)"));
            })
            .response
            .on_disabled_hover_text(
                "Temperature dependency is not yet implemented in the core model.",
            );

            ui.add_space(10.0);

            ui.add(egui::Slider::new(&mut self.cycles_to_simulate, 100.0..=10000.0).text("Simulation Range - Cycles"));

            ui.add_space(15.0);
            ui.separator();
            ui.heading("View Mode");
            ui.selectable_value(&mut self.view_mode, CapacityFadeViewMode::Plot, "📈 Plot View");
            ui.selectable_value(&mut self.view_mode, CapacityFadeViewMode::DataGrid, "🔢 Numerical Data Grid");
            if !self.overrides.is_empty() && ui.button("↻ Clear Overrides").clicked() {
                self.overrides.clear();
            }
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            if self.view_mode == CapacityFadeViewMode::DataGrid {
                ui.heading("Capacity Fade Numerical Grid");
                ui.label("Double-click State of Health (SoH) cells to perform scenario overwrites.");
                ui.separator();
                let mut grid_widget = self.grid_widget.clone();
                let mut adapter = CapacityGridAdapter { tool: self };
                grid_widget.show(ui, &mut adapter);
                self.grid_widget = grid_widget;
            } else {
                ui.heading("Capacity Fade Projection");
                ui.label("Estimation based on Power Law Model for Li-ion batteries.");

            let model = PowerLawModel::standard();
            let dod_val = DepthOfDischarge::new(self.dod);

            // Generate plot points
            let points: Vec<[f64; 2]> = (0..=100)
                .map(|i| {
                    let cycle = (self.cycles_to_simulate / 100.0) * (i as f64);
                    if let Some(&ov) = self.overrides.get(&i) {
                        [cycle, ov]
                    } else if let (Ok(c), Ok(d)) = (Cycles::new(cycle), dod_val.clone()) {
                        let capacity = model.capacity(c, d);
                        [cycle, capacity.as_f64()]
                    } else {
                        [cycle, 0.0]
                    }
                })
                .collect();

            let line = Line::new("Capacity", PlotPoints::new(points));

            Plot::new("capacity_fade_plot")
                .view_aspect(2.0)
                .x_axis_label("Cycles")
                .y_axis_label("State of Health (Capacity)")
                .include_y(0.0)
                .include_y(1.0)
                .show(ui, |plot_ui| {
                    plot_ui.line(line);
                    // Add a horizontal line at 70% (End of Life)
                    plot_ui.hline(
                        HLine::new("End of Life (70%)", 0.7)
                            .color(egui::Color32::RED)
                            .style(egui_plot::LineStyle::Dashed { length: 10.0 }),
                    );
                    // Add a horizontal line at 80% (First Life)
                    plot_ui.hline(
                        HLine::new("First Life (80%)", 0.8)
                            .color(egui::Color32::YELLOW)
                            .style(egui_plot::LineStyle::Dashed { length: 10.0 }),
                    );
                });

            ui.add_space(10.0);

            // Calculate N70 for current DoD
            let n70 = if let Ok(d) = dod_val {
                model.n70(d).as_f64()
            } else {
                0.0
            };
            render_copyable_metric(
                ui,
                "Projected Cycle Life (to 70%)",
                &format!("{:.0} cycles", n70),
            );
            }
        });
    }
}

// [cite:mmwave_radiotherapy_setup]


inventory::submit! {
    crate::framework::ToolMetadata {
        name: "CapacityFadeTool",
        domain: "battery_degradation",
        tags: &[],
        build: || Box::new(CapacityFadeTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for CapacityFadeTool {
    fn theory_description(&self) -> String { "Theoretical context not available.".into() }
    fn phonetic_description(&self) -> String { "Theoretical context not available.".into() }
    fn theory_citation(&self) -> String { "Uncited".into() }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> { std::collections::HashMap::new() }
}
