use crate::framework::InteractiveTool;
use crate::reflective_ui::render_copyable_metric;
use crate::widgets::{GridDataSource, NumericalDataGrid};
use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};
use math_explorer::epidemiology::compartmental::SIRModel;

#[derive(Clone, Copy, PartialEq)]
enum SirViewMode {
    Plot,
    DataGrid,
}

pub struct SirTool {
    n: f64,
    i0: f64,
    beta: f64,
    gamma: f64,
    duration: f64,

    view_mode: SirViewMode,
    grid_widget: NumericalDataGrid,

    // Cached plot data
    s_points: Vec<[f64; 2]>,
    i_points: Vec<[f64; 2]>,
    r_points: Vec<[f64; 2]>,
}

struct SirGridAdapter<'a> {
    tool: &'a mut SirTool,
}

impl<'a> GridDataSource for SirGridAdapter<'a> {
    fn num_rows(&self) -> usize {
        self.tool.s_points.len()
    }

    fn num_cols(&self) -> usize {
        6
    }

    fn header(&self, col: usize) -> String {
        match col {
            0 => "Step".to_string(),
            1 => "Time (days)".to_string(),
            2 => "Susceptible (S)".to_string(),
            3 => "Infected (I)".to_string(),
            4 => "Recovered (R)".to_string(),
            5 => "Total Population (N)".to_string(),
            _ => String::new(),
        }
    }

    fn cell_value(&self, row: usize, col: usize) -> f64 {
        if row >= self.tool.s_points.len() {
            return 0.0;
        }
        match col {
            0 => row as f64,
            1 => self.tool.s_points[row][0],
            2 => self.tool.s_points[row][1],
            3 => self.tool.i_points.get(row).map_or(0.0, |p| p[1]),
            4 => self.tool.r_points.get(row).map_or(0.0, |p| p[1]),
            5 => {
                let s = self.cell_value(row, 2);
                let i = self.cell_value(row, 3);
                let r = self.cell_value(row, 4);
                s + i + r
            }
            _ => 0.0,
        }
    }

    fn set_cell_value(&mut self, row: usize, col: usize, val: f64) {
        if row < self.tool.s_points.len() {
            if row == 0 {
                if col == 3 {
                    self.tool.i0 = val;
                    self.tool.recalculate();
                } else if col == 2 {
                    self.tool.n = val + self.tool.i0;
                    self.tool.recalculate();
                }
            } else {
                match col {
                    2 => self.tool.s_points[row][1] = val,
                    3 => self.tool.i_points[row][1] = val,
                    4 => self.tool.r_points[row][1] = val,
                    _ => {}
                }
            }
        }
    }

    fn is_editable(&self, _row: usize, col: usize) -> bool {
        (2..=4).contains(&col)
    }
}

impl Default for SirTool {
    fn default() -> Self {
        let mut tool = Self {
            n: 1000.0,
            i0: 10.0,
            beta: 0.5,
            gamma: 0.1,
            duration: 100.0,
            view_mode: SirViewMode::Plot,
            grid_widget: NumericalDataGrid::new(),
            s_points: vec![],
            i_points: vec![],
            r_points: vec![],
        };
        tool.recalculate();
        tool
    }
}

impl SirTool {
    fn recalculate(&mut self) {
        self.s_points.clear();
        self.i_points.clear();
        self.r_points.clear();

        // Ensure parameters are valid before passing to model
        let n = self.n.max(1.0);
        let i0 = self.i0.clamp(0.0, n);
        let beta = self.beta.max(0.0);
        let gamma = self.gamma.max(0.0);

        if let Ok(mut model) = SIRModel::new(n, i0, beta, gamma) {
            let dt = 0.1;
            let steps = (self.duration / dt) as usize;

            for i in 0..=steps {
                let t = i as f64 * dt;
                let state = model.state();
                self.s_points.push([t, state.s]);
                self.i_points.push([t, state.i]);
                self.r_points.push([t, state.r]);

                model.step(dt);
            }
        }
    }
}

impl InteractiveTool for SirTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable { self }
    fn name(&self) -> &'static str {
        "SIR Model"
    }

    fn show(&mut self, ctx: &egui::Context) {
        eframe::egui::CentralPanel::default().show(ctx, |ui| {
            self.show_ui(ui);
        });
    }
    #[allow(clippy::too_many_lines)]
    fn show_ui(&mut self, ui: &mut egui::Ui) {
        ui.vertical(|ui| {
            ui.heading("Parameters");
            let mut changed = false;

            changed |= ui
                .add(egui::Slider::new(&mut self.n, 100.0..=100_000.0).text("Population (N)"))
                .changed();
            changed |= ui
                .add(
                    egui::Slider::new(&mut self.i0, 1.0..=self.n / 2.0)
                        .text("Initial Infected (I0)"),
                )
                .changed();
            changed |= ui
                .add(egui::Slider::new(&mut self.beta, 0.0..=5.0).text("Transmission Rate (beta)"))
                .changed();
            changed |= ui
                .add(egui::Slider::new(&mut self.gamma, 0.0..=1.0).text("Recovery Rate (gamma)"))
                .changed();
            changed |= ui
                .add(egui::Slider::new(&mut self.duration, 10.0..=365.0).text("Duration (days)"))
                .changed();

            if changed {
                self.recalculate();
            }

            let r0 = if self.gamma > 0.0 {
                self.beta / self.gamma
            } else {
                f64::INFINITY
            };
            render_copyable_metric(ui, "Basic Reproduction Number (R₀)", &format!("{:.2}", r0));

            ui.separator();
            ui.heading("View Mode");
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.view_mode, SirViewMode::Plot, "📈 Plot View");
                ui.selectable_value(&mut self.view_mode, SirViewMode::DataGrid, "🔢 Numerical Data Grid");
            });

            ui.separator();
            if self.view_mode == SirViewMode::DataGrid {
                ui.heading("SIR Numerical Simulation Data Grid");
                ui.label("Double-click S, I, R cells to edit simulation values.");
                ui.separator();
                let mut grid_widget = self.grid_widget.clone();
                let mut adapter = SirGridAdapter { tool: self };
                grid_widget.show(ui, &mut adapter);
                self.grid_widget = grid_widget;
            } else {
                ui.heading("Simulation");

                let plot = Plot::new("sir_plot")
                    .view_aspect(2.0)
                    .legend(egui_plot::Legend::default());

                plot.show(ui, |plot_ui| {
                    plot_ui.line(
                        Line::new("Susceptible", PlotPoints::new(self.s_points.clone()))
                            .color(egui::Color32::BLUE),
                    );
                    plot_ui.line(
                        Line::new("Infected", PlotPoints::new(self.i_points.clone()))
                            .color(egui::Color32::RED),
                    );
                    plot_ui.line(
                        Line::new("Recovered", PlotPoints::new(self.r_points.clone()))
                            .color(egui::Color32::GREEN),
                    );
                });
            }
        });
    }
}

// [cite:graph_parameters_rust]


inventory::submit! {
    crate::framework::ToolMetadata {
        name: "SirTool",
        domain: "epidemiology",
        tags: &[],
        build: || Box::new(SirTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for SirTool {
    fn theory_description(&self) -> String { "Theoretical context not available.".into() }
    fn phonetic_description(&self) -> String { "Theoretical context not available.".into() }
    fn theory_citation(&self) -> String { "Uncited".into() }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> { std::collections::HashMap::new() }
}
