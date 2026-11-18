use crate::framework::InteractiveTool;
use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};

pub struct TemperatureAnomaliesTool {
    time_series: Vec<[f64; 2]>,
}

impl Default for TemperatureAnomaliesTool {
    fn default() -> Self {
        Self {
            time_series: math_explorer::climate::dataset::get_temperature_anomalies(),
        }
    }
}

impl InteractiveTool for TemperatureAnomaliesTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable { self }
    fn name(&self) -> &'static str {
        "Temperature Anomalies"
    }
    fn domain(&self) -> &'static str {
        "climate"
    }

    fn save_state(&self) -> Option<String> {
        #[derive(serde::Serialize, serde::Deserialize)]
        struct TempState {
            time_series: Vec<[f64; 2]>,
        }
        serde_json::to_string(&TempState {
            time_series: self.time_series.clone(),
        }).ok()
    }

    fn load_state(&mut self, state: &str) {
        #[derive(serde::Serialize, serde::Deserialize)]
        struct TempState {
            time_series: Vec<[f64; 2]>,
        }
        if let Ok(s) = serde_json::from_str::<TempState>(state) {
            self.time_series = s.time_series;
        }
    }

    fn show(&mut self, ctx: &egui::Context) {
        eframe::egui::CentralPanel::default().show(ctx, |ui| {
            self.show_ui(ui);
        });
    }
    fn show_ui(&mut self, ui: &mut egui::Ui) {
        ui.label("Global Temperature Anomalies:");

        let plot = Plot::new("temperature_anomalies_plot")
            .view_aspect(2.0)
            .x_axis_formatter(|x, _range| format!("{:.0}", x.value))
            .y_axis_formatter(|y, _range| format!("{:.2}°C", y.value))
            .legend(egui_plot::Legend::default());

        plot.show(ui, |plot_ui| {
            let points = PlotPoints::new(self.time_series.clone());
            let line = Line::new("Temperature Anomaly", points).width(2.0_f32);
            plot_ui.line(line);
        });
    }
}

// [cite:dwarf_galaxy_empirical_dependencies]


inventory::submit! {
    crate::framework::ToolMetadata {
        id: "TemperatureAnomaliesTool",
        display_name: "Temperature Anomalies",
        domain: "climate",
        tags: &[],
        build: || Box::new(TemperatureAnomaliesTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for TemperatureAnomaliesTool {
    fn theory_description(&self) -> String {
        "Global surface temperature anomalies relative to the 1951-1980 climatological baseline: $\\Delta T(t) = T(t) - T_{\\text{baseline}}$, tracking global warming trends and radiative forcing.".into()
    }
    fn phonetic_description(&self) -> String { "Global surface temperature anomalies and baseline climate theory".into() }
    fn theory_citation(&self) -> String { "Improvements in the GISTEMP Uncertainty Analysis (Lenssen et al., 2019)".into() }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> { std::collections::HashMap::new() }
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Global Temperature Anomalies",
    "domain": "climate",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "smoothing_years": 5,
    "baseline_start": 1951
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "temperature_anomalies_starter",
        title: "Global Temperature Anomalies",
        description: "Zonal surface temperature variance relative to 1951-1980 climatological baseline means.",
        category: "Climatology",
        icon: "🌡️",
        domain: "climate",
        tool_name: Some("TemperatureAnomaliesTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}
