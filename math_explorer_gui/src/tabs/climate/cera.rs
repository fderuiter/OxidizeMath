use crate::accessibility::AccessibleHoverText;
use crate::framework::recovery::{DiagnosticReport, FiniteStateGuard, RecoveryAction};
use crate::framework::InteractiveTool;
use eframe::egui;
use math_explorer::climate::cera::Cera;
use math_explorer::climate::config::CeraConfig;
use nalgebra::DMatrix;

pub struct CeraTool {
    pub(crate) in_channels: usize,
    pub(crate) latent_channels: usize,
    pub(crate) aligned_channels: usize,
    pub(crate) num_levels: usize,
    pub(crate) output_size: usize,
    pub(crate) prediction_result: Option<Result<Vec<f64>, String>>,
    pub(crate) diverged: bool,
    pub(crate) diagnostic_report: Option<DiagnosticReport>,
}

impl Default for CeraTool {
    fn default() -> Self {
        Self {
            in_channels: 2,
            latent_channels: 4,
            aligned_channels: 2,
            num_levels: 10,
            output_size: 5,
            prediction_result: None,
            diverged: false,
            diagnostic_report: None,
        }
    }
}

impl FiniteStateGuard for CeraTool {
    fn is_finite_state(&self) -> bool {
        !self.diverged
    }

    fn diagnostic_report(&self) -> Option<&DiagnosticReport> {
        self.diagnostic_report.as_ref()
    }

    fn diverged(&self) -> bool {
        self.diverged
    }

    fn clamp_parameters(&mut self) {
        self.in_channels = self.in_channels.clamp(1, 10);
        self.latent_channels = self.latent_channels.clamp(1, 20);
        self.aligned_channels = self.aligned_channels.clamp(1, self.latent_channels);
        self.num_levels = self.num_levels.clamp(1, 50);
        self.output_size = self.output_size.clamp(1, 20);
        self.diverged = false;
        self.diagnostic_report = None;
    }

    fn reset_parameters(&mut self) {
        self.in_channels = 2;
        self.latent_channels = 4;
        self.aligned_channels = 2;
        self.num_levels = 10;
        self.output_size = 5;
        self.prediction_result = None;
        self.diverged = false;
        self.diagnostic_report = None;
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct CeraState {
    in_channels: usize,
    latent_channels: usize,
    aligned_channels: usize,
    num_levels: usize,
    output_size: usize,
}

impl InteractiveTool for CeraTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }
    fn name(&self) -> &'static str {
        "CERA Model"
    }
    fn domain(&self) -> &'static str {
        "climate"
    }
    fn export_capabilities(&self) -> crate::framework::ExportCapabilities {
        crate::framework::ExportCapabilities::none()
    }

    fn has_finite_parameters(&self) -> bool {
        FiniteStateGuard::is_finite_state(self)
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&CeraState {
            in_channels: self.in_channels,
            latent_channels: self.latent_channels,
            aligned_channels: self.aligned_channels,
            num_levels: self.num_levels,
            output_size: self.output_size,
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let s: CeraState =
            serde_json::from_str(state).map_err(|e| format!("Invalid CERA state: {e}"))?;
        if s.in_channels == 0
            || s.latent_channels == 0
            || s.aligned_channels == 0
            || s.num_levels == 0
            || s.output_size == 0
        {
            return Err("CERA channel and level counts must be positive".into());
        }
        self.in_channels = s.in_channels;
        self.latent_channels = s.latent_channels;
        self.aligned_channels = s.aligned_channels;
        self.num_levels = s.num_levels;
        self.output_size = s.output_size;
        self.prediction_result = None;
        self.diverged = false;
        self.diagnostic_report = None;
        Ok(())
    }

    fn clamp_parameters(&mut self) -> bool {
        FiniteStateGuard::clamp_parameters(self);
        true
    }

    #[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
    fn show(&mut self, ctx: &egui::Context) {
        eframe::egui::CentralPanel::default().show(ctx, |ui| {
            self.show_ui(ui);
        });
    }

    #[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
    fn show_ui(&mut self, ui: &mut egui::Ui) {
        ui.label(
            "Configure CERA (Climate-invariant Encoding through Representation Alignment) Model:",
        );

        if self.diverged {
            if let Some(action) = crate::framework::recovery::render_recovery_banner(
                ui,
                self.diagnostic_report.as_ref(),
            ) {
                match action {
                    RecoveryAction::AutoScaleStepSize => {
                        self.num_levels = (self.num_levels / 2).max(1);
                        self.diverged = false;
                        self.diagnostic_report = None;
                    }
                    RecoveryAction::ClampParameters => {
                        FiniteStateGuard::clamp_parameters(self);
                    }
                    RecoveryAction::ResetParameters => {
                        self.reset_parameters();
                    }
                }
            }
            ui.separator();
        }

        ui.horizontal(|ui| {
            ui.add(egui::Slider::new(&mut self.in_channels, 1..=10).text("Input Channels:"));
        });

        ui.horizontal(|ui| {
            ui.add(egui::Slider::new(&mut self.latent_channels, 1..=20).text("Latent Channels:"));
        });

        ui.horizontal(|ui| {
            ui.add(
                egui::Slider::new(&mut self.aligned_channels, 1..=self.latent_channels)
                    .text("Aligned Channels:"),
            );
        });

        ui.horizontal(|ui| {
            ui.add(egui::Slider::new(&mut self.num_levels, 1..=50).text("Number of Levels:"));
        });

        ui.horizontal(|ui| {
            ui.add(egui::Slider::new(&mut self.output_size, 1..=20).text("Output Size:"));
        });

        if ui
            .button("▶ Run Prediction")
            .accessible_hover_text("Execute the CERA model with random inputs")
            .clicked()
        {
            let config = CeraConfig {
                in_channels: self.in_channels,
                latent_channels: self.latent_channels,
                aligned_channels: self.aligned_channels,
                num_levels: self.num_levels,
                output_size: self.output_size,
                epochs: 1,
                batch_size: 1,
                learning_rate: 0.01,
                lambda_pred: 1.0,
                lambda_emd: 0.1,
            };

            match Cera::new(config) {
                Ok(model) => {
                    let inputs = DMatrix::from_fn(self.num_levels, self.in_channels, |_, _| {
                        // allow(entropy_guard)
                        rand::random::<f32>()
                    });
                    let prediction = model.predict(&inputs);

                    let has_non_finite = prediction.iter().any(|v| !v.is_finite());
                    if has_non_finite {
                        self.diverged = true;
                        self.diagnostic_report = Some(DiagnosticReport::new(
                            0.0,
                            0,
                            "prediction",
                            "CERA neural network output diverged: non-finite prediction value detected.",
                        ));
                        self.prediction_result = None;
                    } else {
                        self.diverged = false;
                        self.diagnostic_report = None;
                        let mut result_vec = Vec::new();
                        for val in prediction.iter() {
                            result_vec.push(*val as f64);
                        }
                        self.prediction_result = Some(Ok(result_vec));
                    }
                }
                Err(e) => {
                    self.prediction_result = Some(Err(e));
                }
            }
        }

        ui.separator();

        if !self.diverged {
            match &self.prediction_result {
                Some(Ok(predictions)) => {
                    ui.label("Prediction Results:");
                    for (i, val) in predictions.iter().enumerate() {
                        ui.label(format!("Output {}: {:.4}", i, val));
                    }
                }
                Some(Err(err)) => {
                    let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
                    ui.colored_label(palette.error, format!("Error: {}", err));
                }
                None => {
                    ui.label("Click 'Run Prediction' to test the model with random inputs.");
                }
            }
        }
    }
}

// [cite:cera_framework]

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "CeraTool",
        display_name: "CERA Model",
        domain: "climate",
        tags: &["cera", "climate"],
        build: || Box::new(CeraTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for CeraTool {
    fn theory_description(&self) -> String {
        format!(
            "Climate-Equivariant Representation Architecture (CERA) aligning atmospheric fields across climate regimes with {} input channels, {} latent channels, and {} vertical pressure levels.",
            self.in_channels, self.latent_channels, self.num_levels
        )
    }
    fn phonetic_description(&self) -> String {
        "Climate-Equivariant Representation Architecture theory".into()
    }
    fn theory_citation(&self) -> String {
        "CERA: A Framework for Improved Generalization of Machine Learning Models to Changed Climates (Liu et al., 2025)".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "CERA Climate Alignment",
    "domain": "climate",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "in_channels": 4,
    "latent_channels": 8,
    "aligned_channels": 4,
    "num_levels": 12,
    "output_size": 2
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "cera_starter",
        title: "CERA Climate Alignment",
        description: "Climate-invariant multi-level latent representation alignment across atmospheric regimes.",
        category: "Climate Modeling",
        icon: "🌍",
        domain: "climate",
        tool_name: Some("CeraTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}
