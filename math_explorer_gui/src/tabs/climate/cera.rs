use crate::accessibility::AccessibleHoverText;
use crate::framework::recovery::{DiagnosticReport, FiniteStateGuard, RecoveryAction};
use crate::framework::InteractiveTool;
use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};
use math_explorer::climate::{
    autoencoder::Autoencoder, cera::Cera, config::CeraConfig, predictor::Predictor,
    training::CeraTrainer,
};
use nalgebra::DMatrix;
use rand::Rng;

const MAX_IN_CHANNELS: usize = 10;
const MAX_LATENT_CHANNELS: usize = 20;
const MAX_NUM_LEVELS: usize = 50;
const MAX_OUTPUT_SIZE: usize = 20;
const TRAIN_BATCH_SIZE: usize = 2;
const TRAIN_SAMPLES: usize = 4;
const EPOCHS_PER_CLICK: usize = 3;

type PersistedCeraModel = Cera<Autoencoder, Predictor>;
type ModelArchSignature = (usize, usize, usize, usize, usize);

pub struct CeraTool {
    pub(crate) in_channels: usize,
    pub(crate) latent_channels: usize,
    pub(crate) aligned_channels: usize,
    pub(crate) num_levels: usize,
    pub(crate) output_size: usize,
    pub(crate) prediction_result: Option<Result<Vec<f64>, String>>,
    pub(crate) diverged: bool,
    pub(crate) diagnostic_report: Option<DiagnosticReport>,
    pub(crate) model: Option<PersistedCeraModel>,
    pub(crate) model_signature: Option<ModelArchSignature>,
    pub(crate) loss_history: Vec<f32>,
    pub(crate) epochs_trained: usize,
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
            model: None,
            model_signature: None,
            loss_history: Vec::new(),
            epochs_trained: 0,
        }
    }
}

impl CeraTool {
    fn current_signature(&self) -> ModelArchSignature {
        (
            self.in_channels,
            self.latent_channels,
            self.aligned_channels,
            self.num_levels,
            self.output_size,
        )
    }

    pub(crate) fn clear_persisted_model(&mut self) {
        self.model = None;
        self.model_signature = None;
        self.loss_history.clear();
        self.epochs_trained = 0;
    }

    fn build_training_batches(&self) -> (DMatrix<f32>, DMatrix<f32>, DMatrix<f32>) {
        let mut rng = oxidize_core::rng::OxidizeRng::new(0x00CE_2A13_2026_0001);
        let rows = TRAIN_SAMPLES * self.num_levels;
        let levels = self.num_levels.max(1) as f32;
        let out_dim = self.output_size.max(1) as f32;

        let control_inputs = DMatrix::from_fn(rows, self.in_channels, |r, c| {
            let level_frac = (r % self.num_levels.max(1)) as f32 / levels;
            let channel_phase = (c + 1) as f32 * 0.25;
            (1.0 - 0.5 * level_frac) * channel_phase + (rng.r#gen::<f32>() - 0.5) * 0.2
        });
        let warm_inputs = DMatrix::from_fn(rows, self.in_channels, |r, c| {
            let level_frac = (r % self.num_levels.max(1)) as f32 / levels;
            let channel_phase = (c + 1) as f32 * 0.25;
            0.35 + (1.0 - 0.45 * level_frac) * channel_phase + (rng.r#gen::<f32>() - 0.5) * 0.2
        });
        let control_targets = DMatrix::from_fn(TRAIN_SAMPLES, self.output_size, |s, c| {
            let base = -0.4 + 0.8 * ((c + 1) as f32 / out_dim);
            base + (s as f32 * 0.03) + (rng.r#gen::<f32>() - 0.5) * 0.05
        });
        (control_inputs, control_targets, warm_inputs)
    }

    pub(crate) fn train_epochs(&mut self, epochs: usize) {
        let bounded_epochs = epochs.clamp(1, 5);
        let sig = self.current_signature();
        if self.model.is_none() || self.model_signature != Some(sig) {
            let config = CeraConfig {
                in_channels: self.in_channels,
                latent_channels: self.latent_channels,
                aligned_channels: self.aligned_channels,
                num_levels: self.num_levels,
                output_size: self.output_size,
                epochs: bounded_epochs,
                batch_size: TRAIN_BATCH_SIZE,
                learning_rate: 0.05,
                lambda_pred: 0.2,
                lambda_emd: 0.1,
            };
            match Cera::new(config) {
                Ok(new_model) => {
                    self.model = Some(new_model);
                    self.model_signature = Some(sig);
                    self.loss_history.clear();
                    self.epochs_trained = 0;
                }
                Err(e) => {
                    self.clear_persisted_model();
                    self.prediction_result = Some(Err(e));
                    return;
                }
            }
        }

        let (control_inputs, control_targets, warm_inputs) = self.build_training_batches();
        let eval_sample = control_inputs.rows(0, self.num_levels).clone_owned();

        if let Some(model) = self.model.as_mut() {
            model.config.epochs = bounded_epochs;
            let mut trainer = CeraTrainer::new(model);
            let epoch_losses = trainer.train(&control_inputs, &control_targets, &warm_inputs);
            let prediction = model.predict(&eval_sample);

            let non_finite = epoch_losses.iter().any(|v| !v.is_finite())
                || prediction.iter().any(|v| !v.is_finite());
            if non_finite {
                self.diverged = true;
                self.diagnostic_report = Some(DiagnosticReport::new(
                    0.0,
                    self.epochs_trained,
                    "prediction",
                    "CERA neural network output diverged: non-finite loss or prediction detected.",
                ));
                self.prediction_result = None;
            } else {
                self.diverged = false;
                self.diagnostic_report = None;
                self.epochs_trained += epoch_losses.len();
                self.loss_history.extend(epoch_losses);
                let result_vec: Vec<f64> = prediction.iter().map(|v| *v as f64).collect();
                self.prediction_result = Some(Ok(result_vec));
            }
        }
    }

    fn draw_controls(&mut self, ui: &mut egui::Ui) {
        let mut arch_changed = false;
        arch_changed |= ui
            .add(
                egui::Slider::new(&mut self.in_channels, 1..=MAX_IN_CHANNELS)
                    .text("Input Channels"),
            )
            .changed();
        if ui
            .add(
                egui::Slider::new(&mut self.latent_channels, 1..=MAX_LATENT_CHANNELS)
                    .text("Latent Channels"),
            )
            .changed()
        {
            self.aligned_channels = self.aligned_channels.min(self.latent_channels).max(1);
            arch_changed = true;
        }
        arch_changed |= ui
            .add(
                egui::Slider::new(&mut self.aligned_channels, 1..=self.latent_channels)
                    .text("Aligned Channels"),
            )
            .changed();
        arch_changed |= ui
            .add(
                egui::Slider::new(&mut self.num_levels, 1..=MAX_NUM_LEVELS)
                    .text("Number of Levels"),
            )
            .changed();
        arch_changed |= ui
            .add(egui::Slider::new(&mut self.output_size, 1..=MAX_OUTPUT_SIZE).text("Output Size"))
            .changed();

        if arch_changed {
            self.clear_persisted_model();
            self.prediction_result = None;
        }

        ui.horizontal(|ui| {
            if ui
                .button("▶ Train & Predict (3 Epochs)")
                .accessible_hover_text(
                    "Train the persisted CERA model with CeraTrainer and evaluate atmospheric outputs",
                )
                .clicked()
            {
                self.train_epochs(EPOCHS_PER_CLICK);
            }
            if ui
                .button("↻ Reset Weights")
                .accessible_hover_text("Reset persisted CERA model weights and loss history")
                .clicked()
            {
                self.clear_persisted_model();
                self.prediction_result = None;
            }
        });
    }

    fn draw_results(&self, ui: &mut egui::Ui) {
        if !self.loss_history.is_empty() {
            let latest_loss = self.loss_history.last().copied().unwrap_or(0.0);
            let initial_loss = self.loss_history.first().copied().unwrap_or(latest_loss);
            ui.label(format!(
                "Trained Epochs: {} | Initial CERA Loss: {:.4} → Current CERA Loss: {:.4}",
                self.epochs_trained, initial_loss, latest_loss
            ));
            let pts: Vec<[f64; 2]> = self
                .loss_history
                .iter()
                .enumerate()
                .map(|(i, &l)| [(i + 1) as f64, l as f64])
                .collect();
            Plot::new("cera_loss_plot")
                .height(160.0)
                .x_axis_formatter(|x, _| format!("Epoch {:.0}", x.value))
                .show(ui, |plot_ui| {
                    plot_ui.line(Line::new("CERA Loss", PlotPoints::new(pts)).width(2.0_f32));
                });
            ui.separator();
        }

        match &self.prediction_result {
            Some(Ok(predictions)) => {
                ui.label("Trained Atmospheric Channel Predictions:");
                for (i, val) in predictions.iter().enumerate() {
                    ui.label(format!("Output Channel {}: {:.4}", i, val));
                }
            }
            Some(Err(err)) => {
                let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
                ui.colored_label(palette.error, format!("Error: {err}"));
            }
            None => {
                ui.label(
                    "Click 'Train & Predict (3 Epochs)' to train the persisted CERA model and inspect aligned outputs.",
                );
            }
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
        self.in_channels = self.in_channels.clamp(1, MAX_IN_CHANNELS);
        self.latent_channels = self.latent_channels.clamp(1, MAX_LATENT_CHANNELS);
        self.aligned_channels = self.aligned_channels.clamp(1, self.latent_channels);
        self.num_levels = self.num_levels.clamp(1, MAX_NUM_LEVELS);
        self.output_size = self.output_size.clamp(1, MAX_OUTPUT_SIZE);
        self.diverged = false;
        self.diagnostic_report = None;
        self.clear_persisted_model();
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
        self.clear_persisted_model();
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
        self.state_bounds().validate_json(state)?;
        let s: CeraState =
            serde_json::from_str(state).map_err(|e| format!("Invalid CERA state: {e}"))?;
        if !(1..=MAX_IN_CHANNELS).contains(&s.in_channels)
            || !(1..=MAX_LATENT_CHANNELS).contains(&s.latent_channels)
            || !(1..=s.latent_channels).contains(&s.aligned_channels)
            || !(1..=MAX_NUM_LEVELS).contains(&s.num_levels)
            || !(1..=MAX_OUTPUT_SIZE).contains(&s.output_size)
        {
            return Err(format!(
                "CERA parameters out of bounds (expected in_channels in 1..={MAX_IN_CHANNELS}, latent_channels in 1..={MAX_LATENT_CHANNELS}, aligned_channels in 1..=latent_channels, num_levels in 1..={MAX_NUM_LEVELS}, output_size in 1..={MAX_OUTPUT_SIZE})"
            ));
        }
        self.in_channels = s.in_channels;
        self.latent_channels = s.latent_channels;
        self.aligned_channels = s.aligned_channels;
        self.num_levels = s.num_levels;
        self.output_size = s.output_size;
        self.prediction_result = None;
        self.diverged = false;
        self.diagnostic_report = None;
        self.clear_persisted_model();
        Ok(())
    }

    fn clamp_parameters(&mut self) -> bool {
        FiniteStateGuard::clamp_parameters(self);
        true
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        egui::CentralPanel::default().show_inside(ui, |ui| {
            ui.heading("CERA Climate Representation Alignment");
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
                            self.clear_persisted_model();
                        }
                        RecoveryAction::ClampParameters => FiniteStateGuard::clamp_parameters(self),
                        RecoveryAction::ResetParameters => self.reset_parameters(),
                    }
                }
                ui.separator();
            }

            self.draw_controls(ui);
            ui.separator();
            if !self.diverged {
                self.draw_results(ui);
            }
        });
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
            "Climate-invariant Encoding through Representation Alignment (CERA) optimizing $L = (1 - \\lambda_{{\\text{{pred}}}} - \\lambda_{{\\text{{emd}}}}) L_{{\\text{{recon}}}} + \\lambda_{{\\text{{pred}}}} L_{{\\text{{pred}}}} + \\lambda_{{\\text{{emd}}}} \\text{{EMD}}(Z_c, Z_w)$ across {} input channels, {} latent ({} aligned) channels, and {} pressure levels.",
            self.in_channels, self.latent_channels, self.aligned_channels, self.num_levels
        )
    }
    fn phonetic_description(&self) -> String {
        "Climate-invariant Encoding through Representation Alignment autoencoder and Earth Mover Distance alignment theory.".into()
    }
    fn theory_citation(&self) -> String {
        "CERA: A Framework for Improved Generalization of Machine Learning Models to Changed Climates (Liu et al., 2025)".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "in_channels".into(),
            "Number of atmospheric input variables per vertical level.".into(),
        );
        map.insert(
            "latent_channels".into(),
            "Total compressed autoencoder latent channels.".into(),
        );
        map.insert(
            "aligned_channels".into(),
            "Subset of latent channels aligned via Earth Mover's Distance.".into(),
        );
        map
    }
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": { "schema_version": "1.0", "title": "CERA Climate Alignment", "domain": "climate", "timestamp": "2026-11-16T00:00:00Z" },
  "config": { "in_channels": 4, "latent_channels": 8, "aligned_channels": 4, "num_levels": 12, "output_size": 2 }
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

#[cfg(test)]
#[path = "cera_tests.rs"]
mod tests;
