#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use egui_plot::{ExportPrimitive, SharedPoints};
use math_explorer::climate::{cera::Cera, config::CeraConfig};

use crate::app::ThemeMode;
use crate::framework::{ExportCell, ExportTable};

use super::{CeraTool, TRAIN_BATCH_SIZE};

impl CeraTool {
    fn preview_or_trained_predictions(&self) -> Option<(Vec<f64>, f64)> {
        if self.diverged {
            return None;
        }
        if let Some(Ok(ref preds)) = self.prediction_result {
            if !preds.is_empty() && preds.iter().all(|v| v.is_finite()) {
                let last_loss = self.loss_history.last().copied().unwrap_or(0.0) as f64;
                return Some((preds.clone(), last_loss));
            }
        }
        let config = CeraConfig {
            in_channels: self.in_channels.max(1),
            latent_channels: self.latent_channels.max(1),
            aligned_channels: self.aligned_channels.clamp(1, self.latent_channels.max(1)),
            num_levels: self.num_levels.max(1),
            output_size: self.output_size.max(1),
            epochs: 1,
            batch_size: TRAIN_BATCH_SIZE,
            learning_rate: 0.05,
            lambda_pred: 0.2,
            lambda_emd: 0.1,
        };
        let model = Cera::new(config).ok()?;
        let (control_inputs, control_targets, _) = self.build_training_batches();
        let eval_sample = control_inputs.rows(0, self.num_levels.max(1)).clone_owned();
        let raw_pred = model.predict(&eval_sample);
        if raw_pred.iter().any(|v| !v.is_finite()) {
            return None;
        }
        let preds: Vec<f64> = raw_pred.iter().map(|&v| f64::from(v)).collect();
        let target_row = control_targets.row(0);
        let mse = preds
            .iter()
            .enumerate()
            .map(|(i, &p)| {
                let t = f64::from(target_row.get(i).copied().unwrap_or(0.0));
                (p - t) * (p - t)
            })
            .sum::<f64>()
            / (preds.len().max(1) as f64);
        Some((preds, mse))
    }

    pub(super) fn build_cera_plot_primitives(&self) -> Vec<ExportPrimitive> {
        let Some((preds, preview_loss)) = self.preview_or_trained_predictions() else {
            return Vec::new();
        };
        let palette = ThemeMode::Dark.palette();
        let mut prims = Vec::with_capacity(preds.len() + 3);
        let x_max = (preds.len().saturating_sub(1) as f64 + 0.6).max(1.0);
        prims.push(ExportPrimitive::Polyline {
            name: "Zero Baseline".to_string(),
            points: SharedPoints::from(vec![[-0.6, 0.0], [x_max, 0.0]]),
            color: palette.graph_edge,
            stroke_width: 1.0,
            closed: false,
            fill: None,
        });
        for (idx, &val) in preds.iter().enumerate() {
            let x = idx as f64;
            let span = if val.abs() < 0.02 { 0.02 } else { val };
            let (lo, hi) = if span >= 0.0 {
                (0.0, span)
            } else {
                (span, 0.0)
            };
            let fill = palette.series[idx % palette.series.len()];
            prims.push(ExportPrimitive::Bar {
                name: format!("Channel {idx}"),
                min: [x - 0.34, lo],
                max: [x + 0.34, hi],
                fill,
                stroke_color: palette.node_stroke,
                stroke_width: 1.2,
            });
        }
        let loss_pts: Vec<[f64; 2]> = if self.loss_history.is_empty() {
            vec![[0.0, preview_loss], [x_max, preview_loss]]
        } else {
            let denom = (self.loss_history.len().saturating_sub(1)).max(1) as f64;
            self.loss_history
                .iter()
                .enumerate()
                .map(|(i, &l)| [(i as f64 / denom) * x_max, f64::from(l)])
                .collect()
        };
        prims.push(ExportPrimitive::Polyline {
            name: "CERA Loss".to_string(),
            points: SharedPoints::from(loss_pts),
            color: palette.accent,
            stroke_width: 2.0,
            closed: false,
            fill: None,
        });
        prims
    }

    pub(super) fn build_cera_export_tables(&self) -> Vec<ExportTable> {
        let Some((preds, preview_loss)) = self.preview_or_trained_predictions() else {
            return Vec::new();
        };
        let pred_rows = preds
            .iter()
            .enumerate()
            .map(|(ch, &val)| {
                vec![
                    ExportCell::Int(ch as i64),
                    ExportCell::Float(val),
                    ExportCell::Int(self.epochs_trained as i64),
                    ExportCell::Int(self.in_channels as i64),
                    ExportCell::Int(self.latent_channels as i64),
                    ExportCell::Int(self.aligned_channels as i64),
                    ExportCell::Int(self.num_levels as i64),
                ]
            })
            .collect();
        let pred_table = ExportTable::new(
            "Atmospheric Channel Predictions",
            vec![
                "channel".into(),
                "predicted_value".into(),
                "epochs_trained".into(),
                "in_channels".into(),
                "latent_channels".into(),
                "aligned_channels".into(),
                "num_levels".into(),
            ],
            pred_rows,
        );
        let loss_rows = if self.loss_history.is_empty() {
            vec![vec![ExportCell::Int(0), ExportCell::Float(preview_loss)]]
        } else {
            self.loss_history
                .iter()
                .enumerate()
                .map(|(idx, &loss)| {
                    vec![
                        ExportCell::Int((idx + 1) as i64),
                        ExportCell::Float(f64::from(loss)),
                    ]
                })
                .collect()
        };
        let loss_table = ExportTable::new(
            "Training Loss History",
            vec!["epoch".into(), "cera_loss".into()],
            loss_rows,
        );
        vec![pred_table, loss_table]
    }
}

impl scientific_metadata::theory::TheoryDescribable for CeraTool {
    fn maturity_tier(&self) -> scientific_metadata::theory::MaturityTier {
        scientific_metadata::theory::MaturityTier::Experimental
    }
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
        std::collections::HashMap::from([
            (
                "in_channels".into(),
                "Number of atmospheric input variables per vertical level.".into(),
            ),
            (
                "latent_channels".into(),
                "Total compressed autoencoder latent channels.".into(),
            ),
            (
                "aligned_channels".into(),
                "Subset of latent channels aligned via Earth Mover's Distance.".into(),
            ),
        ])
    }
}
