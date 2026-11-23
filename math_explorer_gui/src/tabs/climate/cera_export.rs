#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::CeraTool;
use crate::app::SemanticPalette;
use crate::framework::export_table::{ExportCell, ExportTable};
use crate::framework::ExportCapabilities;
use eframe::egui;
use egui_plot::export_primitives::ExportPrimitive;

impl CeraTool {
    pub(super) fn current_export_capabilities(&self) -> ExportCapabilities {
        let has_loss = !self.loss_history.is_empty();
        let has_preds = matches!(&self.prediction_result, Some(Ok(v)) if !v.is_empty());
        if has_loss || has_preds {
            ExportCapabilities::plot()
        } else {
            ExportCapabilities::unpopulated(
                "Click 'Train & Predict (3 Epochs)' to train the CERA model before exporting.",
            )
        }
    }

    pub(super) fn build_export_primitives(&self) -> Vec<ExportPrimitive> {
        let palette = SemanticPalette::from_visuals(&egui::Visuals::dark());
        let mut prims = Vec::new();

        let loss_coords: Vec<[f64; 2]> = self
            .loss_history
            .iter()
            .enumerate()
            .filter(|(_, l)| l.is_finite())
            .map(|(idx, &loss)| [(idx + 1) as f64, f64::from(loss)])
            .collect();
        if !loss_coords.is_empty() {
            prims.push(ExportPrimitive::Polyline {
                name: "Training Loss".into(),
                points: loss_coords.clone(),
                color: palette.accent,
                stroke_width: 2.2,
                closed: false,
                fill: None,
            });
            prims.push(ExportPrimitive::Points {
                name: "Epoch Loss".into(),
                points: loss_coords,
                color: palette.accent,
                radius: 3.5,
            });
        }

        if let Some(Ok(preds)) = &self.prediction_result {
            for (ch, &val) in preds.iter().enumerate() {
                if val.is_finite() {
                    let cx = (ch + 1) as f64 + 0.25;
                    let y0 = val.min(0.0);
                    let y1 = val.max(0.0);
                    prims.push(ExportPrimitive::Bar {
                        name: format!("Channel {ch} Prediction"),
                        min: [cx - 0.18, y0],
                        max: [cx + 0.18, y1.max(y0 + 1e-4)],
                        fill: palette.axis_y,
                        stroke_color: palette.node_stroke,
                        stroke_width: 1.0,
                    });
                }
            }
        }
        prims
    }

    pub(super) fn build_export_tables(&self) -> Vec<ExportTable> {
        let mut tables = Vec::new();
        if !self.loss_history.is_empty() {
            let loss_rows: Vec<Vec<ExportCell>> = self
                .loss_history
                .iter()
                .enumerate()
                .map(|(idx, &loss)| {
                    vec![
                        ExportCell::UInt((idx + 1) as u64),
                        ExportCell::Float(f64::from(loss)),
                    ]
                })
                .collect();
            tables.push(ExportTable::new(
                "Training Loss",
                vec!["epoch".into(), "total_loss".into()],
                loss_rows,
            ));
        }

        if let Some(Ok(preds)) = &self.prediction_result {
            if !preds.is_empty() {
                let pred_rows: Vec<Vec<ExportCell>> = preds
                    .iter()
                    .enumerate()
                    .map(|(ch, &val)| {
                        vec![ExportCell::UInt(ch as u64), ExportCell::Float(val)]
                    })
                    .collect();
                tables.push(ExportTable::new(
                    "Channel Predictions",
                    vec!["channel".into(), "predicted_value".into()],
                    pred_rows,
                ));
            }
        }
        tables
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cera_tool_reduces_loss_and_emits_distinct_channels() {
        let mut tool = CeraTool::default();
        assert!(!tool.current_export_capabilities().dataset);
        tool.train_epochs(3);
        tool.train_epochs(3);
        assert!(tool.current_export_capabilities().dataset);
        assert!(tool.current_export_capabilities().figure);

        assert_eq!(tool.epochs_trained, 6);
        assert_eq!(tool.loss_history.len(), 6);
        let first_loss = *tool.loss_history.first().unwrap();
        let last_loss = *tool.loss_history.last().unwrap();
        assert!(
            last_loss < first_loss,
            "Expected CERA loss to decrease across epochs ({first_loss} -> {last_loss})"
        );

        let preds = tool
            .prediction_result
            .as_ref()
            .expect("prediction populated")
            .as_ref()
            .expect("valid prediction");
        assert_eq!(preds.len(), tool.output_size);
        assert!(
            preds.windows(2).any(|w| (w[0] - w[1]).abs() > 1e-6),
            "Expected distinct channel outputs, got {preds:?}"
        );
    }
}
