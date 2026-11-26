#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use std::sync::Arc;

use crate::async_sim::SimulationRunner;
use crate::framework::{ExportCell, ExportTable};
use math_commons::math_kernel::types::flatten_2d_index;

use super::runner::{OpticsFrameData, OpticsWaveRunner};
use super::tool::OpticsWaveTool;

impl OpticsWaveTool {
    fn resolve_export_frame(&self) -> OpticsFrameData {
        if let Some(ref frame) = self.latest_frame {
            if frame.field.len() == self.grid_size * self.grid_size {
                return frame.clone();
            }
        }
        if let Some(snap) = self.worker.latest_snapshot() {
            if let Some(frame) = snap
                .structured_data
                .as_ref()
                .and_then(|d| d.downcast_ref::<OpticsFrameData>())
                .filter(|f| f.field.len() == self.grid_size * self.grid_size)
            {
                return frame.clone();
            }
        }
        let mut preview = OpticsWaveRunner::new(Arc::clone(&self.params), self.grid_size);
        preview.step();
        preview
            .get_snapshot()
            .structured_data
            .and_then(|b| b.downcast::<OpticsFrameData>().ok().map(|d| *d))
            .expect("OpticsWaveRunner::get_snapshot always attaches OpticsFrameData")
    }

    fn build_detector_table(&self, frame: &OpticsFrameData) -> ExportTable {
        let (preset_label, frequency, detector_x) = {
            let p = self.params.read().unwrap_or_else(|e| e.into_inner());
            (p.preset.label().to_string(), p.frequency, p.detector_x)
        };
        let r = &frame.readout;
        ExportTable::new(
            "Detector Readout",
            vec![
                "preset".to_string(),
                "frequency".to_string(),
                "detector_x".to_string(),
                "peak_irradiance".to_string(),
                "mean_irradiance".to_string(),
                "fringe_visibility".to_string(),
                "lambertian_intensity".to_string(),
                "lws_range_resolution".to_string(),
            ],
            vec![vec![
                ExportCell::Text(preset_label),
                ExportCell::Float(frequency),
                ExportCell::Float(detector_x),
                ExportCell::Float(r.peak_irradiance),
                ExportCell::Float(r.mean_irradiance),
                ExportCell::Float(r.fringe_visibility),
                ExportCell::Float(r.lambertian_intensity),
                ExportCell::Float(r.lws_range_resolution),
            ]],
        )
    }

    pub(super) fn build_optics_export_tables(&self) -> Vec<ExportTable> {
        let frame = self.resolve_export_frame();
        let n = self.grid_size.max(1);
        let scale = 1.0 / (n as f64);
        let readout_table = self.build_detector_table(&frame);
        let grid_table = ExportTable::from_bounded_grid_2d(
            "Wave Field",
            n,
            n,
            vec![
                "x".to_string(),
                "y".to_string(),
                "x_norm".to_string(),
                "y_norm".to_string(),
                "field".to_string(),
                "irradiance".to_string(),
                "refractive_index".to_string(),
                "obstacle".to_string(),
            ],
            |x, y| {
                let idx = flatten_2d_index(x, y, n);
                vec![
                    ExportCell::Int(x as i64),
                    ExportCell::Int(y as i64),
                    ExportCell::Float(x as f64 * scale),
                    ExportCell::Float(y as f64 * scale),
                    ExportCell::Float(frame.field.get(idx).copied().unwrap_or(0.0)),
                    ExportCell::Float(frame.irradiance.get(idx).copied().unwrap_or(0.0)),
                    ExportCell::Float(frame.refractive_index.get(idx).copied().unwrap_or(1.0)),
                    ExportCell::Bool(frame.obstacle.get(idx).copied().unwrap_or(0) != 0),
                ]
            },
        );
        vec![readout_table, grid_table]
    }
}
