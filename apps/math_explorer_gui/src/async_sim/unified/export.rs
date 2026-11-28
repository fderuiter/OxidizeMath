use eframe::egui::{Color32, ColorImage};
use math_commons::math_kernel::types::flatten_2d_index;

use crate::async_sim::StateSnapshot;
use crate::framework::{ExportCell, ExportTable};

use super::{UnifiedModel, UnifiedSimTool};

fn build_snapshot_metrics_table(snapshot: &StateSnapshot) -> Option<ExportTable> {
    #[cfg(feature = "physics")]
    if let Some(lbm) = snapshot
        .structured_data
        .as_ref()
        .and_then(|d| {
            d.downcast_ref::<crate::tabs::fluid_dynamics::lattice_boltzmann::LbmSnapshotMetrics>()
        })
        .filter(|_| snapshot.custom_data.len() >= 6)
    {
        return Some(ExportTable::new(
            "Hydrodynamic Metrics",
            vec![
                "f_drag".to_string(),
                "f_lift".to_string(),
                "drag_coeff".to_string(),
                "lift_coeff".to_string(),
                "max_vorticity".to_string(),
                "solid_cells".to_string(),
                "core_intact".to_string(),
                "step_count".to_string(),
            ],
            vec![vec![
                ExportCell::Float(snapshot.custom_data[0]),
                ExportCell::Float(snapshot.custom_data[1]),
                ExportCell::Float(snapshot.custom_data[2]),
                ExportCell::Float(lbm.lift_coeff),
                ExportCell::Float(lbm.max_vorticity),
                ExportCell::Int(snapshot.custom_data[3].round() as i64),
                ExportCell::Bool(snapshot.custom_data[4] >= 0.5),
                ExportCell::Int(snapshot.custom_data[5].round() as i64),
            ]],
        ));
    }

    if snapshot.custom_data.is_empty() {
        return None;
    }
    let headers = (0..snapshot.custom_data.len())
        .map(|i| format!("metric_{i}"))
        .collect();
    let row = snapshot
        .custom_data
        .iter()
        .map(|&v| ExportCell::Float(v))
        .collect();
    Some(ExportTable::new("Snapshot Metrics", headers, vec![row]))
}

fn build_snapshot_raster_table(snapshot: &StateSnapshot) -> Option<ExportTable> {
    if snapshot.width == 0 || snapshot.height == 0 {
        return None;
    }
    let pixels = snapshot.pixels.read().unwrap_or_else(|e| e.into_inner());
    if pixels.len() != snapshot.width * snapshot.height {
        return None;
    }
    Some(ExportTable::from_bounded_grid_2d(
        "Raster Field",
        snapshot.width,
        snapshot.height,
        vec![
            "x".to_string(),
            "y".to_string(),
            "r".to_string(),
            "g".to_string(),
            "b".to_string(),
            "luminance".to_string(),
        ],
        |x, y| {
            let c = pixels
                .get(flatten_2d_index(x, y, snapshot.width))
                .copied()
                .unwrap_or(Color32::BLACK);
            let lum =
                (0.2126 * f64::from(c.r()) + 0.7152 * f64::from(c.g()) + 0.0722 * f64::from(c.b()))
                    / 255.0;
            vec![
                ExportCell::Int(x as i64),
                ExportCell::Int(y as i64),
                ExportCell::Int(i64::from(c.r())),
                ExportCell::Int(i64::from(c.g())),
                ExportCell::Int(i64::from(c.b())),
                ExportCell::Float(lum),
            ]
        },
    ))
}

impl<M: UnifiedModel> UnifiedSimTool<M> {
    fn fallback_model_snapshot(&self) -> StateSnapshot {
        let p = self.params.read().unwrap_or_else(|e| e.into_inner());
        M::new(&p).get_snapshot()
    }

    pub(super) fn build_unified_raster_export(&self) -> Option<ColorImage> {
        let owned_fallback;
        let snapshot = if let Some(ref s) = self.stepped_snapshot {
            s
        } else if let Some(s) = self.controller.latest_snapshot() {
            s
        } else {
            owned_fallback = self.fallback_model_snapshot();
            &owned_fallback
        };
        let guard = snapshot.pixels.read().unwrap_or_else(|e| e.into_inner());
        if guard.is_empty() || snapshot.width == 0 || snapshot.height == 0 {
            return None;
        }
        Some(ColorImage::new(
            [snapshot.width, snapshot.height],
            guard.clone(),
        ))
    }

    pub(super) fn build_unified_export_tables(&self) -> Vec<ExportTable> {
        if !M::is_raster_model() {
            return Vec::new();
        }
        let owned_fallback;
        let snapshot = if let Some(ref s) = self.stepped_snapshot {
            s
        } else if let Some(s) = self.controller.latest_snapshot() {
            s
        } else {
            owned_fallback = self.fallback_model_snapshot();
            &owned_fallback
        };
        let mut tables = Vec::with_capacity(2);
        if let Some(metrics) = build_snapshot_metrics_table(snapshot) {
            tables.push(metrics);
        }
        if let Some(grid) = build_snapshot_raster_table(snapshot) {
            tables.push(grid);
        }
        tables
    }
}
