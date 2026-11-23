#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Marching-squares vector isoline extraction from `StateSnapshot::structured_data` for [`super::MorphogenesisTool`].

use super::runner::MorphoSnapshotData;
use super::MorphogenesisTool;
use crate::async_sim::{SimulationRunner, StateSnapshot};
use eframe::egui;
use egui_plot::export::ExportOptions;
use egui_plot::export_primitives::{export_primitives_svg, has_drawable_primitives};
use egui_plot::ExportPrimitive;
use math_commons::contours::{
    extract_multi_contours, order_paths_nearest_neighbor, GridBounds2D,
};

const CONTOUR_QUANTILES: [f64; 5] = [0.2, 0.35, 0.5, 0.65, 0.8];

pub(super) fn snapshot_u_slice(snap: &StateSnapshot) -> Option<&[f64]> {
    let expected = snap.width.saturating_mul(snap.height);
    snap.structured_data
        .as_ref()
        .and_then(|d| {
            d.downcast_ref::<MorphoSnapshotData>()
                .map(|m| m.u_data.as_slice())
                .or_else(|| d.downcast_ref::<Vec<f64>>().map(Vec::as_slice))
        })
        .filter(|u| u.len() == expected)
}

impl MorphogenesisTool {
    /// Resolves the latest `f64` activator grid `(width, height, u_values)` from stepped state,
    /// `StateSnapshot::structured_data`, cached UI frame data, or a cold runner snapshot.
    pub(crate) fn resolve_structured_u_grid(&self) -> Option<(usize, usize, Vec<f64>)> {
        if let Some((w, h, ref u)) = self.stepped_u_grid {
            if w >= 2 && h >= 2 && u.len() == w * h {
                return Some((w, h, u.clone()));
            }
        }
        if let Some(snapshot) = self.worker.latest_snapshot() {
            if snapshot.width >= 2 && snapshot.height >= 2 {
                if let Some(u_slice) = snapshot_u_slice(snapshot) {
                    return Some((snapshot.width, snapshot.height, u_slice.to_vec()));
                }
            }
        }
        if !self.dirty
            && self.width >= 2
            && self.height >= 2
            && self.cached_u_grid.len() == self.width * self.height
        {
            return Some((self.width, self.height, self.cached_u_grid.clone()));
        }
        if self.width >= 2 && self.height >= 2 {
            let runner = Self::create_kinetics_runner(
                &self.params,
                &self.kinetics_state,
                &self.mask_rle,
                self.width,
                self.height,
            );
            let snap = runner.get_snapshot();
            if let Some(u_slice) = snapshot_u_slice(&snap) {
                return Some((snap.width, snap.height, u_slice.to_vec()));
            }
        }
        None
    }

    /// Extracts multi-level marching-squares isolines from the `f64` activator concentration grid.
    pub(crate) fn extract_isoline_primitives(&self) -> Vec<ExportPrimitive> {
        let Some((w, h, u_grid)) = self.resolve_structured_u_grid() else {
            return Vec::new();
        };
        let (min_u, max_u) = u_grid
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
                if v.is_finite() {
                    (lo.min(v), hi.max(v))
                } else {
                    (lo, hi)
                }
            });
        if !min_u.is_finite() || !max_u.is_finite() || (max_u - min_u) <= 1e-9 {
            return Vec::new();
        }
        let Ok(bounds) = GridBounds2D::new(0.0, w as f64, 0.0, h as f64) else {
            return Vec::new();
        };
        let levels: Vec<f64> = CONTOUR_QUANTILES
            .iter()
            .map(|&q| min_u + q * (max_u - min_u))
            .collect();
        let Ok(contour_sets) = extract_multi_contours(&u_grid, w, h, &bounds, &levels) else {
            return Vec::new();
        };

        let palette = crate::app::SemanticPalette::from_visuals(&egui::Visuals::light());
        let mut primitives = Vec::new();
        for (lvl_idx, (_, raw_paths)) in contour_sets.into_iter().enumerate() {
            let color = palette.series_color(lvl_idx);
            let level_val = levels[lvl_idx];
            for path in order_paths_nearest_neighbor(&raw_paths) {
                let pts = path.polyline_points();
                if pts.len() >= 2 {
                    primitives.push(ExportPrimitive::Polyline {
                        name: format!("Isoline u={level_val:.3}"),
                        points: pts,
                        color,
                        stroke_width: 1.5,
                        closed: path.closed,
                        fill: None,
                    });
                }
            }
        }
        primitives
    }

    pub(crate) fn export_isoline_svg(&self, options: &ExportOptions) -> Option<Result<String, String>> {
        let prims = self.extract_isoline_primitives();
        has_drawable_primitives(&prims).then(|| export_primitives_svg(&prims, options))
    }
}

#[cfg(test)]
mod tests {
    #![cfg_attr(any(), verified(opt_out = "tests"))]

    use super::super::MorphogenesisTool;
    use crate::framework::InteractiveTool;
    use egui_plot::export::ExportOptions;
    use egui_plot::export_primitives::{export_primitives_svg, ExportPrimitive};

    #[test]
    fn test_morphogenesis_export_plot_primitives_emits_vector_isolines_without_raster_image() {
        let mut tool = MorphogenesisTool::default();
        // Use stable stripe parameters and step to form smooth concentration gradients
        tool.apply_named_preset(0.14, 0.86, 1.0, 50.0, 0.01);
        tool.step_simulation();

        let prims = tool.export_plot_primitives();
        assert!(
            !prims.is_empty(),
            "MorphogenesisTool must emit marching-squares isoline primitives"
        );
        assert!(
            prims
                .iter()
                .all(|p| matches!(p, ExportPrimitive::Polyline { .. })),
            "MorphogenesisTool isolines must be pure Polyline primitives"
        );

        let svg = export_primitives_svg(&prims, &ExportOptions::default()).unwrap();
        assert!(svg.contains("<polyline") || svg.contains("<polygon"));
        assert!(!svg.contains("<image"));
    }
}
