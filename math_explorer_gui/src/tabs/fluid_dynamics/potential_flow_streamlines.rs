#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Streamline rendering and offscreen vector export helpers for [`super::potential_flow::PotentialFlowTool`].

use super::potential_flow::PotentialFlowTool;
use eframe::egui;
use egui_plot::ExportPrimitive;
use math_commons::contours::order_paths_nearest_neighbor;

pub(crate) const fn default_show_streamlines() -> bool {
    true
}

pub(crate) const fn default_streamline_count() -> usize {
    14
}

impl PotentialFlowTool {
    /// Computes ordered RK4 + $\psi$-deduplicated streamlines for the active potential flow field.
    pub(crate) fn compute_streamline_paths(&self) -> Vec<(Vec<[f64; 2]>, bool)> {
        if !self.show_streamlines
            || !self.view_range.is_finite()
            || self.view_range <= 0.0
            || !(2..=64).contains(&self.streamline_count)
        {
            return Vec::new();
        }
        let traced = self
            .field
            .trace_streamlines(self.view_range, self.streamline_count);
        let ordered = order_paths_nearest_neighbor(&traced);
        ordered
            .into_iter()
            .filter(|p| p.points.len() >= 2)
            .map(|p| (p.polyline_points(), p.closed))
            .collect()
    }

    pub(crate) fn build_export_primitives(&self) -> Vec<ExportPrimitive> {
        let mut primitives = Vec::new();
        let palette = crate::app::SemanticPalette::from_visuals(&egui::Visuals::light());
        let streamline_color = palette.series_color(1);
        for (idx, (pts, closed)) in self.compute_streamline_paths().into_iter().enumerate() {
            primitives.push(ExportPrimitive::Polyline {
                name: format!("Streamline {}", idx + 1),
                points: pts,
                color: streamline_color,
                stroke_width: 1.8,
                closed,
                fill: None,
            });
        }
        let arrow_color = palette.series_color(0);
        let (origins, tips) = self.compute_arrow_field();
        for (origin, tip) in origins.into_iter().zip(tips) {
            primitives.push(ExportPrimitive::Arrow {
                name: "Vector Field".to_string(),
                origin,
                tip,
                tip_length: Some(8.0),
                color: arrow_color,
            });
        }
        primitives
    }

    pub(crate) fn build_export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let mut datasets = Vec::new();
        let (origins, tips) = self.compute_arrow_field();
        let mut arrow_pts = Vec::with_capacity(origins.len() * 2);
        for (o, t) in origins.into_iter().zip(tips) {
            arrow_pts.push(o);
            arrow_pts.push(t);
        }
        if !arrow_pts.is_empty() {
            datasets.push(("Vector Field".to_string(), arrow_pts));
        }
        for (idx, (pts, _)) in self.compute_streamline_paths().into_iter().enumerate() {
            datasets.push((format!("Streamline {}", idx + 1), pts));
        }
        datasets
    }
}

#[cfg(test)]
mod tests {
    #![cfg_attr(any(), verified(opt_out = "tests"))]

    use super::super::potential_flow::PotentialFlowTool;
    use crate::framework::InteractiveTool;
    use egui_plot::ExportPrimitive;

    #[test]
    fn test_potential_flow_exports_both_streamlines_and_velocity_arrows() {
        let tool = PotentialFlowTool::default();
        let prims = tool.export_plot_primitives();
        let poly_count = prims
            .iter()
            .filter(|p| matches!(p, ExportPrimitive::Polyline { .. }))
            .count();
        let arrow_count = prims
            .iter()
            .filter(|p| matches!(p, ExportPrimitive::Arrow { .. }))
            .count();
        assert!(poly_count >= 4, "expected RK4 streamline polylines");
        assert!(arrow_count >= 20, "expected velocity vector arrows");
    }
}
