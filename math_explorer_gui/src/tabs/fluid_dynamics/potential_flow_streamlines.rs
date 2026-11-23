#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Streamline rendering and offscreen vector export helpers for [`super::potential_flow::PotentialFlowTool`].

use super::potential_flow::PotentialFlowTool;
use eframe::egui;
use egui_plot::{ExportPrimitive, PlotPoint};
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
                points: pts.into(),
                color: streamline_color,
                stroke_width: 1.8,
                closed,
                fill: None,
            });
        }
        let arrow_color = palette.series_color(0);
        let mut origins: Vec<PlotPoint> = Vec::new();
        let mut tips: Vec<PlotPoint> = Vec::new();
        self.populate_arrow_buffers(&mut origins, &mut tips);
        for (origin, tip) in origins.into_iter().zip(tips) {
            primitives.push(ExportPrimitive::Arrow {
                name: "Vector Field".to_string(),
                origin: [origin.x, origin.y],
                tip: [tip.x, tip.y],
                tip_length: Some(8.0),
                color: arrow_color,
            });
        }
        primitives
    }

    pub(crate) fn build_export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let mut datasets = Vec::new();
        let mut origins: Vec<PlotPoint> = Vec::new();
        let mut tips: Vec<PlotPoint> = Vec::new();
        self.populate_arrow_buffers(&mut origins, &mut tips);
        let mut arrow_pts = Vec::with_capacity(origins.len() * 2);
        for (o, t) in origins.into_iter().zip(tips) {
            arrow_pts.push([o.x, o.y]);
            arrow_pts.push([t.x, t.y]);
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

impl scientific_metadata::theory::TheoryDescribable for PotentialFlowTool {
    fn theory_description(&self) -> String {
        "Two-dimensional incompressible irrotational potential flow governed by Laplace's equation $\\nabla^2 \\phi = 0$ for velocity potential $\\phi$, where fluid velocity is given by $\\vec{V} = \\nabla \\phi = (u, v)$ with linear superposition of elementary solutions.".into()
    }
    fn phonetic_description(&self) -> String {
        "Incompressible irrotational potential flow and Laplace equation theory".into()
    }
    fn theory_citation(&self) -> String {
        "Elementary Fluid Dynamics (Acheson, 1990); An Introduction to Fluid Dynamics (Batchelor, 1967)".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "Uniform".into(),
            "Uniform freestream flow phi = U (x cos(alpha) + y sin(alpha))".into(),
        );
        map.insert(
            "Source/Sink".into(),
            "Radial flow phi = (m / 2 pi) ln(r)".into(),
        );
        map.insert(
            "Vortex".into(),
            "Circulation flow phi = (Gamma / 2 pi) theta".into(),
        );
        map.insert(
            "Doublet".into(),
            "Source-sink dipole limit phi = (mu / 2 pi) cos(theta) / r".into(),
        );
        map
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "vector_field",
        title: "Vector Field",
        description: "ψ = U·y·(1 - R²/r²): Potential streamlines and 2D velocity vector field past a cylinder.",
        category: "Fluid Mechanics",
        icon: "🌊",
        domain: "fluid_dynamics",
        tool_name: Some("PotentialFlowTool"),
        preset_generator: || crate::starter_gallery::PRESET_VECTOR_FIELD,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "source_sink",
        title: "Source-Sink Pair",
        description: "ψ = (q/2π)·(θ₁ - θ₂): Superposition of equal source and sink singularities.",
        category: "Fluid Mechanics",
        icon: "🔄",
        domain: "fluid_dynamics",
        tool_name: Some("PotentialFlowTool"),
        preset_generator: || crate::starter_gallery::PRESET_SOURCE_SINK,
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
