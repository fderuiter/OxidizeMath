//! Client-side Vietoris-Rips simplicial filtration and persistent homology workbench.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

pub mod export;
pub mod io;
pub mod plots;
pub mod presets;
#[cfg(test)]
mod tests;

use crate::framework::{ExportCapabilities, InteractiveTool};
use crate::url_state::MAX_DECODED_URL_STATE_BYTES;
use eframe::egui;
use egui_plot::PlotPoint;
use io::{parse_point_cloud_text, ImportedCloud, PendingIoTasks, MAX_URL_CUSTOM_POINTS_2D};
use math_explorer::pure_math::statistics::tda::core::{NdPointCloud, Point2D, PointCloud};
use math_explorer::pure_math::statistics::tda::{
    compute_rips_persistence_from_distances, vietoris_rips_from_distance_matrix,
    RipsPersistenceResult, SimplicialComplex, MAX_FILTRATION_TRIANGLES,
};
use plots::PersistencePlotMode;
use presets::{
    deserialize_tool_state, generate_preset_points, serialize_tool_state, GeneratorDescriptor,
    ShapePreset,
};

#[derive(Clone, Debug)]
pub(super) enum CloudSource {
    Generator(GeneratorDescriptor),
    Custom2D(Vec<Point2D>),
    CustomNd {
        cloud: NdPointCloud,
        axis_x: usize,
        axis_y: usize,
    },
}

/// Interactive Vietoris-Rips & Persistent Homology workbench supporting 2D and N-D clouds up to 250 points.
pub struct VietorisRipsTool {
    pub(super) radius: f64,
    pub(super) max_slider_radius: f64,
    pub(super) source: CloudSource,
    pub(super) plot_mode: PersistencePlotMode,
    pub(super) min_persistence: f64,
    pub(super) points: Vec<Point2D>,
    pub(super) distance_matrix: Vec<Vec<f64>>,
    pub(super) persistence: RipsPersistenceResult,
    pub(super) complex: SimplicialComplex,
    pub(super) betti: (usize, usize),
    pub(super) complex_truncated: bool,
    pub(super) exceeds_url_bounds: bool,
    pub(super) error_msg: Option<String>,
    pub(super) status_msg: Option<String>,
    pub(super) cached_triangles: Vec<[PlotPoint; 3]>,
    pub(super) cached_edges: Vec<[PlotPoint; 2]>,
    pub(super) cached_vertices: Vec<PlotPoint>,
    pub(super) io_tasks: PendingIoTasks,
    revision: u64,
    #[cfg(test)]
    pub(super) complex_builds: usize,
}

impl Default for VietorisRipsTool {
    fn default() -> Self {
        let desc = GeneratorDescriptor::default();
        let mut tool = Self {
            radius: desc.preset.default_radius(),
            max_slider_radius: 15.0,
            source: CloudSource::Generator(desc),
            plot_mode: PersistencePlotMode::Diagram,
            min_persistence: 0.0,
            points: Vec::new(),
            distance_matrix: Vec::new(),
            persistence: RipsPersistenceResult {
                intervals: Vec::new(),
                max_radius: 0.0,
                truncated_triangles: false,
            },
            complex: SimplicialComplex::new(),
            betti: (0, 0),
            complex_truncated: false,
            exceeds_url_bounds: false,
            error_msg: None,
            status_msg: None,
            cached_triangles: Vec::new(),
            cached_edges: Vec::new(),
            cached_vertices: Vec::new(),
            io_tasks: PendingIoTasks::default(),
            revision: 1,
            #[cfg(test)]
            complex_builds: 0,
        };
        let pts = generate_preset_points(&desc);
        let _ = tool.rebuild_from_2d(CloudSource::Generator(desc), pts, tool.radius);
        tool
    }
}

impl VietorisRipsTool {
    #[must_use]
    pub fn is_triangle_truncated(&self) -> bool {
        self.persistence.truncated_triangles || self.complex_truncated
    }

    #[must_use]
    pub fn active_generator(&self) -> Option<GeneratorDescriptor> {
        match self.source {
            CloudSource::Generator(desc) => Some(desc),
            _ => None,
        }
    }

    #[must_use]
    pub fn ambient_dimension(&self) -> usize {
        match &self.source {
            CloudSource::CustomNd { cloud, .. } => cloud.dimension(),
            _ => 2,
        }
    }

    #[must_use]
    pub fn projection_axes(&self) -> (usize, usize) {
        match &self.source {
            CloudSource::CustomNd { axis_x, axis_y, .. } => (*axis_x, *axis_y),
            _ => (0, 1),
        }
    }

    pub fn select_shape_preset(&mut self, preset: ShapePreset) {
        let mut desc = self.active_generator().unwrap_or_default();
        desc.preset = preset;
        let radius = preset.default_radius();
        let pts = generate_preset_points(&desc);
        if let Err(err) = self.rebuild_from_2d(CloudSource::Generator(desc), pts, radius) {
            self.error_msg = Some(err);
        } else {
            self.status_msg = None;
        }
    }

    pub fn apply_generator(&mut self, desc: GeneratorDescriptor) {
        let pts = generate_preset_points(&desc);
        if let Err(err) = self.rebuild_from_2d(CloudSource::Generator(desc), pts, self.radius) {
            self.error_msg = Some(err);
        }
    }

    pub fn set_radius(&mut self, radius: f64) {
        self.radius = radius;
        self.update_complex();
    }

    pub fn set_plot_mode(&mut self, mode: PersistencePlotMode) {
        if self.plot_mode != mode {
            self.plot_mode = mode;
            self.bump_revision();
        }
    }

    pub fn set_min_persistence(&mut self, min_p: f64) {
        if min_p.is_finite() && min_p >= 0.0 {
            self.min_persistence = min_p;
            self.bump_revision();
        }
    }

    pub fn set_projection_axes(&mut self, axis_x: usize, axis_y: usize) {
        let CloudSource::CustomNd { cloud, .. } = &self.source else {
            return;
        };
        match cloud.project_2d(axis_x, axis_y) {
            Ok(projected) => {
                self.points = projected.points;
                if let CloudSource::CustomNd {
                    axis_x: ax,
                    axis_y: ay,
                    ..
                } = &mut self.source
                {
                    *ax = axis_x;
                    *ay = axis_y;
                }
                self.refresh_plot_geometry();
                self.bump_revision();
            }
            Err(e) => self.error_msg = Some(format!("Projection error: {e}")),
        }
    }

    /// Parses and applies an imported `.csv`, `.tsv`, or `.json` file without mutating state on error.
    pub fn import_point_cloud_text(&mut self, filename: &str, content: &str) -> Result<(), String> {
        let imported = parse_point_cloud_text(filename, content)?;
        match imported {
            ImportedCloud::Points2D(pts) => {
                let count = pts.len();
                self.rebuild_from_2d(CloudSource::Custom2D(pts), Vec::new(), self.radius)?;
                self.radius =
                    (self.persistence.max_radius * 0.25).clamp(0.05, self.max_slider_radius);
                self.update_complex();
                self.status_msg = Some(format!("Imported {count} 2D points from {filename}"));
            }
            ImportedCloud::PointsNd(nd) => {
                let (count, dim) = (nd.size(), nd.dimension());
                self.rebuild_from_nd(nd, 0, 1, self.radius)?;
                self.radius =
                    (self.persistence.max_radius * 0.25).clamp(0.05, self.max_slider_radius);
                self.update_complex();
                self.status_msg = Some(format!(
                    "Imported {count} points in ℝ^{dim} from {filename}"
                ));
            }
        }
        Ok(())
    }

    fn bump_revision(&mut self) {
        self.revision = self.revision.wrapping_add(1).max(1);
    }

    fn clear_cache(&mut self) {
        self.cached_triangles.clear();
        self.cached_edges.clear();
        self.cached_vertices.clear();
    }

    pub(super) fn rebuild_from_2d(
        &mut self,
        source: CloudSource,
        explicit_pts: Vec<Point2D>,
        radius: f64,
    ) -> Result<(), String> {
        let pts = match &source {
            CloudSource::Custom2D(custom) => custom.clone(),
            _ => explicit_pts,
        };
        let cloud = PointCloud::new(pts.clone()).map_err(|e| format!("{e}"))?;
        let dist = cloud.distance_matrix();
        let persistence =
            compute_rips_persistence_from_distances(&dist).map_err(|e| format!("{e}"))?;
        let (complex, truncated) = self.build_complex_slice(&dist, radius)?;

        self.source = source;
        self.points = pts;
        self.distance_matrix = dist;
        self.max_slider_radius = (persistence.max_radius * 1.05).max(15.0).max(radius);
        self.persistence = persistence;
        self.radius = radius;
        self.apply_complex(complex, truncated);
        self.refresh_url_budget_flag();
        self.bump_revision();
        Ok(())
    }

    pub(super) fn rebuild_from_nd(
        &mut self,
        cloud: NdPointCloud,
        axis_x: usize,
        axis_y: usize,
        radius: f64,
    ) -> Result<(), String> {
        let projected = cloud
            .project_2d(axis_x, axis_y)
            .map_err(|e| format!("{e}"))?;
        let dist = cloud.distance_matrix();
        let persistence =
            compute_rips_persistence_from_distances(&dist).map_err(|e| format!("{e}"))?;
        let (complex, truncated) = self.build_complex_slice(&dist, radius)?;

        self.source = CloudSource::CustomNd {
            cloud,
            axis_x,
            axis_y,
        };
        self.points = projected.points;
        self.distance_matrix = dist;
        self.max_slider_radius = (persistence.max_radius * 1.05).max(15.0).max(radius);
        self.persistence = persistence;
        self.radius = radius;
        self.apply_complex(complex, truncated);
        self.refresh_url_budget_flag();
        self.bump_revision();
        Ok(())
    }

    fn build_complex_slice(
        &mut self,
        dist: &[Vec<f64>],
        radius: f64,
    ) -> Result<(SimplicialComplex, bool), String> {
        #[cfg(test)]
        {
            self.complex_builds += 1;
        }
        vietoris_rips_from_distance_matrix(dist, radius, MAX_FILTRATION_TRIANGLES)
            .map_err(|e| e.to_string())
    }

    fn apply_complex(&mut self, complex: SimplicialComplex, truncated: bool) {
        self.complex = complex;
        self.complex_truncated = truncated;
        self.betti = self.persistence.betti_numbers_at(self.radius);
        self.refresh_plot_geometry();
        self.error_msg = None;
    }

    fn refresh_plot_geometry(&mut self) {
        self.cached_triangles = self
            .complex
            .get_simplices(2)
            .into_iter()
            .filter_map(|t| {
                let (p0, p1, p2) = (
                    self.points.get(t.vertices[0])?,
                    self.points.get(t.vertices[1])?,
                    self.points.get(t.vertices[2])?,
                );
                Some([
                    PlotPoint::new(p0.x, p0.y),
                    PlotPoint::new(p1.x, p1.y),
                    PlotPoint::new(p2.x, p2.y),
                ])
            })
            .collect();
        self.cached_edges = self
            .complex
            .get_simplices(1)
            .into_iter()
            .filter_map(|e| {
                let (p0, p1) = (
                    self.points.get(e.vertices[0])?,
                    self.points.get(e.vertices[1])?,
                );
                Some([PlotPoint::new(p0.x, p0.y), PlotPoint::new(p1.x, p1.y)])
            })
            .collect();
        self.cached_vertices = self
            .points
            .iter()
            .map(|p| PlotPoint::new(p.x, p.y))
            .collect();
    }

    fn update_complex(&mut self) {
        self.bump_revision();
        if self.points.is_empty() || self.distance_matrix.len() != self.points.len() {
            self.clear_cache();
            self.error_msg = Some("Error building point cloud: empty point cloud".to_string());
            return;
        }
        let dist = self.distance_matrix.clone();
        match self.build_complex_slice(&dist, self.radius) {
            Ok((complex, truncated)) => self.apply_complex(complex, truncated),
            Err(e) => {
                self.clear_cache();
                self.error_msg = Some(format!("Error building complex: {e}"));
            }
        }
    }

    pub(super) fn refresh_url_budget_flag(&mut self) {
        let over_cap = match &self.source {
            CloudSource::Generator(_) => false,
            CloudSource::Custom2D(pts) => pts.len() > MAX_URL_CUSTOM_POINTS_2D,
            CloudSource::CustomNd { cloud, .. } => cloud.size() * cloud.dimension() > 128,
        };
        let over_bytes = self
            .save_state()
            .is_some_and(|s| s.len() + 220 > MAX_DECODED_URL_STATE_BYTES);
        self.exceeds_url_bounds = over_cap || over_bytes;
    }
}

impl InteractiveTool for VietorisRipsTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }

    fn name(&self) -> &'static str {
        "Vietoris-Rips Complex"
    }

    fn domain(&self) -> &'static str {
        "geometry_topology"
    }

    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        export::build_export_datasets(self)
    }

    fn export_plot_primitives(&self) -> Vec<egui_plot::ExportPrimitive> {
        export::build_export_primitives(self)
    }

    fn state_revision(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.revision.hash(&mut h);
        self.radius.to_bits().hash(&mut h);
        self.min_persistence.to_bits().hash(&mut h);
        self.plot_mode.hash(&mut h);
        self.points.len().hash(&mut h);
        h.finish().max(1)
    }

    fn has_finite_parameters(&self) -> bool {
        self.radius.is_finite()
            && self.min_persistence.is_finite()
            && self
                .points
                .iter()
                .all(|p| p.x.is_finite() && p.y.is_finite())
    }

    fn save_state(&self) -> Option<String> {
        serialize_tool_state(self)
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
        deserialize_tool_state(self, state)
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        plots::render_workbench(self, ui);
    }
}

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "VietorisRipsTool",
        display_name: "Vietoris-Rips Complex",
        domain: "geometry_topology",
        tags: &["tda", "persistent homology", "barcode", "betti"],
        build: || Box::new(VietorisRipsTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for VietorisRipsTool {
    fn theory_description(&self) -> String {
        let (b0, b1) = self.betti;
        format!(
            "Topological Data Analysis (TDA) via Vietoris-Rips simplicial filtration and exact $\\mathbb{{Z}}_2$ persistent homology: $\\sigma = [v_0, \\dots, v_k]$ enters at $\\max_{{i,j}} d(v_i, v_j) \\le \\epsilon$ ($\\epsilon = {:.2}$, $\\beta_0 = {b0}$, $\\beta_1 = {b1}$, {} points in $\\mathbb{{R}}^{}$).",
            self.radius,
            self.points.len(),
            self.ambient_dimension()
        )
    }

    fn phonetic_description(&self) -> String {
        "Vietoris-Rips simplicial filtration and Z2 persistent homology theory".into()
    }

    fn theory_citation(&self) -> String {
        "Computational Topology: An Introduction (Edelsbrunner & Harer, 2010)".into()
    }

    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
}

// [cite:graph_parameters_rust]
