//! Client-side Vietoris-Rips simplicial filtration, 3D chaos homology, and TDA workbench.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

pub mod chaos_bridge;
#[cfg(test)]
mod chaos_tests;
pub mod export;
mod filtration;
pub mod handoff;
pub mod io;
pub mod plots;
pub mod presets;
mod sculpt;
pub mod state;
#[cfg(test)]
mod tests;
mod ui_3d;
#[cfg(all(test, target_arch = "wasm32"))]
mod wasm_tests;

pub use io::MAX_VIETORIS_RIPS_POINTS;

use crate::framework::{Camera3D, ExportCapabilities, InteractiveTool};
use crate::url_state::MAX_DECODED_URL_STATE_BYTES;
use eframe::egui;
use egui_plot::PlotPoint;
use io::{parse_point_cloud_text, ImportedCloud, PendingIoTasks, MAX_URL_CUSTOM_POINTS_2D};
use math_explorer::pure_math::statistics::tda::core::{NdPointCloud, Point2D};
use math_explorer::pure_math::statistics::tda::{
    farthest_point_landmarks, PersistenceInterval, Point3D, RipsPersistenceResult,
    SimplicialComplex, TdaFiltrationWorkspace,
};
use plots::PersistencePlotMode;
use presets::{generate_preset_points, GeneratorDescriptor, ShapePreset};
use state::{deserialize_tool_state, serialize_tool_state, SculptBrush, VrSourceMode};

/// Maximum 3D landmark budget (`64`) for chaotic attractors, Takens embeddings, and zero-alloc sweeps.
pub const MAX_CHAOS_LANDMARKS: usize = 64;

/// Number of radii sampled per zero-allocation persistence barcode sweep.
pub const FILTRATION_RADII_STEPS: usize = 24;

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

/// Interactive Vietoris-Rips & Persistent Homology workbench supporting 2D/N-D clouds up to 250 points
/// and 3D chaotic strange-attractor / Takens delay-embedding filtrations.
pub struct VietorisRipsTool {
    pub(crate) mode: VrSourceMode,
    pub(crate) brush: SculptBrush,
    pub(crate) brush_radius: f64,
    pub(crate) active_drag_vertex: Option<usize>,
    pub(super) radius: f64,
    pub(super) max_slider_radius: f64,
    pub(super) source: CloudSource,
    pub(super) plot_mode: PersistencePlotMode,
    pub(super) min_persistence: f64,
    pub(super) points: Vec<Point2D>,
    pub(crate) points_3d: Vec<Point3D>,
    pub(crate) view_center_3d: [f64; 3],
    pub(crate) camera: Camera3D,
    pub(crate) chaos: chaos_bridge::VrChaosState,
    pub(super) distance_matrix: Vec<Vec<f64>>,
    pub(super) persistence: RipsPersistenceResult,
    pub(super) complex: SimplicialComplex,
    pub(super) betti: (usize, usize),
    pub(super) complex_truncated: bool,
    pub(super) exceeds_url_bounds: bool,
    pub(crate) workspace: TdaFiltrationWorkspace,
    pub(crate) filtration_radii: [f64; FILTRATION_RADII_STEPS],
    pub(crate) barcode_intervals: Vec<PersistenceInterval>,
    pub(super) error_msg: Option<String>,
    pub(super) status_msg: Option<String>,
    pub(super) cached_triangles: Vec<[PlotPoint; 3]>,
    pub(super) cached_edges: Vec<[PlotPoint; 2]>,
    pub(super) cached_vertices: Vec<PlotPoint>,
    pub(crate) cached_triangles_3d: Vec<[Point3D; 3]>,
    pub(crate) cached_edges_3d: Vec<[Point3D; 2]>,
    pub(super) io_tasks: PendingIoTasks,
    pub(super) revision: u64,
    #[cfg(test)]
    pub(super) complex_builds: usize,
}

impl Default for VietorisRipsTool {
    fn default() -> Self {
        let desc = GeneratorDescriptor::default();
        let mut tool = Self {
            mode: VrSourceMode::Points2D,
            brush: SculptBrush::OrbitOrPan,
            brush_radius: 0.8,
            active_drag_vertex: None,
            radius: desc.preset.default_radius(),
            max_slider_radius: 15.0,
            source: CloudSource::Generator(desc),
            plot_mode: PersistencePlotMode::Diagram,
            min_persistence: 0.0,
            points: Vec::new(),
            points_3d: Vec::with_capacity(MAX_CHAOS_LANDMARKS),
            view_center_3d: [0.0, 0.0, 25.0],
            camera: Camera3D::new(0.35, 0.55, 1.0),
            chaos: chaos_bridge::VrChaosState::default(),
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
            workspace: TdaFiltrationWorkspace::with_capacity(MAX_CHAOS_LANDMARKS),
            filtration_radii: [0.0; FILTRATION_RADII_STEPS],
            barcode_intervals: Vec::with_capacity(256),
            error_msg: None,
            status_msg: None,
            cached_triangles: Vec::new(),
            cached_edges: Vec::new(),
            cached_vertices: Vec::new(),
            cached_triangles_3d: Vec::new(),
            cached_edges_3d: Vec::new(),
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
        if self.mode.is_3d() {
            return None;
        }
        match self.source {
            CloudSource::Generator(desc) => Some(desc),
            _ => None,
        }
    }

    #[must_use]
    pub fn ambient_dimension(&self) -> usize {
        if self.mode.is_3d() {
            return 3;
        }
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
        let radius = if self.mode.is_3d() {
            desc.preset.default_radius()
        } else {
            self.radius
        };
        let pts = generate_preset_points(&desc);
        if let Err(err) = self.rebuild_from_2d(CloudSource::Generator(desc), pts, radius) {
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

    /// Ingests a one-click 3D chaotic landmark handoff from `AttractorPlotter`.
    pub fn ingest_chaos_handoff(
        &mut self,
        handoff: handoff::ChaosLandmarkHandoff,
    ) -> Result<(), String> {
        let subsampled = farthest_point_landmarks(&handoff.points, MAX_CHAOS_LANDMARKS)
            .map_err(|e| format!("{e:?}"))?;
        self.chaos.sigma = handoff.sigma.clamp(0.0, 50.0);
        self.chaos.rho = handoff.rho.clamp(0.0, 100.0);
        self.chaos.beta = handoff.beta.clamp(0.0, 10.0);
        self.chaos.dt = handoff.dt.clamp(0.001, 0.05);
        self.load_validated_3d_points(subsampled, 7.5)
    }

    pub(crate) fn ingest_live_stream_landmarks(
        &mut self,
        handoff: handoff::ChaosLandmarkHandoff,
    ) -> Result<(), String> {
        let subsampled = farthest_point_landmarks(&handoff.points, MAX_CHAOS_LANDMARKS)
            .map_err(|e| format!("{e:?}"))?;
        self.chaos.sigma = handoff.sigma.clamp(0.0, 50.0);
        self.chaos.rho = handoff.rho.clamp(0.0, 100.0);
        self.chaos.beta = handoff.beta.clamp(0.0, 10.0);
        self.chaos.dt = handoff.dt.clamp(0.001, 0.05);
        let prev_mode = self.mode;
        self.load_validated_3d_points(subsampled, self.radius)?;
        self.mode = prev_mode;
        Ok(())
    }

    pub(super) fn bump_revision(&mut self) {
        self.revision = self.revision.wrapping_add(1).max(1);
    }

    pub(super) fn refresh_url_budget_flag(&mut self) {
        let over_cap = if self.mode.is_3d() {
            false
        } else {
            match &self.source {
                CloudSource::Generator(_) => false,
                CloudSource::Custom2D(pts) => pts.len() > MAX_URL_CUSTOM_POINTS_2D,
                CloudSource::CustomNd { cloud, .. } => cloud.size() * cloud.dimension() > 128,
            }
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
        (self.mode as u8).hash(&mut h);
        self.radius.to_bits().hash(&mut h);
        self.min_persistence.to_bits().hash(&mut h);
        self.plot_mode.hash(&mut h);
        self.points.len().hash(&mut h);
        self.points_3d.len().hash(&mut h);
        self.camera.pitch.to_bits().hash(&mut h);
        self.camera.yaw.to_bits().hash(&mut h);
        self.camera.zoom.to_bits().hash(&mut h);
        self.chaos.rho.to_bits().hash(&mut h);
        self.chaos.takens_lag.hash(&mut h);
        h.finish().max(1)
    }

    fn has_finite_parameters(&self) -> bool {
        self.radius.is_finite()
            && self.min_persistence.is_finite()
            && self.brush_radius.is_finite()
            && self.camera.is_valid()
            && self.chaos.has_finite_parameters()
            && self
                .points
                .iter()
                .all(|p| p.x.is_finite() && p.y.is_finite())
            && self.points_3d.iter().all(Point3D::is_finite)
    }

    fn save_state(&self) -> Option<String> {
        serialize_tool_state(self)
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
        deserialize_tool_state(self, state)
    }

    fn step_simulation(&mut self) {
        if matches!(self.mode, VrSourceMode::Lorenz3D | VrSourceMode::Takens3D) {
            self.chaos.advance_live_steps(5);
            let _ = self.rebuild_chaos_landmarks();
        }
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
        tags: &["tda", "persistent homology", "barcode", "betti", "takens", "lorenz"],
        build: || Box::new(VietorisRipsTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for VietorisRipsTool {
    fn theory_description(&self) -> String {
        let (b0, b1) = self.betti;
        let count = if self.mode.is_3d() {
            self.points_3d.len()
        } else {
            self.points.len()
        };
        format!(
            "Topological Data Analysis (TDA) via Vietoris-Rips simplicial filtration and exact $\\mathbb{{Z}}_2$ persistent homology: $\\sigma = [v_0, \\dots, v_k]$ enters at $\\max_{{i,j}} d(v_i, v_j) \\le \\epsilon$ ($\\epsilon = {:.2}$, $\\beta_0 = {b0}$, $\\beta_1 = {b1}$, {count} points in $\\mathbb{{R}}^{}$).",
            self.radius,
            self.ambient_dimension()
        )
    }

    fn phonetic_description(&self) -> String {
        "Vietoris-Rips simplicial filtration, Z2 persistent homology, and Takens delay embedding theory".into()
    }

    fn theory_citation(&self) -> String {
        "Computational Topology: An Introduction (Edelsbrunner & Harer, 2010); Detecting Strange Attractors in Turbulence (Takens, 1981)".into()
    }

    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
}

// [cite:graph_parameters_rust]
