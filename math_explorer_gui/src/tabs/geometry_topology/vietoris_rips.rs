#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::framework::{Camera3D, InteractiveTool};
use eframe::egui;
use egui_plot::PlotPoint;
use math_explorer::pure_math::statistics::tda::{
    PersistenceInterval, Point2D, Point3D, SimplicialComplex, TdaFiltrationWorkspace,
    farthest_point_landmarks,
};
use rand::Rng;

#[path = "vietoris_rips/chaos_bridge.rs"]
pub mod chaos_bridge;
#[path = "vietoris_rips/filtration.rs"]
mod filtration;
#[path = "vietoris_rips/handoff.rs"]
pub mod handoff;
#[path = "vietoris_rips/presets.rs"]
mod presets;
#[path = "vietoris_rips/sculpt.rs"]
mod sculpt;
#[path = "vietoris_rips/state.rs"]
pub mod state;
#[path = "vietoris_rips/ui.rs"]
mod ui;

#[cfg(test)]
#[path = "vietoris_rips/tests.rs"]
mod tests;
#[cfg(all(test, target_arch = "wasm32"))]
#[path = "vietoris_rips/wasm_tests.rs"]
mod wasm_tests;

/// Maximum landmark count supported in 2D and 3D Vietoris-Rips filtrations.
pub const MAX_VIETORIS_RIPS_POINTS: usize = 64;
/// Default vertex count for legacy 2D random initialization.
pub const DEFAULT_2D_POINTS: usize = 20;
/// Number of radii sampled per persistence barcode sweep (`64 choose 3 = 41,664` max checks).
pub const FILTRATION_RADII_STEPS: usize = 24;

/// Interactive 2D/3D Vietoris-Rips persistent homology, Takens embedding, and chaos mashup tool.
pub struct VietorisRipsTool {
    pub(crate) mode: state::VrSourceMode,
    pub(crate) brush: state::SculptBrush,
    pub(crate) brush_radius: f64,
    pub(crate) active_drag_vertex: Option<usize>,
    pub(crate) radius: f64,
    pub(crate) points: Vec<Point2D>,
    pub(crate) points_3d: Vec<Point3D>,
    pub(crate) view_center_3d: [f64; 3],
    pub(crate) camera: Camera3D,
    pub(crate) chaos: chaos_bridge::VrChaosState,
    pub(crate) complex: SimplicialComplex,
    pub(crate) workspace: TdaFiltrationWorkspace,
    pub(crate) filtration_radii: [f64; FILTRATION_RADII_STEPS],
    pub(crate) barcode_intervals: Vec<PersistenceInterval>,
    pub(crate) beta_0: usize,
    pub(crate) beta_1: usize,
    pub(crate) error_msg: Option<String>,
    pub(crate) cached_triangles: Vec<[PlotPoint; 3]>,
    pub(crate) cached_edges: Vec<[PlotPoint; 2]>,
    pub(crate) cached_vertices: Vec<PlotPoint>,
    pub(crate) cached_triangles_3d: Vec<[Point3D; 3]>,
    pub(crate) cached_edges_3d: Vec<[Point3D; 2]>,
    pub(crate) revision: u64,
    #[cfg(test)]
    pub(crate) complex_builds: usize,
}

impl Default for VietorisRipsTool {
    fn default() -> Self {
        let mut tool = Self {
            mode: state::VrSourceMode::Points2D,
            brush: state::SculptBrush::OrbitOrPan,
            brush_radius: 0.8,
            active_drag_vertex: None,
            radius: 1.5,
            points: Vec::with_capacity(MAX_VIETORIS_RIPS_POINTS),
            points_3d: Vec::with_capacity(MAX_VIETORIS_RIPS_POINTS),
            view_center_3d: [0.0, 0.0, 25.0],
            camera: Camera3D::new(0.35, 0.55, 1.0),
            chaos: chaos_bridge::VrChaosState::default(),
            complex: SimplicialComplex::new(),
            workspace: TdaFiltrationWorkspace::with_capacity(MAX_VIETORIS_RIPS_POINTS),
            filtration_radii: [0.0; FILTRATION_RADII_STEPS],
            barcode_intervals: Vec::with_capacity(256),
            beta_0: 0,
            beta_1: 0,
            error_msg: None,
            cached_triangles: Vec::new(),
            cached_edges: Vec::new(),
            cached_vertices: Vec::new(),
            cached_triangles_3d: Vec::new(),
            cached_edges_3d: Vec::new(),
            revision: 1,
            #[cfg(test)]
            complex_builds: 0,
        };
        tool.generate_points(DEFAULT_2D_POINTS);
        tool
    }
}

impl VietorisRipsTool {
    pub(crate) fn generate_points(&mut self, count: usize) {
        let mut rng = oxidize_core::rng::OxidizeRng::default();
        let capped = count.clamp(1, MAX_VIETORIS_RIPS_POINTS);
        self.points.clear();
        for _ in 0..capped {
            self.points
                .push(Point2D::new(rng.gen_range(0.0..10.0), rng.gen_range(0.0..10.0)));
        }
        self.update_complex();
    }

    /// Ingests a one-click 3D chaotic landmark handoff from `AttractorPlotter`.
    pub fn ingest_chaos_handoff(
        &mut self,
        handoff: handoff::ChaosLandmarkHandoff,
    ) -> Result<(), String> {
        let subsampled = farthest_point_landmarks(&handoff.points, MAX_VIETORIS_RIPS_POINTS)
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
        let subsampled = farthest_point_landmarks(&handoff.points, MAX_VIETORIS_RIPS_POINTS)
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

    fn state_revision(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.revision.hash(&mut h);
        (self.mode as u8).hash(&mut h);
        self.radius.to_bits().hash(&mut h);
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
            && self.brush_radius.is_finite()
            && self.camera.is_valid()
            && self.chaos.has_finite_parameters()
            && self.points.iter().all(|p| p.x.is_finite() && p.y.is_finite())
            && self.points_3d.iter().all(Point3D::is_finite)
    }

    fn save_state(&self) -> Option<String> {
        self.serialize_tool_state()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.deserialize_tool_state(state)
    }

    fn step_simulation(&mut self) {
        if matches!(
            self.mode,
            state::VrSourceMode::Lorenz3D | state::VrSourceMode::Takens3D
        ) {
            self.chaos.advance_live_steps(5);
            let _ = self.rebuild_chaos_landmarks();
        }
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let pts = if self.mode.is_3d() {
            self.points_3d.iter().map(|&p| self.project_3d(p)).collect()
        } else {
            self.points.iter().map(|p| [p.x, p.y]).collect()
        };
        let bars = self
            .barcode_intervals
            .iter()
            .map(|iv| [iv.birth, iv.death])
            .collect();
        vec![
            ("Landmarks".to_string(), pts),
            ("PersistenceIntervals".to_string(), bars),
        ]
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        self.render_tool_ui(ui);
    }
}

// [cite:graph_parameters_rust]

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "VietorisRipsTool",
        display_name: "Vietoris-Rips Complex",
        domain: "geometry_topology",
        tags: &["tda", "homology", "persistence", "takens", "lorenz"],
        build: || Box::new(VietorisRipsTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for VietorisRipsTool {
    fn theory_description(&self) -> String {
        let count = if self.mode.is_3d() {
            self.points_3d.len()
        } else {
            self.points.len()
        };
        format!(
            "Topological Data Analysis (TDA) via Vietoris-Rips filtration over $\\mathbb{{Z}}_2$: $\\beta_0 = {}$, $\\beta_1 = \\dim(\\ker \\partial_1) - \\operatorname{{rank}}_{{\\mathbb{{Z}}_2}}(\\partial_2) = {}$ at $\\epsilon = {:.2}$ across {} landmarks.",
            self.beta_0, self.beta_1, self.radius, count
        )
    }

    fn phonetic_description(&self) -> String {
        "Vietoris-Rips persistent homology and Takens delay embedding theory".into()
    }

    fn theory_citation(&self) -> String {
        "Computational Topology: An Introduction (Edelsbrunner & Harer, 2010); Detecting Strange Attractors in Turbulence (Takens, 1981)".into()
    }

    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
}
