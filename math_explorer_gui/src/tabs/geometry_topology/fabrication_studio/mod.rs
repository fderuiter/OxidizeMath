#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! `FabricationStudioTool`: converts 2D Turing fields, potential-flow streamlines, and
//! aperiodic Penrose / Einstein monotile patches into millimeter `.svg` stroke paths for
//! laser cutters (`#layer-cut`) and pen plotters (`#layer-score`).

pub mod generators;
pub mod state;
mod ui;

#[cfg(test)]
mod tests;

pub use generators::{build_fabrication_artifact, FabricationArtifact};
pub use state::{FabricationSourceKind, FabricationStudioState, FabricationTilingSelector};

use crate::framework::{ExportCapabilities, InteractiveTool, StarterPresetMetadata, ToolMetadata};
use eframe::egui;
use egui_plot::{export_fabrication_svg, ExportPrimitive};
use scientific_metadata::theory::TheoryDescribable;
use std::collections::HashMap;
use std::hash::Hasher;
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Mutex};

/// Client-side millimeter SVG fabrication studio for laser cutters and pen plotters.
pub struct FabricationStudioTool {
    pub(crate) state: FabricationStudioState,
    pub(crate) cached_artifact: Mutex<Option<(u64, Arc<FabricationArtifact>)>>,
    pub(crate) last_error: Option<String>,
    pub(crate) dirty: bool,
    pub(crate) status_banner: Option<(String, bool)>,
    pub(crate) save_rx: Option<Receiver<Result<String, String>>>,
}

impl Default for FabricationStudioTool {
    fn default() -> Self {
        Self {
            state: FabricationStudioState::default(),
            cached_artifact: Mutex::new(None),
            last_error: None,
            dirty: true,
            status_banner: None,
            save_rx: None,
        }
    }
}

impl FabricationStudioTool {
    pub(crate) fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    pub(crate) fn artifact_or_build(&self) -> Result<Arc<FabricationArtifact>, String> {
        let rev = self.state_revision();
        let mut guard = self
            .cached_artifact
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if let Some((cached_rev, ref art)) = *guard {
            if cached_rev == rev {
                return Ok(Arc::clone(art));
            }
        }
        let built = Arc::new(build_fabrication_artifact(&self.state)?);
        *guard = Some((rev, Arc::clone(&built)));
        Ok(built)
    }

    pub(crate) fn ensure_artifact(&mut self) {
        match self.artifact_or_build() {
            Ok(_) => {
                self.last_error = None;
                self.dirty = false;
            }
            Err(err) => {
                *self
                    .cached_artifact
                    .lock()
                    .unwrap_or_else(|e| e.into_inner()) = None;
                self.last_error = Some(err);
                self.dirty = false;
            }
        }
    }

    /// Serializes the active geometry into a millimeter-accurate layered fabrication `.svg` string.
    pub fn serialize_fabrication_svg(&self) -> Result<String, String> {
        let artifact = self.artifact_or_build()?;
        export_fabrication_svg(
            &artifact.cut_paths_mm,
            &artifact.score_paths_mm,
            &self.state.svg_options(),
        )
    }

    /// Encodes a standalone `#tab=geometry_topology&tool=FabricationStudioTool&state=...` share link.
    pub fn encode_studio_share_url(&self) -> Result<String, String> {
        let state_json = self
            .save_state()
            .ok_or_else(|| "Failed to serialize FabricationStudioTool state".to_string())?;
        let config: serde_json::Value = serde_json::from_str(&state_json)
            .map_err(|e| format!("Invalid studio JSON state: {e}"))?;
        let envelope =
            crate::presets::serialize_compact_preset_envelope(self.name(), self.domain(), config)?;
        let fragment = crate::url_state::encode_share_fragment(
            "geometry_topology",
            Some("FabricationStudioTool"),
            Some(&envelope),
        )?;
        Ok(crate::url_state::build_share_url_from_fragment(&fragment))
    }

    pub(crate) fn default_svg_filename(&self) -> String {
        let slug = match self.state.source {
            FabricationSourceKind::TuringField => "turing_coaster",
            FabricationSourceKind::PotentialFlow => "potential_streamlines",
            FabricationSourceKind::AperiodicTiling => match self.state.tiling_kind {
                FabricationTilingSelector::PenroseP2 => "penrose_p2",
                FabricationTilingSelector::PenroseP3 => "penrose_p3",
                FabricationTilingSelector::EinsteinHat => "einstein_hat",
                FabricationTilingSelector::EinsteinSpectre => "einstein_spectre",
            },
        };
        format!("fabrication_{slug}.svg")
    }
}

impl InteractiveTool for FabricationStudioTool {
    fn name(&self) -> &'static str {
        "Fabrication Studio"
    }

    fn domain(&self) -> &'static str {
        "geometry_topology"
    }

    fn theory(&self) -> &dyn TheoryDescribable {
        self
    }

    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }

    fn export_custom_svg(
        &self,
        _options: &egui_plot::export::ExportOptions,
    ) -> Option<Result<String, String>> {
        Some(self.serialize_fabrication_svg())
    }

    fn export_plot_primitives(&self) -> Vec<ExportPrimitive> {
        let Ok(art) = self.artifact_or_build() else {
            return Vec::new();
        };
        let mut prims = Vec::with_capacity(art.score_paths_mm.len() + art.cut_paths_mm.len());
        let h = art.bed_height_mm;
        for (idx, poly) in art.score_paths_mm.iter().enumerate() {
            prims.push(ExportPrimitive::Polyline {
                name: format!("Score {}", idx + 1),
                points: poly.points.iter().map(|&[x, y]| [x, h - y]).collect(),
                color: egui::Color32::from_rgb(56, 189, 248),
                stroke_width: 1.5,
                closed: poly.closed,
                fill: None,
            });
        }
        for (idx, poly) in art.cut_paths_mm.iter().enumerate() {
            prims.push(ExportPrimitive::Polyline {
                name: format!("Cut {}", idx + 1),
                points: poly.points.iter().map(|&[x, y]| [x, h - y]).collect(),
                color: egui::Color32::from_rgb(244, 63, 94),
                stroke_width: 2.0,
                closed: poly.closed,
                fill: None,
            });
        }
        prims
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let Ok(art) = self.artifact_or_build() else {
            return Vec::new();
        };
        let h = art.bed_height_mm;
        let mut ds = Vec::new();
        for (idx, poly) in art.cut_paths_mm.iter().enumerate() {
            ds.push((
                format!("Cut Path {}", idx + 1),
                poly.points.iter().map(|&[x, y]| [x, h - y]).collect(),
            ));
        }
        for (idx, poly) in art.score_paths_mm.iter().enumerate() {
            ds.push((
                format!("Score Path {}", idx + 1),
                poly.points.iter().map(|&[x, y]| [x, h - y]).collect(),
            ));
        }
        ds
    }

    fn state_revision(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.state.hash_into(&mut h);
        h.finish().max(1)
    }

    fn has_finite_parameters(&self) -> bool {
        self.state.validate().is_ok()
    }

    fn clamp_parameters(&mut self) -> bool {
        self.state.clamp_in_place();
        self.mark_dirty();
        true
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&self.state).ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let parsed: FabricationStudioState = serde_json::from_str(state)
            .map_err(|e| format!("Invalid FabricationStudioTool state: {e}"))?;
        parsed.validate()?;
        self.state = parsed;
        self.mark_dirty();
        Ok(())
    }

    #[cfg(test)]
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    #[cfg(test)]
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        self.poll_async_io(&ctx);
        self.draw_side_panel(ui);
        self.draw_bed_preview(ui);
    }
}

impl TheoryDescribable for FabricationStudioTool {
    fn theory_description(&self) -> String {
        "Converts 2D scalar fields $u(x, y)$ via marching-squares linear edge interpolation and saddle disambiguation, incompressible stream functions $\\psi(x, y)$ via 4th-order Runge-Kutta integration, and aperiodic substitution tilings (`Penrose P2/P3`, Einstein `Hat` and chiral `Spectre`) into deduplicated, nearest-neighbor travel-ordered millimeter SVG `<path>` geometry.".to_string()
    }

    fn phonetic_description(&self) -> String {
        "Two dimensional marching squares contours, Runge Kutta streamlines, and aperiodic Einstein Hat and Penrose tilings for millimeter vector fabrication.".to_string()
    }

    fn theory_citation(&self) -> String {
        "Smith, Myers, Kaplan, & Goodman-Strauss (2023), An aperiodic monotile; Lorensen & Cline (1987), Marching Cubes; Penrose (1974), The role of aesthetics in pure and applied mathematical research.".to_string()
    }

    fn available_descriptions(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert(
            "Einstein Hat".to_string(),
            "13-edge Tile(1, sqrt(3)) polykite monotile covering the plane non-periodically without translational symmetry.".to_string(),
        );
        map.insert(
            "Marching Squares".to_string(),
            "Piecewise-linear 2D level-set extraction disambiguating saddle cells via bilinear center interpolation.".to_string(),
        );
        map
    }
}

/// Starter preset JSON for Einstein Hat aperiodic monotile patch on A4.
pub const PRESET_FABRICATION_EINSTEIN_HAT: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Einstein Hat Monotile Sheet",
    "domain": "geometry_topology",
    "timestamp": "2026-01-01T00:00:00Z"
  },
  "config": {
    "source": "AperiodicTiling",
    "bed_mode": "A4",
    "custom_width_mm": 200.0,
    "custom_height_mm": 200.0,
    "margin_mm": 12.0,
    "optimize_travel": true,
    "show_pen_travel": true,
    "turing_a": 0.1,
    "turing_b": 0.9,
    "turing_d_u": 1.0,
    "turing_d_v": 45.0,
    "turing_dt": 0.04,
    "turing_grid_size": 48,
    "turing_steps": 140,
    "turing_score_levels": 4,
    "turing_coaster_border": true,
    "flow_elements": [
      { "kind": "Uniform", "strength": 10.0, "x": 0.0, "y": 0.0, "angle_deg": 0.0 },
      { "kind": "Doublet", "strength": 251.32741228718345, "x": 0.0, "y": 0.0, "angle_deg": 0.0 }
    ],
    "flow_view_range": 10.0,
    "flow_streamline_count": 16,
    "tiling_kind": "EinsteinHat",
    "tiling_depth": 2
  }
}"#;

/// Starter preset JSON for Schnakenberg Turing laser-cut & scored coaster on A4.
pub const PRESET_FABRICATION_TURING_COASTER: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Turing Laser Coaster (A4)",
    "domain": "geometry_topology",
    "timestamp": "2026-01-01T00:00:00Z"
  },
  "config": {
    "source": "TuringField",
    "bed_mode": "A4",
    "custom_width_mm": 210.0,
    "custom_height_mm": 297.0,
    "margin_mm": 10.0,
    "optimize_travel": true,
    "show_pen_travel": false,
    "turing_a": 0.12,
    "turing_b": 0.88,
    "turing_d_u": 1.0,
    "turing_d_v": 40.0,
    "turing_dt": 0.04,
    "turing_grid_size": 48,
    "turing_steps": 150,
    "turing_score_levels": 4,
    "turing_coaster_border": true,
    "flow_elements": [
      { "kind": "Uniform", "strength": 10.0, "x": 0.0, "y": 0.0, "angle_deg": 0.0 }
    ],
    "flow_view_range": 10.0,
    "flow_streamline_count": 16,
    "tiling_kind": "EinsteinHat",
    "tiling_depth": 1
  }
}"#;

/// Starter preset JSON for AxiDraw potential-flow cylinder streamlines on US Letter.
pub const PRESET_FABRICATION_FLOW_PLOTTER: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Cylinder Streamline Plot (US Letter)",
    "domain": "geometry_topology",
    "timestamp": "2026-01-01T00:00:00Z"
  },
  "config": {
    "source": "PotentialFlow",
    "bed_mode": "UsLetter",
    "custom_width_mm": 215.9,
    "custom_height_mm": 279.4,
    "margin_mm": 15.0,
    "optimize_travel": true,
    "show_pen_travel": true,
    "turing_a": 0.1,
    "turing_b": 0.9,
    "turing_d_u": 1.0,
    "turing_d_v": 45.0,
    "turing_dt": 0.04,
    "turing_grid_size": 48,
    "turing_steps": 140,
    "turing_score_levels": 4,
    "turing_coaster_border": true,
    "flow_elements": [
      { "kind": "Uniform", "strength": 10.0, "x": 0.0, "y": 0.0, "angle_deg": 0.0 },
      { "kind": "Doublet", "strength": 251.32741228718345, "x": 0.0, "y": 0.0, "angle_deg": 0.0 }
    ],
    "flow_view_range": 10.0,
    "flow_streamline_count": 18,
    "tiling_kind": "PenroseP3",
    "tiling_depth": 2
  }
}"#;

inventory::submit! {
    ToolMetadata {
        id: "FabricationStudioTool",
        display_name: "Fabrication Studio",
        domain: "geometry_topology",
        tags: &[
            "svg",
            "laser",
            "plotter",
            "axidraw",
            "lightburn",
            "penrose",
            "einstein",
            "hat",
            "spectre",
            "turing",
            "contours",
        ],
        build: || Box::new(FabricationStudioTool::default()),
    }
}

inventory::submit! {
    StarterPresetMetadata {
        id: "fabrication_einstein_hat",
        title: "Einstein Hat Monotile",
        description: "Deduplicated Tile(1, √3) aperiodic polykite patch with separate cut perimeter and single-stroke interior score paths.",
        category: "Geometry & Fabrication",
        icon: "🧩",
        domain: "geometry_topology",
        tool_name: Some("FabricationStudioTool"),
        preset_generator: || PRESET_FABRICATION_EINSTEIN_HAT,
    }
}

inventory::submit! {
    StarterPresetMetadata {
        id: "fabrication_turing_coaster",
        title: "Turing Laser Coaster",
        description: "Schnakenberg reaction-diffusion marching-squares isolines inside a circular A4 laser-cut coaster perimeter.",
        category: "Geometry & Fabrication",
        icon: "🔥",
        domain: "geometry_topology",
        tool_name: Some("FabricationStudioTool"),
        preset_generator: || PRESET_FABRICATION_TURING_COASTER,
    }
}

inventory::submit! {
    StarterPresetMetadata {
        id: "fabrication_cylinder_streamlines",
        title: "Cylinder Streamline Plot",
        description: "Travel-ordered RK4 potential-flow streamlines and stagnation body contour formatted in millimeters for pen plotters.",
        category: "Geometry & Fabrication",
        icon: "✒️",
        domain: "geometry_topology",
        tool_name: Some("FabricationStudioTool"),
        preset_generator: || PRESET_FABRICATION_FLOW_PLOTTER,
    }
}

// [cite:graph_parameters_rust]
