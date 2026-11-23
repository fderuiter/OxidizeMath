#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Interactive 3D Knot, Artin Braid, Reidemeister Isotopy, and Laurent Skein Studio.

// [cite:knot_braid_topology]

mod model;
mod presets;
mod ui_controls;
mod viewport;

pub(crate) use model::{DiagramEditOp, StudioComputedModel, StudioFamily};

use crate::framework::{
    Camera3D, ExportCapabilities, InteractiveTool, Scene3DBuilder, ToolStateBounds,
};
use eframe::egui;
use math_explorer::pure_math::knot_theory::build_link_tubular_mesh;

/// Maximum serialized Reidemeister edit steps stored in shareable URL / JSON state.
pub const MAX_REIDEMEISTER_HISTORY: usize = 24;

const fn default_tube_radius() -> f64 {
    0.14
}
const fn default_braid_strands() -> usize {
    2
}
fn default_braid_generators() -> Vec<i32> {
    vec![1, 1, 1]
}
const fn default_torus_p() -> usize {
    2
}
const fn default_torus_q() -> i32 {
    3
}
const fn default_lissajous_freqs() -> [usize; 3] {
    [3, 4, 7]
}
const fn default_lissajous_phases() -> [f64; 3] {
    [0.1, 0.7, 0.2]
}
const fn default_major_radius() -> f64 {
    2.0
}
const fn default_minor_radius() -> f64 {
    0.65
}
const fn default_curve_samples() -> usize {
    72
}
const fn default_radial_segments() -> usize {
    12
}
const fn default_true() -> bool {
    true
}

#[derive(serde::Serialize)]
struct KnotBraidSaveState<'a> {
    tube_radius: f64,
    family: StudioFamily,
    braid_strands: usize,
    braid_generators: &'a [i32],
    torus_p: usize,
    torus_q: i32,
    lissajous_freqs: [usize; 3],
    lissajous_phases: [f64; 3],
    major_radius: f64,
    minor_radius: f64,
    curve_samples: usize,
    radial_segments: usize,
    show_bishop_frames: bool,
    show_tube_rings: bool,
    camera: Camera3D,
    #[serde(skip_serializing_if = "<[DiagramEditOp]>::is_empty")]
    reidemeister_ops: &'a [DiagramEditOp],
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct KnotBraidState {
    #[serde(default = "default_tube_radius")]
    tube_radius: f64,
    family: StudioFamily,
    #[serde(default = "default_braid_strands")]
    braid_strands: usize,
    #[serde(default = "default_braid_generators")]
    braid_generators: Vec<i32>,
    #[serde(default = "default_torus_p")]
    torus_p: usize,
    #[serde(default = "default_torus_q")]
    torus_q: i32,
    #[serde(default = "default_lissajous_freqs")]
    lissajous_freqs: [usize; 3],
    #[serde(default = "default_lissajous_phases")]
    lissajous_phases: [f64; 3],
    #[serde(default = "default_major_radius")]
    major_radius: f64,
    #[serde(default = "default_minor_radius")]
    minor_radius: f64,
    #[serde(default = "default_curve_samples")]
    curve_samples: usize,
    #[serde(default = "default_radial_segments")]
    radial_segments: usize,
    #[serde(default)]
    show_bishop_frames: bool,
    #[serde(default = "default_true")]
    show_tube_rings: bool,
    #[serde(default)]
    camera: Option<Camera3D>,
    #[serde(default)]
    pitch: Option<f32>,
    #[serde(default)]
    yaw: Option<f32>,
    #[serde(default)]
    zoom: Option<f32>,
    #[serde(default)]
    reidemeister_ops: Vec<DiagramEditOp>,
}

/// Interactive 3D Knot, Artin Braid, Reidemeister Isotopy, and Laurent Skein Studio.
pub struct KnotBraidStudioTool {
    pub(crate) tube_radius: f64,
    pub(crate) family: StudioFamily,
    pub(crate) braid_strands: usize,
    pub(crate) braid_generators: Vec<i32>,
    pub(crate) torus_p: usize,
    pub(crate) torus_q: i32,
    pub(crate) lissajous_freqs: [usize; 3],
    pub(crate) lissajous_phases: [f64; 3],
    pub(crate) major_radius: f64,
    pub(crate) minor_radius: f64,
    pub(crate) curve_samples: usize,
    pub(crate) radial_segments: usize,
    pub(crate) show_bishop_frames: bool,
    pub(crate) show_tube_rings: bool,
    pub(crate) reidemeister_ops: Vec<DiagramEditOp>,
    pub(crate) selected_crossing: usize,
    pub(crate) second_crossing: usize,
    pub(crate) selected_segment: usize,
    pub(crate) second_segment: usize,
    pub(crate) selected_braid_idx: usize,
    pub(crate) status_feedback: Option<String>,
    pub(crate) camera: Camera3D,
    pub(crate) scene_builder: Scene3DBuilder,
    pub(crate) model: StudioComputedModel,
    cached_key: u64,
}

impl Default for KnotBraidStudioTool {
    fn default() -> Self {
        let camera = Camera3D::new(0.5, 0.45, 1.0);
        let mut tool = Self {
            tube_radius: default_tube_radius(),
            family: StudioFamily::ArtinBraid,
            braid_strands: default_braid_strands(),
            braid_generators: default_braid_generators(),
            torus_p: default_torus_p(),
            torus_q: default_torus_q(),
            lissajous_freqs: default_lissajous_freqs(),
            lissajous_phases: default_lissajous_phases(),
            major_radius: default_major_radius(),
            minor_radius: default_minor_radius(),
            curve_samples: default_curve_samples(),
            radial_segments: default_radial_segments(),
            show_bishop_frames: false,
            show_tube_rings: true,
            reidemeister_ops: Vec::new(),
            selected_crossing: 0,
            second_crossing: 1,
            selected_segment: 0,
            second_segment: 1,
            selected_braid_idx: 0,
            status_feedback: None,
            camera,
            scene_builder: Scene3DBuilder::new(camera),
            model: StudioComputedModel::empty(),
            cached_key: 0,
        };
        tool.sync_model();
        tool
    }
}

fn validate_loaded_state(s: &KnotBraidState) -> Result<(), String> {
    if !s.tube_radius.is_finite() || !(0.03..=0.5).contains(&s.tube_radius) {
        return Err("tube_radius must be finite in [0.03, 0.5]".to_string());
    }
    if !(2..=8).contains(&s.braid_strands) || s.braid_generators.len() > 64 {
        return Err("braid_strands or braid_generators out of bounds".to_string());
    }
    for &g in &s.braid_generators {
        let idx = g.unsigned_abs() as usize;
        if idx == 0 || idx >= s.braid_strands {
            return Err(format!(
                "invalid Artin generator {g} for B_{}",
                s.braid_strands
            ));
        }
    }
    if !(2..=8).contains(&s.torus_p) || s.torus_q == 0 || !(-11..=11).contains(&s.torus_q) {
        return Err("torus_p or torus_q out of bounds".to_string());
    }
    if s.lissajous_freqs.iter().any(|&f| !(1..=9).contains(&f))
        || s.lissajous_phases
            .iter()
            .any(|p| !p.is_finite() || !(-6.5..=6.5).contains(p))
    {
        return Err("Lissajous frequencies or phases out of bounds".to_string());
    }
    if !s.major_radius.is_finite()
        || !(1.0..=4.0).contains(&s.major_radius)
        || !s.minor_radius.is_finite()
        || !(0.15..=1.5).contains(&s.minor_radius)
        || s.minor_radius >= s.major_radius
    {
        return Err("major_radius / minor_radius out of bounds".to_string());
    }
    if !(24..=180).contains(&s.curve_samples)
        || !(6..=32).contains(&s.radial_segments)
        || s.reidemeister_ops.len() > MAX_REIDEMEISTER_HISTORY
    {
        return Err("sampling resolution or Reidemeister history out of bounds".to_string());
    }
    Ok(())
}

impl InteractiveTool for KnotBraidStudioTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }

    fn name(&self) -> &'static str {
        "Knot, Braid & Skein Studio"
    }

    fn domain(&self) -> &'static str {
        "geometry_topology"
    }

    fn state_bounds(&self) -> ToolStateBounds {
        ToolStateBounds::new(crate::url_state::MAX_DECODED_URL_STATE_BYTES, 12, 128)
    }

    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot().with_3d_solid()
    }

    fn export_mesh_3d(&self) -> Option<Result<oxidize_core::mesh::Mesh, String>> {
        Some(
            build_link_tubular_mesh(&self.model.link_3d, self.tube_radius, self.radial_segments)
                .map_err(|e| e.to_string()),
        )
    }

    fn state_revision(&self) -> u64 {
        use std::hash::{Hash as _, Hasher as _};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.geometry_hash().hash(&mut h);
        self.show_bishop_frames.hash(&mut h);
        self.show_tube_rings.hash(&mut h);
        self.camera.hash_state(&mut h);
        h.finish().max(1)
    }

    fn has_finite_parameters(&self) -> bool {
        self.tube_radius.is_finite()
            && (0.03..=0.5).contains(&self.tube_radius)
            && (2..=8).contains(&self.braid_strands)
            && self.braid_generators.len() <= 64
            && (2..=8).contains(&self.torus_p)
            && self.torus_q != 0
            && (-11..=11).contains(&self.torus_q)
            && self.major_radius.is_finite()
            && self.minor_radius.is_finite()
            && self.minor_radius < self.major_radius
            && self.lissajous_phases.iter().all(|p| p.is_finite())
            && (24..=180).contains(&self.curve_samples)
            && (6..=32).contains(&self.radial_segments)
            && self.camera.is_valid()
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&KnotBraidSaveState {
            tube_radius: self.tube_radius,
            family: self.family,
            braid_strands: self.braid_strands,
            braid_generators: &self.braid_generators,
            torus_p: self.torus_p,
            torus_q: self.torus_q,
            lissajous_freqs: self.lissajous_freqs,
            lissajous_phases: self.lissajous_phases,
            major_radius: self.major_radius,
            minor_radius: self.minor_radius,
            curve_samples: self.curve_samples,
            radial_segments: self.radial_segments,
            show_bishop_frames: self.show_bishop_frames,
            show_tube_rings: self.show_tube_rings,
            camera: self.camera,
            reidemeister_ops: &self.reidemeister_ops,
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
        let loaded: KnotBraidState = serde_json::from_str(state)
            .map_err(|e| format!("Invalid KnotBraidStudio state: {e}"))?;
        validate_loaded_state(&loaded)?;
        let camera = Camera3D::from_preset_fields(
            loaded.camera,
            loaded.pitch,
            loaded.yaw,
            loaded.zoom,
            Some(Camera3D::new(0.5, 0.45, 1.0)),
        )?;
        self.tube_radius = loaded.tube_radius;
        self.family = loaded.family;
        self.braid_strands = loaded.braid_strands;
        self.braid_generators = loaded.braid_generators;
        self.torus_p = loaded.torus_p;
        self.torus_q = loaded.torus_q;
        self.lissajous_freqs = loaded.lissajous_freqs;
        self.lissajous_phases = loaded.lissajous_phases;
        self.major_radius = loaded.major_radius;
        self.minor_radius = loaded.minor_radius;
        self.curve_samples = loaded.curve_samples;
        self.radial_segments = loaded.radial_segments;
        self.show_bishop_frames = loaded.show_bishop_frames;
        self.show_tube_rings = loaded.show_tube_rings;
        self.reidemeister_ops = loaded.reidemeister_ops;
        self.camera.apply_pose(camera);
        self.cached_key = 0;
        self.sync_model();
        Ok(())
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        self.sync_model();
        self.draw_side_panel(ui);
        self.draw_viewports(ui);
    }

    #[cfg(test)]
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    #[cfg(test)]
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }
}

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "KnotBraidStudio",
        display_name: "Knot, Braid & Skein Studio",
        domain: "geometry_topology",
        tags: &["knot", "braid", "topology", "jones", "alexander"],
        build: || Box::new(KnotBraidStudioTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for KnotBraidStudioTool {
    fn theory_description(&self) -> String {
        format!(
            "3D spatial knot and Artin braid closure studio ({}): Bishop parallel-transport frames, \
             Gauss linking/writhe integrals, Reidemeister I/II/III ambient isotopies, Fox/Burau \
             Alexander polynomial, and Kauffman/Jones Laurent skein invariants.",
            self.family.label()
        )
    }

    fn phonetic_description(&self) -> String {
        "Three-dimensional knot theory, Artin braid closures, and Laurent skein invariants".into()
    }

    fn theory_citation(&self) -> String {
        "Kauffman (1987), State Models and the Jones Polynomial; Birman (1974), Braids, Links, and Mapping Class Groups".into()
    }

    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "ArtinBraid".into(),
            "Infinite Artin braid group B_n closed via Alexander's theorem".into(),
        );
        map.insert(
            "ReidemeisterMoves".into(),
            "Planar ambient isotopy moves preserving Alexander and Jones polynomials".into(),
        );
        map.insert(
            "BorromeanRings".into(),
            "Brunnian 3-component link with pairwise zero Gauss linking numbers".into(),
        );
        map
    }
}

#[cfg(test)]
mod tests;

#[cfg(all(test, not(target_arch = "wasm32")))]
mod screenshot_tests;

#[cfg(all(test, target_arch = "wasm32"))]
mod wasm_tests;
