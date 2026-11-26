#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

#[path = "crystal_export.rs"]
mod crystal_export;

use crate::framework::{
    Camera3D, ExportCapabilities, InteractiveTool, Plot3DViewport, Scene3DBuilder,
};
use eframe::egui;
use egui_plot::MarkerShape;
use math_explorer::physics::solid_state::lattice::{
    BodyCenteredCubic, CrystalSystem, FaceCenteredCubic, SimpleCubic, UnitCell,
};
use nalgebra::Point3;
use oxidize_core::mesh::{ColoredMesh, Mesh};

/// Supported cubic Bravais crystal systems in [`CrystalViewer`].
#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy, serde::Serialize, serde::Deserialize)]
#[allow(clippy::enum_variant_names)]
pub enum CrystalSystemEnum {
    /// Primitive simple cubic lattice ($1$ atom per cell).
    SimpleCubic,
    /// Body-centered cubic lattice ($2$ atoms per cell).
    BodyCenteredCubic,
    /// Face-centered cubic lattice ($4$ atoms per cell).
    FaceCenteredCubic,
}

impl CrystalSystemEnum {
    /// Returns a readable crystallographic system title for UI selectors.
    pub fn label(&self) -> &'static str {
        match self {
            CrystalSystemEnum::SimpleCubic => "Simple Cubic",
            CrystalSystemEnum::BodyCenteredCubic => "Body-Centered Cubic (BCC)",
            CrystalSystemEnum::FaceCenteredCubic => "Face-Centered Cubic (FCC)",
        }
    }

    pub(super) fn generate(&self, a: f64) -> UnitCell {
        match self {
            CrystalSystemEnum::SimpleCubic => SimpleCubic.generate(a),
            CrystalSystemEnum::BodyCenteredCubic => BodyCenteredCubic.generate(a),
            CrystalSystemEnum::FaceCenteredCubic => FaceCenteredCubic.generate(a),
        }
    }
}

/// Interactive 3D crystallographic Bravais unit-cell viewer built on `Plot3DViewport` and `Scene3DBuilder`.
pub struct CrystalViewer {
    system: CrystalSystemEnum,
    lattice_constant: f64,
    pub(crate) camera: Camera3D,
    pub(crate) scene_builder: Scene3DBuilder,
    atoms_scratch: Vec<Point3<f64>>,
}

impl Default for CrystalViewer {
    fn default() -> Self {
        let camera = Camera3D::new(0.5, 0.5, 1.0);
        Self {
            system: CrystalSystemEnum::FaceCenteredCubic,
            lattice_constant: 2.0,
            camera,
            scene_builder: Scene3DBuilder::new(camera),
            atoms_scratch: Vec::new(),
        }
    }
}

impl CrystalViewer {
    /// Populates `self.scene_builder` with world `Point3<f64>` unit-cell edges and atomic sites.
    pub fn populate_scene(&mut self, edge_color: egui::Color32, atom_color: egui::Color32) {
        let a = self.lattice_constant;
        let half = a * 0.5;
        self.scene_builder.begin(self.camera);
        self.scene_builder
            .set_origin_offset(Point3::new(half, half, half));

        let corners = [
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(a, 0.0, 0.0),
            Point3::new(a, a, 0.0),
            Point3::new(0.0, a, 0.0),
            Point3::new(0.0, 0.0, a),
            Point3::new(a, 0.0, a),
            Point3::new(a, a, a),
            Point3::new(0.0, a, a),
        ];
        let edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ];
        let stroke = egui::Stroke::new(1.0_f32, edge_color);
        for (start_idx, end_idx) in edges {
            self.scene_builder.add_segment(
                "Edge",
                corners[start_idx],
                corners[end_idx],
                stroke,
                false,
            );
        }

        let unit_cell = self.system.generate(a);
        self.atoms_scratch.clear();
        self.atoms_scratch.extend(
            unit_cell
                .atomic_positions
                .iter()
                .map(|p| Point3::new(p.x, p.y, p.z)),
        );
        self.scene_builder.add_points(
            "Atoms",
            &self.atoms_scratch,
            5.0_f32,
            atom_color,
            MarkerShape::Circle,
            true,
        );
    }

    fn draw_side_panel(&mut self, parent_ui: &mut egui::Ui) {
        let panel_id = crate::framework::scoped_panel_id(parent_ui, "crystal_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            ui.heading("Crystal Lattice");
            ui.separator();

            ui.label("Crystal System");
            egui::ComboBox::from_id_salt("crystal_system_combo")
                .selected_text(self.system.label())
                .show_ui(ui, |ui| {
                    for variant in [
                        CrystalSystemEnum::SimpleCubic,
                        CrystalSystemEnum::BodyCenteredCubic,
                        CrystalSystemEnum::FaceCenteredCubic,
                    ] {
                        ui.selectable_value(&mut self.system, variant, variant.label());
                    }
                });

            ui.add(
                egui::Slider::new(&mut self.lattice_constant, 0.5..=5.0)
                    .text("Lattice Constant (a)"),
            );

            ui.separator();
            egui::CollapsingHeader::new("View")
                .default_open(true)
                .show(ui, |ui| self.camera.ui(ui));
        });
    }
}

#[derive(serde::Serialize)]
struct CrystalSaveState {
    system: CrystalSystemEnum,
    lattice_constant: f64,
    camera: Camera3D,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct CrystalState {
    system: CrystalSystemEnum,
    lattice_constant: f64,
    #[serde(default)]
    camera: Option<Camera3D>,
    #[serde(default)]
    yaw: Option<f32>,
    #[serde(default)]
    pitch: Option<f32>,
    #[serde(default)]
    zoom: Option<f32>,
}

impl InteractiveTool for CrystalViewer {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }
    fn name(&self) -> &'static str {
        "Crystal Lattice Viewer"
    }
    fn domain(&self) -> &'static str {
        "solid_state"
    }

    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot().with_3d_solid()
    }

    fn export_colored_mesh_3d(&self) -> Option<Result<ColoredMesh, String>> {
        Some(self.build_crystal_colored_mesh())
    }

    fn export_mesh_3d(&self) -> Option<Result<Mesh, String>> {
        Some(self.build_crystal_colored_mesh().map(|cm| cm.mesh))
    }

    fn state_revision(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.system.hash(&mut h);
        self.lattice_constant.to_bits().hash(&mut h);
        self.camera.hash_state(&mut h);
        h.finish().max(1)
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&CrystalSaveState {
            system: self.system,
            lattice_constant: self.lattice_constant,
            camera: self.camera,
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
        let s: CrystalState =
            serde_json::from_str(state).map_err(|e| format!("Invalid CrystalViewer state: {e}"))?;
        if !s.lattice_constant.is_finite() || !(0.5..=5.0).contains(&s.lattice_constant) {
            return Err(format!(
                "lattice_constant out of [0.5, 5.0]: {}",
                s.lattice_constant
            ));
        }
        let normalized_zoom = s.zoom.map(|z| {
            if (10.0_f32..=200.0_f32).contains(&z) {
                z / 50.0_f32
            } else if z > 200.0_f32 {
                f32::NAN
            } else {
                z
            }
        });
        let camera = Camera3D::from_preset_fields(
            s.camera,
            s.pitch,
            s.yaw,
            normalized_zoom,
            Some(Camera3D::default()),
        )?;
        self.system = s.system;
        self.lattice_constant = s.lattice_constant;
        self.camera.apply_pose(camera);
        Ok(())
    }

    fn show_inside(&mut self, parent_ui: &mut egui::Ui) {
        self.draw_side_panel(parent_ui);
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
            self.populate_scene(palette.graph_edge, palette.axis_x);
            let mut camera = self.camera;
            Plot3DViewport::new("crystal_plot", 3.5).show(ui, &mut camera, |plot_ui| {
                self.scene_builder.finish(plot_ui);
            });
            self.camera = camera;
        });
    }
}

// [cite:modular_polynomials_review]

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "CrystalViewer",
        display_name: "Crystal Lattice Viewer",
        domain: "solid_state",
        tags: &["bravais lattice", "bcc", "fcc", "unit cell", "crystallography", "3d"],
        build: || Box::new(CrystalViewer::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for CrystalViewer {
    fn theory_description(&self) -> String {
        let (sys_name, formula) = match self.system {
            CrystalSystemEnum::SimpleCubic => (
                "Simple Cubic (SC)",
                "1 atom per unit cell, coordination number 6, packing fraction $\\pi/6 \\approx 0.52$",
            ),
            CrystalSystemEnum::BodyCenteredCubic => (
                "Body-Centered Cubic (BCC)",
                "2 atoms per unit cell, coordination number 8, packing fraction $\\sqrt{3}\\pi/8 \\approx 0.68$",
            ),
            CrystalSystemEnum::FaceCenteredCubic => (
                "Face-Centered Cubic (FCC)",
                "4 atoms per unit cell, coordination number 12, close-packed fraction $\\pi/(3\\sqrt{2}) \\approx 0.74$",
            ),
        };
        format!(
            "Crystallographic Bravais lattice geometry for {sys_name} with lattice constant $a = {:.2}\\,\\text{{\\AA}}$: {formula}.",
            self.lattice_constant
        )
    }
    fn phonetic_description(&self) -> String {
        "Crystallographic Bravais lattice and unit cell geometry theory".into()
    }
    fn theory_citation(&self) -> String {
        "Introduction to Solid State Physics (Kittel, 2004); Solid State Physics (Ashcroft & Mermin, 1976)".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "Simple Cubic".into(),
            "Primitive cubic lattice with basis at (0,0,0)".into(),
        );
        map.insert(
            "BCC".into(),
            "Body-centered lattice with basis at (0,0,0) and (1/2, 1/2, 1/2)".into(),
        );
        map.insert(
            "FCC".into(),
            "Face-centered close-packed cubic lattice".into(),
        );
        map
    }
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Bravais Crystal Lattice",
    "domain": "solid_state",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "system": "FaceCenteredCubic",
    "lattice_constant": 3.0,
    "camera": {
      "pitch": 0.4,
      "yaw": 0.6,
      "zoom": 1.3
    }
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "crystal_viewer_starter",
        title: "Bravais Crystal Lattice",
        description: "3D periodic unit cell atomic coordinates for Simple Cubic, BCC, and FCC Bravais lattices.",
        category: "Crystallography",
        icon: "💎",
        domain: "solid_state",
        tool_name: Some("CrystalViewer"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}

#[cfg(test)]
mod tests {
    #![cfg_attr(any(), verified(opt_out = "tests"))]

    use super::*;

    #[test]
    fn crystal_viewer_camera_preset_and_export_work() {
        let tool = CrystalViewer {
            camera: Camera3D::new(0.35, 0.9, 1.5),
            ..Default::default()
        };
        let saved = tool.save_state().expect("state should serialize");

        let mut restored = CrystalViewer::default();
        restored
            .try_load_state(&saved)
            .expect("saved state should load");
        assert!((restored.camera.pitch - 0.35).abs() < 1e-6);
        assert!((restored.camera.yaw - 0.9).abs() < 1e-6);
        assert!((restored.camera.zoom - 1.5).abs() < 1e-6);

        let legacy = r#"{"system":"BodyCenteredCubic","lattice_constant":2.5,"pitch":0.4,"yaw":0.6,"zoom":1.3}"#;
        restored
            .try_load_state(legacy)
            .expect("legacy pitch/yaw/zoom preset should load");
        assert!((restored.camera.zoom - 1.3).abs() < 1e-6);

        let ctx = egui::Context::default();
        let _ = ctx.run(egui::RawInput::default(), |ctx| restored.show(ctx));
        assert!(!egui_plot::get_last_plot_datasets().is_empty());
        assert!(!egui_plot::get_last_plot_datasets_3d().is_empty());

        for sys in [
            CrystalSystemEnum::SimpleCubic,
            CrystalSystemEnum::BodyCenteredCubic,
            CrystalSystemEnum::FaceCenteredCubic,
        ] {
            restored.system = sys;
            let cm = restored
                .export_colored_mesh_3d()
                .expect("crystal 3D hook present")
                .expect("crystal mesh builds");
            assert_eq!(cm.colors.len(), cm.mesh.vertices.len());
            assert_eq!(
                crate::tabs::geometry_topology::export_utils::count_open_and_nonmanifold_edges(
                    &cm.mesh
                ),
                (0, 0)
            );
            assert!(crate::tabs::geometry_topology::export_utils::signed_volume(&cm.mesh) > 0.0);
        }
    }
}
