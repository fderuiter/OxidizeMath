use crate::framework::{Camera3D, ExportCapabilities, InteractiveTool, Plot3DViewport};
use eframe::egui;
use egui_plot::{Line, PlotPoints, Points};
use math_explorer::physics::solid_state::lattice::{
    BodyCenteredCubic, CrystalSystem, FaceCenteredCubic, SimpleCubic, UnitCell,
};
use nalgebra::Vector3;
use oxidize_core::mesh::{
    append_colored_part, build_cylinder, build_uv_sphere, ColoredMesh, Mesh, Point3D, VertexColor,
};

#[derive(Debug, PartialEq, Eq, Clone, Copy, serde::Serialize, serde::Deserialize)]
#[allow(clippy::enum_variant_names)]
pub enum CrystalSystemEnum {
    SimpleCubic,
    BodyCenteredCubic,
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

    fn generate(&self, a: f64) -> UnitCell {
        match self {
            CrystalSystemEnum::SimpleCubic => SimpleCubic.generate(a),
            CrystalSystemEnum::BodyCenteredCubic => BodyCenteredCubic.generate(a),
            CrystalSystemEnum::FaceCenteredCubic => FaceCenteredCubic.generate(a),
        }
    }
}

pub struct CrystalViewer {
    system: CrystalSystemEnum,
    lattice_constant: f64,
    camera: Camera3D,
}

impl Default for CrystalViewer {
    fn default() -> Self {
        Self {
            system: CrystalSystemEnum::FaceCenteredCubic,
            lattice_constant: 2.0,
            camera: Camera3D::new(0.5, 0.5, 1.0),
        }
    }
}

impl CrystalViewer {
    fn project(&self, p: Vector3<f64>) -> [f64; 2] {
        let half = self.lattice_constant * 0.5;
        self.camera
            .project(&[p.x - half, p.y - half, p.z - half])
    }

    fn draw_unit_cell_box(&self, ui: &mut egui_plot::PlotUi) {
        let a = self.lattice_constant;
        let corners = [
            Vector3::new(0.0, 0.0, 0.0), // 0
            Vector3::new(a, 0.0, 0.0),   // 1
            Vector3::new(a, a, 0.0),     // 2
            Vector3::new(0.0, a, 0.0),   // 3
            Vector3::new(0.0, 0.0, a),   // 4
            Vector3::new(a, 0.0, a),     // 5
            Vector3::new(a, a, a),       // 6
            Vector3::new(0.0, a, a),     // 7
        ];

        let edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0), // Bottom face
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4), // Top face
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7), // Verticals
        ];

        let edge_color =
            crate::app::SemanticPalette::from_visuals(&ui.ctx().style().visuals).graph_edge;
        for (start_idx, end_idx) in edges {
            let start = self.project(corners[start_idx]);
            let end = self.project(corners[end_idx]);
            ui.line(
                Line::new("Edge", PlotPoints::new(vec![start, end]))
                    .color(edge_color)
                    .width(1.0_f32),
            );
        }
    }

    fn build_crystal_colored_mesh(&self) -> Result<ColoredMesh, String> {
        if !self.lattice_constant.is_finite() || self.lattice_constant <= 0.0 {
            return Err("Lattice constant must be finite and positive.".to_string());
        }
        let unit_cell = self.system.generate(self.lattice_constant);
        let [v1, v2, v3] = unit_cell.lattice_vectors;
        let center_offset = (v1 + v2 + v3) * 0.5;
        let to_p3 = |v: Vector3<f64>| -> Point3D {
            let c = v - center_offset;
            Point3D::new(c.x as f32, c.y as f32, c.z as f32)
        };

        let a = self.lattice_constant as f32;
        let atom_radius = (0.14 * a).max(0.05);
        let corner_radius = (0.10 * a).max(0.04);
        let strut_radius = (0.03 * a).max(0.015);
        let mut combined = ColoredMesh::default();

        let basis_color = VertexColor::new(235, 92, 74);
        for &pos in &unit_cell.atomic_positions {
            let sphere =
                build_uv_sphere(to_p3(pos), atom_radius, 16, 12).map_err(|e| e.to_string())?;
            append_colored_part(&mut combined, &sphere, basis_color);
        }

        let corners = [
            Vector3::zeros(),
            v1,
            v1 + v2,
            v2,
            v3,
            v1 + v3,
            v1 + v2 + v3,
            v2 + v3,
        ];
        let periodic_color = VertexColor::new(88, 176, 236);
        for corner in &corners {
            let overlaps_basis = unit_cell
                .atomic_positions
                .iter()
                .any(|p| (p - corner).norm() < 1e-5);
            if !overlaps_basis {
                let sphere = build_uv_sphere(to_p3(*corner), corner_radius, 14, 10)
                    .map_err(|e| e.to_string())?;
                append_colored_part(&mut combined, &sphere, periodic_color);
            }
        }
        let corner_pts = corners.map(to_p3);
        Self::append_unit_cell_struts(&mut combined, &corner_pts, strut_radius)?;
        Ok(combined)
    }

    fn append_unit_cell_struts(
        combined: &mut ColoredMesh,
        corners: &[Point3D; 8],
        strut_radius: f32,
    ) -> Result<(), String> {
        let strut_color = VertexColor::new(142, 156, 178);
        const EDGES: [(usize, usize); 12] = [
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
        for (i0, i1) in EDGES {
            let cyl = build_cylinder(corners[i0], corners[i1], strut_radius, 12)
                .map_err(|e| e.to_string())?;
            append_colored_part(combined, &cyl, strut_color);
        }
        Ok(())
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
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable { self }
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

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&CrystalSaveState {
            system: self.system,
            lattice_constant: self.lattice_constant,
            camera: self.camera,
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
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
        let panel_id = crate::framework::scoped_panel_id(parent_ui, "crystal_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            ui.heading("Crystal Lattice");
            ui.separator();

            ui.label("Crystal System");
            egui::ComboBox::from_id_salt("crystal_system_combo")
                .selected_text(self.system.label())
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut self.system,
                        CrystalSystemEnum::SimpleCubic,
                        CrystalSystemEnum::SimpleCubic.label(),
                    );
                    ui.selectable_value(
                        &mut self.system,
                        CrystalSystemEnum::BodyCenteredCubic,
                        CrystalSystemEnum::BodyCenteredCubic.label(),
                    );
                    ui.selectable_value(
                        &mut self.system,
                        CrystalSystemEnum::FaceCenteredCubic,
                        CrystalSystemEnum::FaceCenteredCubic.label(),
                    );
                });

            ui.add(egui::Slider::new(&mut self.lattice_constant, 0.5..=5.0).text("Lattice Constant (a)"));

            ui.separator();
            egui::CollapsingHeader::new("View")
                .default_open(true)
                .show(ui, |ui| self.camera.ui(ui));
        });

        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
            let mut camera = self.camera;
            Plot3DViewport::new("crystal_plot", 3.5).show(ui, &mut camera, |plot_ui| {
                self.draw_unit_cell_box(plot_ui);

                let unit_cell = self.system.generate(self.lattice_constant);
                let points: Vec<[f64; 2]> = unit_cell
                    .atomic_positions
                    .iter()
                    .map(|p| self.project(*p))
                    .collect();

                plot_ui.points(
                    Points::new("Atoms", points)
                        .radius(5.0_f32)
                        .color(palette.axis_x),
                );
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
        tags: &[],
        build: || Box::new(CrystalViewer::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for CrystalViewer {
    fn theory_description(&self) -> String {
        let (sys_name, formula) = match self.system {
            CrystalSystemEnum::SimpleCubic => ("Simple Cubic (SC)", "1 atom per unit cell, coordination number 6, packing fraction $\\pi/6 \\approx 0.52$"),
            CrystalSystemEnum::BodyCenteredCubic => ("Body-Centered Cubic (BCC)", "2 atoms per unit cell, coordination number 8, packing fraction $\\sqrt{3}\\pi/8 \\approx 0.68$"),
            CrystalSystemEnum::FaceCenteredCubic => ("Face-Centered Cubic (FCC)", "4 atoms per unit cell, coordination number 12, close-packed fraction $\\pi/(3\\sqrt{2}) \\approx 0.74$"),
        };
        format!(
            "Crystallographic Bravais lattice geometry for {} with lattice constant $a = {:.2}\\,\\text{{\\AA}}$: {}.",
            sys_name, self.lattice_constant, formula
        )
    }
    fn phonetic_description(&self) -> String { "Crystallographic Bravais lattice and unit cell geometry theory".into() }
    fn theory_citation(&self) -> String { "Introduction to Solid State Physics (Kittel, 2004); Solid State Physics (Ashcroft & Mermin, 1976)".into() }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert("Simple Cubic".into(), "Primitive cubic lattice with basis at (0,0,0)".into());
        map.insert("BCC".into(), "Body-centered lattice with basis at (0,0,0) and (1/2, 1/2, 1/2)".into());
        map.insert("FCC".into(), "Face-centered close-packed cubic lattice".into());
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
