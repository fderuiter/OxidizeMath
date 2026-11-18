use crate::framework::InteractiveTool;
use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints, Points};
use math_explorer::physics::solid_state::lattice::{
    BodyCenteredCubic, CrystalSystem, FaceCenteredCubic, SimpleCubic, UnitCell,
};
use nalgebra::Vector3;

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

    // Camera
    yaw: f32,
    pitch: f32,
    zoom: f32,
}

impl Default for CrystalViewer {
    fn default() -> Self {
        Self {
            system: CrystalSystemEnum::FaceCenteredCubic,
            lattice_constant: 2.0,
            yaw: 0.5,
            pitch: 0.5,
            zoom: 50.0,
        }
    }
}

impl CrystalViewer {
    fn project(&self, p: Vector3<f64>) -> [f64; 2] {
        let center = Vector3::new(
            self.lattice_constant / 2.0,
            self.lattice_constant / 2.0,
            self.lattice_constant / 2.0,
        );
        let p_centered = p - center;

        // Rotation
        let cy = (self.yaw as f64).cos();
        let sy = (self.yaw as f64).sin();
        let cp = (self.pitch as f64).cos();
        let sp = (self.pitch as f64).sin();

        // Rotate around Y (Yaw)
        let x1 = p_centered.x * cy - p_centered.z * sy;
        let z1 = p_centered.x * sy + p_centered.z * cy;
        let y1 = p_centered.y;

        // Rotate around X (Pitch)
        let y2 = y1 * cp - z1 * sp;
        // let z2 = y1 * sp + z1 * cp; // Depth, ignored for 2D projection

        [x1 * (self.zoom as f64), y2 * (self.zoom as f64)]
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
}

#[derive(serde::Serialize, serde::Deserialize)]
struct CrystalState {
    system: CrystalSystemEnum,
    lattice_constant: f64,
    yaw: f32,
    pitch: f32,
    zoom: f32,
}

impl InteractiveTool for CrystalViewer {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable { self }
    fn name(&self) -> &'static str {
        "Crystal Lattice Viewer"
    }
    fn domain(&self) -> &'static str {
        "solid_state"
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&CrystalState {
            system: self.system,
            lattice_constant: self.lattice_constant,
            yaw: self.yaw,
            pitch: self.pitch,
            zoom: self.zoom,
        })
        .ok()
    }

    fn load_state(&mut self, state: &str) {
        if let Ok(s) = serde_json::from_str::<CrystalState>(state) {
            self.system = s.system;
            self.lattice_constant = s.lattice_constant;
            self.yaw = s.yaw;
            self.pitch = s.pitch;
            self.zoom = s.zoom;
        }
    }

    fn show(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left("crystal_controls").show(ctx, |ui| {
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
            ui.heading("Camera");
            ui.label("Yaw");
            ui.drag_angle(&mut self.yaw);
            ui.label("Pitch");
            ui.drag_angle(&mut self.pitch);
            ui.add(egui::Slider::new(&mut self.zoom, 10.0..=200.0).text("Zoom"));
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
            Plot::new("crystal_plot")
                .data_aspect(1.0)
                .show(ui, |plot_ui| {
                    // Draw Box
                    self.draw_unit_cell_box(plot_ui);

                    // Draw Atoms
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
            ui.label("Drag 'Yaw' and 'Pitch' to rotate.");
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
    "structure": "FCC",
    "unit_cells": 2
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "crystal_viewer_starter",
        title: "Bravais Crystal Lattice",
        description: "3D periodic atomic unit cell crystallographic coordinates (FCC, BCC, Diamond cubic structures).",
        category: "Crystallography",
        icon: "💎",
        domain: "solid_state",
        tool_name: Some("CrystalViewer"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}
