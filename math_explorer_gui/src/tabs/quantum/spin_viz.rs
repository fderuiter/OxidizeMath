#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

#[path = "spin_export.rs"]
mod spin_export;
#[path = "spin_interaction.rs"]
pub mod spin_interaction;

use std::f64::consts::{PI, TAU};

use crate::framework::{Camera3D, ExportCapabilities, InteractiveTool, Scene3DBuilder};
use eframe::egui;
use egui_plot::{LineStyle, MarkerShape};
use math_explorer::physics::quantum::{evolve_state_in_place, spin, QuantumOperator, QuantumState};
use nalgebra::{DMatrix, DVector, Point3};
use num_complex::Complex;
use spin_interaction::{b_field_visual_tip, BlochSteerTarget};

/// Interactive quantum spin-1/2 Bloch sphere visualizer built on `Plot3DViewport` and `Scene3DBuilder`.
pub struct SpinVisualizer {
    pub(crate) psi: QuantumState,
    pub(crate) b_field: [f64; 3],
    pub(crate) initial_bloch_angles: Option<[f64; 2]>,
    pub(crate) time: f64,
    pub(crate) paused: bool,
    pub(crate) camera: Camera3D,
    pub(crate) scene_builder: Scene3DBuilder,
    pub(crate) active_drag: Option<BlochSteerTarget>,
    pub(crate) steer_target: BlochSteerTarget,
    pub(crate) last_plot_rect: Option<egui::Rect>,
    pub(crate) last_plot_bounds: Option<[f64; 4]>,
    sx: QuantumOperator,
    sy: QuantumOperator,
    sz: QuantumOperator,
    propagator: QuantumOperator,
    cached_key: Option<[u64; 4]>,
    scratch: DVector<Complex<f64>>,
}

impl Default for SpinVisualizer {
    fn default() -> Self {
        let camera = Camera3D::new(0.5, 0.5, 1.0);
        Self {
            psi: QuantumState::spin_zero(),
            b_field: [0.0, 0.0, 1.0],
            initial_bloch_angles: None,
            time: 0.0,
            paused: true,
            camera,
            scene_builder: Scene3DBuilder::new(camera),
            active_drag: None,
            steer_target: BlochSteerTarget::InitialState,
            last_plot_rect: None,
            last_plot_bounds: None,
            sx: spin::sigma_x(),
            sy: spin::sigma_y(),
            sz: spin::sigma_z(),
            propagator: QuantumOperator::new(DMatrix::identity(2, 2)),
            cached_key: None,
            scratch: DVector::from_element(2, Complex::new(0.0, 0.0)),
        }
    }
}

impl SpinVisualizer {
    /// Resets the quantum state to `initial_bloch_angles` (or $|0\rangle$ along $+Z$) and zeroes elapsed time.
    pub fn reset(&mut self) {
        if let Some([theta, phi]) = self.initial_bloch_angles {
            spin::set_spin_bloch_angles_in_place(&mut self.psi, theta, phi);
        } else {
            spin::set_spin_bloch_angles_in_place(&mut self.psi, 0.0, 0.0);
        }
        self.time = 0.0;
    }

    /// Pauses continuous playback and advances a single $\Delta t = 0.05$ Zeeman step.
    pub fn step_once(&mut self) {
        self.paused = true;
        self.step(0.05);
    }

    fn refresh_propagator(&mut self, dt: f64) {
        let [bx, by, bz] = self.b_field;
        let key = [bx.to_bits(), by.to_bits(), bz.to_bits(), dt.to_bits()];
        if self.cached_key != Some(key) {
            let b_norm = (bx * bx + by * by + bz * bz).sqrt();
            if b_norm > 1.0e-15 && b_norm.is_finite() && dt.is_finite() {
                let (sin_a, cos_a) = (0.5 * b_norm * dt).sin_cos();
                let inv_b = 1.0 / b_norm;
                let (nx, ny, nz) = (bx * inv_b, by * inv_b, bz * inv_b);
                self.propagator.matrix[(0, 0)] = Complex::new(cos_a, -nz * sin_a);
                self.propagator.matrix[(0, 1)] = Complex::new(-ny * sin_a, -nx * sin_a);
                self.propagator.matrix[(1, 0)] = Complex::new(ny * sin_a, -nx * sin_a);
                self.propagator.matrix[(1, 1)] = Complex::new(cos_a, nz * sin_a);
            } else {
                self.propagator.matrix[(0, 0)] = Complex::new(1.0, 0.0);
                self.propagator.matrix[(0, 1)] = Complex::new(0.0, 0.0);
                self.propagator.matrix[(1, 0)] = Complex::new(0.0, 0.0);
                self.propagator.matrix[(1, 1)] = Complex::new(1.0, 0.0);
            }
            self.cached_key = Some(key);
        }
    }

    pub(crate) fn step(&mut self, dt: f64) {
        self.refresh_propagator(dt);
        evolve_state_in_place(&mut self.psi, &self.propagator, &mut self.scratch);
        self.psi.normalize_in_place();
        self.time += dt;
    }

    fn bloch_circle_3d(axis: usize) -> [Point3<f64>; 61] {
        let mut ring = [Point3::origin(); 61];
        for (i, pt) in ring.iter_mut().enumerate() {
            let theta = (i as f64) * std::f64::consts::TAU / 60.0;
            let (s, c) = theta.sin_cos();
            *pt = match axis {
                0 => Point3::new(c, s, 0.0),
                1 => Point3::new(0.0, c, s),
                _ => Point3::new(c, 0.0, s),
            };
        }
        ring
    }

    /// Populates `self.scene_builder` with Bloch sphere rings, axes, magnetic field $\vec{B}$, and spin state vector.
    pub fn populate_scene(
        &mut self,
        ex: f64,
        ey: f64,
        ez: f64,
        edge_color: egui::Color32,
        accent_color: egui::Color32,
        dark_mode: bool,
    ) {
        let b_color = if dark_mode {
            crate::app::palette::DARK_PALETTE.warning
        } else {
            crate::app::palette::LIGHT_PALETTE.warning
        };
        self.scene_builder.begin(self.camera);
        let ring_stroke = egui::Stroke::new(1.0_f32, edge_color);
        for axis in 0..3 {
            let circle = Self::bloch_circle_3d(axis);
            self.scene_builder
                .add_polyline("", &circle, ring_stroke, LineStyle::Solid, false);
        }
        self.scene_builder.add_axes(1.2, dark_mode);
        if let Some((b_tip, _)) = b_field_visual_tip(self.b_field) {
            self.scene_builder.add_segment(
                "B Field",
                Point3::origin(),
                b_tip,
                egui::Stroke::new(2.4_f32, b_color),
                true,
            );
            self.scene_builder.add_points(
                "B Tip",
                &[b_tip],
                5.0_f32,
                b_color,
                MarkerShape::Diamond,
                true,
            );
            self.scene_builder
                .add_label(Point3::from(b_tip.coords * 1.07), "B", b_color);
        }
        let tip = Point3::new(ex, ey, ez);
        self.scene_builder.add_segment(
            "State",
            Point3::origin(),
            tip,
            egui::Stroke::new(3.0_f32, accent_color),
            true,
        );
        self.scene_builder.add_points(
            "Spin Tip",
            &[tip],
            5.0_f32,
            accent_color,
            MarkerShape::Circle,
            true,
        );
    }
}

#[derive(serde::Serialize)]
struct SpinVisualizerSaveState {
    b_field: [f64; 3],
    paused: bool,
    camera: Camera3D,
    #[serde(skip_serializing_if = "Option::is_none")]
    initial_bloch_angles: Option<[f64; 2]>,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct SpinVisualizerState {
    b_field: [f64; 3],
    paused: bool,
    #[serde(default)]
    camera: Option<Camera3D>,
    #[serde(default)]
    pitch: Option<f32>,
    #[serde(default)]
    yaw: Option<f32>,
    #[serde(default)]
    zoom: Option<f32>,
    #[serde(default)]
    initial_bloch_angles: Option<[f64; 2]>,
}

impl InteractiveTool for SpinVisualizer {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }
    fn name(&self) -> &'static str {
        "Spin Dynamics (Bloch Sphere)"
    }
    fn domain(&self) -> &'static str {
        "quantum"
    }

    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot().with_3d_solid()
    }

    fn export_colored_mesh_3d(&self) -> Option<Result<oxidize_core::mesh::ColoredMesh, String>> {
        Some(self.build_bloch_colored_mesh_3d())
    }

    fn export_mesh_3d(&self) -> Option<Result<oxidize_core::mesh::Mesh, String>> {
        Some(self.build_bloch_colored_mesh_3d().map(|cm| cm.mesh))
    }

    fn state_revision(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        for component in self.b_field {
            component.to_bits().hash(&mut h);
        }
        if let Some([theta, phi]) = self.initial_bloch_angles {
            1_u8.hash(&mut h);
            theta.to_bits().hash(&mut h);
            phi.to_bits().hash(&mut h);
        } else {
            0_u8.hash(&mut h);
        }
        self.time.to_bits().hash(&mut h);
        self.paused.hash(&mut h);
        self.camera.hash_state(&mut h);
        h.finish().max(1)
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&SpinVisualizerSaveState {
            b_field: self.b_field,
            paused: self.paused,
            camera: self.camera,
            initial_bloch_angles: self.initial_bloch_angles,
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
        let loaded: SpinVisualizerState = serde_json::from_str(state)
            .map_err(|e| format!("Invalid SpinVisualizer state: {e}"))?;
        if !loaded
            .b_field
            .iter()
            .all(|v| v.is_finite() && (-5.0..=5.0).contains(v))
        {
            return Err("b_field components must be finite in [-5.0, 5.0]".to_string());
        }
        if let Some([theta, phi]) = loaded.initial_bloch_angles {
            if !theta.is_finite()
                || !(0.0..=PI).contains(&theta)
                || !phi.is_finite()
                || !(-TAU..=TAU).contains(&phi)
            {
                return Err(
                    "initial_bloch_angles must have finite theta in [0, pi] and phi in [-2pi, 2pi]"
                        .to_string(),
                );
            }
        }
        let camera = Camera3D::from_preset_fields(
            loaded.camera,
            loaded.pitch,
            loaded.yaw,
            loaded.zoom,
            Some(Camera3D::default()),
        )?;
        self.b_field = loaded.b_field;
        self.paused = loaded.paused;
        self.initial_bloch_angles = loaded
            .initial_bloch_angles
            .map(|[th, ph]| [th, spin_interaction::wrap_azimuth_rad(ph)]);
        self.camera.apply_pose(camera);
        self.reset();
        Ok(())
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        if !self.paused {
            self.step(0.05);
            ui.ctx().request_repaint();
        }

        let ex = self.sx.expectation_value(&self.psi).re;
        let ey = self.sy.expectation_value(&self.psi).re;
        let ez = self.sz.expectation_value(&self.psi).re;

        self.draw_controls(ui, ex, ey, ez);
        self.draw_bloch_sphere(ui, ex, ey, ez);
    }
}

// [cite:quantum_mechanics]
// module = "quantum"

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "SpinVisualizer",
        display_name: "Spin Dynamics (Bloch Sphere)",
        domain: "quantum",
        tags: &[],
        build: || Box::new(SpinVisualizer::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for SpinVisualizer {
    fn theory_description(&self) -> String {
        format!(
            "Quantum spin-1/2 state $|\\psi\\rangle = \\alpha |0\\rangle + \\beta |1\\rangle$ on the Bloch sphere evolved under Zeeman interaction Hamiltonian $H = -\\vec{{\\mu}} \\cdot \\vec{{B}} = -\\frac{{g \\mu_B}}{{\\hbar}} \\vec{{S}} \\cdot \\vec{{B}}$ with magnetic field $\\vec{{B}} = [{:.2}, {:.2}, {:.2}]$.",
            self.b_field[0], self.b_field[1], self.b_field[2]
        )
    }
    fn phonetic_description(&self) -> String {
        "Quantum spin-1/2 Bloch sphere dynamics and Zeeman effect theory".into()
    }
    fn theory_citation(&self) -> String {
        "Principles of Quantum Mechanics (Shankar, 2012); Quantum Mechanics: Concepts and Applications (Zettili, 2009)".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
}

#[cfg(test)]
#[path = "spin_studio_tests.rs"]
mod tests;

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Bloch Sphere Spin State",
    "domain": "quantum",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "b_field": [1.0, 0.0, 1.0],
    "paused": false,
    "camera": {
      "pitch": 0.5,
      "yaw": 0.5,
      "zoom": 1.0
    }
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "spin_viz_starter",
        title: "Bloch Sphere Spin State",
        description: "Quantum spin-1/2 state vector on the Bloch sphere evolving under a 3D Zeeman magnetic field.",
        category: "Quantum States",
        icon: "🌐",
        domain: "quantum",
        tool_name: Some("SpinVisualizer"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}
