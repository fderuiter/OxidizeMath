#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::accessibility::AccessibleTheoryHover;
use crate::framework::{
    Camera3D, InteractiveTool, PlaybackAction, PlaybackTransport, Plot3DViewport, Scene3DBuilder,
};
use eframe::egui;
use egui_plot::{LineStyle, MarkerShape};
use math_explorer::physics::quantum::{
    evolve_state_in_place, spin, time_evolution_operator, QuantumOperator, QuantumState,
};
use nalgebra::{DMatrix, DVector, Point3};
use num_complex::Complex;

/// Interactive quantum spin-1/2 Bloch sphere visualizer built on `Plot3DViewport` and `Scene3DBuilder`.
pub struct SpinVisualizer {
    psi: QuantumState,
    b_field: [f64; 3],
    time: f64,
    paused: bool,
    pub(crate) camera: Camera3D,
    pub(crate) scene_builder: Scene3DBuilder,
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
            time: 0.0,
            paused: true,
            camera,
            scene_builder: Scene3DBuilder::new(camera),
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
    /// Resets the quantum state to $|0\rangle$ along $+Z$ and zeroes elapsed time.
    pub fn reset(&mut self) {
        self.psi = QuantumState::spin_zero();
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
            let h_matrix = (&self.sx.matrix * Complex::new(bx, 0.0)
                + &self.sy.matrix * Complex::new(by, 0.0)
                + &self.sz.matrix * Complex::new(bz, 0.0))
                * Complex::new(0.5, 0.0);
            let hamiltonian = QuantumOperator::new(h_matrix);
            self.propagator = time_evolution_operator(&hamiltonian, dt, 1.0);
            self.cached_key = Some(key);
        }
    }

    fn step(&mut self, dt: f64) {
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

    /// Populates `self.scene_builder` with Bloch sphere rings, axes, state vector, and spin tip.
    pub fn populate_scene(
        &mut self,
        ex: f64,
        ey: f64,
        ez: f64,
        edge_color: egui::Color32,
        accent_color: egui::Color32,
        dark_mode: bool,
    ) {
        self.scene_builder.begin(self.camera);
        let ring_stroke = egui::Stroke::new(1.0_f32, edge_color);
        for axis in 0..3 {
            let circle = Self::bloch_circle_3d(axis);
            self.scene_builder
                .add_polyline("", &circle, ring_stroke, LineStyle::Solid, false);
        }
        self.scene_builder.add_axes(1.2, dark_mode);
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

    fn draw_controls(&mut self, parent_ui: &mut egui::Ui, ex: f64, ey: f64, ez: f64) {
        let panel_id = crate::framework::scoped_panel_id(parent_ui, "spin_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            ui.heading("Controls");
            ui.separator();

            ui.group(|ui| {
                ui.add(
                    egui::Slider::new(&mut self.b_field[0], -5.0..=5.0)
                        .text("Magnetic Field B - Bx"),
                );
                ui.add(egui::Slider::new(&mut self.b_field[1], -5.0..=5.0).text("By"));
                ui.add(egui::Slider::new(&mut self.b_field[2], -5.0..=5.0).text("Bz"));
            });

            ui.separator();
            if let Some(action) = PlaybackTransport::in_thread(
                "spin_viz_transport",
                "Spin Dynamics (Bloch Sphere)",
                !self.paused,
            )
            .show(ui)
            {
                match action {
                    PlaybackAction::Play => self.paused = false,
                    PlaybackAction::Pause => self.paused = true,
                    PlaybackAction::Step => self.step_once(),
                    PlaybackAction::Reset => self.reset(),
                    PlaybackAction::StepBack | PlaybackAction::Scrub(_) => {}
                }
            }

            ui.separator();
            ui.heading("View");
            self.camera.ui(ui);

            ui.separator();
            ui.label(format!("Ex (x): {ex:.3}"));
            ui.label(format!("Ey (y): {ey:.3}"));
            ui.label(format!("Ez (z): {ez:.3}"));
        });
    }

    fn draw_bloch_sphere(&mut self, parent_ui: &mut egui::Ui, ex: f64, ey: f64, ez: f64) {
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
            let dark_mode = ui.visuals().dark_mode;
            self.populate_scene(ex, ey, ez, palette.graph_edge, palette.accent, dark_mode);
            let mut camera = self.camera;
            let response = Plot3DViewport::new("bloch_sphere", 1.5)
                .show(ui, &mut camera, |plot_ui| {
                    self.scene_builder.finish(plot_ui);
                })
                .response;
            self.camera = camera;
            response.accessible_theory_hover(&self.psi);
        });
    }
}

#[derive(serde::Serialize)]
struct SpinVisualizerSaveState {
    b_field: [f64; 3],
    paused: bool,
    camera: Camera3D,
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

    fn state_revision(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        for component in self.b_field {
            component.to_bits().hash(&mut h);
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
        let camera = Camera3D::from_preset_fields(
            loaded.camera,
            loaded.pitch,
            loaded.yaw,
            loaded.zoom,
            Some(Camera3D::default()),
        )?;
        self.b_field = loaded.b_field;
        self.paused = loaded.paused;
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
mod tests {
    #![cfg_attr(any(), verified(opt_out = "tests"))]

    use super::*;

    #[test]
    fn test_spin_visualizer_initialization_and_reset() {
        let mut viz = SpinVisualizer::default();
        assert_eq!(viz.time, 0.0);
        assert!(viz.paused);

        let ez = viz.sz.expectation_value(&viz.psi).re;
        assert!((ez - 1.0).abs() < 1e-6);

        viz.step_once();
        assert!(viz.paused);
        assert!((viz.time - 0.05).abs() < 1e-9);

        viz.reset();
        assert_eq!(viz.time, 0.0);
        let ez_reset = viz.sz.expectation_value(&viz.psi).re;
        assert!((ez_reset - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cached_operators_match_pauli() {
        let viz = SpinVisualizer::default();
        let sx_ref = spin::sigma_x();
        let sy_ref = spin::sigma_y();
        let sz_ref = spin::sigma_z();

        assert_eq!(viz.sx.matrix, sx_ref.matrix);
        assert_eq!(viz.sy.matrix, sy_ref.matrix);
        assert_eq!(viz.sz.matrix, sz_ref.matrix);
    }

    #[test]
    fn test_spin_visualizer_propagator_cache_and_parameter_changes() {
        use math_explorer::physics::quantum::evolve_state;

        let mut viz = SpinVisualizer::default();
        assert_eq!(viz.cached_key, None);
        assert_eq!(viz.scratch.len(), 2);

        viz.b_field = [1.0, 0.5, -0.5];
        viz.step(0.05);
        let key_initial = viz.cached_key;
        assert!(key_initial.is_some());
        let ptr_initial = viz.propagator.matrix.as_ptr();

        viz.step(0.05);
        assert_eq!(viz.cached_key, key_initial);
        assert_eq!(viz.propagator.matrix.as_ptr(), ptr_initial);

        viz.step(0.02);
        let key_dt_changed = viz.cached_key;
        assert_ne!(key_dt_changed, key_initial);

        viz.reset();
        viz.b_field = [2.0, 0.0, 0.0];
        let mut expected = QuantumState::spin_zero();
        let h_mat = (&viz.sx.matrix * Complex::new(2.0, 0.0)) * Complex::new(0.5, 0.0);
        let h_op = QuantumOperator::new(h_mat);
        for _ in 0..4 {
            viz.step(0.05);
            expected = evolve_state(&expected, &h_op, 0.05, 1.0).normalize();
        }
        assert_ne!(viz.cached_key, key_dt_changed);
        assert!((&viz.psi.vector - &expected.vector).norm() < 1e-12);
    }

    #[test]
    fn test_spin_visualizer_viewport_and_plot_export() {
        let mut viz = SpinVisualizer::default();
        let ctx = egui::Context::default();
        let _ = ctx.run(egui::RawInput::default(), |ctx| viz.show(ctx));
        assert!(!egui_plot::get_last_plot_datasets().is_empty());
        assert!(!egui_plot::get_last_plot_datasets_3d().is_empty());
        assert!(!egui_plot::get_last_plot_primitives().is_empty());
    }
}

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
