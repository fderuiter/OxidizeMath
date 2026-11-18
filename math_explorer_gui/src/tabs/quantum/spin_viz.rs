use crate::accessibility::AccessibleTheoryHover;
use crate::framework::{InteractiveTool, PlaybackAction, PlaybackTransport};
use eframe::egui;
use egui_plot::{Line, Plot, PlotPoint, Points};
use math_explorer::physics::quantum::{evolve_state, spin, QuantumOperator, QuantumState};
use num_complex::Complex;

pub struct SpinVisualizer {
    psi: QuantumState,
    b_field: [f64; 3],
    time: f64,
    paused: bool,
    camera: crate::framework::Camera3D,
    sx: QuantumOperator,
    sy: QuantumOperator,
    sz: QuantumOperator,
}

impl Default for SpinVisualizer {
    fn default() -> Self {
        Self {
            psi: QuantumState::spin_zero(),
            b_field: [0.0, 0.0, 1.0],
            time: 0.0,
            paused: true,
            camera: crate::framework::Camera3D::new(0.5, 0.5, 1.0),
            sx: spin::sigma_x(),
            sy: spin::sigma_y(),
            sz: spin::sigma_z(),
        }
    }
}

impl SpinVisualizer {
    pub fn reset(&mut self) {
        self.psi = QuantumState::spin_zero();
        self.time = 0.0;
    }

    pub fn step_once(&mut self) {
        self.paused = true;
        self.step(0.05);
    }

    fn step(&mut self, dt: f64) {
        let [bx, by, bz] = self.b_field;
        let h_matrix = (&self.sx.matrix * Complex::new(bx, 0.0)
            + &self.sy.matrix * Complex::new(by, 0.0)
            + &self.sz.matrix * Complex::new(bz, 0.0))
            * Complex::new(0.5, 0.0);

        let hamiltonian = QuantumOperator::new(h_matrix);
        self.psi = evolve_state(&self.psi, &hamiltonian, dt, 1.0).normalize();
        self.time += dt;
    }

    fn project(&self, point: [f64; 3]) -> [f64; 2] {
        self.camera.project(&point)
    }

    fn project_circle(&self, axis: usize) -> [PlotPoint; 61] {
        let mut ring = [PlotPoint::new(0.0, 0.0); 61];
        for (i, pt) in ring.iter_mut().enumerate() {
            let theta = i as f64 * std::f64::consts::TAU / 60.0;
            let p3 = match axis {
                0 => [theta.cos(), theta.sin(), 0.0],
                1 => [0.0, theta.cos(), theta.sin()],
                _ => [theta.cos(), 0.0, theta.sin()],
            };
            let [x, y] = self.project(p3);
            *pt = PlotPoint::new(x, y);
        }
        ring
    }

    fn draw_controls(&mut self, ctx: &egui::Context, ex: f64, ey: f64, ez: f64) {
        egui::SidePanel::left("spin_controls").show(ctx, |ui| {
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

    fn draw_bloch_sphere(&mut self, ctx: &egui::Context, ex: f64, ey: f64, ez: f64) {
        egui::CentralPanel::default().show(ctx, |ui| {
            let equator = self.project_circle(0);
            let meridian = self.project_circle(1);
            let meridian2 = self.project_circle(2);

            let to_pt = |p: [f64; 3]| {
                let [x, y] = self.project(p);
                PlotPoint::new(x, y)
            };
            let origin = to_pt([0.0, 0.0, 0.0]);
            let end = to_pt([ex, ey, ez]);
            let vec_line = [origin, end];
            let x_axis = [origin, to_pt([1.2, 0.0, 0.0])];
            let y_axis = [origin, to_pt([0.0, 1.2, 0.0])];
            let z_axis = [origin, to_pt([0.0, 0.0, 1.2])];
            let tip = [end];

            let response = Plot::new("bloch_sphere")
                .data_aspect(1.0)
                .view_aspect(1.0)
                .show(ui, |plot_ui| {
                    plot_ui.line(Line::new("", &equator[..]).color(egui::Color32::GRAY));
                    plot_ui.line(Line::new("", &meridian[..]).color(egui::Color32::GRAY));
                    plot_ui.line(Line::new("", &meridian2[..]).color(egui::Color32::GRAY));
                    plot_ui.line(Line::new("", &x_axis[..]).color(egui::Color32::RED));
                    plot_ui.line(Line::new("", &y_axis[..]).color(egui::Color32::GREEN));
                    plot_ui.line(Line::new("", &z_axis[..]).color(egui::Color32::BLUE));
                    plot_ui.line(
                        Line::new("State", &vec_line[..])
                            .color(egui::Color32::YELLOW)
                            .width(3.0_f32),
                    );
                    plot_ui.points(
                        Points::new("", &tip[..])
                            .radius(5.0_f32)
                            .color(egui::Color32::YELLOW),
                    );
                })
                .response;

            self.camera.handle_interaction(&response, ui);
            response.accessible_theory_hover(&self.psi);
        });
    }
}

impl InteractiveTool for SpinVisualizer {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }
    fn name(&self) -> &'static str {
        "Spin Dynamics (Bloch Sphere)"
    }

    fn show(&mut self, ctx: &egui::Context) {
        if !self.paused {
            self.step(0.05);
            ctx.request_repaint();
        }

        let ex = self.sx.expectation_value(&self.psi).re;
        let ey = self.sy.expectation_value(&self.psi).re;
        let ez = self.sz.expectation_value(&self.psi).re;

        self.draw_controls(ctx, ex, ey, ez);
        self.draw_bloch_sphere(ctx, ex, ey, ez);
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
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Bloch Sphere Spin State",
    "domain": "quantum",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "theta": 1.047,
    "phi": 0.785,
    "magnetic_b": 1.0
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "spin_viz_starter",
        title: "Bloch Sphere Spin State",
        description: "|ψ⟩ = cos(θ/2)|0⟩ + e^(iφ)sin(θ/2)|1⟩: Qubit pure state spherical coordinates and Larmor precession.",
        category: "Quantum States",
        icon: "🌐",
        domain: "quantum",
        tool_name: Some("SpinVisualizer"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}
