use crate::accessibility::{AccessibleHoverText, AccessibleTheoryHover};
use eframe::egui;
use egui_plot::{Line, Plot, PlotPoint, Points};
use scientific_metadata::theory::TheoryDescribable;
use math_explorer::physics::quantum::{evolve_state, spin, QuantumOperator, QuantumState};
use num_complex::Complex;

use crate::framework::InteractiveTool;

pub struct SpinVisualizer {
    psi: QuantumState,
    b_field: [f64; 3], // [Bx, By, Bz]
    time: f64,
    paused: bool,
    camera: crate::framework::Camera3D,
    sx: QuantumOperator,
    sy: QuantumOperator,
    sz: QuantumOperator,
}

impl Default for SpinVisualizer {
    fn default() -> Self {
        // Initial state |0> (spin up)
        let psi = QuantumState::spin_zero();

        Self {
            psi,
            b_field: [0.0, 0.0, 1.0], // Default B field along Z
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
        // Reset to |0>
        self.psi = QuantumState::spin_zero();
        self.time = 0.0;
    }

    fn step(&mut self, dt: f64) {
        let bx = self.b_field[0];
        let by = self.b_field[1];
        let bz = self.b_field[2];

        // H = 0.5 * (Bx*Sx + By*Sy + Bz*Sz)
        let h_matrix = (&self.sx.matrix * Complex::new(bx, 0.0)
            + &self.sy.matrix * Complex::new(by, 0.0)
            + &self.sz.matrix * Complex::new(bz, 0.0))
            * Complex::new(0.5, 0.0);

        let hamiltonian = QuantumOperator::new(h_matrix);

        // Evolve
        self.psi = evolve_state(&self.psi, &hamiltonian, dt, 1.0);
        self.time += dt;
        self.psi = self.psi.normalize();
    }

    fn project(&self, point: [f64; 3]) -> [f64; 2] {
        self.camera.project(&point)
    }
}

impl InteractiveTool for SpinVisualizer {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable { self }
    fn name(&self) -> &'static str {
        "Spin Dynamics (Bloch Sphere)"
    }

    #[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
    fn show(&mut self, ctx: &egui::Context) {
        if !self.paused {
            self.step(0.05);
            ctx.request_repaint();
        }

        // Calculate expectation values once per frame
        let ex = self.sx.expectation_value(&self.psi).re;
        let ey = self.sy.expectation_value(&self.psi).re;
        let ez = self.sz.expectation_value(&self.psi).re;

        egui::SidePanel::left("spin_controls").show(ctx, |ui| {
            ui.heading("Controls");
            ui.separator();

            ui.group(|ui| {
                ui.add(egui::Slider::new(&mut self.b_field[0], -5.0..=5.0).text("Magnetic Field B - Bx"));
                ui.add(egui::Slider::new(&mut self.b_field[1], -5.0..=5.0).text("By"));
                ui.add(egui::Slider::new(&mut self.b_field[2], -5.0..=5.0).text("Bz"));
            });

            ui.separator();
            ui.horizontal(|ui| {
                if ui
                    .button(if self.paused { "▶ Play" } else { "⏸ Pause" })
                    .clicked()
                {
                    self.paused = !self.paused;
                }
                if ui
                    .button("🔄 Reset")
                    .accessible_hover_text(format!(
                        "Reset to {}",
                        QuantumState::spin_zero().theory_description()
                    ))
                    .clicked()
                {
                    self.reset();
                }
            });

            ui.separator();
            ui.heading("View");
            self.camera.ui(ui);

            ui.separator();

            ui.label(format!("Ex (x): {:.3}", ex));
            ui.label(format!("Ey (y): {:.3}", ey));
            ui.label(format!("Ez (z): {:.3}", ez));
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            // Generate sphere wireframe with fixed stack arrays
            let mut equator = [PlotPoint::new(0.0, 0.0); 61];
            for (i, pt) in equator.iter_mut().enumerate() {
                let theta = i as f64 * std::f64::consts::TAU / 60.0;
                let [x, y] = self.project([theta.cos(), theta.sin(), 0.0]);
                *pt = PlotPoint::new(x, y);
            }

            let mut meridian = [PlotPoint::new(0.0, 0.0); 61];
            for (i, pt) in meridian.iter_mut().enumerate() {
                let theta = i as f64 * std::f64::consts::TAU / 60.0;
                let [x, y] = self.project([0.0, theta.cos(), theta.sin()]);
                *pt = PlotPoint::new(x, y);
            }

            let mut meridian2 = [PlotPoint::new(0.0, 0.0); 61];
            for (i, pt) in meridian2.iter_mut().enumerate() {
                let theta = i as f64 * std::f64::consts::TAU / 60.0;
                let [x, y] = self.project([theta.cos(), 0.0, theta.sin()]);
                *pt = PlotPoint::new(x, y);
            }

            // Vector & Axes
            let origin = {
                let [x, y] = self.project([0.0, 0.0, 0.0]);
                PlotPoint::new(x, y)
            };
            let end = {
                let [x, y] = self.project([ex, ey, ez]);
                PlotPoint::new(x, y)
            };
            let vec_line = [origin, end];

            let x_axis = [
                origin,
                {
                    let [x, y] = self.project([1.2, 0.0, 0.0]);
                    PlotPoint::new(x, y)
                },
            ];
            let y_axis = [
                origin,
                {
                    let [x, y] = self.project([0.0, 1.2, 0.0]);
                    PlotPoint::new(x, y)
                },
            ];
            let z_axis = [
                origin,
                {
                    let [x, y] = self.project([0.0, 0.0, 1.2]);
                    PlotPoint::new(x, y)
                },
            ];
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

                    // Draw state vector (only named line in legend)
                    plot_ui.line(
                        Line::new("State", &vec_line[..])
                            .color(egui::Color32::YELLOW)
                            .width(3.0_f32),
                    );

                    // Draw point at tip
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

// [cite:quantum_mechanics]
// module = "quantum"


inventory::submit! {
    crate::framework::ToolMetadata {
        name: "SpinVisualizer",
        domain: "quantum",
        tags: &[],
        build: || Box::new(SpinVisualizer::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for SpinVisualizer {
    fn theory_description(&self) -> String { "Theoretical context not available.".into() }
    fn phonetic_description(&self) -> String { "Theoretical context not available.".into() }
    fn theory_citation(&self) -> String { "Uncited".into() }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> { std::collections::HashMap::new() }
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

        viz.step(0.1);
        assert!(viz.time > 0.0);

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
