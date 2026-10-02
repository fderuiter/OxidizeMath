#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::accessibility::AccessibleTheoryHover;
use crate::framework::InteractiveTool;
use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};
use math_explorer::physics::chaos::lorenz::{LorenzBuilder, LorenzState, LorenzSystem};
use nalgebra::Vector3;
use std::collections::VecDeque;

pub struct AttractorPlotter {
    system: LorenzSystem,

    // Simulation Control
    paused: bool,
    simulation_speed: usize,
    dt: f64,
    diverged: bool,

    // Visualization
    history: VecDeque<Vector3<f64>>, // Store as Vector3 for easier math
    max_points: usize,

    camera: crate::framework::Camera3D,
}

impl Default for AttractorPlotter {
    fn default() -> Self {
        let initial_state = LorenzState::new(1.0, 1.0, 1.0);
        let system = LorenzBuilder::new().build(initial_state);

        Self {
            system,
            paused: false,
            simulation_speed: 5,
            dt: 0.01,
            diverged: false,
            history: VecDeque::with_capacity(2000),
            max_points: 2000,
            camera: crate::framework::Camera3D::new(0.0, 0.0, 1.0),
        }
    }
}

impl AttractorPlotter {
    #[must_use]
    #[allow(dead_code)]
    pub fn diverged(&self) -> bool {
        self.diverged
    }

    pub fn reset(&mut self) {
        let initial_state = LorenzState::new(1.0, 1.0, 1.0);
        // Preserve parameters if finite, but reset state
        let sigma = if self.system.sigma.is_finite() { self.system.sigma } else { 10.0 };
        let rho = if self.system.rho.is_finite() { self.system.rho } else { 28.0 };
        let beta = if self.system.beta.is_finite() { self.system.beta } else { 8.0 / 3.0 };

        self.system = LorenzBuilder::new()
            .sigma(sigma)
            .rho(rho)
            .beta(beta)
            .build(initial_state);

        self.history.clear();
        self.diverged = false;
        self.paused = false;
    }

    pub fn reset_parameters(&mut self) {
        self.system.sigma = 10.0;
        self.system.rho = 28.0;
        self.system.beta = 8.0 / 3.0;
        self.dt = 0.01;
        self.reset();
    }

    /// Projects 3D point to 2D screen space based on rotation
    fn project(&self, p: Vector3<f64>) -> [f64; 2] {
        // Center the attractor roughly. Lorenz attractor Z ranges approx 0-50.
        // Centering it makes rotation look more natural.
        let center_offset = Vector3::new(0.0, 0.0, 25.0);
        let p_centered = p - center_offset;

        self.camera.project(&[p_centered.x, p_centered.y, p_centered.z])
    }
}

impl InteractiveTool for AttractorPlotter {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable { self }
    fn name(&self) -> &'static str {
        "Attractor Plotter"
    }

    #[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
    fn show(&mut self, ctx: &egui::Context) {
        // --- Simulation ---
        if !self.paused && !self.diverged {
            for _ in 0..self.simulation_speed {
                let current_vec = self.system.state.vec;
                if !current_vec.x.is_finite()
                    || !current_vec.y.is_finite()
                    || !current_vec.z.is_finite()
                {
                    self.diverged = true;
                    self.paused = true;
                    break;
                }

                self.system.step(self.dt);

                let vec = self.system.state.vec;
                if !vec.x.is_finite() || !vec.y.is_finite() || !vec.z.is_finite() {
                    self.diverged = true;
                    self.paused = true;
                    break;
                }

                if self.history.len() >= self.max_points {
                    self.history.pop_front();
                }
                self.history.push_back(vec);
            }
            ctx.request_repaint();
        }

        // --- UI ---
        egui::SidePanel::left("attractor_controls").show(ctx, |ui| {
            ui.heading("Lorenz Attractor");
            ui.separator();

            ui.collapsing("Parameters", |ui| {
                if ui.add(egui::Slider::new(&mut self.system.sigma, 0.0..=50.0).text("Prandtl Number (σ)")).changed() {
                    self.diverged = false;
                }

                if ui.add(egui::Slider::new(&mut self.system.rho, 0.0..=100.0).text("Rayleigh Number (ρ)")).changed() {
                    self.diverged = false;
                }

                if ui.add(egui::Slider::new(&mut self.system.beta, 0.0..=10.0).text("Geometric Factor (β)")).changed() {
                    self.diverged = false;
                }
            });

            ui.collapsing("Simulation", |ui| {
                ui.horizontal(|ui| {
                    if ui
                        .button(if self.paused { "▶ Play" } else { "⏸ Pause" })
                        .clicked()
                    {
                        self.paused = !self.paused;
                        if !self.paused && self.diverged {
                            self.diverged = false;
                        }
                    }
                    if ui.button("↻ Reset").clicked() {
                        self.reset();
                    }
                });

                if ui.add(egui::Slider::new(&mut self.simulation_speed, 1..=50).text("Speed (steps/frame)")).changed() {
                    self.diverged = false;
                }

                if ui.add(egui::Slider::new(&mut self.dt, 0.001..=0.05).text("Time Step (dt)")).changed() {
                    self.diverged = false;
                }

                if ui.add(egui::Slider::new(&mut self.max_points, 100..=10000).text("Max Points")).changed() {
                    self.diverged = false;
                }
            });

            ui.collapsing("View", |ui| {
                self.camera.ui(ui);
            });

            ui.separator();
            let state = self.system.state.vec;
            ui.label(format!("X: {:.2}", state.x));
            ui.label(format!("Y: {:.2}", state.y));
            ui.label(format!("Z: {:.2}", state.z));
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            if self.diverged {
                ui.group(|ui| {
                    ui.horizontal(|ui| {
                        ui.label(
                            egui::RichText::new("⚠️ Solver Divergence Warning: Non-finite state detected (NaN or Inf). Simulation halted.")
                                .color(egui::Color32::RED)
                                .strong(),
                        );
                        if ui.button("↻ Reset Parameters").clicked() {
                            self.reset_parameters();
                        }
                        if ui.button("↻ Reset State").clicked() {
                            self.reset();
                        }
                    });
                });
            }

            let points: Vec<[f64; 2]> = self.history.iter().map(|p| self.project(*p)).collect();

            let response = Plot::new("attractor_plot")
                .data_aspect(1.0)
                .show(ui, |plot_ui| {
                    plot_ui.line(
                        Line::new("Trajectory", PlotPoints::new(points))
                            .color(egui::Color32::from_rgb(100, 200, 255)),
                    );
                })
                .response;
            
            self.camera.handle_interaction(&response, ui);
            
            response.accessible_theory_hover(&self.system);

            ui.label("Drag 'Yaw' and 'Pitch' in the side panel to rotate the view.");
        });
    }
}

inventory::submit! {
    crate::framework::ToolMetadata {
        name: "AttractorPlotter",
        domain: "chaos",
        tags: &[],
        build: || Box::new(AttractorPlotter::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for AttractorPlotter {
    fn theory_description(&self) -> String { "Theoretical context not available.".into() }
    fn phonetic_description(&self) -> String { "Theoretical context not available.".into() }
    fn theory_citation(&self) -> String { "Uncited".into() }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> { std::collections::HashMap::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attractor_divergence_detection() {
        let mut plotter = AttractorPlotter::default();
        assert!(!plotter.diverged());

        // Introduce a non-finite state
        plotter.system.state.vec.x = f64::NAN;

        let ctx = egui::Context::default();
        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            plotter.show(ctx);
        });

        assert!(plotter.diverged());
        assert!(plotter.paused);
        // Verify history contains no non-finite values
        assert!(plotter.history.iter().all(|v| v.x.is_finite() && v.y.is_finite() && v.z.is_finite()));

        // Reset clears diverged flag cleanly
        plotter.reset();
        assert!(!plotter.diverged());
    }

    #[test]
    fn test_attractor_reset_parameters() {
        let mut plotter = AttractorPlotter::default();
        plotter.system.sigma = f64::NAN;
        plotter.diverged = true;

        plotter.reset_parameters();
        assert!(!plotter.diverged());
        assert_eq!(plotter.system.sigma, 10.0);
    }
}
