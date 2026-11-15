#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::accessibility::AccessibleTheoryHover;
use crate::framework::InteractiveTool;
use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};
use math_explorer::physics::chaos::lorenz::{LorenzBuilder, LorenzState, LorenzSystem};
use nalgebra::Vector3;
use std::collections::VecDeque;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SolverExecutionState {
    Ready,
    Running,
    Paused,
    Diverged {
        invalid_vec: Vector3<f64>,
        diverged_at_step: usize,
    },
}

pub struct AttractorPlotter {
    system: LorenzSystem,

    // Simulation Control
    execution_state: SolverExecutionState,
    step_count: usize,
    simulation_speed: usize,
    dt: f64,

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
            execution_state: SolverExecutionState::Ready,
            step_count: 0,
            simulation_speed: 5,
            dt: 0.01,
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
        matches!(self.execution_state, SolverExecutionState::Diverged { .. })
    }

    #[must_use]
    pub fn is_paused(&self) -> bool {
        matches!(
            self.execution_state,
            SolverExecutionState::Paused | SolverExecutionState::Diverged { .. }
        )
    }

    #[must_use]
    pub fn is_finite_state(&self) -> bool {
        let v = self.system.state.vec;
        v.x.is_finite() && v.y.is_finite() && v.z.is_finite()
            && self.system.sigma.is_finite() && self.system.rho.is_finite()
            && self.system.beta.is_finite() && self.dt.is_finite()
    }

    pub fn try_resume(&mut self) -> Result<(), &'static str> {
        let v = self.system.state.vec;
        if !v.x.is_finite() || !v.y.is_finite() || !v.z.is_finite() {
            return Err("Cannot resume simulation: state coordinates are non-finite.");
        }
        if !self.system.sigma.is_finite() || !self.system.rho.is_finite() || !self.system.beta.is_finite() || !self.dt.is_finite() {
            return Err("Cannot resume simulation: system parameters are non-finite.");
        }
        self.execution_state = SolverExecutionState::Running;
        Ok(())
    }

    pub fn on_slider_changed(&mut self) {
        if self.diverged() && self.is_finite_state() {
            self.execution_state = SolverExecutionState::Ready;
        }
    }

    pub fn reset(&mut self) {
        let initial_state = LorenzState::new(1.0, 1.0, 1.0);
        let sigma = if self.system.sigma.is_finite() { self.system.sigma } else { 10.0 };
        let rho = if self.system.rho.is_finite() { self.system.rho } else { 28.0 };
        let beta = if self.system.beta.is_finite() { self.system.beta } else { 8.0 / 3.0 };
        if !self.dt.is_finite() || self.dt <= 0.0 {
            self.dt = 0.01;
        }

        self.system = LorenzBuilder::new().sigma(sigma).rho(rho).beta(beta).build(initial_state);
        self.history.clear();
        self.step_count = 0;
        self.execution_state = SolverExecutionState::Ready;
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
    fn domain(&self) -> &'static str {
        "chaos"
    }

    fn save_state(&self) -> Option<String> {
        #[derive(serde::Serialize, serde::Deserialize)]
        struct AttractorState {
            sigma: f64,
            rho: f64,
            beta: f64,
            dt: f64,
            simulation_speed: usize,
            paused: bool,
        }

        serde_json::to_string(&AttractorState {
            sigma: self.system.sigma,
            rho: self.system.rho,
            beta: self.system.beta,
            dt: self.dt,
            simulation_speed: self.simulation_speed,
            paused: self.is_paused(),
        })
        .ok()
    }

    fn load_state(&mut self, state: &str) {
        #[derive(serde::Serialize, serde::Deserialize)]
        struct AttractorState {
            sigma: f64,
            rho: f64,
            beta: f64,
            dt: f64,
            simulation_speed: usize,
            paused: bool,
        }

        if let Ok(s) = serde_json::from_str::<AttractorState>(state) {
            self.system.sigma = s.sigma;
            self.system.rho = s.rho;
            self.system.beta = s.beta;
            self.dt = s.dt;
            self.simulation_speed = s.simulation_speed;
            self.reset();
            if s.paused {
                self.execution_state = SolverExecutionState::Paused;
            }
        }
    }

    fn step_simulation(&mut self) {
        if !self.diverged() {
            if self.execution_state == SolverExecutionState::Ready {
                self.execution_state = SolverExecutionState::Running;
            }
            for _ in 0..self.simulation_speed {
                let current_vec = self.system.state.vec;
                if !current_vec.x.is_finite() || !current_vec.y.is_finite() || !current_vec.z.is_finite() {
                    self.execution_state = SolverExecutionState::Diverged { invalid_vec: current_vec, diverged_at_step: self.step_count };
                    break;
                }

                self.system.step(self.dt);
                self.step_count += 1;

                let vec = self.system.state.vec;
                if !vec.x.is_finite() || !vec.y.is_finite() || !vec.z.is_finite() {
                    self.execution_state = SolverExecutionState::Diverged { invalid_vec: vec, diverged_at_step: self.step_count };
                    break;
                }

                if self.history.len() >= self.max_points {
                    self.history.pop_front();
                }
                self.history.push_back(vec);
            }
        }
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let points: Vec<[f64; 2]> = self.history.iter().map(|p| self.project(*p)).collect();
        vec![("Trajectory".to_string(), points)]
    }

    #[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
    fn show(&mut self, ctx: &egui::Context) {
        // --- Simulation ---
        if !self.is_paused() {
            self.step_simulation();
            ctx.request_repaint();
        }

        // --- UI ---
        egui::SidePanel::left("attractor_controls").show(ctx, |ui| {
            ui.heading("Lorenz Attractor");
            ui.separator();

            ui.collapsing("Parameters", |ui| {
                if ui.add(egui::Slider::new(&mut self.system.sigma, 0.0..=50.0).text("Prandtl Number (σ)")).changed() {
                    self.on_slider_changed();
                }

                if ui.add(egui::Slider::new(&mut self.system.rho, 0.0..=100.0).text("Rayleigh Number (ρ)")).changed() {
                    self.on_slider_changed();
                }

                if ui.add(egui::Slider::new(&mut self.system.beta, 0.0..=10.0).text("Geometric Factor (β)")).changed() {
                    self.on_slider_changed();
                }
            });

            ui.collapsing("Simulation", |ui| {
                ui.horizontal(|ui| {
                    if ui
                        .button(if self.is_paused() { "▶ Play" } else { "⏸ Pause" })
                        .clicked()
                    {
                        if self.is_paused() {
                            let _ = self.try_resume();
                        } else {
                            self.execution_state = SolverExecutionState::Paused;
                        }
                    }
                    if ui.button("↻ Reset").clicked() {
                        self.reset();
                    }
                });

                if ui.add(egui::Slider::new(&mut self.simulation_speed, 1..=50).text("Speed (steps/frame)")).changed() {
                    self.on_slider_changed();
                }

                if ui.add(egui::Slider::new(&mut self.dt, 0.001..=0.05).text("Time Step (dt)")).changed() {
                    self.on_slider_changed();
                }

                if ui.add(egui::Slider::new(&mut self.max_points, 100..=10000).text("Max Points")).changed() {
                    self.on_slider_changed();
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
            if let SolverExecutionState::Diverged {
                invalid_vec,
                diverged_at_step,
            } = self.execution_state
            {
                ui.group(|ui| {
                    ui.horizontal(|ui| {
                        ui.label(
                            egui::RichText::new(format!(
                                "⚠️ Solver Divergence Warning: Non-finite state detected at step {} (coordinates: [{:.2}, {:.2}, {:.2}]). Simulation halted.",
                                diverged_at_step, invalid_vec.x, invalid_vec.y, invalid_vec.z
                            ))
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
        assert!(!plotter.is_paused());
        assert_eq!(plotter.execution_state, SolverExecutionState::Ready);

        // Introduce a non-finite state
        plotter.system.state.vec.x = f64::NAN;

        let ctx = egui::Context::default();
        let _ = ctx.run(egui::RawInput::default(), |ctx| plotter.show(ctx));

        assert!(plotter.diverged() && plotter.is_paused());
        match plotter.execution_state {
            SolverExecutionState::Diverged { invalid_vec, diverged_at_step } => {
                assert!(invalid_vec.x.is_nan() && diverged_at_step == 0);
            }
            _ => panic!("Expected Diverged execution state"),
        }

        // Verify history contains no non-finite values
        assert!(plotter.history.iter().all(|v| v.x.is_finite() && v.y.is_finite() && v.z.is_finite()));

        // Reset clears diverged flag cleanly and restores Ready
        plotter.reset();
        assert!(!plotter.diverged() && !plotter.is_paused());
        assert_eq!(plotter.execution_state, SolverExecutionState::Ready);
        assert_eq!(plotter.step_count, 0);
    }

    #[test]
    fn test_attractor_reset_parameters() {
        let mut plotter = AttractorPlotter::default();
        plotter.system.sigma = f64::NAN;
        plotter.execution_state = SolverExecutionState::Diverged {
            invalid_vec: Vector3::new(f64::NAN, 1.0, 1.0),
            diverged_at_step: 10,
        };

        plotter.reset_parameters();
        assert!(!plotter.diverged());
        assert_eq!(plotter.execution_state, SolverExecutionState::Ready);
        assert_eq!(plotter.step_count, 0);
        assert_eq!(plotter.system.sigma, 10.0);
    }

    #[test]
    fn test_attractor_transition_validation_non_finite_rejection() {
        let mut plotter = AttractorPlotter::default();
        plotter.system.state.vec.x = f64::NAN;

        let ctx = egui::Context::default();
        let _ = ctx.run(egui::RawInput::default(), |ctx| plotter.show(ctx));

        assert!(plotter.diverged() && plotter.is_paused());

        // Resume attempt must fail while coordinates are non-finite
        assert!(plotter.try_resume().is_err() && plotter.diverged());

        // Slider modifications must remain inert while state is non-finite
        plotter.system.sigma = 15.0;
        plotter.on_slider_changed();
        assert!(plotter.diverged());

        plotter.dt = 0.02;
        plotter.on_slider_changed();
        assert!(plotter.diverged());

        // Explicit reset restores Ready state and finite coordinates
        plotter.reset();
        assert_eq!(plotter.execution_state, SolverExecutionState::Ready);
        assert!(!plotter.diverged() && !plotter.is_paused() && plotter.is_finite_state());

        // Resume now succeeds
        assert!(plotter.try_resume().is_ok());
        assert_eq!(plotter.execution_state, SolverExecutionState::Running);
    }

    #[test]
    fn test_attractor_parameter_sanitization_and_rejection() {
        let mut plotter = AttractorPlotter::default();
        plotter.system.sigma = f64::NAN;
        plotter.dt = f64::NAN;

        // try_resume must reject non-finite parameters
        assert!(plotter.try_resume().is_err());

        // Reset sanitizes non-finite parameters to default values
        plotter.reset();
        assert_eq!(plotter.system.sigma, 10.0);
        assert_eq!(plotter.dt, 0.01);
        assert_eq!(plotter.execution_state, SolverExecutionState::Ready);
    }

    #[test]
    fn test_attractor_step_counting_and_divergence_metadata() {
        let mut plotter = AttractorPlotter::default();
        assert_eq!(plotter.step_count, 0);

        plotter.step_simulation();
        assert_eq!(plotter.execution_state, SolverExecutionState::Running);
        assert_eq!(plotter.step_count, plotter.simulation_speed);

        let datasets = plotter.export_datasets();
        let options = egui_plot::export::ExportOptions::default();
        if let Ok(png_data) = egui_plot::export::export_custom_png(&datasets, &options) {
            let _ = std::fs::write("/tmp/attractor_plotter_trajectory.png", png_data);
        }

        // Force divergence during stepping
        plotter.system.state.vec.y = f64::INFINITY;
        plotter.step_simulation();

        assert!(plotter.diverged());
        match plotter.execution_state {
            SolverExecutionState::Diverged { invalid_vec, diverged_at_step } => {
                assert!(invalid_vec.y.is_infinite());
                assert_eq!(diverged_at_step, plotter.simulation_speed);
            }
            _ => panic!("Expected Diverged execution state"),
        }
    }
}
