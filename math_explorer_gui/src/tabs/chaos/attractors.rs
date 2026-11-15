#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::accessibility::AccessibleTheoryHover;
use crate::framework::InteractiveTool;
use eframe::egui::{self, Color32, CornerRadius, Stroke, Vec2};
use egui_plot::{Callout, Line, Plot, PlotPoints};
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

    pub show_annotations: bool,
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
            show_annotations: true,
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
            #[serde(default = "default_annotations_enabled")]
            show_annotations: bool,
        }

        fn default_annotations_enabled() -> bool {
            true
        }

        serde_json::to_string(&AttractorState {
            sigma: self.system.sigma,
            rho: self.system.rho,
            beta: self.system.beta,
            dt: self.dt,
            simulation_speed: self.simulation_speed,
            paused: self.is_paused(),
            show_annotations: self.show_annotations,
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
            #[serde(default = "default_annotations_enabled")]
            show_annotations: bool,
        }

        fn default_annotations_enabled() -> bool {
            true
        }

        if let Ok(s) = serde_json::from_str::<AttractorState>(state) {
            self.system.sigma = s.sigma;
            self.system.rho = s.rho;
            self.system.beta = s.beta;
            self.dt = s.dt;
            self.simulation_speed = s.simulation_speed;
            self.show_annotations = s.show_annotations;
            self.reset();
            if s.paused {
                self.execution_state = SolverExecutionState::Paused;
            }
        }
    }

    fn apply_preset_json(&mut self, _json_str: &str) -> Result<(), String> {
        self.reset_parameters();
        self.execution_state = SolverExecutionState::Running;
        Ok(())
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
            if ui.button("🎨 Starter Gallery").clicked() {
                let cur = ctx.data(|d| d.get_temp::<bool>(egui::Id::new("SHOW_STARTER_GALLERY")).unwrap_or(false));
                ctx.data_mut(|d| d.insert_temp(egui::Id::new("SHOW_STARTER_GALLERY"), !cur));
            }
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
            ui.checkbox(&mut self.show_annotations, "Show Fixed Points");

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

                    if self.show_annotations
                        && self.system.rho > 1.0
                        && self.system.beta > 0.0
                        && self.system.rho.is_finite()
                        && self.system.beta.is_finite()
                    {
                        let x = (self.system.beta * (self.system.rho - 1.0)).sqrt();
                        let z = self.system.rho - 1.0;
                        if x.is_finite() && z.is_finite() {
                            let c_pos = self.project(Vector3::new(x, x, z));
                            let c_neg = self.project(Vector3::new(-x, -x, z));

                            let visuals = plot_ui.ctx().style().visuals.clone();
                            let bg_fill = Color32::from_rgba_unmultiplied(
                                visuals.window_fill().r(),
                                visuals.window_fill().g(),
                                visuals.window_fill().b(),
                                230,
                            );
                            let frame_stroke = visuals.widgets.noninteractive.bg_stroke;

                            let callout_pos = Callout::new(
                                c_pos,
                                format!("Fixed Point C+ ({:.1}, {:.1}, {:.1})", x, x, z),
                            )
                            .offset(Vec2::new(35.0, -25.0))
                            .leader_stroke(Stroke::new(1.5_f32, Color32::from_rgb(231, 76, 60)))
                            .frame(bg_fill, frame_stroke)
                            .corner_radius(CornerRadius::same(4))
                            .text_color(visuals.text_color());

                            let callout_neg = Callout::new(
                                c_neg,
                                format!("Fixed Point C- ({:.1}, {:.1}, {:.1})", -x, -x, z),
                            )
                            .offset(Vec2::new(-35.0, 25.0))
                            .leader_stroke(Stroke::new(1.5_f32, Color32::from_rgb(155, 89, 182)))
                            .frame(bg_fill, frame_stroke)
                            .corner_radius(CornerRadius::same(4))
                            .text_color(visuals.text_color());

                            plot_ui.callout(callout_pos);
                            plot_ui.callout(callout_neg);
                        }
                    }
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
mod attractor_tests;
