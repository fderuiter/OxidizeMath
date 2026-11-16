use crate::async_sim::unified::{UnifiedModel, UnifiedSimTool};
use crate::async_sim::{SimCommand, StateSnapshot};
use eframe::egui::Color32;
use math_commons::math_kernel::types::flatten_2d_index;
use math_commons::math_kernel::colormap::heatmap_color;
use scientific_metadata::theory::{ParameterConstraint, TheoryDescribable};
use math_explorer::physics::fluid_dynamics::lattice_boltzmann::{
    BgkCollision, LatticeBoltzmannD2Q9,
};
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

#[cfg_attr(any(), verified(opt_out = "gui_tool"))]
pub struct LbmUnified {
    solver: LatticeBoltzmannD2Q9<BgkCollision>,
    step_count: usize,
    diverged: bool,
    diagnostic_report: Option<crate::async_sim::recovery::DiagnosticReport>,
    last_valid_pixels: RefCell<Vec<Color32>>,
}

impl UnifiedModel for LbmUnified {
    fn domain() -> &'static str {
        "fluid_dynamics"
    }

    fn new(params: &HashMap<String, f64>) -> Self {
        let viscosity = *params.get("viscosity").unwrap_or(&0.02);
        let width = 100;
        let height = 50;
        let tau = 3.0 * viscosity + 0.5;

        let mut solver = LatticeBoltzmannD2Q9::new(width, height, tau);
        solver.set_inlet(0, 20, 5, 10, 0.1, 0.0);

        let center_x = 30;
        let center_y = 25;
        let radius = 5;
        for y in 0..height {
            for x in 0..width {
                let dx = x as i32 - center_x;
                let dy = y as i32 - center_y;
                if dx * dx + dy * dy <= radius * radius {
                    solver.set_obstacle(x, y, true);
                }
            }
        }

        Self {
            solver,
            step_count: 0,
            diverged: false,
            diagnostic_report: None,
            last_valid_pixels: RefCell::new(Vec::new()),
        }
    }

    fn step(&mut self, params: &HashMap<String, f64>) {
        if self.diverged {
            return;
        }
        self.step_count += 1;
        let viscosity = *params.get("viscosity").unwrap_or(&0.02);

        // Finite-state parameter check
        if !viscosity.is_finite() || viscosity <= 0.0 {
            self.trigger_divergence(viscosity, "viscosity");
            return;
        }

        let tau = 3.0 * viscosity + 0.5;
        if !tau.is_finite() || tau <= 0.5 {
            self.trigger_divergence(viscosity, "viscosity (tau <= 0.5)");
            return;
        }

        self.solver.collision_model.tau = tau;
        self.solver.set_inlet(0, 0, 2, self.solver.height(), 0.1, 0.0);
        self.solver.step();

        // Evaluate f64::is_finite() during every numerical step
        let is_finite = self.solver.state.velocity_x().iter().all(|v| v.is_finite())
            && self.solver.state.velocity_y().iter().all(|v| v.is_finite())
            && self.solver.state.density().iter().all(|d| d.is_finite());

        if !is_finite {
            self.trigger_divergence(viscosity, "velocity (u)");
        }
    }

    fn get_snapshot(&self) -> StateSnapshot {
        let width = self.solver.width();
        let height = self.solver.height();

        let pixels = if self.diverged && !self.last_valid_pixels.borrow().is_empty() {
            self.last_valid_pixels.borrow().clone()
        } else {
            let mut px = vec![Color32::BLACK; width * height];
            for y in 0..height {
                for x in 0..width {
                    let pixel_idx = flatten_2d_index(x, y, width);
                    if self.solver.is_obstacle(x, height - 1 - y) {
                        px[pixel_idx] = Color32::from_gray(50);
                    } else {
                        let v = self.solver.get_velocity_magnitude(x, height - 1 - y);
                        let normalized_v = if v.is_finite() {
                            (v * 1000.0 / 255.0).clamp(0.0, 1.0)
                        } else {
                            0.0
                        };
                        let rgb = heatmap_color(normalized_v);
                        px[pixel_idx] = Color32::from_rgb(rgb[0], rgb[1], rgb[2]);
                    }
                }
            }
            if !self.diverged {
                *self.last_valid_pixels.borrow_mut() = px.clone();
            }
            px
        };

        StateSnapshot {
            width,
            height,
            pixels: Arc::new(std::sync::RwLock::new(pixels)),
            custom_data: Vec::new(),
            structured_data: None,
        }
    }

    fn process_command(&mut self, cmd: SimCommand, params: &HashMap<String, f64>) {
        match cmd {
            SimCommand::Reset => {
                *self = Self::new(params);
            }
            SimCommand::Custom(ref action) if action == "ClampParameters" => {
                let mut clamped_params = params.clone();
                Self::clamp_parameters(&mut clamped_params);
                *self = Self::new(&clamped_params);
            }
            SimCommand::ApplyBrush { cx, cy, r, is_obstacle } => {
                let width = self.solver.width() as i32;
                let height = self.solver.height() as i32;
                for y in (cy - r)..=(cy + r) {
                    for x in (cx - r)..=(cx + r) {
                        if x >= 0 && x < width && y >= 0 && y < height && (x - cx) * (x - cx) + (y - cy) * (y - cy) <= r * r {
                            self.solver.set_obstacle(x as usize, y as usize, is_obstacle);
                        }
                    }
                }
            }
            SimCommand::ClearObstacles => {
                self.solver.clear_obstacles();
            }
            SimCommand::Custom(ref action) if action == "Clear Obstacles" => {
                self.solver.clear_obstacles();
            }
            _ => {}
        }
    }

    fn custom_actions() -> Vec<&'static str> {
        vec!["Clear Obstacles"]
    }

    fn diverged(&self) -> bool {
        self.diverged
    }

    fn diagnostic_report(&self) -> Option<crate::async_sim::recovery::DiagnosticReport> {
        self.diagnostic_report.clone()
    }

    fn default_parameters() -> HashMap<String, f64> {
        let mut map = HashMap::new();
        map.insert("viscosity".to_string(), 0.02);
        map
    }

    fn clamp_parameters(params: &mut HashMap<String, f64>) {
        if let Some(v) = params.get_mut("viscosity") {
            *v = if !v.is_finite() || *v < 0.01 { 0.02 } else { v.clamp(0.01, 0.5) };
        }
    }

    fn tool_source() -> &'static str {
        "LatticeBoltzmannTool"
    }

    fn parameters() -> HashMap<String, ParameterConstraint> {
        let mut map = HashMap::new();
        map.insert("viscosity".to_string(), ParameterConstraint { min: 0.00333, max: 0.5, step: 0.01 });
        map
    }

    fn name() -> &'static str {
        "Lattice Boltzmann (Demo)"
    }
}

impl LbmUnified {
    fn trigger_divergence(&mut self, viscosity: f64, var_name: &str) {
        self.diverged = true;
        let t_div = self.step_count as f64 * 0.01;
        let report = crate::async_sim::recovery::DiagnosticReport::new(
            t_div,
            self.step_count,
            var_name,
            format!(
                "Lattice Boltzmann numerical divergence: non-finite state encountered in {} at step {}",
                var_name, self.step_count
            ),
        );
        crate::async_sim::recovery::emit_divergence_telemetry(
            "LatticeBoltzmannTool",
            t_div,
            self.step_count,
            var_name,
            "Lattice Boltzmann (Demo)",
            viscosity,
        );
        self.diagnostic_report = Some(report);
    }
}

inventory::submit! {
    crate::framework::ToolMetadata {
        name: "LatticeBoltzmannTool",
        domain: "fluid_dynamics",
        tags: &[],
        build: || Box::new(UnifiedSimTool::<LbmUnified>::new()),
    }
}

impl TheoryDescribable for LbmUnified {
    fn theory_description(&self) -> String {
        LatticeBoltzmannD2Q9::<BgkCollision>::new(1, 1, 1.0).theory_description()
    }
    
    fn phonetic_description(&self) -> String {
        LatticeBoltzmannD2Q9::<BgkCollision>::new(1, 1, 1.0).phonetic_description()
    }
    
    fn theory_citation(&self) -> String {
        LatticeBoltzmannD2Q9::<BgkCollision>::new(1, 1, 1.0).theory_citation()
    }
    
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        LatticeBoltzmannD2Q9::<BgkCollision>::new(1, 1, 1.0).available_descriptions()
    }
}
