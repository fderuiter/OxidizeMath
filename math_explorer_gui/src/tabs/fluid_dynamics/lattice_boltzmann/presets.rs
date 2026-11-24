//! Starter gallery presets for [`super::LatticeBoltzmannTool`] (cylinder baseline,
//! bluff-plate wake shedding, and minimum-drag airfoil challenge).

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::super::lbm_atlas::LbmCartographer;
use super::super::lbm_fairing::FairingStamp;
use super::super::lbm_fairing::{CHALLENGE_LBM_FAIRING, LBM_REF_DIAMETER};
use super::runner::sync_solver_obstacles;
use super::state::{
    bluff_plate_obstacle_grid, count_solid_cells, cylinder_obstacle_grid, encode_obstacle_mask,
    enforce_area_budget, streamlined_airfoil_obstacle_grid, LbmToolState, LbmViewMode,
    AREA_BUDGET_MAX, AREA_BUDGET_MIN, DEFAULT_AREA_BUDGET, DEFAULT_BRUSH_RADIUS,
    DEFAULT_INFLOW_SPEED, DEFAULT_STEPS_PER_FRAME, DEFAULT_VISCOSITY, INFLOW_SPEED_MAX,
    INFLOW_SPEED_MIN, LBM_CELL_COUNT, LBM_HEIGHT, LBM_INLET_GUARD_COLS, LBM_WIDTH, VISCOSITY_MAX,
    VISCOSITY_MIN,
};
use super::LatticeBoltzmannTool;
use crate::accessibility::AccessibleHoverText;
use crate::async_sim::atlas::{AtlasTeleportEvent, RegimeCartographer};
use crate::async_sim::SimCommand;
use crate::framework::recovery::{DiagnosticReport, FiniteStateGuard};
use crate::framework::InteractiveTool;
use eframe::egui;
use math_commons::math_kernel::types::flatten_2d_index;
use math_explorer::physics::fluid_dynamics::lattice_boltzmann::{
    stamp_naca4_airfoil, stamp_venturi_nozzle, BgkCollision, LatticeBoltzmannD2Q9, Naca4Airfoil,
};
use scientific_metadata::theory::{MaturityTier, ParameterConstraint, TheoryDescribable};
use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

impl LatticeBoltzmannTool {
    pub(crate) fn reset_parameters_and_flow(&mut self) {
        self.viscosity = DEFAULT_VISCOSITY;
        self.inflow_speed = DEFAULT_INFLOW_SPEED;
        self.steps_per_frame = DEFAULT_STEPS_PER_FRAME;
        self.brush_radius = DEFAULT_BRUSH_RADIUS;
        self.area_budget = DEFAULT_AREA_BUDGET;
        self.enforce_challenge_invariants_and_score();
        self.push_shared_config();
        self.reset_flow_field();
    }

    pub(super) fn current_tool_state(&self) -> LbmToolState {
        LbmToolState {
            viscosity: self.viscosity,
            inflow_speed: self.inflow_speed,
            steps_per_frame: self.steps_per_frame,
            brush_radius: self.brush_radius,
            area_budget: self.area_budget,
            view_mode: self.view_mode,
            obstacle_mask: encode_obstacle_mask(&self.obstacles),
            mask_hex: None,
            obstacles_rle: None,
        }
    }

    pub(super) fn build_wind_tunnel_cartographer(
        &self,
        axis_keys: (&'static str, &'static str),
    ) -> Option<Arc<dyn RegimeCartographer>> {
        let char_len = if self.telemetry.frontal_height > 0 {
            self.telemetry.frontal_height as f64
        } else {
            LBM_REF_DIAMETER
        };
        let theory = self.theory_parameters();
        LbmCartographer::with_theory_and_char_length(&theory, char_len, axis_keys)
            .map(|c| Arc::new(c) as Arc<dyn RegimeCartographer>)
    }

    pub(super) fn apply_wind_tunnel_teleport(&mut self, ev: AtlasTeleportEvent) {
        if self.active_challenge == Some(CHALLENGE_LBM_FAIRING) {
            return;
        }
        for (key, val) in [(ev.x_key, ev.x_val), (ev.y_key, ev.y_val)] {
            if !val.is_finite() {
                continue;
            }
            match key {
                "viscosity" => self.viscosity = val.clamp(VISCOSITY_MIN, VISCOSITY_MAX),
                "inlet_velocity" | "inflow_speed" => {
                    self.inflow_speed = val.clamp(INFLOW_SPEED_MIN, INFLOW_SPEED_MAX);
                }
                _ => {}
            }
        }
        self.push_shared_config();
        self.reset_flow_field();
        let _ = self.worker.send_command(SimCommand::Start);
    }

    pub(super) fn build_snapshot_solver(&self) -> LatticeBoltzmannD2Q9<BgkCollision> {
        let state_clone = self
            .shared_lattice
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        let tau = 3.0 * self.viscosity.clamp(VISCOSITY_MIN, VISCOSITY_MAX) + 0.5;
        let mut solver = LatticeBoltzmannD2Q9::new(LBM_WIDTH, LBM_HEIGHT, tau);
        if state_clone.width() == LBM_WIDTH
            && state_clone.height() == LBM_HEIGHT
            && state_clone.density().iter().all(|v| v.is_finite())
        {
            solver.state = state_clone;
        } else {
            solver.set_inlet(0, 0, LBM_WIDTH, LBM_HEIGHT, self.inflow_speed.max(0.0), 0.0);
        }
        sync_solver_obstacles(&mut solver, &self.obstacles);
        solver
    }

    pub(crate) fn apply_fairing_stamp(&mut self, stamp: FairingStamp) {
        let mut mask = [false; LBM_CELL_COUNT];
        stamp.write_mask(&mut mask);
        self.apply_obstacle_preset(mask.to_vec());
    }

    pub(crate) fn apply_naca_2412_preset(&mut self, aoa_deg: f64) {
        let mut temp = LatticeBoltzmannD2Q9::<BgkCollision>::new(LBM_WIDTH, LBM_HEIGHT, 0.56);
        let airfoil = Naca4Airfoil::from_code(
            2412,
            26.0,
            0.32 * LBM_WIDTH as f64,
            0.5 * LBM_HEIGHT as f64,
            aoa_deg,
        );
        stamp_naca4_airfoil(&mut temp, &airfoil);
        let grid = Self::extract_solver_obstacles(&temp);
        self.apply_obstacle_preset(grid);
    }

    pub(crate) fn apply_venturi_preset(&mut self) {
        let mut temp = LatticeBoltzmannD2Q9::<BgkCollision>::new(LBM_WIDTH, LBM_HEIGHT, 0.56);
        stamp_venturi_nozzle(&mut temp, 0.44);
        let grid = Self::extract_solver_obstacles(&temp);
        if self.active_challenge.is_none() {
            let needed = count_solid_cells(&grid);
            self.area_budget = self
                .area_budget
                .max(needed)
                .clamp(AREA_BUDGET_MIN, AREA_BUDGET_MAX);
        }
        self.apply_obstacle_preset(grid);
    }

    fn extract_solver_obstacles(solver: &LatticeBoltzmannD2Q9<BgkCollision>) -> Vec<bool> {
        let mut grid = vec![false; LBM_CELL_COUNT];
        for y in 0..LBM_HEIGHT.min(solver.height()) {
            for x in LBM_INLET_GUARD_COLS..LBM_WIDTH.min(solver.width()) {
                grid[flatten_2d_index(x, y, LBM_WIDTH)] = solver.is_obstacle(x, y);
            }
        }
        grid
    }

    pub(crate) fn enforce_current_budget(&mut self) {
        enforce_area_budget(&mut self.obstacles, self.area_budget);
        self.enforce_challenge_invariants_and_score();
        self.sync_obstacles_to_worker();
    }

    pub(crate) fn apply_obstacle_preset(&mut self, mut grid: Vec<bool>) {
        enforce_area_budget(&mut grid, self.area_budget);
        self.obstacles = grid;
        self.enforce_challenge_invariants_and_score();
        self.mask_generation = self.mask_generation.wrapping_add(1);
        {
            let mut cfg = self
                .shared_config
                .write()
                .unwrap_or_else(|e| e.into_inner());
            cfg.viscosity = self.viscosity;
            cfg.inflow_speed = self.inflow_speed;
            cfg.obstacles.clone_from(&self.obstacles);
            cfg.mask_generation = self.mask_generation;
        }
        self.reset_flow_field();
    }

    pub(super) fn draw_profile_buttons(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.label("Tunnel Profiles:");
        ui.horizontal_wrapped(|ui| {
            let mut applied = false;
            for (id, label, hover) in [
                (
                    0_u8,
                    "● Cylinder",
                    "Load circular cylinder vortex-street benchmark",
                ),
                (
                    1,
                    "▮ Bluff Plate",
                    "Load high-drag flat bluff plate obstacle",
                ),
                (
                    2,
                    "➤ Min-Drag Airfoil",
                    "Load streamlined airfoil benchmark",
                ),
                (
                    3,
                    "✈ NACA 2412",
                    "Stamp cambered NACA 2412 airfoil at 8° AoA",
                ),
                (4, "⏳ Venturi", "Stamp converging-diverging Venturi nozzle"),
                (
                    7,
                    "✕ Clear",
                    "Remove all solid obstacles from the wind tunnel",
                ),
            ] {
                if ui.button(label).accessible_hover_text(hover).clicked() {
                    self.apply_profile_button_id(id);
                    applied = true;
                }
            }
            for stamp in [FairingStamp::TeardropFairing, FairingStamp::SymmetricWedge] {
                if ui
                    .button(format!("♦ {}", stamp.label()))
                    .accessible_hover_text("Stamp fairing geometry around protected R_core = 4")
                    .clicked()
                {
                    self.apply_fairing_stamp(stamp);
                    applied = true;
                }
            }
            if applied {
                ctx.request_repaint();
            }
        });
    }

    fn apply_profile_button_id(&mut self, id: u8) {
        match id {
            0 => self.apply_obstacle_preset(cylinder_obstacle_grid()),
            1 => self.apply_obstacle_preset(bluff_plate_obstacle_grid()),
            2 => self.apply_obstacle_preset(streamlined_airfoil_obstacle_grid()),
            3 => self.apply_naca_2412_preset(8.0),
            4 => self.apply_venturi_preset(),
            _ => self.apply_obstacle_preset(vec![false; LBM_CELL_COUNT]),
        }
    }
}

impl FiniteStateGuard for LatticeBoltzmannTool {
    fn is_finite_state(&self) -> bool {
        self.has_finite_parameters() && !self.worker.state().is_faulted()
    }

    fn diagnostic_report(&self) -> Option<&DiagnosticReport> {
        self.worker.diagnostic_report()
    }

    fn diverged(&self) -> bool {
        !FiniteStateGuard::is_finite_state(self) || self.worker.diagnostic_report().is_some()
    }

    fn reset_parameters(&mut self) {
        self.reset_parameters_and_flow();
    }
}

impl TheoryDescribable for LatticeBoltzmannTool {
    fn maturity_tier(&self) -> MaturityTier {
        MaturityTier::Certified
    }

    fn theory_description(&self) -> String {
        LatticeBoltzmannD2Q9::<BgkCollision>::new(1, 1, 1.0).theory_description()
    }

    fn phonetic_description(&self) -> String {
        LatticeBoltzmannD2Q9::<BgkCollision>::new(1, 1, 1.0).phonetic_description()
    }

    fn theory_citation(&self) -> String {
        LatticeBoltzmannD2Q9::<BgkCollision>::new(1, 1, 1.0).theory_citation()
    }

    fn available_descriptions(&self) -> HashMap<String, String> {
        LatticeBoltzmannD2Q9::<BgkCollision>::new(1, 1, 1.0).available_descriptions()
    }

    fn theory_parameters(&self) -> HashMap<String, ParameterConstraint> {
        HashMap::from([
            (
                "viscosity".to_string(),
                ParameterConstraint {
                    min: VISCOSITY_MIN,
                    max: VISCOSITY_MAX,
                    step: 0.005,
                },
            ),
            (
                "inflow_speed".to_string(),
                ParameterConstraint {
                    min: INFLOW_SPEED_MIN,
                    max: INFLOW_SPEED_MAX,
                    step: 0.005,
                },
            ),
            (
                "inlet_velocity".to_string(),
                ParameterConstraint {
                    min: 0.015,
                    max: INFLOW_SPEED_MAX,
                    step: 0.005,
                },
            ),
        ])
    }
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "LBM Flow Simulation",
    "domain": "fluid_dynamics",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "viscosity": 0.02
  }
}"#;

fn build_preset_envelope(title: &str, state: &LbmToolState) -> String {
    serde_json::json!({
        "metadata": {
            "schema_version": "1.0",
            "title": title,
            "domain": "fluid_dynamics",
            "timestamp": "2026-11-22T00:00:00Z"
        },
        "config": state,
    })
    .to_string()
}

fn bluff_plate_preset_json() -> &'static str {
    static JSON: OnceLock<String> = OnceLock::new();
    JSON.get_or_init(|| {
        build_preset_envelope(
            "Bluff-Plate Wake Shedding",
            &LbmToolState {
                viscosity: 0.015,
                inflow_speed: 0.11,
                steps_per_frame: 8,
                brush_radius: 2,
                area_budget: DEFAULT_AREA_BUDGET,
                view_mode: LbmViewMode::Vorticity,
                obstacle_mask: encode_obstacle_mask(&bluff_plate_obstacle_grid()),
                mask_hex: None,
                obstacles_rle: None,
            },
        )
    })
}

fn min_drag_challenge_preset_json() -> &'static str {
    static JSON: OnceLock<String> = OnceLock::new();
    JSON.get_or_init(|| {
        build_preset_envelope(
            "Minimum-Drag Airfoil Challenge",
            &LbmToolState {
                viscosity: 0.015,
                inflow_speed: 0.11,
                steps_per_frame: 8,
                brush_radius: 2,
                area_budget: 220,
                view_mode: LbmViewMode::Vorticity,
                obstacle_mask: encode_obstacle_mask(&streamlined_airfoil_obstacle_grid()),
                mask_hex: None,
                obstacles_rle: None,
            },
        )
    })
}

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "LatticeBoltzmannTool",
        display_name: "Lattice Boltzmann Wind Tunnel",
        domain: "fluid_dynamics",
        tags: &[
            "lbm",
            "d2q9",
            "bgk",
            "vortex street",
            "navier-stokes",
            "fluid",
            "cfd",
            "aerodynamics",
            "drag",
            "vorticity",
        ],
        build: || Box::new(LatticeBoltzmannTool::default()),
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "lattice_boltzmann_starter",
        title: "LBM Flow Simulation",
        description: "f_i(x + e_i·dt, t + dt) = f_i(x, t) - (f_i - f_i^eq)/τ: Mesoscopic D2Q9 cylinder wind tunnel.",
        category: "Computational Fluids",
        icon: "🌊",
        domain: "fluid_dynamics",
        tool_name: Some("LatticeBoltzmannTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "lattice_boltzmann_bluff_plate",
        title: "Bluff-Plate Wake Shedding",
        description: "High-drag vertical flat plate shedding alternating signed vorticity eddies (C_D > 2).",
        category: "Computational Fluids",
        icon: "🧱",
        domain: "fluid_dynamics",
        tool_name: Some("LatticeBoltzmannTool"),
        preset_generator: bluff_plate_preset_json,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "lattice_boltzmann_min_drag_challenge",
        title: "Minimum-Drag Airfoil Challenge",
        description: "Sculpt within a 220-cell solid-area budget to beat a streamlined airfoil's C_D.",
        category: "Computational Fluids",
        icon: "✈️",
        domain: "fluid_dynamics",
        tool_name: Some("LatticeBoltzmannTool"),
        preset_generator: min_drag_challenge_preset_json,
    }
}
