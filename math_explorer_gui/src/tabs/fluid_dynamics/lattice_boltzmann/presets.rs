//! Starter gallery presets for [`super::LatticeBoltzmannTool`] (cylinder baseline,
//! bluff-plate wake shedding, and minimum-drag airfoil challenge).

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::state::{
    bluff_plate_obstacle_grid, encode_obstacle_mask, streamlined_airfoil_obstacle_grid,
    LbmToolState, LbmViewMode, DEFAULT_AREA_BUDGET, INFLOW_SPEED_MAX, INFLOW_SPEED_MIN,
    VISCOSITY_MAX, VISCOSITY_MIN,
};
use super::LatticeBoltzmannTool;
use crate::framework::recovery::{DiagnosticReport, FiniteStateGuard};
use crate::framework::InteractiveTool;
use math_explorer::physics::fluid_dynamics::lattice_boltzmann::{
    BgkCollision, LatticeBoltzmannD2Q9,
};
use scientific_metadata::theory::{ParameterConstraint, TheoryDescribable};
use std::collections::HashMap;
use std::sync::OnceLock;

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
                    step: 0.01,
                },
            ),
            (
                "inflow_speed".to_string(),
                ParameterConstraint {
                    min: INFLOW_SPEED_MIN,
                    max: INFLOW_SPEED_MAX,
                    step: 0.01,
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
            },
        )
    })
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
