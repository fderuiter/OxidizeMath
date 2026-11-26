//! Interactive **Enzyme Kinetics & Reaction-Diffusion** studio (`BiochemicalDynamicsTool`)
//! inside [`super::BiochemicalTab`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

pub mod presets;
mod ui;

#[cfg(test)]
mod tests;

pub use presets::{BiochemicalDynamicsState, BiochemicalTheory};

use crate::framework::recovery::{DiagnosticReport, FiniteStateGuard};
use crate::framework::{ExportCapabilities, InteractiveTool, ToolStateBounds};
use eframe::egui;
use math_explorer::biology::diffusion::FiniteDifference1D;
use math_explorer::biology::kinetics::{HillKinetics, KineticsModel, MichaelisMenten};
use math_explorer::biology::reaction_diffusion::{HillSubstrateReaction, ReactionDiffusionSystem};
use math_explorer::math_kernel::types::StepSize;
use math_explorer::pure_math::analysis::ode::Euler;
use scientific_metadata::theory::TheoryDescribable;
use std::hash::{Hash, Hasher};

const COOPERATIVE_FRAME_BUDGET_SECS: f64 = 0.008;
const BASAL_SYNTHESIS_RATE: f64 = 0.02;

type BiochemicalRdSystem = ReactionDiffusionSystem<
    HillSubstrateReaction,
    FiniteDifference1D,
    Euler<math_explorer::biology::reaction_diffusion::ChemicalState>,
>;

/// Interactive studio coupling [`MichaelisMenten`] and [`HillKinetics`] enzyme saturation curves
/// with a zero-allocation two-species 1D [`ReactionDiffusionSystem`].
pub struct BiochemicalDynamicsTool {
    /// Serializable URL and preset configuration.
    pub state: BiochemicalDynamicsState,
    pub(crate) rd_system: BiochemicalRdSystem,
    pub(crate) sim_time: f64,
    pub(crate) step_count: usize,
    pub(crate) last_step_ms: f64,
    pub(crate) diagnostic: Option<DiagnosticReport>,
    theory: BiochemicalTheory,
}

impl Default for BiochemicalDynamicsTool {
    fn default() -> Self {
        let state = BiochemicalDynamicsState::default();
        let rd_system = build_rd_system(&state).expect("default biochemical RD parameters valid");
        let mut tool = Self {
            state,
            rd_system,
            sim_time: 0.0,
            step_count: 0,
            last_step_ms: 0.0,
            diagnostic: None,
            theory: BiochemicalTheory,
        };
        tool.seed_initial_pulse();
        tool
    }
}

fn build_rd_system(state: &BiochemicalDynamicsState) -> Result<BiochemicalRdSystem, String> {
    let reaction = HillSubstrateReaction::new(
        state.v_max,
        state.k_m,
        state.hill_n,
        state.substrate_reservoir,
        state.feed_rate,
        state.degradation_rate,
        BASAL_SYNTHESIS_RATE,
    )
    .map_err(|e| format!("Invalid Hill reaction parameters: {e}"))?;

    ReactionDiffusionSystem::builder()
        .num_species(2)
        .grid_size(state.grid_points)
        .reaction(reaction)
        .diffusion(FiniteDifference1D::new(StepSize(0.5)))
        .diffusion_coeffs(vec![state.diffusion_activator, state.diffusion_substrate])
        .build()
        .map_err(|e| format!("Failed to construct ReactionDiffusionSystem: {e}"))
}

impl BiochemicalDynamicsTool {
    fn seed_initial_pulse(&mut self) {
        let n = self.rd_system.state.grid_size();
        let center = (n as f64) * 0.5;
        let sigma_sq = ((n as f64) * 0.09).powi(2).max(1.0);
        for i in 0..n {
            let dist_sq = ((i as f64) - center).powi(2);
            let pulse = 1.6 * (-dist_sq / (2.0 * sigma_sq)).exp();
            self.rd_system.state.species_mut(0)[i] = 0.08 + pulse;
            self.rd_system.state.species_mut(1)[i] = self.state.substrate_reservoir;
        }
        self.sim_time = 0.0;
        self.step_count = 0;
    }

    /// Rebuilds the pre-allocated 1D [`ReactionDiffusionSystem`] from current parameters.
    pub fn rebuild_reaction_diffusion(&mut self) {
        if let Err(msg) = self.state.validate() {
            self.state.playing = false;
            self.diagnostic = Some(DiagnosticReport::new(
                self.sim_time,
                self.step_count,
                "parameters",
                msg,
            ));
            return;
        }
        match build_rd_system(&self.state) {
            Ok(sys) => {
                self.rd_system = sys;
                self.diagnostic = None;
                self.seed_initial_pulse();
            }
            Err(msg) => {
                self.state.playing = false;
                self.diagnostic = Some(DiagnosticReport::new(
                    self.sim_time,
                    self.step_count,
                    "reaction_diffusion",
                    msg,
                ));
            }
        }
    }

    /// Advances the 1D reaction-diffusion system by one step using zero heap allocations.
    pub fn step_once(&mut self) {
        if let Err(msg) = self.state.validate() {
            self.state.playing = false;
            self.diagnostic = Some(DiagnosticReport::new(
                self.sim_time,
                self.step_count,
                "parameters",
                msg,
            ));
            return;
        }
        let start = crate::web_io::PlatformInstant::now();
        self.rd_system.step(self.state.dt);
        self.last_step_ms = start.elapsed().as_secs_f64() * 1000.0;
        self.sim_time += self.state.dt;
        self.step_count += 1;

        let finite = self
            .rd_system
            .state
            .species(0)
            .iter()
            .all(|v| v.is_finite())
            && self
                .rd_system
                .state
                .species(1)
                .iter()
                .all(|v| v.is_finite());
        if !finite {
            self.state.playing = false;
            self.diagnostic = Some(DiagnosticReport::new(
                self.sim_time,
                self.step_count,
                "state",
                "Non-finite concentration detected in 1D ReactionDiffusionSystem; step halted.",
            ));
        }
    }

    pub(crate) fn step_cooperative_frame(&mut self) {
        let frame_start = crate::web_io::PlatformInstant::now();
        for _ in 0..4 {
            self.step_once();
            if !self.state.playing
                || frame_start.elapsed().as_secs_f64() >= COOPERATIVE_FRAME_BUDGET_SECS
            {
                break;
            }
        }
    }

    /// Samples `(MichaelisMenten, HillKinetics)` velocity curves across `0.0..=max_substrate`.
    pub fn sample_kinetics_curves(&self) -> (Vec<[f64; 2]>, Vec<[f64; 2]>) {
        let samples = 120_usize;
        let mm = MichaelisMenten::new(self.state.v_max.max(0.0), self.state.k_m.max(1e-6));
        let hill = HillKinetics::new(
            self.state.v_max.max(0.0),
            self.state.k_m.max(1e-6),
            self.state.hill_n.max(0.1),
        );
        let mut mm_pts = Vec::with_capacity(samples + 1);
        let mut hill_pts = Vec::with_capacity(samples + 1);

        if let (Ok(mm_model), Ok(hill_model)) = (mm, hill) {
            for i in 0..=samples {
                let s = self.state.max_substrate * (i as f64) / (samples as f64);
                let v_mm = mm_model.reaction_velocity(s).unwrap_or(0.0);
                let v_hill = hill_model.reaction_velocity(s).unwrap_or(0.0);
                mm_pts.push([s, v_mm]);
                hill_pts.push([s, v_hill]);
            }
        }
        (mm_pts, hill_pts)
    }

    /// Samples the active 1D activator $u(x)$ and substrate $s(x)$ spatial concentration profiles.
    pub fn sample_spatial_profiles(&self) -> (Vec<[f64; 2]>, Vec<[f64; 2]>) {
        let u = self.rd_system.state.species(0);
        let s = self.rd_system.state.species(1);
        let dx = 0.5_f64;
        let u_pts = u
            .iter()
            .enumerate()
            .map(|(i, &val)| [(i as f64) * dx, val])
            .collect();
        let s_pts = s
            .iter()
            .enumerate()
            .map(|(i, &val)| [(i as f64) * dx, val])
            .collect();
        (u_pts, s_pts)
    }
}

impl FiniteStateGuard for BiochemicalDynamicsTool {
    fn is_finite_state(&self) -> bool {
        self.diagnostic.is_none()
            && self.state.validate().is_ok()
            && self
                .rd_system
                .state
                .species(0)
                .iter()
                .all(|v| v.is_finite())
            && self
                .rd_system
                .state
                .species(1)
                .iter()
                .all(|v| v.is_finite())
    }

    fn diagnostic_report(&self) -> Option<&DiagnosticReport> {
        self.diagnostic.as_ref()
    }

    fn diverged(&self) -> bool {
        !FiniteStateGuard::is_finite_state(self)
    }

    fn auto_scale_step_size(&mut self) {
        let next_dt = if self.state.dt.is_finite() && self.state.dt > 0.0 {
            self.state.dt * 0.5
        } else {
            0.01
        };
        self.state.dt = next_dt.clamp(0.001, 0.05);
        FiniteStateGuard::clamp_parameters(self);
    }

    fn clamp_parameters(&mut self) {
        fn clean(val: f64, lo: f64, hi: f64, fallback: f64) -> f64 {
            if val.is_finite() {
                val.clamp(lo, hi)
            } else {
                fallback
            }
        }
        self.state.v_max = clean(self.state.v_max, 0.1, 20.0, 2.4);
        self.state.k_m = clean(self.state.k_m, 0.05, 10.0, 1.2);
        self.state.hill_n = clean(self.state.hill_n, 0.5, 8.0, 2.5);
        self.state.max_substrate = clean(self.state.max_substrate, 1.0, 40.0, 8.0);
        self.state.substrate_reservoir = clean(self.state.substrate_reservoir, 0.2, 10.0, 2.2);
        self.state.feed_rate = clean(self.state.feed_rate, 0.01, 2.0, 0.30);
        self.state.degradation_rate = clean(self.state.degradation_rate, 0.01, 2.0, 0.45);
        self.state.diffusion_activator = clean(self.state.diffusion_activator, 0.005, 2.0, 0.08);
        self.state.diffusion_substrate = clean(self.state.diffusion_substrate, 0.005, 2.0, 0.40);
        self.state.grid_points = self.state.grid_points.clamp(16, 256);
        self.state.dt = clean(self.state.dt, 0.001, 0.1, 0.02);
        self.diagnostic = None;
        self.rebuild_reaction_diffusion();
    }

    fn reset_parameters(&mut self) {
        self.state = BiochemicalDynamicsState::default();
        self.diagnostic = None;
        self.rebuild_reaction_diffusion();
    }
}

impl InteractiveTool for BiochemicalDynamicsTool {
    fn name(&self) -> &'static str {
        "Enzyme Kinetics & Reaction-Diffusion"
    }

    fn domain(&self) -> &'static str {
        "biochemical"
    }

    fn theory(&self) -> &dyn TheoryDescribable {
        &self.theory
    }

    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let (mm_pts, hill_pts) = self.sample_kinetics_curves();
        let (u_pts, s_pts) = self.sample_spatial_profiles();
        vec![
            ("Michaelis-Menten Velocity".to_string(), mm_pts),
            (
                format!("Hill Cooperative Velocity (n={:.2})", self.state.hill_n),
                hill_pts,
            ),
            ("1D Activator Concentration u(x)".to_string(), u_pts),
            ("1D Substrate Concentration s(x)".to_string(), s_pts),
        ]
    }

    fn step_simulation(&mut self) {
        for _ in 0..4 {
            self.step_once();
        }
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        self.render_studio(ui);
    }

    fn has_finite_parameters(&self) -> bool {
        FiniteStateGuard::is_finite_state(self)
    }

    fn clamp_parameters(&mut self) -> bool {
        FiniteStateGuard::clamp_parameters(self);
        true
    }

    fn state_bounds(&self) -> ToolStateBounds {
        crate::url_state::URL_JSON_BOUNDS
    }

    fn state_revision(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.state.view.hash(&mut h);
        self.state.v_max.to_bits().hash(&mut h);
        self.state.k_m.to_bits().hash(&mut h);
        self.state.hill_n.to_bits().hash(&mut h);
        self.state.max_substrate.to_bits().hash(&mut h);
        self.state.substrate_reservoir.to_bits().hash(&mut h);
        self.state.feed_rate.to_bits().hash(&mut h);
        self.state.degradation_rate.to_bits().hash(&mut h);
        self.state.diffusion_activator.to_bits().hash(&mut h);
        self.state.diffusion_substrate.to_bits().hash(&mut h);
        self.state.grid_points.hash(&mut h);
        self.state.dt.to_bits().hash(&mut h);
        self.state.playing.hash(&mut h);
        h.finish().max(1)
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&self.state).ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
        let parsed: BiochemicalDynamicsState =
            serde_json::from_str(state).map_err(|e| format!("Invalid biochemical state: {e}"))?;
        parsed.validate()?;
        self.state = parsed;
        self.diagnostic = None;
        self.rebuild_reaction_diffusion();
        Ok(())
    }

    #[cfg(test)]
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    #[cfg(test)]
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }
}

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "BiochemicalDynamicsTool",
        display_name: "Enzyme Kinetics & Reaction-Diffusion",
        domain: "biochemical",
        tags: &[
            "kinetics",
            "michaelis-menten",
            "hill",
            "cooperativity",
            "enzyme",
            "reaction-diffusion",
            "allosteric",
        ],
        build: || Box::new(BiochemicalDynamicsTool::default()),
    }
}
