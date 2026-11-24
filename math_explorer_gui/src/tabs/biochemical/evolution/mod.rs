//! Interactive **Evolutionary Replicator Lab** (`EvolutionaryBiologyTool`) inside
//! [`super::BiochemicalTab`], backed by [`HawkDovePopulation`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

pub mod presets;
mod ui;

#[cfg(test)]
mod tests;

pub use presets::{EvolutionaryBiologyState, EvolutionaryTheory};

use crate::framework::recovery::{DiagnosticReport, FiniteStateGuard};
use crate::framework::{ExportCapabilities, InteractiveTool, ToolStateBounds};
use eframe::egui;
use math_explorer::biology::evolution::HawkDovePopulation;
use math_explorer::primitives::UnitInterval;
use math_explorer::pure_math::analysis::ode::{ArrayState, RungeKutta4, SolverExt};
use scientific_metadata::theory::TheoryDescribable;
use std::hash::{Hash, Hasher};

/// Interactive studio exploring [`HawkDovePopulation`] replicator dynamics, mixed ESS
/// coexistence ($p^* = V / C$), and 1D evolutionary phase portraits.
pub struct EvolutionaryBiologyTool {
    /// Serializable URL and preset configuration.
    pub state: EvolutionaryBiologyState,
    pub(crate) primary_hawk_traj: Vec<[f64; 2]>,
    pub(crate) low_traj: Vec<[f64; 2]>,
    pub(crate) high_traj: Vec<[f64; 2]>,
    pub(crate) phase_curve: Vec<[f64; 2]>,
    pub(crate) diagnostic: Option<DiagnosticReport>,
    theory: EvolutionaryTheory,
}

impl Default for EvolutionaryBiologyTool {
    fn default() -> Self {
        let mut tool = Self {
            state: EvolutionaryBiologyState::default(),
            primary_hawk_traj: Vec::new(),
            low_traj: Vec::new(),
            high_traj: Vec::new(),
            phase_curve: Vec::new(),
            diagnostic: None,
            theory: EvolutionaryTheory,
        };
        tool.recompute();
        tool
    }
}

impl EvolutionaryBiologyTool {
    /// Returns the active [`HawkDovePopulation`] domain model.
    pub fn population(&self) -> HawkDovePopulation {
        HawkDovePopulation::new(self.state.resource_value, self.state.injury_cost)
    }

    /// Recomputes RK4 replicator trajectories and the 1D phase velocity curve $\dot{p}_H(p_H)$.
    pub fn recompute(&mut self) {
        if let Err(msg) = self.state.validate() {
            self.diagnostic = Some(DiagnosticReport::new(0.0, 0, "parameters", msg));
            return;
        }
        self.diagnostic = None;
        let pop = self.population();
        self.primary_hawk_traj = integrate_replicator_trajectory(
            &pop,
            self.state.initial_hawk_freq,
            self.state.horizon,
            self.state.dt,
        );
        self.low_traj =
            integrate_replicator_trajectory(&pop, 0.08, self.state.horizon, self.state.dt);
        self.high_traj =
            integrate_replicator_trajectory(&pop, 0.92, self.state.horizon, self.state.dt);

        let phase_samples = 120_usize;
        self.phase_curve.clear();
        self.phase_curve.reserve(phase_samples + 1);
        for i in 0..=phase_samples {
            let p = (i as f64) / (phase_samples as f64);
            self.phase_curve.push([p, pop.replicator_velocity(p)]);
        }
    }
}

fn integrate_replicator_trajectory(
    pop: &HawkDovePopulation,
    initial_p: f64,
    horizon: f64,
    dt: f64,
) -> Vec<[f64; 2]> {
    let steps = ((horizon / dt).round() as usize).clamp(4, 4_000);
    let mut out = Vec::with_capacity(steps + 1);
    let p_clamped = UnitInterval::new(initial_p.clamp(0.001, 0.999)).map_or(0.5, |u| u.value());
    let mut state = ArrayState([p_clamped]);
    let mut rk4 = RungeKutta4::new(&state);
    let mut t = 0.0_f64;
    out.push([t, state.0[0]]);
    for _ in 0..steps {
        state = rk4.solve(pop, t, &state, dt);
        state.0[0] = state.0[0].clamp(0.0, 1.0);
        t += dt;
        out.push([t, state.0[0]]);
    }
    out
}

impl FiniteStateGuard for EvolutionaryBiologyTool {
    fn is_finite_state(&self) -> bool {
        self.diagnostic.is_none() && self.state.validate().is_ok()
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
            0.05
        };
        self.state.dt = next_dt.clamp(0.005, 0.125);
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
        self.state.resource_value = clean(self.state.resource_value, 0.1, 20.0, 2.0);
        self.state.injury_cost = clean(self.state.injury_cost, 0.1, 30.0, 4.0);
        self.state.initial_hawk_freq = clean(self.state.initial_hawk_freq, 0.01, 0.99, 0.85);
        self.state.horizon = clean(self.state.horizon, 2.0, 80.0, 20.0);
        self.state.dt = clean(self.state.dt, 0.005, 0.25, 0.05);
        self.diagnostic = None;
        self.recompute();
    }

    fn reset_parameters(&mut self) {
        self.state = EvolutionaryBiologyState::default();
        self.diagnostic = None;
        self.recompute();
    }
}

impl InteractiveTool for EvolutionaryBiologyTool {
    fn name(&self) -> &'static str {
        "Evolutionary Replicator Lab"
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
        let dove_pts: Vec<[f64; 2]> = self
            .primary_hawk_traj
            .iter()
            .map(|&[t, p]| [t, 1.0 - p])
            .collect();
        vec![
            (
                "Hawk Frequency p_H(t)".to_string(),
                self.primary_hawk_traj.clone(),
            ),
            ("Dove Frequency p_D(t)".to_string(), dove_pts),
            (
                "Replicator Phase Velocity dp_H/dt".to_string(),
                self.phase_curve.clone(),
            ),
        ]
    }

    fn step_simulation(&mut self) {
        self.state.horizon = (self.state.horizon + 2.0).min(80.0);
        self.recompute();
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
        self.state.resource_value.to_bits().hash(&mut h);
        self.state.injury_cost.to_bits().hash(&mut h);
        self.state.initial_hawk_freq.to_bits().hash(&mut h);
        self.state.horizon.to_bits().hash(&mut h);
        self.state.dt.to_bits().hash(&mut h);
        self.state.show_ensemble.hash(&mut h);
        h.finish().max(1)
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&self.state).ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
        let parsed: EvolutionaryBiologyState =
            serde_json::from_str(state).map_err(|e| format!("Invalid evolution state: {e}"))?;
        parsed.validate()?;
        self.state = parsed;
        self.diagnostic = None;
        self.recompute();
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
        id: "EvolutionaryBiologyTool",
        display_name: "Evolutionary Replicator Lab",
        domain: "biochemical",
        tags: &[
            "evolution",
            "hawk-dove",
            "replicator",
            "ess",
            "maynard-smith",
            "game-theory",
            "population",
        ],
        build: || Box::new(EvolutionaryBiologyTool::default()),
    }
}
