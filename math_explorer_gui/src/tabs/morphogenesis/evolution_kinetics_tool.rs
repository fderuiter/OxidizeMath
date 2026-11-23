#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::framework::recovery::{DiagnosticReport, FiniteStateGuard};
use crate::framework::{ExportCapabilities, InteractiveTool};
use crate::presets::checked_ode_step_count;
use eframe::egui;
use math_explorer::biology::evolution::HawkDovePopulation;
use math_explorer::biology::kinetics::{HillKinetics, KineticsModel, MichaelisMenten};
use math_explorer::pure_math::analysis::ode::{ArrayState, RungeKutta4, Solver};
use scientific_metadata::theory::TheoryDescribable;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

/// Scientific metadata descriptor for [`EvolutionKineticsTool`].
#[derive(Clone, Debug, Default)]
pub struct EvolutionKineticsTheory;

impl TheoryDescribable for EvolutionKineticsTheory {
    fn theory_description(&self) -> String {
        "Couples evolutionary game replicator dynamics dp_H/dt = p_H(1 - p_H)(W_H - W_D) for the Hawk-Dove contest (ESS p* = min(V/C, 1)) with Michaelis-Menten and cooperative Hill enzyme saturation kinetics v([S]) = V_max [S]^n / (K_m^n + [S]^n).".to_string()
    }

    fn phonetic_description(&self) -> String {
        "Hawk-Dove evolutionary replicator equation d p sub H d t equals p sub H times 1 minus p sub H times W sub H minus W sub D alongside Michaelis-Menten and Hill enzyme saturation curves.".to_string()
    }

    fn theory_citation(&self) -> String {
        "Maynard Smith, J., & Price, G. R. (1973). The logic of animal conflict. Nature, 246, 15-18; Michaelis, L., & Menten, M. L. (1913). Die Kinetik der Invertinwirkung. Biochemische Zeitschrift, 49, 333-369; Hill, A. V. (1910). The possible effects of the aggregation of the molecules of haemoglobin on its dissociation curves. Journal of Physiology, 40, iv-vii.".to_string()
    }

    fn available_descriptions(&self) -> HashMap<String, String> {
        let mut descs = HashMap::new();
        descs.insert(
            "hawk_dove_ess".to_string(),
            "Evolutionarily Stable Strategy p* = min(V/C, 1) balancing resource reward V against injury cost C.".to_string(),
        );
        descs.insert(
            "hill_cooperativity".to_string(),
            "Hill cooperativity exponent n governing hyperbolic (n=1) versus sigmoidal (n>1) allosteric binding.".to_string(),
        );
        descs
    }

    fn theory_latex_equation(&self) -> Option<String> {
        Some(
            r"\frac{dp_H}{dt} = p_H(1-p_H)\frac{V - p_H C}{2},\quad v([S]) = \frac{V_{\max}[S]^n}{K_m^n + [S]^n}"
                .to_string(),
        )
    }
}

/// Complete validated URL and preset state schema for [`EvolutionKineticsTool`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvolutionKineticsToolState {
    /// Contested resource value $V$ (`0.1..=20.0`).
    pub resource_value: f64,
    /// Cost of escalated Hawk-Hawk injury $C$ (`0.1..=20.0`).
    pub cost_of_injury: f64,
    /// Initial Hawk strategy proportion $p_H(0)$ (`0.001..=0.999`).
    pub initial_hawk_freq: f64,
    /// Integration horizon $T$ (`1.0..=100.0`).
    pub time_horizon: f64,
    /// RK4 integration time step $\Delta t$ (`0.005..=0.5`).
    pub dt: f64,
    /// Enzyme maximum velocity $V_{\max}$ (`0.1..=50.0`).
    pub v_max: f64,
    /// Half-saturation Michaelis constant $K_m$ (`0.05..=20.0`).
    pub k_m: f64,
    /// Hill cooperativity coefficient $n$ (`0.2..=8.0`).
    pub hill_n: f64,
    /// Highlighted substrate probe concentration `[S]` (`0.0..=30.0`).
    pub substrate_probe: f64,
}

impl Default for EvolutionKineticsToolState {
    fn default() -> Self {
        Self {
            resource_value: 2.0,
            cost_of_injury: 4.0,
            initial_hawk_freq: 0.15,
            time_horizon: 25.0,
            dt: 0.05,
            v_max: 10.0,
            k_m: 1.5,
            hill_n: 3.0,
            substrate_probe: 2.0,
        }
    }
}

impl EvolutionKineticsToolState {
    /// Validates finiteness, domain ranges, and bounded ODE step counts.
    pub fn validate(&self) -> Result<(), String> {
        let checks = [
            ("resource_value", self.resource_value, 0.1, 20.0),
            ("cost_of_injury", self.cost_of_injury, 0.1, 20.0),
            ("initial_hawk_freq", self.initial_hawk_freq, 0.001, 0.999),
            ("time_horizon", self.time_horizon, 1.0, 100.0),
            ("dt", self.dt, 0.005, 0.5),
            ("v_max", self.v_max, 0.1, 50.0),
            ("k_m", self.k_m, 0.05, 20.0),
            ("hill_n", self.hill_n, 0.2, 8.0),
            ("substrate_probe", self.substrate_probe, 0.0, 30.0),
        ];
        for (name, val, min, max) in checks {
            if !val.is_finite() || val < min || val > max {
                return Err(format!(
                    "Evolution/Kinetics parameter '{name}' = {val} must be finite within [{min}, {max}]"
                ));
            }
        }
        if checked_ode_step_count(self.dt, self.time_horizon).is_none() {
            return Err("Invalid ODE step count for time_horizon / dt".to_string());
        }
        Ok(())
    }

    /// Analytical Evolutionarily Stable Strategy Hawk frequency $p^* = \mathrm{clamp}(V / C, 0, 1)$.
    pub fn analytical_ess(&self) -> f64 {
        if self.cost_of_injury <= 1e-12 {
            1.0
        } else {
            (self.resource_value / self.cost_of_injury).clamp(0.0, 1.0)
        }
    }
}

/// Interactive GUI tool exposing [`HawkDovePopulation`], [`MichaelisMenten`], and [`HillKinetics`].
pub struct EvolutionKineticsTool {
    pub(crate) state: EvolutionKineticsToolState,
    pub(crate) hawk_curve: Vec<[f64; 2]>,
    pub(crate) dove_curve: Vec<[f64; 2]>,
    pub(crate) ess_line: Vec<[f64; 2]>,
    pub(crate) mm_curve: Vec<[f64; 2]>,
    pub(crate) hill_curve: Vec<[f64; 2]>,
    theory: EvolutionKineticsTheory,
}

impl Default for EvolutionKineticsTool {
    fn default() -> Self {
        let mut tool = Self {
            state: EvolutionKineticsToolState::default(),
            hawk_curve: Vec::new(),
            dove_curve: Vec::new(),
            ess_line: Vec::new(),
            mm_curve: Vec::new(),
            hill_curve: Vec::new(),
            theory: EvolutionKineticsTheory,
        };
        tool.recompute();
        tool
    }
}

impl EvolutionKineticsTool {
    /// Recomputes both the Hawk-Dove RK4 replicator trajectory and enzyme saturation curves.
    pub fn recompute(&mut self) {
        let steps = checked_ode_step_count(self.state.dt, self.state.time_horizon)
            .unwrap_or(500)
            .min(4_000);
        let pop = HawkDovePopulation::new(self.state.resource_value, self.state.cost_of_injury);
        let mut y = ArrayState([self.state.initial_hawk_freq.clamp(0.001, 0.999)]);
        let mut rk4 = RungeKutta4::new(&y);

        self.hawk_curve.clear();
        self.dove_curve.clear();
        self.hawk_curve.reserve(steps + 1);
        self.dove_curve.reserve(steps + 1);

        let mut t = 0.0;
        for _ in 0..=steps {
            let p_h = y.0[0].clamp(0.0, 1.0);
            self.hawk_curve.push([t, p_h]);
            self.dove_curve.push([t, 1.0 - p_h]);
            rk4.step(&pop, t, &mut y, self.state.dt);
            y.0[0] = y.0[0].clamp(0.0, 1.0);
            t += self.state.dt;
        }

        let ess = self.state.analytical_ess();
        self.ess_line = vec![[0.0, ess], [self.state.time_horizon, ess]];

        self.mm_curve.clear();
        self.hill_curve.clear();
        let mm = MichaelisMenten::new(self.state.v_max.max(0.1), self.state.k_m.max(0.05))
            .expect("validated MichaelisMenten");
        let hill = HillKinetics::new(
            self.state.v_max.max(0.1),
            self.state.k_m.max(0.05),
            self.state.hill_n.max(0.2),
        )
        .expect("validated HillKinetics");

        let max_s = (self.state.k_m * 5.0)
            .max(self.state.substrate_probe * 1.25)
            .max(5.0);
        let num_samples = 160usize;
        for i in 0..=num_samples {
            let s = max_s * (i as f64) / (num_samples as f64);
            let v_mm = mm.reaction_velocity(s).unwrap_or(0.0);
            let v_hill = hill.reaction_velocity(s).unwrap_or(0.0);
            self.mm_curve.push([s, v_mm]);
            self.hill_curve.push([s, v_hill]);
        }
    }

    /// Evaluates `(v_mm, v_hill)` at the active `substrate_probe`.
    pub fn probe_velocities(&self) -> (f64, f64) {
        let s = self.state.substrate_probe.max(0.0);
        let v_mm = MichaelisMenten::new(self.state.v_max.max(0.1), self.state.k_m.max(0.05))
            .and_then(|m| m.reaction_velocity(s))
            .unwrap_or(0.0);
        let v_hill = HillKinetics::new(
            self.state.v_max.max(0.1),
            self.state.k_m.max(0.05),
            self.state.hill_n.max(0.2),
        )
        .and_then(|h| h.reaction_velocity(s))
        .unwrap_or(0.0);
        (v_mm, v_hill)
    }
}

impl FiniteStateGuard for EvolutionKineticsTool {
    fn is_finite_state(&self) -> bool {
        self.has_finite_parameters()
            && self.hawk_curve.iter().all(|p| p[1].is_finite())
            && self.hill_curve.iter().all(|p| p[1].is_finite())
    }

    fn diagnostic_report(&self) -> Option<&DiagnosticReport> {
        None
    }

    fn diverged(&self) -> bool {
        !FiniteStateGuard::is_finite_state(self)
    }

    fn auto_scale_step_size(&mut self) {
        self.state.dt = (self.state.dt * 0.5).clamp(0.005, 0.5);
        self.recompute();
    }

    fn clamp_parameters(&mut self) {
        let sanitize = |v: f64, lo: f64, hi: f64, def: f64| {
            if v.is_nan() {
                def
            } else {
                v.clamp(lo, hi)
            }
        };
        let d = EvolutionKineticsToolState::default();
        self.state.resource_value =
            sanitize(self.state.resource_value, 0.1, 20.0, d.resource_value);
        self.state.cost_of_injury =
            sanitize(self.state.cost_of_injury, 0.1, 20.0, d.cost_of_injury);
        self.state.initial_hawk_freq = sanitize(
            self.state.initial_hawk_freq,
            0.001,
            0.999,
            d.initial_hawk_freq,
        );
        self.state.time_horizon = sanitize(self.state.time_horizon, 1.0, 100.0, d.time_horizon);
        self.state.dt = sanitize(self.state.dt, 0.005, 0.5, d.dt);
        self.state.v_max = sanitize(self.state.v_max, 0.1, 50.0, d.v_max);
        self.state.k_m = sanitize(self.state.k_m, 0.05, 20.0, d.k_m);
        self.state.hill_n = sanitize(self.state.hill_n, 0.2, 8.0, d.hill_n);
        self.state.substrate_probe =
            sanitize(self.state.substrate_probe, 0.0, 30.0, d.substrate_probe);
        self.recompute();
    }

    fn reset_parameters(&mut self) {
        self.state = EvolutionKineticsToolState::default();
        self.recompute();
    }
}

impl InteractiveTool for EvolutionKineticsTool {
    fn name(&self) -> &'static str {
        "Evolution & Enzyme Kinetics"
    }

    fn domain(&self) -> &'static str {
        "biology"
    }

    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        vec![
            ("Hawk Frequency p_H(t)".to_string(), self.hawk_curve.clone()),
            ("Dove Frequency p_D(t)".to_string(), self.dove_curve.clone()),
            ("Analytical ESS p*".to_string(), self.ess_line.clone()),
            ("Michaelis-Menten v([S])".to_string(), self.mm_curve.clone()),
            ("Hill Kinetics v([S])".to_string(), self.hill_curve.clone()),
        ]
    }

    fn step_simulation(&mut self) {
        self.state.time_horizon = (self.state.time_horizon + 2.0).clamp(1.0, 100.0);
        self.recompute();
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        self.draw_controls(ui);
        self.draw_canvas(ui);
    }

    fn theory(&self) -> &dyn TheoryDescribable {
        &self.theory
    }

    fn state_revision(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.state.resource_value.to_bits().hash(&mut h);
        self.state.cost_of_injury.to_bits().hash(&mut h);
        self.state.initial_hawk_freq.to_bits().hash(&mut h);
        self.state.time_horizon.to_bits().hash(&mut h);
        self.state.dt.to_bits().hash(&mut h);
        self.state.v_max.to_bits().hash(&mut h);
        self.state.k_m.to_bits().hash(&mut h);
        self.state.hill_n.to_bits().hash(&mut h);
        self.state.substrate_probe.to_bits().hash(&mut h);
        h.finish().max(1)
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&self.state).ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let parsed: EvolutionKineticsToolState = serde_json::from_str(state)
            .map_err(|e| format!("Invalid EvolutionKinetics state: {e}"))?;
        parsed.validate()?;
        self.state = parsed;
        self.recompute();
        Ok(())
    }

    fn has_finite_parameters(&self) -> bool {
        self.state.resource_value.is_finite()
            && self.state.cost_of_injury.is_finite()
            && self.state.initial_hawk_freq.is_finite()
            && self.state.time_horizon.is_finite()
            && self.state.dt.is_finite()
            && self.state.v_max.is_finite()
            && self.state.k_m.is_finite()
            && self.state.hill_n.is_finite()
            && self.state.substrate_probe.is_finite()
    }

    fn clamp_parameters(&mut self) -> bool {
        FiniteStateGuard::clamp_parameters(self);
        true
    }
}

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "EvolutionKineticsTool",
        display_name: "Evolution & Enzyme Kinetics",
        domain: "biology",
        tags: &[
            "evolution",
            "hawk-dove",
            "replicator",
            "ess",
            "michaelis-menten",
            "hill-kinetics",
        ],
        build: || Box::new(EvolutionKineticsTool::default()),
    }
}
