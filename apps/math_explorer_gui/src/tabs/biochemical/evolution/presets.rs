//! Presets, URL-safe state schema, [`TheoryDescribable`], and `StarterPresetMetadata` cards
//! for [`super::EvolutionaryBiologyTool`].

use scientific_metadata::theory::TheoryDescribable;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Plot layout selector inside [`super::EvolutionaryBiologyTool`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EvolutionPlotView {
    /// Stacked replicator frequency trajectories $p_H(t)$ and 1D phase line $\dot{p}_H(p_H)$.
    #[default]
    TrajectoryAndPhase,
    /// Replicator frequency trajectories converging toward ESS $p^*$.
    TrajectoryOnly,
    /// 1D replicator phase velocity $\dot{p}_H = p_H(1 - p_H)(V - p_H C)/2$.
    PhaseLineOnly,
}

impl EvolutionPlotView {
    /// All selectable plot views.
    pub const ALL: [Self; 3] = [
        Self::TrajectoryAndPhase,
        Self::TrajectoryOnly,
        Self::PhaseLineOnly,
    ];

    /// Human-readable label.
    pub const fn label(self) -> &'static str {
        match self {
            Self::TrajectoryAndPhase => "Trajectories + Replicator Phase Line",
            Self::TrajectoryOnly => "ESS Convergence Trajectories",
            Self::PhaseLineOnly => "1D Replicator Phase Portrait",
        }
    }
}

/// Compact `< 4 KiB` URL-safe state for [`super::EvolutionaryBiologyTool`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvolutionaryBiologyState {
    /// Active plot layout mode.
    pub view: EvolutionPlotView,
    /// Contested resource value $V$ (`0.1..=20.0`).
    pub resource_value: f64,
    /// Injury cost from escalated Hawk-Hawk combat $C$ (`0.1..=30.0`).
    pub injury_cost: f64,
    /// Primary initial Hawk frequency $p_0 \in [0.01, 0.99]$.
    pub initial_hawk_freq: f64,
    /// Integration time horizon $T$ (`2.0..=80.0`).
    pub horizon: f64,
    /// Replicator integration step $\Delta t$ (`0.005..=0.25`).
    pub dt: f64,
    /// Whether to display an ensemble fan of secondary initial conditions (`0.08`, `0.92`).
    pub show_ensemble: bool,
}

impl Default for EvolutionaryBiologyState {
    fn default() -> Self {
        Self {
            view: EvolutionPlotView::TrajectoryAndPhase,
            resource_value: 2.0,
            injury_cost: 4.0,
            initial_hawk_freq: 0.85,
            horizon: 20.0,
            dt: 0.05,
            show_ensemble: true,
        }
    }
}

impl EvolutionaryBiologyState {
    /// Validates all parameters against `URL_JSON_BOUNDS` and evolutionary game invariants.
    pub fn validate(&self) -> Result<(), String> {
        let checks = [
            ("resource_value", self.resource_value, 0.1, 20.0),
            ("injury_cost", self.injury_cost, 0.1, 30.0),
            ("initial_hawk_freq", self.initial_hawk_freq, 0.01, 0.99),
            ("horizon", self.horizon, 2.0, 80.0),
            ("dt", self.dt, 0.005, 0.25),
        ];
        for (name, val, min, max) in checks {
            if !val.is_finite() || val < min || val > max {
                return Err(format!(
                    "Evolutionary parameter '{name}' = {val} must be finite within [{min}, {max}]"
                ));
            }
        }
        Ok(())
    }
}

/// [`TheoryDescribable`] metadata for Maynard Smith & Price Hawk-Dove replicator dynamics.
#[derive(Clone, Debug, Default)]
pub struct EvolutionaryTheory;

impl TheoryDescribable for EvolutionaryTheory {
    fn theory_description(&self) -> String {
        "Models frequency-dependent natural selection in the Maynard Smith & Price Hawk-Dove game \
         under continuous replicator dynamics dp/dt = p(1 - p)(W_H(p) - W_D(p)) = p(1 - p)(V - p*C)/2. \
         When injury cost exceeds resource value (C > V), neither pure strategy is evolutionarily \
         stable and trajectories converge globally from (0, 1) to the polymorphic mixed ESS p* = V / C."
            .to_string()
    }

    fn phonetic_description(&self) -> String {
        "Hawk-Dove evolutionary replicator dynamics d p d t equals p times one minus p times V minus \
         p C over two, converging to mixed evolutionarily stable frequency V over C."
            .to_string()
    }

    fn theory_citation(&self) -> String {
        "Maynard Smith, J., & Price, G. R. (1973). The logic of animal conflict. Nature, \
         246(5427), 15-18."
            .to_string()
    }

    fn available_descriptions(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert(
            "Polymorphic ESS (C > V)".to_string(),
            "High injury cost prevents pure Hawk invasion, stabilizing a mixed population p* = V / C."
                .to_string(),
        );
        map.insert(
            "Pure Hawk Escalation (V >= C)".to_string(),
            "When resource value outweighs injury cost, fighting strictly dominates and p* = 1."
                .to_string(),
        );
        map
    }

    fn theory_latex_equation(&self) -> Option<String> {
        Some(
            r"\dot{p}_H = \frac{1}{2} p_H (1 - p_H)(V - p_H C),\qquad p^* = \min\!\left(1, \frac{V}{C}\right)"
                .to_string(),
        )
    }
}

/// Starter preset JSON showcasing polymorphic Hawk-Dove coexistence ($V = 2, C = 6 \implies p^* = 1/3$).
pub const PRESET_HAWK_DOVE_ESS: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Hawk-Dove Mixed ESS Coexistence",
    "domain": "biochemical",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "view": "TrajectoryAndPhase",
    "resource_value": 2.0,
    "injury_cost": 6.0,
    "initial_hawk_freq": 0.88,
    "horizon": 24.0,
    "dt": 0.05,
    "show_ensemble": true
  }
}"#;

/// Starter preset JSON showcasing pure Hawk dominance when resource value exceeds injury cost ($V = 5, C = 3$).
pub const PRESET_HAWK_ESCALATION: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Escalated Conflict Hawk Dominance",
    "domain": "biochemical",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "view": "TrajectoryAndPhase",
    "resource_value": 5.0,
    "injury_cost": 3.0,
    "initial_hawk_freq": 0.12,
    "horizon": 16.0,
    "dt": 0.04,
    "show_ensemble": true
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "evolution_hawk_dove_ess",
        title: "Hawk-Dove Mixed ESS Coexistence",
        description: "dp/dt = ½p(1-p)(V - pC): Replicator convergence from boundary initial conditions to polymorphic ESS p* = V/C = 1/3.",
        category: "Evolutionary Biology",
        icon: "🦅",
        domain: "biochemical",
        tool_name: Some("EvolutionaryBiologyTool"),
        preset_generator: || PRESET_HAWK_DOVE_ESS,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "evolution_hawk_escalation",
        title: "Escalated Conflict Hawk Dominance",
        description: "V > C regime where payoff to escalated fighting exceeds injury cost, driving fixation to pure Hawk p* = 1.",
        category: "Evolutionary Biology",
        icon: "⚔",
        domain: "biochemical",
        tool_name: Some("EvolutionaryBiologyTool"),
        preset_generator: || PRESET_HAWK_ESCALATION,
    }
}

// [cite:biology]
