//! Serialization state, Kelly edge badges, theory metadata, and starter presets for Parrondo's Paradox.

use super::ParrondoParadoxTool;
use crate::framework::{StarterPresetMetadata, ToolMetadata};
use math_explorer::error::KellyError;
use math_explorer::pure_math::statistics::kelly::{Odds, UnitInterval, kelly_fraction};
use scientific_metadata::theory::TheoryDescribable;
use serde::{Deserialize, Serialize};

/// Active workspace layout and export focus for [`ParrondoParadoxTool`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ParrondoViewMode {
    /// Combined multi-panel dashboard; exports the `(p_A, epsilon)` paradox phase wedge.
    #[default]
    SplitOverview,
    /// Full-width ensemble wealth trajectories and exact Markov drift lines.
    WealthTrajectories,
    /// Full-width modulo-`M` stationary and empirical trap-occupancy bar chart.
    TrapOccupancy,
    /// Full-width `(p_A, epsilon)` paradox phase-region map.
    PhaseWedge,
}

impl ParrondoViewMode {
    pub(super) const ALL: [Self; 4] = [
        Self::SplitOverview,
        Self::WealthTrajectories,
        Self::TrapOccupancy,
        Self::PhaseWedge,
    ];

    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::SplitOverview => "Split Overview",
            Self::WealthTrajectories => "Trajectories & Drift",
            Self::TrapOccupancy => "Trap Occupancy",
            Self::PhaseWedge => "Phase Wedge",
        }
    }
}

/// Even-money Kelly criterion diagnosis derived from a strategy's stationary step drift.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KellyEdgeBadge {
    /// `kelly_fraction` returned [`KellyError::NoEdge`] ($p_{\text{eff}} \le 0.5$).
    NoEdge { win_prob: f64 },
    /// `kelly_fraction` returned a positive optimal fraction $f^* > 0$.
    PositiveEdge {
        optimal_fraction: f64,
        win_prob: f64,
    },
}

impl KellyEdgeBadge {
    #[must_use]
    pub fn evaluate(expected_step_gain: f64) -> Self {
        let win_prob = (0.5 * (1.0 + expected_step_gain)).clamp(1e-6, 1.0 - 1e-6);
        let Ok(prob) = UnitInterval::new(win_prob) else {
            return Self::NoEdge { win_prob };
        };
        let Ok(even_money_odds) = Odds::new(2.0) else {
            return Self::NoEdge { win_prob };
        };
        match kelly_fraction(&prob, &even_money_odds) {
            Ok(frac) => Self::PositiveEdge {
                optimal_fraction: frac.value(),
                win_prob,
            },
            Err(KellyError::NoEdge { .. } | _) => Self::NoEdge { win_prob },
        }
    }

    #[must_use]
    pub const fn is_no_edge(self) -> bool {
        matches!(self, Self::NoEdge { .. })
    }

    #[must_use]
    pub fn badge_text(self) -> String {
        match self {
            Self::NoEdge { .. } => "NoEdge (f* = 0%)".to_string(),
            Self::PositiveEdge {
                optimal_fraction, ..
            } => format!("Kelly f* = {:.2}%", optimal_fraction * 100.0),
        }
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ParrondoToolState {
    pub(super) p_a_base: f64,
    pub(super) p_b0_base: f64,
    pub(super) p_b1_base: f64,
    pub(super) epsilon: f64,
    pub(super) modulus: usize,
    pub(super) schedule: String,
    pub(super) num_walkers: usize,
    pub(super) num_steps: usize,
    pub(super) seed: u64,
    pub(super) view_mode: ParrondoViewMode,
}

impl TheoryDescribable for ParrondoParadoxTool {
    fn theory_description(&self) -> String {
        let sched = self.active_schedule.as_string();
        let gain_sched = self.report.as_ref().map_or(0.0, |r| r.gain_schedule);
        format!(
            "Discrete-time modulo-$M$ Brownian ratchet ($M = {}$, $\\epsilon = {:.4}$) coupling losing games $A$ and $B$ via schedule `{}` ($P_{{\\text{{sched}}}} = \\prod_k P_{{s_k}}$) to break detailed balance and yield stationary step drift $E[\\Delta W] = {:+.5}$.",
            self.modulus, self.epsilon, sched, gain_sched
        )
    }

    fn phonetic_description(&self) -> String {
        "Parrondo paradox discrete time modulo M Markov Brownian ratchet and Kelly criterion edge reversal".to_string()
    }

    fn theory_citation(&self) -> String {
        "Harmer, G. P. & Abbott, D. (1999). Losing strategies can win by Parrondo's paradox. Nature, 402, 864; Parrondo, J. M. R., Harmer, G. P. & Abbott, D. (2000). New paradoxical games based on Brownian ratchets. Physical Review Letters, 85(24), 5226-5229; Kelly, J. L. (1956). Bell System Technical Journal, 35(4), 917-926.".to_string()
    }

    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "Game A & Game B (Losing)".to_string(),
            "Isolated games satisfy negative expectation under bias epsilon > 0, triggering KellyError::NoEdge on standalone bankroll sizing.".to_string(),
        );
        map.insert(
            "Periodic Schedule Ratchet".to_string(),
            "Alternating P_A and P_B breaks detailed balance on Z/MZ and evacuates probability mass from the W mod M = 0 losing trap.".to_string(),
        );
        map.insert(
            "Phase Wedge (p_A, epsilon)".to_string(),
            "Parameter wedge bounded by E[Delta W_A] = 0, E[Delta W_B] = 0, and E[Delta W_sched] = 0 where two losing games synthesize positive drift.".to_string(),
        );
        map
    }
}

inventory::submit! {
    ToolMetadata {
        id: "ParrondoParadoxTool",
        display_name: "Parrondo's Paradox",
        domain: "financial_math",
        tags: &["markov", "ratchet", "kelly", "paradox", "probability"],
        build: || Box::new(ParrondoParadoxTool::default()),
    }
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Parrondo Brownian Ratchet",
    "domain": "financial_math",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "p_a_base": 0.5,
    "p_b0_base": 0.1,
    "p_b1_base": 0.75,
    "epsilon": 0.004,
    "modulus": 3,
    "schedule": "ABB",
    "num_walkers": 320,
    "num_steps": 200,
    "seed": 1337,
    "view_mode": "SplitOverview"
  }
}"#;

inventory::submit! {
    StarterPresetMetadata {
        id: "parrondo_paradox_starter",
        title: "Parrondo Brownian Ratchet",
        description: "Modulo-M Markov ratchet turning two individually losing Kelly NoEdge games into positive drift.",
        category: "Stochastic Ratchets",
        icon: "🎲",
        domain: "financial_math",
        tool_name: Some("ParrondoParadoxTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}

// [cite:clinical_trials_statistics]
