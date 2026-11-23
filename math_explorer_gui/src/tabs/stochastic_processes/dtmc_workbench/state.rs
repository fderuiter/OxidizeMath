#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! State serialization, theory metadata, starter presets, and plot exports for [`super::DtmcTensorWorkbenchTool`].

use super::{DtmcTensorWorkbenchTool, MAX_MC_STEPS, MIN_MC_STEPS};
use crate::framework::{StarterPresetMetadata, ToolMetadata};
use egui_plot::ExportPrimitive;
use scientific_metadata::theory::TheoryDescribable;
use serde::{Deserialize, Serialize};

/// Canonical DTMC and `TransitionTensor` regime presets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum DtmcModelPreset {
    /// Irreducible, aperiodic 3-state chain with editable rows.
    ErgodicChain,
    /// Bipartite 3-state period-2 walk solved via QR nullspace (`[0.25, 0.5, 0.25]`).
    #[default]
    PeriodTwoWalk,
    /// Multi-absorbing chain demonstrating `stationary == None` and `B = N * R` limiting projection.
    AsymmetricAbsorbing,
    /// Non-stationary shot-clock `TransitionTensor` interpolated across $t \in [3, 24]$ s.
    ShotClockTensor,
}

impl DtmcModelPreset {
    pub(super) const ALL: [Self; 4] = [
        Self::PeriodTwoWalk,
        Self::AsymmetricAbsorbing,
        Self::ErgodicChain,
        Self::ShotClockTensor,
    ];

    /// Short UI label for the preset.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::ErgodicChain => "Ergodic 3-State",
            Self::PeriodTwoWalk => "Period-2 Walk",
            Self::AsymmetricAbsorbing => "Asymmetric Absorbing",
            Self::ShotClockTensor => "Shot-Clock Tensor",
        }
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct DtmcWorkbenchState {
    pub(super) preset: DtmcModelPreset,
    pub(super) row0_stay: f64,
    pub(super) row0_to1: f64,
    pub(super) row1_to0: f64,
    pub(super) row1_stay: f64,
    pub(super) shot_clock_time: f64,
    pub(super) initial_state: usize,
    pub(super) target_state: usize,
    pub(super) mc_steps: usize,
    pub(super) seed: u64,
}

impl DtmcWorkbenchState {
    pub(super) fn validate(&self) -> Result<(), String> {
        let floats_ok = [
            self.row0_stay,
            self.row0_to1,
            self.row1_to0,
            self.row1_stay,
            self.shot_clock_time,
        ]
        .into_iter()
        .all(f64::is_finite);
        if !floats_ok {
            return Err("Non-finite DTMC parameter".to_string());
        }
        if !(0.0..=1.0).contains(&self.row0_stay)
            || !(0.0..=1.0).contains(&self.row0_to1)
            || self.row0_stay + self.row0_to1 > 1.0 + 1e-9
            || !(0.0..=1.0).contains(&self.row1_to0)
            || !(0.0..=1.0).contains(&self.row1_stay)
            || self.row1_to0 + self.row1_stay > 1.0 + 1e-9
            || !(3.0..=24.0).contains(&self.shot_clock_time)
            || self.initial_state > 2
            || self.target_state > 2
            || !(MIN_MC_STEPS..=MAX_MC_STEPS).contains(&self.mc_steps)
        {
            return Err("DTMC parameter out of bounds".to_string());
        }
        Ok(())
    }
}

impl TheoryDescribable for DtmcTensorWorkbenchTool {
    fn theory_description(&self) -> String {
        let closed_count = self.classes.iter().filter(|c| c.is_closed).count();
        format!(
            "Discrete-time Markov chain (`{}`) partitioned via iterative Tarjan SCC into {} communicating class(es) ({} closed); solves $(P^T - I)\\pi = 0$ via QR nullspace and projects transient weights through $B = (I - Q)^{{-1}} R$.",
            self.preset.label(),
            self.classes.len(),
            closed_count
        )
    }

    fn phonetic_description(&self) -> String {
        "Discrete time Markov chain Tarjan communicating classes Q R stationary nullspace fundamental matrix absorption and transition tensor interpolation".to_string()
    }

    fn theory_citation(&self) -> String {
        "Norris, J. R. (1997). Markov Chains. Cambridge University Press; Tarjan, R. (1972). Depth-first search and linear graph algorithms. SIAM Journal on Computing, 1(2), 146-160.".to_string()
    }

    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert("DTMC & Transition Tensor".to_string(), self.theory_description());
        map
    }
}

inventory::submit! {
    ToolMetadata {
        id: "DtmcTensorWorkbenchTool",
        display_name: "DTMC & Transition Tensor",
        domain: "stochastic_processes",
        tags: &["markov", "dtmc", "tarjan", "stationary", "absorbing", "tensor"],
        build: || Box::new(DtmcTensorWorkbenchTool::default()),
    }
}

const PRESET_PERIODIC_QR_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Periodic Walk & QR Nullspace",
    "domain": "stochastic_processes",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "preset": "PeriodTwoWalk",
    "row0_stay": 0.0,
    "row0_to1": 1.0,
    "row1_to0": 0.5,
    "row1_stay": 0.0,
    "shot_clock_time": 12.0,
    "initial_state": 0,
    "target_state": 2,
    "mc_steps": 1600,
    "seed": 1337
  }
}"#;

const PRESET_SHOT_CLOCK_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Shot-Clock Transition Tensor",
    "domain": "stochastic_processes",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "preset": "ShotClockTensor",
    "row0_stay": 0.5,
    "row0_to1": 0.475,
    "row1_to0": 0.0,
    "row1_stay": 1.0,
    "shot_clock_time": 8.5,
    "initial_state": 0,
    "target_state": 1,
    "mc_steps": 1500,
    "seed": 2026
  }
}"#;

inventory::submit! {
    StarterPresetMetadata {
        id: "dtmc_periodic_qr_starter",
        title: "Periodic Walk & QR Nullspace",
        description: "3-state period-2 walk where power iteration oscillates but QR nullspace recovers [0.25, 0.5, 0.25].",
        category: "Markov Chains",
        icon: "🔁",
        domain: "stochastic_processes",
        tool_name: Some("DtmcTensorWorkbenchTool"),
        preset_generator: || PRESET_PERIODIC_QR_JSON,
    }
}

inventory::submit! {
    StarterPresetMetadata {
        id: "dtmc_shot_clock_tensor_starter",
        title: "Shot-Clock Transition Tensor",
        description: "Time-indexed non-stationary absorbing possession tensor interpolated across the 24-second clock.",
        category: "Markov Chains",
        icon: "⏱",
        domain: "stochastic_processes",
        tool_name: Some("DtmcTensorWorkbenchTool"),
        preset_generator: || PRESET_SHOT_CLOCK_JSON,
    }
}

impl DtmcTensorWorkbenchTool {
    pub(super) fn build_export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let mut datasets = vec![(
            "Empirical Visit Frequency".to_string(),
            (0..3)
                .map(|s| [s as f64, self.empirical_freq[s]])
                .collect(),
        )];
        if let Some(pi) = self.stationary {
            datasets.push((
                "QR Stationary pi".to_string(),
                (0..3).map(|s| [s as f64, pi[s]]).collect(),
            ));
        }
        if let Some(lim) = self.limiting {
            datasets.push((
                "Limiting Distribution (B=N*R)".to_string(),
                (0..3).map(|s| [s as f64, lim[s]]).collect(),
            ));
        }
        datasets
    }

    pub(super) fn build_export_primitives(&self) -> Vec<ExportPrimitive> {
        let pal = crate::app::palette::DARK_PALETTE;
        let mut prims = Vec::new();
        for s in 0..3 {
            let x = s as f64;
            if let Some(pi) = self.stationary {
                prims.push(ExportPrimitive::Bar {
                    name: format!("pi[{s}]"),
                    min: [x - 0.27, 0.0],
                    max: [x - 0.09, pi[s]],
                    fill: pal.series[0],
                    stroke_color: pal.accent,
                    stroke_width: 1.0,
                });
            }
            if let Some(lim) = self.limiting {
                prims.push(ExportPrimitive::Bar {
                    name: format!("lim[{s}]"),
                    min: [x - 0.09, 0.0],
                    max: [x + 0.09, lim[s]],
                    fill: pal.series[1],
                    stroke_color: pal.accent,
                    stroke_width: 1.0,
                });
            }
            prims.push(ExportPrimitive::Bar {
                name: format!("emp[{s}]"),
                min: [x + 0.09, 0.0],
                max: [x + 0.27, self.empirical_freq[s]],
                fill: pal.series[2],
                stroke_color: pal.accent,
                stroke_width: 1.0,
            });
        }
        prims
    }
}

// [cite:clinical_trials_statistics]
