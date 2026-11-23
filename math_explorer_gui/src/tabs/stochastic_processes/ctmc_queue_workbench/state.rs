#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! State serialization, presets, theory metadata, and plot exports for [`super::CtmcQueueWorkbenchTool`].

use super::{CtmcQueueWorkbenchTool, MAX_CAPACITY, MAX_SERVERS, MIN_CAPACITY, MIN_SERVERS};
use crate::framework::{StarterPresetMetadata, ToolMetadata};
use egui_plot::ExportPrimitive;
use scientific_metadata::theory::TheoryDescribable;
use serde::{Deserialize, Serialize};

/// Active continuous-time stochastic regime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ContinuousModelMode {
    /// Slow-rate preconditioned QR nullspace $G^T \pi = 0$ vs Gillespie sojourn times.
    #[default]
    SlowRateCtmc,
    /// Multi-server $M/M/c$ Erlang-C birth-death queue vs truncated CTMC generator.
    MmcBirthDeathQueue,
    /// Lewis-Shedler thinning for inhomogeneous Poisson intensity profiles.
    InhomogeneousPoisson,
}

impl ContinuousModelMode {
    pub(super) const ALL: [Self; 3] = [
        Self::SlowRateCtmc,
        Self::MmcBirthDeathQueue,
        Self::InhomogeneousPoisson,
    ];

    /// Short UI label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::SlowRateCtmc => "Slow-Rate CTMC",
            Self::MmcBirthDeathQueue => "M/M/c Erlang-C",
            Self::InhomogeneousPoisson => "Poisson Thinning",
        }
    }
}

/// Fixed parametric inhomogeneous intensity profile selector (no symbolic AST).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PoissonVariant {
    /// Diurnal sinusoidal intensity $\lambda_0 + A \sin(\omega t)$.
    #[default]
    Sinusoidal,
    /// Piecewise-constant surge window $[0.3 T, 0.65 T]$.
    StepBurst,
}

impl PoissonVariant {
    /// Short UI label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Sinusoidal => "Sinusoidal",
            Self::StepBurst => "Step Burst",
        }
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct CtmcQueueState {
    pub(super) mode: ContinuousModelMode,
    pub(super) poisson_variant: PoissonVariant,
    pub(super) rate_log10: f64,
    pub(super) forward_weight: f64,
    pub(super) backward_weight: f64,
    pub(super) servers: usize,
    pub(super) arrival_rate: f64,
    pub(super) service_rate: f64,
    pub(super) capacity: usize,
    pub(super) horizon: f64,
    pub(super) seed: u64,
}

impl CtmcQueueState {
    pub(super) fn validate(&self) -> Result<(), String> {
        let floats_ok = [
            self.rate_log10,
            self.forward_weight,
            self.backward_weight,
            self.arrival_rate,
            self.service_rate,
            self.horizon,
        ]
        .into_iter()
        .all(f64::is_finite);
        if !floats_ok {
            return Err("Non-finite CTMC/Queue parameter".to_string());
        }
        let max_stable_lambda = self.servers as f64 * self.service_rate;
        if !(-7.0..=0.0).contains(&self.rate_log10)
            || !(0.2..=5.0).contains(&self.forward_weight)
            || !(0.2..=5.0).contains(&self.backward_weight)
            || !(MIN_SERVERS..=MAX_SERVERS).contains(&self.servers)
            || !(0.2..=25.0).contains(&self.arrival_rate)
            || !(0.5..=10.0).contains(&self.service_rate)
            || self.arrival_rate >= max_stable_lambda
            || !(MIN_CAPACITY..=MAX_CAPACITY).contains(&self.capacity)
            || self.capacity < self.servers
            || !(5.0..=100.0).contains(&self.horizon)
        {
            return Err("CTMC/Queue parameter out of bounds or unstable".to_string());
        }
        Ok(())
    }
}

impl TheoryDescribable for CtmcQueueWorkbenchTool {
    fn theory_description(&self) -> String {
        format!(
            "Continuous-time stochastic workbench (`{}`): solves rate-preconditioned $G^T \\pi = 0$ via QR decomposition without Padé time-horizon sensitivity, computes overflow-safe $M/M/c$ Erlang-C Little's Law metrics, and simulates inhomogeneous Poisson arrivals via Lewis-Shedler thinning.",
            self.mode.label()
        )
    }

    fn phonetic_description(&self) -> String {
        "Continuous time Markov chain generator nullspace Gillespie jump process M M c Erlang C birth death queue and Lewis Shedler Poisson thinning".to_string()
    }

    fn theory_citation(&self) -> String {
        "Erlang, A. K. (1917). Solution of some problems in the theory of probabilities of significance in automatic telephone exchanges; Lewis, P. A. W., & Shedler, G. S. (1979). Simulation of nonhomogeneous Poisson processes by thinning. Naval Research Logistics Quarterly.".to_string()
    }

    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "CTMC, Poisson & M/M/c Queues".to_string(),
            self.theory_description(),
        );
        map
    }
}

inventory::submit! {
    ToolMetadata {
        id: "CtmcQueueWorkbenchTool",
        display_name: "CTMC, Poisson & M/M/c Queues",
        domain: "stochastic_processes",
        tags: &["ctmc", "gillespie", "poisson", "thinning", "queueing", "erlang"],
        build: || Box::new(CtmcQueueWorkbenchTool::default()),
    }
}

const PRESET_SLOW_CTMC_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Slow-Rate CTMC QR Nullspace (1e-7)",
    "domain": "stochastic_processes",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "mode": "SlowRateCtmc",
    "poisson_variant": "Sinusoidal",
    "rate_log10": -7.0,
    "forward_weight": 1.0,
    "backward_weight": 3.0,
    "servers": 3,
    "arrival_rate": 4.5,
    "service_rate": 2.0,
    "capacity": 20,
    "horizon": 25.0,
    "seed": 1337
  }
}"#;

const PRESET_MMC_ERLANG_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "M/M/c Erlang-C Multi-Server Queue",
    "domain": "stochastic_processes",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "mode": "MmcBirthDeathQueue",
    "poisson_variant": "Sinusoidal",
    "rate_log10": -4.0,
    "forward_weight": 1.0,
    "backward_weight": 2.0,
    "servers": 3,
    "arrival_rate": 4.8,
    "service_rate": 2.0,
    "capacity": 24,
    "horizon": 30.0,
    "seed": 2026
  }
}"#;

inventory::submit! {
    StarterPresetMetadata {
        id: "ctmc_slow_rate_nullspace_starter",
        title: "Slow-Rate CTMC QR Nullspace (1e-7)",
        description: "Solves G^T pi = 0 directly at 1e-7 transition scale where fixed-horizon Pade exp(Gt) fails to mix.",
        category: "Continuous-Time & Queues",
        icon: "⚡",
        domain: "stochastic_processes",
        tool_name: Some("CtmcQueueWorkbenchTool"),
        preset_generator: || PRESET_SLOW_CTMC_JSON,
    }
}

inventory::submit! {
    StarterPresetMetadata {
        id: "mmc_erlang_c_queue_starter",
        title: "M/M/c Erlang-C Multi-Server Queue",
        description: "3-server birth-death queue comparing closed-form Erlang-C probabilities against truncated CTMC steady_state.",
        category: "Continuous-Time & Queues",
        icon: "📊",
        domain: "stochastic_processes",
        tool_name: Some("CtmcQueueWorkbenchTool"),
        preset_generator: || PRESET_MMC_ERLANG_JSON,
    }
}

impl CtmcQueueWorkbenchTool {
    pub(super) fn build_export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        match self.mode {
            ContinuousModelMode::SlowRateCtmc => {
                let mut ds = vec![(
                    "Gillespie Sojourn Fraction".to_string(),
                    (0..3).map(|i| [i as f64, self.ctmc_empirical[i]]).collect(),
                )];
                if let Some(pi) = self.ctmc_steady {
                    ds.push((
                        "QR Steady State pi".to_string(),
                        (0..3).map(|i| [i as f64, pi[i]]).collect(),
                    ));
                }
                ds
            }
            ContinuousModelMode::MmcBirthDeathQueue => vec![
                (
                    "Analytic Erlang-C p_k".to_string(),
                    self.queue_analytic_pk
                        .iter()
                        .enumerate()
                        .map(|(k, &p)| [k as f64, p])
                        .collect(),
                ),
                (
                    "Truncated CTMC pi_k".to_string(),
                    self.queue_ctmc_pk
                        .iter()
                        .enumerate()
                        .map(|(k, &p)| [k as f64, p])
                        .collect(),
                ),
            ],
            ContinuousModelMode::InhomogeneousPoisson => vec![
                (
                    "Thinned Arrivals N(t)".to_string(),
                    self.primary_curve.iter().map(|p| [p.x, p.y]).collect(),
                ),
                (
                    "Cumulative Mean Lambda(t)".to_string(),
                    self.secondary_curve.iter().map(|p| [p.x, p.y]).collect(),
                ),
            ],
        }
    }

    pub(super) fn build_export_primitives(&self) -> Vec<ExportPrimitive> {
        let pal = crate::app::palette::DARK_PALETTE;
        match self.mode {
            ContinuousModelMode::SlowRateCtmc => self.build_ctmc_export_bars(&pal),
            ContinuousModelMode::MmcBirthDeathQueue => self.build_queue_export_bars(&pal),
            ContinuousModelMode::InhomogeneousPoisson => vec![
                ExportPrimitive::Polyline {
                    name: "Thinned N(t)".to_string(),
                    points: self.primary_curve.iter().map(|p| [p.x, p.y]).collect(),
                    color: pal.series[0],
                    stroke_width: 2.0_f32,
                    closed: false,
                    fill: None,
                },
                ExportPrimitive::Polyline {
                    name: "Mean Lambda(t)".to_string(),
                    points: self.secondary_curve.iter().map(|p| [p.x, p.y]).collect(),
                    color: pal.series[1],
                    stroke_width: 2.0_f32,
                    closed: false,
                    fill: None,
                },
            ],
        }
    }

    fn build_ctmc_export_bars(&self, pal: &crate::app::palette::SemanticPalette) -> Vec<ExportPrimitive> {
        let mut prims = Vec::new();
        for s in 0..3 {
            let x = s as f64;
            if let Some(pi) = self.ctmc_steady {
                prims.push(ExportPrimitive::Bar {
                    name: format!("qr_pi[{s}]"),
                    min: [x - 0.22, 0.0],
                    max: [x - 0.02, pi[s]],
                    fill: pal.series[0],
                    stroke_color: pal.accent,
                    stroke_width: 1.0_f32,
                });
            }
            prims.push(ExportPrimitive::Bar {
                name: format!("sojourn[{s}]"),
                min: [x + 0.02, 0.0],
                max: [x + 0.22, self.ctmc_empirical[s]],
                fill: pal.series[1],
                stroke_color: pal.accent,
                stroke_width: 1.0_f32,
            });
        }
        prims
    }

    fn build_queue_export_bars(&self, pal: &crate::app::palette::SemanticPalette) -> Vec<ExportPrimitive> {
        self.queue_analytic_pk
            .iter()
            .enumerate()
            .flat_map(|(k, &pk)| {
                let x = k as f64;
                let ctmc_p = self.queue_ctmc_pk.get(k).copied().unwrap_or(0.0);
                [
                    ExportPrimitive::Bar {
                        name: format!("p_{k}"),
                        min: [x - 0.2, 0.0],
                        max: [x, pk],
                        fill: pal.series[0],
                        stroke_color: pal.accent,
                        stroke_width: 1.0_f32,
                    },
                    ExportPrimitive::Bar {
                        name: format!("ctmc_{k}"),
                        min: [x, 0.0],
                        max: [x + 0.2, ctmc_p],
                        fill: pal.series[2],
                        stroke_color: pal.accent,
                        stroke_width: 1.0_f32,
                    },
                ]
            })
            .collect()
    }
}

// [cite:clinical_trials_statistics]
